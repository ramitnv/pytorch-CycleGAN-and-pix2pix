import torch
from torch import nn as nn

from models.avsg_map_encoder import MapEncoder
from models.sub_modules import PointNet, MLP
from util.helper_func import init_net, set_spectral_norm_normalization
from models.avsg_func import get_extra_D_inputs

###############################################################################

def define_D(opt, gpu_ids=None):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    if gpu_ids is None:
        gpu_ids = []

    if opt.netD == 'SceneDiscriminator':
        net = SceneDiscriminator(opt)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % opt.netD)

    if opt.use_spectral_norm_D:
        net = set_spectral_norm_normalization(net)
    net = init_net(net, opt.init_type, opt.init_gain, gpu_ids)
    return net


##############################################################################

class SceneDiscriminator(nn.Module):

    def __init__(self, opt):
        super(SceneDiscriminator, self).__init__()
        self.opt = opt
        self.device = opt.device
        self.batch_size = opt.batch_size
        self.max_num_agents = opt.max_num_agents
        self.agent_feat_vec_coord_labels = opt.agent_feat_vec_coord_labels
        self.dim_agent_feat_vec = len(opt.agent_feat_vec_coord_labels)
        self.dim_discr_agents_enc = opt.dim_discr_agents_enc
        self.dim_latent_map = opt.dim_latent_map
        self.map_enc = MapEncoder(opt)
        self.agents_enc = PointNet(d_in=self.dim_agent_feat_vec,
                                   d_out=self.dim_discr_agents_enc,
                                   d_hid=self.dim_discr_agents_enc,
                                   n_layers=opt.n_discr_pointnet_layers,
                                   opt=opt)
        self.out_mlp = MLP(d_in=self.dim_latent_map + self.dim_discr_agents_enc,
                           d_out=1,
                           d_hid=self.dim_discr_agents_enc,
                           n_layers=opt.n_discr_out_mlp_layers,
                           opt=opt)
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        self.collisions_enc = PointNet(d_in=2,  # s1,s2
                                   d_out=1,
                                   d_hid=32,
                                   n_layers=3,
                                   opt=opt)
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def forward(self, conditioning, agents_feat_vecs, extra_D_input=None):
        if not extra_D_input:
            extra_D_input = get_extra_D_inputs(conditioning, agents_feat_vecs, self.opt)

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        '''
        The extra_D_input is used to extend the agents feature vectors:
        out_of_road_indicator &  collisions_indicators  for 'front', 'back', 'left', 'right' (5 features  total)
        The collision indicator at each segment at each i_agent is calculated by
        taking as input the s1,s2 with all agents paired with  i_agent and passing through a PointNet
        '''
        agents_exists = conditioning['agents_exists']
        out_of_road_indicators = extra_D_input['out_of_road_indicators']
        collisions_indicators = extra_D_input['collisions_indicators']
        segs_names = ['front', 'back', 'left', 'right']

        batch_size, max_n_agents = agents_exists.shape
        agents_feat_vecs = nn.functional.pad(agents_feat_vecs, (0, 9))
        assert agents_feat_vecs.shape[-1] == 14
        agents_feat_vecs[:, :, 5] = out_of_road_indicators
        agents_feat_vecs[:, :, 5] = out_of_road_indicators
        # agent_incoming_collisions =
        for i_agent in range(max_n_agents):
            self.collisions_enc()

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        map_feat = conditioning['map_feat']
        map_latent = self.map_enc(map_feat)
        agents_latent = self.agents_enc(agents_feat_vecs)
        scene_latent = torch.cat([map_latent, agents_latent], dim=1)
        pred_fake = self.out_mlp(scene_latent)
        ''' 
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        '''
        return pred_fake


##############################################################################


def get_gradient_penalty(netD, conditioning, real_samp, fake_samp, model, constant=1.0):
    """Calculate the gradient penalty loss,
    similar to the WGAN-GP paper https://arxiv.org/abs/1704.00028
    but I took the sum of gradients at x_real and at x_fake
     (it makes more sense thant to take the gradient at the interpolation point, as in the paper)

    Arguments:
        netD (network)              -- discriminator network
        real_samp (tensor array)    -- real images
        fake_samp (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if model.gan_mode != 'WGANGP':
        return None
    gradient_penalty = torch.tensor(0., device=model.device)
    for samp in real_samp, fake_samp:
        is_requires_grad = samp.requires_grad
        samp.requires_grad_(True)
        d_out = netD(conditioning, samp)
        gradients = torch.autograd.grad(outputs=d_out, inputs=samp,
                                        grad_outputs=torch.ones_like(d_out),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_samp.size(0), -1)  # flat the data
        gradient_penalty += ((gradients + 1e-16).norm(2, dim=1) - constant).square().mean()  # added eps
        samp.requires_grad_(is_requires_grad)
    return gradient_penalty

###############################################################################

# def cal_gradient_penalty(netD, conditioning, real_samp, fake_samp, model, type='mixed', constant=1.0):
#     """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
#
#     Arguments:
#         netD (network)              -- discriminator network
#         real_samp (tensor array)    -- real images
#         fake_samp (tensor array)    -- generated images from the generator
#         device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
#         type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
#         constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
#         lambda_gp (float)           -- weight for this loss
#
#     Returns the gradient penalty loss
#     """
#     device = model.device
#     if model.gan_mode != 'WGANGP':
#         return None
#     if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
#         interpolates_v = real_samp
#     elif type == 'fake':
#         interpolates_v = fake_samp
#     elif type == 'mixed':
#         # Based on "Improved Training of Wasserstein GANs" Gulrajani et. al. 2017
#         alpha = torch.rand(real_samp.shape[0], 1, device=device)
#         alpha = alpha.expand(real_samp.shape[0], real_samp.nelement() // real_samp.shape[0]).contiguous().view(
#             *real_samp.shape)
#         interpolates_v = alpha * real_samp + ((1 - alpha) * fake_samp)
#     else:
#         raise NotImplementedError('{} not implemented'.format(type))
#     interpolates_v.requires_grad_(True)
#     disc_interpolates = netD(conditioning, interpolates_v)
#     gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates_v,
#                                     grad_outputs=torch.ones(disc_interpolates.size()).to(device),
#                                     create_graph=True, retain_graph=True, only_inputs=True)
#     gradients = gradients[0].view(real_samp.size(0), -1)  # flat the data
#     gradient_penalty = ((gradients + 1e-16).norm(2, dim=1) - constant).square().mean()  # added eps
#     return gradient_penalty

###############################################################################