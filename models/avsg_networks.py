import torch
import torch.nn as nn
import torch.nn.functional as nnf
from models.avsg_moudules import MLP, PointNet
from models.avsg_map_encoder import MapEncoder
from models.avsg_agents_decoder import get_agents_decoder

##############################################################################################

class SceneGenerator(nn.Module):

    def __init__(self, opt):
        super(SceneGenerator, self).__init__()
        self.device = opt.device
        self.max_num_agents = opt.max_num_agents
        self.dim_latent_map = opt.dim_latent_map
        self.dim_agent_feat_vec = len(opt.agent_feat_vec_coord_labels)
        self.dim_agent_noise = opt.dim_agent_noise
        self.map_enc = MapEncoder(opt)
        self.agents_dec = get_agents_decoder(opt, self.device)
        # Debug - print parameter names:  [x[0] for x in self.named_parameters()]
        self.batch_size = opt.batch_size


    def forward(self, conditioning):
        """Standard forward"""
        map_feat = conditioning['map_feat']
        n_agents = conditioning['n_agents']
        map_latent = self.map_enc(map_feat)
        latent_noise_std = 1.0
        latent_noise = torch.randn(self.max_num_agents, self.dim_agent_noise, device=self.device) * latent_noise_std
        agents_feat_vecs = self.agents_dec(map_latent, latent_noise, n_agents)
        return agents_feat_vecs


#########################################################################################


class SceneDiscriminator(nn.Module):

    def __init__(self, opt):
        super(SceneDiscriminator, self).__init__()
        self.device = opt.device
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

    def forward(self, conditioning, agents_feat_vecs):
        map_feat = conditioning['map_feat']
        n_agents = conditioning['n_agents']

        """In case the number of agents in the conditioning is less than  max_num_agents 
        pad the input with zeros"""
        if n_agents < self.max_num_agents:
            pad = (0, 0, 0, self.max_num_agents - n_agents)
            agents_feat_vecs = nnf.pad(agents_feat_vecs, pad, mode='constant', value=0.0)

        """Standard forward."""
        map_latent = self.map_enc(map_feat)
        agents_latent = self.agents_enc(agents_feat_vecs)
        scene_latent = torch.cat([map_latent, agents_latent])
        pred_fake = self.out_mlp(scene_latent)
        ''' 
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        '''
        return pred_fake
#########################################################################################
