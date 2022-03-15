import torch
from torch import nn as nn
from torch.nn.init import trunc_normal_
from models.avsg_agents_decoder import get_agents_decoder
from models.avsg_map_encoder import MapEncoder
from util.helper_func import init_net


###############################################################################

def define_G(opt, gpu_ids=None):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """

    if gpu_ids is None:
        gpu_ids = []

    if opt.netG == 'SceneGenerator':
        net = SceneGenerator(opt)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % opt.netG)
    return init_net(net, opt.init_type, opt.init_gain, gpu_ids)


###############################################################################

class SceneGenerator(nn.Module):

    def __init__(self, opt):
        super(SceneGenerator, self).__init__()
        self.device = opt.device
        self.max_num_agents = opt.max_num_agents
        self.dim_latent_map = opt.dim_latent_map
        self.dim_agent_feat_vec = len(opt.agent_feat_vec_coord_labels)
        self.dim_agent_noise = opt.dim_agent_noise
        self.latent_noise_trunc_stds = opt.latent_noise_trunc_stds
        self.map_enc = MapEncoder(opt)
        self.agents_dec = get_agents_decoder(opt, self.device)
        # Debug - print parameter names:  [x[0] for x in self.named_parameters()]
        self.batch_size = opt.batch_size

    def forward(self, conditioning):
        """Standard forward"""
        map_feat = conditioning['map_feat']
        n_agents_per_scene = conditioning['n_agents_in_scene']
        agents_exists = conditioning['agents_exists']
        batch_size = len(n_agents_per_scene)
        map_latent = self.map_enc(map_feat)
        latent_noise = self.generate_latent_noise(batch_size, n_agents_per_scene, self.latent_noise_trunc_stds)
        agents_feat_vecs = self.agents_dec(map_latent, latent_noise, n_agents_per_scene, agents_exists)
        return agents_feat_vecs

    def generate_latent_noise(self, batch_size, n_agents_per_scene, latent_noise_trunc_stds):
        # Agents decoder gets a latent noise that is non-zero only in coordinates “associated” with an agent
        # (i.e., if there are only n agents to produce then only n/max_agents_num of the vector is non-zero)
        latent_noise = torch.zeros(batch_size, self.max_num_agents, self.dim_agent_noise, device=self.device)
        for i_scene in range(batch_size):
            n_agents = n_agents_per_scene[i_scene]
            trunc_normal_(latent_noise[i_scene, :n_agents], mean=0., std=1.,
                          a=-latent_noise_trunc_stds, b=+latent_noise_trunc_stds)

        return latent_noise

    ###############################################################################


def get_out_of_road_penalty(conditioning, agents, opt):
    polygon_types = opt.polygon_types
    i_centroid_x = opt.agent_feat_vec_coord_labels.index('centroid_x')
    i_centroid_y = opt.agent_feat_vec_coord_labels.index('centroid_y')
    i_lanes_mid = polygon_types.index('lanes_mid')
    i_lanes_left = polygon_types.index('lanes_left')
    i_lanes_right = polygon_types.index('lanes_right')
    map_feat = conditioning['map_feat']
    map_elems_points = map_feat['map_elems_points']
    map_elems_exists = map_feat['map_elems_exists']
    batch_size, n_polygon_types, max_num_elem, max_points_per_elem, coord_dim = map_elems_points.shape
    agents_exists = conditioning['agents_exists']
    n_agents_in_scene = conditioning['n_agents_in_scene']
    max_n_agents = agents.shape[1]

    # Get lanes points, set infinity in non-valid coordinates
    lanes_mid_points = map_elems_points[:, i_lanes_mid]
    lanes_mid_points[torch.logical_not(map_elems_exists[:, i_lanes_mid])] = torch.inf
    lanes_left_points = map_elems_points[:, i_lanes_left]
    lanes_left_points[torch.logical_not(map_elems_exists[:, i_lanes_left])] = torch.inf
    lanes_right_points = map_elems_points[:, i_lanes_right]
    lanes_right_points[torch.logical_not(map_elems_exists[:, i_lanes_right])] = torch.inf

    # Get agents centroids, set infinity in non-valid coordinates
    agents_centroids = agents[:, :, [i_centroid_x, i_centroid_y]]
    agents_centroids[torch.logical_not(agents_exists)] = torch.inf

    ###  Now we transform the tensors to be with a common dimensions of
    #    [batch_size, max_n_agents, max_num_elem, max_points_per_elem, coord_dim]
    agents_centroids = agents_centroids.unsqueeze(2).unsqueeze(2).expand(-1, -1, max_num_elem, max_points_per_elem, -1)
    lanes_mid_points = lanes_mid_points.unsqueeze(1).expand(-1, max_n_agents, -1, -1, -1)

    #  Compute dists of agents centroids to mid-lane points   [batch_size, max_n_agents, max_num_elem, max_points_per_elem]
    dists_sqr_agent_to_mid_points = ((agents_centroids - lanes_mid_points) ** 2).sum(dim=-1)
    dists_sqr_agent_to_mid_points = dists_sqr_agent_to_mid_points.view(batch_size, max_n_agents,
                                                                       max_num_elem * max_points_per_elem)

    # find the closest mid-lane point to each agent
    min_dists_sqr_agent_to_mid_points = dists_sqr_agent_to_mid_points.min(dim=2)
    i_closest_mid = min_dists_sqr_agent_to_mid_points.indices
    i_closest_mid = i_closest_mid.view(batch_size * max_n_agents)
    lanes_mid_points = lanes_mid_points.view(batch_size, max_n_agents, max_num_elem * max_points_per_elem, coord_dim)
    lanes_mid_points = lanes_mid_points.reshape(batch_size * max_n_agents, max_num_elem * max_points_per_elem,
                                                coord_dim)
    closest_mid_points = lanes_mid_points[torch.arange(lanes_mid_points.shape[0]), i_closest_mid, :]
    closest_mid_points = closest_mid_points.view(batch_size, max_n_agents, coord_dim)

    # penalize fake agents if there is any left-lane or right-lane point closer to mid_point than the agents' centroid
    lanes_left_points = lanes_left_points.view(batch_size, max_num_elem * max_points_per_elem, coord_dim)
    lanes_left_points = lanes_left_points.unsqueeze(1).expand(-1, max_n_agents, -1, -1)
    lanes_right_points = lanes_right_points.view(batch_size, max_num_elem * max_points_per_elem, coord_dim)
    lanes_right_points = lanes_right_points.unsqueeze(1).expand(-1, max_n_agents, -1, -1)
    closest_mid_points = closest_mid_points.unsqueeze(2).expand(-1, -1, max_num_elem * max_points_per_elem, -1)
    dist_sqr_closest_mid_to_left = ((lanes_left_points - closest_mid_points) ** 2).sum(dim=-1).min(dim=-1).values
    dist_sqr_closest_mid_to_right = ((lanes_right_points - closest_mid_points) ** 2).sum(dim=-1).min(dim=-1).values
    dists_sqr_agent_to_mid_points = min_dists_sqr_agent_to_mid_points.values
    # set zero to all non-valid coordinates
    dists_sqr_agent_to_mid_points[torch.isinf(dists_sqr_agent_to_mid_points)] = 0
    dist_sqr_closest_mid_to_left[torch.isinf(dist_sqr_closest_mid_to_left)] = 0
    dist_sqr_closest_mid_to_right[torch.isinf(dist_sqr_closest_mid_to_right)] = 0
    dists_sqr_agent_to_mid_points[torch.isnan(dists_sqr_agent_to_mid_points)] = 0
    dist_sqr_closest_mid_to_left[torch.isnan(dist_sqr_closest_mid_to_left)] = 0
    dist_sqr_closest_mid_to_right[torch.isnan(dist_sqr_closest_mid_to_right)] = 0

    f_relu = nn.ReLU()
    penalty = f_relu(dists_sqr_agent_to_mid_points - dist_sqr_closest_mid_to_left).square().sum(dim=1) \
              + f_relu(dists_sqr_agent_to_mid_points - dist_sqr_closest_mid_to_right).square().sum(dim=1)
    penalty = penalty.sum()  # sum over batch
    assert not torch.any(torch.isnan(penalty))
    assert not torch.any(torch.isinf(penalty))
    return penalty
