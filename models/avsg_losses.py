import torch
from torch import sqrt
from torch.nn.functional import elu

from models.avsg_generator import get_distance_to_closest_lane_points

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
    max_n_agents = agents.shape[1]

    # Get lanes points
    lanes_mid_points = map_elems_points[:, i_lanes_mid]

    # Get agents centroids, set infinity in non-valid coordinates
    agents_centroids = agents[:, :, [i_centroid_x, i_centroid_y]]

    #   Now we transform the tensors to be with a common dimensions of
    #   [batch_size, max_n_agents, max_num_elem, max_points_per_elem, coord_dim]
    agents_centroids = agents_centroids.unsqueeze(2).unsqueeze(2).expand(-1, -1, max_num_elem, max_points_per_elem, -1)
    lanes_mid_points = lanes_mid_points.unsqueeze(1).expand(-1, max_n_agents, -1, -1, -1)

    #  Compute dists of agents centroids to mid-lane points
    #  [batch_size, max_n_agents, max_num_elem, max_points_per_elem]
    d_sqr_agent_to_mid = (agents_centroids - lanes_mid_points).square().sum(dim=-1)

    # Set distance=inf in non-existent agents and points, so that invalid indices wouldn't be chosen in the min()
    d_sqr_agent_to_mid[torch.logical_not(agents_exists)] = torch.inf
    invalids = torch.logical_not(map_elems_exists[:, i_lanes_mid])
    invalids = invalids.unsqueeze(1).unsqueeze(-1).expand(-1, max_n_agents, -1, max_points_per_elem)
    d_sqr_agent_to_mid[invalids] = torch.inf

    # find the indices of the closest mid-lane point to each agent
    d_sqr_agent_to_mid = d_sqr_agent_to_mid.view(batch_size, max_n_agents, max_num_elem * max_points_per_elem)
    min_dists_sqr_agent_to_mid_points = d_sqr_agent_to_mid.min(dim=2)
    i_closest_mid = min_dists_sqr_agent_to_mid_points.indices
    d_sqr_agent_to_mid = min_dists_sqr_agent_to_mid_points.values
    i_closest_mid = i_closest_mid.view(batch_size * max_n_agents)
    lanes_mid_points = lanes_mid_points.reshape(batch_size * max_n_agents, max_num_elem * max_points_per_elem,
                                                coord_dim)
    # Select the mid-lane points with minimal distance per agent:
    closest_mid_points = lanes_mid_points[torch.arange(lanes_mid_points.shape[0]), i_closest_mid, :]
    closest_mid_points = closest_mid_points.view(batch_size, max_n_agents, coord_dim)

    # find min dist from the "closest_mid_points" to a left \ right lane point
    d_sqr_mid_to_left = get_distance_to_closest_lane_points(map_elems_points[:, i_lanes_left],
                                                            map_elems_exists[:, i_lanes_left],
                                                            closest_mid_points)
    d_sqr_mid_to_right = get_distance_to_closest_lane_points(map_elems_points[:, i_lanes_right],
                                                             map_elems_exists[:, i_lanes_right],
                                                             closest_mid_points)
    # get valid indices
    d_sqr_mid_to_left = d_sqr_mid_to_left.flatten()
    d_sqr_mid_to_right = d_sqr_mid_to_right.flatten()
    d_sqr_agent_to_mid = d_sqr_agent_to_mid.flatten()
    invalids = torch.isinf(d_sqr_agent_to_mid) + torch.isnan(d_sqr_agent_to_mid) \
               + torch.isinf(d_sqr_mid_to_left) + torch.isnan(d_sqr_mid_to_left) \
               + torch.isinf(d_sqr_mid_to_right) + torch.isnan(d_sqr_mid_to_right)
    valids = torch.logical_not(invalids)

    # penalize fake agents if there is any left-lane or right-lane point closer to mid_point than the agents' centroid

    # sum over all scenes and all agents the penalty for out-of-road agents
    # penalty := ELU(dist_agent_to_mid - left_to_mid) + ELU(dist_agent_to_mid - right_to_mid)
    penalty = elu(sqrt(d_sqr_agent_to_mid[valids]) - sqrt(d_sqr_mid_to_left[valids])).sum() \
              + elu(sqrt(d_sqr_agent_to_mid[valids]) - sqrt(d_sqr_mid_to_right[valids])).sum()

    return penalty

###############################################################################
