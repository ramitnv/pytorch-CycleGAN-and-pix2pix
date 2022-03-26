import torch
from torch import sqrt, linalg as LA
from torch.nn.functional import elu
import numpy as np


###############################################################################

class ProjectionToAgentFeat(object):
    '''
    Project the generator output to the feature vectors domain
    '''

    def __init__(self, opt, device):
        self.device = device
        self.max_num_agents = opt.max_num_agents
        self.agent_feat_vec_coord_labels = opt.agent_feat_vec_coord_labels
        self.dim_agent_feat_vec = len(opt.agent_feat_vec_coord_labels)
        # code now supports only this feature format:
        assert self.agent_feat_vec_coord_labels == [
            'centroid_x',  # [0]  Real number
            'centroid_y',  # [1]  Real number
            'yaw_cos',  # [2]  in range [-1,1],  sin(yaw)^2 + cos(yaw)^2 = 1
            'yaw_sin',  # [3]  in range [-1,1],  sin(yaw)^2 + cos(yaw)^2 = 1
            'speed',  # [4] Real non-negative
        ]

    def __call__(self, agents_vecs, n_agents_per_scene, agents_exists):
        '''
        agents_vecs [batch_size x max_num_agents x dim_agent_feat_vec)]
        # Coordinates 0,1 are centroid x,y - no need to project
        # Coordinates 2,3 are yaw_cos, yaw_sin - project to unit circle by normalizing to L norm ==1
        # Coordinate 4 is speed project to positive numbers
        '''
        eps = 1e-12
        agents_vecs = torch.cat([
            agents_vecs[:, :, :2],
            agents_vecs[:, :, 2:4] / (LA.vector_norm(agents_vecs[:, :, 2:4], ord=2, dim=2, keepdims=True) + eps),
            torch.abs(agents_vecs[:, :, 4].unsqueeze(-1))  # should we use F.softplus ?
        ], dim=2)
        # Set zero at non existent agents
        agents_vecs[agents_exists.logical_not()] = 0.
        return agents_vecs


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

    # Get agents centroids
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

def get_distance_to_closest_lane_points(lanes_points, lanes_points_exists, reference_points):
    '''
    returns distances from the "reference_points" to the corresponding closest left/right lane points
    note: reference_points changes dimensions
    '''
    batch_size, max_num_elem, max_points_per_elem, coord_dim = lanes_points.shape
    max_n_agents = reference_points.shape[1]
    invalids = torch.logical_not(lanes_points_exists)

    # change the two tensors into a common shape:
    lanes_points = lanes_points.view(batch_size, max_num_elem * max_points_per_elem, coord_dim)
    lanes_points = lanes_points.unsqueeze(1).expand(-1, max_n_agents, -1, -1)
    invalids = invalids.unsqueeze(1).repeat(1, max_n_agents, max_points_per_elem)
    reference_points = reference_points.unsqueeze(2).expand(-1, -1, max_num_elem * max_points_per_elem, -1)

    # find min dist from the "reference_points" to a left \ right lane point
    d_sqr_ref_to_lane = (lanes_points - reference_points).square().sum(dim=-1)
    d_sqr_ref_to_lane[invalids] = torch.inf  # Set distance=inf in non-existent points
    d_sqr_ref_to_closest_lane_point = d_sqr_ref_to_lane.min(dim=-1).values

    return d_sqr_ref_to_closest_lane_point


###############################################################################


def get_collisions_penalty(conditioning, agents, opt):
    batch_size, max_n_agents, n_feat = agents.shape
    i_centroid_x = opt.agent_feat_vec_coord_labels.index('centroid_x')
    i_centroid_y = opt.agent_feat_vec_coord_labels.index('centroid_y')
    i_yaw_cos = opt.agent_feat_vec_coord_labels.index('yaw_cos')
    i_yaw_sin = opt.agent_feat_vec_coord_labels.index('yaw_sin')
    extent_length = opt.default_agent_extent_length
    extent_width = opt.default_agent_extent_width
    agents_exists = conditioning['agents_exists']
    centroids = agents[:, :, [i_centroid_x, i_centroid_y]]
    front_direction = agents[:, :, [i_yaw_cos, i_yaw_sin]]
    front_vec = front_direction * extent_length * 0.5
    rot_mat = torch.tensor(([0, -1.], [1., 0])).to(
        opt.device)  # explanation: the original direction vec is (cos(a), sin(a)) we want to rotate by +90 degrees, (cos(pi/2+a), sin(pi/2 + a) = (-sin(a) , +cos(a)) = rot_mat @ (cos(a), sin(a))
    left_vec = front_direction @ rot_mat * extent_width * 0.5

    # find the  middle of each of the 4 segments (sides of the car)
    segs_mids = {'front': centroids + front_vec, 'back': centroids - front_vec,
               'left': centroids + left_vec, 'right': centroids - left_vec}

    # find a vector that goes from the center of each segment to one of its edges (doesn't matter which of the two)
    segs_vecs = {'front': + left_vec, 'back': - left_vec,
                'left': - front_vec, 'right': + front_vec}

    for i_agent1 in range(max_n_agents - 1):
        for i_agent2 in range(i_agent1 + 1, max_n_agents):
            # find valid scenes = both agents exists,
            valids = agents_exists[:, i_agent1] * agents_exists[:, i_agent2]
            for seg1_name, seg1_vecs in segs_vecs.items():
                for seg2_name, seg2_vecs in segs_vecs.items():
                    seg1_mids = segs_mids[seg1_name]
                    seg2_mids = segs_mids[seg2_name]
                    seg1_mid = seg1_mids[valids, i_agent1, :]
                    seg2_mid = seg2_mids[valids, i_agent2, :]
                    seg1_vec = seg1_vecs[valids, i_agent1, :]
                    seg2_vec = seg1_vecs[valids, i_agent2, :]
                    # find the deviation of the intersection point from the middle of the segment
                    # if it is inside the segment, then it is a collision between the cars4
                    # add to the penalty the distance of the impact from the corner,
                    # use ReLU or ELU to only penalize collisions
                    # See: https://math.stackexchange.com/a/406895
                    # determinant  = (x_dir_1) * (-y_dir_2) - (- x_dir_2) * (y_dir_1)
                    # = (x_dir_2) * (y_dir_1) -(x_dir_1) * (y_dir_2)
                    determinant = seg2_vec[:, 0] * seg1_vec[:, 1] - seg1_vec[:, 0] * seg2_vec[:, 1]
                    epsilon = 1e-6
                    # find valid scenes = where there is a cross of the infinite lines
                    # of the two corresponding segments
                    is_cross = determinant.abs() > epsilon
                    pass

    return
