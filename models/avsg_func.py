import torch
from torch import sqrt, linalg as LA
from torch.nn.functional import elu
from util.common_util import append_to_field

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

def get_extra_D_inputs(conditioning, fake_agents, opt):
    extra_D_inputs = {'out_of_road_indicators': get_out_of_road_indicators(conditioning, fake_agents, opt),
                      'collisions_indicators': get_collisions_indicators(conditioning, fake_agents, opt)}
    return extra_D_inputs


###############################################################################

def get_out_of_road_indicators(conditioning, agents, opt):
    '''
       # out_of_road_indicators [scene_id x agent_idx] = the distance for which the agent centroid is out-of-road
    '''
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

    # out_of_road_indicators [scene_id x agent_idx] = the distance for which the agent centroid is out-of-road
    #  := ELU(dist_agent_to_mid - left_to_mid) + ELU(dist_agent_to_mid - right_to_mid)
    out_of_road_indicators = elu(sqrt(d_sqr_agent_to_mid) - sqrt(d_sqr_mid_to_left)) \
                             + elu(sqrt(d_sqr_agent_to_mid) - sqrt(d_sqr_mid_to_right))
    out_of_road_indicators[valids.logical_not()] = 0.
    out_of_road_indicators = out_of_road_indicators.view(batch_size, max_n_agents)
    return out_of_road_indicators


###############################################################################
def get_out_of_road_penalty(conditioning, extra_D_inputs, opt):

    n_agents_in_scene = conditioning['n_agents_in_scene']
    out_of_road_indicators = extra_D_inputs['out_of_road_indicators']
    # Average over agents
    out_of_road_penalty = out_of_road_indicators.sum(axis=-1) / n_agents_in_scene
    return out_of_road_penalty.mean()  # average over batch

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


def get_collisions_indicators(conditioning, agents, opt):
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
        opt.device)  # explanation: the original direction vec is (cos(a), sin(a)) we want to rotate by +90 degrees,
    # (cos(pi/2+a), sin(pi/2 + a) = (-sin(a) , +cos(a)) = rot_mat @ (cos(a), sin(a))
    left_vec = front_direction @ rot_mat * extent_width * 0.5

    # find the  middle of each of the 4 segments (sides of the car)
    segs_mids = {'front': centroids + front_vec, 'back': centroids - front_vec,
                 'left': centroids + left_vec, 'right': centroids - left_vec}

    # find a vector that goes from the center of each segment to one of its edges (doesn't matter which of the two)
    segs_vecs = {'front': + left_vec, 'back': - left_vec,
                 'left': - front_vec, 'right': + front_vec}

    collisions_indicators = {}
    for i_agent1 in range(max_n_agents - 1):
        for i_agent2 in range(i_agent1 + 1, max_n_agents):
            # find valid scenes = both agents IDs exists,
            valids = agents_exists[:, i_agent1] * agents_exists[:, i_agent2]
            if valids.sum() == 0:
                continue
            for seg1_name, seg1_vecs in segs_vecs.items():
                for seg2_name, seg2_vecs in segs_vecs.items():
                    # get the segments middle points and direction vectors (normalized to be 0.5 * segment_length)
                    L1_p = segs_mids[seg1_name][:, i_agent1, :]
                    L2_p = segs_mids[seg2_name][:, i_agent2, :]
                    L1_v = seg1_vecs[:, i_agent1, :]
                    L2_v = seg2_vecs[:, i_agent2, :]

                    # find the deviation of the intersection point from the middle of the segment
                    # See: https://math.stackexchange.com/a/406895
                    # our problem to solve in a matrix form:  A f = d
                    # where
                    # A = [[L1_v_x, -L2_v_x], [L1_v_y, -L2_v_y]]
                    # s = [s1, s2]
                    # d = [L2_p_x - L1_p_x, L2_p_y - L1_p_y] = [dx, dy]

                    # determinant(A)  = (L1_v_x) * (-L2_v_y) - (-L2_v_x) * (L1_v_y)
                    # = (L2_v_x) * (L1_v_y) -(L1_v_x) * (L2_v_y)
                    determinant = L2_v[:, 0] * L1_v[:, 1] - L1_v[:, 0] * L2_v[:, 1]
                    epsilon = 1e-6

                    # find valid scenes = where there is a cross of the two infinite lines
                    # (might not be inside the segments)
                    is_cross = determinant.abs() > epsilon
                    if is_cross.sum() == 0:
                        continue
                    valids = valids * is_cross

                    # A^{-1} = (1/determinant) * [[ -L2_v_y, L2_v_x], [-L1_v_y, L1_v_x]]
                    # s = A^{-1} d
                    # s1 = (1/determinant) * (-L2_v_y * dx + L2_v_x * dy) = (L2_v_x * dy - L2_v_y * dx) / determinant
                    # s2 = (1/determinant) * (-L1_v_y * dx + L1_v_x * dy) = (L1_v_x * dy - L1_v_y * dx) / determinant
                    d = L2_p - L1_p
                    s1 = torch.zeros(batch_size, device=opt.device)
                    s2 = torch.zeros(batch_size, device=opt.device)
                    s1[valids] = (L2_v[valids, 0] * d[valids, 1] - L2_v[valids, 1] * d[valids, 0]) / determinant[valids]
                    s2[valids] = (L1_v[valids, 0] * d[valids, 1] - L1_v[valids, 1] * d[valids, 0]) / determinant[valids]

                    collisions_indicators[(i_agent1, i_agent2, seg1_name, seg2_name)] = (s1, s2, valids)
    collisions_indicators['batch_size'] = batch_size
    return collisions_indicators


###############################################################################
def get_collisions_penalty(conditioning, extra_D_inputs, opt):
    agents_exists = conditioning['agents_exists']
    batch_size,  max_n_agents = agents_exists.shape
    collisions_indicators = extra_D_inputs['collisions_indicators']
    segs_names = ['front', 'back', 'left', 'right']
    penalty = torch.tensor(0., device=opt.device)
    for i_agent1 in range(max_n_agents - 1):
        for i_agent2 in range(i_agent1 + 1, max_n_agents):
            for seg1_name in segs_names:
                for seg2_name in segs_names:
                    s1, s2, valids = collisions_indicators[(i_agent1, i_agent2, seg1_name, seg2_name)]
                    if valids.sum() == 0:
                        continue
                    curr_penalty = (1 + elu(1 - s1[valids].abs())) * (1 + elu(1 - s2[valids].abs()))
                    penalty += curr_penalty.sum()
    # s1,s2 are the distances of the intersections from the middle of the corresponding segments
    # # if the intersection is in both segment (|s1| < 1 and |s2| < 1),
    # # then it is a collision between the cars and a penalty is added
    # # we used
    # # 1 + elu(1 - |s|) = 2 - |s|, if  |s| <=1 , exp(1 - |s|),  if |s| > 1,
    # # since its positive and monotone increasing in t - so we get lower penalty with larger |s|
    # # (higher |s| means farther from collision)
    # # the max penalty is with s==0 (mid segment collision)
    # # but when |s|>1, then we have an exp decaying function (decays rapidly when getting far from collision)
    # # note that since 1+elu(t)  is non-negative , this logic holds also for optimizing s1 and s2 jointly

    penalty /= batch_size  # we want average over batch_size
    assert not torch.isnan(penalty)
    return penalty
###############################################################################
