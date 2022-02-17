import numpy as np


#########################################################################################


def get_single_conditioning_from_batch(conditioning_batch, i_map):
    map_feat_batch = conditioning_batch['map_feat']
    # get single map from batch:
    map_feat = {k: map_feat_batch[k][i_map].unsqueeze(0) for k in map_feat_batch.keys()}
    conditioning = {'map_feat': map_feat,
                    'n_agents_in_scene': conditioning_batch['n_agents_in_scene'][i_map].unsqueeze(0),
                    'agents_exists':  conditioning_batch['agents_exists'][i_map].unsqueeze(0)}
    return conditioning

#########################################################################################

def agents_feat_vecs_to_dicts(agents_feat_vecs, agents_exists, opt):
    assert agents_feat_vecs.ndim == 3  # [n_scenes==1 x max_n_agents x feat_dim]
    assert agents_feat_vecs.shape[0] == 1
    assert agents_exists.shape[0] == 1

    # code now supports only this feature format:
    assert opt.agent_feat_vec_coord_labels == [
        'centroid_x',  # [0]  Real number
        'centroid_y',  # [1]  Real number
        'yaw_cos',  # [2]  in range [-1,1],  sin(yaw)^2 + cos(yaw)^2 = 1
        'yaw_sin',  # [3]  in range [-1,1],  sin(yaw)^2 + cos(yaw)^2 = 1
        'speed',  # [4] Real non-negative
    ]
    agents_feat_dicts = []
    for i_agent, is_exist in enumerate(agents_exists[0]):
        if not is_exist:
            continue
        agent_feat_vec = agents_feat_vecs[0, i_agent].detach().cpu().numpy()
        agent_feat_dict = ({'centroid': agent_feat_vec[:2],
                            'yaw': np.arctan2(agent_feat_vec[3], agent_feat_vec[2]),
                            'speed': agent_feat_vec[4],
                            'extent': np.array([opt.default_agent_extent_length, opt.default_agent_extent_width]),
                            'agent_label_id': 0  # CAR
                            })
        agents_feat_dicts.append(agent_feat_dict)
    return agents_feat_dicts


#########################################################################################

def get_agents_descriptions(agents_feat_dicts):
    txt_descript = []
    for i, ag in enumerate(agents_feat_dicts):
        x, y = ag['centroid']
        yaw_deg = np.degrees(ag['yaw'])
        length, width = ag['extent']
        if ag['agent_label_id'] == 0:
            type_label = 'Car'
        elif ag['agent_label_id'] == 1:
            type_label = 'Cyclist'
        elif ag['agent_label_id'] == 2:
            type_label = 'Pedestrian'
        else:
            raise ValueError
        txt_descript.append(
            f"#{i}, {type_label}, ({x:.1f},{y:.1f}), {yaw_deg:.1f}\u00B0, {length:.1f}\u00D7{width:.1f}")
    return txt_descript

#########################################################################################
#
# def calc_agents_feats_stats(dataset, agent_feat_vec_coord_labels, device, num_agents, polygon_types):
#     ##### Find data normalization parameters
#
#     dim_agent_feat_vec = len(agent_feat_vec_coord_labels)
#     sum_agent_feat = torch.zeros(dim_agent_feat_vec, device=device)
#     count = 0
#     for scene in dataset:
#         agents_feat_dict = scene['agents_feat']
#         is_valid, agents_feat_vec = filter_and_preprocess_agent_feat(agents_feat_dict, num_agents, agent_feat_vec_coord_labels, device)
#         if is_valid:
#             sum_agent_feat += agents_feat_vec.sum(dim=0)  # sum all agents in the scene
#             count += agents_feat_vec.shape[0]  # count num agents summed
#     agent_feat_mean = sum_agent_feat / count  # avg across all agents in all scenes
#
#     count = 0
#     sum_sqr_div_agent_feat = torch.zeros(dim_agent_feat_vec, device=device)
#     for scene in dataset:
#         agents_feat_dict = scene['agents_feat']
#         is_valid, agents_feat_vec = filter_and_preprocess_agent_feat(agents_feat_dict, num_agents, agent_feat_vec_coord_labels, device)
#         if is_valid:
#             count += agents_feat_vec.shape[0]  # count num agents summed
#             sum_sqr_div_agent_feat += torch.sum(
#                 torch.pow(agents_feat_vec - agent_feat_mean, 2), dim=0)  # sum all agents in the scene
#     agent_feat_std = torch.sqrt(sum_sqr_div_agent_feat / count)
#
#     return agent_feat_mean, agent_feat_std
#########################################################################################
#
#
# def get_normalized_agent_feat(self, feat):
#     nrm_feat = torch.clone(feat)
#     nrm_feat[:, self.agent_feat_to_nrm] -= self.agent_feat_mean[self.agent_feat_to_nrm]
#     nrm_feat[:, self.agent_feat_to_nrm] /= self.agent_feat_std[self.agent_feat_to_nrm]
#     return nrm_feat
#########################################################################################
