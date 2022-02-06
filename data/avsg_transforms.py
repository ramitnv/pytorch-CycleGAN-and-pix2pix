import numpy as np
import torch
from util.util import to_num


#########################################################################################

class SetActorsNum(object):
    """
    take the closest max_num_agents agents to the ego (the agents are ordered this way in the data)
    also shuffle their indexing
    """

    def __init__(self, opt):
        self.max_num_agents = opt.max_num_agents
        self.shuffle_agents_inds_flag = opt.shuffle_agents_inds_flag
        self.agent_feat_vec_coord_labels = opt.agent_feat_vec_coord_labels

    def __call__(self, sample, dataset_props):
        agents_feat = sample['agents_feat']
        agents_num_orig = agents_feat['agents_num']
        agents_feat_vecs_orig = agents_feat['agents_feat_vecs']
        # agents_exists_orig = agents_feat['agents_exists']
        device = agents_feat_vecs_orig.device
        agent_feat_dim = agents_feat_vecs_orig.shape[1]
        agents_num = min(agents_num_orig, self.max_num_agents)
        inds = np.arange(to_num(agents_num))
        if self.shuffle_agents_inds_flag:
            np.random.shuffle(inds)
        agens_feat_vecs = torch.zeros((self.max_num_agents, agent_feat_dim),
                                      device=device)
        agens_feat_vecs[:agents_num] = agents_feat_vecs_orig[inds]
        agents_exists = torch.zeros(self.max_num_agents, device=device, dtype=torch.bool)
        agents_exists[:agents_num] = torch.tensor(True)
        sample['agents_feat']['agents_feat_vecs'] = agens_feat_vecs
        sample['agents_feat']['agents_num'] = agents_num * torch.ones_like(agents_num_orig)
        sample['agents_feat']['agents_exists'] = agents_exists
        return sample


#########################################################################################


class PreprocessSceneData(object):
    def __init__(self, opt):
        self.agent_feat_vec_coord_labels = opt.agent_feat_vec_coord_labels
        assert opt.agent_feat_vec_coord_labels == [
            'centroid_x',  # [0]  Real number
            'centroid_y',  # [1]  Real number
            'yaw_cos',  # [2]  in range [-1,1],  sin(yaw)^2 + cos(yaw)^2 = 1
            'yaw_sin',  # [3]  in range [-1,1],  sin(yaw)^2 + cos(yaw)^2 = 1
            'speed',  # [4] Real non-negative
        ]
        self.dim_agent_feat_vec = len(self.agent_feat_vec_coord_labels)
        self.augmentation_type = opt.augmentation_type
        self.polygon_name_order = opt.polygon_name_order

    def __call__(self, sample, dataset_props):

        agents_feat = sample['agents_feat']
        map_feat = sample['map_feat']
        agents_feat_vecs_orig = agents_feat['agents_feat_vecs']
        agents_exists = agents_feat['agents_exists']
        agents_num = agents_feat['agents_num']
        device = agents_feat_vecs_orig.device

        dataset_agent_feat_vec_coord_labels = dataset_props['agent_feat_vec_coord_labels']
        agents_feat_vecs = torch.zeros((agents_num, self.dim_agent_feat_vec), device=device)
        for i_coord, label in enumerate(self.agent_feat_vec_coord_labels):
            i_coord_orig = dataset_agent_feat_vec_coord_labels.index(label)
            agents_feat_vecs[:, i_coord] = agents_feat_vecs_orig[:, i_coord_orig]

        if self.augmentation_type == 'none':
            pass
        elif self.augmentation_type == 'rotate_and_translate':

            # --------------------------------------
            # Random augmentation: rotation & translation
            # --------------------------------------

            aug_rot = np.random.rand(1).squeeze() * 2 * np.pi
            rot_mat = np.array([[np.cos(aug_rot), -np.sin(aug_rot)],
                                [np.sin(aug_rot), np.cos(aug_rot)]])
            rot_mat = torch.from_numpy(rot_mat).to(device=device)

            pos_shift_std = 50  # [m]
            pos_shift = torch.randn_like(agents_feat_vecs[0][:2]) * pos_shift_std

            for i_agent, feat_vec in enumerate(agents_feat_vecs):
                # Rotate the centroid (x,y)
                feat_vec[:2] = rot_mat @ feat_vec[:2]
                # Rotate the yaw angle (in unit vec form)
                feat_vec[2:4] = rot_mat @ feat_vec[2:4]
                # Translate centroid
                feat_vec[:2] += pos_shift
                agents_feat_vecs[i_agent] = feat_vec

            map_elems_points = map_feat['map_elems_points']
            x_vals = map_elems_points[:, :, :, 0]
            y_vals = map_elems_points[:, :, :, 1]
            x_vals, y_vals = rot_mat[0, 0] * x_vals + rot_mat[0, 1] * y_vals, \
                             rot_mat[1, 0] * x_vals + rot_mat[1, 1] * y_vals,
            x_vals, y_vals = x_vals + pos_shift[0], y_vals + pos_shift[1]
            map_feat['map_elems_points'][:, :, :, 0] = x_vals
            map_feat['map_elems_points'][:, :, :, 1] = y_vals

        elif self.augmentation_type == 'Gaussian_data':
            # Replace all the agent features data to gaussian samples... for debug
            agents_feat_vecs = agents_feat_vecs * 0 + torch.randn_like(agents_feat_vecs)
            # Set zero to all map features
            map_feat['map_elems_points'] *= 0
        else:
            raise NotImplementedError(f'Unrecognized opt.augmentation_type  {self.augmentation_type}')

        conditioning = {'map_feat': map_feat, 'n_agents_in_scene': agents_num, 'agents_exists': agents_exists}
        sample = {'conditioning': conditioning, 'agents_feat_vecs': agents_feat_vecs}
        return sample
