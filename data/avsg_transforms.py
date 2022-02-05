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

    def __call__(self, sample):
        agents_feat = sample['agents_feat']
        agents_num_orig = agents_feat['agents_num']
        agents_feat_vecs = agents_feat['agents_data']
        device = agents_feat_vecs.device
        agent_feat_dim = agents_feat_vecs.shape[1]
        agents_num = min(agents_num_orig, self.max_num_agents)
        inds = np.arange(to_num(agents_num))
        if self.shuffle_agents_inds_flag:
            np.random.shuffle(inds)
        sample['agents_feat']['agents_num'] = agents_num * torch.ones_like(agents_num_orig)
        agens_feat_vecs = torch.zeros((self.max_num_agents, agent_feat_dim),
                                      device=device)
        agens_feat_vecs[:agents_num] = sample['agents_feat']['agents_data'][inds]
        sample['agents_feat']['agents_data'] = agens_feat_vecs
        return sample

#########################################################################################


class PreprocessSceneData(object):
    def __init__(self, opt):
        # We assume this order of coordinates:
        assert opt.agent_feat_vec_coord_labels == ['centroid_x', 'centroid_y',
                                                   'yaw_cos', 'yaw_sin',
                                                   'extent_length', 'extent_width', 'speed',
                                                   'is_CAR', 'is_CYCLIST', 'is_PEDESTRIAN']
        self.augmentation_type = opt.augmentation_type
        self.polygon_name_order = opt.polygon_name_order

    def __call__(self, sample):
        agents_feat = sample['agents_feat']
        map_feat = sample['map_feat']
        agents_feat_vecs = agents_feat['agents_data']
        agents_num = agents_feat['agents_num']
        device = agents_feat_vecs.device
        if self.augmentation_type == 'none':
            pass
        elif self.augmentation_type == 'rotate_and_translate':

            # --------------------------------------
            # Random augmentation: rotation & translation
            # --------------------------------------

            aug_rot = np.random.rand(1).squeeze() * 2 * np.pi
            rot_mat = np.array([[np.cos(aug_rot), -np.sin(aug_rot)],
                                [np.sin(aug_rot), np.cos(aug_rot)]])
            rot_mat = torch.from_numpy(rot_mat).to(device=device,
                                                   dtype=torch.float32)

            pos_shift_std = 50  # [m]
            pos_shift = torch.randn_like(agents_feat_vecs[0][:2]) * pos_shift_std

            for i_agent, ag in enumerate(agents_feat_vecs):
                # Rotate the centroid (x,y)
                ag[:2] = rot_mat @ ag[:2]
                # Rotate the yaw angle (in unit vec form)
                ag[2:4] = rot_mat @ ag[2:4]
                # Translate centroid
                ag[:2] += pos_shift
                sample['agents_feat']['agents_data'][i_agent] = ag

            map_elems_points = sample['map_feat']['map_elems_points']
            x_vals = map_elems_points[:, :, :, 0]
            y_vals = map_elems_points[:, :, :, 1]
            x_vals, y_vals = rot_mat[0, 0] * x_vals + rot_mat[0, 1] * y_vals, \
                             rot_mat[1, 0] * x_vals + rot_mat[1, 1] * y_vals,
            x_vals, y_vals = x_vals + pos_shift[0], y_vals + pos_shift[1]
            sample['map_feat']['map_elems_points'][:, :, :, 0] = x_vals
            sample['map_feat']['map_elems_points'][:, :, :, 1] = y_vals

        elif self.augmentation_type == 'Gaussian_data':
            # Replace all the agent features data to gaussian samples... for debug
            agents_feat_vecs = agents_feat_vecs * 0 + torch.randn_like(agents_feat_vecs)
            sample['agents_feat']['agents_data'] = agents_feat_vecs
            # Set zero to all map features
            sample['map_feat']['map_elems_points'] *= 0
        else:
            raise NotImplementedError(f'Unrecognized opt.augmentation_type  {self.augmentation_type}')

        conditioning = {'map_feat': sample['map_feat'],
                        'n_agents_in_scene': agents_num}
        sample = {'conditioning': conditioning,
                  'agents_feat_vecs': agents_feat_vecs}
        return sample
