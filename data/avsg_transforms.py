import numpy as np
import torch

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
        agents_num = min(agents_num_orig, self.max_num_agents)
        inds = np.arange(agents_num)
        if self.shuffle_agents_inds_flag:
            np.random.shuffle(inds)
        sample['agents_feat']['agents_num'] = agents_num
        sample['agents_feat']['agents_data'] = sample['agents_feat']['agents_data'][inds]
        return sample


#########################################################################################


class PreprocessSceneData(object):
    def __init__(self, opt):
        self.device = opt.device
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
        agents_num_orig = agents_feat['agents_num']
        if self.augmentation_type == 'none':
            pass
        elif self.augmentation_type == 'rotate_and_translate':

            # --------------------------------------
            # Random augmentation: rotation & translation
            # --------------------------------------
            aug_rot = np.random.rand(1).squeeze() * 2 * np.pi
            rot_mat = np.array([[np.cos(aug_rot), -np.sin(aug_rot)],
                                [np.sin(aug_rot), np.cos(aug_rot)]])
            rot_mat = torch.from_numpy(rot_mat).to(device=self.device, dtype=torch.float32)

            pos_shift_std = 50  # [m]
            pos_shift = torch.rand(2, device=self.device) * pos_shift_std

            for i_agent, ag in enumerate(agents_feat_vecs):
                # Rotate the centroid (x,y)
                ag[:2] = rot_mat @ ag[:2]
                # Rotate the yaw angle (in unit vec form)
                ag[2:4] = rot_mat @ ag[2:4]
                # Translate centroid
                ag[:2] += pos_shift
                sample['agents_feat']['agents_data'][i_agent] = ag

            for poly_type in self.polygon_name_order:
                elems = map_feat[poly_type]
                for i_elem, poly_elem in enumerate(elems):
                    poly_elem = (rot_mat @ poly_elem) + pos_shift
                    sample['map_feat'][poly_type][i_elem] = poly_elem

        elif self.augmentation_type == 'Gaussian_data':
            # Replace all the agent features data to gaussian samples... for debug
            agents_feat_vecs = agents_feat_vecs * 0 + torch.randn_like(agents_feat_vecs)
            sample['agents_feat']['agents_data'] = agents_feat_vecs
            # Set zero to all map features
            for poly_type in opt.polygon_name_order:
                elems = scenes_batch['map_feat'][poly_type][i_scene]
                for i_elem, poly_elem in elems:
                    for i_point, point in enumerate(poly_elem):
                        if scenes_batch['map_feat'][poly_type + '_valid'][i_scene, i_elem, i_point]:
                            scenes_batch['map_feat'][poly_type][i_scene, i_elem, i_point] *= 0

        else:
            raise NotImplementedError(f'Unrecognized opt.augmentation_type  {opt.augmentation_type}')
        conditioning = {'map_feat': scenes_batch['map_feat'], 'n_actors_in_scene': scenes_batch['n_actors_in_scene']}
        real_actors = scenes_batch['agents_feat']
        # sample['agents_feat']['agents_num'] = agents_num
        # sample['agents_feat']['agents_data'] = sample['agents_feat']['agents_data'][inds]

        conditioning = {}
        sample = {'conditioning':  conditioning,
                  'agents_feat_vecs': agents_feat_vecs}
        return sample
