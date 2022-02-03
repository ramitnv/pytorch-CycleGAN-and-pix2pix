"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import os.path
import pickle
import numpy as np
import torch
from data.base_dataset import BaseDataset

#########################################################################################

def agents_feat_dicts_to_vecs(agents_feat_dicts, opt):
    dim_agent_feat_vec = len(opt.agent_feat_vec_coord_labels)
    assert opt.agent_feat_vec_coord_labels == ['centroid_x', 'centroid_y', 'yaw_cos', 'yaw_sin',
                                               'extent_length', 'extent_width', 'speed',
                                               'is_CAR', 'is_CYCLIST', 'is_PEDESTRIAN']
    agents_feat_vecs = []
    for agent_dict in agents_feat_dicts:
        agent_feat_vec = torch.zeros(dim_agent_feat_vec, device=opt.device)
        agent_feat_vec[0] = agent_dict['centroid'][0]
        agent_feat_vec[1] = agent_dict['centroid'][1]
        agent_feat_vec[2] = np.cos(agent_dict['yaw'])
        agent_feat_vec[3] = np.sin(agent_dict['yaw'])
        agent_feat_vec[4] = float(agent_dict['extent'][0])
        agent_feat_vec[5] = float(agent_dict['extent'][1])
        agent_feat_vec[6] = float(agent_dict['speed'])
        # agent type ['CAR', 'CYCLIST', 'PEDESTRIAN'] is represented in one-hot encoding
        agent_feat_vec[7] = agent_dict['agent_label_id'] == 0
        agent_feat_vec[8] = agent_dict['agent_label_id'] == 1
        agent_feat_vec[9] = agent_dict['agent_label_id'] == 2
        assert agent_feat_vec[7:].sum() == 1
        agents_feat_vecs.append(agent_feat_vec)
    agents_feat_vecs = torch.stack(agents_feat_vecs)
    return agents_feat_vecs


#########################################################################################
#########################################################################################


def select_actors_from_scene(agents_feat_dicts, opt):
    # Take the max_num_agents closest to ego, and shuffle their order
    max_num_agents = opt.max_num_agents

    actors_dists_to_ego = [np.linalg.norm(agent_dict['centroid'][:]) for agent_dict in agents_feat_dicts]

    agents_dists_order = np.argsort(actors_dists_to_ego)

    inds = agents_dists_order[:max_num_agents]  # take the closest agent to the ego
    np.random.shuffle(inds)  # shuffle so that the ego won't always be first

    return inds


#########################################################################################

def avsg_data_collate(batch, opt):
    c_batch = dict()
    batch_size = len(batch)
    dim_agent_feat_vec = len(opt.agent_feat_vec_coord_labels)
    n_actors_in_scene = []
    agent_inds_per_scene = []
    for i_scene, samp in enumerate(batch):
        inds = select_actors_from_scene(samp['agents_feat'], opt)
        agent_inds_per_scene.append(inds)
        n_actors_in_scene.append(len(inds))

    max_n_actors = max(n_actors_in_scene)
    c_batch['agents_feat'] = torch.zeros((batch_size, max_n_actors, dim_agent_feat_vec), device=opt.device)
    for i_scene, samp in enumerate(batch):
        agents_feat_dicts = samp['agents_feat']
        inds = agent_inds_per_scene[i_scene]
        agents_feat_vecs = agents_feat_dicts_to_vecs([agents_feat_dicts[i] for i in inds], opt)
        c_batch['agents_feat'][i_scene, :agents_feat_vecs.shape[0], :] = agents_feat_vecs

    c_batch['n_actors_in_scene'] = n_actors_in_scene
    c_batch['map_feat'] = dict()
    for poly_type in opt.polygon_name_order:
        max_n_elem = max([len(x['map_feat'][poly_type]) for x in batch])
        max_n_points = 0
        for i_scene, samp in enumerate(batch):
            for i_elem, elem_points in enumerate(samp['map_feat'][poly_type]):
                max_n_points = max(max_n_points, len(elem_points))
        c_batch['map_feat'][poly_type] = torch.zeros((batch_size, max_n_elem, max_n_points, 2), device=opt.device)
        c_batch['map_feat'][poly_type + '_valid'] = torch.zeros((batch_size, max_n_elem, max_n_points),
                                                                device=opt.device, dtype=torch.bool)
        for i_scene, samp in enumerate(batch):
            for i_elem, elem_points in enumerate(samp['map_feat'][poly_type]):
                c_batch['map_feat'][poly_type][i_scene, i_elem, :elem_points.shape[0], :] = elem_points
                c_batch['map_feat'][poly_type + '_valid'][i_scene, i_elem, :elem_points.shape[0]] = True
    return c_batch


#########################################################################################


class AvsgDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        # parser.set_defaults(max_dataset_size=float("inf"))  # specify dataset-specific default values
        return parser

    def __init__(self, opt, data_path):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        info_file_path = os.path.join(data_path, 'info_data.pkl')
        with open(info_file_path, 'rb') as fid:
            self.dataset = pickle.loads(fid.read())
            print('Loaded dataset file ', data_path)

            # print(f"Total number of scenes: {self.dataset.n_scenes}")

            # for k, v in self.dataset.items():
            #     if len(v) > opt.max_dataset_size:
            #         print(f"Field {k} is truncated from {len(v)} to {opt.max_dataset_size}")
            #         self.dataset[k] = self.dataset[k][:opt.max_dataset_size]

        # get the image paths of your dataset;
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        # self.transform = get_transform(opt)
        self.transform = []


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helper functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        agents_feat = self.dataset['agents_feat'][index]
        map_feat = self.dataset['map_feat'][index]

        return {'agents_feat': agents_feat, 'map_feat': map_feat}

    def __len__(self):
        """Return the total number of scenes."""
        return len(self.dataset['map_feat'])

#########################################################################################

