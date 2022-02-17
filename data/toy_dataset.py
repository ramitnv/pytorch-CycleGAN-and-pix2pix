"""Dataset class

    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of samples.
"""

import pickle
from pathlib import Path
import h5py
import numpy as np
import torch

from data.avsg_transforms import sample_sanity_check
from data.base_dataset import BaseDataset


#########################################################################################


class ToyDataset(BaseDataset):
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
        parser.add_argument('--map_data_type',
                            default='zeros',
                            type=str,
                            help='If "zero" then set map_feat=0, if a number, then load this scene index for all samples')

        parser.add_argument('--agent_feat_vec_coord_labels',
                            default=['centroid_x',  # [0]  Real number
                                     'centroid_y',  # [1]  Real number
                                     'yaw_cos',  # [2]  in range [-1,1],  sin(yaw)^2 + cos(yaw)^2 = 1
                                     'yaw_sin',  # [3]  in range [-1,1],  sin(yaw)^2 + cos(yaw)^2 = 1
                                     'speed',  # [4] Real non-negative
                                     ],
                            type=list)
        # This agent extent is used (instead of being loaded from data)  if  agent_feat_vec_coord_labels
        # does not include extent_length and extent_width features
        parser.add_argument('--default_agent_extent_length', type=float, default=4.)  # [m]
        parser.add_argument('--default_agent_extent_width', type=float, default=1.5)  # [m]

        parser.add_argument('--max_num_agents', type=int, default=4, help='')
        parser.add_argument('--num_agents', type=int, default=4, help=' number of agents in a scene')

        parser.add_argument('--theta_type', type=str, default='zero', help=' "zero" | "uniform')

        return parser

    #########################################################################################
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
        self.map_data_type = opt.map_data_type
        self.data_path = data_path
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        info_file_path = Path(data_path, 'info').with_suffix('.pkl')
        with info_file_path.open('rb') as fid:
            dataset_info = pickle.load(fid)
        self.dataset_props = dataset_info['dataset_props']
        self.saved_mats_info = dataset_info['saved_mats_info']
        self.n_scenes = self.dataset_props['n_scenes']
        print('Loaded dataset file ', data_path)
        print(f"Total number of scenes: {self.n_scenes}")
        opt.polygon_types = self.dataset_props['polygon_types']
        opt.closed_polygon_types = self.dataset_props['closed_polygon_types']
        opt.agent_feat_vec_dim = len(opt.agent_feat_vec_coord_labels)
        self.agent_feat_vec_dim = opt.agent_feat_vec_dim
        self.agent_feat_vec_coord_labels = opt.agent_feat_vec_coord_labels
        self.max_num_agents = opt.max_num_agents
        self.num_agents = opt.num_agents
        self.theta_type = opt.theta_type
        #########################################################################################

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
        dataset_props = self.dataset_props
        saved_mats_info = self.saved_mats_info
        agents_feat = {}
        map_feat = {}

        polygon_types = self.dataset_props['polygon_types']
        max_num_elem = self.dataset_props['max_num_elem']
        max_points_per_elem = self.dataset_props['max_points_per_elem']
        coord_dim = self.dataset_props['coord_dim']
        max_num_agents = self.max_num_agents
        num_agents = self.num_agents
        n_polygon_types = len(polygon_types)

        #####  Set map fields ['map_elems_points', 'map_elems_n_points_orig', 'map_elems_exists']
        if self.map_data_type == 'zeros':
            # Set zero to all map features
            map_feat['map_elems_points'] = torch.zeros((n_polygon_types, max_num_elem, max_points_per_elem, coord_dim),
                                                       dtype=torch.float32, device=self.device)
            map_feat['map_elems_n_points_orig'] = torch.zeros((n_polygon_types, max_num_elem),
                                                              dtype=torch.int16, device=self.device)
            map_feat['map_elems_n_points_orig'] = torch.zeros((n_polygon_types, max_num_elem),
                                                              dtype=torch.bool, device=self.device)
        else:
            map_scene_idx = int(self.map_data_type)
            file_path = Path(self.data_path, 'data').with_suffix('.h5')
            with h5py.File(file_path, 'r') as h5f:
                for mat_name, mat_info in saved_mats_info.items():
                    mat_sample = np.array(h5f[mat_name][map_scene_idx])
                    mat_sample = torch.from_numpy(mat_sample).to(device=self.device)
                    if mat_info['entity'] == 'map':
                        map_feat[mat_name] = mat_sample

        ##### Set agents_feat fields ['agents_feat_vecs', 'agents_num', 'agents_exists']
        x_range = (-20, 20)
        y_range = (-20, 20)
        agents_feat_vecs = torch.zeros((max_num_agents, self.agent_feat_vec_dim), dtype=torch.float32, device=self.device)
        coord_labels = self.agent_feat_vec_coord_labels
        agents_feat_vecs[:num_agents, coord_labels.index('centroid_x')] \
            = x_range[0] + (x_range[1] - x_range[0]) * torch.rand(num_agents, dtype=torch.float32, device=self.device)
        agents_feat_vecs[:num_agents, coord_labels.index('centroid_y')]\
            = y_range[0] + (y_range[1] - y_range[0]) * torch.rand(num_agents, dtype=torch.float32, device=self.device)

        if self.theta_type == 'uniform':
            thetas = 2 * torch.pi * torch.rand(num_agents, dtype=torch.float32, device=self.device)  # uniform angle
        elif self.theta_type == 'zero':
            thetas = torch.zeros(num_agents, dtype=torch.float32, device=self.device)  # zero angle
        else:
            raise NotImplementedError
        agents_feat_vecs[:num_agents, coord_labels.index('yaw_cos')] = torch.cos(thetas)
        agents_feat_vecs[:num_agents, coord_labels.index('yaw_sin')] = torch.sin(thetas)

        agents_feat_vecs[:num_agents, coord_labels.index('speed')] = torch.ones(num_agents, dtype=torch.float32, device=self.device)

        sample = {'agents_feat': agents_feat, 'map_feat': map_feat}
        assert sample_sanity_check(sample)
        return sample

    ########################################################################################

    def __len__(self):
        """Return the total number of scenes."""
        return self.n_scenes

#########################################################################################
