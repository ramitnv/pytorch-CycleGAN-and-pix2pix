"""Dataset class

    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of samples.
"""

import pathlib
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from data.avsg_transforms import SelectAgents, PreprocessSceneData, ReadAgentsVecs, sample_sanity_check
from data.base_dataset import BaseDataset

is_windows = hasattr(sys, 'getwindowsversion')
if is_windows:
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

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

        # ~~~~  Agents features
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

        parser.add_argument('--max_num_agents', type=int, default=4, help=' number of agents in a scene')

        parser.add_argument('--augmentation_type', type=str, default='rotate_and_translate',
                            help=" 'none' | 'rotate_and_translate")
        parser.add_argument('--shuffle_agents_inds_flag', type=int, default=1, help="")
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
        self.data_path = data_path
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        info_file_path = Path(data_path, 'info').with_suffix('.pkl')
        with info_file_path.open('rb') as fid:
            dataset_info = pickle.load(fid)
        self.dataset_props = dataset_info['dataset_props']
        self.saved_mats_info = dataset_info['saved_mats_info']
        self.n_scenes = self.dataset_props['n_scenes']
        print('Loaded dataset file ', data_path)
        print(f"Total number of scenes loaded: {self.dataset_props['n_scenes']}")
        opt.polygon_types = self.dataset_props['polygon_types']
        opt.closed_polygon_types = self.dataset_props['closed_polygon_types']
        opt.agent_feat_vec_dim = len(opt.agent_feat_vec_coord_labels)
        self.transforms = [SelectAgents(opt), ReadAgentsVecs(opt, self.dataset_props), PreprocessSceneData(opt)]

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
        file_path = Path(self.data_path, 'data').with_suffix('.h5')
        with h5py.File(file_path, 'r') as h5f:
            for mat_name, mat_info in saved_mats_info.items():
                mat_sample = np.array(h5f[mat_name][index])
                mat_sample = torch.from_numpy(mat_sample).to(device=self.device)
                if mat_info['entity'] == 'map':
                    map_feat[mat_name] = mat_sample
                else:
                    agents_feat[mat_name] = mat_sample
        sample = {'agents_feat': agents_feat, 'map_feat': map_feat}
        for fn in self.transforms:
            sample = fn(sample)

        assert sample_sanity_check(sample)
        return sample
    ########################################################################################

    def __len__(self):
        """Return the total number of scenes."""
        return self.n_scenes

#########################################################################################
