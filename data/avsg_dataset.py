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

import pickle
import numpy as np
from data.base_dataset import BaseDataset
from pathlib import Path
import sys
import pathlib

is_windows = hasattr(sys, 'getwindowsversion')
if is_windows:
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

#########################################################################################



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
def load_samples(self, inds):
    aaa = self.dataset_info
    agents_feat = None
    map_feat = None
    return {'agents_feat': agents_feat, 'map_feat': map_feat}


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
        info_file_path = Path(data_path, 'info_data').with_suffix('.pkl')
        with info_file_path.open('rb') as fid:
            dataset_info = pickle.load(fid)
            self.dataset_info = dataset_info
            print('Loaded dataset file ', data_path)
            print(f"Total number of scenes: {self.dataset_info['n_scenes']}")

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
        return self.load_samples([index])

    def data_collate(self, batch):        
        return self.load_samples(batch)

    def __len__(self):
        """Return the total number of scenes."""
        return len(self.dataset_info['n_scenes'])

#########################################################################################

