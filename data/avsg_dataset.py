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
import torch
from data.avsg_transforms import SetActorsNum, PreprocessSceneData

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
        # parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        # parser.set_defaults(max_dataset_size=float("inf"))  # specify dataset-specific default values
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
        info_file_path = Path(data_path, 'info_data').with_suffix('.pkl')
        with info_file_path.open('rb') as fid:
            dataset_info = pickle.load(fid)
            self.dataset_props = dataset_info['dataset_props']
            self.saved_mats_info = dataset_info['saved_mats_info']
            self.n_scenes = self.dataset_props['n_scenes']
            print('Loaded dataset file ', data_path)
            print(f"Total number of scenes: {self.n_scenes}")
        self.transforms = [SetActorsNum(opt), PreprocessSceneData(opt)]

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
        for mat_name, mat_info in saved_mats_info.items():
            data_shape = mat_info['shape']
            data_type = mat_info['dtype']
            # Load the memmap data in Read-only mode:
            file_path = Path(self.data_path, mat_name).with_suffix('.dat')
            sample_shape = data_shape[1:] # ize as the data matrix, but with only 1 scene
            n_bytes = data_type.itemsize
            offset = n_bytes * index
            memmap_arr = np.memmap(str(file_path),
                                   dtype=data_type,
                                   mode='r',
                                   shape=sample_shape,
                                   offset=offset)
            mat = torch.from_numpy(memmap_arr).to(device=self.device)
            if mat_info['entity'] == 'map':
                map_feat[mat_name] = mat
            else:
                agents_feat[mat_name] = mat
        sample = {'agents_feat': agents_feat, 'map_feat': map_feat}
        for fn in self.transforms:
            sample = fn(sample)
        return sample

    ########################################################################################

    def __len__(self):
        """Return the total number of scenes."""
        return self.n_scenes

#########################################################################################

