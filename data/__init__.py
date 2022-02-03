"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""

from data.base_dataset import BaseDataset

def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = get_dataset_class_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def get_dataset_class_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    if dataset_name == 'avsg':
        from data.avsg_dataset import AvsgDataset
        dataset_class = AvsgDataset
    else:
        raise NotImplementedError()
    return dataset_class



def get_collate_fn_using_name(dataset_name, opt):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    if dataset_name == 'avsg':
        from data.avsg_dataset import avsg_data_collate
        collate_fn = lambda batch: avsg_data_collate(batch, opt)
    else:
        raise NotImplementedError()
    return collate_fn
