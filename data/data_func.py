import torch
import torch.utils.data as data_utils
from . import get_dataset_class_using_name


def create_dataloader(opt, data_path):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    """
    dataset_class = get_dataset_class_using_name(opt.dataset_mode)
    dataset_obj = dataset_class(opt, data_path)

    if opt.data_size_limit > 0:
        indices = torch.randperm(len(dataset_obj))[:opt.data_size_limit]
        dataset_obj = data_utils.Subset(dataset_obj, indices)
        print(f'Dataset reduced to {len(dataset_obj)} scenes')

    print(f"dataset [{type(dataset_obj).__name__}] was created, data loaded from {data_path}")
    data_loader = data_utils.DataLoader(
        dataset_obj,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.num_threads))
    data_iterator = iter(data_loader)
    data_gen = {'data_loader': data_loader, 'data_iterator': data_iterator}
    return data_gen


def get_next_batch_cyclic(data_gen):
    """ get sample from iterator, if it finishes then restart  """
    data_loader = data_gen['data_loader']
    data_iterator = data_gen['data_iterator']
    try:
        batch_data = next(data_iterator)
    except StopIteration:
        #  just restart the iterator and re-use the samples
        data_iterator = iter(data_loader)
        batch_data = next(data_iterator)
    return batch_data
