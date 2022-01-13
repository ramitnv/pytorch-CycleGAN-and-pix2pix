
import torch.utils.data

from . import get_dataset_class_using_name, get_collate_fn_using_name


def create_dataloader(opt, data_path):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    """
    dataset_class = get_dataset_class_using_name(opt.dataset_mode)
    collate_fn = get_collate_fn_using_name(opt.dataset_mode, opt)
    dataset_object = dataset_class(opt, data_path)
    print("dataset [%s] was created" % type(dataset_object).__name__)
    data_loader = torch.utils.data.DataLoader(
        dataset_object,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.num_threads),
        collate_fn=collate_fn)
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
