''''

Run using:
 $ python -m run_check_map_enc
 --dataset_mode avsg  --model avsg_check_map_enc --data_path_train datasets/avsg_data/l5kit_train.pkl --data_path_val datasets/avsg_data/l5kit_sample.pkl
* To change dataset files change --data_path_train and --data_path_val
* To run only on CPU add: --gpu_ids -1
* Name the experiment with --name

Note: if you get CUDA Uknown error, try $ apt-get install nvidia-modprobe
'''
import os
import time

from data.data_func import create_dataloader
from models import create_model
from options.train_options import TrainOptions

if __name__ == '__main__':


    opt = TrainOptions().parse()  # get training options
    assert opt.model == 'avsg_check_map_enc'
    assert os.path.isfile(opt.data_path_val)
    train_dataset = create_dataloader(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(train_dataset)  # get the number of images in the dataset.
    print('The number of training samples = %d' % dataset_size)
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    n_epochs = opt.n_epochs
    print('Number of training epochs: ', n_epochs)
    ##########
    # Train
    #########
    iter_print_freq = 20
    start_time = time.time()
    total_iter = 0
    for i_epoch in range(n_epochs):
        for i_batch, data in enumerate(train_dataset):
            # unpack data from dataset and apply preprocessing
            is_valid = model.set_input(data)
            if not is_valid:
                # if the data sample is not valid to use
                continue
            # calculate loss functions, get gradients, update network weights
            model.optimize_parameters()
            # update learning rates *after* first step (https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
            model.update_learning_rate()
            if total_iter % iter_print_freq == 0:
                model.set_input(data)
                model.forward()  # run inference
                loss = model.loss_criterion(model.prediction, model.ground_truth)
                print(f'Epoch {i_epoch}, batch {i_batch}, total_iter {total_iter},  loss {loss}')
            total_iter += 1
        print(f'End of epoch {i_epoch}, elapsed time {time.time() - start_time}')

    ##########
    # Test
    ##########
    del train_dataset
    model.eval()
    eval_dataset = create_dataloader(opt, data_root=opt.data_path_val)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(eval_dataset)  # get the number of images in the dataset.
    print('The number of test samples = %d' % dataset_size)
    n_loss_calc = 0
    loss_sum = 0
    for i, data in enumerate(eval_dataset):
        model.set_input(data)  # unpack data from data loader
        model.forward()  # run inference
        loss = model.loss_criterion(model.prediction, model.ground_truth)
        loss_sum += loss
        n_loss_calc += 1
    loss_avg = loss_sum / n_loss_calc
    print(f'Average test loss over  {dataset_size} samples is loss {loss_avg}')
