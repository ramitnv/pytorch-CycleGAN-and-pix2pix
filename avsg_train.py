"""General-purpose training script for image-to-image translation.

* To run training:
$ python -m avsg_train
 --dataset_mode avsg  --model avsg --dataroot datasets/avsg_data/l5kit_sample.pkl  --data_eval datasets/avsg_data/l5kit_sample.pkl

* Replace l5kit_sample.pkl with l5kit_train.pkl or l5kit_train_full.pkl for larger datasets

* To run only on CPU add: --gpu_ids -1

* To yse wandb logging,
run $ wandb login
and add run parameter --use_wandb

* To limit the datasets size --max_dataset_size 1000

* Name the experiment with --name



This script works for various models (with option '--model') and
different datasets (with option '--dataset_mode').
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').
It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.


Note: if you get CUDA Unknown error, try $ apt-get install nvidia-modprobe
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from avsg_utils import pre_process_scene_data
import itertools

if __name__ == '__main__':
    run_start_time = time.time()
    opt = TrainOptions().parse()  # get training options
    train_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(train_dataset)  # get the number of images in the dataset.
    print('The number of training samples = %d' % dataset_size)
    opt.dataroot = opt.data_eval
    eval_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    eval_dataset_size = len(eval_dataset)  # get the number of images in the dataset.
    print('The number of test samples = %d' % eval_dataset_size)
    validation_data_gen = itertools.cycle(eval_dataset)

    model = create_model(opt)  # create a model given opt.model and other options
    opt.device = model.device
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    tot_iters = 0  # the total number of training iterations
    start_time = time.time()
    for i_epoch in range(opt.start_epoch,
                         opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <start_epoch>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        for i_batch, scenes_batch in enumerate(train_dataset):  # inner loop within one epoch

            # unpack data from dataset and apply preprocessing:
            real_actors, conditioning = pre_process_scene_data(scenes_batch, opt)

            model.train()

            # calculate loss functions, get gradients, update network weights:
            model.optimize_parameters(real_actors, conditioning)

            # update learning rates (must be after first model update step):
            model.update_learning_rate()

            # print training losses and save logging information to the log file and wandb charts:
            if tot_iters % opt.print_freq == 0:
                visualizer.print_current_metrics(model, opt, conditioning, validation_data_gen, i_epoch, i_batch,
                                                 tot_iters, run_start_time)
            # Display visualizations:
            if tot_iters > 0 and tot_iters % opt.display_freq == 0:
                visualizer.display_current_results(model,  real_actors, conditioning, validation_data_gen, opt, i_epoch,
                                                   i_batch, tot_iters)

            # cache our latest model every <save_latest_freq> iterations:
            if tot_iters > 0 and tot_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, tot_iters %d)' % (i_epoch, tot_iters))
                save_suffix = 'iter_%d' % tot_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            tot_iters += 1

        # cache our model every <save_epoch_freq> epochs:
        if i_epoch > 0 and i_epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (i_epoch, tot_iters))
            model.save_networks('latest')
            model.save_networks(i_epoch)

        print(f'End of epoch {i_epoch+1}/{opt.n_epochs + opt.n_epochs_decay}'
              f', epoch run time {(time.time() - epoch_start_time):.2f} sec')
