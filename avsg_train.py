"""General-purpose training script for image-to-image translation.

* To run training:
$ python -m avsg_train
 --dataset_mode avsg  --model avsg --data_path_train datasets/avsg_data/l5kit_sample.pkl  --data_path_val datasets/avsg_data/l5kit_sample.pkl

* Replace l5kit_sample.pkl with l5kit_train.pkl or l5kit_train_full.pkl for larger datasets

* To run only on CPU add: --gpu_ids -1

* To use wandb logging,
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
from data.avsg_dataset import get_cyclic_data_generator
from models import create_model
from options.train_options import TrainOptions
from util.visualizer import Visualizer

if __name__ == '__main__':
    run_start_time = time.time()
    opt = TrainOptions().parse()  # get training options
    train_data_gen = get_cyclic_data_generator(opt, data_root=opt.data_path_train)
    val_data_gen = get_cyclic_data_generator(opt, data_root=opt.data_path_val)

    model = create_model(opt)  # create a model given opt.model and other options
    opt.device = model.device
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    start_time = time.time()
    for i in range(opt.n_iters):
        iter_start_time = time.time()  # timer for entire epoch

        model.train()
        model.optimize_discriminator(train_data_gen, opt)
        model.optimize_generator(train_data_gen, opt)

        # update learning rates (must be after first model update step):
        model.update_learning_rates()

        # print training losses and save logging information to the log file and wandb charts:
        if i % opt.print_freq == 0:
            visualizer.print_current_metrics(model, opt, conditioning, val_data_gen, i_epoch, i_batch,
                                             tot_iters, run_start_time)
        # Display visualizations:
        if i > 0 and i % opt.display_freq == 0:
            visualizer.display_current_results(model, real_actors, conditioning, val_data_gen, opt, i_epoch,
                                               i_batch, tot_iters)

        # cache our latest model every <save_latest_freq> iterations:
        if i > 0 and i % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, tot_iters %d)' % (i_epoch, tot_iters))
            save_suffix = 'iter_%d' % i if opt.save_by_iter else 'latest'
            model.save_networks(save_suffix)


        print(f'End of iteration {i+1}/{opt.n_iters}'
              f', epoch run time {(time.time() - epoch_start_time):.2f} sec')
