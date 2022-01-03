import numpy as np
import os
import ntpath
import time
import torch
from . import util, html
from avsg_utils import agents_feat_vecs_to_dicts, pre_process_scene_data, get_agents_descriptions
from models.networks import cal_gradient_penalty
from util.avsg_visualization_utils import visualize_scene_feat

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

##############################################################################################

class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses   a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    # ==========================================================================

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """

        self.opt = opt  # cache the option
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.saved = False
        self.use_wandb = opt.use_wandb
        self.current_fig_index = (0, 0)
        self.plotted_inds = []

        if self.use_wandb:
            self.wandb_run = wandb.init(project='SceneGen', name=opt.name, config=opt) if not wandb.run else wandb.run
            self.wandb_run._label(repo='SceneGen')

        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # ==========================================================================

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    # ==========================================================================

    def display_current_results(self, model, train_dataset, eval_dataset, opt, i_epoch, i_epoch_iter, total_iters,
                                run_start_time, file_type='jpg'):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        fig_index = total_iters
        self.plotted_inds.append(fig_index)
        visuals_dict, wandb_logs = get_metrics_stats_and_images(model, train_dataset, eval_dataset, opt, i_epoch,
                                                                i_epoch_iter, total_iters, run_start_time)


        # save images to an HTML file if they haven't been saved.
        if self.use_html and not self.saved:
            self.saved = True
            # save images to the disk
            for label, image in visuals_dict.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, f'e{i_epoch+1}_i{i_epoch_iter+1}_{label}.{file_type}')
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=0)
            for ind in self.plotted_inds[::-1]:
                webpage.add_header(f'epoch {ind[0]}, iter {ind[1]}')
                ims, txts, links = [], [], []
                for label, image_numpy in visuals_dict.items():
                    img_path = f'e{ind[0]}_i{ind[1]}_{label}.{file_type}'
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

        print(f'Figure saved. epoch #{i_epoch}, epoch_iter #{i_epoch_iter}, total_iter #{total_iters}')

    # ==========================================================================

    def print_current_metrics(self, model, eval_dataset, i_epoch, i_epoch_iter, total_iters):
        """  print training losses and save logging information to the log file and wandblog charts


        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs

        """
        train_metrics = model.train_log_metrics_G | model.train_log_metrics_D

        scenes_batch = next(eval_dataset)
        real_actors, conditioning = pre_process_scene_data(scenes_batch, model.opt)
        _, val_metrics_G = model.get_G_losses(conditioning, real_actors)
        _, val_metrics_D = model.get_D_losses(conditioning, real_actors)

        val_metrics = val_metrics_G | val_metrics_D

        message = f'(epoch: {1 + i_epoch}, batch: {1 + i_epoch_iter}, tot_iters: {1+total_iters}) '
        message += 'Train: '
        for name, v in train_metrics.items():
            message += f'{name}: {v:.2f} '

        message += '\nValidation: '
        for name, v in val_metrics.items():
            message += f'{name}: {v:.2f} '

        # print the message
        print(message)

        # update wandb charts
        if self.use_wandb:
            for name, v in train_metrics.items():
                self.wandb_run.log({f'train/{name}': v})
            for name, v in val_metrics_G.items():
                self.wandb_run.log({f'val/{name}': v})

        # save log file
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

##############################################################################################


def get_metrics_stats_and_images(model, train_dataset, eval_dataset, opt, i_epoch, epoch_iter, total_iters,
                                 run_start_time):
    """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""

    datasets = {'train': train_dataset, 'val': eval_dataset}
    wandb_logs = {}
    visuals_dict = {}
    model.eval()
    stats_n_maps = opt.stats_n_maps  # how many maps to average over the metrics
    vis_n_maps = opt.vis_n_maps  # how many maps to visualize
    vis_n_generator_runs = opt.vis_n_generator_runs  # how many sampled fake agents per map to visualize
    g_var_n_generator_runs = opt.g_var_n_generator_runs  # how many sampled fake agents per map to caclulate G out variance

    metrics = dict()
    metrics_type_names = ['G/out_variability', 'G/loss_GAN', 'G/loss_reconstruct', 'G/loss_total',
                          'D/loss_classify_real', 'D/loss_classify_fake', 'D/loss_grad_penalty',
                          'D/loss_total', 'D/logit(fake)', 'D/logit(real)']

    for dataset_name, dataset in datasets.items():
        metrics_names = [f'{dataset_name}/{type_name}' for type_name in metrics_type_names]

        for metric_name in metrics_names:
            metrics[metric_name] = np.zeros(stats_n_maps)

        assert vis_n_generator_runs >= 1
        map_id = 0
        log_label = 'null'
        for scene_data in dataset:
            if map_id >= stats_n_maps:
                break
            real_agents_vecs, conditioning = pre_process_scene_data(scene_data, opt)

            if map_id < vis_n_maps:
                # Add an image of the map & real agents to wandb logs
                log_label = f"{dataset_name}_epoch#{1 + i_epoch} iter#{1 + epoch_iter} Map#{1 + map_id}"
                img, wandb_img = get_wandb_image(model, conditioning, real_agents_vecs, label='real_agents')
                visuals_dict[f'{dataset_name}_map_{map_id}_real_agents'] = img
                if opt.use_wandb:
                    wandb_logs[log_label] = [wandb_img]

            for i_generator_run in range(vis_n_generator_runs):
                fake_agents_vecs = model.netG(conditioning).detach()  # detach since we don't backpropp

                # calculate the metrics for only for the first generated agents set per map:
                if i_generator_run == 0:
                    # Feed real agents set to discriminator
                    d_out_for_real = model.netD(conditioning,
                                                real_agents_vecs).detach()  # detach since we don't backpropp
                    # pred_is_real_for_real_binary = (pred_is_real_for_real > 0).to(torch.float32)
                    d_out_for_fake = model.netD(conditioning,
                                                fake_agents_vecs).detach()  # detach since we don't backpropp
                    # pred_is_real_for_fake_binary = (pred_is_real_for_fake > 0).to(torch.float32)
                    loss_D_fake = model.criterionGAN(prediction=d_out_for_fake,
                                                     target_is_real=False)  # D wants to correctly classsify
                    loss_D_real = model.criterionGAN(prediction=d_out_for_real,
                                                     target_is_real=True)  # D wants to correctly classsify
                    loss_G_GAN = model.criterionGAN(prediction=d_out_for_fake,
                                                    target_is_real=True)  # G tries to make D wrongly classify the fake sample (make D output "True"
                    loss_G_reconstruct = model.criterion_reconstruct(fake_agents_vecs, real_agents_vecs)

                    loss_D_grad_penalty = cal_gradient_penalty(model.netD, conditioning, real_agents_vecs,
                                                               fake_agents_vecs, model)

                    metrics[f'{dataset_name}/G/loss_GAN'][map_id] = loss_G_GAN
                    metrics[f'{dataset_name}/G/loss_reconstruct'][map_id] = loss_G_reconstruct
                    metrics[f'{dataset_name}/G/loss_total'][
                        map_id] = loss_G_GAN + loss_G_reconstruct * opt.lambda_reconstruct
                    metrics[f'{dataset_name}/D/loss_classify_real'][map_id] = loss_D_real
                    metrics[f'{dataset_name}/D/loss_classify_fake'][map_id] = loss_D_fake
                    metrics[f'{dataset_name}/D/loss_grad_penalty'][map_id] = loss_D_grad_penalty
                    metrics[f'{dataset_name}/D/loss_total'][
                        map_id] = loss_D_fake + loss_D_real + model.lambda_gp * loss_D_grad_penalty
                    metrics[f'{dataset_name}/D/logit(fake)'][map_id] = d_out_for_fake
                    metrics[f'{dataset_name}/D/logit(real)'][map_id] = d_out_for_real

                # Add an image of the map & fake agents to wandb logs
                if map_id < vis_n_maps and i_generator_run < vis_n_generator_runs:
                    img, wandb_img = get_wandb_image(model, conditioning, fake_agents_vecs, label='real_agents')
                    visuals_dict[f'{dataset_name}_map_#{map_id + 1}_fake_#{i_generator_run + 1}'] = img
                    if opt.use_wandb:
                        wandb_logs[log_label].append(wandb_img)

            samples_fake_agents_vecs = []
            for i_generator_run in range(g_var_n_generator_runs):
                samples_fake_agents_vecs.append(model.netG(conditioning).detach())
            samples_fake_agents_vecs = torch.stack(samples_fake_agents_vecs)
            # calculate variance across samples:
            feat_var_across_samples = samples_fake_agents_vecs.var(dim=0)
            # Avg all out feat:
            metrics[f'{dataset_name}/G/out_variability'][map_id] = feat_var_across_samples.mean()
            map_id += 1

    # Average over the maps:
    for key, val in metrics.items():
        metrics[key] = val.mean()

    # additional metrics:
    metrics['run/LR'] = model.lr
    metrics['run/epoch'] = 1 + i_epoch
    metrics['run/total_iters'] = total_iters
    metrics['run/run_hours'] = (time.time() - run_start_time) / 60 ** 2

    if opt.use_wandb:
        wandb.log(metrics)
    print('Eval metrics: ' + ', '.join([f'{key}: {val:.2f}' for key, val in metrics.items()]))
    if opt.isTrain:
        model.train()
    return visuals_dict, wandb_logs


#########################################################################################

def get_wandb_image(model, conditioning, agents_vecs, label='real_agents'):
    agents_feat_dicts = agents_feat_vecs_to_dicts(agents_vecs)
    real_map = conditioning['map_feat']
    img = visualize_scene_feat(agents_feat_dicts, real_map)
    pred_is_real = torch.sigmoid(model.netD(conditioning, agents_vecs)).item()
    caption = f'{label}\npred_is_real={pred_is_real:.2}\n'
    caption += '\n'.join(get_agents_descriptions(agents_feat_dicts))
    wandb_img = wandb.Image(img, caption=caption)
    return img, wandb_img

#########################################################################################
##############################################################################################

def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256, use_wandb=False, file_type='png'):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []
    ims_dict = {}
    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = f'{name}_{label}.{file_type}'
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
        if use_wandb:
            ims_dict[label] = wandb.Image(im)
    webpage.add_images(ims, txts, links, width=width)
    if use_wandb:
        wandb.log(ims_dict)
##############################################################################################