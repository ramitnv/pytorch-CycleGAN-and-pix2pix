import ntpath
import os
import time

import torch
from util.util import append_to_field, num_to_str, to_num
from avsg_utils import agents_feat_vecs_to_dicts, pre_process_scene_data, get_agents_descriptions, \
    get_single_conditioning_from_batch
from util.avsg_visualization_utils import visualize_scene_feat
from . import util, html

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


##############################################################################################

class Visualizer:
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
        self.use_wandb = opt.use_wandb
        self.current_fig_index = 0
        self.plotted_inds = []
        self.records = dict()
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

    # ==========================================================================

    def print_current_metrics(self, model, i, opt, train_conditioning, val_data_gen, run_start_time):
        """  print training losses and save logging information to the log file and wandblog charts


        Parameters:
            i_epoch (int) -- current epoch
            i_batch (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)

        """
        model.eval()
        metrics = {'train': {'G': None, 'D': None}, 'val': {'G': None, 'D': None}}
        metrics['train']['G'] = model.train_log_metrics_G
        metrics['train']['D'] = model.train_log_metrics_D

        val_batch = next(val_data_gen)
        val_real_actors, val_conditioning = pre_process_scene_data(val_batch, opt)
        _, metrics['val']['G'] = model.get_G_losses(opt, val_real_actors, val_conditioning)
        _, metrics['val']['D'] = model.get_D_losses(opt, val_real_actors, val_conditioning)

        # add some more metrics
        # additional metrics:
        metrics['run'] = {'Iteration': i + 1, 'LR_G': model.lr_G, 'LR_D': model.lr_D,
                          'run_hours': (time.time() - run_start_time) / 60 ** 2}

        # sample several fake agents per map to calculate G out variance
        for conditioning, data_type in [(train_conditioning, 'train'), (val_conditioning, 'val')]:
            samples_fake_agents_vecs = []
            for i_generator_run in range(opt.G_variability_n_runs):
                samples_fake_agents_vecs.append(model.netG(conditioning).detach())
            samples_fake_agents_vecs = torch.stack(samples_fake_agents_vecs)
            # calculate variance across samples:
            feat_var_across_samples = samples_fake_agents_vecs.var(dim=0)
            # Average all output coordinates:
            metrics[data_type]['G']['G_out_variability'] = feat_var_across_samples.mean()

        # print to console
        message = '(' + ', '.join([f'{name}: {num_to_str(v)} ' for name, v in metrics['run'].items()]) + ')'
        for data_type in ['train', 'val']:
            for net_type in ['G', 'D']:
                message += f'\n{data_type}: ' + ''.join([f'{name}: {num_to_str(v)} '
                                                         for name, v in metrics[data_type][net_type].items()])
        print(message)

        # save to log file
        with open(self.log_name, "a") as log_file:
            log_file.write(f'{message}\n')

        # update wandb charts
        if self.use_wandb:
            for name, v in metrics['run'].items():
                self.wandb_run.log({f'run/{name}': v})
                append_to_field(self.records, f'run/{name}', to_num(v))
            for data_type in ['train', 'val']:
                for net_type in ['G', 'D']:
                    for name, v in metrics[data_type][net_type].items():
                        key_label = f'{data_type}/{net_type}/{name}'
                        self.wandb_run.log({key_label: v})
                        append_to_field(self.records, key_label, v)
            append_to_field(self.records, 'i', i)

            losses_seqs, losses_labels = self.get_loss_series(net_type='G', data_type='train')
            wandb.log({'D_Train_Losses_Scaled': wandb.plot.line_series(xs=self.records['i'],
                                                                       ys=losses_seqs,
                                                                       keys=losses_labels,
                                                                       title='D_Train_Losses_Scaled',
                                                                       xname='iter')})
        if opt.isTrain:
            model.train()

    # ==========================================================================

    def get_loss_series(self, net_type='G', data_type='train'):
        losses_labels = [k for k in self.records.keys() if k.startswith(f'{data_type}/{net_type}')]
        losses_seqs = [self.records[label] for label in losses_labels]
        return losses_seqs, losses_labels

    # ==========================================================================

    def display_current_results(self, model, i, opt, train_conditioning, train_real_actors, val_data_gen,
                                file_type='jpg'):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        fig_index = i
        self.plotted_inds.append(fig_index)
        visuals_dict, wandb_logs = get_images(model, i, opt, train_conditioning, train_real_actors, val_data_gen)

        # save images to an HTML file if they haven't been saved.
        if self.use_html:
            # save images to the disk
            for label, image in visuals_dict.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, f'i{i + 1}_{label}.{file_type}')
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=0)
            for ind in self.plotted_inds[::-1]:
                webpage.add_header(f'tot_iter {ind}')
                ims, txts, links = [], [], []
                for label, image_numpy in visuals_dict.items():
                    img_path = f'iter{ind}_{label}.{file_type}'
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()
        if self.use_wandb:
            if fig_index != self.current_fig_index:
                self.current_fig_index = fig_index
                for log_label, log_data in wandb_logs.items():
                    self.wandb_run.log({log_label: log_data})

        print(f'Figure saved for iteration #{i + 1}')

    # ==========================================================================


def get_images(model, i, opt, train_conditioning, train_real_actors, val_data_gen):
    """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""

    vis_n_maps = min(opt.vis_n_maps, opt.batch_size)  # how many maps to visualize
    vis_n_generator_runs = opt.vis_n_generator_runs  # how many sampled fake agents per map to visualize
    validation_batch = next(val_data_gen)
    val_real_actors, val_conditioning = pre_process_scene_data(validation_batch, opt)
    scenes_batches_dict = {'train': (train_real_actors, train_conditioning), 'val': (val_real_actors, val_conditioning)}
    wandb_logs = {}
    visuals_dict = {}
    model.eval()

    for dataset_name, scenes_batch in scenes_batches_dict.items():

        real_agents_vecs_batch, conditioning_batch = scenes_batch

        for i_map in range(vis_n_maps):
            # take data of current scene:
            real_agents_vecs = real_agents_vecs_batch[i_map]
            conditioning = get_single_conditioning_from_batch(conditioning_batch, i_map)

            # Add an image of the map & real agents to wandb logs
            log_label = f"{dataset_name}_iter_{i + 1}_map_{i_map + 1}"
            img, wandb_img = get_wandb_image(model, conditioning, real_agents_vecs, label='real')
            visuals_dict[f'{dataset_name}_iter_{i + 1}_map_{i_map + 1}_real_agents'] = img
            if opt.use_wandb:
                wandb_logs[log_label] = [wandb_img]

            for i_generator_run in range(vis_n_generator_runs):
                fake_agents_vecs = model.netG(conditioning).detach().squeeze()  # detach since we don't backpropp
                # Add an image of the map & fake agents to wandb logs
                img, wandb_img = get_wandb_image(model, conditioning, fake_agents_vecs,
                                                 label=f'fake_{1 + i_generator_run}')
                visuals_dict[f'{dataset_name}_iter_{i + 1}_map_{i_map + 1}_fake_{i_generator_run + 1}'] = img
                if opt.use_wandb:
                    wandb_logs[log_label].append(wandb_img)
    if opt.isTrain:
        model.train()
    return visuals_dict, wandb_logs


#########################################################################################

def get_wandb_image(model, conditioning, agents_vecs, label='real_agents'):
    if agents_vecs.ndim == 1:
        agents_vecs = agents_vecs.unsqueeze(0)
    assert agents_vecs.ndim == 2  # [n_agents x feat_dim]
    real_map = conditioning['map_feat']
    agents_feat_dicts = agents_feat_vecs_to_dicts(agents_vecs)
    img = visualize_scene_feat(agents_feat_dicts, real_map)
    pred_is_real = torch.sigmoid(model.netD(conditioning, agents_vecs)).item()
    caption = f'{label}\npred_is_real={pred_is_real:.2}\n'
    caption += '\n'.join(get_agents_descriptions(agents_feat_dicts))
    wandb_img = wandb.Image(img, caption=caption)
    return img, wandb_img


##############################################################################################

def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256, use_wandb=False, file_type='png'):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these images (see html.py for more details)
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
