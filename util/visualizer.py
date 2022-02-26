import sys
import os
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.avsg_utils import agents_feat_vecs_to_dicts, get_agents_descriptions, \
    get_single_conditioning_from_batch
from data.data_func import get_next_batch_cyclic
from util.avsg_visualization_utils import visualize_scene_feat
from util.util import append_to_field, num_to_str, to_num

import wandb

warnings.filterwarnings("ignore", "I found a path object that I don't think is part of a bar chart. Ignoring.")

is_windows = hasattr(sys, 'getwindowsversion')
##############################################################################################

class Visualizer:
    """This class includes several functions that can display/save images and print/save logging information.

     """

    # ==========================================================================

    def __init__(self, opt):
        """Initialize the Visualizer class

        """

        self.opt = opt  # cache the option
        self.name = opt.name
        self.wandb_online = opt.wandb_online
        self.records = {}
        exp_name = opt.name
        if self.wandb_online:
            # https://docs.wandb.ai/guides/track/advanced/environment-variables
            os.environ["WANDB_MODE"] = "run"
        else:
            os.environ["WANDB_MODE"] = "offline"

        self.wandb_run = wandb.init(project='SceneGen', name=exp_name, config=opt) if not wandb.run else wandb.run

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

        val_batch = get_next_batch_cyclic(val_data_gen)
        val_conditioning, val_real_actors = val_batch['conditioning'], val_batch['agents_feat_vecs']

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
            metrics[data_type]['G']['G_out_variability'] = feat_var_across_samples.mean().item()

        # print to console
        message = '(' + ', '.join([f'{name}: {num_to_str(v, perc=3)} ' for name, v in metrics['run'].items()]) + ')'
        for data_type in ['train', 'val']:
            for net_type in ['G', 'D']:
                message += f'\n{data_type}: ' + ''.join([f'{name}: {num_to_str(v, perc=3)} '
                                                         for name, v in metrics[data_type][net_type].items()])
        print(message)

        # save to log file
        with open(self.log_name, "a") as log_file:
            log_file.write(f'{message}\n')

        if opt.isTrain:
            model.train()

        # update wandb charts
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

        loss_terms_D = [('D_Loss_Total', 'train/D/loss_D', 1),
                        ('D_Loss_on_Fake', 'train/D/loss_D_classify_fake', 1),
                        ('D_Loss_on_Real', 'train/D/loss_D_classify_real', 1),
                        ('Lam*(Weights_Norm)', 'train/D/loss_D_weights_norm',
                         opt.lamb_loss_D_weights_norm,
                         ('Lam*(Grad_Penalty)', 'train/D/loss_D_grad_penalty',
                          opt.lamb_loss_D_grad_penalty))
                        ]
        self.plot_weighted_loss_summary(loss_terms_D, 'D_Train_Losses_Weighted')

        loss_terms_G = [('G_Loss_Total', 'train/G/loss_G', 1),
                        ('G_Loss_GAN', "train/G/loss_G_GAN", 1),
                        ('Lam*(G_Loss_Feat_Match', "train/G/loss_G_feat_match",
                         opt.lamb_loss_G_feat_match),
                        ('Lam*(Weights_Norm)', "train/G/loss_G_weights_norm",
                         opt.lamb_loss_G_weights_norm),
                        ]
        self.plot_weighted_loss_summary(loss_terms_G, 'G_Train_Losses_Weighted')

    # ==========================================================================

    def plot_weighted_loss_summary(self, loss_terms, log_name):
        iter_grid = np.array(self.records['i'])
        for loss_term in loss_terms:
            loss_name = loss_term[1]
            loss_label = loss_term[0]
            loss_lambda_weight = loss_term[2]
            if loss_name not in self.records.keys():
                continue
            loss_seq = np.array(self.records[loss_name]) * loss_lambda_weight
            plt.plot(iter_grid, loss_seq, label=loss_label)
        plt.legend()
        self.wandb_run.log({log_name: plt})

    # =========================================================== ===============

    def get_loss_series(self, net_type='G', data_type='train'):
        losses_labels = [k for k in self.records.keys() if k.startswith(f'{data_type}/{net_type}')]
        losses_seqs = [self.records[label] for label in losses_labels]
        return losses_seqs, losses_labels

    # ==========================================================================

    def display_current_results(self, model, i, opt, train_conditioning, train_real_actors, val_data_gen):
        """Display current results
b
         """
        wandb_logs = get_images(model, i, opt, train_conditioning, train_real_actors, val_data_gen)
        if wandb_logs:
            for log_label, log_data in wandb_logs.items():
                if is_windows:  # https://stackoverflow.com/q/70615413
                    log_label.replace('/', '_')
                self.wandb_run.log({log_label: log_data})

        ##############################################################################################



def get_images(model, i, opt, train_conditioning, train_real_actors, val_data_gen):
    """Return visualization images. train.py will display these images with visdom, and save the images  """

    vis_n_maps = min(opt.vis_n_maps, opt.batch_size)  # how many maps to visualize
    vis_n_generator_runs = opt.vis_n_generator_runs  # how many sampled fake agents per map to visualize
    validation_batch = get_next_batch_cyclic(val_data_gen)
    val_conditioning, val_real_actors = validation_batch['conditioning'], validation_batch['agents_feat_vecs']

    wandb_logs = {}
    if opt.num_last_images_to_save <= 0:
        return wandb_logs
    model.eval()
    for dataset_name, real_agents_vecs_batch, conditioning_batch \
            in [('train', train_real_actors, train_conditioning), ('val', val_real_actors, val_conditioning)]:
        for i_map in range(vis_n_maps):
            # take data of current scene:
            real_agents_vecs = real_agents_vecs_batch[i_map].unsqueeze(0)
            conditioning = get_single_conditioning_from_batch(conditioning_batch, i_map)
            # create an image of the map & real agents
            img, wandb_img = get_wandb_image(model, conditioning, real_agents_vecs, opt, label='real',
                                             title=f'map_{i_map + 1}_real')
            log_label = f"{dataset_name}_images/id_{i % opt.num_last_images_to_save}_map_{i_map + 1}"
            wandb_logs[log_label] = [wandb_img]
            for i_generator_run in range(vis_n_generator_runs):
                # create an image of the map & fake agents
                fake_agents_vecs = model.netG(conditioning).detach()  # detach since we don't backpropp
                # Add an image of the map & fake agents to wandb logs
                img, wandb_img = get_wandb_image(model, conditioning, fake_agents_vecs, opt,
                                                 label=f'fake_{1 + i_generator_run}',
                                                 title=f'map_{i_map + 1}_fake_{1 + i_generator_run}')
                wandb_logs[log_label].append(wandb_img)
    if opt.isTrain:
        model.train()
    return wandb_logs

##############################################################################################



#########################################################################################

def get_wandb_image(model, conditioning, agents_vecs, opt, label='real_agents', title=''):
    # change data to format used for the plot function:
    agents_exists = conditioning['agents_exists']
    agents_feat_dicts = agents_feat_vecs_to_dicts(agents_vecs, agents_exists, opt)
    real_map = {k: v[0].detach().cpu().numpy() for k, v in conditioning['map_feat'].items()}
    img = visualize_scene_feat(agents_feat_dicts, real_map, opt, title=title)
    pred_is_real = torch.sigmoid(model.netD(conditioning, agents_vecs)).item()
    caption = f'{label}\npred_is_real={pred_is_real:.2}\n'
    caption += '\n'.join(get_agents_descriptions(agents_feat_dicts))
    wandb_img = wandb.Image(img, caption=caption)
    return img, wandb_img


##############################################################################################
