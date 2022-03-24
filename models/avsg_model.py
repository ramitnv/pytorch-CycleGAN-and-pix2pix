"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""

import torch

from models.avsg_discriminator import cal_gradient_penalty
from models.avsg_generator import define_G
from .avsg_func import get_out_of_road_penalty
from util.helper_func import get_net_weights_norm, sum_regularization_terms
from .avsg_discriminator import define_D
from .base_model import BaseModel
from .sub_modules import GANLoss


#########################################################################################


class AvsgModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # ~~~~  Data
        if is_train:
            parser.add_argument('--augmentation_type', type=str, default='rotate_and_translate',
                                help=" 'none' | 'rotate_and_translate")

            parser.add_argument('--shuffle_agents_inds_flag', type=int, default=1,  help="")

            # ~~~~ general model settings
            parser.add_argument('--dim_agent_noise', type=int, default=16, help='Scene latent noise dimension')
            parser.add_argument('--dim_latent_map', type=int, default=32, help='Scene latent noise dimension')
            parser.add_argument('--n_point_net_layers', type=int, default=4, help='PointNet layers number')
            parser.add_argument('--use_layer_norm', type=int, default=0, help='0 or 1')
            parser.add_argument('--use_spectral_norm_D', type=int, default=1, help='0 or 1')
            parser.add_argument('--point_net_aggregate_func', type=str, default='sum', help='sum / max ')

            parser.add_argument('--lamb_loss_G_out_of_road', type=float, default=0.001, help=" ")


            # ~~~~ map encoder settings
            parser.add_argument('--dim_latent_polygon_elem', type=int, default=8, help='')
            parser.add_argument('--dim_latent_polygon_type', type=int, default=16, help='')
            parser.add_argument('--kernel_size_conv_polygon', type=int, default=5, help='')
            parser.add_argument('--n_conv_layers_polygon', type=int, default=4, help='')
            parser.add_argument('--n_layers_poly_types_aggregator', type=int, default=4, help='')
            parser.add_argument('--n_layers_sets_aggregator', type=int, default=4, help='')
            parser.add_argument('--n_layers_scene_embedder_out', type=int, default=4, help='')

            # ~~~~ discriminator encoder settings
            parser.add_argument('--dim_discr_agents_enc', type=int, default=16, help='')
            parser.add_argument('--n_discr_out_mlp_layers', type=int, default=3, help='')
            parser.add_argument('--n_discr_pointnet_layers', type=int, default=3, help='')

            # ~~~~   Agents decoder options
            parser.add_argument('--agents_decoder_model', type=str, default='Sequential',
                                help="| 'MLP' | 'Sequential' ")
            parser.add_argument('--agents_dec_in_layers', type=int, default=4, help='')
            parser.add_argument('--agents_dec_out_layers', type=int, default=4, help='')
            parser.add_argument('--agents_dec_n_stacked_rnns', type=int, default=3, help='')
            parser.add_argument('--agents_dec_dim_hid', type=int, default=512, help='')
            parser.add_argument('--agents_dec_use_bias', type=int, default=1)
            parser.add_argument('--agents_dec_mlp_n_layers', type=int, default=4)
            parser.add_argument('--gru_attn_layers', type=int, default=3, help='')


            # ~~~~ Display settings
            parser.add_argument('--vis_n_maps', type=int, default=2, help='')
            parser.add_argument('--vis_n_generator_runs', type=int, default=3, help='')
            parser.add_argument('--G_variability_n_runs', type=int, default=5, help='')
        return parser

    #########################################################################################

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options
///
        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        opt.device = self.device
        self.wandb_online = opt.wandb_online

        # specify the models you want to save to the disk.
        # The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks
        self.netG = define_G(opt, self.gpu_ids)
        if self.isTrain:
            self.netD = define_D(opt, self.gpu_ids)
            # define loss functions
            self.criterionGAN = GANLoss(opt.gan_mode,
                                        target_real_label=opt.target_real_label,
                                        target_fake_label=opt.target_fake_label).to(self.device)
            if opt.feat_match_loss_type == 'L1':
                self.criterion_feat_match = torch.nn.L1Loss()
            elif opt.feat_match_loss_type == 'MSE':
                self.criterion_feat_match = torch.nn.MSELoss()
            else:
                raise NotImplementedError

            self.gan_mode = opt.gan_mode
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            if opt.optimizer_type == 'Adam':
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr_G, betas=(opt.beta1, 0.999))
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_D, betas=(opt.beta1, 0.999))
            elif opt.optimizer_type == 'SGD':
                self.optimizer_G = torch.optim.SGD(self.netG.parameters(), lr=opt.lr_G, momentum=opt.sgd_momentum)
                self.optimizer_D = torch.optim.SGD(self.netD.parameters(), lr=opt.lr_D, momentum=opt.sgd_momentum)
            else:
                raise NotImplementedError
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            ##  Debug
            # print('calculating the statistics (mean & std) of the agents features...')
            # from avsg_utils import calc_agents_feats_stats
            # print(calc_agents_feats_stats(dataset, opt.agent_feat_vec_coord_labels, opt.device, opt.num_agents))
        #########################################################################################

    def get_D_losses(self, opt, real_agents, conditioning):
        """Calculate loss for the discriminator"""

        # we use conditional GANs; we need to feed both input and output to the discriminator
        fake_agents = self.netG(conditioning)

        # Feed fake agents to discriminator and calculate its prediction loss
        fake_agents_detached = fake_agents.detach()  # stop backprop to the generator by detaching

        # (optional) add noise to D's inputs, as a stabilization technique
        if opt.added_noise_std_for_D_in > 0:
            input_noise = torch.empty_like(real_agents)
            torch.nn.init.trunc_normal_(input_noise, mean=0., std=opt.added_noise_std_for_D_in, a=-1., b=1.)
            fake_agents_detached += input_noise
            torch.nn.init.trunc_normal_(input_noise, mean=0., std=opt.added_noise_std_for_D_in, a=-1., b=1.)
            real_agents += input_noise

        d_out_for_fake = self.netD(conditioning, fake_agents_detached)

        # the loss is 0 if D correctly classify as fake
        loss_D_classify_fake = self.criterionGAN(prediction=d_out_for_fake, target_is_real=False)

        # Feed real (loaded from data) agents to discriminator and calculate its prediction loss
        d_out_for_real = self.netD(conditioning, real_agents)

        # the loss is 0 if D correctly classify as not fake
        loss_D_classify_real = self.criterionGAN(prediction=d_out_for_real, target_is_real=True)

        loss_D_grad_penalty = cal_gradient_penalty(self.netD, conditioning, real_agents,
                                                   fake_agents_detached, self)

        loss_D_weights_norm = get_net_weights_norm(self.netD, opt.type_weights_norm_D)

        # combine losses

        reg_total = sum_regularization_terms([
            (opt.lamb_loss_D_grad_penalty, loss_D_grad_penalty),
            (opt.lamb_loss_D_weights_norm, loss_D_weights_norm)])

        loss_D = loss_D_classify_fake + loss_D_classify_real + reg_total

        log_metrics = {"loss_D": loss_D,
                       "loss_D_classify_fake": loss_D_classify_fake,
                       "loss_D_classify_real": loss_D_classify_real,
                       "loss_D_grad_penalty": loss_D_grad_penalty,
                       "loss_D_weights_norm": loss_D_weights_norm,
                       "d_real": d_out_for_real,
                       "d_fake": d_out_for_fake}
        log_metrics = {name: val.mean().item() for name, val in log_metrics.items() if val is not None}
        return loss_D, log_metrics

    #########################################################################################

    def get_G_losses(self, opt, real_agents, conditioning):
        """Calculate loss terms for the generator"""

        fake_agents = self.netG(conditioning)

        d_out_for_fake = self.netD(conditioning, fake_agents)

        # G aims to fool D to wrongly classify the fake sample (make D output "True")
        loss_G_GAN = self.criterionGAN(prediction=d_out_for_fake, target_is_real=True)

        if opt.lamb_loss_G_feat_match > 0:
            loss_G_feat_match = self.criterion_feat_match(fake_agents, real_agents)
        else:
            loss_G_feat_match = None

        loss_G_weights_norm = get_net_weights_norm(self.netG, opt.type_weights_norm_G)

        loss_G_out_of_road = get_out_of_road_penalty(conditioning, fake_agents, opt)

        # combine losses
        reg_total = sum_regularization_terms(
            [(opt.lamb_loss_G_feat_match, loss_G_feat_match),
             (opt.lamb_loss_G_weights_norm, loss_G_weights_norm),
             (opt.lamb_loss_G_out_of_road, loss_G_out_of_road)])

        loss_G = loss_G_GAN + reg_total

        log_metrics = {"loss_G": loss_G,
                       "loss_G_GAN": loss_G_GAN,
                       "loss_G_feat_match": loss_G_feat_match,
                       "d_fake": d_out_for_fake,
                       "loss_G_weights_norm": loss_G_weights_norm,
                       "loss_G_out_of_road": loss_G_out_of_road}
        log_metrics = {name: val.mean().item() for name, val in log_metrics.items() if val is not None}
        return loss_G, log_metrics

    #########################################################################################

    def optimize_discriminator(self, opt, real_agents, conditioning):
        """Update network weights; it will be called in every training iteration."""

        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.set_requires_grad(self.netG, False)  # disable backprop for G
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        loss_D, log_metrics_D = self.get_D_losses(opt, real_agents, conditioning)
        loss_D.backward()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # Save for logging:
        self.train_log_metrics_D = log_metrics_D

    #########################################################################################

    def optimize_generator(self, opt, real_agents, conditioning):
        """Update network weights; it will be called in every training iteration."""

        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netG, True)  # enable backprop for G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        loss_G, log_metrics_G = self.get_G_losses(opt, real_agents, conditioning)
        loss_G.backward()  # calculate gradients for G
        self.optimizer_G.step()  # update G's weights
        # Save for logging:
        self.train_log_metrics_G = log_metrics_G

    #########################################################################################
