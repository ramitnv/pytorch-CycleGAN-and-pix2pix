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
from models.networks import cal_gradient_penalty
from . import networks
from .base_model import BaseModel
from util.helper_func import get_net_weights_norm


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
        parser.add_argument('--data_path_val', type=str, default='', help='Path for validation dataset file')
        parser.add_argument('--augmentation_type', type=str, default='rotate_and_translate',
                            help=" 'none' | 'rotate_and_translate' | 'Gaussian_data' ")

        # ~~~~  General model settings
        if is_train:
            parser.set_defaults(gan_mode='wgangp',  # 'the type of GAN objective. [vanilla| lsgan | wgangp].
                                # vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
                                netD='SceneDiscriminator',
                                netG='SceneGenerator')
            parser.add_argument('--agents_decoder_model', type=str,
                                default='MLP')  # | 'MLP' | 'LSTM'
            # parser.add_argument('--num_samples_pack', type=int,
            #                     default=1)  # accumulate  m samples to classify with D (as in PacGAN)

        if is_train:
            # ~~~~  Training optimization settings
            parser.set_defaults(
                n_iter=1000,
                batch_size=64,
                lr=0.02,
                lr_policy='constant',  # [linear | step | plateau | cosine | constant]
                num_threads=0,  # threads for loading data, can increase to 4 for faster run if no mem issues
                save_latest_freq=1e6,  # frequency of saving the latest results
            )
            parser.add_argument('--reconstruct_loss_type', type=str, default='MSE', help=" 'L1' | 'MSE' ")
            parser.add_argument('--lambda_reconstruct', type=float, default=5e-4, help='weight for reconstruct_loss ')
            parser.add_argument('--lambda_gp', type=float, default=1., help='weight for gradient penalty in WGANGP')

            parser.add_argument('--type_weights_norm_D', type=str, default="Frobenius",
                                help=" None / Frobenius / L1 / Nuclear")
            parser.add_argument('--type_weights_norm_G', type=str, default="None",
                                help=" None / Frobenius / L1 / Nuclear")
            parser.add_argument('--lambda_weights_norm_D', type=float, default=1e-4, help=" ")
            parser.add_argument('--lambda_weights_norm_G', type=float, default=1e-4, help=" ")

        # ~~~~  Map features
        parser.add_argument('--polygon_name_order', type=list,
                            default=['lanes_mid', 'lanes_left', 'lanes_right', 'crosswalks'], help='')
        parser.add_argument('--closed_polygon_types', type=list,
                            default=['crosswalks'], help='')
        parser.add_argument('--max_points_per_poly', type=int, default=20,
                            help='Maximal number of points per polygon element')

        # ~~~~  Agents features
        parser.add_argument('--agent_feat_vec_coord_labels',
                            default=['centroid_x',  # [0]  Real number
                                     'centroid_y',  # [1]  Real number
                                     'yaw_cos',  # [2]  in range [-1,1],  sin(yaw)^2 + cos(yaw)^2 = 1
                                     'yaw_sin',  # [3]  in range [-1,1],  sin(yaw)^2 + cos(yaw)^2 = 1
                                     'extent_length',  # [4] Real positive
                                     'extent_width',  # [5] Real positive
                                     'speed',  # [6] Real non-negative
                                     'is_CAR',  # [7] 0 or 1
                                     'is_CYCLIST',  # [8] 0 or 1
                                     'is_PEDESTRIAN',  # [9]  0 or 1
                                     ],
                            type=list)
        parser.add_argument('--max_num_agents', type=int, default=4, help=' number of agents in a scene')

        if is_train:
            # ~~~~ general model settings
            parser.add_argument('--dim_agent_noise', type=int, default=16, help='Scene latent noise dimension')
            parser.add_argument('--dim_latent_map', type=int, default=32, help='Scene latent noise dimension')
            parser.add_argument('--n_point_net_layers', type=int, default=3, help='PointNet layers number')
            parser.add_argument('--use_layer_norm', type=int, default=1, help='0 or 1')
            parser.add_argument('--point_net_aggregate_func', type=str, default='sum', help='sum / max ')

            # ~~~~ map encoder settings
            parser.add_argument('--dim_latent_polygon_elem', type=int, default=8, help='')
            parser.add_argument('--dim_latent_polygon_type', type=int, default=16, help='')
            parser.add_argument('--kernel_size_conv_polygon', type=int, default=5, help='')
            parser.add_argument('--n_conv_layers_polygon', type=int, default=3, help='')
            parser.add_argument('--n_layers_poly_types_aggregator', type=int, default=3, help='')
            parser.add_argument('--n_layers_sets_aggregator', type=int, default=3, help='')
            parser.add_argument('--n_layers_scene_embedder_out', type=int, default=3, help='')

            # ~~~~ discriminator encoder settings
            parser.add_argument('--dim_discr_agents_enc', type=int, default=16, help='')
            parser.add_argument('--n_discr_out_mlp_layers', type=int, default=3, help='')
            parser.add_argument('--n_discr_pointnet_layers', type=int, default=3, help='')

            # ~~~~   Agents decoder options
            parser.add_argument('--agents_dec_in_layers', type=int, default=3, help='')
            parser.add_argument('--agents_dec_out_layers', type=int, default=3, help='')
            parser.add_argument('--agents_dec_n_stacked_rnns', type=int, default=3, help='')
            parser.add_argument('--agents_dec_dim_hid', type=int, default=512, help='')
            parser.add_argument('--agents_dec_use_bias', type=int, default=1)
            parser.add_argument('--agents_dec_mlp_n_layers', type=int, default=4)
            parser.add_argument('--gru_attn_layers', type=int, default=3, help='')

            # ~~~~ Display settings
            parser.set_defaults(
                print_freq=5,
                display_freq=10)
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
        self.use_wandb = opt.use_wandb
        self.polygon_name_order = opt.polygon_name_order
        self.agent_feat_vec_coord_labels = opt.agent_feat_vec_coord_labels
        self.dim_agent_feat_vec = len(self.agent_feat_vec_coord_labels)

        # specify the models you want to save to the disk.
        # The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks
        self.netG = networks.define_G(opt, self.gpu_ids)
        if self.isTrain:
            self.netD = networks.define_D(opt, self.gpu_ids)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            if opt.reconstruct_loss_type == 'L1':
                self.criterion_reconstruct = torch.nn.L1Loss()
            elif opt.reconstruct_loss_type == 'MSE':
                self.criterion_reconstruct = torch.nn.MSELoss()
            else:
                raise NotImplementedError
            self.gan_mode = opt.gan_mode
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            ##  Debug
            # print('calculating the statistics (mean & std) of the agents features...')
            # from avsg_utils import calc_agents_feats_stats
            # print(calc_agents_feats_stats(dataset, opt.agent_feat_vec_coord_labels, opt.device, opt.num_agents))
        #########################################################################################

    def get_D_losses(self, opt, real_actors, conditioning):
        """Calculate loss for the discriminator"""

        # we use conditional GANs; we need to feed both input and output to the discriminator
        fake_agents = self.netG(conditioning)

        # Feed fake agents to discriminator and calculate its prediction loss
        fake_agents_detached = fake_agents.detach()  # stop backprop to the generator by detaching
        d_out_for_fake = self.netD(conditioning, fake_agents_detached)

        # the loss is 0 if D correctly classify as fake
        loss_D_classify_fake = self.criterionGAN(prediction=d_out_for_fake, target_is_real=False)

        # Feed real (loaded from data) agents to discriminator and calculate its prediction loss
        d_out_for_real = self.netD(conditioning, real_actors)

        # the loss is 0 if D correctly classify as not fake
        loss_D_classify_real = self.criterionGAN(prediction=d_out_for_real, target_is_real=True)

        loss_D_grad_penalty = cal_gradient_penalty(self.netD, conditioning, real_actors,
                                                   fake_agents_detached, self)

        loss_D_weights_norm = get_net_weights_norm(self.netD, opt.type_weights_norm_D)

        # combine losses
        loss_D = loss_D_classify_fake + loss_D_classify_real
        reg_losses = [(opt.lambda_gp, loss_D_grad_penalty),
                      (opt.lambda_weights_norm_D, loss_D_weights_norm)]
        for (lamb, loss) in reg_losses:
            if loss is not None:
                loss_D += lamb * loss

        log_metrics = {"loss_D": loss_D,
                       "loss_D_classify_fake": loss_D_classify_fake,
                       "loss_D_classify_real": loss_D_classify_real,
                       "loss_D_grad_penalty": loss_D_grad_penalty,
                       "loss_D_weights_norm": loss_D_weights_norm,
                       "D_logit(real)": d_out_for_real, "D_logit(fake_detach)": d_out_for_fake}
        log_metrics = {name: val.mean().item() for name, val in log_metrics.items() if val is not None}
        return loss_D, log_metrics

    #########################################################################################

    def get_G_losses(self, opt, real_actors, conditioning):
        """Calculate loss terms for the generator"""

        fake_actors = self.netG(conditioning)

        d_out_for_fake = self.netD(conditioning, fake_actors)

        # G aims to fool D to wrongly classify the fake sample (make D output "True")
        loss_G_GAN = self.criterionGAN(prediction=d_out_for_fake, target_is_real=True)

        # Second, we want G(map) = map, since the generator acts also as an encoder-decoder for the map
        loss_G_reconstruct = self.criterion_reconstruct(fake_actors, real_actors)

        loss_G_weights_norm = get_net_weights_norm(self.netG, opt.type_weights_norm_G)

        # combine losses
        loss_G = loss_G_GAN
        reg_losses = [(opt.lambda_reconstruct, loss_G_reconstruct),
                      (opt.lambda_weights_norm_G, loss_G_weights_norm)]
        for (lamb, loss) in reg_losses:
            if loss is not None:
                loss_G += lamb * loss

        log_metrics = {"loss_G": loss_G, "loss_G_GAN": loss_G_GAN,
                       "loss_G_reconstruct": loss_G_reconstruct,
                       "D_logit(fake)": d_out_for_fake,
                       "loss_G_weights_norm": loss_G_weights_norm}
        log_metrics = {name: val.mean().item() for name, val in log_metrics.items() if val is not None}
        return loss_G, log_metrics

    #########################################################################################

    def optimize_discriminator(self, opt, real_actors, conditioning):
        """Update network weights; it will be called in every training iteration."""

        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.set_requires_grad(self.netG, False)  # disable backprop for G
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        loss_D, log_metrics_D = self.get_D_losses(opt, real_actors, conditioning)
        loss_D.backward()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # Save for logging:
        self.train_log_metrics_D = log_metrics_D

    #########################################################################################

    def optimize_generator(self, opt, real_actors, conditioning):
        """Update network weights; it will be called in every training iteration."""

        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netG, True)  # enable backprop for G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        loss_G, log_metrics_G = self.get_G_losses(opt, real_actors, conditioning)
        loss_G.backward()  # calculate gradients for G
        self.optimizer_G.step()  # update G's weights
        # Save for logging:
        self.train_log_metrics_G = log_metrics_G

    #########################################################################################
