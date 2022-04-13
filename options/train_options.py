from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def __init__(self):
        super().__init__()
        self.isTrain = None

    def initialize(self, parser):
        self.isTrain = True
        parser = BaseOptions.initialize(self, parser)

        # data parameters
        parser.add_argument('--data_size_limit', type=int, default=0, help='Limits dataset size, if positive num.')

        # Optimization  parameters
        parser.add_argument('--n_iter', type=int, default=50000, help='number of total iterations')
        parser.add_argument('--n_steps_G', type=int, default=1, help='number of generator update steps per iteration')
        parser.add_argument('--n_steps_D', type=int, default=1, help='number of generator update steps per iteration')
        parser.add_argument('--lr_G', type=float, default=0.002, help='initial learning rate for ADAM optimizer of G')
        parser.add_argument('--lr_D', type=float, default=0.002, help='initial learning rate for ADAM optimizer of D')
        parser.add_argument('--optimizer_type', type=str, default='Adam', help='SGD / Adam')
        parser.add_argument('--sgd_momentum', type=float, default=0.9, help='momentum')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr_policy', type=str, default='constant',
                            help='learning rate policy. [linear | step | plateau | cosine | constant]')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='multiply by a gamma every lr_decay_iters iterations if lr_policy==step')
        parser.add_argument('--lr_decay_factor', type=float, default=0.1,
                            help='multiply by this every lr_decay_iters iterations if lr_policy==step')

        # Objective function parameters
        parser.add_argument('--gan_mode', type=str, default='WGANGP',
                            help='the type of GAN objective. [vanilla| LSGAN | WGANGP]. '
                                 'vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--feat_match_loss_type', type=str, default='MSE', help=" 'L1' | 'MSE' ")
        parser.add_argument('--lamb_loss_G_feat_match', type=float, default=0, help='weight for feat_match_loss ')
        parser.add_argument('--lamb_loss_D_grad_penalty', type=float, default=5.,  help='weight for gradient penalty in WGANGP')
        parser.add_argument('--type_weights_norm_G', type=str, default='None',  help=" None / Frobenius / L1 / Nuclear")
        parser.add_argument('--lamb_loss_G_weights_norm', type=float, default=0, help=" ")
        parser.add_argument('--type_weights_norm_D', type=str, default='None', help=" 'None' / 'Frobenius' / 'L1' / 'Nuclear' ")
        parser.add_argument('--lamb_loss_D_weights_norm', type=float, default=0, help=" ")
        parser.add_argument('--target_real_label', type=float, default=0.9,
                            help="The label of real samples, use value>0 for label-smoothing")
        parser.add_argument('--target_fake_label', type=float, default=0.1,
                            help="The label of fake samples, use value<1 for label-smoothing")
        parser.add_argument('--latent_noise_trunc_stds', type=float, default=0.7)
        parser.add_argument('--added_noise_std_for_D_in', type=float, default=0.05)

        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        #   visualization parameters
        parser.add_argument('--display_freq', type=int, default=500,
                            help='frequency of generating visualization images (non-positive number = no images')
        parser.add_argument('--print_freq', type=int, default=5,
                            help='frequency of showing training results on console')

        return parser
