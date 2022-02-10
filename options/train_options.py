from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def __init__(self):
        super().__init__()
        self.isTrain = None

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        #   visualization parameters
        parser.add_argument('--display_freq', type=int, default=200, help='frequency of showing training results on screen')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=5, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--n_iter', type=int, default=50000, help='number of total iterations')
        parser.add_argument('--n_steps_G', type=int, default=3, help='number of generator update steps per iteration')
        parser.add_argument('--n_steps_D', type=int, default=1, help='number of generator update steps per iteration')

        parser.add_argument('--optimizer_type', type=str, default='Adam', help='SGD / Adam')
        parser.add_argument('--lr_G', type=float, default=0.0002, help='initial learning rate for ADAM optimizer of G')
        parser.add_argument('--lr_D', type=float, default=0.00002, help='initial learning rate for ADAM optimizer of D')
        parser.add_argument('--sgd_momentum', type=float, default=0.9, help='momentum')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--gan_mode', type=str, default='wgangp', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--lr_policy', type=str, default='cosine', help='learning rate policy. [linear | step | plateau | cosine | constant]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations if lr_policy==step' )
        parser.add_argument('--lr_decay_factor', type=float, default=0.1, help='multiply by this every lr_decay_iters iterations if lr_policy==step')

        parser.add_argument('--reconstruct_loss_type', type=str, default='L1', help=" 'L1' | 'MSE' ")
        parser.add_argument('--lamb_loss_G_reconstruct', type=float, default=0, help='weight for reconstruct_loss ')
        parser.add_argument('--lamb_loss_D_grad_penalty', type=float, default=10., help='weight for gradient penalty in WGANGP')
        parser.add_argument('--type_weights_norm_D', type=str, default="Frobenius",
                            help=" None / Frobenius / L1 / Nuclear")
        parser.add_argument('--type_weights_norm_G', type=str, default="Frobenius",
                            help=" None / Frobenius / L1 / Nuclear")
        parser.add_argument('--lamb_loss_D_weights_norm', type=float, default=1e-4, help=" ")
        parser.add_argument('--lamb_loss_G_weights_norm', type=float, default=0, help=" ")

        self.isTrain = True
        return parser
