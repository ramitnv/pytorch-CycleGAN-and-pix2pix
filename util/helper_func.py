import functools

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler


#########################################################################################

def run_validation(model, eval_dataset, n_batches=0):
    is_training = model.isTrain
    model.eval()
    n_loss_calc = 0
    loss_sum = 0
    for i_batch, data in enumerate(eval_dataset):
        if n_batches and n_batches <= i_batch:
            break
        model.set_input(data)  # unpack data from data loader
        model.forward()  # run inference
        loss = model.loss_criterion(model.prediction, model.ground_truth)
        loss_sum += loss
        n_loss_calc += 1
    loss_avg = loss_sum / n_loss_calc
    if is_training:
        model.train()
    return loss_avg


#########################################################################################

class Identity(nn.Module):
    def forward(self, x):
        return x


#########################################################################################

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


#########################################################################################

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'constant':
        scheduler = lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=0)
    elif opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.start_epoch - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=opt.lr_decay_factor)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_iter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


#########################################################################################

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


#########################################################################################

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=None):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if gpu_ids is None:
        gpu_ids = []
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


##########################################################################################

def sum_regularization_terms(reg_losses):
    weighted_terms = [lamb * loss for (lamb, loss) in reg_losses if loss is not None]
    if not weighted_terms:
        return 0.0
    else:
        return torch.stack(weighted_terms).sum()


#########################################################################################
def get_net_weights_norm(net, norm_type):
    if norm_type == 'None':
        return None
    tot_norm = 0
    for param in net.parameters():
        if norm_type == 'Frobenius':
            tot_norm += torch.norm(param, p='fro')
        elif norm_type == 'L1':
            tot_norm += torch.norm(param, p=1)
        elif norm_type == 'Nuclear':
            tot_norm += torch.norm(param, p='nuc')
    return tot_norm


########################################################################################

def set_spectral_norm_normalization(net):
    names = []  # for debug
    for name, module in net.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d,
                               nn.Conv2d, nn.ConvTranspose2d)):
            names.append(name)
            torch.nn.utils.parametrizations.spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12,
                                                          dim=None)
    return net
