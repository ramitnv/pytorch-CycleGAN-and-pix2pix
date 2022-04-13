
import torch
import torch.nn.functional as F
from torch import nn as nn


class MLP(nn.Module):

    def __init__(self, d_in, d_out, d_hid, n_layers, opt, bias=True):
        super(MLP, self).__init__()
        assert n_layers >= 1
        self.device = opt.device
        self.use_layer_norm = opt.use_layer_norm
        self.d_in = d_in
        self.d_out = d_out
        self.d_hid = d_hid
        self.n_layers = n_layers
        layer_dims = [d_in] + (n_layers - 1) * [d_hid] + [d_out]
        modules_list = []
        for i_layer in range(n_layers - 1):
            layer_d_in = layer_dims[i_layer]
            layer_d_out = layer_dims[i_layer + 1]
            modules_list.append(nn.Linear(layer_d_in, layer_d_out, bias=bias, device=self.device))
            if self.use_layer_norm:
                modules_list.append(nn.LayerNorm(layer_d_out, device=self.device))
            modules_list.append(nn.LeakyReLU())
        modules_list.append(nn.Linear(layer_dims[-2], d_out, bias=bias, device=self.device))
        self.net = nn.Sequential(*modules_list)
        self.layer_dims = layer_dims

    def forward(self, in_vec):
        return self.net(in_vec)


#########################################################################################


class PointNet(nn.Module):

    def __init__(self, d_in, d_out, d_hid, n_layers, opt):
        super(PointNet, self).__init__()
        self.device = opt.device
        self.use_layer_norm = opt.use_layer_norm
        self.n_layers = n_layers
        self.d_in = d_in
        self.d_out = d_out
        self.d_hid = d_hid
        self.point_net_aggregate_func = opt.point_net_aggregate_func
        self.layer_dims = [d_in] + (n_layers - 1) * [d_hid] + [d_out]
        self.linearA = nn.ModuleList()
        self.linearB = nn.ModuleList()
        for i_layer in range(n_layers - 1):
            # each layer the function that operates on each element in the set x is
            # f(x) = ReLu(A x + B * (sum over all non x elements) )
            layer_dims = ()
            self.linearA.append(nn.Linear(in_features=self.layer_dims[i_layer],
                                          out_features=self.layer_dims[i_layer + 1],
                                          device=self.device))
            self.linearB.append(nn.Linear(in_features=self.layer_dims[i_layer],
                                          out_features=self.layer_dims[i_layer + 1],
                                          device=self.device))
        self.out_layer = nn.Linear(d_hid, d_out, device=self.device)
        if self.use_layer_norm:
            self.layer_normalizer = nn.LayerNorm(d_hid, device=self.device)

    def forward(self, in_set):
        """'
             each layer the function that operates on each element in the set x is
            f(elem y) = ReLu(A y + B * (sum over all non y elements) )
            where A and B are the same for all elements, and are layer dependent.
            After that the elements are aggregated by max-pool
             and finally  a linear layer gives the output

            input is a tensor of size [batch_size x num_set_elements x elem_dim]

        """

        h = in_set # [batch_size x n_elements x feat_dim]
        batch_size = h.shape[0]
        n_elements = h.shape[1]
        feat_dim = h.shape[2]
        for i_layer in range(self.n_layers - 1):
            linearA = self.linearA[i_layer]
            linearB = self.linearB[i_layer]
            # find for each element coord now yje the sum over all elements in its set
            h_sum = h.sum(dim=-2)
            h_sum = h_sum.repeat(n_elements, 1, 1)
            h_sum = torch.permute(h_sum, (1, 0, 2))
            sum_without_elem = h_sum - h
            h = torch.reshape(h, (batch_size*n_elements, -1)) # we run the same linear transform on each elem in each sample
            sum_without_elem = torch.reshape(sum_without_elem, (batch_size*n_elements, -1))
            h_new = linearA(h) + linearB(sum_without_elem)
            h = torch.reshape(h_new, (batch_size, n_elements, -1))
            if self.use_layer_norm:
                h = self.layer_normalizer(h)
            h = F.leaky_relu(h)
        # apply permutation invariant aggregation over all elements
        if self.point_net_aggregate_func == 'max':
            h = h.max(dim=-2)
        elif self.point_net_aggregate_func == 'sum':
            h = h.sum(dim=-2)
        else:
            raise NotImplementedError
        h = self.out_layer(h)
        return h


###############################################################################

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'LSGAN':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['WGANGP']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - typically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and ground truth labels.

        Parameters:
            prediction (tensor) - - typically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['LSGAN', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'WGANGP':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        else:
            raise ValueError('Invalid gan_mode')
        return loss