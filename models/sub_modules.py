import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            modules_list.append(nn.ReLU())
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
            f(x) = ReLu(A x + B * (sum over all non x elements) )
            where A and B are the same for all elements, and are layer dependent.
            After that the elements are aggregated by max-pool
             and finally  a linear layer gives the output

            input is a tensor of size [batch_size x num_set_elements x elem_dim]

        """

        h = in_set
        n_elements = in_set.shape[-2]
        for i_layer in range(self.n_layers - 1):
            linearA = self.linearA[i_layer]
            linearB = self.linearB[i_layer]
            pre_layer_sum = h.sum(dim=-2)
            h_new_lst = []
            for i_elem in range(n_elements):
                h_elem = h[i_elem]
                sum_without_elem = pre_layer_sum - h_elem
                h_new = linearA(h_elem) + linearB(sum_without_elem)
                h_new_lst.append(h_new)
            h = torch.stack(h_new_lst)
            if self.use_layer_norm:
                h = self.layer_normalizer(h)
            h = F.relu(h)
        # apply permutation invariant aggregation over all elements

        if self.point_net_aggregate_func == 'max':
            h = h.max(dim=-2)
        elif self.point_net_aggregate_func == 'sum':
            h = h.sum(dim=-2)
        else:
            raise NotImplementedError
        h = self.out_layer(h)
        return h
