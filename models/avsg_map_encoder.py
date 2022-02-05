"""
Inspired by
https://github.com/lyft/l5kit/blob/master/l5kit/l5kit/planning/vectorized/open_loop_model.py

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.avsg_utils import get_poly_n_points_per_element
from util.util import make_tensor_1d
from models.sub_modules import MLP, PointNet


class PolygonEncoder(nn.Module):

    def __init__(self, dim_latent, n_conv_layers, kernel_size, device):
        super(PolygonEncoder, self).__init__()
        self.device = device
        self.dim_latent = dim_latent
        self.n_conv_layers = n_conv_layers
        self.kernel_size = kernel_size
        self.conv_layers = []
        for i_layer in range(self.n_conv_layers):
            if i_layer == 0:
                in_channels = 2  # in the input each point has 2 channels (x,y)
            else:
                in_channels = self.dim_latent
            self.conv_layers.append(nn.Conv1d(in_channels=in_channels,
                                              out_channels=self.dim_latent,
                                              kernel_size=self.kernel_size,
                                              padding='same',
                                              padding_mode='circular',
                                              device=self.device))
        self.layers = nn.ModuleList(self.conv_layers)
        self.out_layer = nn.Linear(self.dim_latent, self.dim_latent, device=self.device)

    def forward(self, poly_elems_points, poly_elems_exists):
        """Standard forward
        poly_elems_points  [batch_size  x n_elements x n_points x 2d  ]
        poly_elems_exists [batch_size  x n_elements]
        """
        batch_size = poly_elems_points.shape[0]
        n_elements_max = poly_elems_points.shape[1]
        n_points = poly_elems_points.shape[2]

        # fit to conv1d input dimensions [batch_size  x n_elements x in_channels=2  x n_points]
        h = torch.permute(poly_elems_points, (0, 1, 3, 2))

        # we combine the i_scene and i_element codinates, since all elements in all scenes go through same conv

        h = torch.reshape(h, (batch_size*n_elements_max, 2, n_points))  # [(batch_size*n_elements) x in_channels=2  x n_points]

        # We use several layers of  1d circular convolution followed by ReLu (equivariant layers)
        # and finally sum the output - this is all in all - a shift-invariant operator
        for i_layer in range(self.n_conv_layers):
            h = self.conv_layers[i_layer](h)
            h = F.relu(h)
        # reshape back:
        h = torch.reshape(h, (batch_size, n_elements_max, self.dim_latent, n_points)) # [batch_size, n_elements x out_channels  x n_points]
        h = h.sum(dim=-1) # [batch_size, n_elements x out_channels]
        h = self.out_layer(h)
        return h


#########################################################################################


class MapEncoder(nn.Module):

    def __init__(self, opt):
        super(MapEncoder, self).__init__()
        self.device = opt.device
        self.polygon_name_order = opt.polygon_name_order
        self.closed_polygon_types = opt.closed_polygon_types
        self.dim_latent_polygon_elem = opt.dim_latent_polygon_elem
        n_polygon_types = len(opt.polygon_name_order)
        self.dim_latent_polygon_type = opt.dim_latent_polygon_type
        self.dim_latent_map = opt.dim_latent_map
        self.poly_encoder = nn.ModuleDict()
        self.sets_aggregators = nn.ModuleDict()
        for poly_type in self.polygon_name_order:
            self.poly_encoder[poly_type] = PolygonEncoder(dim_latent=self.dim_latent_polygon_elem,
                                                          n_conv_layers=opt.n_conv_layers_polygon,
                                                          kernel_size=opt.kernel_size_conv_polygon,
                                                          device=self.device)
            self.sets_aggregators[poly_type] = PointNet(d_in=self.dim_latent_polygon_elem,
                                                        d_out=self.dim_latent_polygon_type,
                                                        d_hid=self.dim_latent_polygon_type,
                                                        n_layers=opt.n_layers_sets_aggregator,
                                                        opt=opt)
        self.poly_types_aggregator = MLP(d_in=self.dim_latent_polygon_type * n_polygon_types,
                                         d_out=self.dim_latent_map,
                                         d_hid=self.dim_latent_map,
                                         n_layers=opt.n_layers_poly_types_aggregator,
                                         opt=opt)

    def forward(self, map_feat):
        """Standard forward
        """
        map_elems_exists = map_feat['map_elems_exists']  # True for coordinates of valid poly elements
        map_elems_points = map_feat['map_elems_points']  # coordinates of the polygon elements
        latents_per_poly_type = []
        for i_poly_type, poly_type in enumerate(self.polygon_name_order):
            # Get the latent embedding of all elements of this type of polygons:
            poly_encoder = self.poly_encoder[poly_type]
            poly_elems_points = map_elems_points[:, i_poly_type, :, :]  # [batch_size x n_points x 2 dims]
            poly_elems_exists = map_elems_exists[:, i_poly_type, :]     # [batch_size]
            n_polys = poly_elems_exists.sum()
            if n_polys == 0:
                # if there are no polygon of this type in the scene:
                latent_poly_type = torch.zeros(self.dim_latent_polygon_type, device=self.device)
            else:
                poly_elems_latent = poly_encoder(poly_elems_points, poly_elems_exists)
                # Run PointNet to aggregate all polygon elements of this  polygon type
                latent_poly_type = self.sets_aggregators[poly_type](poly_elems_latent)
            latents_per_poly_type.append(latent_poly_type)
        poly_types_latents = torch.cat(latents_per_poly_type)
        map_latent = self.poly_types_aggregator(poly_types_latents)
        return map_latent
