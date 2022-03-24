import torch
import torch.nn as nn

from models.avsg_func import ProjectionToAgentFeat
from models.sub_modules import MLP


#########################################################################

def get_agents_decoder(opt, device):
    if opt.agents_decoder_model == 'Sequential':
        return AgentsDecoderSequential(opt, device)
    if opt.agents_decoder_model == 'MLP':
        return AgentsDecoderMLP(opt, device)
    else:
        raise NotImplementedError


#########################################################################################

class AgentsDecoderMLP(nn.Module):
    def __init__(self, opt, device):
        super(AgentsDecoderMLP, self).__init__()
        self.device = device
        self.agents_dec_dim_hid = opt.agents_dec_dim_hid
        self.max_num_agents = opt.max_num_agents
        self.agent_feat_vec_coord_labels = opt.agent_feat_vec_coord_labels
        self.dim_agent_feat_vec = len(opt.agent_feat_vec_coord_labels)
        self.dim_latent_map = opt.dim_latent_map
        self.dim_agent_noise = opt.dim_agent_noise
        self.d_in = self.dim_agent_noise * self.max_num_agents + opt.dim_latent_map
        self.d_out = self.dim_agent_feat_vec * self.max_num_agents
        self.project_to_agent_feat = ProjectionToAgentFeat(opt, device)
        self.decoder = MLP(d_in=self.d_in,
                           d_out=self.d_out,
                           d_hid=self.agents_dec_dim_hid,
                           n_layers=opt.agents_dec_mlp_n_layers,
                           opt=opt,
                           bias=opt.agents_dec_use_bias)

    def forward(self, map_latent, latent_noise, n_agents_per_scene, agents_exists):
        batch_size = latent_noise.shape[0]
        latent_noise = torch.reshape(latent_noise, (batch_size, self.max_num_agents * self.dim_agent_noise))
        in_vec = torch.cat([map_latent, latent_noise], dim=1)
        out_vec = self.decoder(in_vec)
        out_vec = torch.reshape(out_vec, (batch_size, self.max_num_agents, self.dim_agent_feat_vec))
        # Apply projection of each output vector to the feature vectors domain:
        agents_feat_vecs = self.project_to_agent_feat(out_vec, n_agents_per_scene, agents_exists)
        return agents_feat_vecs

#########################################################################################

class AgentsDecoderSequential(nn.Module):
    def __init__(self, opt, device):
        super(AgentsDecoderSequential, self).__init__()
        self.device = device
        self.agents_dec_dim_hid = opt.agents_dec_dim_hid
        self.max_num_agents = opt.max_num_agents
        self.agent_feat_vec_coord_labels = opt.agent_feat_vec_coord_labels
        self.dim_agent_feat_vec = len(opt.agent_feat_vec_coord_labels)
        self.dim_latent_map = opt.dim_latent_map
        self.dim_agent_noise = opt.dim_agent_noise
        self.d_in = self.dim_agent_noise * self.max_num_agents + opt.dim_latent_map
        self.d_out = self.dim_agent_feat_vec * self.max_num_agents
        self.project_to_agent_feat = ProjectionToAgentFeat(opt, device)
        self.decoder = MLP(d_in=self.d_in,
                           d_out=self.d_out,
                           d_hid=self.agents_dec_dim_hid,
                           n_layers=opt.agents_dec_mlp_n_layers,
                           opt=opt,
                           bias=opt.agents_dec_use_bias)

    def forward(self, map_latent, latent_noise, n_agents_per_scene, agents_exists):
        batch_size = latent_noise.shape[0]
        latent_noise = torch.reshape(latent_noise, (batch_size, self.max_num_agents * self.dim_agent_noise))
        in_vec = torch.cat([map_latent, latent_noise], dim=1)
        out_vec = self.decoder(in_vec)
        out_vec = torch.reshape(out_vec, (batch_size, self.max_num_agents, self.dim_agent_feat_vec))
        # Apply projection of each output vector to the feature vectors domain:
        agents_feat_vecs = self.project_to_agent_feat(out_vec, n_agents_per_scene, agents_exists)
        return agents_feat_vecs

#########################################################################################