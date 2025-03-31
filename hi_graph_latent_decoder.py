import math
import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from einops import rearrange, repeat
import constants
import copy
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv import TransformerConv
import numpy as np
from inspect import isfunction
import matplotlib.pyplot as plt
import os
from interaction_net import PropagationNet
import utils

###########################################################
# Graph Encoder/Decoder with Transformers and MLPs
###########################################################

class LongformerNHopEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        dim_feedforward=2048,
        max_hop=4,
        num_layers=1,
        max_position=1024
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.max_hop = max_hop
        self.num_layers = num_layers
        self.max_position = max_position

        self.local_attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

        self.feedforwards = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Linear(dim_feedforward, d_model)
            )
            for _ in range(num_layers)
        ])

        self.global_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.global_feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )

        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers * 2 + 2)
        ])

        self.positional_encoding = PositionalEncoding(d_model)

        self.hop_biases = nn.ParameterList([
            nn.Parameter(torch.empty(max_position, max_position))
            for _ in range(max_hop)
        ])

        self.reset_parameters()
        
    def reset_parameters(self):
        for i, bias_param in enumerate(self.hop_biases):
            hop_idx = i + 1
            scale = float(self.max_hop - i) / float(self.max_hop)
            with torch.no_grad():
                bias_param.fill_(scale)
        
    def create_n_hop_mask(self, edge_index, num_nodes, hop):
        adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=edge_index.device)
        adjacency[edge_index[0], edge_index[1]] = 1.0

        current_adjacency = adjacency.clone()
        for _ in range(hop - 1):
            adjacency = adjacency @ current_adjacency
            adjacency = adjacency.clamp(max=1.0)

        return adjacency

    def forward(self, x, edge_index):
        batch_size, seq_len, _ = x.size()

        x = self.positional_encoding(x)

        mask_sum = torch.zeros((seq_len, seq_len), device=x.device, dtype=torch.float32)
        for hop_i in range(1, self.max_hop + 1):
            hop_idx = hop_i - 1
            adjacency = self.create_n_hop_mask(edge_index, seq_len, hop_i)
            bias_matrix = self.hop_biases[hop_idx][:seq_len, :seq_len]
            mask_sum += adjacency * bias_matrix
            
        norm_index = 0
        for layer_idx in range(self.num_layers):
            attn_module = self.local_attentions[layer_idx]
            local_out, _ = attn_module(x, x, x, attn_mask=mask_sum)
            x = x + local_out
            x = self.norms[norm_index](x)
            norm_index += 1

            ff_out = self.feedforwards[layer_idx](x)
            x = x + ff_out
            x = self.norms[norm_index](x)
            norm_index += 1

        global_attn_out, _ = self.global_attention(x, x, x)
        x = x + global_attn_out
        x = self.norms[norm_index](x)
        norm_index += 1

        global_ff_out = self.global_feedforward(x)
        x = x + global_ff_out
        x = self.norms[norm_index](x)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)

class HiGraphLatentEncoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        g2m_edge_index,
        m2m_edge_index,
        mesh_up_edge_index,
        hidden_dim,
        intra_level_layers,
        n_hop=4,
        hidden_layers=1,
        output_dist='diagonal',
    ):
        super().__init__()

        self.g2m_gnn = PropagationNet(
            g2m_edge_index,
            hidden_dim,
            hidden_layers=hidden_layers,
            update_edges=False,
        )

        self.mesh_up_gnns = nn.ModuleList(
            [
                PropagationNet(
                    edge_index,
                    hidden_dim,
                    hidden_layers=hidden_layers,
                    update_edges=False,
                )
                for edge_index in mesh_up_edge_index
            ]
        )

        self.intra_up_transformers = nn.ModuleList([
            LongformerNHopEncoderLayer(
                d_model=hidden_dim,
                num_heads=8,
                dim_feedforward=4 * hidden_dim,
                max_hop=n_hop,
                num_layers=intra_level_layers,
                max_position=642
            )] + [
            LongformerNHopEncoderLayer(
                d_model=hidden_dim,
                num_heads=8,
                dim_feedforward=4 * hidden_dim,
                max_hop=n_hop,
                num_layers=intra_level_layers,
                max_position=162
            ) 
        ])

        if output_dist == 'diagonal':
            self.output_dim = latent_dim * 2
        else:
            self.output_dim = latent_dim
        self.latent_param_map = utils.make_mlp(
            [hidden_dim] * (hidden_layers + 1) + [self.output_dim],
            layer_norm=False,
        )
        self.output_dist = output_dist
        self.n_hop = n_hop
        self.m2m_edge_index = m2m_edge_index

    def forward(self, grid_current_emb, graph_emb):
        current_mesh_rep = self.g2m_gnn(
            grid_current_emb, graph_emb["mesh"][0], graph_emb["g2m"]
        )

        current_mesh_rep = self.intra_up_transformers[0](
            current_mesh_rep, self.m2m_edge_index[0]
        )

        for idx, (up_gnn, intra_transformer, mesh_up_level_rep, m2m_level_rep, mesh_level_rep) in enumerate(zip(
            self.mesh_up_gnns,
            self.intra_up_transformers[1:],
            graph_emb["mesh_up"],
            graph_emb["m2m"][1:],
            graph_emb["mesh"][1:],
        )):
            new_node_rep = up_gnn(current_mesh_rep, mesh_level_rep, mesh_up_level_rep)
            
            current_mesh_rep = intra_transformer(
                new_node_rep, self.m2m_edge_index[idx + 1]
            )

        latent_params = self.latent_param_map(current_mesh_rep)

        if self.output_dist == 'diagonal':
            mean, logvar = torch.chunk(latent_params, 2, dim=-1)
        else:
            mean = latent_params
            logvar = None

        return mean, logvar

class HiGraphLatentDecoder(nn.Module):
    def __init__(self,
                 g2m_edge_index,
                 m2m_edge_index,
                 m2g_edge_index,
                 mesh_up_edge_index,
                 mesh_down_edge_index,
                 hidden_dim,
                 hidden_layers=1,
                 intra_level_layers=1,
                 n_hop=4,
                 output_std=False):
        super().__init__()

        latent_dim = hidden_dim

        self.grid_update_mlp = utils.make_mlp(
            [hidden_dim] * (hidden_layers + 2)
        )

        self.latent_embedder = utils.make_mlp(
            [latent_dim] + [hidden_dim] * (hidden_layers + 1)
        )
        
        self.g2m_gnn = PropagationNet(
            g2m_edge_index,
            hidden_dim,
            hidden_layers=hidden_layers,
            update_edges=False,
        )

        self.mesh_up_gnns = nn.ModuleList(
            [
                PropagationNet(
                    edge_index,
                    hidden_dim,
                    hidden_layers=hidden_layers,
                    update_edges=False,
                )
                for edge_index in mesh_up_edge_index
            ]
        )

        self.intra_up_transformers = nn.ModuleList([
            LongformerNHopEncoderLayer(
                d_model=hidden_dim,
                num_heads=8,
                dim_feedforward=4 * hidden_dim,
                max_hop=n_hop,
                num_layers=intra_level_layers,
                max_position=642
            )] + [
            LongformerNHopEncoderLayer(
                d_model=hidden_dim,
                num_heads=8,
                dim_feedforward=4 * hidden_dim,
                max_hop=n_hop,
                num_layers=intra_level_layers,
                max_position=162
            ) 
        ])

        self.mesh_down_gnns = nn.ModuleList(
            [
                PropagationNet(
                    edge_index,
                    hidden_dim,
                    hidden_layers=hidden_layers,
                    update_edges=False,
                )
                for edge_index in mesh_down_edge_index
            ]
        )

        self.intra_down_transformers = nn.ModuleList([
            LongformerNHopEncoderLayer(
                d_model=hidden_dim,
                num_heads=8,
                dim_feedforward=4 * hidden_dim,
                max_hop=n_hop,
                num_layers=intra_level_layers,
                max_position=642
            )
            for _ in list(m2m_edge_index)[:-1]
        ])

        self.m2g_gnn = PropagationNet(
            m2g_edge_index,
            hidden_dim,
            hidden_layers=hidden_layers,
            update_edges=False,
        )

        self.output_std = output_std
        if self.output_std:
            output_dim = 2 * constants.GRID_STATE_DIM
        else:
            output_dim = constants.GRID_STATE_DIM

        self.param_map = utils.make_mlp(
            [hidden_dim] * (hidden_layers + 1) + [output_dim], layer_norm=False
        )

        self.n_hop = n_hop
        self.m2m_edge_index = m2m_edge_index

    def forward(self, grid_rep, latent_sample, last_state, graph_emb):
        latent_rep = self.latent_embedder(latent_sample)
        
        current_mesh_rep = self.g2m_gnn(
            grid_rep, graph_emb["mesh"][0], graph_emb["g2m"]
        )

        mesh_level_reps = []
        m2m_level_reps = []

        for idx, (up_gnn, intra_transformer, mesh_up_level_rep, m2m_level_rep, mesh_level_rep) in enumerate(zip(
            self.mesh_up_gnns,
            self.intra_up_transformers[:-1],
            graph_emb["mesh_up"],
            graph_emb["m2m"][:-1],
            graph_emb["mesh"][1:-1] + [latent_rep],
        )):
            new_mesh_rep = intra_transformer(
                current_mesh_rep, self.m2m_edge_index[idx]
            )

            mesh_level_reps.append(new_mesh_rep)
            m2m_level_reps.append(m2m_level_rep)

            current_mesh_rep = up_gnn(new_mesh_rep, mesh_level_rep, mesh_up_level_rep)

        prev_mesh_rep = current_mesh_rep
        current_mesh_rep = self.intra_up_transformers[-1](
            current_mesh_rep, list(self.m2m_edge_index)[-1]
        )

        for idx, (down_gnn, intra_transformer, mesh_down_level_rep, m2m_level_rep, mesh_level_rep) in enumerate(zip(
            reversed(self.mesh_down_gnns),
            reversed(self.intra_down_transformers),
            reversed(graph_emb["mesh_down"]),
            reversed(m2m_level_reps),
            reversed(mesh_level_reps),
        )):
            new_mesh_rep = down_gnn(current_mesh_rep, mesh_level_rep, mesh_down_level_rep)

            current_mesh_rep = intra_transformer(
                new_mesh_rep, list(self.m2m_edge_index)[-(idx + 2)]
            )
            
        residual_grid_rep = grid_rep + self.grid_update_mlp(
            grid_rep
        )

        grid_rep = self.m2g_gnn(
            current_mesh_rep, residual_grid_rep, graph_emb["m2g"]
        )

        state_params = self.param_map(
            grid_rep
        )

        if self.output_std:
            mean_delta, std_raw = state_params.chunk(2, dim=-1)
            pred_std = nn.functional.softplus(std_raw)
        else:
            mean_delta = state_params
            pred_std = None

        pred_mean = last_state + mean_delta

        return pred_mean, pred_std

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, maxH=64, maxW=64):
        super().__init__()
        self.d_model = d_model
        self.maxH = maxH
        self.maxW = maxW

        pe = torch.zeros(maxH, maxW, d_model)
        d_model_half = d_model // 2
        row_freq = torch.exp(torch.arange(0, d_model_half, 2, dtype=torch.float32) * 
                             -(math.log(10000.0)/d_model_half))
        col_freq = torch.exp(torch.arange(0, d_model_half, 2, dtype=torch.float32) * 
                             -(math.log(10000.0)/d_model_half))

        for r in range(maxH):
            for i in range(0, d_model_half, 2):
                pe[r, :, i]   = math.sin(r * row_freq[i//2])
                pe[r, :, i+1] = math.cos(r * row_freq[i//2])

        for c in range(maxW):
            for j in range(0, d_model_half, 2):
                pe[:, c, d_model_half + j]   = math.sin(c * col_freq[j//2])
                pe[:, c, d_model_half + j+1] = math.cos(c * col_freq[j//2])

        self.register_buffer('pe', pe)

    def forward(self, x2d):
        B, C, H, W = x2d.shape
        if C > self.d_model:
            raise ValueError(f"Channel({C}) > posenc dim({self.d_model})?")

        pos_slice = self.pe[:H, :W, :C].permute(2,0,1)  # (C, H, W)
        # broadcast to (1, C, H, W)
        x2d = x2d + pos_slice.unsqueeze(0)
        return x2d
    
class GridEncoder(nn.Module):
    def __init__(self, var_dim, embed_dim, H, W, n_conv_layers=3, hidden_ch=64):
        super().__init__()
        self.var_dim = var_dim
        self.embed_dim = embed_dim
        self.H = H
        self.W = W

        self.pos_enc2d = PositionalEncoding2D(d_model=var_dim, maxH=H, maxW=W)

        self.initial_conv = nn.Sequential(
            nn.Conv2d(var_dim, hidden_ch, kernel_size=3, padding=1),
            nn.SiLU()
        )

        convs = []
        for i in range(n_conv_layers - 1):
            convs.append(nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1))
            convs.append(nn.SiLU())
        self.conv_blocks = nn.Sequential(*convs)

        hidden_fc = embed_dim * 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_ch, hidden_fc),
            nn.SiLU(),
            nn.Linear(hidden_fc, embed_dim)
        )

    def forward(self, grid):
        B, HW, V = grid.shape
        if HW != self.H*self.W:
            raise ValueError("grid size mismatch")

        x2d = grid.reshape(B, self.H, self.W, V).permute(0,3,1,2).contiguous()  
        x2d = self.pos_enc2d(x2d)

        feat = self.initial_conv(x2d)
        feat = self.conv_blocks(feat)

        feat_mean = feat.mean(dim=[2,3])

        out = self.fc(feat_mean)
        return out

class AdaLNZeroBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, x, cond_3d):
        B, N, D = x.shape
        cond_3d = cond_3d.view(B, 3, D)

        alpha = cond_3d[:, 0, :]
        gamma = cond_3d[:, 1, :]
        beta  = cond_3d[:, 2, :]

        x_ln = self.ln(x)

        gamma_ = gamma.unsqueeze(1).expand(B, N, D)
        beta_  = beta.unsqueeze(1).expand(B, N, D)

        out_ln = gamma_ * x_ln + beta_

        return out_ln, alpha

class NHopAttentionBlockwithTimeEmbedding(nn.Module):
    def __init__(self,
                 d_model,
                 num_heads,
                 max_hop=4,
                 max_position=1024,
                 dim_feedforward=2048,
                 dropout=0.1,
                 use_pos_enc=False,
                 num_layers=1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_pos_enc = use_pos_enc
        self.num_layers = num_layers

        if use_pos_enc:
            self.pos_enc = PositionalEncoding1D(d_model, max_len=max_position)
        else:
            self.pos_enc = None

        self.max_hop = max_hop
        self.max_position = max_position

        self.hop_biases = nn.ParameterList([
            nn.Parameter(torch.empty(max_position, max_position))
            for _ in range(max_hop)
        ])

        self.mha_layers = nn.ModuleList()
        self.ff_layers = nn.ModuleList()
        self.adaLN1_blocks = nn.ModuleList()
        self.adaLN2_blocks = nn.ModuleList()

        for _ in range(num_layers):
            self.mha_layers.append(nn.MultiheadAttention(d_model, num_heads, batch_first=True))
            self.ff_layers.append(
                nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, d_model)
                )
            )
            self.adaLN1_blocks.append(AdaLNZeroBlock(d_model))
            self.adaLN2_blocks.append(AdaLNZeroBlock(d_model))

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for i, p in enumerate(self.hop_biases):
            scale = float(self.max_hop - i)/float(self.max_hop)
            nn.init.constant_(p, scale)

    def create_n_hop_mask(self, edge_index, num_nodes):
        adjacency_1 = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=edge_index.device)
        adjacency_1[edge_index[0], edge_index[1]] = 1.0

        mask_sum = torch.zeros_like(adjacency_1)
        current = adjacency_1.clone()

        for hop in range(self.max_hop):
            mask_sum += current * self.hop_biases[hop][:num_nodes, :num_nodes]
            current = (current @ adjacency_1).clamp_(max=1.0)

        return mask_sum

    def forward(self, x, edge_index, cond_block=None):
        if cond_block is None:
            raise ValueError("Need cond_block for AdaLN (NHopAttn)")

        B, N, D = x.shape
        attn_mask = self.create_n_hop_mask(edge_index, N)

        if self.pos_enc is not None:
            x = self.pos_enc(x)

        for layer_idx in range(self.num_layers):
            start = layer_idx * (6*D)
            end   = (layer_idx+1) * (6*D)
            cond_6d = cond_block[:, start:end]

            cond1 = cond_6d[:, :3*D]
            cond2 = cond_6d[:, 3*D:]

            x_ln, alpha1 = self.adaLN1_blocks[layer_idx](x, cond1)
            attn_out, _ = self.mha_layers[layer_idx](x_ln, x_ln, x_ln, attn_mask=attn_mask)
            alpha1_ = alpha1.unsqueeze(1).expand(B,N,D)
            x = x + alpha1_ * self.dropout(attn_out)

            x_ln2, alpha2 = self.adaLN2_blocks[layer_idx](x, cond2)
            ff_out = self.ff_layers[layer_idx](x_ln2)
            alpha2_ = alpha2.unsqueeze(1).expand(B,N,D)
            x = x + alpha2_ * self.dropout(ff_out)

        return x

class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0)/d_model)
        )
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        B,N,D = x.shape
        x = x + self.pe[:, :N, :D]
        return x
    
def timestep_embedding(timesteps, emb_dim=256):
    import math
    half = emb_dim//2
    freqs = torch.exp(
        math.log(10000.) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device)/half
    )
    args = timesteps[:,None].float() * freqs[None]
    emb_sin = torch.sin(args)
    emb_cos = torch.cos(args)
    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb

class CondModel(nn.Module):
    def __init__(self, d_model, num_layers=1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6*d_model*num_layers)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, cond):
        return self.net(cond)

class Denoiser(nn.Module):
    def __init__(self,
                 in_dim=256,
                 out_dim=256,
                 d_model=768,
                 max_hop=4,
                 num_heads=12,
                 dim_feedforward=768 * 4,
                 dropout=0.0,
                 max_position=1024,
                 grid_H=64, grid_W=32, var_dim=256, n_cnn_layers=3,
                 nhop_num_layers=12,
                 use_uncond_token=True
                ):
        super().__init__()

        self.d_model = d_model
        
        self.in_proj = nn.Linear(in_dim, d_model)
        
        self.use_uncond_token = use_uncond_token
        if use_uncond_token:
            self.uncond_token = nn.Parameter(torch.zeros(d_model))
        else:
            self.uncond_token = None 

        self.grid_encoder = GridEncoder(
            var_dim=var_dim,
            embed_dim=d_model,
            H=grid_H,
            W=grid_W,
            n_conv_layers=n_cnn_layers,
            hidden_ch=512
        )

        self.first_nhop_condModel = CondModel(d_model, num_layers=nhop_num_layers)
        self.first_nhop_block = NHopAttentionBlockwithTimeEmbedding(
            d_model=d_model,
            num_heads=num_heads,
            max_hop=max_hop,
            max_position=max_position,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_pos_enc=True,
            num_layers=nhop_num_layers
        )

        self.out_proj = nn.Linear(d_model, out_dim)

    def get_t_embed(self, t):
        B = t.shape[0]
        t_emb256 = timestep_embedding(t, 256)
        hidden = self.d_model
        net = nn.Sequential(
            nn.Linear(256, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.d_model)
        ).to(t.device)
        return net(t_emb256)

    def forward(self, x, edge_index, grid=None, t=None):
        B, N, C = x.shape
        
        x = self.in_proj(x)

        grid_enc = self.grid_encoder(grid) 
        t_emb = self.get_t_embed(t)
        cond_sum = grid_enc + t_emb

        cond_block = self.first_nhop_condModel(cond_sum)

        x_nhop = self.first_nhop_block(x, edge_index, cond_block=cond_block)

        out = self.out_proj(x_nhop)
        return out