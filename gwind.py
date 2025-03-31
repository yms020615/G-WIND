# Third-party
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import distributions as tdists
import wandb
from diffusers import DPMSolverMultistepScheduler

# First-party
import constants, metrics, utils, vis
from ar_model import ARModel
from hi_graph_latent_decoder import *

BOUNDARIES = {
    'NorthAmerica': {  # 9 x 14
        'lat_range': (15, 65),
        'lon_range': (220, 300)
    },
    'SouthAmerica': {  # 14 x 11
        'lat_range': (-55, 20),
        'lon_range': (270, 330)
    },
    'Europe': {        # 7 x 7
        'lat_range': (30, 65),
        'lon_range': (0, 40)
    },
    'SouthAsia': {     # 11 x 16
        'lat_range': (-15, 45),
        'lon_range': (25, 110)
    },
    'EastAsia': {      # 11 x 15
        'lat_range': (5, 65),
        'lon_range': (70, 150)
    },
    'Australia': {     # 11 x 14
        'lat_range': (-50, 10),
        'lon_range': (100, 180)
    },
    'Africa': {
        'lat_range': (-35, 38),
        'lon_range': (0, 52)
    },
    'Antarctica': {
        'lat_range': (-90, -60),
        'lon_range': (0, 360)
    },
    'Arctic': {
        'lat_range': (66, 90),
        'lon_range': (0, 360)
    },
    'Nino3_4': {
        'lat_range': (-5, 5),
        'lon_range': (190, 240)
    },
}

def prepare_region_masks(lat_1d: torch.Tensor, lon_1d: torch.Tensor, boundaries: dict):
    H = lat_1d.shape[0]  # 32
    W = lon_1d.shape[0]  # 64

    region_masks_1d = {}

    for region_name, info in boundaries.items():
        lat_min, lat_max = info['lat_range']
        lon_min, lon_max = info['lon_range']

        lat_mask = (lat_1d >= lat_min) & (lat_1d <= lat_max)  # (32,)
        lon_mask = (lon_1d >= lon_min) & (lon_1d <= lon_max)  # (64,)

        region_mask_2d = lat_mask.unsqueeze(-1) & lon_mask.unsqueeze(0)

        region_mask_1d = region_mask_2d.flatten()

        region_masks_1d[region_name] = region_mask_1d

        true_count = region_mask_1d.sum().item()
        print(
            f"[{region_name}] lat_range={lat_min}~{lat_max}, "
            f"lon_range={lon_min}~{lon_max} -> True grid points: {true_count}"
        )

    return region_masks_1d

def build_lat_lon_centered(num_lat=32, num_lon=64):
    lat_step = 180.0 / num_lat  # 5.625
    lon_step = 360.0 / num_lon  # 5.625

    lat_1d = torch.tensor([
        -90 + (k + 0.5) * lat_step
        for k in range(num_lat)
    ])
    lon_1d = torch.tensor([
        0 + (k + 0.5) * lon_step
        for k in range(num_lon)
    ])
    return lat_1d, lon_1d

class GWIND(ARModel):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
                
        assert (
            args.n_example_pred <= args.batch_size
        ), "Can not plot more examples than batch size in GWIND"
        self.sample_obs_noise = bool(args.sample_obs_noise)
        self.ensemble_size = args.ensemble_size
        
        self.kl_beta = 0.0
        self.alpha = 0.0
        self.kl_beta_max = args.kl_beta
        self.alpha_max = 1.0
        self.kl_beta_ramp_epochs = 20
        self.alpha_ramp_epochs = 300
        self.freeze_decoder_until_epoch = 300
        
        self.crps_weight = args.crps_weight
        
        self.lat_1d, self.lon_1d = build_lat_lon_centered()
        self.region_masks_1d = prepare_region_masks(self.lat_1d, self.lon_1d, BOUNDARIES)

        self.hierarchical_graph, graph_ldict = utils.load_graph(args.graph)
        for name, attr_value in graph_ldict.items():
            if isinstance(attr_value, torch.Tensor):
                self.register_buffer(name, attr_value, persistent=False)
            else:
                setattr(self, name, attr_value)

        grid_current_dim = self.grid_dim + constants.GRID_STATE_DIM
        g2m_dim = self.g2m_features.shape[1]
        m2g_dim = self.m2g_features.shape[1]

        self.mlp_blueprint_end = [args.hidden_dim] * (args.hidden_layers + 1)
        self.grid_prev_embedder = utils.make_mlp(
            [self.grid_dim] + self.mlp_blueprint_end
        )
        self.grid_prev_embedder_for_denoiser = utils.make_mlp(
            [self.grid_dim] + self.mlp_blueprint_end
        )
        self.grid_current_embedder = utils.make_mlp(
            [grid_current_dim] + self.mlp_blueprint_end
        ) 

        self.g2m_embedder = utils.make_mlp([g2m_dim] + self.mlp_blueprint_end)
        self.m2g_embedder = utils.make_mlp([m2g_dim] + self.mlp_blueprint_end)
        if self.hierarchical_graph:
            print("Loaded hierarchical graph with structure:")
            level_mesh_sizes = [
                mesh_feat.shape[0] for mesh_feat in self.mesh_static_features
            ]
            self.num_mesh_nodes = level_mesh_sizes[-1]
            num_levels = len(self.mesh_static_features)
            for level_index, level_mesh_size in enumerate(level_mesh_sizes):
                same_level_edges = self.m2m_features[level_index].shape[0]
                print(
                    f"level {level_index} - {level_mesh_size} nodes, "
                    f"{same_level_edges} same-level edges"
                )

                if level_index < (num_levels - 1):
                    up_edges = self.mesh_up_features[level_index].shape[0]
                    down_edges = self.mesh_down_features[level_index].shape[0]
                    print(f"  {level_index}<->{level_index+1}")
                    print(f" - {up_edges} up edges, {down_edges} down edges")
            mesh_dim = self.mesh_static_features[0].shape[1]
            m2m_dim = self.m2m_features[0].shape[1]
            mesh_up_dim = self.mesh_up_features[0].shape[1]
            mesh_down_dim = self.mesh_down_features[0].shape[1]

            self.mesh_embedders = torch.nn.ModuleList(
                [
                    utils.make_mlp([mesh_dim] + self.mlp_blueprint_end)
                    for _ in range(num_levels)
                ]
            )
            self.mesh_up_embedders = torch.nn.ModuleList(
                [
                    utils.make_mlp([mesh_up_dim] + self.mlp_blueprint_end)
                    for _ in range(num_levels - 1)
                ]
            )
            self.mesh_down_embedders = torch.nn.ModuleList(
                [
                    utils.make_mlp([mesh_down_dim] + self.mlp_blueprint_end)
                    for _ in range(num_levels - 1)
                ]
            )
            self.embedd_m2m = (
                max(
                    args.prior_processor_layers,
                    args.encoder_processor_layers,
                    args.processor_layers,
                )
                > 0
            )
            if self.embedd_m2m:
                self.m2m_embedders = torch.nn.ModuleList(
                    [
                        utils.make_mlp([m2m_dim] + self.mlp_blueprint_end)
                        for _ in range(num_levels)
                    ]
                )
        else:
            self.num_mesh_nodes, mesh_static_dim = (
                self.mesh_static_features.shape
            )
            print(
                f"Loaded graph with {self.num_grid_nodes + self.num_mesh_nodes}"
                f"nodes ({self.num_grid_nodes} grid, "
                f"{self.num_mesh_nodes} mesh)"
            )
            mesh_static_dim = self.mesh_static_features.shape[1]
            self.mesh_embedder = utils.make_mlp(
                [mesh_static_dim] + self.mlp_blueprint_end
            )
            m2m_dim = self.m2m_features.shape[1]
            self.m2m_embedder = utils.make_mlp(
                [m2m_dim] + self.mlp_blueprint_end
            )

        latent_dim = args.latent_dim if args.latent_dim is not None else args.hidden_dim
        self.latent_dim = latent_dim

        self.prior = HiGraphLatentEncoder(
            latent_dim,
            self.g2m_edge_index,
            self.m2m_edge_index,
            self.mesh_up_edge_index,
            args.hidden_dim,
            args.prior_processor_layers,
            hidden_layers=args.hidden_layers,
            output_dist="isotropic",
        )

        self.encoder = HiGraphLatentEncoder(
            latent_dim,
            self.g2m_edge_index,
            self.m2m_edge_index,
            self.mesh_up_edge_index,
            args.hidden_dim,
            args.encoder_processor_layers,
            hidden_layers=args.hidden_layers,
            output_dist="diagonal", 
        )

        self.decoder = HiGraphLatentDecoder(
            self.g2m_edge_index,
            self.m2m_edge_index,
            self.m2g_edge_index,
            self.mesh_up_edge_index,
            self.mesh_down_edge_index,
            args.hidden_dim,
            args.hidden_layers,
            args.processor_layers
        )
        
        self.pretraining = False
        self.freeze_encoder_decoder = False
        self.load_only_encoder_decoder = False
        
        self.denoiser = Denoiser(in_dim=args.hidden_dim, out_dim=args.hidden_dim)
    
        if self.freeze_encoder_decoder:
            self.prior.eval()
            for p in self.prior.parameters():
                p.requires_grad = False

            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False

            self.decoder.eval()
            for p in self.decoder.parameters():
                p.requires_grad = False

            self.grid_prev_embedder.eval()
            for p in self.grid_prev_embedder.parameters():
                p.requires_grad = False

            self.grid_current_embedder.eval()
            for p in self.grid_current_embedder.parameters():
                p.requires_grad = False

            self.g2m_embedder.eval()
            for p in self.g2m_embedder.parameters():
                p.requires_grad = False

            self.m2g_embedder.eval()
            for p in self.m2g_embedder.parameters():
                p.requires_grad = False

            self.mesh_embedders.eval()
            for module in self.mesh_embedders:
                for p in module.parameters():
                    p.requires_grad = False

            self.mesh_up_embedders.eval()
            for module in self.mesh_up_embedders:
                for p in module.parameters():
                    p.requires_grad = False

            self.mesh_down_embedders.eval()
            for module in self.mesh_down_embedders:
                for p in module.parameters():
                    p.requires_grad = False

            self.m2m_embedders.eval()
            for module in self.m2m_embedders:
                for p in module.parameters():
                    p.requires_grad = False

        else:
            self.prior.train()
            for p in self.prior.parameters():
                p.requires_grad = True

            self.encoder.train()
            for p in self.encoder.parameters():
                p.requires_grad = True

            self.decoder.train()
            for p in self.decoder.parameters():
                p.requires_grad = True

            self.grid_prev_embedder.train()
            for p in self.grid_prev_embedder.parameters():
                p.requires_grad = True

            self.grid_current_embedder.train()
            for p in self.grid_current_embedder.parameters():
                p.requires_grad = True

            self.g2m_embedder.train()
            for p in self.g2m_embedder.parameters():
                p.requires_grad = True

            self.m2g_embedder.train()
            for p in self.m2g_embedder.parameters():
                p.requires_grad = True

            self.mesh_embedders.train()
            for module in self.mesh_embedders:
                for p in module.parameters():
                    p.requires_grad = True

            self.mesh_up_embedders.train()
            for module in self.mesh_up_embedders:
                for p in module.parameters():
                    p.requires_grad = True

            self.mesh_down_embedders.train()
            for module in self.mesh_down_embedders:
                for p in module.parameters():
                    p.requires_grad = True

            self.m2m_embedders.train()
            for module in self.m2m_embedders:
                for p in module.parameters():
                    p.requires_grad = True
        
        self.timesteps = 1000
        self.sampling_steps = 20
        
        self.scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=self.timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='linear',
            solver_order=2,
            prediction_type='epsilon',
            thresholding=False,
            dynamic_thresholding_ratio=0.995,
            sample_max_value=1.0,
            algorithm_type='dpmsolver++',
            solver_type='midpoint',
            lower_order_final=True,
            use_karras_sigmas=True
        )
        
        betas = torch.linspace(self.scheduler.config.beta_start, self.scheduler.config.beta_end, self.scheduler.config.num_train_timesteps)
        self.betas = betas
        self.scheduler.config["betas"] = betas
    
        alphas = 1.0 - betas
        self.alphas = alphas
        
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod = alphas_cumprod
        
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        self.alphas_cumprod_prev = alphas_cumprod_prev
        
        logvar_init = 0.0  
        self.logvar = torch.full(fill_value=logvar_init, size=(self.timesteps,))

        self.learn_logvar = True  
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

        if not self.pretraining:
            self.val_metrics.update(
                {
                    "spread_squared": [],
                    "ens_mse": [],
                    "crps_ens": [],
                }
            )
            self.test_metrics.update(
                {
                    "spread_squared": [],
                    "ens_mse": [],
                    "ens_mae": [],
                    "crps_ens": [],
                }
            )

    def embedd_current(
        self,
        prev_state=None,
        prev_prev_state=None,
        forcing=None,
        current_state=None,
    ):
        batch_size = (
            prev_prev_state.shape[0]
            if prev_prev_state is not None
            else prev_state.shape[0]
        )

        # Feature concatenation with dynamic checking
        grid_features = []
        if prev_prev_state is not None:
            grid_features.append(prev_prev_state)
            if prev_state is None:
                grid_features.append(prev_prev_state)

        if prev_state is not None:
            grid_features.append(prev_state)
        if forcing is not None:
            grid_features.append(forcing)
        grid_features.append(self.expand_to_batch(self.grid_static_features, batch_size))

        if current_state is not None:
            grid_features.append(current_state)  # Include current state
            grid_features = torch.cat(grid_features, dim=-1)  # (B, num_grid_nodes, grid_current_dim)
            return self.grid_current_embedder(grid_features)
        elif prev_state is not None and prev_prev_state is not None:
            grid_features = torch.cat(grid_features, dim=-1)  # (B, num_grid_nodes, grid_prev_dim)
            return self.grid_prev_embedder(grid_features)
        elif prev_prev_state is not None:
            grid_features = torch.cat(grid_features, dim=-1)  # (B, num_grid_nodes, grid_prev_prev_dim)
            return self.grid_prev_prev_embedder(grid_features)
        else:
            raise ValueError("At least one of prev_state, prev_prev_state, or current_state must be provided.")
        
    def embedd_all(self, prev_state, prev_prev_state, forcing):
        batch_size = prev_state.shape[0]

        grid_features = torch.cat(
            (
                prev_prev_state,
                prev_state,
                forcing,
                self.expand_to_batch(self.grid_static_features, batch_size),
            ),
            dim=-1,
        )  # (B, num_grid_nodes, grid_dim)

        grid_emb = self.grid_prev_embedder(grid_features)
        # (B, num_grid_nodes, d_h)

        # Graph embedding
        graph_emb = {
            "g2m": self.expand_to_batch(
                self.g2m_embedder(self.g2m_features), batch_size
            ),  # (B, M_g2m, d_h)
            "m2g": self.expand_to_batch(
                self.m2g_embedder(self.m2g_features), batch_size
            ),  # (B, M_m2g, d_h)
        }

        if self.hierarchical_graph:
            graph_emb["mesh"] = [
                self.expand_to_batch(emb(node_static_features), batch_size)
                for emb, node_static_features in zip(
                    self.mesh_embedders,
                    self.mesh_static_features,
                )
            ]  # each (B, num_mesh_nodes[l], d_h)

            if self.embedd_m2m:
                graph_emb["m2m"] = [
                    self.expand_to_batch(emb(edge_feat), batch_size)
                    for emb, edge_feat in zip(
                        self.m2m_embedders, self.m2m_features
                    )
                ]
            else:
                # Need a placeholder otherwise, just use raw features
                graph_emb["m2m"] = list(self.m2m_features)

            graph_emb["mesh_up"] = [
                self.expand_to_batch(emb(edge_feat), batch_size)
                for emb, edge_feat in zip(
                    self.mesh_up_embedders, self.mesh_up_features
                )
            ]
            graph_emb["mesh_down"] = [
                self.expand_to_batch(emb(edge_feat), batch_size)
                for emb, edge_feat in zip(
                    self.mesh_down_embedders, self.mesh_down_features
                )
            ]
        else:
            graph_emb["mesh"] = self.expand_to_batch(
                self.mesh_embedder(self.mesh_static_features), batch_size
            )  # (B, num_mesh_nodes, d_h)
            graph_emb["m2m"] = self.expand_to_batch(
                self.m2m_embedder(self.m2m_features), batch_size
            )  # (B, M_m2m, d_h)

        return grid_emb, graph_emb
    
    def embedd_all_for_denoiser(self, prev_state, prev_prev_state, forcing):
        batch_size = prev_state.shape[0]

        grid_features = torch.cat(
            (
                prev_prev_state,
                prev_state,
                forcing,
                self.expand_to_batch(self.grid_static_features, batch_size),
            ),
            dim=-1,
        )  # (B, num_grid_nodes, grid_dim)

        grid_emb = self.grid_prev_embedder_for_denoiser(grid_features)
        # (B, num_grid_nodes, d_h)

        return grid_emb
    
    def on_train_epoch_start(self):
        epoch = self.current_epoch

        # kl_beta update
        if self.pretraining:
            if epoch >= self.kl_beta_ramp_epochs:
                factor = 1.0
            else:
                factor = float(epoch) / float(self.kl_beta_ramp_epochs)
        else:
            factor = 1.0
        # factor = 1.0
        self.kl_beta = self.kl_beta_max * factor
    
    def training_step(self, batch):
        device = self.device
    
        init_states, target_states, forcing_features, sample_times = batch
        init_states = init_states.to(device)
        target_states = target_states.to(device)
        forcing_features = forcing_features.to(device)
        sample_times = sample_times.to(device)

        total_loss = 0

        if self.pretraining:
            prev_prev_state = init_states[:, 0]
            prev_state = init_states[:, 1]
            
            pred_steps = forcing_features.shape[1]
            preds = []
            
            for i in range(pred_steps): # range(self.current_epoch + 2):
                current_state = target_states[:, i]
                forcing = forcing_features[:, i]
            
                grid_prev_emb, graph_emb = self.embedd_all(prev_state, prev_prev_state, forcing)

                grid_current_emb = self.embedd_current(prev_state, prev_prev_state, forcing, current_state)

                mean, logvar = self.prior(grid_prev_emb, graph_emb)
                mean_va, logvar_va = self.encoder(grid_current_emb, graph_emb)

                std = torch.exp(0.5 * logvar) if logvar is not None else torch.ones_like(mean)
                std_va = torch.exp(0.5 * logvar_va) if logvar_va is not None else torch.ones_like(mean_va)

                prior_dist = tdists.Normal(mean, std)
                var_dist = tdists.Normal(mean_va, std_va)

                z_va = var_dist.rsample()

                pred_mean, model_pred_std = self.decoder(
                    grid_rep=grid_prev_emb,
                    latent_sample=z_va,
                    last_state=prev_state,
                    graph_emb=graph_emb,
                )

                if self.output_std:
                    pred_std = model_pred_std
                else:
                    pred_std = self.per_var_std
                    
                preds.append(pred_mean)

                entry_likelihoods = -self.loss(
                    pred_mean,
                    current_state,
                    pred_std,
                    mask=self.interior_mask_bool,
                    grid_weights=self.grid_weights,
                    average_grid=False,
                    sum_vars=False,
                )

                likelihood_term = torch.sum(entry_likelihoods, dim=(1, 2))  # (B,)
                nll_loss = -torch.mean(likelihood_term)

                kl_loss = torch.sum(
                    torch.distributions.kl_divergence(var_dist, prior_dist),
                    dim=(1, 2)
                ).mean()
                kl_loss_weighted = self.kl_beta * kl_loss

                elbo_loss = nll_loss + kl_loss_weighted
                total_loss += elbo_loss
                
                new_state = pred_mean
                prev_prev_state = prev_state
                prev_state = new_state

            self.log("pretrain_nll_loss", nll_loss, prog_bar=True)
            self.log("pretrain_kl_loss", kl_loss, prog_bar=True)
            self.log("pretrain_elbo", elbo_loss, prog_bar=True)
            self.log("kl_beta", self.kl_beta, prog_bar=True)
            self.log("train_loss", total_loss, prog_bar=True)

            return total_loss

        else:
            B = init_states.size(0)

            prev_prev_state = init_states[:, 0]
            prev_state = init_states[:, 1] 

            pred_steps = forcing_features.shape[1]

            nll_losses = []
            kl_losses = []
            eps_losses = []
            eps_losses2 = []
            recon_losses = []
            
            self.scheduler.set_timesteps(self.timesteps, device=device)
            
            self.denoiser.train()
            
            recon_x_sample = []

            for i in range(pred_steps):
                forcing = forcing_features[:, i].to(device)
                current_state = target_states[:, i].to(device)

                grid_prev_emb, graph_emb = self.embedd_all(prev_state, prev_prev_state, forcing)
                grid_prev_emb_d = self.embedd_all_for_denoiser(prev_state, prev_prev_state, forcing)
                mean, logvar = self.prior(grid_prev_emb, graph_emb)
                
                grid_current_emb = self.embedd_current(prev_state, prev_prev_state, forcing, current_state)
                mean_va, logvar_va = self.encoder(grid_current_emb, graph_emb)
                
                std = torch.exp(0.5 * logvar) if logvar is not None else torch.ones_like(mean)
                std_va = torch.exp(0.5 * logvar_va) if logvar_va is not None else torch.ones_like(mean_va)
                
                prior_dist = tdists.Normal(mean, std)
                var_dist = tdists.Normal(mean_va, std_va)

                z = prior_dist.rsample()
                z_va = var_dist.rsample()

                t = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,), device=device).long()

                self.alphas_cumprod = self.alphas_cumprod.to(device)
                alpha_bar_t = self.alphas_cumprod[t].view(B, *((1,) * (z_va.ndim - 1))).to(device)
                sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
                sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)

                epsilon = torch.randn_like(z_va) * std_va
                Z_t = sqrt_alpha_bar_t * (z_va - mean_va) + sqrt_one_minus_alpha_bar_t * epsilon

                epsilon_pred = self.denoiser(
                    x=Z_t,
                    edge_index=self.m2m_edge_index[1],
                    grid=grid_prev_emb_d,
                    t=t,
                )
                
                loss_simple = ((epsilon_pred - epsilon)**2 / (std_va**2)).mean(dim=(1,2))
                logvar_values = self.logvar[t]
                eps_loss_val = loss_simple / torch.exp(logvar_values) + logvar_values
                
                eps_losses.append(eps_loss_val)

                Z_0_t = (Z_t - sqrt_one_minus_alpha_bar_t * epsilon_pred) / sqrt_alpha_bar_t + mean_va

                recon_x, _ = self.decoder(
                    grid_rep=grid_prev_emb,
                    latent_sample=Z_0_t,
                    last_state=prev_state,
                    graph_emb=graph_emb,
                )
                recon_loss_val = torch.mean((recon_x - current_state) ** 2)
                recon_losses.append(recon_loss_val)

                pred_std = self.per_var_std
                entry_likelihoods = -self.loss(
                    recon_x,
                    current_state,
                    pred_std,
                    mask=self.interior_mask_bool,
                    grid_weights=self.grid_weights,
                    average_grid=False,
                    sum_vars=False,
                )
                likelihood_term = torch.sum(entry_likelihoods, dim=(1, 2))
                nll_loss = -torch.mean(likelihood_term)
                nll_losses.append(nll_loss)

                kl_loss = torch.sum(
                    torch.distributions.kl_divergence(var_dist, prior_dist),
                    dim=(1, 2)
                ).mean()
                kl_losses.append(kl_loss)

                new_state = recon_x
                prev_prev_state = prev_state
                prev_state = new_state
                
            if self.crps_weight > 0:
                S = 5
                recon_x_samples = []

                for s in range(S):
                    recon_x_sample = []
                    
                    prev_prev_state = init_states[:, 0]
                    prev_state = init_states[:, 1]
                    
                    for i in range(pred_steps):
                        forcing = forcing_features[:, i].to(device)
                        current_state = target_states[:, i].to(device)

                        grid_prev_emb, graph_emb = self.embedd_all(prev_state, prev_prev_state, forcing)
                        grid_prev_emb_d = self.embedd_all_for_denoiser(prev_state, prev_prev_state, forcing)

                        grid_current_emb = self.embedd_current(prev_state, prev_prev_state, forcing, current_state)
                        mean_va, logvar_va = self.encoder(grid_current_emb, graph_emb)
                        
                        std = torch.exp(0.5 * logvar) if logvar is not None else torch.ones_like(mean)
                        std_va = torch.exp(0.5 * logvar_va) if logvar_va is not None else torch.ones_like(mean_va)
                        
                        prior_dist = tdists.Normal(mean, std)
                        var_dist = tdists.Normal(mean_va, std_va)

                        z = prior_dist.rsample()
                        z_va = var_dist.rsample()
                
                        t = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,), device=device).long()

                        self.alphas_cumprod = self.alphas_cumprod.to(device)
                        alpha_bar_t = self.alphas_cumprod[t].view(B, *((1,) * (z.ndim - 1))).to(device)
                        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
                        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)

                        epsilon = torch.randn_like(z) * std_va
                        Z_t = sqrt_alpha_bar_t * (z_va - mean_va) + sqrt_one_minus_alpha_bar_t * epsilon

                        epsilon_pred = self.denoiser(Z_t, self.m2m_edge_index[1], grid_prev_emb_d, t)
                            
                        Z_0_t = (Z_t - sqrt_one_minus_alpha_bar_t * epsilon_pred) / sqrt_alpha_bar_t + mean_va

                        recon_x_s, _ = self.decoder(
                            grid_rep=grid_prev_emb,
                            latent_sample=Z_0_t,
                            last_state=prev_state,
                            graph_emb=graph_emb,
                        )
                        
                        recon_x_sample.append(recon_x_s)
                        
                        new_state = recon_x_s
                        prev_prev_state = prev_state
                        prev_state = new_state
                        
                    recon_x_samples.append(torch.stack(recon_x_sample, dim=1))

                recon_x_samples = torch.stack(recon_x_samples, dim=1)

                crps_estimate = metrics.crps_loss(
                    recon_x_samples,
                    target_states,
                    self.per_var_std,
                    grid_weights=self.grid_weights,
                    mask=self.interior_mask_bool,
                )
                crps_loss = torch.mean(crps_estimate)

                total_loss += self.crps_weight * crps_loss
                self.log("crps_loss", crps_loss, prog_bar=True)

            nll_loss_all = torch.mean(torch.stack(nll_losses)) if len(nll_losses) > 0 else 0.0
            kl_loss_all = torch.mean(torch.stack(kl_losses)) if len(kl_losses) > 0 else 0.0
            eps_loss_all = torch.mean(torch.stack(eps_losses)) if len(eps_losses) > 0 else 0.0
            eps_loss_all2 = torch.mean(torch.stack(eps_losses2)) if len(eps_losses2) > 0 else 0.0
            recon_loss_all = torch.mean(torch.stack(recon_losses)) if len(recon_losses) > 0 else 0.0

            kl_loss_weighted = self.kl_beta * kl_loss_all

            elbo_loss = nll_loss_all + kl_loss_weighted

            total_loss += eps_loss_all
            
            self.log("train_loss", total_loss, prog_bar=True)
            self.log("nll_loss", nll_loss_all, prog_bar=True)
            self.log("kl_loss", kl_loss_all, prog_bar=True)
            self.log("eps_loss", eps_loss_all, prog_bar=True)
            self.log("eps_loss_no_snr", eps_loss_all2, prog_bar=True)
            self.log("recon_loss", recon_loss_all, prog_bar=True)
            
            self.log("Z_0_t_mean", torch.mean(Z_0_t), prog_bar=True)
            self.log("z_mean", torch.mean(z), prog_bar=True)
            self.log("z_std", torch.std(z), prog_bar=True)
            self.log("recon_x_mean", torch.mean(recon_x), prog_bar=True)
            
            return total_loss

    def plot_examples(self, batch, n_examples, prediction=None):
        """
        Plot ensemble forecast + mean and std
        """
        init_states, target_states, forcing_features, sample_times = batch

        trajectories, _ = self.sample_prediction(
            init_states,
            forcing_features,
            target_states,
            sample_times,
            steps=40,
            ensemble_size=self.ensemble_size,
        )
        # (B, S, pred_steps, num_grid_nodes, d_f)

        # Rescale to original data scale
        traj_rescaled = trajectories * self.data_std + self.data_mean
        target_rescaled = target_states * self.data_std + self.data_mean

        # Compute mean and std of ensemble
        ens_mean = torch.mean(
            traj_rescaled, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)
        ens_std = torch.std(
            traj_rescaled, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)

        # Iterate over the examples
        for traj_slice, target_slice, ens_mean_slice, ens_std_slice in zip(
            traj_rescaled[:n_examples],
            target_rescaled[:n_examples],
            ens_mean[:n_examples],
            ens_std[:n_examples],
        ):
            # traj_slice is (S, pred_steps, num_grid_nodes, d_f)
            # others are (pred_steps, num_grid_nodes, d_f)
            self.plotted_examples += 1  # Increment already here

            # Note: min and max values can not be in ensemble mean
            var_vmin = (
                torch.minimum(
                    traj_slice.flatten(0, 2).min(dim=0)[0],
                    target_slice.flatten(0, 1).min(dim=0)[0],
                )
                .cpu()
                .numpy()
            )  # (d_f,)
            var_vmax = (
                torch.maximum(
                    traj_slice.flatten(0, 2).max(dim=0)[0],
                    target_slice.flatten(0, 1).max(dim=0)[0],
                )
                .cpu()
                .numpy()
            )  # (d_f,)
            var_vranges = list(zip(var_vmin, var_vmax))

            # Iterate over prediction horizon time steps
            for t_i, (samples_t, target_t, ens_mean_t, ens_std_t) in enumerate(
                zip(
                    traj_slice.transpose(0, 1),
                    # (pred_steps, S, num_grid_nodes, d_f)
                    target_slice,
                    ens_mean_slice,
                    ens_std_slice,
                ),
                start=1,
            ):
                if t_i not in [20, 40]:
                    continue
                
                time_title_part = f"t={t_i} ({self.step_length*t_i} h)"
                # Create one figure per variable at this time step
                var_names = [
                    constants.PARAM_NAMES_SHORT[var_i]
                    for var_i in constants.EVAL_PLOT_VARS
                ]
                var_figs = [
                    vis.plot_ensemble_prediction(
                        samples_t[:, :, var_i],
                        target_t[:, var_i],
                        ens_mean_t[:, var_i],
                        ens_std_t[:, var_i],
                        self.interior_mask[:, 0],
                        title=(
                            f"{var_name} "
                            f"({constants.PARAM_UNITS[var_i]}), "
                            f"{time_title_part}"
                        ),
                        vrange=var_vranges[var_i],
                    )
                    for var_i, var_name in zip(
                        constants.EVAL_PLOT_VARS, var_names
                    )
                ]

                example_title = f"example_{self.plotted_examples}"
                wandb.log(
                    {
                        f"{var_name}_{example_title}": wandb.Image(fig)
                        for var_name, fig in zip(var_names, var_figs)
                    }
                )
                plt.close(
                    "all"
                )  # Close all figs for this time step, saves memory

    def sample_prediction(self, init_states, forcing_features, current_state, sample_times, steps=1, ensemble_size=1):
        device = self.device
        B = init_states.shape[0]
        prev_prev_state = init_states[:, 0].to(device)
        prev_state = init_states[:, 1].to(device)
        forcing_features = forcing_features.to(device)
        true_states = current_state.to(device)
        
        predictions = []
        
        for s in range(ensemble_size):
            current_prev_state = prev_state.clone()
            current_prev_prev_state = prev_prev_state.clone()

            pred_list = []
            for i in range(steps):
                forcing = forcing_features[:, i]
                current_state = true_states[:, i]

                grid_prev_emb, graph_emb = self.embedd_all(current_prev_state, current_prev_prev_state, forcing)
                grid_prev_emb_d = self.embedd_all_for_denoiser(current_prev_state, current_prev_prev_state, forcing)

                mean, logvar = self.prior(grid_prev_emb, graph_emb)

                if logvar is not None:
                    std = torch.exp(0.5 * logvar)
                else:
                    std = torch.ones_like(mean)

                prior_dist = tdists.Normal(mean, std)
                z = prior_dist.rsample()
                
                latent = torch.randn_like(mean) * std

                self.scheduler.set_timesteps(self.sampling_steps, device=device)

                # sampler loop
                for t in self.scheduler.timesteps:
                    t_tensor = torch.full((B,), t, dtype=torch.long, device=device)

                    epsilon_pred = self.denoiser(
                        latent,
                        self.m2m_edge_index[1],
                        grid_prev_emb_d, 
                        t_tensor
                    )

                    latent = self.scheduler.step(epsilon_pred, t, latent).prev_sample
                    
                pred_mean, pred_std = self.decoder(
                    grid_rep=grid_prev_emb,
                    latent_sample=latent + mean,
                    last_state=current_prev_state,
                    graph_emb=graph_emb,
                )

                pred_list.append(pred_mean)

                current_prev_prev_state = current_prev_state
                current_prev_state = pred_mean

            pred_list = torch.stack(pred_list, dim=1)
            predictions.append(pred_list)

        predictions = torch.stack(predictions, dim=0)
        predictions = predictions.permute(1, 0, 2, 3, 4).contiguous()
        
        return predictions, None
    
    def validation_step(self, batch, batch_idx):
        device = self.device
        init_states, target_states, forcing_features, sample_times = batch
        init_states = init_states.to(device)
        target_states = target_states.to(device)
        forcing_features = forcing_features.to(device)
        sample_times = sample_times.to(device)

        self.prior.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.denoiser.eval()

        with torch.no_grad():
            if self.pretraining:
                prev_prev_state = init_states[:, 0]
                prev_state = init_states[:, 1]
                current_state = target_states[:, 0]
                forcing = forcing_features[:, 0]

                grid_prev_emb, graph_emb = self.embedd_all(prev_state, prev_prev_state, forcing)

                mean, logvar = self.prior(
                    grid_prev_emb, graph_emb
                )

                if logvar is not None:
                    std = torch.exp(0.5 * logvar)
                else:
                    std = torch.ones_like(mean)

                prior_dist = tdists.Normal(mean, std)
                z = prior_dist.rsample()

                reconstructed_grid, _ = self.decoder(
                    grid_rep=grid_prev_emb,
                    latent_sample=z,
                    last_state=prev_state,
                    graph_emb=graph_emb,
                )
                
                recon_loss = torch.mean((reconstructed_grid - current_state) ** 2)

                self.log("val_mean_loss", recon_loss, prog_bar=True)

                return recon_loss
            else:
                pred_steps = target_states.shape[1]
                ensemble_size = self.ensemble_size
                predictions, pred_std = self.sample_prediction(
                    init_states, forcing_features, target_states, sample_times, steps=pred_steps, ensemble_size=ensemble_size
                )
                pred_std = pred_std if pred_std is not None else self.per_var_std

                target = target_states
                ens_mean = torch.mean(predictions, dim=1)
                time_step_loss = torch.mean(
                    self.loss(ens_mean, target, pred_std, mask=self.interior_mask_bool, grid_weights=self.grid_weights),
                    dim=0,
                )
                mean_loss = torch.mean(time_step_loss)

                val_log_dict = {
                    f"val_loss_unroll{step}": time_step_loss[step - 1]
                    for step in self.val_log_leads if step <= predictions.shape[1]
                }
                val_log_dict["val_mean_loss"] = mean_loss
                self.log_dict(val_log_dict, on_step=False, on_epoch=True, sync_dist=True)

                if ensemble_size > 1:
                    crps_batch = metrics.crps_ens(
                        predictions,
                        target,
                        pred_std,
                        grid_weights=self.grid_weights,
                        mask=self.interior_mask_bool,
                        sum_vars=False,
                    )
                    self.val_metrics["crps_ens"].append(crps_batch)

                spread_squared_batch = metrics.spread_squared(
                    predictions,
                    target,
                    pred_std,
                    grid_weights=self.grid_weights,
                    mask=self.interior_mask_bool,
                    sum_vars=False,
                )
                self.val_metrics["spread_squared"].append(spread_squared_batch)

                # Store MSEs
                ens_mean_batch = metrics.mse(
                    ens_mean,
                    target,
                    None,
                    grid_weights=self.grid_weights,
                    mask=self.interior_mask_bool,
                    sum_vars=False,
                )
                self.val_metrics["ens_mse"].append(ens_mean_batch)

    def log_spsk_ratio(self, metric_vals, prefix):
        # Compute mean spsk_ratio
        spread_squared_tensor = self.all_gather_cat(
            torch.cat(metric_vals["spread_squared"], dim=0)
        )  # (N_eval, pred_steps, d_f)
        ens_mse_tensor = self.all_gather_cat(
            torch.cat(metric_vals["ens_mse"], dim=0)
        )  # (N_eval, pred_steps, d_f)

        # Do not log during sanity check?
        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            # Note that spsk_ratio is scale-invariant, so do not have to rescale
            spread = torch.sqrt(torch.mean(spread_squared_tensor, dim=0))
            skill = torch.sqrt(torch.mean(ens_mse_tensor, dim=0))
            # Both (pred_steps, d_f)

            # Include finite sample correction
            spsk_ratios = np.sqrt(
                (self.ensemble_size + 1) / self.ensemble_size
            ) * (
                spread / skill
            )  # (pred_steps, d_f)
            log_dict = self.create_metric_log_dict(
                spsk_ratios, prefix, "spsk_ratio"
            )

            log_dict[f"{prefix}_mean_spsk_ratio"] = torch.mean(
                spsk_ratios
            )  # log mean
            wandb.log(log_dict)

    def on_validation_epoch_end(self):
        if not self.pretraining:
            self.log_spsk_ratio(self.val_metrics, "val")
        super().on_validation_epoch_end()

    def test_step(self, batch, batch_idx):
        init_states, target_states, forcing_features, sample_times = batch
        device = self.device
        init_states = init_states.to(device)
        target_states = target_states.to(device)
        forcing_features = forcing_features.to(device)
        sample_times = sample_times.to(device)

        self.prior.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.denoiser.eval()

        with torch.no_grad():
            pred_steps = target_states.shape[1]
            ensemble_size = self.ensemble_size
            predictions, pred_std = self.sample_prediction(
                init_states, forcing_features, target_states, sample_times, steps=pred_steps, ensemble_size=ensemble_size
            )
            pred_std = pred_std if pred_std is not None else self.per_var_std

            target = target_states
            ens_mean = torch.mean(predictions, dim=1)
            time_step_loss = torch.mean(
                self.loss(ens_mean, target, pred_std, mask=self.interior_mask_bool, grid_weights=self.grid_weights),
                dim=0,
            )
            mean_loss = torch.mean(time_step_loss)

        test_log_dict = {
            f"test_loss_unroll{step}": time_step_loss[step - 1]
            for step in self.val_log_leads
        }
        test_log_dict["test_mean_loss"] = mean_loss
        self.log_dict(test_log_dict, on_step=False, on_epoch=True, sync_dist=True)

        spread_squared_batch = metrics.spread_squared(
            predictions,
            target,
            pred_std,
            grid_weights=self.grid_weights,
            mask=self.interior_mask_bool,
            sum_vars=False,
        )
        self.test_metrics["spread_squared"].append(spread_squared_batch)

        for metric_name in ("mse", "mae"):
            metric_func = metrics.get_metric(metric_name)

            global_vals = metric_func(
                ens_mean,
                target,
                None,
                grid_weights=self.grid_weights,
                mask=self.interior_mask_bool,
                sum_vars=False,
            )
            global_key = f"ens_{metric_name}"
            if global_key not in self.test_metrics:
                self.test_metrics[global_key] = []
            self.test_metrics[global_key].append(global_vals)

            for region_name, region_mask_1d in self.region_masks_1d.items():
                region_vals = metric_func(
                    ens_mean,
                    target,
                    None,
                    grid_weights=self.grid_weights,
                    mask=region_mask_1d,
                    sum_vars=False,
                )
                region_key = f"ens_{metric_name}_{region_name}"
                if region_key not in self.test_metrics:
                    self.test_metrics[region_key] = []
                self.test_metrics[region_key].append(region_vals)

        crps_batch = metrics.crps_ens(
            predictions,
            target,
            pred_std,
            grid_weights=self.grid_weights,
            mask=self.interior_mask_bool,
            sum_vars=False,
        )
        if "crps_ens" not in self.test_metrics:
            self.test_metrics["crps_ens"] = []
        self.test_metrics["crps_ens"].append(crps_batch)

        for region_name, region_mask_1d in self.region_masks_1d.items():
            region_crps = metrics.crps_ens(
                predictions,
                target,
                pred_std,
                grid_weights=self.grid_weights,
                mask=region_mask_1d,
                sum_vars=False,
            )
            region_key = f"crps_ens_{region_name}"
            if region_key not in self.test_metrics:
                self.test_metrics[region_key] = []
            self.test_metrics[region_key].append(region_crps)

        if self.output_std:
            mean_pred_std = torch.mean(
                pred_std[..., self.interior_mask_bool, :], dim=-2
            )  # (B, pred_steps, d_f)
            self.test_metrics["output_std"].append(mean_pred_std)

        spatial_loss = self.loss(ens_mean, target, pred_std, average_grid=False)
        log_spatial_losses = spatial_loss[:, self.val_log_leads - 1]
        self.spatial_loss_maps.append(log_spatial_losses)
        
        # Plot example predictions (on rank 0 only)
        if (
            self.trainer.is_global_zero
            and self.plotted_examples < self.n_example_pred
        ):
            # Need to plot more example predictions
            n_additional_examples = min(
                predictions.shape[0], self.n_example_pred - self.plotted_examples
            )

            self.plot_examples(
                batch, n_additional_examples, prediction=predictions
            )

        return mean_loss

    def on_test_epoch_end(self):
        """
        Compute test metrics and make plots at the end of test epoch.
        Will gather stored tensors and perform plotting and logging on rank 0.
        """
        super().on_test_epoch_end()
        self.log_spsk_ratio(self.test_metrics, "test")
