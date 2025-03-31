# Standard library
import os

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# First-party
import constants, metrics, utils, vis


class ARModel(pl.LightningModule):
    """
    Generic auto-regressive weather model.
    Abstract class that can be extended.
    """

    # pylint: disable=arguments-differ
    # Disable to override args/kwargs from superclass

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.lr = args.lr

        # Load static features for grid/data
        static_data_dict = utils.load_static_data(args.dataset)
        for static_data_name, static_data_tensor in static_data_dict.items():
            self.register_buffer(
                static_data_name, static_data_tensor, persistent=False
            )

        # Double grid output dim. to also output std.-dev.
        self.output_std = bool(args.output_std)
        if self.output_std:
            self.grid_output_dim = (
                2 * constants.GRID_STATE_DIM
            )  # Pred. dim. in grid cell
        else:
            self.grid_output_dim = (
                constants.GRID_STATE_DIM
            )  # Pred. dim. in grid cell

            # Store constant per-variable std.-dev. weighting
            # Note that this is the inverse of the multiplicative weighting
            # in wMSE/wMAE
            self.register_buffer(
                "per_var_std",
                self.step_diff_std / torch.sqrt(self.param_weights),
                persistent=False,
            )

        # grid_dim from data + static
        (
            self.num_grid_nodes,
            grid_static_dim,
        ) = self.grid_static_features.shape  # 63784 = 268x238
        self.grid_dim = (
            2 * constants.GRID_STATE_DIM
            + grid_static_dim
            + constants.GRID_FORCING_DIM
        )

        # Instantiate loss function
        self.loss = metrics.get_metric(args.loss)

        self.boundary_forcing = self.border_mask is not None
        if self.boundary_forcing:
            # Pre-compute interior mask for use in loss function
            interior_mask_tensor = 1.0 - self.border_mask
        else:
            # Still set as constant 1 (no border), for plotting
            interior_mask_tensor = torch.ones(self.num_grid_nodes, 1)

        self.register_buffer(
            "interior_mask", interior_mask_tensor, persistent=False
        )  # (num_grid_nodes, 1), 1 for non-border

        # For validation and testing
        self.step_length = args.step_length  # Number of hours per pred. step
        self.val_metrics = {
            # "mse": [],
        }
        self.test_metrics = {
            # "mse": [],
            # "mae": [],
        }
        if self.output_std:
            self.test_metrics["output_std"] = []  # Treat as metric

        # Do not try to log at lead times not forecasted
        self.val_log_leads = constants.VAL_STEP_LOG_ERRORS[
            constants.VAL_STEP_LOG_ERRORS <= args.eval_leads
        ]
        self.val_plot_vars = {
            var_i: ts[ts <= args.eval_leads]
            for var_i, ts in constants.VAL_PLOT_VARS.items()
        }
        self.eval_leads = args.eval_leads

        # For making restoring of optimizer state optional (slight hack)
        self.opt_state = None

        # For example plotting
        self.n_example_pred = args.n_example_pred
        self.plotted_examples = 0

        # For storing spatial loss maps during evaluation
        self.spatial_loss_maps = []

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        if self.opt_state:
            optimizer.load_state_dict(self.opt_state)

        if self.trainer is not None:
            steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
            total_steps = steps_per_epoch * self.trainer.max_epochs
        else:
            steps_per_epoch = 1000
            total_steps = steps_per_epoch * 10

        warmup_steps = int(0.05 * total_steps)

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        lr_scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "name": "lr_scheduler"
        }

        return [optimizer], [lr_scheduler_config]
    
    '''
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, betas=(0.9, 0.95)
        )
        if self.opt_state:
            opt.load_state_dict(self.opt_state)

        return opt
    '''

    @property
    def interior_mask_bool(self):
        """
        Get the interior mask as a boolean (N,) mask.
        If not using boundary forcing, returns None.
        """
        if not self.boundary_forcing:
            return None

        return self.interior_mask[:, 0].to(torch.bool)

    @staticmethod
    def expand_to_batch(x, batch_size):
        """
        Expand tensor with initial batch dimension
        """
        return x.unsqueeze(0).expand(batch_size, -1, -1)

    def predict_step(self, prev_state, prev_prev_state, forcing):
        """
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
        forcing: (B, num_grid_nodes, forcing_dim)
        """
        raise NotImplementedError("No prediction step implemented")

    def all_gather_cat(self, tensor_to_gather):
        """
        Gather tensors across all ranks, and concatenate across dim. 0
        (instead of stacking in new dim. 0)

        tensor_to_gather: (d1, d2, ...), distributed over K ranks

        returns: (K*d1, d2, ...)
        """
        return self.all_gather(tensor_to_gather).flatten(0, 1)

    # newer lightning versions requires batch_idx argument, even if unused
    # pylint: disable-next=unused-argument
    def validation_step(self, batch, batch_idx):
        """
        Run validation on single batch
        """
        prediction, target, pred_std = self.common_step(batch)

        time_step_loss = torch.mean(
            self.loss(
                prediction,
                target,
                pred_std,
                mask=self.interior_mask_bool,
                grid_weights=self.grid_weights,
            ),
            dim=0,
        )  # (time_steps-1)
        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        val_log_dict = {
            f"val_loss_unroll{step}": time_step_loss[step - 1]
            for step in self.val_log_leads
        }
        val_log_dict["val_mean_loss"] = mean_loss
        self.log_dict(
            val_log_dict, on_step=False, on_epoch=True, sync_dist=True
        )

        # Store MSEs
        entry_mses = metrics.mse(
            prediction,
            target,
            pred_std,
            grid_weights=self.grid_weights,
            mask=self.interior_mask_bool,
            sum_vars=False,
        )  # (B, pred_steps, d_f)
        self.val_metrics["mse"].append(entry_mses)

    def on_validation_epoch_end(self):
        """
        Compute val metrics at the end of val epoch
        """
        # Create error maps for all test metrics
        self.aggregate_and_plot_metrics(self.val_metrics, prefix="val")

        # Clear lists with validation metrics values
        for metric_list in self.val_metrics.values():
            metric_list.clear()

    # pylint: disable-next=unused-argument
    def test_step(self, batch, batch_idx):
        """
        Run test on single batch
        """
        prediction, target, pred_std = self.common_step(batch)
        # prediction: (B, pred_steps, num_grid_nodes, d_f)
        # pred_std: (B, pred_steps, num_grid_nodes, d_f) or (d_f,)

        time_step_loss = torch.mean(
            self.loss(
                prediction,
                target,
                pred_std,
                mask=self.interior_mask_bool,
                grid_weights=self.grid_weights,
            ),
            dim=0,
        )  # (time_steps-1,)
        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        test_log_dict = {
            f"test_loss_unroll{step}": time_step_loss[step - 1]
            for step in self.val_log_leads
        }
        test_log_dict["test_mean_loss"] = mean_loss

        self.log_dict(
            test_log_dict, on_step=False, on_epoch=True, sync_dist=True
        )

        # Compute all evaluation metrics for error maps
        # Note: explicitly list metrics here, as test_metrics can contain
        # additional ones, computed differently, but that should be aggregated
        # on_test_epoch_end
        for metric_name in ("mse", "mae"):
            metric_func = metrics.get_metric(metric_name)
            batch_metric_vals = metric_func(
                prediction,
                target,
                pred_std,
                grid_weights=self.grid_weights,
                mask=self.interior_mask_bool,
                sum_vars=False,
            )  # (B, pred_steps, d_f)
            self.test_metrics[metric_name].append(batch_metric_vals)

        if self.output_std:
            # Store output std. per variable, spatially averaged
            mean_pred_std = torch.mean(
                pred_std[..., self.interior_mask_bool, :], dim=-2
            )  # (B, pred_steps, d_f)
            self.test_metrics["output_std"].append(mean_pred_std)

        # Save per-sample spatial loss for specific times
        # Note: Do not include grid weighting for this plot
        spatial_loss = self.loss(
            prediction, target, pred_std, average_grid=False
        )  # (B, pred_steps, num_grid_nodes)
        log_spatial_losses = spatial_loss[:, self.val_log_leads - 1]
        self.spatial_loss_maps.append(log_spatial_losses)
        # (B, N_log, num_grid_nodes)

        # Plot example predictions (on rank 0 only)
        if (
            self.trainer.is_global_zero
            and self.plotted_examples < self.n_example_pred
        ):
            # Need to plot more example predictions
            n_additional_examples = min(
                prediction.shape[0], self.n_example_pred - self.plotted_examples
            )

            self.plot_examples(
                batch, n_additional_examples, prediction=prediction
            )

    def plot_examples(self, batch, n_examples, prediction=None):
        """
        Plot the first n_examples forecasts from batch

        batch: batch with data to plot corresponding forecasts for
        n_examples: number of forecasts to plot
        prediction: (B, pred_steps, num_grid_nodes, d_f), existing prediction.
            Generate if None.
        """
        if prediction is None:
            prediction, target = self.common_step(batch)

        target = batch[1]

        # Rescale to original data scale
        prediction_rescaled = prediction * self.data_std + self.data_mean
        target_rescaled = target * self.data_std + self.data_mean

        # Iterate over the examples
        for pred_slice, target_slice in zip(
            prediction_rescaled[:n_examples], target_rescaled[:n_examples]
        ):
            # Each slice is (pred_steps, num_grid_nodes, d_f)
            self.plotted_examples += 1  # Increment already here

            var_vmin = (
                torch.minimum(
                    pred_slice.flatten(0, 1).min(dim=0)[0],
                    target_slice.flatten(0, 1).min(dim=0)[0],
                )
                .cpu()
                .numpy()
            )  # (d_f,)
            var_vmax = (
                torch.maximum(
                    pred_slice.flatten(0, 1).max(dim=0)[0],
                    target_slice.flatten(0, 1).max(dim=0)[0],
                )
                .cpu()
                .numpy()
            )  # (d_f,)
            var_vranges = list(zip(var_vmin, var_vmax))

            # Iterate over prediction horizon time steps
            for t_i, (pred_t, target_t) in enumerate(
                zip(pred_slice, target_slice), start=1
            ):
                # Create one figure per variable at this time step
                var_names = [
                    constants.PARAM_NAMES_SHORT[var_i]
                    for var_i in constants.EVAL_PLOT_VARS
                ]
                var_figs = [
                    vis.plot_prediction(
                        pred_t[:, var_i],
                        target_t[:, var_i],
                        self.interior_mask[:, 0],
                        title=(
                            f"{var_name} "
                            f"({constants.PARAM_UNITS[var_i]}), "
                            f"t={t_i} ({self.step_length*t_i} h)"
                        ),
                        vrange=var_vranges[var_i],
                    )
                    for var_i, var_name in zip(
                        constants.EVAL_PLOT_VARS, var_names
                    )
                ]

                example_i = self.plotted_examples
                wandb.log(
                    {
                        f"{var_name}_example_{example_i}": wandb.Image(fig)
                        for var_name, fig in zip(var_names, var_figs)
                    }
                )
                plt.close(
                    "all"
                )  # Close all figs for this time step, saves memory

            # Save pred and target as .pt files
            torch.save(
                pred_slice.cpu(),
                os.path.join(
                    wandb.run.dir, f"example_pred_{self.plotted_examples}.pt"
                ),
            )
            torch.save(
                target_slice.cpu(),
                os.path.join(
                    wandb.run.dir, f"example_target_{self.plotted_examples}.pt"
                ),
            )

    def create_metric_log_dict(self, metric_tensor, prefix, metric_name):
        """
        Put together a dict with everything to log for one metric.
        Also saves plots as pdf and csv if using test prefix.

        metric_tensor: (pred_steps, d_f), metric values per time and variable
        prefix: string, prefix to use for logging
        metric_name: string, name of the metric

        Return:
        log_dict: dict with everything to log for given metric
        """
        log_dict = {}
        metric_fig = vis.plot_error_map(metric_tensor)
        full_log_name = f"{prefix}_{metric_name}"
        log_dict[full_log_name] = wandb.Image(metric_fig)

        if prefix == "test":
            # Save pdf
            metric_fig.savefig(
                os.path.join(wandb.run.dir, f"{full_log_name}.pdf")
            )
            # Save errors also as csv
            np.savetxt(
                os.path.join(wandb.run.dir, f"{full_log_name}.csv"),
                metric_tensor.cpu().numpy(),
                delimiter=",",
            )

        # Check if metrics are watched, log exact values for specific vars
        if full_log_name in constants.METRICS_WATCH:
            for var_i, timesteps in constants.VAR_LEADS_METRICS_WATCH.items():
                var = constants.PARAM_NAMES_SHORT[var_i]
                log_dict.update(
                    {
                        f"{full_log_name}_{var}_step_{step}": metric_tensor[
                            step - 1, var_i
                        ]  # 1-indexed in constants
                        for step in timesteps
                        if step <= self.eval_leads
                    }
                )

        return log_dict

    def aggregate_and_plot_metrics(self, metrics_dict, prefix):
        """
        Aggregate and create error map plots for all metrics in metrics_dict

        metrics_dict: dictionary with metric_names and list of tensors
            with step-evals.
        prefix: string, prefix to use for logging
        """
        log_dict = {}
        for metric_name, metric_val_list in metrics_dict.items():
            metric_tensor = self.all_gather_cat(
                torch.cat(metric_val_list, dim=0)
            )  # (N_eval, pred_steps, d_f)

            if self.trainer.is_global_zero:
                metric_tensor_averaged = torch.mean(metric_tensor, dim=0)
                # (pred_steps, d_f)

                # Take square root after averaging to change squared metrics
                if "mse" in metric_name:
                    metric_tensor_averaged = torch.sqrt(metric_tensor_averaged)
                    metric_name = metric_name.replace("mse", "rmse")
                elif metric_name.endswith("_squared"):
                    metric_tensor_averaged = torch.sqrt(metric_tensor_averaged)
                    metric_name = metric_name[: -len("_squared")]

                # Note: we here assume rescaling for all metrics is linear
                metric_rescaled = metric_tensor_averaged * self.data_std
                # (pred_steps, d_f)
                log_dict.update(
                    self.create_metric_log_dict(
                        metric_rescaled, prefix, metric_name
                    )
                )

        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            wandb.log(log_dict)  # Log all
            plt.close("all")  # Close all figs

    def on_test_epoch_end(self):
        self.aggregate_and_plot_metrics(self.test_metrics, prefix="test")

    def on_load_checkpoint(self, checkpoint):
        loaded_state_dict = checkpoint["state_dict"]
    
        if "g2m_gnn.grid_mlp.0.weight" in loaded_state_dict:
            replace_keys = list(
                filter(
                    lambda key: key.startswith("g2m_gnn.grid_mlp"),
                    loaded_state_dict.keys(),
                )
            )
            for old_key in replace_keys:
                new_key = old_key.replace("g2m_gnn.grid_mlp", "encoding_grid_mlp")
                loaded_state_dict[new_key] = loaded_state_dict[old_key]
                del loaded_state_dict[old_key]
        
        if self.load_only_encoder_decoder:
            keys_to_remove = []
            for key in list(loaded_state_dict.keys()):
                if key.startswith("denoiser.") or key.startswith("condition_encoder.") or key.startswith("time_embedders.") or key.startswith("leadtime_embedders."):
                    keys_to_remove.append(key)
    
            for k in keys_to_remove:
                del loaded_state_dict[k]
    
            if len(keys_to_remove) > 0:
                print(f"[on_load_checkpoint] Removed {len(keys_to_remove)} keys from denoiser/condition_encoder.")
             