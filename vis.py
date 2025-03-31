# Third-party
import cartopy
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

# First-party
import constants, utils


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_error_map(errors, title=None):
    """
    Plot a heatmap of errors of different variables at different
    predictions horizons
    errors: (pred_steps, d_f)
    """
    errors_np = errors.T.cpu().numpy()  # (d_f, pred_steps)
    d_f, pred_steps = errors_np.shape

    # Normalize all errors to [0,1] for color map
    max_errors = errors_np.max(axis=1)  # d_f
    errors_norm = errors_np / np.expand_dims(max_errors, axis=1)

    fig, ax = plt.subplots(figsize=(25, 25))

    ax.imshow(
        errors_norm,
        cmap="OrRd",
        vmin=0,
        vmax=1.0,
        interpolation="none",
        aspect="auto",
        alpha=0.8,
    )

    # ax and labels
    for (j, i), error in np.ndenumerate(errors_np):
        # Numbers > 9999 will be too large to fit
        if 0.001 < error < 9999:
            formatted_error = f"{error:.3f}"
        else:
            formatted_error = f"{error:.2E}"

        ax.text(
            i,
            j,
            formatted_error,
            ha="center",
            va="center",
            usetex=False,
            fontsize=8,
        )

    # Ticks and labels
    label_size = 15
    ax.set_xticks(np.arange(pred_steps))
    pred_hor_i = np.arange(pred_steps) + 1  # Prediction horiz. in index
    pred_hor_h = constants.TIME_STEP_LENGTH * pred_hor_i  # Lead time in hours
    ax.set_xticklabels(pred_hor_h, size=label_size)
    ax.set_xlabel("Lead time (h)", size=label_size)

    ax.set_yticks(np.arange(d_f))
    y_ticklabels = [
        f"{name} ({unit})"
        for name, unit in zip(
            constants.PARAM_NAMES_SHORT, constants.PARAM_UNITS
        )
    ]
    ax.set_yticklabels(y_ticklabels, rotation=0, size=label_size)

    if title:
        ax.set_title(title, size=15)

    return fig


def plot_on_axis(
    ax, data, obs_mask=None, vmin=None, vmax=None, ax_title=None, cmap="plasma"
):
    """
    Plot weather state on given axis
    """
    # Set up masking of border region
    if obs_mask is None:
        pixel_alpha = 1
    else:
        mask_reshaped = obs_mask.reshape(*constants.GRID_SHAPE)
        pixel_alpha = (
            mask_reshaped.clamp(0.7, 1).cpu().numpy()
        )  # Faded border region

    ax.set_global()
    ax.coastlines()  # Add coastline outlines
    data_grid = data.reshape(*constants.GRID_SHAPE).cpu().numpy().T
    im = ax.imshow(
        data_grid,
        origin="lower",
        extent=constants.GRID_LIMITS,
        transform=cartopy.crs.PlateCarree(),
        alpha=pixel_alpha,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    if ax_title:
        ax.set_title(ax_title, size=15)
    return im


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_prediction(pred, target, obs_mask=None, title=None, vrange=None):
    """
    Plot example prediction and grond truth.
    Each has shape (N_grid,)
    """
    # Get common scale for values
    if vrange is None:
        vmin = min(vals.min().cpu().item() for vals in (pred, target))
        vmax = max(vals.max().cpu().item() for vals in (pred, target))
    else:
        vmin, vmax = vrange

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 7), subplot_kw={"projection": constants.MAP_PROJ}
    )

    # Plot pred and target
    for ax, data in zip(axes, (target, pred)):
        im = plot_on_axis(ax, data, obs_mask, vmin, vmax)

    # Ticks and labels
    axes[0].set_title("Ground Truth", size=15)
    axes[1].set_title("Prediction", size=15)
    cbar = fig.colorbar(im, aspect=30)
    cbar.ax.tick_params(labelsize=10)

    if title:
        fig.suptitle(title, size=20)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_ensemble_prediction(
    samples, target, ens_mean, ens_std, obs_mask=None, title=None, vrange=None
):
    """
    Plot example predictions, ground truth, mean and std.-dev.
    from ensemble forecast with a brighter style & rectangular projection.

    Args:
        samples (Tensor): (S, N_grid,) shape, ensemble samples
        target (Tensor):  (N_grid,) shape, ground truth
        ens_mean (Tensor): (N_grid,) shape, ensemble mean
        ens_std (Tensor):  (N_grid,) shape, ensemble std. dev.
        obs_mask (Tensor, optional): (N_grid,) shape, mask for observed regions
        title (str, optional): plot title
        vrange (tuple, optional): (vmin, vmax) for main data (not std dev)
    """
    # ----- [1] Color scale 설정 -----
    if vrange is None:
        vmin = min(vals.min().cpu().item() for vals in (samples, target))
        vmax = max(vals.max().cpu().item() for vals in (samples, target))
    else:
        vmin, vmax = vrange

    # ----- [2] Figure, Subplots 생성 -----
    #   - 직사각형 투영: PlateCarree()를 사용하면 지도 왜곡이 덜합니다.
    #   - figsize를 키우고, constrained_layout=True 로 레이아웃 자동 조정
    fig, axes = plt.subplots(
        3, 3,
        figsize=(16, 14),
        subplot_kw={"projection": ccrs.PlateCarree()},  # 직사각형 투영
        constrained_layout=True
    )
    axes = axes.flatten()

    # ----- [2-1] 축(axes) 기본 설정 (coastlines, gridlines 등) -----
    #   - 전 지구 영역 설정(원하면 잘라내기 가능)
    for ax in axes:
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax.coastlines(resolution='110m', color='black', linewidth=1)
        # 격자선(gridlines) 표시 (draw_labels=True 하면 경위도 라벨이 표시됨)
        gl = ax.gridlines(draw_labels=False, color='gray', alpha=0.4, linestyle='--')
        # gl.top_labels = gl.right_labels = False  # 필요시 라벨 제거

    # ----- [3] Ground Truth (좌상단) -----
    #   - 더 밝은 컬러맵 예: "turbo", "rainbow", "Spectral" 등
    gt_im = plot_on_axis(
        axes[0],
        target,
        obs_mask=obs_mask,
        vmin=vmin,
        vmax=vmax,
        cmap="turbo",  # 밝은 톤의 컬러맵
        ax_title="Ground Truth",
    )

    # ----- [4] Ensemble Mean (중상단) -----
    mean_im = plot_on_axis(
        axes[1],
        ens_mean - target,
        obs_mask=obs_mask,
        vmin=(ens_mean - target).min().cpu().item(),
        vmax=(ens_mean - target).max().cpu().item(),
        cmap="turbo",
        ax_title="(Ens. Mean) - (Ground Truth)",
    )

    # ----- [5] Ensemble Std (우상단) -----
    #   - 표준편차는 다른 컬러맵(예: 'YlOrRd', 'hot') 사용
    std_im = plot_on_axis(
        axes[2],
        ens_std,
        obs_mask=obs_mask,
        cmap="YlOrRd",
        ax_title="Ens. Std."
    )

    # ----- [6] Ensemble Members (아랫줄) -----
    #   - 최대 6개까지만 표시
    max_members_to_plot = 6
    for i, ax in enumerate(axes[3:3+max_members_to_plot]):
        if i < samples.shape[0]:
            plot_on_axis(
                ax,
                samples[i],
                obs_mask=obs_mask,
                vmin=vmin,
                vmax=vmax,
                cmap="turbo",
                ax_title=f"Member {i+1}",
            )
        else:
            ax.axis("off")

    # 나머지 unused subplot 숨기기
    for ax in axes[3 + samples.shape[0]:]:
        ax.axis("off")

    # ----- [7] Colorbar 배치 -----
    #   - 메인 colorbar(ground truth, ensemble mean, etc.)
    cbar_ax = fig.add_axes([0.25, 0.3, 0.5, 0.02])  # (left, bottom, width, height)
    cb = fig.colorbar(gt_im, cax=cbar_ax, orientation='horizontal')
    cb.ax.tick_params(labelsize=10)

    #   - 표준편차(std) 전용 colorbar를 아래쪽에 가로 배치
    stdbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.02])
    std_cb = fig.colorbar(std_im, cax=stdbar_ax, orientation='horizontal')
    std_cb.ax.tick_params(labelsize=10)
    
    cbar_ax2 = fig.add_axes([0.25, 0.35, 0.5, 0.02])  # (left, bottom, width, height)
    cb2 = fig.colorbar(mean_im, cax=cbar_ax2, orientation='horizontal')
    cb2.ax.tick_params(labelsize=10)

    # ----- [8] Title 설정 -----
    if title:
        fig.suptitle(title, fontsize=18, fontweight="bold")

    return fig

'''
def plot_ensemble_prediction(
    samples, target, ens_mean, ens_std, obs_mask=None, title=None, vrange=None
):
    """
    Plot example predictions, ground truth, mean and std.-dev.
    from ensemble forecast

    samples: (S, N_grid,)
    target: (N_grid,)
    ens_mean: (N_grid,)
    ens_std: (N_grid,)
    obs_mask: (N_grid,)
    (optional) title: title of plot
    (optional) vrange: tuple of length with common min and max of values
        (not for std.)
    """
    # Get common scale for values
    if vrange is None:
        vmin = min(vals.min().cpu().item() for vals in (samples, target))
        vmax = max(vals.max().cpu().item() for vals in (samples, target))
    else:
        vmin, vmax = vrange

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(15, 15),
        subplot_kw={"projection": constants.MAP_PROJ},
    )
    axes = axes.flatten()

    # Plot target, ensemble mean and std.
    gt_im = plot_on_axis(
        axes[0],
        target,
        obs_mask=obs_mask,
        vmin=vmin,
        vmax=vmax,
        ax_title="Ground Truth",
    )
    plot_on_axis(
        axes[1],
        ens_mean,
        obs_mask=obs_mask,
        vmin=vmin,
        vmax=vmax,
        ax_title="Ens. Mean",
    )
    std_im = plot_on_axis(
        axes[2], ens_std, obs_mask=obs_mask, ax_title="Ens. Std."
    )  # Own vrange

    # Plot samples
    for member_i, (ax, member) in enumerate(
        zip(axes[3:], samples[:6]), start=1
    ):
        plot_on_axis(
            ax,
            member,
            obs_mask=obs_mask,
            vmin=vmin,
            vmax=vmax,
            ax_title=f"Member {member_i}",
        )

    # Turn off unused axes
    for ax in axes[(3 + samples.shape[0]) :]:
        ax.axis("off")

    # Add colorbars
    values_cbar = fig.colorbar(
        gt_im, ax=axes[:2], aspect=60, location="bottom", shrink=0.9
    )
    values_cbar.ax.tick_params(labelsize=10)
    std_cbar = fig.colorbar(std_im, aspect=30, location="bottom", shrink=0.9)
    std_cbar.ax.tick_params(labelsize=10)

    if title:
        fig.suptitle(title, size=20)

    return fig
'''


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_spatial_error(error, obs_mask=None, title=None, vrange=None):
    """
    Plot errors over spatial map
    Error and obs_mask has shape (N_grid,)
    """
    # Get common scale for values
    if vrange is None:
        vmin = error.min().cpu().item()
        vmax = error.max().cpu().item()
    else:
        vmin, vmax = vrange

    fig, ax = plt.subplots(
        figsize=(5, 4.8), subplot_kw={"projection": constants.MAP_PROJ}
    )

    im = plot_on_axis(ax, error, obs_mask, vmin, vmax, cmap="OrRd")

    # Ticks and labels
    cbar = fig.colorbar(im, aspect=30)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.get_offset_text().set_fontsize(10)
    cbar.formatter.set_powerlimits((-3, 3))

    if title:
        fig.suptitle(title, size=10)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_latent_samples(prior_samples, vi_samples, title=None):
    """
    Plot samples of latent variable drawn from prior and
    variational distribution

    prior_samples: (samples, N_mesh, d_latent)
    vi_samples: (samples, N_mesh, d_latent)

    Returns:
    fig: the plot figure
    """
    num_samples, num_mesh_nodes, latent_dim = prior_samples.shape
    plot_dims = min(latent_dim, 3)  # Plot first 3 dimensions
    img_side_size = int(np.sqrt(num_mesh_nodes))

    # Check if number of nodes is a square
    if img_side_size**2 != num_mesh_nodes:
        # Number of mesh nodes is not a square number, can not directly plot
        # latent samples as images"
        # Fix this by not plotting all nodes (choose amount to work as image)
        num_mesh_subset = img_side_size**2
        prior_samples = prior_samples[:, :num_mesh_subset]
        vi_samples = vi_samples[:, :num_mesh_subset]

    # Get common scale for values
    vmin = min(
        vals[..., :plot_dims].min().cpu().item()
        for vals in (prior_samples, vi_samples)
    )
    vmax = max(
        vals[..., :plot_dims].max().cpu().item()
        for vals in (prior_samples, vi_samples)
    )

    # Create figure
    fig, axes = plt.subplots(num_samples, 2 * plot_dims, figsize=(20, 16))

    # Plot samples
    for row_i, (axes_row, prior_sample, vi_sample) in enumerate(
        zip(axes, prior_samples, vi_samples)
    ):

        for dim_i in range(plot_dims):
            prior_sample_reshaped = (
                prior_sample[:, dim_i]
                .reshape(img_side_size, img_side_size)
                .cpu()
                .to(torch.float32)
                .numpy()
            )
            vi_sample_reshaped = (
                vi_sample[:, dim_i]
                .reshape(img_side_size, img_side_size)
                .cpu()
                .to(torch.float32)
                .numpy()
            )
            # Plot every other as prior and vi
            prior_ax = axes_row[2 * dim_i]
            vi_ax = axes_row[2 * dim_i + 1]
            prior_ax.imshow(prior_sample_reshaped, vmin=vmin, vmax=vmax)
            vi_im = vi_ax.imshow(vi_sample_reshaped, vmin=vmin, vmax=vmax)

            if row_i == 0:
                # Add titles at top of columns
                prior_ax.set_title(f"d{dim_i} (prior)", size=15)
                vi_ax.set_title(f"d{dim_i} (vi)", size=15)

    # Remove ticks from all axes
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    # Add colorbar
    cbar = fig.colorbar(vi_im, ax=axes, aspect=60, location="bottom")
    cbar.ax.tick_params(labelsize=15)

    if title:
        fig.suptitle(title, size=20)

    return fig
