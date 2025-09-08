import numpy as np
import rasterio
import os
import logging
from matplotlib import pyplot as plt
from rasterio.plot import plotting_extent
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import SymLogNorm, Normalize, CenteredNorm
from typing import Optional

# Basic configuration with level set to INFO
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reconstruct_output_image(output, rows, cols, valid_cols):
    """Reconstruct the output data to original 2D image array

    Args:
        output (np.ndarray): flattened image array 
        rows (int): number of rows of the original 2D image array
        cols (int): number of columns of the original 2D image array
        valid_cols (np.ndarray): the valid column indices stored in zero-truncated preprocessing step

    Returns:
        output_hmax: reconstructed original image array (shape: (num_samples, rows, cols))
        output_mean: mean value of image array over the number of data samples (shape: (rows, cols))
        output_std: standard deviation of image array over the number of data samples( shape: (rows, cols))
    """
    output_size = output.shape[0]
    indices = np.flatnonzero(valid_cols)
    output_index = [int(i) for i in list(indices)]
    reconstructed_output = np.zeros((output_size, rows * cols))
    reconstructed_output[:,output_index] = output
    output_mean = reconstructed_output.mean(axis=0).reshape(rows, cols)
    output_std = reconstructed_output.std(axis=0).reshape(rows, cols)
    output_hmax = reconstructed_output.reshape((-1, rows, cols))
    # print("Reconstructed output image dimension: ", output_hmax.shape)
    return output_hmax, output_mean, output_std

def _viz_output_image(mean_gt, std_gt, mean_pred, std_pred, qoi, model_name: str, hill_path: Optional[str] = None) -> None:
    """Visualize the reconstructed output image

    Args:
        mean_gt (np.ndarray): mean value of the ground-truth image array
        std_gt (np.ndarray): standard deviation value of the ground-truth image array
        mean_pred (np.ndarray): mean value of the predicted image array
        std_pred (np.ndarray): standard deviation value of the predicted image array
        qoi (str): quantity of interest
        hill_path (str | None): filepath of the background image
    """
    font_size = 12
    tick_size = 10
    if qoi not in ["hmax", "vmax"]:
        raise ValueError(f"Invalid quantity of interest: {qoi}")
    label = "[m]" if qoi == "hmax" else "[m/s]"
    title = "Flow Height" if qoi == "hmax" else "Flow Velocity"
        
    # Masking
    mean_gt_mask = np.ma.masked_where(mean_gt <= 0.0, mean_gt, copy=True)
    std_gt_mask = np.ma.masked_where(mean_gt <= 0.0, std_gt, copy=True)
    mean_pred_mask = np.ma.masked_where(mean_pred <= 0.0, mean_pred, copy=True)
    std_pred_mask = np.ma.masked_where(mean_pred <= 0.0, std_pred, copy=True)
    
    if hill_path is not None:
        with rasterio.open(hill_path, 'r') as hill:
            hill_arr = hill.read(1)
            extent = plotting_extent(hill)
            hill_mask = np.ma.masked_where(hill_arr < -30000, hill_arr, copy=True)
    else:
        hill_arr = None
        extent = None
        hill_mask = None
    
    diff_mean = mean_gt - mean_pred
    diff_std = std_gt - std_pred
    diff_mean_mask = np.ma.masked_where(diff_mean == 0, diff_mean, copy=True)
    diff_std_mask = np.ma.masked_where(diff_std == 0, diff_std, copy=True)
    
    # Plotting
    fig, axs = plt.subplots(
        nrows=2, ncols=3,
        figsize=(8, 8),
    )
    axs = axs.flatten()
    fig.suptitle(
        f'2D {title} with {model_name} Model',
        fontsize=16,
        y=0.98
    )
    max_mean_value = np.max(np.stack([mean_gt, mean_pred]))
    max_std_value = np.max(np.stack([std_gt, std_pred]))
    max_diff_mean_value = np.max(np.abs(diff_mean))
    max_diff_std_value = np.max(np.abs(diff_std))
    image_params = [
        (mean_gt_mask, 'viridis', 0.0, max_mean_value, "Mean Ground-truth " + label),
        (mean_pred_mask, 'viridis', 0.0, max_mean_value, "Mean Prediction " + label),
        (diff_mean_mask, 'RdBu_r', -max_diff_mean_value, max_diff_mean_value, "Difference in mean " + label),
        (std_gt_mask, 'viridis', 0.0, max_std_value, "Std. dev. Ground-truth " + label),
        (std_pred_mask, 'viridis', 0.0, max_std_value, "Std. dev. Prediction " + label),
        (diff_std_mask, 'RdBu_r', -max_diff_std_value, max_diff_std_value, "Difference in std. dev. " + label),
    ]
    for i, (data, cmap, vmin, vmax, cbar_label) in enumerate(image_params):
        ax = axs[i]
        if hill_mask is not None:
            ax.imshow(hill_mask, cmap='Greys', extent=extent)
        im = ax.imshow(data, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax, zorder=1)
        cbar = fig.colorbar(im, ax=ax, location='top', orientation='horizontal', pad=0.05)
        cbar.set_label(cbar_label, fontsize=font_size)
        cbar.ax.tick_params(labelsize=tick_size)
        if i % 3 != 0:
            ax.set_yticklabels([])
        if i < 3:
            ax.set_xticklabels([])
        ax.tick_params(axis='x', labelsize=tick_size)
        ax.tick_params(axis='y', labelsize=tick_size)
        if hill_path is not None:
            ax.set_xticks(np.arange(1489000, 1493600, 1000))
            ax.set_yticks(np.arange(5201000, 5205001, 1000))
            if i % 3 == 0:
                ax.set_ylabel('Northing [x 10$^6$ m]', fontsize=font_size)
                ax.set_yticklabels([5.201, 5.202, 5.203, 5.204, 5.205])
            else:
                ax.set_yticklabels([])
            if i >= 3:
                ax.set_xlabel('Easting [x 10$^6$ m]', fontsize=font_size)
                ax.set_xticklabels([1.498, 1.490, 1.491, 1.492, 1.493])
            else:
                ax.set_xticklabels([])
            ax.tick_params(axis='both', which='both', labelsize=tick_size)
    fig.tight_layout()
    plt.show()
    
def _viz_diff_grid_points_binary(ground_truth: np.ndarray, prediction: np.ndarray, model_name: str):
    diff = ground_truth - prediction
    flag = np.where(diff > 0, 1, np.where(diff < 0, -1, 0))
    # Define colormap and normalization: negative (blue), zero (white), positive (red)
    base_cmap = plt.get_cmap('coolwarm')
    colors = base_cmap(np.linspace(0, 1, 3))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    font_size = 12
    fig, ax = plt.subplots()
    im = ax.imshow(flag, cmap=cmap, norm=norm)
    cbar = fig.colorbar(
                im,
                ax=ax,
                orientation='horizontal',  
                fraction=0.05,              # colorbar thickness (as fraction of original axes)
                pad=0.175,                  # space between plot and colorbar
                ticks=[-1, 0, 1]
            )
    cbar.set_label('Sign of ground-truth – prediction', fontsize=font_size)
    ax.set_title(f"2D Grid Map Difference for {model_name} Model",
                 pad=12)
    ax.set_xlabel('Column index', fontsize=font_size)
    ax.set_ylabel('Row index', fontsize=font_size)
    fig.tight_layout()
    plt.show()
    
def _viz_diff_grid_points_percentage(ground_truths: np.ndarray, predictions: np.ndarray, model_name: str):
    """
    Visualize the percentage error of the prediction with three cases:
      1) mean relative error
      2) best-case relative error
      3) worst-case relative error

    Args:
        ground_truths (np.ndarray): Ground-truth data
        predictions (np.ndarray): Prediction data
        model_name (str): Model name
    """
    assert ground_truths.shape == predictions.shape, \
        "Ground-truth and prediction must have the same shape"
    assert ground_truths.ndim == predictions.ndim == 3, \
        "Ground-truth and prediction must be 3D arrays with the shape of (n_samples, rows, cols)"
    eps = 1e-6
    # Compute per-sample, per-cell relative error (%)    
    diff = ground_truths - predictions
    denom = np.where(np.abs(ground_truths) > eps, ground_truths, np.nan) # avoid division by zero
    err_percent = diff / denom * 100.0      # under (+) / over (−)
    err_percent = np.ma.masked_invalid(err_percent)
    
    # find the run‐indices for "safest" (most neg cells) and "worst" (most pos cells)
    neg_counts = (err_percent < 0).sum(axis=(1,2))   # count over‐estimates per run
    pos_counts = (err_percent > 0).sum(axis=(1,2))   # count under‐estimates per run
    idx_best = int(np.argmax(neg_counts))
    idx_worst  = int(np.argmax(pos_counts))
    
    # aggregate across the sample dimension
    mean_map  = err_percent.mean(axis=0)    # average error
    best_map = err_percent[idx_best]
    worst_map  = err_percent[idx_worst]

    # set a common color‐scale
    all_vals = err_percent.compressed()
    vmax = np.nanmax(np.abs(all_vals))
    norm = SymLogNorm(
        linthresh=1.0, 
        linscale=1.0,
        vmin=-vmax, vmax=+vmax,
        base=10
    )
    cmap = plt.get_cmap('coolwarm')
    
    # plot
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.95, 0.05])
    
    # Create subplots for the main plots
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    
    titles = ['Mean Error',
              'Best-case Error',
              'Worst-case Error']
    maps   = [mean_map, best_map, worst_map]
    
    for i, (ax, data, title) in enumerate(zip(axes, maps, titles)):
        im = ax.imshow(data, cmap=cmap, norm=norm)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Column index', fontsize=10)
        ax.set_ylabel('Row index', fontsize=10) if i == 0 else ax.set_yticklabels([])

    # Add colorbar in the bottom row
    cbar_ax = fig.add_subplot(gs[1, :])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Error (%)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle(f"Mean / Best / Worst Percentage Error Maps for {model_name} Model", fontsize=14, y=1.02)
    plt.show()
    
def _viz_all_2d_maps_in_one_plot(ground_truths: np.ndarray, predictions: np.ndarray, qoi: str, model_name: str, hill_path: Optional[str] = None):
    """
    Visualize all 2D maps in one plot.

    Args:
        ground_truths (np.ndarray): Ground-truth data (shape: (n_samples, rows, cols))
        predictions (np.ndarray): Prediction data (shape: (n_samples, rows, cols))
        qoi (str): Quantity of interest
        data_type (str): Data type
        model_name (str): Model name
        hill_path (str | None): filepath of the background image
    """
    if qoi not in ["hmax", "vmax"]:
        raise ValueError(f"Invalid quantity of interest: {qoi}")
    if ground_truths.shape != predictions.shape:
        raise ValueError("Ground-truth and prediction must have the same shape")
    if ground_truths.ndim != 3 or predictions.ndim != 3:
        raise ValueError("Ground-truth and prediction must be 3D arrays with the shape of (n_samples, rows, cols)")
    gt = ground_truths.mean(axis=0)
    pred = predictions.mean(axis=0)
    
    label = "[m]" if qoi == "hmax" else "[m/s]"
    title = "Mean Flow Height" if qoi == "hmax" else "Mean Flow Velocity"
    font_size = 18
    tick_size = 12
    
    # Compute the difference between ground-truth and prediction
    diff = gt - pred
    diff_mask = np.ma.masked_where(diff == 0, diff, copy=True)
    ground_truth_mask = np.ma.masked_where(gt <= 0.0, gt, copy=True)
    prediction_mask = np.ma.masked_where(pred <= 0.0, pred, copy=True)
    if hill_path is not None:
        with rasterio.open(hill_path, 'r') as hill:
            hill_arr = hill.read(1)
            extent = plotting_extent(hill)
            hill_mask = np.ma.masked_where(hill_arr < -30000, hill_arr, copy=True)
    else:
        hill_arr = None
        extent = None
        hill_mask = None
    max_mean = np.nanmax([ground_truth_mask.max(), prediction_mask.max()])
    max_diff = np.nanmax(np.abs(diff_mask))
    
    # Compute error percentage maps
    eps = 1e-6
    diff = ground_truths - predictions
    denom = np.where(np.abs(ground_truths) > eps, ground_truths, np.nan) # avoid division by zero
    err_percent = diff / denom * 100.0      # under (+) / over (−)
    err_percent = np.ma.masked_invalid(err_percent)
    # find the run‐indices for "safest" (most neg cells) and "worst" (most pos cells)
    neg_counts = (err_percent < 0).sum(axis=(1,2))   # count over‐estimates per run
    pos_counts = (err_percent > 0).sum(axis=(1,2))   # count under‐estimates per run
    idx_best = int(np.argmax(neg_counts))
    idx_worst  = int(np.argmax(pos_counts))
    # aggregate across the sample dimension
    mean_err_map  = err_percent.mean(axis=0)    # average error
    best_err_map = err_percent[idx_best]
    worst_err_map  = err_percent[idx_worst]
    # set a common color‐scale
    all_vals = err_percent.compressed()
    vmax = np.nanmax(np.abs(all_vals))
    err_norm = SymLogNorm(
        linthresh=1.0, 
        linscale=1.0,
        vmin=-vmax, vmax=+vmax,
        base=10
    )
    
    # Plotting
    fig, axes = plt.subplots(
        nrows=1, ncols=6,
        figsize=(30, 6),
        dpi=300,
    )
    specs = [
        (f"Ground-truth {label}", ground_truth_mask, "viridis", Normalize(vmin=0, vmax=max_mean)),
        (f"Prediction {label}", prediction_mask, "viridis", Normalize(vmin=0, vmax=max_mean)),
        (f"Difference {label}", diff_mask, "RdBu_r", CenteredNorm(vcenter=0, halfrange=max_diff)),
        ("Mean Error [%]",  mean_err_map, "coolwarm", err_norm),
        ("Best-case Error [%]", best_err_map, "coolwarm", err_norm),
        ("Worst-case Error [%]",worst_err_map,"coolwarm", err_norm),
    ]

    for i, (cbar_label, data, cmap, norm) in enumerate(specs):
        ax = axes[i]
        if hill_mask is not None:
            ax.imshow(hill_mask, cmap='Greys', extent=extent)
            ax.set_xticks(np.arange(1489000, 1493600, 1000))
            ax.set_yticks(np.arange(5201000, 5205001, 1000))
            if i == 0:
                ax.set_ylabel('Northing [x 10$^6$ m]', fontsize=font_size)
                ax.set_yticklabels([5.201, 5.202, 5.203, 5.204, 5.205])
            else:
                ax.set_yticklabels([])
            ax.set_xlabel('Easting [x 10$^6$ m]', fontsize=font_size)
            ax.set_xticklabels([1.498, 1.490, 1.491, 1.492, 1.493])
        im = ax.imshow(data, cmap=cmap, norm=norm, extent=extent, zorder=1)
        cbar = fig.colorbar(im, ax=ax, location='top', orientation='horizontal', pad=0.08, fraction=0.05)
        cbar.set_label(cbar_label, fontsize=font_size)
        cbar.ax.tick_params(labelsize=tick_size)
        if i != 0:
            ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', labelsize=tick_size)
    fig.suptitle(f"2D {title} Comparison for {model_name}", fontsize=25, y=1.02)
    fig.subplots_adjust(top=0.9)
    fig.tight_layout()
    plt.show()
    
def viz_output_distribution(ground_truth: dict, qoi: str = "hmax"):
    label = "Flow Height [m]" if qoi == "hmax" else "Flow Velocity [m/s]"
    plt.figure()
    plt.hist(ground_truth.flatten(), bins=30, density=True, edgecolor='k')
    plt.title(f'Histogram of Maximum {label}')
    plt.xlabel(label)
    plt.ylabel('Density')
    plt.show()

def viz_prediction(ground_truth, predictions, model_name, problem_type: str = "high_dim_output", is_in_latent_space=False, threshold: float = 0.5):
    assert problem_type in ["high_dim_output", "high_dim_input"], "Problem type must be 'high_dim_output' or 'high_dim_input'"
    if ground_truth.ndim == 1:
        ground_truth = ground_truth.reshape(-1, 1)
    if predictions.ndim == 2 and ground_truth.shape[1] == 1:
        prediction_mean = predictions[:, 0]
        prediction_std = predictions[:, 1]
    elif predictions.ndim == 2:
        prediction_mean = predictions
        prediction_std = None
    elif predictions.ndim == 3:
        prediction_mean = predictions[:, :, 0]
        prediction_std = predictions[:, :, 1]
    else:
        raise ValueError(f"The dimension of predictions must be 2d or 3d np.ndarray, but got {predictions.ndim}.")
    title = f"Prediction v.s. Ground-truth for {model_name} Model"
    if is_in_latent_space:
        title += " in latent space"
    plt.figure()
    y_true = ground_truth.flatten()
    y_pred = prediction_mean.flatten()
    if not is_in_latent_space and problem_type == "high_dim_output":
        y_true = np.where(y_true < threshold, 0, y_true)
        y_pred = np.where(y_pred < threshold, 0, y_pred)
    plt.plot([np.min(y_true),np.max(y_true)], [np.min(y_true),np.max(y_true)], color='black')
    if prediction_std is not None:
        pred_std = prediction_std.flatten()
        plt.errorbar(y_true, y_pred, 
                     yerr=pred_std, 
                     fmt='o', 
                     markersize=6,
                     markeredgewidth=1.0,
                     markerfacecolor='None',
                     ecolor='lightsteelblue',
                     elinewidth=0.5,
                     capsize=2,
                     alpha=0.8,  
                     label='emulator prediction ± std',
                     zorder=1)
    else:
        plt.errorbar(y_true, y_pred, 
                     fmt='o', 
                     markersize=6,
                     markeredgewidth=1.0,
                     markerfacecolor='None',
                     ecolor='lightsteelblue',
                     elinewidth=0.5,
                     capsize=2, 
                     alpha=0.8,  
                     label='emulator prediction', 
                     zorder=1)    
    plt.xlabel('Actual y')
    plt.ylabel('Predicted y')
    plt.xlim(np.min(y_true),np.max(y_true))
    plt.ylim(np.min(y_true),np.max(y_true))
    plt.legend()
    plt.tight_layout()
    plt.title(title)
    plt.show()
    
def viz_residuals(
    ground_truth, predictions, model_name,
    is_in_latent_space=False, threshold: float = 0.5
):
    if ground_truth.ndim == 1:
        ground_truth = ground_truth.reshape(-1, 1)
    if predictions.ndim == 2 and ground_truth.shape[1] == 1:
        prediction_mean = predictions[:, 0]
        prediction_std = predictions[:, 1]
    elif predictions.ndim == 2:
        prediction_mean = predictions
        prediction_std = None
    elif predictions.ndim == 3:
        prediction_mean = predictions[:, :, 0]
        prediction_std = predictions[:, :, 1]
    else:
        raise ValueError(f"The dimension of predictions must be 2d or 3d np.ndarray, but got {predictions.ndim}.")
    title = f"Prediction v.s. Ground-truth for {model_name} Model"
    if is_in_latent_space:
        title += " in latent space"

    y_true = ground_truth.flatten()
    y_pred = prediction_mean.flatten()
    if not is_in_latent_space:
        y_true = np.where(y_true < threshold, 0, y_true)
        y_pred = np.where(y_pred < threshold, 0, y_pred)
    residuals = y_pred - y_true
    plt.figure()
    ax = plt.gca()
    ax.axhline(0, color='black', zorder=1)
    if prediction_std is not None:
        pred_std = prediction_std.flatten()
        ax.errorbar(
            y_true, residuals, yerr=pred_std,
            fmt='o',
            markersize=6,
            markeredgewidth=1.0,
            markerfacecolor='none',
            ecolor='lightsteelblue',
            elinewidth=0.5,
            capsize=2,
            alpha=0.8,
            label='residual ± std'
        )
    else:
        ax.plot(
            y_true, residuals, 'o',
            markersize=6,
            markeredgewidth=1.0,
            markerfacecolor='none',
            alpha=0.8,
            label='residual'
        )
    ax.set_xlabel('Actual $y$', fontsize=12)
    ax.set_ylabel('Residual $y_{pred}-y_{true}$', fontsize=12)
    ax.set_title(f'Residual Plot for {model_name} Model', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()
    
def viz_output_image(ground_truths: dict, prediction_results: dict, output_img_params: dict, model_name: str, qoi: str = "hmax", threshold: float = 0.5):
    valid_cols = output_img_params["filtered_columns"]
    rows = output_img_params["output_img_rows"]
    cols = output_img_params["output_img_cols"]
    bg_img_path = output_img_params["background_img_path"]
    predictions_mean = prediction_results["prediction_mean"]
    predictions_mean_original = prediction_results.get("prediction_mean_original", predictions_mean)
    # Masking
    ground_truths_masked = np.where(ground_truths["original"] < threshold, 0, ground_truths["original"])
    predictions_mean_masked = np.where(predictions_mean_original < threshold, 0, predictions_mean_original)
    # Reconstruct to image visualization
    gts, gt_mean, gt_std = reconstruct_output_image(ground_truths_masked, rows, cols, valid_cols)
    preds, pred_mean, pred_std = reconstruct_output_image(predictions_mean_masked, rows, cols, valid_cols)
    _viz_output_image(gt_mean, gt_std, pred_mean, pred_std, qoi, model_name, bg_img_path)
    return gts, preds, gt_mean, pred_mean, gt_std, pred_std
    
def viz_diff_grid_points(ground_truths: dict, prediction_results: dict, output_img_params: dict, model_name: str, mode: str, threshold: float = 0.5):
    """
    Visualize the difference between ground-truth and prediction in the grid points.

    Args:
        ground_truths (dict): Ground-truth data
        prediction_results (dict): Prediction results
        output_img_params (dict): Output image parameters
        model_name (str): Model name
        mode (str): Mode of visualization plot ["binary", "percentage"]
    """
    valid_cols = output_img_params["filtered_columns"]
    rows = output_img_params["output_img_rows"]
    cols = output_img_params["output_img_cols"]
    predictions_mean = prediction_results["prediction_mean"]
    predictions_mean_original = prediction_results.get("prediction_mean_original", predictions_mean)
    # Masking
    ground_truths_masked = np.where(ground_truths["original"] < threshold, 0, ground_truths["original"])
    predictions_mean_masked = np.where(predictions_mean_original < threshold, 0, predictions_mean_original)
    # Reconstruct to image visualization
    gts, gt_mean, _ = reconstruct_output_image(ground_truths_masked, rows, cols, valid_cols)
    preds, pred_mean, _ = reconstruct_output_image(predictions_mean_masked, rows, cols, valid_cols)
    if mode == "binary":
        _viz_diff_grid_points_binary(gt_mean, pred_mean, model_name)
    elif mode == "percentage":
        _viz_diff_grid_points_percentage(gts, preds, model_name)
    else:
        raise ValueError(f"Mode must be 'binary' or 'percentage', but got {mode}")
    
def viz_all_2d_maps_in_one_plot(
    ground_truth: dict, 
    prediction_results: dict, 
    output_img_params: dict, 
    model_name: str, 
    qoi: str = "hmax", 
    threshold: float = 0.5
    ):
    valid_cols = output_img_params["filtered_columns"]
    rows = output_img_params["output_img_rows"]
    cols = output_img_params["output_img_cols"]
    bg_img_path = output_img_params.get("background_img_path", None)
    predictions_mean = prediction_results["prediction_mean"]
    predictions_mean_original = prediction_results.get("prediction_mean_original", predictions_mean)
    # Masking
    ground_truths_masked = np.where(ground_truth["original"] < threshold, 0, ground_truth["original"])
    predictions_mean_masked = np.where(predictions_mean_original < threshold, 0, predictions_mean_original)
    # Reconstruct to image visualization
    gts, gt_mean, _ = reconstruct_output_image(ground_truths_masked, rows, cols, valid_cols)
    preds, pred_mean, _ = reconstruct_output_image(predictions_mean_masked, rows, cols, valid_cols)
    _viz_all_2d_maps_in_one_plot(gts, preds, qoi, model_name, bg_img_path)
    
def plot_pca_zero_output_hist(ground_truth: np.ndarray, predictions: dict, threshold: float = 0.5):
    """
    Plot the histogram of the zero output.
    """
    # Masking
    ground_truth = np.where(ground_truth < threshold, 0, ground_truth)
    arrs = [ground_truth]
    labels = ['Ground-truth']
    for model_name, prediction in predictions.items():
        prediction = np.where(prediction < threshold, 0, prediction)
        arrs.append(prediction)
        if model_name == "ppgasp":
            labels.append("PPGaSP")
        if model_name == "pca_ppgasp":
            labels.append("PCA-PPGaSP")
        if model_name == "ppgasp_gpytorch":
            labels.append("PPGaSP-GPyTorch")
        if model_name == "pca_ppgasp_gpytorch":
            labels.append("PCA-PPGaSP-GPyTorch")
        if model_name == "correlated_gpytorch":
            labels.append("Correlated GPyTorch")
    
    colors = [f'C{i}' for i in range(len(arrs))]
    zero_counts = [np.count_nonzero(arr == 0) for arr in arrs]
    
    x = np.arange(len(arrs))
    width = 0.4
    fig, ax = plt.subplots()
    ax.bar(x, zero_counts, width, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel('Number of zero elements')
    ax.set_title('Comparison of Number of Zero Elements in Prediction for Different Models', pad=12)
    plt.tight_layout()
    plt.show()

