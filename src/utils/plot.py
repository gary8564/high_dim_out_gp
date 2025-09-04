import numpy as np
import rasterio
import os
import logging
from matplotlib import pyplot as plt
from rasterio.plot import plotting_extent

# Basic configuration with level set to INFO
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def viz_flattened_prediction(ground_truth, prediction_mean, prediction_std, save_dir, model_name):
    plt.figure()
    y_true = ground_truth.flatten()
    y_pred = prediction_mean.flatten()
    pred_std = prediction_std.flatten()
    logger.info(f"Ground-truth shape: {y_true.shape}, Prediction shape: {y_pred.shape}")
    plt.plot([np.min(y_true),np.max(y_true)], [np.min(y_true),np.max(y_true)])
    plt.errorbar(y_true, y_pred, yerr=pred_std, fmt='.', label='emulator', color='pink', mfc='pink', zorder=1)
    plt.xlabel('Actual y')
    plt.ylabel('Predicted y')
    plt.xlim(np.min(y_true),np.max(y_true))
    plt.ylim(np.min(y_true),np.max(y_true))
    plt.legend()
    plt.tight_layout()
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    save_loc = os.path.join(curr_dir, save_dir)
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    plt.savefig(os.path.join(save_loc, f"gt_vs_pred_{model_name}.png"))
    plt.show()
    plt.close()

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

def viz_output_image(mean_gt, std_gt, mean_pred, std_pred, qoi, threshold, model_name, save_dir, hill_path=None) -> None:
    """Visualize the reconstructed output image

    Args:
        mean_gt (np.ndarray): mean value of the ground-truth image array
        std_gt (np.ndarray): standard deviation value of the ground-truth image array
        mean_pred (np.ndarray): mean value of the predicted image array
        std_pred (np.ndarray): standard deviation value of the predicted image array
        qoi (str): quantity of interest
        threshold (float): threshold value to define valid cells from simulations
        model_name (str): name of the model
        save_dir (str): save results to the specified filepath
        hill_path (str | None): filepath of the background image
    """
    font_size = 12
    tick_size = 10
    if qoi not in ["hmax", "vmax"]:
        raise ValueError(f"Invalid quantity of interest: {qoi}")
    label = "flow height [m]" if qoi == "hmax" else "flow velocity [m/s]"
    title = "Flow Height" if qoi == "hmax" else "Flow Velocity"
    # Masking
    mean_gt_mask = np.ma.masked_where(mean_gt < threshold, mean_gt, copy=True)
    std_gt_mask = np.ma.masked_where(mean_gt < threshold, std_gt, copy=True)
    mean_pred_mask = np.ma.masked_where(mean_pred < threshold, mean_pred, copy=True)
    std_pred_mask = np.ma.masked_where(mean_pred < threshold, std_pred, copy=True)
    
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
        (mean_gt_mask, 'viridis', threshold, max_mean_value, "Mean " + label),
        (mean_pred_mask, 'viridis', threshold, max_mean_value, "Mean " + label),
        (diff_mean_mask, 'RdBu_r', -max_diff_mean_value, max_diff_mean_value, 'Difference in mean [m]'),
        (std_gt_mask, 'viridis', threshold, max_std_value, "Std. deviation [m]"),
        (std_pred_mask, 'viridis', threshold, max_std_value, "Std. deviation [m]"),
        (diff_std_mask, 'RdBu_r', -max_diff_std_value, max_diff_std_value, "Difference in std. deviation [m]"),
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
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    save_loc = os.path.join(curr_dir, save_dir)
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    plt.savefig(os.path.join(save_loc, f"2d_massflow_map_{model_name}.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
