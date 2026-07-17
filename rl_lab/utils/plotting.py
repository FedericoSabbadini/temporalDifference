"""Learning-curve and sensitivity plots.

Origin: moving_average is ADAPTED from the course exercises; the multi-seed mean+band
plots are NEW (the course plotted single runs).

House style for every figure: moving-average smoothing, mean line over the 3 seeds,
shaded +/-1 sigma band, labelled axes, legend, grid.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save to disk without a GUI (Tk is blocked on the Windows box)
import matplotlib.pyplot as plt


def moving_average(x, window=20):
    """Running mean, 'valid' mode (output is window-1 shorter).
    
    Args:
        x: 1D array.
        window: Size of the moving average window; used for smoothing the learning curves. 
        Use 1 for no smoothing (e.g. for value-estimate plots).
        
    Returns:
        1D array of the same type as x, smoothed (meaning each element is the mean of the previous `window` elements).
        "valid" mode means the output only includes points where the full window fits.
    """
    if len(x) < window:
        # case of very short arrays (e.g. 1 episode), just return the array itself
        return np.array(x)
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_learning_curves(results_dict, title, save_path=None, window=40,
                         ylabel="Return"):
    """Mean curve +/-1 sigma band, one entry per algorithm/configuration.

    Args:
        results_dict: {label: array of shape (n_seeds, n_episodes)}.
        window: moving-average window (use 1 for the value-estimate plots, where the
            raw variance of MC is exactly what has to be visible).
    """
    plt.figure(figsize=(10, 5))

    for label, data in results_dict.items():
        if data.ndim == 1:
            data = data.reshape(1, -1) # 
        # axis=0 -> across seeds
        mean = np.mean(data, axis=0) # mean over all seeds, for each episode
        std = np.std(data, axis=0) # std over all seeds, for each episode
        sm_mean = moving_average(mean, window)
        sm_std = moving_average(std, window)
        
        x = np.arange(len(sm_mean))
        plt.plot(x, sm_mean, label=label)
        plt.fill_between(x, sm_mean - sm_std, sm_mean + sm_std, alpha=0.2) # +- 0.1 required

    plt.xlabel(f"Episode (smoothed, window={window})")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_sensitivity(results_by_param, param_name, algo_name, title=None,
                     save_path=None, window=40):
    """One mean curve per hyperparameter value.

    Each curve is still the mean over the 3 seeds, but I deliberately omit the sigma
    bands here: with four overlapping curves the shaded regions cover each other and
    the plot becomes unreadable. The seed spread is reported in the tables instead.
    """
    plt.figure(figsize=(10, 5))

    for param_val, data in results_by_param.items():
        if data.ndim == 1:
            data = data.reshape(1, -1)
        mean = np.mean(data, axis=0)
        sm = moving_average(mean, window)
        plt.plot(range(len(sm)), sm, label=f"{param_name}={param_val}")

    plt.xlabel(f"Episode (smoothed, window={window})")
    plt.ylabel("Return")
    plt.title(title or f"{algo_name} - sensitivity to {param_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()
