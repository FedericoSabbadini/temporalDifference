"""Plotting utilities — learning curves with std bands, sensitivity plots.

All plots follow the same style:
  - Moving-average smoothing for readability
  - Mean line + shaded ±1 std band (when multiple seeds)
  - Labeled axes, legend, grid
  - Consistent figure size

Matches the course exercise plotting conventions (moving_average, figsize).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend: save figures to disk without a GUI
                        # (avoids the Tk/_imagingtk DLL load blocked by Windows App Control)
import matplotlib.pyplot as plt


def moving_average(x, window=20):
    """Smooth a 1-D array with a moving average."""
    if len(x) < window:
        return np.array(x)
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_learning_curves(results_dict, title, save_path=None, window=20,
                         ylabel="Return"):
    """Plot learning curves for multiple algorithms with std bands.

    Args:
        results_dict: {label: np.array of shape (n_seeds, n_episodes)}.
        title: plot title.
        save_path: if set, save the figure here.
        window: moving-average window.
        ylabel: y-axis label.
    """
    plt.figure(figsize=(10, 5))

    for label, data in results_dict.items():
        if data.ndim == 1:
            data = data.reshape(1, -1)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        sm_mean = moving_average(mean, window)
        sm_std = moving_average(std, window)
        x = np.arange(len(sm_mean))
        plt.plot(x, sm_mean, label=label)
        plt.fill_between(x, sm_mean - sm_std, sm_mean + sm_std, alpha=0.2)

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
                     save_path=None, window=20):
    """Plot learning curves for different hyperparameter values.

    Args:
        results_by_param: {param_value: np.array (n_seeds, n_episodes)}.
        param_name: parameter name for legend (e.g. "ε").
        algo_name: algorithm name for title.
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
    plt.title(title or f"{algo_name} — sensitivity to {param_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()
