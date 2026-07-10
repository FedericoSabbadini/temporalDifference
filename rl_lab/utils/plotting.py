"""Plotting utilities — learning curves with std bands, sensitivity plots.

Origin: moving_average is ADAPTED from the course exercises.
plot_learning_curves (mean + std bands) and plot_sensitivity are NEW.
The course exercises plotted single runs; here we need multi-seed bands.

All plots follow the same style:
  - Moving-average smoothing for readability
  - Mean line + shaded ±1 std band (when multiple seeds)
  - Labeled axes, legend, grid
  - Consistent figure size (10×5)

This produces the "learning curves (reward vs episodi)" and "sensibilità
agli iperparametri" plots required by the assignment for every environment.
"""

import os
import numpy as np
import matplotlib
# Non-interactive backend: save figures to disk without a GUI. Avoids GUI-backend
# issues (Tk load blocked on Windows App Control; crashes on some macOS setups).
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def moving_average(x, window=20):
    """Smooth a 1-D array with a moving average.

    Uses np.convolve with a uniform kernel of size `window`.
    Returns an array shorter by (window-1) elements ('valid' mode).
    """
    if len(x) < window:
        return np.array(x)  # not enough data to smooth — return as-is
    # np.convolve with a uniform kernel [1/w, 1/w, ..., 1/w] computes the running mean.
    # mode="valid" means output length = len(x) - window + 1 (no padding artifacts).
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_learning_curves(results_dict, title, save_path=None, window=20,
                         ylabel="Return"):
    """Plot learning curves for multiple algorithms with std bands.

    This is the primary visualization tool for the project.
    Each algorithm gets a smoothed mean line with a shaded ±1σ band,
    showing both average performance and variability across seeds.

    Args:
        results_dict: {label: np.array of shape (n_seeds, n_episodes)}.
            Each value is a 2D array where rows are seeds and columns are episodes.
        title: plot title.
        save_path: if set, save the figure here.
        window: moving-average window size (higher = smoother but loses detail).
        ylabel: y-axis label (default "Return" = total episode reward).
    """
    plt.figure(figsize=(10, 5))

    for label, data in results_dict.items():
        if data.ndim == 1:
            data = data.reshape(1, -1)  # single seed → treat as 1-row matrix
        # axis=0 computes mean/std ACROSS seeds (each row is one seed's returns)
        mean = np.mean(data, axis=0)  # (n_episodes,) — average performance
        std = np.std(data, axis=0)    # (n_episodes,) — variability across seeds
        # Smooth both mean and std for readability (raw curves are very noisy)
        sm_mean = moving_average(mean, window)
        sm_std = moving_average(std, window)
        x = np.arange(len(sm_mean))  # x-axis = episode number (after smoothing)
        plt.plot(x, sm_mean, label=label)  # solid line = mean performance
        # Shaded band = ±1 standard deviation (shows seed-to-seed variability)
        plt.fill_between(x, sm_mean - sm_std, sm_mean + sm_std, alpha=0.2)

    plt.xlabel(f"Episode (smoothed, window={window})")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)  # light grid for readability
    plt.tight_layout()  # prevent label clipping

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()  # free memory (important when generating many figures)


def plot_sensitivity(results_by_param, param_name, algo_name, title=None,
                     save_path=None, window=20):
    """Plot learning curves for different hyperparameter values.

    Used for the "sensitivity analysis" required by the assignment:
    how does changing ε, α, or γ affect learning?

    Args:
        results_by_param: {param_value: np.array (n_seeds, n_episodes)}.
        param_name: parameter name for legend (e.g. "ε", "α", "γ").
        algo_name: algorithm name for title (e.g. "Q-Learning").
    """
    plt.figure(figsize=(10, 5))

    for param_val, data in results_by_param.items():
        if data.ndim == 1:
            data = data.reshape(1, -1)
        # Only plot the mean (no std band) to keep sensitivity plots readable
        # with multiple overlapping curves
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
