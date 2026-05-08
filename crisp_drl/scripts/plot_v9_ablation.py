import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize
from pathlib import Path
from collections import defaultdict

BASE_FOLDER = "rollout_data/collect_data"
V9_PREFIX = "20260213"


def parse_folder_name(name: str) -> tuple[str, float, int]:
    """Parse folder name: {datetime}_{hyperparameter}_{success_pct}"""
    parts = name.split("_")
    datetime_str = parts[0]
    hyperparameter = float(parts[1])
    success_pct = int(parts[2])
    return datetime_str, hyperparameter, success_pct


def compute_rollout_stats(folder_path: Path) -> tuple[float, float, int]:
    """Compute success rate, mean successful episode length, and dataset size."""
    n_total = 0
    n_success = 0
    n_actions_total = 0
    success_lengths = []

    for f in sorted(os.listdir(folder_path)):
        if not f.endswith(".npz"):
            continue
        data = np.load(folder_path / f, allow_pickle=True)
        n_total += 1
        n_actions_total += len(data["actions"])
        if data["terminated"]:
            n_success += 1
            success_lengths.append(len(data["actions"]))

    success_rate = n_success / n_total if n_total > 0 else 0.0
    mean_success_length = (
        float(np.mean(success_lengths)) if success_lengths else float("nan")
    )
    dataset_size = n_actions_total
    return success_rate, mean_success_length, dataset_size


def confidence_ellipse(mean, cov, ax, n_std=1.0, **kwargs):
    """Plot a covariance ellipse at n_std standard deviations."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Ensure non-negative eigenvalues (numerical stability)
    eigenvalues = np.maximum(eigenvalues, 0)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)

    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)
    return ellipse


def main():
    parser = argparse.ArgumentParser(
        description="Plot v9 ablation: success rate vs successful episode length."
    )
    parser.add_argument(
        "--no-variance-ellipses",
        action="store_true",
        help="Do not draw covariance/variance ellipses around each hyperparameter point.",
    )
    args = parser.parse_args()

    # Collect all v9 ablation folders
    hyper_data = defaultdict(
        list
    )  # hyperparameter -> [(success_rate, mean_success_length)]
    dataset_size_data = defaultdict(list)  # hyperparameter -> [dataset_size]

    for folder_name in sorted(os.listdir(BASE_FOLDER)):
        if not folder_name.startswith(V9_PREFIX):
            continue
        folder_path = Path(BASE_FOLDER) / folder_name
        if not folder_path.is_dir():
            continue

        _, hyperparameter, _ = parse_folder_name(folder_name)
        print(f"Processing {folder_name} (hyperparameter={hyperparameter})")

        success_rate, mean_success_length, dataset_size = compute_rollout_stats(
            folder_path
        )
        dataset_size_data[hyperparameter].append(dataset_size)
        if not np.isnan(mean_success_length):
            hyper_data[hyperparameter].append((success_rate, mean_success_length))

    # Compute means and covariances per hyperparameter
    hyperparams_sorted = sorted(hyper_data.keys())
    means = []
    covs = []
    labels = []

    for hp in hyperparams_sorted:
        points = np.array(hyper_data[hp])  # shape (n_runs, 2)
        mean = points.mean(axis=0)
        means.append(mean)
        labels.append(hp)
        if len(points) > 1:
            cov = np.cov(points.T)
        else:
            cov = np.zeros((2, 2))
        covs.append(cov)
        print(
            f"  hp={hp}: n={len(points)}, "
            f"success_rate={mean[0]:.1f} ± {np.sqrt(cov[0, 0]):.1f}, "
            f"mean_length={mean[1]:.1f} ± {np.sqrt(cov[1, 1]):.1f}"
        )

    means = np.array(means)

    # Compute mean/std dataset size per hyperparameter (uses all runs).
    dataset_size_means = []
    dataset_size_stds = []
    for hp in hyperparams_sorted:
        sizes = np.array(dataset_size_data[hp], dtype=float)
        dataset_size_means.append(float(np.mean(sizes)))
        dataset_size_stds.append(float(np.std(sizes)))

    dataset_size_means = np.array(dataset_size_means)
    dataset_size_stds = np.array(dataset_size_stds)

    # Plot
    viridis = cm.get_cmap("viridis")

    # Figure 1: mean dataset size over p_rand.
    fig_left, ax_left = plt.subplots(figsize=(7, 6))
    ax_left.errorbar(
        hyperparams_sorted,
        dataset_size_means,
        yerr=dataset_size_stds,
        fmt="-o",
        color="tab:blue",
        ecolor="tab:blue",
        elinewidth=1.2,
        capsize=3,
        markersize=5,
    )
    ax_left.set_xlabel("$p_{rand}$", fontsize=13)
    ax_left.set_ylabel(
        "Dataset Size $\\left\\langle\\mathcal{D}\\right\\rangle$ (number of transitions)",
        fontsize=13,
    )
    ax_left.set_title(
        "Dataset Size $\\left\\langle\\mathcal{D}\\right\\rangle$ vs $p_{rand}$",
        fontsize=14,
    )
    ax_left.grid(True, alpha=0.3)
    fig_left.tight_layout()
    fig_left.savefig("eval/v9_ablation_dataset_size.png", dpi=150, bbox_inches="tight")

    # Figure 2: success vs length.
    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw uncertainty ellipses unless disabled via CLI.
    if not args.no_variance_ellipses:
        for i, (mean, cov, hp) in enumerate(zip(means, covs, labels)):
            if np.any(cov > 0):
                color_value = 0.5 if len(labels) == 1 else i / (len(labels) - 1)
                confidence_ellipse(
                    mean,
                    cov,
                    ax,
                    n_std=1.0,
                    facecolor=viridis(color_value),
                    alpha=0.2,
                    edgecolor=viridis(color_value),
                    linewidth=1.5,
                )

    # Draw line through means (ordered by increasing hyperparameter)
    ax.plot(
        means[:, 0],
        means[:, 1],
        "-",
        color="gray",
        linewidth=1.5,
        alpha=0.7,
        zorder=1,
    )

    # Draw data points
    colors = [
        viridis(0.5 if len(labels) == 1 else i / (len(labels) - 1))
        for i in range(len(labels))
    ]
    ax.scatter(
        means[:, 0],
        means[:, 1],
        c=colors,
        s=80,
        zorder=3,
        edgecolors="black",
        linewidths=0.8,
    )

    # Label each point with its hyperparameter value
    for i, hp in enumerate(labels):
        ax.annotate(
            f"{hp}",
            (means[i, 0], means[i, 1]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Success Rate", fontsize=13)
    ax.set_ylabel("Mean Successful Rollout Length (steps)", fontsize=13)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-5, 125)
    ax.set_title(
        "Dataset Mean Successful Rollout Length vs Success Rate by $p_{rand}$",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)

    # Add colorbar for hyperparameter
    sm = cm.ScalarMappable(
        cmap=viridis,
        norm=Normalize(vmin=min(labels), vmax=max(labels)),
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("$p_{rand}$", fontsize=12)

    fig.tight_layout()
    fig.savefig("eval/v9_ablation_success_vs_length.png", dpi=150, bbox_inches="tight")
    print(
        "\nSaved to eval/v9_ablation_dataset_size.png and eval/v9_ablation_success_vs_length.png"
    )
    plt.show()


if __name__ == "__main__":
    main()
