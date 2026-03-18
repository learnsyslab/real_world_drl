import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
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


def compute_rollout_stats(folder_path: Path) -> tuple[float, float]:
    """Compute success rate and mean successful episode length for a folder."""
    n_total = 0
    n_success = 0
    success_lengths = []

    for f in sorted(os.listdir(folder_path)):
        if not f.endswith(".npz"):
            continue
        data = np.load(folder_path / f, allow_pickle=True)
        n_total += 1
        if data["terminated"]:
            n_success += 1
            success_lengths.append(len(data["actions"]))

    success_rate = n_success / n_total * 100 if n_total > 0 else 0.0
    mean_success_length = np.mean(success_lengths) if success_lengths else float("nan")
    return success_rate, mean_success_length


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
    # Collect all v9 ablation folders
    hyper_data = defaultdict(
        list
    )  # hyperparameter -> [(success_rate, mean_success_length)]

    for folder_name in sorted(os.listdir(BASE_FOLDER)):
        if not folder_name.startswith(V9_PREFIX):
            continue
        folder_path = Path(BASE_FOLDER) / folder_name
        if not folder_path.is_dir():
            continue

        _, hyperparameter, _ = parse_folder_name(folder_name)
        print(f"Processing {folder_name} (hyperparameter={hyperparameter})")

        success_rate, mean_success_length = compute_rollout_stats(folder_path)
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
            f"success_rate={mean[0]:.1f}% ± {np.sqrt(cov[0, 0]):.1f}, "
            f"mean_length={mean[1]:.1f} ± {np.sqrt(cov[1, 1]):.1f}"
        )

    means = np.array(means)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Draw uncertainty ellipses
    for i, (mean, cov, hp) in enumerate(zip(means, covs, labels)):
        if np.any(cov > 0):
            confidence_ellipse(
                mean,
                cov,
                ax,
                n_std=1.0,
                facecolor=plt.cm.viridis(i / (len(labels) - 1)),
                alpha=0.2,
                edgecolor=plt.cm.viridis(i / (len(labels) - 1)),
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
    colors = [plt.cm.viridis(i / (len(labels) - 1)) for i in range(len(labels))]
    scatter = ax.scatter(
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

    ax.set_xlabel("Episode Success Rate (%)", fontsize=13)
    ax.set_ylabel("Mean Successful Episode Length (steps)", fontsize=13)
    ax.set_title(
        "Env v9 Ablation: Success Rate vs Episode Length by Hyperparameter", fontsize=14
    )
    ax.grid(True, alpha=0.3)

    # Add colorbar for hyperparameter
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis,
        norm=plt.Normalize(vmin=min(labels), vmax=max(labels)),
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Hyperparameter Value", fontsize=12)

    plt.tight_layout()
    plt.savefig("eval/v9_ablation_success_vs_length.png", dpi=150, bbox_inches="tight")
    plt.savefig("eval/v9_ablation_success_vs_length.pdf", bbox_inches="tight")
    print("\nSaved to eval/v9_ablation_success_vs_length.png and .pdf")
    plt.show()


if __name__ == "__main__":
    main()
