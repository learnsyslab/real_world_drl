"""Visualize grasping error distributions from pose estimation grasping evaluation."""

import json
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from pathlib import Path


def load_samples(jsonl_path: Path) -> list[dict]:
    """Load samples from JSONL file."""
    samples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def extract_errors(
    samples: list[dict], key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract X, Y, and Z errors from samples for a given key."""
    x_errors = []
    y_errors = []
    z_errors = []
    for sample in samples:
        error = sample.get(key, [0, 0, 0])
        x_errors.append(error[0])
        y_errors.append(error[1])
        z_errors.append(error[2])
    return np.array(x_errors), np.array(y_errors), np.array(z_errors)


def plot_xy_error(
    ax: plt.Axes,
    x_errors: np.ndarray,
    y_errors: np.ndarray,
    z_errors: np.ndarray,
    title: str,
    add_colorbar: bool = True,
) -> None:
    """Create scatter plot of XY error distribution colored by Z error."""
    # Convert to millimeters
    x_mm = x_errors * 1000
    y_mm = y_errors * 1000
    z_mm = z_errors * 1000

    # Scatter plot colored by z-error
    scatter = ax.scatter(
        x_mm,
        y_mm,
        alpha=0.7,
        s=80,
        c=z_mm,
        cmap="coolwarm",
        edgecolors="black",
        linewidth=0.5,
    )

    if add_colorbar:
        cbar = plt.colorbar(scatter, ax=ax, label="Z Error (mm)")
        cbar.ax.tick_params(labelsize=8)

    # Add crosshairs at origin
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    # Mark the origin
    ax.scatter([0], [0], c="red", s=80, marker="+", linewidths=2, zorder=5)

    # Calculate statistics
    mean_x = np.mean(x_mm)
    mean_y = np.mean(y_mm)
    euclidean_errors = np.sqrt(x_mm**2 + y_mm**2)
    mean_euclidean = np.mean(euclidean_errors).item()

    # Mark mean position
    ax.scatter(
        [mean_x],
        [mean_y],
        c="orange",
        s=80,
        marker="x",
        linewidths=2,
        zorder=5,
    )

    # Add circle showing mean euclidean error
    circle = plt.Circle(
        (0, 0),
        mean_euclidean,
        fill=False,
        color="green",
        linestyle="--",
        linewidth=1.5,
    )
    ax.add_patch(circle)

    # Make axes equal and centered
    max_range = max(np.max(np.abs(x_mm)), np.max(np.abs(y_mm)), 0.1) * 1.3
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_aspect("equal")

    ax.set_xlabel("X Error (mm)", fontsize=10)
    ax.set_ylabel("Y Error (mm)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Statistics text
    std_x = np.std(x_mm)
    std_y = np.std(y_mm)
    stats_text = (
        f"N={len(x_mm)}\n"
        f"μX: {mean_x:.2f} mm\n"
        f"μY: {mean_y:.2f} mm\n"
        f"σX: {std_x:.2f} mm\n"
        f"σY: {std_y:.2f} mm\n"
        f"μ|XY|: {mean_euclidean:.2f} mm"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.grid(True, alpha=0.3)


def plot_xz_error(
    ax: plt.Axes,
    x_errors: np.ndarray,
    z_errors: np.ndarray,
    title: str,
) -> dict:
    """Create scatter plot of XZ error distribution (without Y component) with Gaussian fit."""
    # Convert to millimeters
    x_mm = x_errors * 1000
    z_mm = z_errors * 1000

    # Scatter plot
    ax.scatter(
        x_mm,
        z_mm,
        alpha=0.7,
        s=80,
        c="steelblue",
        edgecolors="black",
        linewidth=0.5,
        zorder=3,
    )

    # Add crosshairs at origin
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    # Mark the origin
    ax.scatter([0], [0], c="red", s=80, marker="+", linewidths=2, zorder=5)

    # Calculate statistics
    mean_x = np.mean(x_mm)
    mean_z = np.mean(z_mm)
    min_x = np.min(x_mm)
    max_x = np.max(x_mm)
    min_z = np.min(z_mm)
    max_z = np.max(z_mm)
    std_x = np.std(x_mm)
    std_z = np.std(z_mm)
    euclidean_errors = np.sqrt(x_mm**2 + z_mm**2)
    mean_euclidean = np.mean(euclidean_errors).item()

    # Fit 2D Gaussian: compute covariance matrix and its eigenvalues/eigenvectors
    data = np.vstack([x_mm, z_mm])
    cov = np.cov(data)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort eigenvalues and eigenvectors in descending order
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Compute rotation angle from eigenvectors
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    # Draw 1, 2, 3 sigma ellipses centered at mean
    sigma_colors = ["#2ecc71", "#f39c12", "#e74c3c"]  # green, orange, red
    sigma_labels = ["1σ", "2σ", "3σ"]
    for n_sigma, color, label in zip([1, 2, 3], sigma_colors, sigma_labels):
        # Width and height are 2 * n_sigma * sqrt(eigenvalue)
        width = 2 * n_sigma * np.sqrt(eigenvalues[0])
        height = 2 * n_sigma * np.sqrt(eigenvalues[1])
        ellipse = Ellipse(
            xy=(mean_x, mean_z),
            width=width,
            height=height,
            angle=angle,
            fill=False,
            color=color,
            linewidth=1.5,
            linestyle="-",
            label=label,
            zorder=2,
        )
        ax.add_patch(ellipse)

    # Mark mean position
    ax.scatter(
        [mean_x],
        [mean_z],
        c="orange",
        s=80,
        marker="x",
        linewidths=2,
        zorder=5,
        label=f"Mean ({mean_x:.2f}, {mean_z:.2f})",
    )

    # Make axes equal and centered
    # Account for 3-sigma ellipse size
    max_ellipse_extent = 3 * max(np.sqrt(eigenvalues[0]), np.sqrt(eigenvalues[1]))
    max_range = (
        max(
            np.max(np.abs(x_mm)),
            np.max(np.abs(z_mm)),
            abs(mean_x) + max_ellipse_extent,
            abs(mean_z) + max_ellipse_extent,
            0.1,
        )
        * 1.2
    )
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_aspect("equal")

    ax.set_xlabel("X Error (mm)", fontsize=10)
    ax.set_ylabel("Z Error (mm)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Compute correlation coefficient
    corr = cov[0, 1] / (std_x * std_z) if std_x > 0 and std_z > 0 else 0

    # Statistics text
    stats_text = (
        f"N={len(x_mm)}\n"
        f"μX: {mean_x:.2f} mm\n"
        f"μZ: {mean_z:.2f} mm\n"
        f"σX: {std_x:.2f} mm\n"
        f"σZ: {std_z:.2f} mm\n"
        f"ρ: {corr:.3f}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Return statistics for printing
    return {
        "min_x": min_x,
        "max_x": max_x,
        "mean_x": mean_x,
        "min_z": min_z,
        "max_z": max_z,
        "mean_z": mean_z,
    }


def main():
    # Path to the samples file
    jsonl_path = Path("eval/pose_estimation_grasping") / "samples.jsonl"

    if not jsonl_path.exists():
        print(f"Error: File not found: {jsonl_path}")
        return

    # Load and process data
    samples = load_samples(jsonl_path)
    print(f"Loaded {len(samples)} samples")

    # Extract errors for each type
    est_x, est_y, est_z = extract_errors(samples, "estimation_error")
    ctrl_x, ctrl_y, ctrl_z = extract_errors(samples, "controller_error")
    open_x, open_y, open_z = extract_errors(samples, "total_error_gripper_open")
    closed_x, closed_y, closed_z = extract_errors(samples, "total_error_gripper_closed")

    # Create figure with 4 subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot XY errors with Z as color
    plot_xy_error(axes[0, 0], est_x, est_y, est_z, "Estimation Error (XY, Z-colored)")
    plot_xy_error(
        axes[0, 1], ctrl_x, ctrl_y, ctrl_z, "Controller Error (XY, Z-colored)"
    )
    plot_xy_error(
        axes[1, 0], open_x, open_y, open_z, "Total Error Gripper Open (XY, Z-colored)"
    )

    # Plot XZ error for gripper closed (without Y)
    stats = plot_xz_error(
        axes[1, 1], closed_x, closed_z, "Total Error Gripper Closed (XZ, no Y)"
    )

    plt.tight_layout()

    # Save the plot
    output_path = (
        Path("eval/pose_estimation_grasping") / "grasping_error_distribution.png"
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    # Print statistics for total_error_gripper_closed
    print("\n" + "=" * 50)
    print("Total Error Gripper Closed Statistics (XZ)")
    print("=" * 50)
    print(f"X Error (mm):")
    print(f"  Min:  {stats['min_x']:.4f}")
    print(f"  Max:  {stats['max_x']:.4f}")
    print(f"  Mean: {stats['mean_x']:.4f}")
    print(f"Z Error (mm):")
    print(f"  Min:  {stats['min_z']:.4f}")
    print(f"  Max:  {stats['max_z']:.4f}")
    print(f"  Mean: {stats['mean_z']:.4f}")
    print("=" * 50)

    plt.show()


if __name__ == "__main__":
    main()
