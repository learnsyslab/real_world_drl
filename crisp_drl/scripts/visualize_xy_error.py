"""Visualize XY error distribution as a scatter plot centered at (0,0)."""

import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path


def load_samples(jsonl_path: Path) -> list[dict]:
    """Load samples from JSONL file."""
    samples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def extract_xyz_errors(
    samples: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract X, Y, and Z errors from samples."""
    x_errors = []
    y_errors = []
    z_errors = []
    for sample in samples:
        error = sample.get("pose_estimation.goal_position_error", [0, 0, 0])
        x_errors.append(error[0])
        y_errors.append(error[1])
        z_errors.append(error[2])
    return np.array(x_errors), np.array(y_errors), np.array(z_errors)


def plot_xy_error_distribution(
    x_errors: np.ndarray,
    y_errors: np.ndarray,
    z_errors: np.ndarray,
    output_path: str = "",
):
    """Create scatter plot of XY error distribution centered at (0,0), colored by Z error."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Convert to millimeters for better readability
    x_mm = x_errors * 1000
    y_mm = y_errors * 1000
    z_mm = z_errors * 1000 + 10.9  # Compensate for goal position offset in Z

    # Scatter plot colored by z-error
    scatter = ax.scatter(
        x_mm,
        y_mm,
        alpha=0.7,
        s=100,
        c=z_mm,
        cmap="coolwarm",
        edgecolors="black",
        linewidth=0.5,
    )

    # Add colorbar for z-error
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.08)
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label("z-error [mm]")
    cbar.ax.tick_params(labelsize=10)

    # Add sample numbers inside the circles
    # for i, (x, y) in enumerate(zip(x_mm, y_mm)):
    #     ax.annotate(
    #         str(i + 1),
    #         (x, y),
    #         ha="center",
    #         va="center",
    #         fontsize=7,
    #         color="white",
    #         fontweight="bold",
    #     )

    # Add crosshairs at origin
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    gt_legend = ax.scatter(
        [0],
        [0],
        c="red",
        s=100,
        marker="+",
        linewidths=2,
        zorder=5,
        label="Ground Truth",
    )

    # Calculate and display statistics
    mean_x = np.mean(x_mm)
    mean_y = np.mean(y_mm)
    std_x = np.std(x_mm)
    std_y = np.std(y_mm)
    euclidean_errors = np.sqrt(x_mm**2 + y_mm**2)
    mean_euclidean = np.mean(euclidean_errors).item()

    # Mark mean position
    mean_legend = ax.scatter(
        [mean_x],
        [mean_y],
        c="black",
        s=100,
        marker="x",
        linewidths=2,
        zorder=5,
        label=f"Mean ({mean_x:.2f}, {mean_y:.2f})",
    )

    # Add circle showing mean euclidean error

    # Make axes equal and centered
    max_range = max(np.max(np.abs(x_mm)), np.max(np.abs(y_mm))) * 1.2
    ax.set_xlim(-2.58, 0.2)
    ax.set_ylim(-1.1, 0.2)
    ax.set_xticks(np.arange(-2.5, 0.2, 0.25))
    ax.set_yticks(np.arange(-1.0, 0.2, 0.25))
    ax.set_aspect("equal")

    # Labels and title
    ax.set_xlabel("x-error [mm]")
    ax.set_ylabel("y-error [mm]")
    ax.set_title("PE Error Distribution", fontsize=14)

    # Calculate z statistics
    mean_z = np.mean(z_mm)
    std_z = np.std(z_mm)

    # Add statistics text box
    # stats_text = (
    #     f"N = {len(x_mm)}\n"
    #     f"Mean X: {mean_x:.3f} mm\n"
    #     f"Mean Y: {mean_y:.3f} mm\n"
    #     f"Mean Z: {mean_z:.3f} mm\n"
    #     f"Std X: {std_x:.3f} mm\n"
    #     f"Std Y: {std_y:.3f} mm\n"
    #     f"Std Z: {std_z:.3f} mm\n"
    #     f"Mean Euclidean (XY): {mean_euclidean:.3f} mm"
    # )
    # ax.text(
    #     0.02,
    #     0.98,
    #     stats_text,
    #     transform=ax.transAxes,
    #     fontsize=10,
    #     verticalalignment="top",
    #     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    # )

    sample_legend = Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="C0",
        markeredgecolor="black",
        markersize=10,
        linestyle="None",
        label="Samples",
    )
    ax.legend(handles=[sample_legend, mean_legend, gt_legend], loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path != "":
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    plt.show()


def main():
    # Path to the samples file
    jsonl_path = Path("eval/pose_estimation") / "samples.jsonl"

    if not jsonl_path.exists():
        print(f"Error: File not found: {jsonl_path}")
        return

    # Load and process data
    samples = load_samples(jsonl_path)
    print(f"Loaded {len(samples)} samples")

    x_errors, y_errors, z_errors = extract_xyz_errors(samples)

    # Create visualization
    output_path = Path("eval/pose_estimation") / "xy_error_distribution.png"
    plot_xy_error_distribution(
        x_errors, y_errors, z_errors, output_path=str(output_path)
    )


if __name__ == "__main__":
    main()
