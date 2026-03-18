"""Visualize XY error distribution as a scatter plot centered at (0,0)."""

import json
import matplotlib.pyplot as plt
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
    fig, ax = plt.subplots(figsize=(10, 8))

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
    cbar = plt.colorbar(scatter, ax=ax, label="Z Error (mm)")
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

    # Mark the origin
    ax.scatter(
        [0],
        [0],
        c="red",
        s=100,
        marker="+",
        linewidths=2,
        zorder=5,
        label="Origin (0,0)",
    )

    # Calculate and display statistics
    mean_x = np.mean(x_mm)
    mean_y = np.mean(y_mm)
    std_x = np.std(x_mm)
    std_y = np.std(y_mm)
    euclidean_errors = np.sqrt(x_mm**2 + y_mm**2)
    mean_euclidean = np.mean(euclidean_errors).item()

    # Mark mean position
    ax.scatter(
        [mean_x],
        [mean_y],
        c="orange",
        s=100,
        marker="x",
        linewidths=2,
        zorder=5,
        label=f"Mean ({mean_x:.2f}, {mean_y:.2f})",
    )

    # Add circle showing mean euclidean error
    circle = plt.Circle(  # pyright: ignore[reportPrivateImportUsage]
        (0, 0),
        mean_euclidean,
        fill=False,
        color="green",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean Euclidean Error: {mean_euclidean:.2f} mm",
    )
    ax.add_patch(circle)

    # Make axes equal and centered
    max_range = max(np.max(np.abs(x_mm)), np.max(np.abs(y_mm))) * 1.2
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_aspect("equal")

    # Labels and title
    ax.set_xlabel("X Error (mm)", fontsize=12)
    ax.set_ylabel("Y Error (mm)", fontsize=12)
    ax.set_title("XY Goal Position Error Distribution", fontsize=14, fontweight="bold")

    # Calculate z statistics
    mean_z = np.mean(z_mm)
    std_z = np.std(z_mm)

    # Add statistics text box
    stats_text = (
        f"N = {len(x_mm)}\n"
        f"Mean X: {mean_x:.3f} mm\n"
        f"Mean Y: {mean_y:.3f} mm\n"
        f"Mean Z: {mean_z:.3f} mm\n"
        f"Std X: {std_x:.3f} mm\n"
        f"Std Y: {std_y:.3f} mm\n"
        f"Std Z: {std_z:.3f} mm\n"
        f"Mean Euclidean (XY): {mean_euclidean:.3f} mm"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.legend(loc="upper right")
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
