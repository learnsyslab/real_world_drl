#!/usr/bin/env python3
"""
Script to compute evaluation metrics from a JSONL file containing episode data.

Computes:
- Average successful episode length (mean and stddev)
- Success rate (mean and stddev)
- Plots comparing goal position offsets for successful vs failed cases
- Plot showing perfect actions for failed cases
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np


def load_data(jsonl_path: Path) -> list[dict]:
    """Load data from a JSONL file."""
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def compute_metrics(data: list[dict]) -> dict:
    """Compute evaluation metrics from the data."""
    successful_episodes = [ep for ep in data if ep["episode_successful"]]
    failed_episodes = [ep for ep in data if not ep["episode_successful"]]

    # Success rate
    total_episodes = len(data)
    num_successful = len(successful_episodes)
    success_rate = num_successful / total_episodes if total_episodes > 0 else 0.0

    # For success rate stddev, we treat each episode as a Bernoulli trial
    # Variance of Bernoulli = p * (1 - p), stddev = sqrt(p * (1 - p) / n)
    success_rate_stddev = (
        np.sqrt(success_rate * (1 - success_rate) / total_episodes)
        if total_episodes > 0
        else 0.0
    )

    # Successful episode lengths
    successful_lengths = [ep["length"] for ep in successful_episodes if "length" in ep]
    avg_successful_length = np.mean(successful_lengths) if successful_lengths else 0.0
    successful_length_stddev = np.std(successful_lengths) if successful_lengths else 0.0

    return {
        "total_episodes": total_episodes,
        "num_successful": num_successful,
        "num_failed": len(failed_episodes),
        "success_rate": success_rate,
        "success_rate_stddev": success_rate_stddev,
        "avg_successful_length": avg_successful_length,
        "successful_length_stddev": successful_length_stddev,
    }


def extract_goal_offsets(data: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Extract goal position offsets for successful and failed episodes."""
    successful_offsets = []
    failed_offsets = []

    for ep in data:
        offset = ep["reset.goal_position.offset"]
        if ep["episode_successful"]:
            successful_offsets.append(offset)
        else:
            failed_offsets.append(offset)

    return np.array(successful_offsets), np.array(failed_offsets)


def extract_failed_perfect_actions(data: list[dict]) -> np.ndarray:
    """Extract perfect actions for failed episodes."""
    failed_actions = []
    for ep in data:
        if not ep["episode_successful"]:
            failed_actions.append(ep["observation.perfect_action"])
    return np.array(failed_actions)


def extract_grasped_deltas(data: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Extract grasped delta offsets for successful and failed episodes."""
    successful_deltas = []
    failed_deltas = []

    for ep in data:
        delta = ep["reset.grasped.delta"]
        if ep["episode_successful"]:
            successful_deltas.append(delta)
        else:
            failed_deltas.append(delta)

    return np.array(successful_deltas), np.array(failed_deltas)


def plot_goal_offsets_2d(
    successful_offsets: np.ndarray,
    failed_offsets: np.ndarray,
    output_dir: Path,
    show_plot: bool = False,
):
    """Create a 2D scatter plot of goal position offsets (X vs Y)."""
    fig, ax = plt.subplots(figsize=(10, 8))

    if len(successful_offsets) > 0:
        ax.scatter(
            successful_offsets[:, 0] * 1000,  # Convert to mm
            successful_offsets[:, 1] * 1000,
            c="green",
            alpha=0.6,
            label=f"Successful (n={len(successful_offsets)})",
            s=50,
        )

    if len(failed_offsets) > 0:
        ax.scatter(
            failed_offsets[:, 0] * 1000,
            failed_offsets[:, 1] * 1000,
            c="red",
            alpha=0.8,
            label=f"Failed (n={len(failed_offsets)})",
            s=100,
            marker="x",
        )

    # Compute max absolute value for centered limits
    all_offsets = np.vstack(
        [successful_offsets, failed_offsets]
        if len(successful_offsets) > 0 and len(failed_offsets) > 0
        else [successful_offsets]
        if len(successful_offsets) > 0
        else [failed_offsets]
        if len(failed_offsets) > 0
        else [np.array([[0, 0]])]
    )
    max_val = np.max(np.abs(all_offsets)) * 1.1 * 1000
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    ax.set_xlabel("X Offset (mm)", fontsize=12)
    ax.set_ylabel("Y Offset (mm)", fontsize=12)
    ax.set_title("Goal Position Offsets: Successful vs Failed Episodes", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    ax.axvline(x=0, color="k", linestyle="-", linewidth=0.5)
    ax.set_aspect("equal")

    plt.tight_layout()
    output_path = output_dir / "goal_offsets_2d.png"
    plt.savefig(output_path, dpi=150)
    if show_plot:
        plt.show()
    plt.close()
    print(f"Saved: {output_path}")


def plot_grasped_deltas_xz(
    successful_deltas: np.ndarray,
    failed_deltas: np.ndarray,
    output_dir: Path,
    show_plot: bool = False,
):
    """Create a 2D scatter plot of grasped delta offsets (X vs Z)."""
    fig, ax = plt.subplots(figsize=(10, 8))

    if len(successful_deltas) > 0:
        ax.scatter(
            successful_deltas[:, 0] * 1000,  # Convert to mm
            successful_deltas[:, 2] * 1000,  # Z axis (index 2)
            c="green",
            alpha=0.6,
            label=f"Successful (n={len(successful_deltas)})",
            s=50,
        )

    if len(failed_deltas) > 0:
        ax.scatter(
            failed_deltas[:, 0] * 1000,
            failed_deltas[:, 2] * 1000,  # Z axis (index 2)
            c="red",
            alpha=0.8,
            label=f"Failed (n={len(failed_deltas)})",
            s=100,
            marker="x",
        )

    # Compute max absolute value for centered limits
    all_deltas = np.vstack(
        [successful_deltas, failed_deltas]
        if len(successful_deltas) > 0 and len(failed_deltas) > 0
        else [successful_deltas]
        if len(successful_deltas) > 0
        else [failed_deltas]
        if len(failed_deltas) > 0
        else [np.array([[0, 0, 0]])]
    )
    max_val_x = np.max(all_deltas[:, 0]) * 1000
    min_val_x = np.min(all_deltas[:, 0]) * 1000
    max_val_z = np.max(all_deltas[:, 2]) * 1000
    min_val_z = np.min(all_deltas[:, 2]) * 1000
    delta_x = max_val_x - min_val_x
    delta_z = max_val_z - min_val_z
    ax.set_xlim(min_val_x - delta_x * 0.1, max_val_x + delta_x * 0.1)
    ax.set_ylim(min_val_z - delta_z * 0.1, max_val_z + delta_z * 0.1)

    ax.set_xlabel("X Delta (mm)", fontsize=12)
    ax.set_ylabel("Z Delta (mm)", fontsize=12)
    ax.set_title(
        "Grasped Delta Offsets (X vs Z): Successful vs Failed Episodes", fontsize=14
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    ax.axvline(x=0, color="k", linestyle="-", linewidth=0.5)
    ax.set_aspect("equal")

    # Draw ellipse at x=0, y=1 with width=3 and height=1
    ellipse = Ellipse(
        xy=(0, 1), width=3, height=1, fill=False, edgecolor="orange", linestyle="--"
    )
    ax.add_patch(ellipse)

    plt.tight_layout()
    output_path = output_dir / "grasped_deltas_xz.png"
    plt.savefig(output_path, dpi=150)
    if show_plot:
        plt.show()
    plt.close()
    print(f"Saved: {output_path}")


def plot_episode_length_vs_position(
    data: list[dict], output_dir: Path, show_plot: bool = False
):
    """Create a scatter plot of successful episodes' starting positions colored by episode length.

    X and Y axes show goal position offsets, points are colored by episode length.
    """
    successful_episodes = [ep for ep in data if ep["episode_successful"]]

    if len(successful_episodes) == 0:
        print("No successful episodes to plot.")
        return

    # Extract starting positions and lengths
    x_positions = []
    y_positions = []
    lengths = []

    for ep in successful_episodes:
        starting_pos = ep["reset.goal_position.offset"]
        x_positions.append(starting_pos[0] * 1000)  # Convert to mm
        y_positions.append(starting_pos[1] * 1000)  # Convert to mm
        lengths.append(ep["length"])

    x_positions = np.array(x_positions)
    y_positions = np.array(y_positions)
    lengths = np.array(lengths)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create scatter plot with color map based on episode length
    scatter = ax.scatter(
        x_positions,
        y_positions,
        c=lengths,
        cmap="viridis",
        s=100,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )

    # Compute max absolute value for centered limits
    max_val = np.max(np.abs(np.concatenate([x_positions, y_positions]))) * 1.1
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Episode Length", fontsize=12)

    ax.set_xlabel("X Offset (mm)", fontsize=12)
    ax.set_ylabel("Y Offset (mm)", fontsize=12)
    ax.set_title(
        f"Successful Episodes: Starting Position with Episode Length Coloring (n={len(successful_episodes)})",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    ax.axvline(x=0, color="k", linestyle="-", linewidth=0.5)
    ax.set_aspect("equal")

    plt.tight_layout()
    output_path = output_dir / "episode_length_vs_position.png"
    plt.savefig(output_path, dpi=150)
    if show_plot:
        plt.show()
    plt.close()
    print(f"Saved: {output_path}")


def plot_failed_perfect_actions(
    failed_actions: np.ndarray, output_dir: Path, show_plot: bool = False
):
    """Create a 2D scatter plot showing perfect actions (X vs Y) for failed cases."""
    if len(failed_actions) == 0:
        print("No failed episodes to plot perfect actions for.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(
        failed_actions[:, 0],
        failed_actions[:, 1],
        c="red",
        alpha=0.8,
        s=100,
        marker="x",
        label=f"Failed (n={len(failed_actions)})",
    )

    # Add mean point
    mean_x = np.mean(failed_actions[:, 0])
    mean_y = np.mean(failed_actions[:, 1])
    ax.scatter(
        mean_x,
        mean_y,
        c="blue",
        s=200,
        marker="o",
        edgecolors="black",
        linewidths=2,
        label=f"Mean: ({mean_x:.6f}, {mean_y:.6f})",
        zorder=5,
    )

    # Compute max absolute value for centered limits
    max_val = np.max(np.max(np.abs(failed_actions[:, :2]))) * 1.1
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    ax.set_xlabel("Perfect Action X", fontsize=12)
    ax.set_ylabel("Perfect Action Y", fontsize=12)
    ax.set_title("Perfect Actions for Failed Episodes (X vs Y)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    ax.axvline(x=0, color="k", linestyle="-", linewidth=0.5)
    ax.set_aspect("equal")

    plt.tight_layout()
    output_path = output_dir / "failed_perfect_actions.png"
    plt.savefig(output_path, dpi=150)
    if show_plot:
        plt.show()
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute evaluation metrics from episode data JSONL file."
    )
    parser.add_argument(
        "jsonl_path",
        type=str,
        help="Path to the JSONL file containing episode evaluation data.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots. Defaults to the same directory as the input file.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show plots in an interactive matplotlib window.",
    )

    args = parser.parse_args()

    jsonl_path = Path(args.jsonl_path)
    if not jsonl_path.exists():
        print(f"Error: File not found: {jsonl_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else jsonl_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {jsonl_path}")
    data = load_data(jsonl_path)
    print(f"Loaded {len(data)} episodes.")

    # Compute metrics
    metrics = compute_metrics(data)

    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    print(f"Total Episodes:           {metrics['total_episodes']}")
    print(f"Successful Episodes:      {metrics['num_successful']}")
    print(f"Failed Episodes:          {metrics['num_failed']}")
    print("-" * 60)
    print(
        f"Success Rate:             {metrics['success_rate']:.4f} ± {metrics['success_rate_stddev']:.4f}"
    )
    print(
        f"                          ({metrics['success_rate'] * 100:.2f}% ± {metrics['success_rate_stddev'] * 100:.2f}%)"
    )
    print("-" * 60)
    print(
        f"Avg Successful Length:    {metrics['avg_successful_length']:.2f} ± {metrics['successful_length_stddev']:.2f}"
    )
    print("=" * 60 + "\n")

    # Extract data for plots
    successful_offsets, failed_offsets = extract_goal_offsets(data)
    successful_deltas, failed_deltas = extract_grasped_deltas(data)
    failed_actions = extract_failed_perfect_actions(data)

    # Create plots
    print("Generating plots...")
    plot_goal_offsets_2d(successful_offsets, failed_offsets, output_dir, args.plot)
    plot_grasped_deltas_xz(successful_deltas, failed_deltas, output_dir, args.plot)
    plot_episode_length_vs_position(data, output_dir, args.plot)
    plot_failed_perfect_actions(failed_actions, output_dir, args.plot)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
