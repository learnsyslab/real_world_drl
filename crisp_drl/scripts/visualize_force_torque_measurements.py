"""Visualize force/torque observation sequences over time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COMPONENT_NAMES = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
COMPONENT_LABELS = [
    "Force X",
    "Force Y",
    "Force Z",
    "Torque X",
    "Torque Y",
    "Torque Z",
]


def load_records(jsonl_path: Path) -> list[dict]:
    """Load all JSONL records from a force/torque measurement log."""
    records: list[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records[-1:]


def extract_episode_observations(record: dict, episode_index: int) -> np.ndarray:
    """Convert one JSONL record into an (N, 6) observation array."""
    observations = np.asarray(record.get("observations", []), dtype=np.float32)
    if observations.size == 0:
        return observations.reshape(0, 6)

    if observations.ndim != 2 or observations.shape[1] != 6:
        raise ValueError(
            f"Record {episode_index} has shape {observations.shape}, expected (N, 6)."
        )
    return observations


def build_timeline(
    records: list[dict], episode_gap: int = 15
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str | None]]]:
    """Stack episodes into one timeline with gaps between records."""
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    episode_boundaries: list[tuple[int, str | None]] = []

    offset = 0
    for episode_index, record in enumerate(records):
        observations = extract_episode_observations(record, episode_index)
        if observations.shape[0] == 0:
            continue

        x_values = np.arange(observations.shape[0], dtype=np.int32) + offset
        all_x.append(x_values)
        all_y.append(observations)
        episode_boundaries.append((offset, record.get("datetime")))

        if episode_index < len(records) - 1:
            separator_x = np.array([int(x_values[-1]) + 1], dtype=np.int32)
            separator_y = np.full((1, 6), np.nan, dtype=np.float32)
            all_x.append(separator_x)
            all_y.append(separator_y)

        offset = int(x_values[-1]) + episode_gap + 1

    if not all_x:
        return np.empty(0, dtype=np.int32), np.empty((0, 6), dtype=np.float32), []

    return np.concatenate(all_x), np.concatenate(all_y, axis=0), episode_boundaries


def plot_force_torque_timeline(
    x_values: np.ndarray,
    observations: np.ndarray,
    episode_boundaries: list[tuple[int, str | None]],
    output_path: Path | None = None,
    show: bool = True,
) -> None:
    """Plot the six force/torque channels over the measurement timeline."""
    fig, axes = plt.subplots(6, 1, sharex=True, figsize=(14, 12))
    color_cycle = plt.cm.tab10  # pyright: ignore[reportAttributeAccessIssue]

    for component_index, axis in enumerate(axes):
        axis.plot(
            x_values,
            observations[:, component_index],
            color=color_cycle(component_index % color_cycle.N),
            linewidth=1.2,
            label=COMPONENT_NAMES[component_index],
        )
        axis.set_ylabel(COMPONENT_LABELS[component_index])
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", fontsize="small", framealpha=0.9)

        for boundary, _ in episode_boundaries[1:]:
            axis.axvline(
                boundary - 0.5,
                color="gray",
                linestyle="--",
                linewidth=0.8,
                alpha=0.6,
            )

    title_parts = ["Force/Torque observations over time"]
    if len(episode_boundaries) == 1 and episode_boundaries[0][1]:
        title_parts.append(f"({episode_boundaries[0][1]})")
    fig.suptitle(" ".join(title_parts), fontsize=16)

    axes[-1].set_xlabel("Step")
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Saved plot to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize force/torque measurements from a checkpoint JSONL file."
    )
    parser.add_argument(
        "jsonl_path",
        nargs="?",
        default=Path("checkpoints/test_box_1/force_torque_measurements.jsonl"),
        type=Path,
        help="Path to the force_torque_measurements.jsonl file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. Defaults to the input path with a .png suffix.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the figure without opening an interactive window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jsonl_path: Path = args.jsonl_path

    if not jsonl_path.exists():
        raise FileNotFoundError(f"File not found: {jsonl_path}")

    records = load_records(jsonl_path)
    if not records:
        raise ValueError(f"No JSONL records found in {jsonl_path}")

    x_values, observations, episode_boundaries = build_timeline(records)
    if observations.size == 0:
        raise ValueError(f"No force/torque observations found in {jsonl_path}")

    output_path = args.output
    if output_path is None:
        output_path = jsonl_path.with_suffix(".png")

    plot_force_torque_timeline(
        x_values,
        observations,
        episode_boundaries,
        output_path=output_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
