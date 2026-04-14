"""Plot success rate vs UTD (pretrain passes) varying two hyperparameters."""

import json
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
SEEDS = [1, 2, 3]
PASSES = {
    "1cft_d200v9_15": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "1cft_d300v9_15": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "1cft_d300v9_20": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "1cft_d300v9_20_75": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "1cft_d400v9_15": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "1cft_d400v9_20": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "1cft_d400v9_20_75": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "1cft_d500v9_20": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "1cft_d500v9_20_75": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "1cft_d400v9_25": [0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5, 6, 7, 8],
    "1cft_d400v9_25_60": [1, 2, 3, 4, 5, 6, 7, 8],
    "1cft_d500v9_25": [0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5, 6, 7, 8],
    "1cft_d500v9_25_60": [1, 2, 3, 4, 5, 6, 7, 8],
    "1cft_d600v9_25": [0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5, 6, 7, 8],
    "1cft_d600v9_25_60": [1, 2, 3, 4, 5, 6, 7, 8],
    "1cft_d500v9_30": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6],
    "1cft_d500v9_30_40": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "1cft_d600v9_30": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6],
    "1cft_d600v9_30_40": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "1cft_d700v9_30": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6],
    "1cft_d700v9_30_40": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "1cft_d800v9_30": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6],
    "1cft_d800v9_30_40": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}
GROUPS = {
    "15": ["1cft_d200v9_15", "1cft_d300v9_15", "1cft_d400v9_15"],
    "20": ["1cft_d300v9_20", "1cft_d400v9_20", "1cft_d500v9_20"],
    "20_75": ["1cft_d300v9_20_75", "1cft_d400v9_20_75", "1cft_d500v9_20_75"],
    "25": ["1cft_d400v9_25", "1cft_d500v9_25", "1cft_d600v9_25"],
    "25_60": ["1cft_d400v9_25_60", "1cft_d500v9_25_60", "1cft_d600v9_25_60"],
    "30": ["1cft_d500v9_30", "1cft_d600v9_30", "1cft_d700v9_30", "1cft_d800v9_30"],
    "30_40": [
        "1cft_d500v9_30_40",
        "1cft_d600v9_30_40",
        "1cft_d700v9_30_40",
        "1cft_d800v9_30_40",
    ],
}
YLIMS = {
    "15": (0.95, 1.05),
    "20": (0.9, 1.05),
    "20_75": (0.9, 1.05),
    "25": (0.9, 1.05),
    "25_60": (0.9, 1.05),
    "30": (0.9, 1.05),
    "30_40": (0.9, 1.05),
}


def load_success_rate(eval_path: Path) -> float | None:
    """Compute success rate from an eval_infos.jsonl file."""
    successes = []
    if not os.path.exists(eval_path):
        print(f"Warning: Does not exist: {eval_path}")
        return -1
    with open(eval_path) as f:
        for line in f:
            info = json.loads(line)
            if info.get("datetime", "") <= "2026-03-02 00:00:00":
                continue
            successes.append(info["episode_successful"])
    if len(successes) == 0:
        print(f"Warning: No recent evals found in {eval_path}")
        # raise ValueError(f"No recent evals found in {eval_path}")
        return -1
    return np.mean(successes)  # pyright: ignore[reportReturnType]


def plot(
    group_name: str,
    hp1_values: list[str],
    ylims: tuple[float, float],
    filename: str,
):
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10  # pyright: ignore[reportAttributeAccessIssue]

    all_passes: set[int] = set()
    for hp1 in hp1_values:
        all_passes.update(PASSES[hp1])

    for i, hp1 in enumerate(hp1_values):
        color = cmap(i / len(hp1_values))
        hp_passes = PASSES[hp1]
        rates = np.full((len(hp_passes), len(SEEDS)), np.nan)

        for j, p in enumerate(hp_passes):
            for k, seed in enumerate(SEEDS):
                dir_name = f"{hp1}_s{seed}"
                eval_path = (
                    CHECKPOINTS_DIR
                    / dir_name
                    / (f"pretrain_{p}" if type(p) is int else f"pretrain_{p:.2f}")
                    / "eval_infos.jsonl"
                )
                sr = load_success_rate(eval_path)
                if sr is not None:
                    rates[j, k] = sr

        if np.all(np.isnan(rates)):
            continue

        mean = np.nanmean(rates, axis=1)
        std = np.nanstd(rates, axis=1)

        valid = ~np.isnan(mean)
        x = np.array(hp_passes)[valid]
        m = mean[valid]
        s = std[valid]

        ax.plot(x, m, marker="o", label=hp1, color=color)
        ax.fill_between(x, m - s, m + s, alpha=0.2, color=color)

    ax.set_xlabel("UTD (pretrain passes)")
    ax.set_ylabel("Success Rate")
    ax.set_title(f"Success Rate vs UTD (group={group_name})")
    ax.set_xticks(sorted(all_passes))
    ax.legend(loc="best", fontsize="small")
    ax.set_ylim(*ylims)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHECKPOINTS_DIR.parent / "plots" / filename, dpi=150)


def main():
    for group_name, hp1_values in GROUPS.items():
        plot(
            group_name,
            hp1_values,
            YLIMS[group_name],
            f"pretrain_success_{group_name}.png",
        )


if __name__ == "__main__":
    main()
