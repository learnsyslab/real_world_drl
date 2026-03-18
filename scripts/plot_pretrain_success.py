"""Plot success rate vs UTD (pretrain passes) for all hyperparameter settings."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
HYPERPARAMETERS = ["0", "3", "6", "7", "75", "8", "85", "9", "95", "100"]
SEEDS = [1, 2, 3]
PASSES = {
    "0": [1, 3, 5, 10, 15, 20, 25, 30],
    "3": [1, 3, 5, 10, 15, 20, 25, 30],
    "6": [1, 3, 5, 10, 15, 20, 25, 30],
    "7": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30],
    "75": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30],
    "8": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30],
    "85": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30],
    "9": [1, 3, 5, 10, 15, 20, 25, 30],
    "95": [1, 3, 5, 10, 15, 20, 25, 30],
    "100": [1, 3, 5, 10, 15, 20, 25, 30],
}
BASE_NAME = "1cam_ft_128_16_full_d300v9"


def load_success_rate(eval_path: Path) -> float | None:
    """Compute success rate from an eval_infos.jsonl file."""
    if not eval_path.exists():
        return None
    successes = []
    with open(eval_path) as f:
        for line in f:
            info = json.loads(line)
            successes.append(info["episode_successful"])
    if not successes:
        return None
    return np.mean(successes)  # pyright: ignore[reportReturnType]


def plot(hyperparameters, passes, title, filename, ylim=(-0.05, 1.05)):
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10  # pyright: ignore[reportAttributeAccessIssue]

    for i, hp in enumerate(hyperparameters):
        color = cmap(i / len(hyperparameters))
        hp_passes = passes[hp]
        # Collect success rates: shape (len(hp_passes), len(SEEDS))
        rates = np.full((len(hp_passes), len(SEEDS)), np.nan)

        for j, p in enumerate(hp_passes):
            for k, seed in enumerate(SEEDS):
                dir_name = f"{BASE_NAME}_{hp}_s{seed}"
                eval_path = (
                    CHECKPOINTS_DIR / dir_name / f"pretrain_{p}" / "eval_infos.jsonl"
                )
                sr = load_success_rate(eval_path)
                if sr is not None:
                    rates[j, k] = sr

        # Skip hyperparameter if no data at all
        if np.all(np.isnan(rates)):
            continue

        mean = np.nanmean(rates, axis=1)
        std = np.nanstd(rates, axis=1)

        valid = ~np.isnan(mean)
        x = np.array(hp_passes)[valid]
        m = mean[valid]
        s = std[valid]

        ax.plot(x, m, marker="o", label=f"hp={hp}", color=color)
        ax.fill_between(x, m - s, m + s, alpha=0.2, color=color)

    ax.set_xlabel("UTD (pretrain passes)")
    ax.set_ylabel("Success Rate")
    ax.set_title(title)
    all_passes = sorted(set(p for ps in passes.values() for p in ps))
    ax.set_xticks(all_passes)
    ax.legend(loc="best", fontsize="small")
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHECKPOINTS_DIR.parent / "scripts" / filename, dpi=150)
    # plt.show()


def main():
    # Plot 1: all hyperparameters, all passes
    plot(
        HYPERPARAMETERS,
        PASSES,
        "Success Rate vs UTD by Hyperparameter",
        "pretrain_success.png",
    )

    # Plot 2: subset of hyperparameters, UTD >= 3
    plot(
        ["6", "7", "75", "8", "85"],
        {hp: [p for p in PASSES[hp] if p >= 1] for hp in ["6", "7", "75", "8", "85"]},
        "Success Rate vs UTD (hp ∈ {6,7,75,8,85})",
        "pretrain_success_subset.png",
        ylim=(0.9, 1.05),
    )

    # Plot 3: subset of hyperparameters, UTD >= 3, different y-axis range
    plot(
        ["75", "8"],
        {hp: [p for p in PASSES[hp] if p >= 1] for hp in ["75", "8"]},
        "Success Rate vs UTD (hp ∈ {75,8})",
        "pretrain_success_subset_small.png",
        ylim=(0.98, 1.02),
    )


if __name__ == "__main__":
    main()
