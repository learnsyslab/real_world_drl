"""Plot success rate vs UTD (pretrain passes) varying two hyperparameters."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
HP1_VALUES = ["1c", "1cft", "2c"]
HP2_VALUES = [100, 150, 200, 300, 400]
SEEDS = [1, 2, 3]
PASSES = {
    100: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20],
    150: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20],
    200: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30],
    300: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20],
    400: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20],
}
YLIMS = {
    100: (-0.05, 1.05),
    150: (-0.05, 1.05),
    200: (0.5, 1.05),
    300: (0.9, 1.05),
    400: (0.9, 1.05),
}


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


def plot(
    hp2: int,
    hp1_values: list[str],
    passes: list[int],
    ylims: tuple[float, float],
    filename: str,
):
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10  # pyright: ignore[reportAttributeAccessIssue]

    for i, hp1 in enumerate(hp1_values):
        color = cmap(i / len(hp1_values))
        rates = np.full((len(passes), len(SEEDS)), np.nan)

        for j, p in enumerate(passes):
            for k, seed in enumerate(SEEDS):
                dir_name = f"{hp1}_d{hp2}v9_s{seed}"
                eval_path = (
                    CHECKPOINTS_DIR / dir_name / f"pretrain_{p}" / "eval_infos.jsonl"
                )
                sr = load_success_rate(eval_path)
                if sr is not None:
                    rates[j, k] = sr

        if np.all(np.isnan(rates)):
            continue

        mean = np.nanmean(rates, axis=1)
        std = np.nanstd(rates, axis=1)

        valid = ~np.isnan(mean)
        x = np.array(passes)[valid]
        m = mean[valid]
        s = std[valid]

        ax.plot(x, m, marker="o", label=f"hp1={hp1}", color=color)
        ax.fill_between(x, m - s, m + s, alpha=0.2, color=color)

    ax.set_xlabel("UTD (pretrain passes)")
    ax.set_ylabel("Success Rate")
    ax.set_title(f"Success Rate vs UTD (d{hp2}v9)")
    ax.set_xticks(passes)
    ax.legend(loc="best", fontsize="small")
    ax.set_ylim(*ylims)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHECKPOINTS_DIR.parent / "scripts" / filename, dpi=150)


def main():
    for hp2 in HP2_VALUES:
        plot(hp2, HP1_VALUES, PASSES[hp2], YLIMS[hp2], f"pretrain_success_d{hp2}.png")


if __name__ == "__main__":
    main()
