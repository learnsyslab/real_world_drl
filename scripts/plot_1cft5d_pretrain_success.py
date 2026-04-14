"""Plot success rate vs pretrain passes for 1cft5d dataset-size experiments."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
PLOTS_DIR = REPO_ROOT / "plots"

EXPERIMENT_PATTERN = re.compile(r"^1cft5d_d(?P<dataset_size>\d+)v9_s(?P<seed>\d+)$")
HALFROT_EXPERIMENT_PATTERN = re.compile(r"^1cft5d_halfrot_d2000v9_s(?P<seed>\d+)$")
PASSES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + list(range(1, 11))


def load_success_rate(eval_path: Path) -> float | None:
    """Compute success rate from an eval_infos.jsonl file."""
    if not eval_path.exists():
        return None

    successes = []
    with eval_path.open() as f:
        for line in f:
            info = json.loads(line)
            successes.append(info["episode_successful"])

    if not successes:
        return None
    return float(np.mean(successes))


def discover_experiments() -> tuple[dict[int, list[int]], list[int]]:
    """Map base and halfrot experiments to available seeds by scanning checkpoint directories."""
    datasets_to_seeds: dict[int, set[int]] = {}
    halfrot_seeds: set[int] = set()

    for path in CHECKPOINTS_DIR.iterdir():
        if not path.is_dir():
            continue

        halfrot_match = HALFROT_EXPERIMENT_PATTERN.match(path.name)
        if halfrot_match is not None:
            halfrot_seeds.add(int(halfrot_match.group("seed")))
            continue

        match = EXPERIMENT_PATTERN.match(path.name)
        if match is None:
            continue

        dataset_size = int(match.group("dataset_size"))
        seed = int(match.group("seed"))
        datasets_to_seeds.setdefault(dataset_size, set()).add(seed)

    return (
        {
            dataset_size: sorted(seeds)
            for dataset_size, seeds in sorted(datasets_to_seeds.items())
        },
        sorted(halfrot_seeds),
    )


def plot_success_vs_passes(
    datasets_to_seeds: dict[int, list[int]], halfrot_seeds: list[int]
) -> Path:
    """Plot mean and std success rate over pretrain passes for each dataset size."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path_zoomed = PLOTS_DIR / "1cft5d_pretrain_success_zoomed.png"
    output_path_full = PLOTS_DIR / "1cft5d_pretrain_success_full.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10  # pyright: ignore[reportAttributeAccessIssue]

    for i, (dataset_size, seeds) in enumerate(datasets_to_seeds.items()):
        rates = np.full((len(PASSES), len(seeds)), np.nan)

        for j, pretrain_pass in enumerate(PASSES):
            for k, seed in enumerate(seeds):
                run_name = f"1cft5d_d{dataset_size}v9_s{seed}"
                eval_path = (
                    CHECKPOINTS_DIR
                    / run_name
                    / (
                        f"pretrain_{pretrain_pass}"
                        if type(pretrain_pass) is int
                        else f"pretrain_{pretrain_pass:.2f}"
                    )
                    / "eval_infos.jsonl"
                )
                success_rate = load_success_rate(eval_path)
                if success_rate is not None:
                    rates[j, k] = success_rate

        if np.all(np.isnan(rates)):
            continue

        mean = np.nanmean(rates, axis=1)
        std = np.nanstd(rates, axis=1)
        valid = ~np.isnan(mean)

        x = np.array(PASSES)[valid]
        y = mean[valid]
        yerr = std[valid]

        color = cmap(i % 10)
        ax.plot(x, y, marker="o", color=color, label=f"dataset={dataset_size}")
        ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)

    if halfrot_seeds:
        rates = np.full((len(PASSES), len(halfrot_seeds)), np.nan)

        for j, pretrain_pass in enumerate(PASSES):
            for k, seed in enumerate(halfrot_seeds):
                run_name = f"1cft5d_halfrot_d2000v9_s{seed}"
                eval_path = (
                    CHECKPOINTS_DIR
                    / run_name
                    / (
                        f"pretrain_{pretrain_pass}"
                        if type(pretrain_pass) is int
                        else f"pretrain_{pretrain_pass:.2f}"
                    )
                    / "eval_infos.jsonl"
                )
                success_rate = load_success_rate(eval_path)
                if success_rate is not None:
                    rates[j, k] = success_rate

        if not np.all(np.isnan(rates)):
            mean = np.nanmean(rates, axis=1)
            std = np.nanstd(rates, axis=1)
            valid = ~np.isnan(mean)

            x = np.array(PASSES)[valid]
            y = mean[valid]
            yerr = std[valid]

            color = cmap(len(datasets_to_seeds) % 10)
            ax.plot(x, y, marker="o", color=color, label="halfrot dataset=2000")
            ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)

    ax.set_xlabel("Pretrain Passes")
    ax.set_ylabel("Success Rate")
    ax.set_title("1cft5d: Success Rate vs Pretrain Passes")
    ax.set_xticks(PASSES)
    ax.set_ylim(0.95, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")

    fig.tight_layout()
    fig.savefig(output_path_zoomed, dpi=150)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(output_path_full, dpi=150)
    return output_path_zoomed


def main() -> None:
    datasets_to_seeds, halfrot_seeds = discover_experiments()
    if not datasets_to_seeds and not halfrot_seeds:
        raise RuntimeError(
            "No matching experiments found for pattern '1cft5d_d{datasetsize}v9_s{seed}' or '1cft5d_halfrot_d2000v9_s{seed}'."
        )

    output_path = plot_success_vs_passes(datasets_to_seeds, halfrot_seeds)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
