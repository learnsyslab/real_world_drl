"""Plot success rate vs pretrain passes for 1cft3d dataset-size experiments."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
PLOTS_DIR = REPO_ROOT / "plots"

EXPERIMENT_PATTERN = re.compile(r"^1cft3d_d(?P<dataset_size>\d+)v9_s(?P<seed>\d+)$")
PASSES = list(range(1, 9))
PLOT_EXCLUDED_DATASET_SIZES = {300}


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


def discover_experiments() -> dict[int, list[int]]:
    """Map dataset size to available seeds by scanning checkpoint directories."""
    datasets_to_seeds: dict[int, set[int]] = {}

    for path in CHECKPOINTS_DIR.iterdir():
        if not path.is_dir():
            continue
        match = EXPERIMENT_PATTERN.match(path.name)
        if match is None:
            continue

        dataset_size = int(match.group("dataset_size"))
        seed = int(match.group("seed"))
        datasets_to_seeds.setdefault(dataset_size, set()).add(seed)

    return {
        dataset_size: sorted(seeds)
        for dataset_size, seeds in sorted(datasets_to_seeds.items())
    }


def build_dataset_color_map() -> dict[int, str]:
    """Map each dataset size (e.g. d300) to a consistent color."""
    # Keep the first four colors and use more distinct colors for higher dataset sizes.
    return {
        200: "tab:blue",
        300: "tab:orange",
        400: "tab:green",
        500: "tab:red",
        600: "tab:purple",
        700: "tab:cyan",
        800: "tab:brown",
        2000: "tab:blue",
    }


def build_latex_summary_table(
    doa: int, rows: list[tuple[int, float, float, int]]
) -> str:
    """Build a LaTeX table of max mean success rate and corresponding stats."""
    rows = sorted(rows, key=lambda row: row[0])
    lines = [
        r"\begin{table}[htpb]",
        r"\centering",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"DoA & $\left|\mathcal{D}\right|$ & Max. Mean \ac{SR} & Corresp. Std. & Corresp. \ac{UTD} \\",
        r"\hline",
    ]

    if rows:
        lines.append(
            rf"\multirow{{{len(rows)}}}{{*}}{{{doa}}} & {rows[0][0]} & {rows[0][1]:.4f} & {rows[0][2]:.4f} & {rows[0][3]} \\",
        )
        for dataset_size, max_mean, corr_std, corr_utd in rows[1:]:
            lines.append(
                f" & {dataset_size} & {max_mean:.3f} & {corr_std:.3f} & {corr_utd} \\\\",
            )

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            rf"\caption{{Maximum mean success rate over \ac{{UTD}} for {doa}\ac{{DoA}} with corresponding standard deviation and \ac{{UTD}}.}}",
            rf"\label{{tab:doa{doa}_sr}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def plot_success_vs_passes(
    datasets_to_seeds: dict[int, list[int]],
) -> tuple[Path, Path]:
    """Plot mean and std success rate over pretrain passes for each dataset size."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOTS_DIR / "1cft3d_pretrain_success.png"
    table_path = PLOTS_DIR / "1cft3d_pretrain_success_table.tex"

    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.cm.tab10  # pyright: ignore[reportAttributeAccessIssue]
    dataset_color_map = build_dataset_color_map()
    summary_rows: list[tuple[int, float, float, int]] = []

    for i, (dataset_size, seeds) in enumerate(datasets_to_seeds.items()):
        rates = np.full((len(PASSES), len(seeds)), np.nan)

        for j, pretrain_pass in enumerate(PASSES):
            for k, seed in enumerate(seeds):
                run_name = f"1cft3d_d{dataset_size}v9_s{seed}"
                eval_path = (
                    CHECKPOINTS_DIR
                    / run_name
                    / f"pretrain_{pretrain_pass}"
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

        best_idx = int(np.nanargmax(y))
        summary_rows.append(
            (
                dataset_size,
                float(y[best_idx]),
                float(yerr[best_idx]),
                int(x[best_idx]),
            )
        )

        if dataset_size in PLOT_EXCLUDED_DATASET_SIZES:
            continue

        color = dataset_color_map.get(dataset_size, cmap(i % 10))
        ax.plot(
            x,
            y,
            marker="o",
            markersize=4.5,
            color=color,
            label=f"$\\left|\\mathcal{{D}}\\right|={dataset_size}$",
        )
        ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)

    ax.set_xlabel("UTD (epochs)")
    ax.set_ylabel("Success Rate")
    ax.set_title("3 DoA, Success Rate vs UTD", fontsize=16)
    ax.set_xticks(PASSES)
    ax.set_xlim(PASSES[0], PASSES[-1])
    ax.set_xticklabels(
        [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
        ]
    )
    ax.set_ylim(0.945, 1.005)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)

    table_tex = build_latex_summary_table(doa=3, rows=summary_rows)
    table_path.write_text(table_tex + "\n")
    print(table_tex)

    return output_path, table_path


def main() -> None:
    datasets_to_seeds = discover_experiments()
    if not datasets_to_seeds:
        raise RuntimeError(
            "No matching experiments found for pattern '1cft3d_d{datasetsize}v9_s{seed}'."
        )

    output_path, table_path = plot_success_vs_passes(datasets_to_seeds)
    print(f"Saved plot to {output_path}")
    print(f"Saved table to {table_path}")


if __name__ == "__main__":
    main()
