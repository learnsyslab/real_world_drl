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
# HALFROT_EXPERIMENT_PATTERN = re.compile(r"^1cft5d_halfrot_d2000v9_s(?P<seed>\d+)$")
PASSES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + list(range(1, 9))


def build_latex_summary_table(
    doa: int, rows: list[tuple[int, float, float, float | int]]
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
            rf"\multirow{{{len(rows)}}}{{*}}{{{doa}}} & {rows[0][0]} & {rows[0][1]:.3f} & {rows[0][2]:.3f} & {rows[0][3]} \\",
        )
        for dataset_size, max_mean, corr_std, corr_utd in rows[1:]:
            lines.append(
                f" & {dataset_size} & {max_mean:.3f} & {corr_std:.3f} & {corr_utd} \\\\",
            )

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{Maximum mean success rate over \ac{UTD} for 5\ac{DoA} with corresponding standard deviation and \ac{UTD}.}",
            r"\label{tab:doa5_sr}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


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

        # halfrot_match = HALFROT_EXPERIMENT_PATTERN.match(path.name)
        # if halfrot_match is not None:
        #     halfrot_seeds.add(int(halfrot_match.group("seed")))
        #     continue

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
) -> tuple[Path, Path]:
    """Plot mean and std success rate over pretrain passes for each dataset size."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path_full = PLOTS_DIR / "1cft5d_pretrain_success_full.png"
    table_path = PLOTS_DIR / "1cft5d_pretrain_success_table.tex"

    fig, ax = plt.subplots(figsize=(9, 6))
    dataset_color_map = build_dataset_color_map()
    cmap = plt.cm.tab10  # pyright: ignore[reportAttributeAccessIssue]
    summary_rows: list[tuple[int, float, float, float | int]] = []

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

        best_idx = int(np.nanargmax(y))
        x_best = x[best_idx].item()
        if np.allclose(x_best, int(round(x_best))):
            x_best = int(round(x_best))
        summary_rows.append(
            (
                dataset_size,
                float(y[best_idx]),
                float(yerr[best_idx]),
                x_best,
            )
        )

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

            best_idx = int(np.nanargmax(y))
            summary_rows.append(
                (
                    2000,
                    float(y[best_idx]),
                    float(yerr[best_idx]),
                    x[best_idx].item(),
                )
            )

            color = dataset_color_map.get(2000, cmap(len(datasets_to_seeds) % 10))
            ax.plot(x, y, marker="o", color=color, label="$\\mathcal{{D}}=2000$")
            ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)

    ax.set_xlabel("UTD (epochs)")
    ax.set_ylabel("Success Rate")
    ax.set_title("5DoA, Success Rate vs UTD", fontsize=16)
    ax.set_xticks(PASSES)
    ax.set_xticklabels(
        [
            "0.1",
            "",
            "",
            "",
            "0.5",
            "",
            "",
            "",
            "",
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
    ax.set_xlim(0, 8)
    # ax.set_ylim(0.945, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize="small")

    # fig.tight_layout()
    # fig.savefig(output_path_zoomed, dpi=150)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(output_path_full, dpi=150)
    summary_rows_by_dataset: dict[int, tuple[int, float, float, float | int]] = {}
    for row in summary_rows:
        dataset_size, max_mean, _, _ = row
        current = summary_rows_by_dataset.get(dataset_size)
        if current is None or max_mean > current[1]:
            summary_rows_by_dataset[dataset_size] = row

    table_tex = build_latex_summary_table(
        doa=5, rows=list(summary_rows_by_dataset.values())
    )
    table_path.write_text(table_tex + "\n")
    print(table_tex)

    return output_path_full, table_path


def main() -> None:
    datasets_to_seeds, halfrot_seeds = discover_experiments()
    if not datasets_to_seeds and not halfrot_seeds:
        raise RuntimeError(
            "No matching experiments found for pattern '1cft5d_d{datasetsize}v9_s{seed}' or '1cft5d_halfrot_d2000v9_s{seed}'."
        )

    output_path, table_path = plot_success_vs_passes(datasets_to_seeds, halfrot_seeds)
    print(f"Saved plot to {output_path}")
    print(f"Saved table to {table_path}")


if __name__ == "__main__":
    main()
