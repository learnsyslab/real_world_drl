"""Plot success rate vs UTD (pretrain passes) varying two hyperparameters."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
HP1_VALUES = ["1cft", "2c", "1c"]
HP2_VALUES = [100, 150, 200, 300, 400]
COMBINED_HP2_VALUES = [400, 300, 200]
TABLE_HP1_VALUES = ["1c", "1cft", "2c"]
SEEDS = [1, 2, 3]
PASSES = {
    100: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    150: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    200: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    300: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    400: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
}
YLIMS = {
    100: (-0.05, 1.05),
    150: (-0.05, 1.05),
    200: (0.945, 1.005),
    300: (0.945, 1.005),
    400: (0.945, 1.005),
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


def hp1_to_str(hp1: str) -> str:
    if hp1 == "1c":
        return "$\\it{1cam}$"
    if hp1 == "1cft":
        return "$\\it{1cam+ft}$"
    if hp1 == "2c":
        return "$\\it{2cam}$"
    return hp1


def hp1_to_str_latex(hp1: str) -> str:
    if hp1 == "1c":
        return "\\textit{1cam}"
    if hp1 == "1cft":
        return "\\textit{1cam+ft}"
    if hp1 == "2c":
        return "\\textit{2cam}"
    return hp1


def i_to_cmap(i: int) -> float | int:
    if i == 0:
        return 0
    if i == 1:
        return 2
    if i == 2:
        return 3


def compute_sr_stats(hp1: str, hp2: int) -> tuple[float, float, int] | None:
    """Return max(mean SR), corresponding std across seeds, and corresponding UTD."""
    passes = PASSES[hp2]
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
        return None

    mean = np.nanmean(rates, axis=1)
    std = np.nanstd(rates, axis=1)
    valid_idx = np.where(~np.isnan(mean))[0]
    if len(valid_idx) == 0:
        return None

    best_rel_idx = int(np.nanargmax(mean[valid_idx]))
    best_idx = int(valid_idx[best_rel_idx])
    return float(mean[best_idx]), float(std[best_idx]), int(passes[best_idx])


def format_sr(value: float, is_best: bool) -> str:
    sr_str = f"{value:.3f}"
    if is_best:
        return f"\\textbf{{{sr_str}}}"
    return sr_str


def build_sensor_choice_table(hp1_values: list[str], hp2_values: list[int]) -> str:
    stats: dict[str, dict[int, tuple[float, float, int] | None]] = {}
    global_best = -np.inf

    for hp1 in hp1_values:
        stats[hp1] = {}
        for hp2 in hp2_values:
            entry = compute_sr_stats(hp1, hp2)
            stats[hp1][hp2] = entry
            if entry is not None and entry[0] > global_best:
                global_best = entry[0]

    lines = [
        "\\begin{table}[b]",
        "\\centering",
        "\\begin{tabular}{llccc}",
        "\\hline",
        "Sensor Choice & $\\left|\\mathcal{D}\\right|$ & Max. Mean \\ac{SR} & Corresp. Std. & Corresp. \\ac{UTD} \\\\",
        "\\hline",
    ]

    for hp1 in hp1_values:
        for i, hp2 in enumerate(hp2_values):
            entry = stats[hp1][hp2]
            sensor_col = (
                f"\\multirow{{{len(hp2_values)}}}{{*}}{{{hp1_to_str_latex(hp1)}}}"
                if i == 0
                else ""
            )

            if entry is None:
                row = (
                    f"{sensor_col} & {hp2} & -- & -- & -- \\\\"
                    if i == 0
                    else f" & {hp2} & -- & -- & -- \\\\"
                )
            else:
                mean, std, utd = entry
                is_best = np.isclose(mean, global_best)
                mean_str = format_sr(mean, is_best)
                row = (
                    f"{sensor_col} & {hp2} & {mean_str} & {std:.3f} & {utd} \\\\"
                    if i == 0
                    else f" & {hp2} & {mean_str} & {std:.3f} & {utd} \\\\"
                )

            lines.append(row)

        lines.append("\\hline")

    lines.extend(
        [
            "\\end{tabular}",
            "\\caption{Maximum mean success rate over \\ac{UTD} for each sensor choice and each $\\left|\\mathcal{D}\\right| \\in \\{400, 300, 200\\}$, with corresponding standard deviation and \\ac{UTD}. Bold indicates the largest mean value across all rows.}",
            "\\label{tab:pretrain-max-sr-sensors}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def plot_combined(hp2_values: list[int], filename: str):
    fig, axes = plt.subplots(
        1, len(hp2_values), figsize=(15, 6), sharey=True, sharex=True
    )
    if len(hp2_values) == 1:
        axes = [axes]
    cmap = plt.cm.tab10  # pyright: ignore[reportAttributeAccessIssue]

    for ax, hp2 in zip(axes, hp2_values):
        hp1_values = [h for h in HP1_VALUES if not (hp2 == 200 and h == "2c")]
        passes = PASSES[hp2]

        for i, hp1 in enumerate(hp1_values):
            color = cmap(i_to_cmap(i))
            rates = np.full((len(passes), len(SEEDS)), np.nan)

            for j, p in enumerate(passes):
                for k, seed in enumerate(SEEDS):
                    dir_name = f"{hp1}_d{hp2}v9_s{seed}"
                    eval_path = (
                        CHECKPOINTS_DIR
                        / dir_name
                        / f"pretrain_{p}"
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
            x = np.array(passes)[valid]
            m = mean[valid]
            s = std[valid]

            ax.plot(x, m, marker="o", label=hp1_to_str(hp1), color=color)
            ax.fill_between(x, m - s, m + s, alpha=0.2, color=color)

        if hp2 == hp2_values[0]:
            ax.legend(loc="lower right", fontsize="small")
        # if hp2 == hp2_values[-1]:
        ax.set_xlabel("UTD (epochs)")
        ax.set_title(f"$\\left|\\mathcal{{D}}\\right|={hp2}$")
        ax.set_xticks(passes)
        ax.set_xlim(passes[0], passes[-1])

        ax.set_ylim(*YLIMS[hp2])
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Success Rate")
    fig.suptitle("Success Rate vs UTD", fontsize=16)
    fig.tight_layout()
    # fig.subplots_adjust(top=0.86)
    fig.savefig(CHECKPOINTS_DIR.parent / "scripts" / filename, dpi=150)


def main():
    plot_combined(COMBINED_HP2_VALUES, "pretrain_success_d400_d300_d200.png")
    latex_table = build_sensor_choice_table(TABLE_HP1_VALUES, COMBINED_HP2_VALUES)
    print(latex_table)
    with open(
        CHECKPOINTS_DIR.parent / "scripts" / "pretrain_sensor_choice_table.tex", "w"
    ) as f:
        f.write(latex_table + "\n")


if __name__ == "__main__":
    main()
