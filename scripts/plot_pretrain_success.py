"""Plot success rate vs UTD (pretrain passes) for all hyperparameter settings."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
HYPERPARAMETERS = ["0", "3", "6", "7", "75", "8", "85", "9", "95", "100"]
SEEDS = [1, 2, 3]
PASSES = {
    "0": [1, 3, 5, 10, 15, 20],
    "3": [1, 3, 5, 10, 15, 20],
    "6": [1, 3, 5, 10, 15, 20],
    "7": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20],
    "75": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20],
    "8": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20],
    "85": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20],
    "9": [1, 3, 5, 10, 15, 20],
    "95": [1, 3, 5, 10, 15, 20],
    "100": [1, 3, 5, 10, 15, 20],
}
BASE_NAME = "1cam_ft_128_16_full_d300v9"
TAB10 = plt.cm.tab10  # pyright: ignore[reportAttributeAccessIssue]
HP_COLORS = {hp: TAB10(i % TAB10.N) for i, hp in enumerate(HYPERPARAMETERS)}


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


def hp_to_str(hp: str) -> str:
    if len(hp) == 3:
        return "1"
    if hp == "0":
        return "0"
    return f"{int(hp) / 100:.2f}" if len(hp) == 2 else f"{int(hp) / 10:.1f}"


def plot_on_ax(ax, hyperparameters, passes, title, ylim=(-0.05, 1.05)):
    for i, hp in enumerate(hyperparameters):
        color = HP_COLORS.get(hp, TAB10(i % TAB10.N))
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

        ax.plot(
            x,
            m,
            marker="o",
            markersize=4.5,
            label="$p_{rand}=" + hp_to_str(hp) + "$",
            color=color,
        )
        ax.fill_between(x, m - s, m + s, alpha=0.2, color=color)

    ax.set_xlabel("UTD (epochs)")
    ax.set_ylabel("Success Rate")
    ax.set_title(title)
    all_passes = sorted(set(p for ps in passes.values() for p in ps))
    ax.set_xticks(all_passes)  # Show all integer ticks in range
    ax.set_xlim(all_passes[0], all_passes[-1])  # Show all integer ticks in range
    ax.legend(loc="lower right", fontsize="small", framealpha=0.9)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)


def max_stats_by_hyperparameter(hyperparameters, passes):
    """Return max mean SR stats per hyperparameter across UTD values."""
    stats = {}
    for hp in hyperparameters:
        hp_passes = passes[hp]
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

        if np.all(np.isnan(rates)):
            continue

        mean = np.nanmean(rates, axis=1)
        std = np.nanstd(rates, axis=1)
        valid = ~np.isnan(mean)
        if not np.any(valid):
            continue

        valid_means = mean[valid]
        valid_stds = std[valid]
        valid_passes = np.array(hp_passes)[valid]
        max_idx = int(np.argmax(valid_means))
        stats[hp] = {
            "max_mean": float(valid_means[max_idx]),
            "std_at_max": float(valid_stds[max_idx]),
            "utd_at_max": int(valid_passes[max_idx]),
        }

    return stats


def build_latex_table(hyperparameters, stats):
    """Build a LaTeX table with p_rand as columns and metrics as rows."""
    available_hps = [
        hp for hp in hyperparameters if hp in stats and hp_to_str(hp) not in {"1"}
    ]
    if not available_hps:
        return ""

    max_value = max(stats[hp]["max_mean"] for hp in available_hps)
    cols = "l" + "c" * len(available_hps)
    header = "$p_{rand}$ & " + " & ".join(hp_to_str(hp) for hp in available_hps)

    mean_cells = []
    std_cells = []
    utd_cells = []
    for hp in available_hps:
        value = stats[hp]["max_mean"]
        std = stats[hp]["std_at_max"]
        utd = stats[hp]["utd_at_max"]
        cell = f"{value:.3f}"
        if np.isclose(value, max_value):
            cell = f"\\textbf{{{cell}}}"
        mean_cells.append(cell)
        std_cells.append(f"{std:.3f}")
        utd_cells.append(str(utd))

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\hline",
        header + " \\\\",
        "\\hline",
        "Max. Mean \\ac{SR} & " + " & ".join(mean_cells) + " \\\\",
        "Corresp. Std. & " + " & ".join(std_cells) + " \\\\",
        "Corresp. \\ac{UTD} & " + " & ".join(utd_cells) + " \\\\",
        "\\hline",
        "\\end{tabular}",
        "\\caption{Maximum mean success rate over UTD for each $p_{rand}$ with corresponding standard deviation. Bold indicates the largest mean value across columns.}",
        "\\label{tab:pretrain-max-sr}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def plot_max_stats_bar_chart(stats, output_path: Path, title: str = "Policy  SR"):
    """Plot the maximum mean SR per hyperparameter with std error bars."""
    available_hps = [hp for hp in HYPERPARAMETERS if hp in stats]
    if not available_hps:
        return

    labels = [hp_to_str(hp) for hp in available_hps]
    means = [stats[hp]["max_mean"] for hp in available_hps]
    stds = [stats[hp]["std_at_max"] for hp in available_hps]
    colors = [
        HP_COLORS.get(hp, TAB10(i % TAB10.N)) for i, hp in enumerate(available_hps)
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(available_hps))
    ax.bar(
        x, means, yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=0.6
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_xlabel("Policy", fontsize=13)
    ax.set_ylabel("SR", fontsize=13)
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: all hyperparameters, all passes
    plot_on_ax(
        axes[0],
        HYPERPARAMETERS,
        PASSES,
        "Success Rate vs UTD",
    )

    # Plot 2: subset of hyperparameters, UTD >= 3
    plot_on_ax(
        axes[1],
        ["6", "7", "75", "8"],
        {hp: [p for p in PASSES[hp] if p >= 1] for hp in ["6", "7", "75", "8"]},
        "Success Rate vs UTD",
        ylim=(0.945, 1.005),
    )

    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / "pretrain_success_combined.png", dpi=150)
    stats = max_stats_by_hyperparameter(HYPERPARAMETERS, PASSES)
    plot_max_stats_bar_chart(stats, PLOTS_DIR / "pretrain_success_max_stats_bar.png")
    latex_table = build_latex_table(HYPERPARAMETERS, stats)
    if latex_table:
        table_path = PLOTS_DIR / "pretrain_success_max_table.tex"
        table_path.write_text(latex_table + "\n", encoding="utf-8")
        print(latex_table)
    # plt.show()

    # Plot 3: subset of hyperparameters, UTD >= 3, different y-axis range
    # plot(
    #     ["75", "8"],
    #     {hp: [p for p in PASSES[hp] if p >= 1] for hp in ["75", "8"]},
    #     "Success Rate vs UTD (hp ∈ {75,8})",
    #     "pretrain_success_subset_small.png",
    #     ylim=(0.945, 1.005),
    # )


if __name__ == "__main__":
    main()
