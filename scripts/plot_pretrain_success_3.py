"""Plot success rate vs UTD (pretrain passes) varying two hyperparameters."""

import json
import math
from pathlib import Path
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
ROLLOUTS_DIR = Path(__file__).resolve().parent.parent / "rollout_data" / "collect_data"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
SEEDS = [1, 2, 3]
PASSES = {
    "1cft_d200v9_15": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "1cft_d300v9_15": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "1cft_d300v9_20": [1, 2, 3, 4, 5, 6, 7, 8],
    "1cft_d300v9_20_75": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "1cft_d400v9_15": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "1cft_d400v9_20": [1, 2, 3, 4, 5, 6, 7, 8],
    "1cft_d400v9_20_75": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "1cft_d500v9_20": [1, 2, 3, 4, 5, 6, 7, 8],
    "1cft_d500v9_20_75": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "1cft_d400v9_25": [0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5],
    "1cft_d400v9_25_60": [1, 2, 3, 4, 5, 6, 7, 8],
    "1cft_d500v9_25": [0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5],
    "1cft_d500v9_25_60": [1, 2, 3, 4, 5, 6, 7, 8],
    "1cft_d600v9_25": [0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5],
    "1cft_d600v9_25_60": [1, 2, 3, 4, 5, 6, 7, 8],
    "1cft_d500v9_30": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3],
    "1cft_d500v9_30_40": [1, 2, 3, 4, 5, 6, 7, 8],
    "1cft_d600v9_30": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3],
    "1cft_d600v9_30_40": [1, 2, 3, 4, 5, 6, 7, 8],
    "1cft_d700v9_30": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3],
    "1cft_d700v9_30_40": [1, 2, 3, 4, 5, 6, 7, 8],
    "1cft_d800v9_30": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3],
    "1cft_d800v9_30_40": [1, 2, 3, 4, 5, 6, 7, 8],
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
    "15": (0.945, 1.005),
    "20": (0.945, 1.005),
    "20_75": (0.945, 1.005),
    "25": (0.945, 1.005),
    "25_60": (0.945, 1.005),
    "30": (0.945, 1.005),
    "30_40": (0.945, 1.005),
}

# Fixed panel layout for one combined 2x4 figure.
PANEL_ORDER = ["15", "legend", "20", "20_75", "25", "25_60", "30", "30_40"]

GROUP_TITLES = {
    "15": "$u_{pe}^{trans}=1.5mm,\\ p_{rand}=0.8,\\ L_{trunc}=150$",
    "20": "$u_{pe}^{trans}=2mm,\\ p_{rand}=0.8,\\ L_{trunc}=180$",
    "20_75": "$u_{pe}^{trans}=2mm,\\ p_{rand}=0.75,\\ L_{trunc}=150$",
    "25": "$u_{pe}^{trans}=2.5mm,\\ p_{rand}=0.8,\\ L_{trunc}=240$",
    "25_60": "$u_{pe}^{trans}=2.5mm,\\ p_{rand}=0.6,\\ L_{trunc}=150$",
    "30": "$u_{pe}^{trans}=3mm,\\ p_{rand}=0.8,\\ L_{trunc}=330$",
    "30_40": "$u_{pe}^{trans}=3mm,\\ p_{rand}=0.4,\\ L_{trunc}=150$",
}

TABLE_LAYOUT = [
    {
        "u": "1.5",
        "blocks": [
            {
                "p": "0.8",
                "L": 150,
                "experiments": ["1cft_d200v9_15", "1cft_d300v9_15", "1cft_d400v9_15"],
            }
        ],
    },
    {
        "u": "2",
        "blocks": [
            {
                "p": "0.8",
                "L": 180,
                "experiments": ["1cft_d300v9_20", "1cft_d400v9_20", "1cft_d500v9_20"],
            },
            {
                "p": "0.75",
                "L": 150,
                "experiments": [
                    "1cft_d300v9_20_75",
                    "1cft_d400v9_20_75",
                    "1cft_d500v9_20_75",
                ],
            },
        ],
    },
    {
        "u": "2.5",
        "blocks": [
            {
                "p": "0.8",
                "L": 240,
                "experiments": ["1cft_d400v9_25", "1cft_d500v9_25", "1cft_d600v9_25"],
            },
            {
                "p": "0.6",
                "L": 150,
                "experiments": [
                    "1cft_d400v9_25_60",
                    "1cft_d500v9_25_60",
                    "1cft_d600v9_25_60",
                ],
            },
        ],
    },
    {
        "u": "3",
        "blocks": [
            {
                "p": "0.8",
                "L": 330,
                "experiments": [
                    "1cft_d500v9_30",
                    "1cft_d600v9_30",
                    "1cft_d700v9_30",
                    "1cft_d800v9_30",
                ],
            },
            {
                "p": "0.4",
                "L": 150,
                "experiments": [
                    "1cft_d500v9_30_40",
                    "1cft_d600v9_30_40",
                    "1cft_d700v9_30_40",
                    "1cft_d800v9_30_40",
                ],
            },
        ],
    },
]


def load_success_rate(eval_path: Path) -> float | None:
    """Compute success rate from an eval_infos.jsonl file."""
    successes = []
    if not os.path.exists(eval_path):
        print(f"Warning: Does not exist: {eval_path}")
        return None
    with open(eval_path) as f:
        for line in f:
            info = json.loads(line)
            if info.get("datetime", "") <= "2026-03-02 00:00:00":
                continue
            successes.append(info["episode_successful"])
    if len(successes) == 0:
        print(f"Warning: No recent evals found in {eval_path}")
        return None
    return np.mean(successes)  # pyright: ignore[reportReturnType]


def extract_dataset_size(hp1_name: str) -> int:
    """Extract dataset size from names like 1cft_d300v9_20_75."""
    d_start = hp1_name.index("_d") + 2
    d_end = hp1_name.index("v", d_start)
    return int(hp1_name[d_start:d_end])


def extract_replay_suffix(exp_name: str) -> str:
    """Extract suffix used in replay-buffer naming from an experiment name."""
    return exp_name.split("v9_", maxsplit=1)[1]


def replay_buffer_path(exp_name: str, seed: int) -> Path:
    """Build replay-buffer path for an experiment and seed."""
    dataset_size = extract_dataset_size(exp_name)
    camera_prefix = exp_name.split("_d", maxsplit=1)[0]
    suffix = extract_replay_suffix(exp_name)
    return (
        ROLLOUTS_DIR
        / f"replay_buffer_{dataset_size}v9_{camera_prefix}_{suffix}_s{seed}.joblib"
    )


def compute_dataset_metrics(
    exp_name: str, truncation_length: int
) -> tuple[float, float, float]:
    """Return mean data-collection SR, MSRL, and average number of actions over seeds."""
    dc_srs = []
    msrls = []
    action_counts = []

    for seed in SEEDS:
        buffer_path = replay_buffer_path(exp_name, seed)
        if not buffer_path.exists():
            raise FileNotFoundError(f"Replay buffer not found: {buffer_path}")

        buffer = joblib.load(buffer_path)
        n_actions = int(buffer.pos_acts)
        n_rollouts = int(buffer.pos_obs - buffer.pos_acts)
        n_success = int(round(buffer.terminateds[: buffer.pos_obs].sum().item()))
        n_unsuccessful = n_rollouts - n_success

        dc_srs.append(n_success / n_rollouts)
        action_counts.append(n_actions)
        if n_success == 0:
            msrls.append(float("nan"))
        else:
            msrls.append((n_actions - truncation_length * n_unsuccessful) / n_success)

    return (
        float(np.mean(dc_srs)),
        float(np.nanmean(msrls)),
        float(np.mean(action_counts)),
    )


def compute_max_sr_stats(exp_name: str) -> tuple[float, float, float] | None:
    """Return max mean SR, corresponding std, and corresponding UTD for one experiment."""
    hp_passes = PASSES[exp_name]
    rates = np.full((len(hp_passes), len(SEEDS)), np.nan)

    for j, p in enumerate(hp_passes):
        for k, seed in enumerate(SEEDS):
            eval_path = (
                CHECKPOINTS_DIR
                / f"{exp_name}_s{seed}"
                / (f"pretrain_{p}" if type(p) is int else f"pretrain_{p:.2f}")
                / "eval_infos.jsonl"
            )
            sr = load_success_rate(eval_path)
            if sr is not None:
                rates[j, k] = sr

    if np.all(np.isnan(rates)):
        return None

    mean = np.nanmean(rates, axis=1)
    std = np.nanstd(rates, axis=1)
    valid = ~np.isnan(mean)
    if not np.any(valid):
        return None

    valid_means = mean[valid]
    valid_stds = std[valid]
    valid_passes = np.array(hp_passes)[valid]
    best_idx = int(np.argmax(valid_means))
    return (
        float(valid_means[best_idx]),
        float(valid_stds[best_idx]),
        float(valid_passes[best_idx]),
    )


def format_utd(utd: float) -> str:
    if math.isclose(utd, round(utd)):
        return str(int(round(utd)))
    return f"{utd:.1f}"


def build_latex_table_pretrain_3hp() -> str:
    """Build the full LaTeX table for pretrain experiments with dataset metrics."""
    metrics_by_exp: dict[str, dict[str, float | str]] = {}

    for group in TABLE_LAYOUT:
        for block in group["blocks"]:
            truncation = int(block["L"])
            for exp_name in block["experiments"]:
                dc_sr, msrl, avg_actions = compute_dataset_metrics(exp_name, truncation)
                max_stats = compute_max_sr_stats(exp_name)
                if max_stats is None:
                    max_sr, corr_std, corr_utd = float("nan"), float("nan"), "--"
                else:
                    max_sr, corr_std, utd = max_stats
                    corr_utd = format_utd(utd)
                metrics_by_exp[exp_name] = {
                    "dataset_size": float(extract_dataset_size(exp_name)),
                    "dc_sr": dc_sr,
                    "msrl": msrl,
                    "avg_actions": avg_actions,
                    "max_sr": max_sr,
                    "corr_std": corr_std,
                    "corr_utd": corr_utd,
                }

    lines = [
        "\\begin{table}[htpb]",
        "\\centering",
        "\\begin{tabular}{l|ccc|ccc|ccc}",
        "\\hline",
        "$u_{pe}^{trans}$ & \\multirow{2}{*}{$p_{rand}$} & \\multirow{2}{*}{$L_{trunc}$} & \\multirow{2}{*}{$\\left|\\mathcal{D}\\right|$} & \\multirow{2}{*}{DC \\ac{SR}} & \\multirow{2}{*}{MSRL} & $\\left\\langle\\mathcal{D}\\right\\rangle$ & Max. & Corresp. & Corresp. \\\\",
        "$\\left[\\si{\\milli\\metre}\\right]$ &  &  &  & &  & $\\left[\\times 10^3\\right]$ & Mean \\ac{SR} & Std. & \\ac{UTD} \\\\",
        "\\hline",
        "\\hline",
    ]

    for group_idx, group in enumerate(TABLE_LAYOUT):
        u_value = str(group["u"])
        blocks = group["blocks"]
        group_rows = sum(len(block["experiments"]) for block in blocks)
        first_row_of_group = True

        for block_idx, block in enumerate(blocks):
            p_value = str(block["p"])
            truncation = int(block["L"])
            experiments = block["experiments"]
            block_dc_sr_avg = float(
                np.mean(
                    [
                        float(metrics_by_exp[exp_name]["dc_sr"])
                        for exp_name in experiments
                    ]
                )
            )
            block_msrl_avg = float(
                np.mean(
                    [
                        float(metrics_by_exp[exp_name]["msrl"])
                        for exp_name in experiments
                    ]
                )
            )

            for exp_idx, exp_name in enumerate(experiments):
                m = metrics_by_exp[exp_name]
                u_col = (
                    f"\\multirow{{{group_rows}}}{{*}}{{{u_value}}}"
                    if first_row_of_group
                    else ""
                )
                p_col = (
                    f"\\multirow{{{len(experiments)}}}{{*}}{{{p_value}}}"
                    if exp_idx == 0
                    else ""
                )
                l_col = (
                    f"\\multirow{{{len(experiments)}}}{{*}}{{{truncation}}}"
                    if exp_idx == 0
                    else ""
                )
                dc_sr_col = (
                    f"\\multirow{{{len(experiments)}}}{{*}}{{{block_dc_sr_avg:.2f}}}"
                    if exp_idx == 0
                    else ""
                )
                msrl_col = (
                    f"\\multirow{{{len(experiments)}}}{{*}}{{{int(round(block_msrl_avg))}}}"
                    if exp_idx == 0
                    else ""
                )

                lines.append(
                    " & ".join(
                        [
                            u_col,
                            p_col,
                            l_col,
                            str(int(m["dataset_size"])),
                            dc_sr_col,
                            msrl_col,
                            f"{float(m['avg_actions']) / 1000.0:.1f}",
                            f"{float(m['max_sr']):.3f}",
                            f"{float(m['corr_std']):.3f}",
                            str(m["corr_utd"]),
                        ]
                    )
                    + " \\\\"
                )
                first_row_of_group = False

            if block_idx < len(blocks) - 1:
                lines.append("\\cline{2-10}")

        lines.append("\\hline\\hline")

    lines.extend(
        [
            "\\end{tabular}",
            "\\caption{Policy performance linked to data collection strategies and $\\left|\\mathcal{D}\\right|$, for increasing $u_{pe}^{trans}$.}",
            "\\label{tab:pe_unc_sr}",
            "\\end{table}",
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
    }


def plot_group_on_axis(
    ax: Axes,
    group_name: str,
    hp1_values: list[str],
    ylims: tuple[float, float],
    dataset_colors: dict[int, str],
    is_leftmost: bool = False,
):
    all_passes: set[float] = set()
    for hp1 in hp1_values:
        all_passes.update(PASSES[hp1])

    for hp1 in hp1_values:
        dataset_size = extract_dataset_size(hp1)
        color = dataset_colors[dataset_size]
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

        ax.plot(x, m, marker="o", markersize=4.5, color=color)
        ax.fill_between(x, m - s, m + s, alpha=0.2, color=color)

    ax.set_xlabel("UTD (epochs)")
    if is_leftmost:
        ax.set_ylabel("Success Rate")
    ax.set_title(GROUP_TITLES.get(group_name, group_name))
    xticks = sorted(all_passes)
    sub_one_count = sum(1 for x in xticks if x < 1)
    hidden_labels: set[float] = set()
    if sub_one_count == 4:
        hidden_labels = {0.4, 0.8}
    elif sub_one_count == 9:
        hidden_labels = {0.2, 0.3, 0.5, 0.6, 0.8, 0.9}

    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [
            ""
            if any(np.isclose(x, hidden) for hidden in hidden_labels)
            else (f"{int(round(x))}" if np.isclose(x, round(x)) else f"{x:.1f}")
            for x in xticks
        ]
    )
    x_min = 0 if sub_one_count > 0 else min(all_passes)
    ax.set_xlim(x_min, max(all_passes))
    ax.set_ylim(*ylims)
    ax.grid(True, alpha=0.3)


def plot_combined(filename: str):
    fig, axes = plt.subplots(4, 2, figsize=(12, 16), sharey=True)
    fig.suptitle("Success Rate vs UTD", fontsize=16)
    axes_flat = axes.flatten()
    dataset_colors = build_dataset_color_map()

    for idx, panel_name in enumerate(PANEL_ORDER):
        ax = axes_flat[idx]
        if panel_name == "legend":
            handles = []
            labels = []
            for size in sorted(dataset_colors):
                handles.append(Line2D([0], [0], color=dataset_colors[size], lw=3))
                labels.append(f"$\\left|\\mathcal{{D}}\\right|={size}$")
            ax.legend(
                handles, labels, loc="center", fontsize="large", title="Dataset size"
            )
            ax.axis("off")
            continue

        plot_group_on_axis(
            ax,
            panel_name,
            GROUPS[panel_name],
            YLIMS[panel_name],
            dataset_colors,
            is_leftmost=(idx % 2 == 0),
        )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(CHECKPOINTS_DIR.parent / "plots" / filename, dpi=150)


def main():
    plot_combined("pretrain_success_grid.png")
    # latex_table = build_latex_table_pretrain_3hp()
    # PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    # table_path = PLOTS_DIR / "pretrain_success_max_table_3hp.tex"
    # table_path.write_text(latex_table + "\n", encoding="utf-8")
    # print(latex_table)


if __name__ == "__main__":
    main()
