"""Plot force/torque time-series from eval LeRobot dataset episodes.

Two data sources, in priority order:

1. ``<dataset_dir>/ft_log/episode_NNNNNN.npz`` (HOME-TO-HOME log written by
   eval_actor._save_ft_log_npz when the wrapper exposes timestamped FT).
   Covers reset prologue (grasp, PE, contact) + policy steps + snap_push,
   including the force-controlled press peaks. Each NPZ has arrays:
     - ``ft`` (T, 6 float64) — fx, fy, fz, mx, my, mz
     - ``timestamps`` (T, float64) — seconds since reset start
     - ``phases`` (T, object/str) — "reset" | "policy" | "snap_push"

2. ``<dataset_dir>/data/chunk-*/episode_NNNNNN.parquet`` (POLICY-PHASE only
   fallback). Used when no NPZ is present (older eval runs). Snap_push press
   peaks will NOT appear because those wrapper-internal env steps are not in
   the parquet rows.

Usage examples:

  # Plot Fz vs time for episode 0 (auto: NPZ if present, parquet else):
  python scripts/plot_ft_episode.py \\
    --dataset_dir rollout_data/eval/<dataset> --episode 0

  # Plot Tz (torque Z) vs time for episode 0:
  python scripts/plot_ft_episode.py \\
    --dataset_dir rollout_data/eval/<dataset> --episode 0 --axes mz

  # All 3 forces (Fx, Fy, Fz) side-by-side:
  python scripts/plot_ft_episode.py \\
    --dataset_dir rollout_data/eval/<dataset> --episode 0 --axes forces

  # All 3 torques (Tx, Ty, Tz) side-by-side:
  python scripts/plot_ft_episode.py \\
    --dataset_dir rollout_data/eval/<dataset> --episode 0 --axes torques

  # All 6 axes in a 2x3 grid:
  python scripts/plot_ft_episode.py \\
    --dataset_dir rollout_data/eval/<dataset> --episode 0 --axes all

  # Overlay all episodes' Fz on one figure (also works with --axes mz):
  python scripts/plot_ft_episode.py \\
    --dataset_dir rollout_data/eval/<dataset> --episode all --overlay

  # Force parquet-only mode (ignore NPZ if present):
  python scripts/plot_ft_episode.py \\
    --dataset_dir rollout_data/eval/<dataset> --episode 0 --source parquet
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


FT_COLS = ("fx", "fy", "fz", "mx", "my", "mz")
FT_UNITS = ("N", "N", "N", "Nm", "Nm", "Nm")
PHASE_COLORS = {
    # initial / generic
    "reset": "tab:olive",
    # reset sub-phases (inter-episode cleanup before wide PE)
    "reset_lift": "tab:olive",
    "reset_putback_hover": "tab:olive",
    "reset_reseat_push": "tab:pink",  # contact event — stand out
    "reset_release_open": "tab:olive",
    "reset_gripper_home": "tab:pink",  # gripper close→open calibration spike
    "reset_home_joint": "tab:olive",
    "reset_zero_orient": "tab:olive",
    # pose estimations (steady camera, motion ~= 0)
    "pe_wide": "tab:purple",
    "pe_refined": "tab:purple",
    "pe_final": "tab:purple",
    "pe_place": "tab:purple",
    # approach motions (move toward something)
    "approach_pe_wide": "tab:cyan",
    "approach_pe_refined": "tab:cyan",
    "approach_grasp": "tab:cyan",
    "descend_grasp": "tab:cyan",
    "descend_grasp_refined": "tab:cyan",
    "approach_goal": "tab:cyan",
    # contact / grasp / post-grasp / RL / snap
    "grasp": "tab:brown",
    "post_grasp_lift": "tab:green",
    "contact": "tab:orange",
    "policy": "tab:blue",
    "snap_push": "tab:red",
    "unknown": "grey",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="LeRobot dataset directory.",
    )
    parser.add_argument(
        "--episode",
        type=str,
        default="0",
        help='Episode index (int) or "all". Default: 0.',
    )
    parser.add_argument(
        "--axes",
        type=str,
        choices=["z", "mz", "forces", "torques", "all"],
        default="z",
        help=(
            'Which channels to plot. "z" = Fz only (single panel); '
            '"mz" = Tz only (single panel); "forces" = Fx,Fy,Fz in a 1x3 '
            'grid; "torques" = Mx,My,Mz in a 1x3 grid; "all" = all 6 '
            'channels in a 2x3 grid. For --overlay, must be "z" or "mz".'
        ),
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["auto", "npz", "parquet"],
        default="auto",
        help='"auto" prefers ft_log/*.npz, falls back to parquet. '
        '"npz" requires the NPZ. "parquet" forces policy-phase-only.',
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for PNGs. Default: plots/ft_<dataset_basename>.",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Plot all selected episodes on a single figure (Fz only).",
    )
    return parser.parse_args()


def _extract_episode_id_from_path(path: Path) -> int:
    m = re.search(r"episode_(\d+)", path.stem)
    return int(m.group(1)) if m else int(1e12)


def find_episode_ids(dataset_dir: Path, prefer_npz: bool) -> list[int]:
    """Return sorted episode IDs available from either source."""
    ids: set[int] = set()
    if prefer_npz:
        for p in (dataset_dir / "ft_log").glob("episode_*.npz"):
            ids.add(_extract_episode_id_from_path(p))
    for p in (dataset_dir / "data").glob("chunk-*/episode_*.parquet"):
        ids.add(_extract_episode_id_from_path(p))
    return sorted(ids)


def load_episode_ft_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (timestamps[T], ft[T, 6], phases[T])."""
    z = np.load(npz_path, allow_pickle=True)
    return (
        np.asarray(z["timestamps"], dtype=np.float64),
        np.asarray(z["ft"], dtype=np.float64),
        np.asarray(z["phases"]),
    )


def load_episode_ft_parquet(
    parquet_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback loader. Returns (timestamps[T], ft[T, 6], phases[T]) where
    phases is all "policy" since parquet rows are policy-phase only."""
    table = pq.read_table(
        parquet_path,
        columns=["timestamp", "observation.state.sensors_bota_ft_sensor"],
    )
    ts_raw = table.column("timestamp").to_pylist()
    timestamps = np.asarray(
        [float(np.asarray(x).squeeze()) for x in ts_raw], dtype=np.float64
    )
    ft_raw = table.column("observation.state.sensors_bota_ft_sensor").to_pylist()
    ft = np.stack([np.asarray(x, dtype=np.float64) for x in ft_raw], axis=0)
    phases = np.array(["policy"] * len(timestamps), dtype=object)
    return timestamps, ft, phases


def resolve_episode_source(
    dataset_dir: Path, ep_id: int, source: str
) -> tuple[str, Path]:
    """Return (mode, path) where mode in {'npz','parquet'}."""
    npz_path = dataset_dir / "ft_log" / f"episode_{ep_id:06d}.npz"
    # parquet may live in any chunk-NNN directory
    parquet_matches = list(
        (dataset_dir / "data").glob(f"chunk-*/episode_{ep_id:06d}.parquet")
    )
    parquet_path = parquet_matches[0] if parquet_matches else None

    if source == "npz":
        if not npz_path.is_file():
            raise FileNotFoundError(f"NPZ not found for ep {ep_id}: {npz_path}")
        return "npz", npz_path
    if source == "parquet":
        if parquet_path is None:
            raise FileNotFoundError(f"Parquet not found for ep {ep_id}")
        return "parquet", parquet_path
    # auto
    if npz_path.is_file():
        return "npz", npz_path
    if parquet_path is not None:
        return "parquet", parquet_path
    raise FileNotFoundError(
        f"Neither NPZ nor parquet found for ep {ep_id} under {dataset_dir}"
    )


def load_episode(
    dataset_dir: Path, ep_id: int, source: str
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    mode, path = resolve_episode_source(dataset_dir, ep_id, source)
    if mode == "npz":
        ts, ft, ph = load_episode_ft_npz(path)
    else:
        ts, ft, ph = load_episode_ft_parquet(path)
    return mode, ts, ft, ph


def load_episode_meta(dataset_dir: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    eps_path = dataset_dir / "meta" / "episodes.jsonl"
    if eps_path.is_file():
        for line in open(eps_path):
            d = json.loads(line)
            out.setdefault(int(d["episode_index"]), {})["length"] = int(
                d.get("length", 0)
            )
    return out


def _phase_boundaries(timestamps: np.ndarray, phases: np.ndarray) -> list[tuple[float, str]]:
    """Return (t, phase_after) for each phase transition."""
    out: list[tuple[float, str]] = []
    if len(phases) == 0:
        return out
    prev = phases[0]
    for i in range(1, len(phases)):
        if phases[i] != prev:
            out.append((float(timestamps[i]), str(phases[i])))
            prev = phases[i]
    return out


def _title_for(ep_id: int, mode: str, n_frames: int, duration_s: float, meta: dict) -> str:
    info = meta.get(ep_id, {})
    extra = f" len(parquet)={info['length']}" if "length" in info else ""
    mode_tag = "HOME-TO-HOME" if mode == "npz" else "POLICY-PHASE only (parquet)"
    return f"Episode {ep_id} — {n_frames} samples ({duration_s:.2f}s) — {mode_tag}{extra}"


def _draw_phase_overlays(
    ax, timestamps: np.ndarray, phases: np.ndarray, mode: str
) -> None:
    if mode != "npz" or len(phases) == 0:
        return
    boundaries = _phase_boundaries(timestamps, phases)
    # Include the implicit first phase at t=0 so the first segment is labeled.
    boundaries = [(float(timestamps[0]), str(phases[0]))] + boundaries
    y_lo, y_hi = ax.get_ylim()
    y_span = y_hi - y_lo
    for i, (t, phase_after) in enumerate(boundaries):
        color = PHASE_COLORS.get(phase_after, "grey")
        # Skip vertical line at the very first sample (t=0) — it's the
        # initial state, not a transition.
        if i > 0:
            ax.axvline(t, color=color, linestyle="--", linewidth=0.7, alpha=0.55)
        # Stagger labels vertically across 4 rows so they don't overlap when
        # many phases are close together. Rotate 90° (vertical text) to
        # save horizontal space.
        row = i % 4
        y_text = y_hi - (0.02 + 0.10 * row) * y_span
        ax.text(
            t, y_text, f" {phase_after}",
            color=color, fontsize=7, va="top", ha="left",
            rotation=90, rotation_mode="anchor",
        )


def plot_single_axis(
    ep_id: int,
    mode: str,
    timestamps: np.ndarray,
    ft: np.ndarray,
    phases: np.ndarray,
    out_path: Path,
    meta: dict,
    axis_idx: int,
) -> None:
    """Single-channel plot for one episode (e.g. Fz or Tz)."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(timestamps, ft[:, axis_idx], color="tab:blue", linewidth=1.0)
    ax.axhline(0, color="grey", linewidth=0.5, alpha=0.5)
    ax.set_xlabel(
        "time [s] (from start of reset)"
        if mode == "npz"
        else "time [s] (from start of policy phase)"
    )
    ax.set_ylabel(f"{FT_COLS[axis_idx]} [{FT_UNITS[axis_idx]}]")
    ax.set_title(
        _title_for(ep_id, mode, len(timestamps),
                   float(timestamps[-1] if len(timestamps) else 0.0), meta)
    )
    ax.grid(alpha=0.3)
    _draw_phase_overlays(ax, timestamps, phases, mode)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_three_axis(
    ep_id: int,
    mode: str,
    timestamps: np.ndarray,
    ft: np.ndarray,
    phases: np.ndarray,
    out_path: Path,
    meta: dict,
    axis_indices: tuple[int, int, int],
) -> None:
    """3-panel plot for a triple of channels (Fx/Fy/Fz or Mx/My/Mz)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True)
    for ax, i in zip(axes, axis_indices):
        ax.plot(timestamps, ft[:, i], color="tab:blue", linewidth=0.9)
        ax.axhline(0, color="grey", linewidth=0.5, alpha=0.5)
        ax.set_title(f"{FT_COLS[i]} [{FT_UNITS[i]}]")
        ax.set_xlabel("time [s]")
        ax.grid(alpha=0.3)
        _draw_phase_overlays(ax, timestamps, phases, mode)
    fig.suptitle(
        _title_for(ep_id, mode, len(timestamps),
                   float(timestamps[-1] if len(timestamps) else 0.0), meta)
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_single_all(
    ep_id: int,
    mode: str,
    timestamps: np.ndarray,
    ft: np.ndarray,
    phases: np.ndarray,
    out_path: Path,
    meta: dict,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.5), sharex=True)
    for i, ax in enumerate(axes.flatten()):
        ax.plot(timestamps, ft[:, i], color="tab:blue", linewidth=0.9)
        ax.axhline(0, color="grey", linewidth=0.5, alpha=0.5)
        ax.set_title(f"{FT_COLS[i]} [{FT_UNITS[i]}]")
        ax.grid(alpha=0.3)
        if i >= 3:
            ax.set_xlabel("time [s]")
        _draw_phase_overlays(ax, timestamps, phases, mode)
    fig.suptitle(
        _title_for(ep_id, mode, len(timestamps),
                   float(timestamps[-1] if len(timestamps) else 0.0), meta)
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_overlay_axis(
    episodes: list[tuple[int, str, np.ndarray, np.ndarray, np.ndarray]],
    out_path: Path,
    axis_idx: int,
) -> None:
    """Overlay one channel across episodes."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    cmap = plt.cm.viridis
    modes = set()
    for idx, (ep_id, mode, ts, ft, _ph) in enumerate(episodes):
        color = cmap(idx / max(1, len(episodes) - 1))
        ax.plot(ts, ft[:, axis_idx], color=color, linewidth=0.7, alpha=0.7,
                label=f"ep {ep_id}")
        modes.add(mode)
    ax.axhline(0, color="grey", linewidth=0.5, alpha=0.5)
    mode_tag = "HOME-TO-HOME" if modes == {"npz"} else (
        "POLICY-PHASE (parquet)" if modes == {"parquet"} else "MIXED"
    )
    ax.set_xlabel("time [s]")
    ax.set_ylabel(f"{FT_COLS[axis_idx]} [{FT_UNITS[axis_idx]}]")
    ax.set_title(
        f"{FT_COLS[axis_idx]} vs time — {len(episodes)} episodes overlaid — {mode_tag}"
    )
    ax.grid(alpha=0.3)
    if len(episodes) <= 25:
        ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    prefer_npz = args.source in ("auto", "npz")
    available = find_episode_ids(args.dataset_dir, prefer_npz=prefer_npz)
    if not available:
        raise SystemExit(f"No episodes found under {args.dataset_dir}.")

    if args.episode == "all":
        selected = available
    else:
        ep_id = int(args.episode)
        if ep_id not in available:
            raise SystemExit(
                f"Episode {ep_id} not found. Available: {available[:5]}…{available[-5:]}"
            )
        selected = [ep_id]

    out_dir = args.output_dir or Path("plots") / f"ft_{args.dataset_dir.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_episode_meta(args.dataset_dir)

    # Map axes selection to the channel index / suffix.
    SINGLE_AXIS = {"z": 2, "mz": 5}            # axis_idx into ft[:, i]
    THREE_AXIS = {                              # axis triplet (0=fx..5=mz)
        "forces": (0, 1, 2),
        "torques": (3, 4, 5),
    }

    if args.overlay:
        if args.axes not in SINGLE_AXIS:
            raise SystemExit(
                f'--overlay requires --axes z or --axes mz (got "{args.axes}")'
            )
        axis_idx = SINGLE_AXIS[args.axes]
        loaded = []
        for ep_id in selected:
            mode, ts, ft, ph = load_episode(args.dataset_dir, ep_id, args.source)
            loaded.append((ep_id, mode, ts, ft, ph))
        out_path = out_dir / f"overlay_{FT_COLS[axis_idx]}_{len(selected)}ep.png"
        plot_overlay_axis(loaded, out_path, axis_idx)
        print(f"wrote {out_path}")
        return

    for ep_id in selected:
        mode, ts, ft, ph = load_episode(args.dataset_dir, ep_id, args.source)
        if args.axes in SINGLE_AXIS:
            axis_idx = SINGLE_AXIS[args.axes]
            out_path = out_dir / f"episode_{ep_id:06d}_{FT_COLS[axis_idx]}.png"
            plot_single_axis(ep_id, mode, ts, ft, ph, out_path, meta, axis_idx)
            peak = float(np.max(np.abs(ft[:, axis_idx]))) if len(ts) else 0.0
            unit = FT_UNITS[axis_idx]
            label = f"|{FT_COLS[axis_idx]}|max={peak:.2f}{unit}"
        elif args.axes in THREE_AXIS:
            axis_indices = THREE_AXIS[args.axes]
            out_path = out_dir / f"episode_{ep_id:06d}_{args.axes}.png"
            plot_three_axis(ep_id, mode, ts, ft, ph, out_path, meta, axis_indices)
            peaks = [
                f"|{FT_COLS[i]}|max={float(np.max(np.abs(ft[:, i]))):.2f}{FT_UNITS[i]}"
                for i in axis_indices
            ]
            label = " ".join(peaks)
        else:  # "all"
            out_path = out_dir / f"episode_{ep_id:06d}_all.png"
            plot_single_all(ep_id, mode, ts, ft, ph, out_path, meta)
            peak = float(np.max(np.abs(ft[:, 2]))) if len(ts) else 0.0
            label = f"|Fz|max={peak:.2f}N"
        dur = float(ts[-1]) if len(ts) else 0.0
        print(
            f"wrote {out_path}  [{mode}]  T={len(ts)} samples, "
            f"duration={dur:.2f}s, {label}"
        )


if __name__ == "__main__":
    main()
