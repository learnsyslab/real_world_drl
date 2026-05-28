"""Offline analyzer for FoundationPose repeatability on the Siemens lid task.

Reads ep{NNN}_raw.npz files written by InsertionWrapperSiemensPE
(crisp_drl/envs/pose_visualizer.dump_raw_npz) and reports per-DoF PE
distribution + visual qualitative strip.

Usage
-----
    python scripts/analyze_pe_repeatability.py <pose_viz_dir>
        [--gt-episode N]   anchor errors to ep{N} (visual GT) in addition to
                           the pseudo-GT (component-wise median)
        [--out-dir DIR]    default <pose_viz_dir>/analysis/
        [--which {coarse,refined,both}]   default: both
        [--top-k K]        qualitative strip size (top-k + bottom-k by trans err)
        [--synthesize N SIGMA_T_MM SIGMA_R_DEG]
                           generate N synthetic NPZ files in <pose_viz_dir>
                           with N(0, sigma) noise and exit. For smoke-testing.
"""

from __future__ import annotations

import argparse
import json
import re
from glob import glob
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Conventions match the wrapper: rot_matrix_to_euler_xyz in
# crisp_drl/agents/shared/insertion_wrapper_s.py and
# axis_angle_from_rotation_matrix in
# crisp_drl/agents/shared/env_wrappers.py. Inlined here to keep the
# analyzer free of ROS / pose-estimator import chains.

def rot_matrix_to_euler_xyz(rot: np.ndarray) -> np.ndarray:
    sy = np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
    if sy < 1e-6:
        return np.array([np.arctan2(-rot[1, 2], rot[1, 1]),
                         np.arctan2(-rot[2, 0], sy), 0.0])
    return np.array([np.arctan2(rot[2, 1], rot[2, 2]),
                     np.arctan2(-rot[2, 0], sy),
                     np.arctan2(rot[1, 0], rot[0, 0])])


def axis_angle_from_rotation_matrix(mat: np.ndarray) -> np.ndarray:
    cos_theta = (np.trace(mat) - 1.0) / 2.0
    theta = float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    if theta < 1e-8:
        return np.zeros(3)
    axis = np.array([mat[2, 1] - mat[1, 2],
                     mat[0, 2] - mat[2, 0],
                     mat[1, 0] - mat[0, 1]])
    denom = 2.0 * np.sin(theta)
    if abs(denom) < 1e-8:
        return np.zeros(3)
    return axis / denom * theta


def rotation_matrix_from_axis_angle(axis_angle: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(axis_angle))
    if theta < 1e-8:
        return np.eye(3)
    axis = axis_angle / theta
    kx, ky, kz = axis
    K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


EP_RE = re.compile(r"ep(\d+)_raw\.npz$")


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #

def load_episodes(viz_dir: Path) -> list[dict]:
    """Walk pe_viz_dir, return one record per episode sorted by ep idx.

    Each record has: ep, world_coarse (4x4), world_refined (4x4),
    det_r_coarse, orth_err_coarse, det_r_refined, orth_err_refined.
    """
    records: dict[int, dict] = {}
    for path in sorted(glob(str(viz_dir / "ep*_raw.npz"))):
        m = EP_RE.search(path)
        if m is None:
            continue
        ep = int(m.group(1))
        d = np.load(path)
        rec = {
            "ep": ep,
            "world_coarse": np.asarray(d["world_pose_coarse"], dtype=np.float64),
            "world_refined": np.asarray(d["world_pose_refined"], dtype=np.float64),
            "det_r_coarse": float(d["pose_check_coarse_det_r"][0]),
            "orth_err_coarse": float(d["pose_check_coarse_orth_err"][0]),
            "det_r_refined": float(d["pose_check_refined_det_r"][0]),
            "orth_err_refined": float(d["pose_check_refined_orth_err"][0]),
        }
        # Newest file wins if --snap_reinforce caused a re-write of the same ep.
        records[ep] = rec
    if not records:
        raise SystemExit(f"No ep*_raw.npz files found in {viz_dir}")
    return [records[k] for k in sorted(records)]


def decompose(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """4x4 SE(3) -> (xyz_mm, rpy_deg)."""
    pos_mm = M[:3, 3] * 1000.0
    rpy_deg = np.rad2deg(rot_matrix_to_euler_xyz(M[:3, :3]))
    return pos_mm, rpy_deg


def axis_angle_deg(M: np.ndarray) -> float:
    return float(np.rad2deg(np.linalg.norm(axis_angle_from_rotation_matrix(M[:3, :3]))))


def rotation_error_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic rotation distance between two 3x3 rotations, in degrees."""
    R_delta = R_a.T @ R_b
    return float(np.rad2deg(np.linalg.norm(axis_angle_from_rotation_matrix(R_delta))))


def build_samples(records: list[dict]) -> dict[str, dict]:
    """Stack per-episode decomposed values into arrays for 'coarse' and 'refined'."""
    out: dict[str, dict] = {}
    for which in ("coarse", "refined"):
        key = f"world_{which}"
        pos = np.stack([decompose(r[key])[0] for r in records], axis=0)  # (N, 3) mm
        rpy = np.stack([decompose(r[key])[1] for r in records], axis=0)  # (N, 3) deg
        rmats = np.stack([r[key][:3, :3] for r in records], axis=0)      # (N, 3, 3)
        eps = np.array([r["ep"] for r in records], dtype=np.int64)
        det_r = np.array([r[f"det_r_{which}"] for r in records])
        orth = np.array([r[f"orth_err_{which}"] for r in records])
        out[which] = dict(
            eps=eps, pos=pos, rpy=rpy, rmats=rmats, det_r=det_r, orth=orth,
        )
    return out


# --------------------------------------------------------------------------- #
# Errors / stats                                                              #
# --------------------------------------------------------------------------- #

def compute_errors(samp: dict, gt_pos: np.ndarray, gt_R: np.ndarray) -> dict:
    """Per-episode signed translation (mm) + rpy (deg) + geodesic (deg) errors."""
    trans_err = samp["pos"] - gt_pos                # (N, 3) mm
    # Signed Euler error is fine because we keep the same chart (no wrap risk
    # within +/- a few deg of pseudo-GT for a fixed lid pose).
    rpy_err = samp["rpy"] - np.rad2deg(rot_matrix_to_euler_xyz(gt_R))
    geo_err = np.array([rotation_error_deg(gt_R, R) for R in samp["rmats"]])
    trans_norm = np.linalg.norm(trans_err, axis=1)
    return dict(
        trans=trans_err, rpy=rpy_err, geo=geo_err, trans_norm=trans_norm,
    )


def dof_stats(arr: np.ndarray, labels: list[str]) -> dict:
    """Return mean / median / std / p95 / max-abs per column."""
    out = {}
    arr = np.atleast_2d(arr)
    if arr.shape[0] < arr.shape[1] and len(labels) == arr.shape[0]:
        arr = arr.T  # (N, k)
    for i, lab in enumerate(labels):
        col = arr[:, i]
        out[lab] = dict(
            mean=float(np.mean(col)),
            median=float(np.median(col)),
            std=float(np.std(col)),
            p95=float(np.percentile(np.abs(col), 95)),
            max_abs=float(np.max(np.abs(col))),
        )
    return out


def summarize(errs: dict) -> dict:
    return dict(
        translation_mm=dof_stats(errs["trans"], ["x", "y", "z"]),
        rotation_deg=dof_stats(errs["rpy"], ["roll", "pitch", "yaw"]),
        geodesic_deg=dof_stats(errs["geo"].reshape(-1, 1), ["theta"]),
        trans_norm_mm=dof_stats(errs["trans_norm"].reshape(-1, 1), ["norm"]),
    )


# --------------------------------------------------------------------------- #
# CSV / JSON persistence                                                      #
# --------------------------------------------------------------------------- #

def write_csv(samples: dict, errs_by_which: dict, path: Path) -> None:
    cols = [
        "ep", "which",
        "x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg",
        "trans_err_x_mm", "trans_err_y_mm", "trans_err_z_mm",
        "rpy_err_roll_deg", "rpy_err_pitch_deg", "rpy_err_yaw_deg",
        "geodesic_err_deg", "trans_norm_mm",
        "det_r", "orth_err",
    ]
    lines = [",".join(cols)]
    for which in ("coarse", "refined"):
        s, e = samples[which], errs_by_which[which]
        for i in range(len(s["eps"])):
            row = [
                s["eps"][i], which,
                *s["pos"][i], *s["rpy"][i],
                *e["trans"][i], *e["rpy"][i],
                e["geo"][i], e["trans_norm"][i],
                s["det_r"][i], s["orth"][i],
            ]
            lines.append(",".join(f"{v:.6g}" if not isinstance(v, str) else v
                                  for v in row))
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Plotting                                                                    #
# --------------------------------------------------------------------------- #

def _scatter_xy(ax, x, y, c, c_label, gt_label):
    sc = ax.scatter(x, y, c=c, cmap="coolwarm", s=80, alpha=0.85,
                    edgecolors="black", linewidth=0.4)
    plt.colorbar(sc, ax=ax, label=c_label)
    ax.axhline(0, color="gray", ls="--", lw=0.7, alpha=0.6)
    ax.axvline(0, color="gray", ls="--", lw=0.7, alpha=0.6)
    ax.scatter([0], [0], c="red", marker="+", s=120, lw=2, zorder=5,
               label=f"{gt_label} (0,0)")
    mx, my = float(np.mean(x)), float(np.mean(y))
    ax.scatter([mx], [my], c="orange", marker="x", s=120, lw=2, zorder=5,
               label=f"Mean ({mx:.2f}, {my:.2f})")
    rng = max(np.max(np.abs(x)), np.max(np.abs(y)), 1e-3) * 1.25
    ax.set_xlim(-rng, rng); ax.set_ylim(-rng, rng)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.legend(loc="upper right",
                                                                 fontsize=8)


def _hist(ax, vals, label, unit):
    ax.hist(vals, bins=20, color="steelblue", edgecolor="black", alpha=0.75)
    s = float(np.std(vals))
    p95 = float(np.percentile(np.abs(vals), 95))
    ax.axvline(np.mean(vals), color="orange", lw=1.5, label=f"mean {np.mean(vals):+.3f}")
    ax.axvline(s, color="green", ls="--", lw=1.0, label=f"+std {s:.3f}")
    ax.axvline(-s, color="green", ls="--", lw=1.0)
    ax.axvline(p95, color="red", ls=":", lw=1.0, label=f"p95(|·|) {p95:.3f}")
    ax.axvline(-p95, color="red", ls=":", lw=1.0)
    ax.set_xlabel(f"{label} ({unit})"); ax.set_ylabel("count")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)


def plot_distribution(samples, errs_by_which, gt_label, which_flag, out_path: Path):
    """4 rows x 3 cols figure. Plots refined by default; if `which_flag=='both'`,
    refined points are filled circles and coarse points are unfilled squares
    overlaid for direct comparison (only on Row 1)."""
    fig = plt.figure(figsize=(15, 18))
    gs = fig.add_gridspec(4, 3, hspace=0.45, wspace=0.35)

    primary = "refined" if which_flag in ("refined", "both") else "coarse"
    e = errs_by_which[primary]

    # ---- Row 1: translation scatters (XY, XZ, YZ) colored by 3rd axis ---- #
    for i, (ix, iy, iz, lbl) in enumerate([
        (0, 1, 2, "Z err (mm)"),
        (0, 2, 1, "Y err (mm)"),
        (1, 2, 0, "X err (mm)"),
    ]):
        ax = fig.add_subplot(gs[0, i])
        _scatter_xy(ax, e["trans"][:, ix], e["trans"][:, iy], e["trans"][:, iz],
                    lbl, gt_label)
        ax.set_xlabel(f"{'XYZ'[ix]} err (mm)")
        ax.set_ylabel(f"{'XYZ'[iy]} err (mm)")
        ax.set_title(f"Translation: {'XYZ'[ix]}–{'XYZ'[iy]}  ({primary})")
        if which_flag == "both":
            ec = errs_by_which["coarse"]
            ax.scatter(ec["trans"][:, ix], ec["trans"][:, iy], s=40,
                       facecolors="none", edgecolors="black", lw=0.6,
                       alpha=0.5, label="coarse")
            ax.legend(loc="upper right", fontsize=7)

    # ---- Row 2: rotation scatter + axis-angle hist + yaw hist ---- #
    ax = fig.add_subplot(gs[1, 0])
    sc = ax.scatter(e["rpy"][:, 0], e["rpy"][:, 1], c=e["rpy"][:, 2],
                    cmap="coolwarm", s=80, alpha=0.85,
                    edgecolors="black", linewidth=0.4)
    plt.colorbar(sc, ax=ax, label="yaw err (deg)")
    ax.axhline(0, color="gray", ls="--", lw=0.7, alpha=0.6)
    ax.axvline(0, color="gray", ls="--", lw=0.7, alpha=0.6)
    ax.scatter([0], [0], c="red", marker="+", s=120, lw=2)
    ax.set_xlabel("roll err (deg)"); ax.set_ylabel("pitch err (deg)")
    ax.set_title(f"Rotation: roll–pitch  ({primary})")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    _hist(ax, e["geo"], "geodesic rotation err θ", "deg")
    ax.set_title(f"Rotation magnitude θ  ({primary})")

    ax = fig.add_subplot(gs[1, 2])
    _hist(ax, e["rpy"][:, 2], "yaw err", "deg")
    ax.set_title(f"Yaw err  ({primary})")

    # ---- Rows 2 & 3: 6 marginal histograms (3 translation, 3 rotation) ---- #
    units = ["mm", "mm", "mm", "deg", "deg", "deg"]
    labels = ["x", "y", "z", "roll", "pitch", "yaw"]
    data = [e["trans"][:, 0], e["trans"][:, 1], e["trans"][:, 2],
            e["rpy"][:, 0], e["rpy"][:, 1], e["rpy"][:, 2]]
    for i, (vals, lab, u) in enumerate(zip(data, labels, units)):
        row = 2 if i < 3 else 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        _hist(ax, vals, f"{lab} err", u)
        ax.set_title(f"{lab} marginal  ({primary})")
    # Coarse → refined drift goes to a separate file via plot_drift().

    fig.suptitle(f"PE distribution vs {gt_label}  (N={len(e['geo'])}, "
                 f"{primary})", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_drift(samples, out_path: Path):
    """Coarse→refined drift: quiver of XY translation drift +
    histograms of |Δtranslation| and |Δrotation|."""
    sc, sr = samples["coarse"], samples["refined"]
    if len(sc["eps"]) == 0 or len(sr["eps"]) == 0:
        return
    # Align by ep idx
    common = sorted(set(sc["eps"]) & set(sr["eps"]))
    if not common:
        return
    idx_c = {int(e): i for i, e in enumerate(sc["eps"])}
    idx_r = {int(e): i for i, e in enumerate(sr["eps"])}
    pos_c = np.stack([sc["pos"][idx_c[e]] for e in common])
    pos_r = np.stack([sr["pos"][idx_r[e]] for e in common])
    rmat_c = np.stack([sc["rmats"][idx_c[e]] for e in common])
    rmat_r = np.stack([sr["rmats"][idx_r[e]] for e in common])
    dpos = pos_r - pos_c
    drot = np.array([rotation_error_deg(rmat_c[i], rmat_r[i])
                     for i in range(len(common))])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    ax.quiver(pos_c[:, 0], pos_c[:, 1], dpos[:, 0], dpos[:, 1],
              angles="xy", scale_units="xy", scale=1.0,
              color="steelblue", width=0.004)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_xlabel("coarse x (mm)"); ax.set_ylabel("coarse y (mm)")
    ax.set_title("Coarse → refined XY drift")
    _hist(axes[1], np.linalg.norm(dpos, axis=1), "|Δtranslation|", "mm")
    axes[1].set_title("Refinement translation magnitude")
    _hist(axes[2], drot, "|Δrotation|", "deg")
    axes[2].set_title("Refinement rotation magnitude")
    fig.suptitle("Coarse → refined PE drift (smaller = more stable two-shot PE)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_quality(samples: dict, out_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for which, color in (("coarse", "steelblue"), ("refined", "darkorange")):
        s = samples[which]
        axes[0].plot(s["eps"], s["det_r"], "o-", color=color, label=which)
        axes[1].plot(s["eps"], s["orth"], "o-", color=color, label=which)
    axes[0].axhline(1.0, color="black", ls="--", lw=0.6, alpha=0.4)
    axes[0].set_ylabel("det(R)  (1 = ideal)")
    axes[1].axhline(0.0, color="black", ls="--", lw=0.6, alpha=0.4)
    axes[1].set_ylabel("orthogonality err")
    axes[1].set_xlabel("episode")
    for ax in axes:
        ax.grid(True, alpha=0.3); ax.legend()
    fig.suptitle("Rotation-matrix validity per episode")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _find_overlay(viz_dir: Path, ep: int, kind: str) -> Optional[Path]:
    matches = sorted(viz_dir.glob(f"ep{ep:03d}_{kind}_*.png"))
    return matches[-1] if matches else None


def plot_qualitative_strip(viz_dir: Path, samples, errs_by_which, top_k: int,
                            which: str, out_path: Path):
    e = errs_by_which[which]
    s = samples[which]
    order = np.argsort(e["trans_norm"])
    best = order[:top_k]
    worst = order[-top_k:][::-1]
    rows = list(best) + list(worst)
    labels = [f"BEST #{i+1}" for i in range(len(best))] + \
             [f"WORST #{i+1}" for i in range(len(worst))]

    n_rows = len(rows)
    kinds = ["coarse", "refined", "sidebyside"]
    fig, axes = plt.subplots(n_rows, len(kinds),
                             figsize=(4.5 * len(kinds), 2.4 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])
    for r, i in enumerate(rows):
        ep = int(s["eps"][i])
        terr = float(e["trans_norm"][i])
        gerr = float(e["geo"][i])
        for c, kind in enumerate(kinds):
            ax = axes[r, c]
            ax.set_axis_off()
            png = _find_overlay(viz_dir, ep, kind)
            if png is None:
                ax.text(0.5, 0.5, f"(no {kind}.png\nfor ep{ep:03d})",
                        ha="center", va="center")
                continue
            img = cv2.imread(str(png))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            if c == 0:
                ax.set_title(f"{labels[r]}  ep{ep:03d}\n"
                             f"|t|={terr:.2f} mm  θ={gerr:.2f}°",
                             fontsize=9, loc="left")
            else:
                ax.set_title(kind, fontsize=9)
    fig.suptitle(f"Qualitative strip ({which}): best {top_k} / worst {top_k} "
                 f"by ‖translation error‖", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Synthesis (smoke test)                                                      #
# --------------------------------------------------------------------------- #

def synthesize(viz_dir: Path, n: int, sigma_t_mm: float, sigma_r_deg: float):
    viz_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    gt_pos = np.array([0.5, 0.0, 0.05])
    gt_R = np.eye(3)
    sigma_t = sigma_t_mm / 1000.0
    sigma_r = np.deg2rad(sigma_r_deg)
    for ep in range(n):
        # Noise: gaussian on translation, axis-angle on rotation.
        dt = rng.normal(scale=sigma_t, size=3)
        da = rng.normal(scale=sigma_r, size=3)
        R = gt_R @ rotation_matrix_from_axis_angle(da)
        M = np.eye(4); M[:3, :3] = R; M[:3, 3] = gt_pos + dt
        Mr = M.copy()
        # Refined: same noise distribution but independent draw.
        dt2 = rng.normal(scale=sigma_t, size=3)
        da2 = rng.normal(scale=sigma_r, size=3)
        R2 = gt_R @ rotation_matrix_from_axis_angle(da2)
        Mr[:3, :3] = R2; Mr[:3, 3] = gt_pos + dt2
        np.savez_compressed(
            viz_dir / f"ep{ep:03d}_raw.npz",
            world_pose_coarse=M, world_pose_refined=Mr,
            pose_check_coarse_det_r=np.array([1.0], dtype=np.float32),
            pose_check_coarse_orth_err=np.array([0.0], dtype=np.float32),
            pose_check_refined_det_r=np.array([1.0], dtype=np.float32),
            pose_check_refined_orth_err=np.array([0.0], dtype=np.float32),
        )
    print(f"Synthesized {n} episodes in {viz_dir} "
          f"(sigma_t={sigma_t_mm} mm, sigma_r={sigma_r_deg} deg)")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("viz_dir", type=Path,
                    help="pe_viz subdirectory containing ep*_raw.npz")
    ap.add_argument("--gt-episode", type=int, default=None,
                    help="anchor errors to ep{N} (visual GT) in addition to "
                         "the pseudo-GT (median)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--which", choices=["coarse", "refined", "both"],
                    default="both")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--synthesize", type=int, default=None,
                    help="generate N synthetic NPZ files into viz_dir and exit")
    ap.add_argument("--sigma-t-mm", type=float, default=0.5,
                    help="(synth) translation std in mm")
    ap.add_argument("--sigma-r-deg", type=float, default=0.5,
                    help="(synth) rotation std in deg")
    args = ap.parse_args()

    if args.synthesize is not None:
        synthesize(args.viz_dir, args.synthesize,
                   args.sigma_t_mm, args.sigma_r_deg)
        return

    out_dir = args.out_dir or (args.viz_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_episodes(args.viz_dir)
    samples = build_samples(records)
    print(f"[loaded] {len(records)} episodes from {args.viz_dir}")

    # Pseudo-GT = component-wise median (robust to PE failures).
    gt_specs = []
    for label, which in (("pseudo_gt_median_coarse", "coarse"),
                         ("pseudo_gt_median_refined", "refined")):
        s = samples[which]
        gt_pos = np.median(s["pos"], axis=0)
        # Use the rmat closest to median rpy as a rotation proxy. Simpler than
        # a proper rotation-mean — fine because dispersion is small.
        med_rpy = np.median(s["rpy"], axis=0)
        i_med = int(np.argmin(np.linalg.norm(s["rpy"] - med_rpy, axis=1)))
        gt_R = s["rmats"][i_med]
        gt_specs.append((label, which, gt_pos, gt_R))

    if args.gt_episode is not None:
        for which in ("coarse", "refined"):
            s = samples[which]
            mask = s["eps"] == args.gt_episode
            if not mask.any():
                print(f"[warn] --gt-episode {args.gt_episode} not in samples "
                      f"({which}); skipping visual-GT for {which}")
                continue
            i = int(np.where(mask)[0][0])
            gt_specs.append(
                (f"visual_gt_ep{args.gt_episode}_{which}", which,
                 s["pos"][i], s["rmats"][i]))

    # Compute errors + stats for every GT spec.
    stats_out = {}
    errs_by_which_pseudo = {}
    for label, which, gt_pos, gt_R in gt_specs:
        errs = compute_errors(samples[which], gt_pos, gt_R)
        stats_out[label] = summarize(errs)
        if label.startswith("pseudo_gt_median_"):
            errs_by_which_pseudo[which] = errs

    # CSV (uses pseudo-GT errors).
    write_csv(samples, errs_by_which_pseudo, out_dir / "pe_samples.csv")
    (out_dir / "pe_stats.json").write_text(json.dumps(stats_out, indent=2))

    # Plots vs pseudo-GT.
    plot_distribution(samples, errs_by_which_pseudo, "pseudo-GT (median)",
                      args.which, out_dir / "pe_distribution.png")
    plot_quality(samples, out_dir / "pe_quality.png")
    plot_drift(samples, out_dir / "pe_drift_coarse_to_refined.png")
    primary_which = "refined" if args.which in ("refined", "both") else "coarse"
    plot_qualitative_strip(args.viz_dir, samples, errs_by_which_pseudo,
                            args.top_k, primary_which,
                            out_dir / "qualitative_strip.png")

    # If a visual GT was given, emit a parallel distribution figure.
    if args.gt_episode is not None:
        errs_by_which_visual = {}
        for label, which, gt_pos, gt_R in gt_specs:
            if not label.startswith("visual_gt_"):
                continue
            errs_by_which_visual[which] = compute_errors(
                samples[which], gt_pos, gt_R)
        if errs_by_which_visual:
            plot_distribution(
                samples, errs_by_which_visual,
                f"visual GT (ep{args.gt_episode})", args.which,
                out_dir / "pe_distribution_visualgt.png")

    print(f"[done] wrote analysis to {out_dir}")
    for k, v in stats_out.items():
        t = v["translation_mm"]
        r = v["rotation_deg"]
        g = v["geodesic_deg"]["theta"]
        print(f"  {k}:")
        print(f"    trans std (mm): x={t['x']['std']:.3f}  "
              f"y={t['y']['std']:.3f}  z={t['z']['std']:.3f}")
        print(f"    rot   std (deg): roll={r['roll']['std']:.3f}  "
              f"pitch={r['pitch']['std']:.3f}  yaw={r['yaw']['std']:.3f}")
        print(f"    geodesic θ: mean={g['mean']:.3f}  p95={g['p95']:.3f}  "
              f"max={g['max_abs']:.3f}")


if __name__ == "__main__":
    main()
