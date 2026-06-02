#!/usr/bin/env python3
"""Policy-only force/torque statistics from force_torque_measurements.jsonl.

For each episode, the lift event is detected with the same robust algorithm
used by `scripts/plot_ft_random_episodes.py` (multi-scale |T| contrast +
small fz-rise bonus, Savitzky-Golay smoothing). Stats are then computed over
the `--policy_steps` samples immediately preceding the lift index — i.e. the
green band on the plots.

Output mirrors `scripts/ft_stats_from_jsonl.py`, just restricted to the
policy window.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from scipy.signal import savgol_filter as _savgol_filter
    _HAVE_SCIPY = True
except ImportError:
    _savgol_filter = None
    _HAVE_SCIPY = False


def _smooth(signal: np.ndarray, win: int, poly: int) -> np.ndarray:
    if signal.size == 0:
        return signal
    win = max(3, win | 1)
    if win >= signal.size:
        win = signal.size if signal.size % 2 == 1 else signal.size - 1
        if win < 3:
            return signal.copy()
    if _HAVE_SCIPY:
        return _savgol_filter(signal, window_length=win, polyorder=min(poly, win - 1))
    kernel = np.ones(win) / win
    return np.convolve(signal, kernel, mode="same")


def detect_lift(
    ft: np.ndarray,
    mark_lo: int = 200,
    mark_hi: int = 600,
    smooth_win: int = 21,
    smooth_poly: int = 3,
    contrast_windows: tuple[int, ...] = (8, 16, 32),
    score_high_threshold: float = 0.6,
    hysteresis_steps: int = 15,
    hysteresis_frac: float = 0.3,
):
    T = ft.shape[0]
    t_norm = np.linalg.norm(ft[:, 3:6], axis=1)
    t_s = _smooth(t_norm, smooth_win, smooth_poly)
    fz_s = _smooth(ft[:, 2], smooth_win, smooth_poly)

    amp_lo = min(150, max(0, T // 3))
    amp_hi = max(amp_lo + 1, T - 20)
    if amp_hi - amp_lo < 20:
        return None, "none", 0.0, f"episode too short (T={T})"
    seg_t = t_s[amp_lo:amp_hi]
    t_high = float(np.percentile(seg_t, 95))
    t_low = float(np.percentile(seg_t, 10))
    t_amp = t_high - t_low
    if t_amp < 0.05:
        return None, "none", t_amp, f"flat |T| (amp={t_amp:.3f})"

    Wmax = max(contrast_windows)
    lo = max(mark_lo, Wmax)
    hi = min(mark_hi, T - Wmax)
    if hi - lo < 2:
        return None, "none", t_amp, "no candidate range"

    idx = np.arange(lo, hi)
    cs_t = np.concatenate(([0.0], np.cumsum(t_s)))
    cs_f = np.concatenate(([0.0], np.cumsum(fz_s)))

    t_drop_avg = np.zeros_like(idx, dtype=np.float64)
    fz_rise_avg = np.zeros_like(idx, dtype=np.float64)
    for w in contrast_windows:
        left_t = (cs_t[idx] - cs_t[idx - w]) / w
        right_t = (cs_t[idx + w] - cs_t[idx]) / w
        left_f = (cs_f[idx] - cs_f[idx - w]) / w
        right_f = (cs_f[idx + w] - cs_f[idx]) / w
        t_drop_avg += (left_t - right_t)
        fz_rise_avg += (right_f - left_f)
    t_drop_avg /= len(contrast_windows)
    fz_rise_avg /= len(contrast_windows)
    score = t_drop_avg + 0.05 * np.maximum(fz_rise_avg, 0.0)
    score_frac = t_drop_avg / max(t_amp, 1e-3)

    order = np.argsort(-score)
    thr = t_low + hysteresis_frac * t_amp
    chosen_local = int(order[0])
    chosen_passes = False
    for local in order:
        i = int(idx[local])
        j = min(T, i + hysteresis_steps)
        if j - i < 3:
            continue
        if np.all(t_s[i:j] < thr):
            chosen_local = int(local)
            chosen_passes = True
            break

    lift_idx = int(idx[chosen_local])
    best_frac = float(score_frac[chosen_local])
    if t_amp >= 0.10 and best_frac >= score_high_threshold and chosen_passes:
        conf = "high"
    else:
        conf = "low"
    return lift_idx, conf, t_amp, f"frac={best_frac:.2f} amp={t_amp:.3f}"


def per_window_stats(ft_window: np.ndarray) -> tuple[dict, dict]:
    f_xyz, t_xyz = ft_window[:, 0:3], ft_window[:, 3:6]
    f_norm = np.linalg.norm(f_xyz, axis=1)
    t_norm = np.linalg.norm(t_xyz, axis=1)
    force = {
        "mean": float(f_norm.mean()),
        "max": float(f_norm.max()),
        "fx_mean": float(f_xyz[:, 0].mean()),
        "fy_mean": float(f_xyz[:, 1].mean()),
        "fz_mean": float(f_xyz[:, 2].mean()),
        "fx_max_abs": float(np.max(np.abs(f_xyz[:, 0]))),
        "fy_max_abs": float(np.max(np.abs(f_xyz[:, 1]))),
        "fz_max_abs": float(np.max(np.abs(f_xyz[:, 2]))),
    }
    torque = {
        "mean": float(t_norm.mean()),
        "max": float(t_norm.max()),
        "tx_mean": float(t_xyz[:, 0].mean()),
        "ty_mean": float(t_xyz[:, 1].mean()),
        "tz_mean": float(t_xyz[:, 2].mean()),
        "tx_max_abs": float(np.max(np.abs(t_xyz[:, 0]))),
        "ty_max_abs": float(np.max(np.abs(t_xyz[:, 1]))),
        "tz_max_abs": float(np.max(np.abs(t_xyz[:, 2]))),
    }
    return force, torque


def agg_stats(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {"mean": None, "std": None, "max": None, "min": None}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size >= 2 else 0.0,
        "max": float(arr.max()),
        "min": float(arr.min()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Default: <jsonl_dir>/ft_statistics_policy.json")
    ap.add_argument("--policy_steps", type=int, default=30,
                    help="Number of steps before lift index to include.")
    ap.add_argument("--include_low_conf", action="store_true",
                    help="Also include low-confidence detections.")
    # Detection params (kept in sync with plot_ft_random_episodes.py)
    ap.add_argument("--mark_lo", type=int, default=200)
    ap.add_argument("--mark_hi", type=int, default=600)
    ap.add_argument("--smooth_win", type=int, default=21)
    ap.add_argument("--smooth_poly", type=int, default=3)
    ap.add_argument("--contrast_windows", default="8,16,32")
    ap.add_argument("--score_high_threshold", type=float, default=0.6)
    ap.add_argument("--hysteresis_steps", type=int, default=15)
    ap.add_argument("--hysteresis_frac", type=float, default=0.3)
    args = ap.parse_args()

    cw = tuple(int(s) for s in args.contrast_windows.split(",") if s)

    out_path = args.output or args.jsonl.with_name("ft_statistics_policy.json")

    episodes: list[dict] = []
    skipped: list[dict] = []
    with args.jsonl.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            obs = np.asarray(d["observations"], dtype=np.float64)
            if obs.ndim != 2 or obs.shape[1] != 6:
                raise ValueError(
                    f"Episode {i}: expected (T,6) observations, got {obs.shape}"
                )
            lift_idx, conf, t_amp, reason = detect_lift(
                obs,
                mark_lo=args.mark_lo,
                mark_hi=args.mark_hi,
                smooth_win=args.smooth_win,
                smooth_poly=args.smooth_poly,
                contrast_windows=cw,
                score_high_threshold=args.score_high_threshold,
                hysteresis_steps=args.hysteresis_steps,
                hysteresis_frac=args.hysteresis_frac,
            )
            if lift_idx is None:
                skipped.append({"episode": i, "reason": f"no lift: {reason}",
                                "datetime": d.get("datetime")})
                continue
            if conf == "low" and not args.include_low_conf:
                skipped.append({"episode": i, "lift_idx": lift_idx,
                                "reason": f"low conf: {reason}",
                                "datetime": d.get("datetime")})
                continue

            policy_lo = max(0, lift_idx - args.policy_steps)
            window = obs[policy_lo:lift_idx]
            if window.shape[0] < 2:
                skipped.append({"episode": i, "lift_idx": lift_idx,
                                "reason": "window too short",
                                "datetime": d.get("datetime")})
                continue
            force, torque = per_window_stats(window)
            episodes.append({
                "episode": i,
                "datetime": d.get("datetime"),
                "lift_idx": lift_idx,
                "policy_window": [policy_lo, lift_idx],
                "policy_length": int(window.shape[0]),
                "conf": conf,
                "t_amp": float(t_amp),
                "force_N": force,
                "torque_Nm": torque,
            })

    if not episodes:
        raise SystemExit("No episodes with detected lift; nothing to aggregate.")

    f_mean = np.array([e["force_N"]["mean"] for e in episodes])
    f_max = np.array([e["force_N"]["max"] for e in episodes])
    t_mean = np.array([e["torque_Nm"]["mean"] for e in episodes])
    t_max = np.array([e["torque_Nm"]["max"] for e in episodes])

    summary = {
        "source": str(args.jsonl),
        "policy_steps": args.policy_steps,
        "include_low_conf": bool(args.include_low_conf),
        "n_episodes_total": len(episodes) + len(skipped),
        "n_episodes_used": len(episodes),
        "n_skipped": len(skipped),
        "n_high_conf": sum(1 for e in episodes if e["conf"] == "high"),
        "n_low_conf": sum(1 for e in episodes if e["conf"] == "low"),
        "force_N": {
            "episode_mean_stats": agg_stats(f_mean),
            "episode_max_stats": agg_stats(f_max),
            "fx_max_abs_over_all": float(
                np.max([e["force_N"]["fx_max_abs"] for e in episodes])
            ),
            "fy_max_abs_over_all": float(
                np.max([e["force_N"]["fy_max_abs"] for e in episodes])
            ),
            "fz_max_abs_over_all": float(
                np.max([e["force_N"]["fz_max_abs"] for e in episodes])
            ),
        },
        "torque_Nm": {
            "episode_mean_stats": agg_stats(t_mean),
            "episode_max_stats": agg_stats(t_max),
            "tx_max_abs_over_all": float(
                np.max([e["torque_Nm"]["tx_max_abs"] for e in episodes])
            ),
            "ty_max_abs_over_all": float(
                np.max([e["torque_Nm"]["ty_max_abs"] for e in episodes])
            ),
            "tz_max_abs_over_all": float(
                np.max([e["torque_Nm"]["tz_max_abs"] for e in episodes])
            ),
        },
        "episodes": episodes,
        "skipped": skipped,
    }

    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")
    print(f"  used    : {summary['n_episodes_used']}/{summary['n_episodes_total']} "
          f"(high={summary['n_high_conf']} low={summary['n_low_conf']})")
    print(f"  skipped : {summary['n_skipped']}")
    fme = summary["force_N"]["episode_mean_stats"]
    fma = summary["force_N"]["episode_max_stats"]
    tme = summary["torque_Nm"]["episode_mean_stats"]
    tma = summary["torque_Nm"]["episode_max_stats"]
    print(f"  policy force_N    mean={fme['mean']:.3f}±{fme['std']:.3f}  "
          f"max={fma['mean']:.3f}±{fma['std']:.3f}  peak={fma['max']:.3f}")
    print(f"  policy torque_Nm  mean={tme['mean']:.4f}±{tme['std']:.4f}  "
          f"max={tma['mean']:.4f}±{tma['std']:.4f}  peak={tma['max']:.4f}")


if __name__ == "__main__":
    main()
