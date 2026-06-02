#!/usr/bin/env python3
"""Compute force/torque statistics from a force_torque_measurements.jsonl file.

Each JSONL line = one episode with `observations`: (T, 6) array of
[fx, fy, fz, tx, ty, tz]. Output mirrors the FT block written by
crisp_drl.agents.shared.eval_actor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def per_episode_stats(ft: np.ndarray) -> tuple[dict, dict]:
    f_xyz, t_xyz = ft[:, 0:3], ft[:, 3:6]
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
    ap.add_argument(
        "jsonl",
        type=Path,
        help="Path to force_torque_measurements.jsonl",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <jsonl_dir>/ft_statistics.json)",
    )
    args = ap.parse_args()

    out_path = args.output or args.jsonl.with_name("ft_statistics.json")

    episodes: list[dict] = []
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
            force, torque = per_episode_stats(obs)
            episodes.append(
                {
                    "episode": i,
                    "datetime": d.get("datetime"),
                    "length": int(obs.shape[0]),
                    "force_N": force,
                    "torque_Nm": torque,
                }
            )

    if not episodes:
        raise SystemExit("No episodes found in input file.")

    f_mean = np.array([e["force_N"]["mean"] for e in episodes])
    f_max = np.array([e["force_N"]["max"] for e in episodes])
    t_mean = np.array([e["torque_Nm"]["mean"] for e in episodes])
    t_max = np.array([e["torque_Nm"]["max"] for e in episodes])
    lengths = np.array([e["length"] for e in episodes])

    summary = {
        "source": str(args.jsonl),
        "n_episodes": len(episodes),
        "episode_length": agg_stats(lengths.astype(np.float64)),
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
    }

    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}  ({len(episodes)} episodes)")
    print(
        f"  force_N  episode_max  mean={summary['force_N']['episode_max_stats']['mean']:.3f} "
        f"max={summary['force_N']['episode_max_stats']['max']:.3f}"
    )
    print(
        f"  torque_Nm episode_max mean={summary['torque_Nm']['episode_max_stats']['mean']:.4f} "
        f"max={summary['torque_Nm']['episode_max_stats']['max']:.4f}"
    )


if __name__ == "__main__":
    main()
