#!/usr/bin/env python3
"""Plot force/torque curves over time for N random episodes from a
force_torque_measurements.jsonl file.

Each episode = one JSONL line with `observations` of shape (T, 6) =
[fx, fy, fz, tx, ty, tz].
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.signal import savgol_filter as _savgol_filter
    _HAVE_SCIPY = True
except ImportError:
    _savgol_filter = None
    _HAVE_SCIPY = False


def _smooth(signal: np.ndarray, win: int, poly: int) -> np.ndarray:
    """Savitzky-Golay smoothing with boxcar fallback."""
    if signal.size == 0:
        return signal
    win = max(3, win | 1)  # force odd, >= 3
    if win >= signal.size:
        win = signal.size if signal.size % 2 == 1 else signal.size - 1
        if win < 3:
            return signal.copy()
    if _HAVE_SCIPY:
        return _savgol_filter(signal, window_length=win, polyorder=min(poly, win - 1))
    kernel = np.ones(win) / win
    return np.convolve(signal, kernel, mode="same")


def load_episodes(path: Path) -> list[dict]:
    eps = []
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            eps.append(
                {
                    "idx": i,
                    "obs": np.asarray(d["observations"], dtype=np.float64),
                    "datetime": d.get("datetime"),
                }
            )
    return eps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("-n", "--n_episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dt", type=float, default=None,
                    help="Sample period in seconds (x-axis time). If omitted, x = step index.")
    ap.add_argument("-o", "--output_dir", type=Path, default=None,
                    help="Output dir for PNGs (default: <jsonl_dir>/ft_curves/)")
    ap.add_argument("--combined", action="store_true",
                    help="Also write one combined multi-row PNG.")
    ap.add_argument("--combined_zoom", action="store_true",
                    help="When set with --combined, x-limit the combined PNG to [zoom_lo, zoom_hi].")
    ap.add_argument("--mark_fz_event", action="store_true", default=True,
                    help="Detect & mark steepest fz drop in [--mark_lo, --mark_hi].")
    ap.add_argument("--no_mark_fz_event", dest="mark_fz_event",
                    action="store_false")
    ap.add_argument("--mark_lo", type=int, default=200)
    ap.add_argument("--mark_hi", type=int, default=600)
    ap.add_argument("--smooth_win", type=int, default=21,
                    help="Savitzky-Golay window length (forced odd).")
    ap.add_argument("--smooth_poly", type=int, default=3,
                    help="Savitzky-Golay polynomial order.")
    ap.add_argument("--policy_steps_before_lift", type=int, default=30,
                    help="Steps before lift event to shade as policy region.")
    ap.add_argument("--detect_method", choices=["robust", "simple"], default="robust",
                    help="robust = multi-scale, amplitude-normalised, fz-corroborated. "
                         "simple = single sliding-window |T| drop (legacy).")
    ap.add_argument("--contrast_windows", default="8,16,32",
                    help="Comma-separated half-window sizes for multi-scale contrast.")
    ap.add_argument("--score_high_threshold", type=float, default=0.6)
    ap.add_argument("--hysteresis_steps", type=int, default=15)
    ap.add_argument("--hysteresis_frac", type=float, default=0.3)
    ap.add_argument("--lift_window", type=int, default=10,
                    help="(simple only) stride for |T| drop.")
    ap.add_argument("--lift_min_drop_nm", type=float, default=0.10,
                    help="(simple only) reject lift if max |T| drop below this.")
    ap.add_argument("--zoom_lo", type=int, default=300)
    ap.add_argument("--zoom_hi", type=int, default=450)
    ap.add_argument("--no_zoom", action="store_true",
                    help="Disable extra zoomed PNGs.")
    ap.add_argument("--debug", action="store_true",
                    help="Print per-episode detection diagnostics for ALL eps in the JSONL.")
    args = ap.parse_args()

    eps = load_episodes(args.jsonl)
    if len(eps) == 0:
        raise SystemExit("No episodes in input.")

    rng = random.Random(args.seed)
    n = min(args.n_episodes, len(eps))
    chosen = rng.sample(eps, n)
    chosen.sort(key=lambda e: e["idx"])

    out_dir = args.output_dir or args.jsonl.with_name("ft_curves")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy-bound forward reference; defined below in main().
    _detect_holder = {}

    def _run_debug():
        # Diagnostic across all episodes (not just sampled).
        method = args.detect_method
        detect = _detect_holder["fn"]
        counts = {"high": 0, "low": 0, "none": 0}
        print(f"[debug] detector={method}  smooth_win={args.smooth_win} poly={args.smooth_poly}  "
              f"window=[{args.mark_lo},{args.mark_hi}]  contrast={contrast_windows}")
        print(f"{'ep':>3} {'T':>4} {'lift':>5} {'score':>7} {'t_amp':>7} {'conf':>5}  reason")
        for e in eps:
            r = detect(e["obs"])
            counts[r["conf"]] = counts.get(r["conf"], 0) + 1
            lift = r["lift_idx"] if r["lift_idx"] is not None else -1
            print(f"{e['idx']:>3} {e['obs'].shape[0]:>4} {lift:>5} "
                  f"{r['score']:>7.3f} {r['t_amp']:>7.3f} {r['conf']:>5}  {r.get('reason','')}")
        print(f"[debug] summary: high={counts.get('high',0)} "
              f"low={counts.get('low',0)} none={counts.get('none',0)}  total={len(eps)}")

    xlabel = "time [s]" if args.dt else "step"
    cmap = ["tab:red", "tab:green", "tab:blue"]
    f_labels = ["fx", "fy", "fz"]
    t_labels = ["tx", "ty", "tz"]

    contrast_windows = tuple(int(s) for s in args.contrast_windows.split(",") if s)

    def detect_lift_simple(ft: np.ndarray):
        lo = max(0, args.mark_lo)
        hi = min(ft.shape[0], args.mark_hi)
        w = max(1, args.lift_window)
        if hi - lo <= w:
            hi = ft.shape[0]
            if hi - lo <= w:
                return {"lift_idx": None, "score": 0.0, "conf": "none",
                        "t_amp": 0.0, "t_s": None, "reason": "window too small"}
        t_norm = np.linalg.norm(ft[:, 3:6], axis=1)
        t_s = _smooth(t_norm, args.smooth_win, args.smooth_poly)
        drops = t_s[lo:hi - w] - t_s[lo + w:hi]
        if drops.size == 0:
            return {"lift_idx": None, "score": 0.0, "conf": "none",
                    "t_amp": 0.0, "t_s": t_s, "reason": "empty"}
        best = int(np.argmax(drops))
        drop = float(drops[best])
        conf = "high" if drop >= args.lift_min_drop_nm else "low"
        return {"lift_idx": lo + best, "score": drop, "conf": conf,
                "t_amp": float(t_norm.max()), "t_s": t_s,
                "reason": f"Δ|T|={drop:.3f}"}

    def detect_lift_robust(ft: np.ndarray):
        """Multi-scale, amplitude-normalised, fz-corroborated lift detector.

        Returns dict: lift_idx, score, conf in {high, low, none}, t_amp, t_s, reason.
        """
        T = ft.shape[0]
        t_norm = np.linalg.norm(ft[:, 3:6], axis=1)
        t_s = _smooth(t_norm, args.smooth_win, args.smooth_poly)
        fz_s = _smooth(ft[:, 2], args.smooth_win, args.smooth_poly)

        # Amplitudes (adaptive). The episode can keep torque elevated after
        # lift (brick still gripped), so we can't rely on the tail being
        # settled. Instead: t_high = 95th pctl of the central band, t_low =
        # 10th pctl of the SAME band -- the moment of lowest torque must
        # appear somewhere if a real release happened.
        amp_lo = min(150, max(0, T // 3))
        amp_hi = max(amp_lo + 1, T - 20)
        if amp_hi - amp_lo < 20:
            return {"lift_idx": None, "score": 0.0, "conf": "none",
                    "t_amp": 0.0, "t_s": t_s, "fz_s": fz_s,
                    "reason": f"episode too short (T={T})"}
        seg_t = t_s[amp_lo:amp_hi]
        seg_f = fz_s[amp_lo:amp_hi]
        t_high = float(np.percentile(seg_t, 95))
        t_low = float(np.percentile(seg_t, 10))
        t_amp = t_high - t_low
        fz_low = float(np.percentile(seg_f, 10))
        fz_high = float(np.percentile(seg_f, 90))
        fz_amp = max(fz_high - fz_low, 0.0)

        if t_amp < 0.05:
            return {"lift_idx": None, "score": 0.0, "conf": "none",
                    "t_amp": t_amp, "t_s": t_s, "fz_s": fz_s,
                    "reason": f"flat |T| (amp={t_amp:.3f} Nm)"}

        Wmax = max(contrast_windows)
        lo = max(args.mark_lo, Wmax)
        hi = min(args.mark_hi, T - Wmax)
        if hi - lo < 2:
            return {"lift_idx": None, "score": 0.0, "conf": "none",
                    "t_amp": t_amp, "t_s": t_s, "fz_s": fz_s,
                    "reason": "no candidate range"}

        idx = np.arange(lo, hi)
        cs_t = np.concatenate(([0.0], np.cumsum(t_s)))
        cs_f = np.concatenate(([0.0], np.cumsum(fz_s)))

        # Multi-scale contrast in PHYSICAL units (Nm for |T|, N for fz).
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

        # Score: torque drop (Nm) + small bonus from positive fz rise (N).
        # fz_weight chosen so a typical 1 N fz rise contributes ~0.05 Nm-equivalent.
        fz_weight = 0.05
        score = t_drop_avg + fz_weight * np.maximum(fz_rise_avg, 0.0)

        # Normalised "fraction-of-amplitude" used for the confidence check.
        score_frac = t_drop_avg / max(t_amp, 1e-3)

        # Pick + hysteresis: prefer the highest-score candidate whose post-
        # lift |T| stays below floor + hysteresis_frac · amp for H steps.
        order = np.argsort(-score)
        H = args.hysteresis_steps
        thr = t_low + args.hysteresis_frac * t_amp
        chosen_local = int(order[0])
        chosen_passes = False
        for local in order:
            i = int(idx[local])
            j = min(T, i + H)
            if j - i < 3:
                continue
            if np.all(t_s[i:j] < thr):
                chosen_local = int(local)
                chosen_passes = True
                break

        lift_idx = int(idx[chosen_local])
        best_score = float(score[chosen_local])
        best_frac = float(score_frac[chosen_local])

        if t_amp >= 0.10 and best_frac >= args.score_high_threshold and chosen_passes:
            conf = "high"
            reason = f"frac={best_frac:.2f} amp={t_amp:.3f}"
        else:
            conf = "low"
            why = []
            if t_amp < 0.10:
                why.append(f"t_amp={t_amp:.3f}<0.10")
            if best_frac < args.score_high_threshold:
                why.append(f"frac={best_frac:.2f}<{args.score_high_threshold}")
            if not chosen_passes:
                why.append("hysteresis fail")
            reason = ", ".join(why)
        return {"lift_idx": lift_idx, "score": best_score, "conf": conf,
                "t_amp": t_amp, "t_s": t_s, "fz_s": fz_s, "reason": reason,
                "frac": best_frac}

    def detect_lift_event(ft: np.ndarray):
        if args.detect_method == "robust":
            return detect_lift_robust(ft)
        return detect_lift_simple(ft)

    _detect_holder["fn"] = detect_lift_event

    if args.debug:
        _run_debug()

    def plot_one(ax_f, ax_t, ep):
        ft = ep["obs"]
        T = ft.shape[0]
        x = np.arange(T) * args.dt if args.dt else np.arange(T)
        scale = args.dt if args.dt else 1

        # ----- force panel
        for k in range(3):
            ax_f.plot(x, ft[:, k], color=cmap[k], lw=1.0, label=f_labels[k])
        f_norm = np.linalg.norm(ft[:, 0:3], axis=1)
        ax_f.plot(x, f_norm, color="black", lw=1.0, ls="--", label="|F|")
        ax_f.set_ylabel("Force [N]")
        ax_f.set_xlabel(xlabel)
        ax_f.grid(alpha=0.3)

        # ----- torque panel
        for k in range(3):
            ax_t.plot(x, ft[:, 3 + k], color=cmap[k], lw=1.0, label=t_labels[k])
        t_norm = np.linalg.norm(ft[:, 3:6], axis=1)
        ax_t.plot(x, t_norm, color="black", lw=1.0, ls="--", label="|T|")
        ax_t.set_ylabel("Torque [Nm]")
        ax_t.set_xlabel(xlabel)
        ax_t.grid(alpha=0.3)

        f_title = f"Ep {ep['idx']}  T={T}  |F|max={f_norm.max():.2f} N"
        t_title = f"Ep {ep['idx']}  |T|max={t_norm.max():.3f} Nm"

        if args.mark_fz_event:
            res = detect_lift_event(ft)
            t_s = res.get("t_s")
            if t_s is not None:
                ax_t.plot(x, t_s, color="dimgray", lw=0.8, alpha=0.7,
                          label="|T| smoothed")
            lift_idx = res["lift_idx"]
            conf = res["conf"]
            if lift_idx is not None:
                if conf == "high":
                    line_col, band_col, band_a = "green", "lime", 0.12
                else:
                    line_col, band_col, band_a = "dimgray", "lightgray", 0.18
                policy_lo = max(0, lift_idx - args.policy_steps_before_lift)
                for ax_ in (ax_f, ax_t):
                    ax_.axvspan(policy_lo * scale, lift_idx * scale,
                                color=band_col, alpha=band_a,
                                label=f"policy ({args.policy_steps_before_lift} steps)")
                tag = "lift" if conf == "high" else "lift?"
                ax_f.axvline(lift_idx * scale, color=line_col, lw=1.4, ls="--",
                             label=f"{tag} @ {lift_idx}")
                ax_t.axvline(lift_idx * scale, color=line_col, lw=1.4, ls="--",
                             label=f"{tag} @ {lift_idx} score={res['score']:.2f}")
                if conf != "high":
                    suffix = f"  [low conf: {res.get('reason','')}]"
                    f_title += suffix
                    t_title += suffix
            else:
                suffix = f"  [lift not detected: {res.get('reason','')}]"
                f_title += suffix
                t_title += suffix

        ax_f.set_title(f_title)
        ax_t.set_title(t_title)
        ax_f.legend(loc="upper right", fontsize=9, ncol=4)
        ax_t.legend(loc="upper right", fontsize=9, ncol=4)

    written = []
    for ep in chosen:
        r = detect_lift_event(ep["obs"])
        lift = r["lift_idx"] if r["lift_idx"] is not None else -1
        print(f"ep {ep['idx']:>3}: lift={lift:>4}  score={r['score']:>6.3f}  "
              f"t_amp={r['t_amp']:>6.3f}  conf={r['conf']}  ({r.get('reason','')})")

        fig, (ax_f, ax_t) = plt.subplots(
            nrows=1, ncols=2, figsize=(16, 5)
        )
        plot_one(ax_f, ax_t, ep)
        fig.suptitle(
            f"FT curves — episode {ep['idx']}  ({ep.get('datetime')})",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out_png = out_dir / f"ep_{ep['idx']:03d}.png"
        fig.savefig(out_png, dpi=140)
        plt.close(fig)
        written.append(out_png)

        if not args.no_zoom:
            fig, (ax_f, ax_t) = plt.subplots(
                nrows=1, ncols=2, figsize=(16, 5)
            )
            plot_one(ax_f, ax_t, ep)
            scale = args.dt if args.dt else 1
            ax_f.set_xlim(args.zoom_lo * scale, args.zoom_hi * scale)
            ax_t.set_xlim(args.zoom_lo * scale, args.zoom_hi * scale)
            fig.suptitle(
                f"FT curves — episode {ep['idx']} (zoom {args.zoom_lo}-{args.zoom_hi})  "
                f"({ep.get('datetime')})",
                fontsize=12,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            out_png = out_dir / f"ep_{ep['idx']:03d}_zoom.png"
            fig.savefig(out_png, dpi=140)
            plt.close(fig)
            written.append(out_png)

    if args.combined:
        fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(14, 3.0 * n))
        if n == 1:
            axes = axes.reshape(1, 2)
        for row, ep in enumerate(chosen):
            plot_one(axes[row, 0], axes[row, 1], ep)
        if args.combined_zoom:
            scale = args.dt if args.dt else 1
            for row in range(n):
                axes[row, 0].set_xlim(args.zoom_lo * scale, args.zoom_hi * scale)
                axes[row, 1].set_xlim(args.zoom_lo * scale, args.zoom_hi * scale)
        title_extra = f" (zoom {args.zoom_lo}-{args.zoom_hi})" if args.combined_zoom else ""
        fig.suptitle(
            f"FT curves — {n} random eps from {args.jsonl.name} (seed={args.seed}){title_extra}",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        suffix = "_zoom" if args.combined_zoom else ""
        combined = out_dir / f"combined_{n}eps{suffix}.png"
        fig.savefig(combined, dpi=130)
        plt.close(fig)
        written.append(combined)

    print(f"Wrote {len(written)} files to {out_dir}/")
    for p in written:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
