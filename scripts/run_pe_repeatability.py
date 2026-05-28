"""Pose-estimation repeatability harness for the Siemens lid task.

Drives only the PE phase of the eval pipeline — no grasp, no policy, no
contact. Each iteration:

  1. Home to ``SiemensConfig.custom_home_position_pe`` (gripper closed).
  2. Apply an XY jitter to the wide-camera viewpoint (±xy_jitter_m, uniform).
  3. Run coarse PE (FoundationPose) and validate the rotation matrix.
  4. Move to the coarse-refined hover (mirrors lines 1505-1521 of
     ``InsertionWrapperSiemensPE.reset``).
  5. Run refined PE.
  6. Save the same artifacts as the eval pipeline:
       <pose_viz_dir>/ep{NNN}_coarse_*.png, ep{NNN}_refined_*.png,
       ep{NNN}_sidebyside_*.png, ep{NNN}_raw.npz
     so ``scripts/analyze_pe_repeatability.py`` works unchanged.
  7. Return to the PE home.

Example
-------
python scripts/run_pe_repeatability.py \\
    --load_policy siemens_1b_pretrain_v1.0_300/pretrain_4 \\
    --use_pose_estimation --pe_align_6dof \\
    --n_iters 50 --xy_jitter_m 0.05 \\
    --pose_viz_dir pe_viz/siemens_pe_only_$(date +%Y%m%d_%H%M%S)
"""

from __future__ import annotations

import argparse
import logging
import signal
import time
from argparse import Namespace
from contextlib import contextmanager

import numpy as np
import rclpy

from crisp_drl.agents.shared.algorithm_config import Config
from crisp_drl.agents.shared.insertion_env_config import SiemensConfig
from crisp_drl.agents.shared.insertion_wrapper_s import InsertionWrapperSiemensPE
from crisp_drl.envs import make_env
from crisp_drl.envs.pose_visualizer import dump_raw_npz


def _override_config_for_task(args: argparse.Namespace, config: Config) -> Config:
    """Mirror of scripts/run_sac_eval._override_config_for_task — inlined to
    keep this harness free of the sys.path hack that would shadow `sam3`."""
    if getattr(args, "use_6dof_grasp", False):
        config.actor_output_dim = 5
        config.actor_nonvision_input_dim = 26
        config.max_action = np.array(
            [
                0.00025,
                0.00025,
                np.deg2rad(0.25),
                np.deg2rad(0.25),
                np.deg2rad(0.25),
            ]
        )
    return config


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def find_pe_wrapper(env) -> InsertionWrapperSiemensPE:
    """Walk the wrapper chain to the InsertionWrapperSiemensPE instance."""
    cur = env
    while cur is not None:
        if isinstance(cur, InsertionWrapperSiemensPE):
            return cur
        cur = getattr(cur, "env", None)
    raise RuntimeError(
        "InsertionWrapperSiemensPE not found in wrapper chain — "
        "create_real_env_s1_pe must have produced it."
    )


@contextmanager
def no_interrupts():
    old = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old)


def first_time_home(pe_wrap: InsertionWrapperSiemensPE) -> None:
    """Mirror the ``first_reset`` branch of ``InsertionWrapperSiemensPE.reset``
    so the cartesian controller is attached before any slow_move_to call.
    Gripper state is left untouched — this harness never grasps."""
    env_u = pe_wrap.env.unwrapped
    print("[pe-repeat] first-time home …")
    env_u.robot.wait_until_ready()  # type: ignore
    current = np.array(env_u.robot.end_effector_pose.position)  # type: ignore
    target = current + np.array([0.0, 0.0, 0.040])
    print(f"  safety lift 40 mm: {current} -> {target}")
    try:
        env_u.move_to(position=target, speed=0.03)  # type: ignore
    except Exception as e:
        print(f"  WARN: safety lift failed ({e!r}); skipping.")
    print("  joint-home to custom_first_home_position …")
    env_u.home(home_config=pe_wrap.env_config.custom_first_home_position)  # type: ignore
    time.sleep(1.5)


def home_pe(pe_wrap: InsertionWrapperSiemensPE) -> None:
    """Joint-home to the PE home pose. Gripper state untouched."""
    env_u = pe_wrap.env.unwrapped
    env_u.home(home_config=pe_wrap.env_config.custom_home_position_pe)  # type: ignore
    time.sleep(1.5)


def jittered_viewpoint(
    pe_wrap: InsertionWrapperSiemensPE,
    rng: np.random.Generator,
    xy_jitter_m: float,
) -> tuple[float, float]:
    """Refresh obs, sample a uniform (dx, dy) in [-jitter, jitter]^2, and
    drive there with slow_move_to. Returns the sampled (dx, dy) for logging."""
    # Refresh obs after the home() above.
    pe_wrap.obs, _ = pe_wrap.env.reset()
    home_xyz = np.asarray(
        pe_wrap.obs["observation.state.cartesian"][:3], dtype=np.float64
    ).copy()
    dx = float(rng.uniform(-xy_jitter_m, xy_jitter_m))
    dy = float(rng.uniform(-xy_jitter_m, xy_jitter_m))
    target = home_xyz + np.array([dx, dy, 0.0])
    print(f"[pe-repeat] jitter (dx, dy) = ({dx*1000:+.2f}, {dy*1000:+.2f}) mm")
    pe_wrap.slow_move_to(
        target,
        relative_pose_euler=None,
        max_step=0.0015,
        distance_err=0.0005,
    )
    return dx, dy


def run_pe_pass(
    pe_wrap: InsertionWrapperSiemensPE,
    ep_idx: int,
    jitter_dxy: tuple[float, float],
    refined_hover_z_m: float,
) -> None:
    """Run coarse PE + slow_move_to refined hover + refined PE, then dump
    NPZ + overlay PNGs in the same format as the eval pipeline."""
    pe_wrap._pe_episode_idx = ep_idx  # used for filename idx

    # --- Coarse PE --- #
    world_D_coarse, coarse_details = pe_wrap.estimate_world_pose_once(
        return_details=True
    )
    coarse_rgb = np.copy(pe_wrap.obs["observation.images.wrist_camera"])
    coarse_depth = np.copy(pe_wrap.obs["observation.images.wrist_depth_camera"])
    coarse_tcp = np.copy(pe_wrap.obs["observation.state.cartesian"])
    coarse_pose_cam = coarse_details["pose_cam"]
    coarse_mask = coarse_details["mask"]
    pose_check_coarse = pe_wrap.validate_world_pose_transform(world_D_coarse, "coarse")
    # Yaw-only alignment: keep wrist roll/pitch fixed (3DoF + yaw motion only).
    full_alignment_rpy = pe_wrap.compute_alignment_rpy(world_D_coarse)
    coarse_alignment_rpy = np.array(
        [0.0, 0.0, float(full_alignment_rpy[2])], dtype=np.float64
    )

    # --- Move to coarse-refined hover (eval lines 1505-1521) --- #
    coarse_grasp_world = (
        world_D_coarse[:3, 3]
        + world_D_coarse[:3, :3] @ pe_wrap.o_T_o_tcpgrasp
    )
    # Refined-PE hover: lift along world Z above the predicted grasp point.
    # Eval pipeline uses 2 cm; default here is higher so the refined
    # viewpoint sees the object from further back.
    coarse_hover_world = coarse_grasp_world + np.array(
        [0.0, 0.0, refined_hover_z_m]
    )
    pe_wrap.slow_move_to(
        coarse_hover_world,
        relative_pose_euler=coarse_alignment_rpy,
        max_step=0.0012,
        distance_err=0.0002,
        rotation_max_step_rad=np.deg2rad(0.5),
    )

    # --- Refined PE --- #
    world_D_refined, refined_details = pe_wrap.estimate_world_pose_once(
        return_details=True
    )
    refined_rgb = np.copy(pe_wrap.obs["observation.images.wrist_camera"])
    refined_depth = np.copy(pe_wrap.obs["observation.images.wrist_depth_camera"])
    refined_tcp = np.copy(pe_wrap.obs["observation.state.cartesian"])
    refined_pose_cam = refined_details["pose_cam"]
    refined_mask = refined_details["mask"]
    pose_check_refined = pe_wrap.validate_world_pose_transform(
        world_D_refined, "refined"
    )
    refined_alignment_rpy = pe_wrap.compute_alignment_rpy(world_D_refined)

    # --- Save artifacts (mirror eval lines 1582-1620) --- #
    if pe_wrap._pose_overlay_renderer is None:
        print("[pe-repeat] WARN: pose overlay renderer is None — "
              "did you pass --pose_viz_dir?")
        return
    paths = pe_wrap._pose_overlay_renderer.save_pair(
        out_dir=pe_wrap.pose_viz_dir,
        episode_idx=ep_idx,
        rgb=coarse_rgb,
        mask=coarse_mask,
        pose_cam_coarse=coarse_pose_cam,
        pose_cam_refined=refined_pose_cam,
        rgb_refined=refined_rgb,
        mask_refined=refined_mask,
    )
    raw_path = dump_raw_npz(
        out_dir=pe_wrap.pose_viz_dir,
        episode_idx=ep_idx,
        rgb_coarse=coarse_rgb,
        rgb_refined=refined_rgb,
        depth_coarse=coarse_depth,
        depth_refined=refined_depth,
        mask_coarse=coarse_mask,
        mask_refined=refined_mask,
        pose_cam_coarse=coarse_pose_cam,
        pose_cam_refined=refined_pose_cam,
        world_pose_coarse=world_D_coarse,
        world_pose_refined=world_D_refined,
        tcp_cartesian_coarse=coarse_tcp,
        tcp_cartesian_refined=refined_tcp,
        applied_alignment_rpy=np.asarray(coarse_alignment_rpy),
        refined_alignment_rpy=np.asarray(refined_alignment_rpy),
        pose_check_coarse_det_r=np.array(
            [pose_check_coarse["det_r"]], dtype=np.float32),
        pose_check_coarse_orth_err=np.array(
            [pose_check_coarse["orth_err"]], dtype=np.float32),
        pose_check_refined_det_r=np.array(
            [pose_check_refined["det_r"]], dtype=np.float32),
        pose_check_refined_orth_err=np.array(
            [pose_check_refined["orth_err"]], dtype=np.float32),
        jitter_dxy=np.array(jitter_dxy, dtype=np.float64),
    )
    print(f"[pe-repeat] ep{ep_idx:03d}: saved {paths} + {raw_path}")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def build_env_args(cli: argparse.Namespace) -> Namespace:
    """Shim the args namespace expected by create_real_env_s1_pe + the
    classifier wrapper that sits above it."""
    args = Namespace(**vars(cli))
    args.task = "siemens"
    args.eval = True
    args.cli_training = False
    args.resume_training = None
    args.pre_train = None
    args.expert_buffer_path = None
    args.save_parquet = False
    args.parquet_dir = None
    args.parquet_task_name = "pe_repeatability"
    args.no_user_prompt = True
    args.use_pose_estimation = True
    args.pe_align_gripper = False
    args.gt_target_goal = False
    args.snap_reinforce = False
    args.pe_3dof = False
    args.hand = False
    args.grasp_z_offset = None
    args.grasp_z_offset_pe = 0.028
    return args


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n_iters", type=int, default=50,
                   help="Number of PE repeatability passes.")
    p.add_argument("--xy_jitter_m", type=float, default=0.05,
                   help="Half-range of XY jitter at the PE home (m). "
                        "Default 0.05 → ±5 cm → 10×10 cm region.")
    p.add_argument("--refined_hover_z_m", type=float, default=0.02,
                   help="Hover height for the refined-PE viewpoint, along "
                        "world Z, above the predicted grasp point (m). "
                        "Eval pipeline uses 0.02; default 0.04 here keeps "
                        "the mesh smaller in the second frame.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pose_viz_dir", type=str, required=True,
                   help="Output directory for overlay PNGs + raw NPZ dumps. "
                        "Same format consumed by analyze_pe_repeatability.py.")
    p.add_argument("--load_policy", type=str, required=True,
                   help="Policy checkpoint name — needed by the classifier "
                        "wrapper that sits above the PE wrapper. Its output "
                        "is not used by this harness.")
    p.add_argument("--load_encoder", type=str, default=None)
    p.add_argument("--success_threshold", type=float, default=7.0)
    p.add_argument("--no_ft_success_threshold", type=float, default=4.0)
    p.add_argument("--no_ft_sensor", action="store_true")
    p.add_argument("--pe_align_6dof", action="store_true")
    p.add_argument("--use_6dof_grasp", action="store_true")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    cli = parse_args()
    rng = np.random.default_rng(cli.seed)

    rclpy.init()
    try:
        args = build_env_args(cli)
        config = Config()
        config = _override_config_for_task(args, config)

        env = make_env.create_real_env_s1_pe(
            alg_config=config, env_config=SiemensConfig(), args=args
        )
        pe_wrap = find_pe_wrapper(env)
        if pe_wrap._pose_overlay_renderer is None:
            raise SystemExit(
                "Pose overlay renderer was not initialised — "
                "this means --pose_viz_dir did not reach the wrapper.")

        first_time_home(pe_wrap)

        for ep in range(cli.n_iters):
            print(f"\n[pe-repeat] ===== iter {ep+1}/{cli.n_iters} =====")
            try:
                home_pe(pe_wrap)
                jitter = jittered_viewpoint(pe_wrap, rng, cli.xy_jitter_m)
                run_pe_pass(pe_wrap, ep, jitter, cli.refined_hover_z_m)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                # PE failures should not kill the whole study — log and move on.
                print(f"[pe-repeat] ep{ep:03d} FAILED: {e!r}; continuing.")

        # Return home one last time.
        home_pe(pe_wrap)
        print(f"\n[pe-repeat] done — artifacts in {pe_wrap.pose_viz_dir}")
        print(f"[pe-repeat] next: python scripts/analyze_pe_repeatability.py "
              f"{pe_wrap.pose_viz_dir}")
    except KeyboardInterrupt:
        logging.info("Ctrl+C — shutting down.")
    finally:
        with no_interrupts():
            if rclpy.ok():  # pyright: ignore[reportPrivateImportUsage]
                rclpy.shutdown()


if __name__ == "__main__":
    main()
