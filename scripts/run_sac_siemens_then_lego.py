"""Sequential Siemens-then-Lego eval in a single process.

Drives two pretrained SAC policies back-to-back on the same robot:
  1. Move to a shared first-PE viewpoint (calibrated separately; see
     `crisp_drl/scripts/calibrate_siemens_full.py` and the 'C' key).
  2. Run the Siemens insertion policy for `--max_episodes_siemens` episodes.
  3. Trigger Siemens last_reset (sweep + dropoff + gripper open).
  4. Prompt the operator to place the Lego brick on its grasp stand.
  5. Move back to the shared viewpoint and run the Lego 3DoF-rz policy for
     `--max_episodes_lego` episodes.

The two existing single-task entry points (`scripts/run_sac.py`,
`scripts/run_sac_eval.py`) are untouched. This script is purely additive.

Example:
    python scripts/run_sac_siemens_then_lego.py \\
        --load_policy_siemens siemens_1b_pretrain_v1.0_300/pretrain_4 \\
        --load_policy_lego    lego_3dof_rz_pretrain_v1/pretrain_4 \\
        --max_episodes_lego   25 \\
        --success_threshold_siemens 7.0 \\
        --success_threshold_lego    7.0 \\
        --snap_reinforce \\
        --pose_viz_dir pe_viz/combined_$(date +%Y%m%d_%H%M%S)

Default --max_episodes_siemens is 1: one insert + snap_push + home to overview,
then the operator places the Lego brick at the transition prompt and the Lego
phase starts. Override with --max_episodes_siemens N for multi-shot Siemens runs
(requires manual lid re-placement between episodes).
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from argparse import ArgumentParser
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rclpy

from crisp_drl.agents.shared.algorithm_config import Config
from crisp_drl.agents.shared.eval_actor import EvalSACActor
from crisp_drl.agents.shared.insertion_env_config import (
    LEGO_BRICK_CONFIGS,
    LegoConfig,
    SiemensConfigFull,
)
from crisp_drl.agents.shared.insertion_wrapper import install_stop_handler
from crisp_drl.envs import make_env, make_rew


# ---------------------------------------------------------------------------
# Per-task Config override (mirrors scripts/run_sac.py:_override_config_for_task)
# ---------------------------------------------------------------------------

def _config_for_task(task: str) -> Config:
    config = Config()
    if task in ("lego_3dof_rz", "lego_3dof_rz_pe"):
        config.actor_output_dim = 3
        config.actor_nonvision_input_dim = 18
        config.max_action = np.array(
            [0.00025, 0.00025, np.deg2rad(0.5)],
        )
    return config


@contextmanager
def _no_interrupts():
    old = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old)


# ---------------------------------------------------------------------------
# Load shared first-PE viewpoint produced by calibrate_siemens_full.py + 'C'
# ---------------------------------------------------------------------------

def _load_combined_first_pe() -> tuple[np.ndarray, np.ndarray]:
    try:
        from crisp_drl.agents.shared import siemens_config_demo as _scd  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "siemens_config_demo.py is missing. Run "
            "`python crisp_drl/scripts/calibrate_siemens_full.py`, press 'C' "
            "with both objects in the wrist-camera frame, then 'P'."
        ) from e

    joints = getattr(_scd, "COMBINED_FIRST_PE_POSITION", None)
    cart = getattr(_scd, "COMBINED_FIRST_PE_CARTESIAN", None)
    if joints is None or cart is None:
        raise SystemExit(
            "siemens_config_demo.py exists but lacks COMBINED_FIRST_PE_POSITION "
            "and/or COMBINED_FIRST_PE_CARTESIAN. Re-run calibrate_siemens_full.py "
            "and press 'C' before 'P'."
        )
    joints = np.asarray(joints, dtype=np.float64).reshape(-1)
    cart = np.asarray(cart, dtype=np.float64).reshape(-1)
    if joints.size != 7 or cart.size != 6:
        raise SystemExit(
            f"Combined first-PE shapes wrong: joints={joints.shape}, "
            f"cart={cart.shape}; expected (7,) and (6,)."
        )
    return joints, cart


# ---------------------------------------------------------------------------
# Args -> per-task namespace plumbing
# ---------------------------------------------------------------------------

def _siemens_args(args, combined_joints: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        task="siemens",
        eval=True,
        cli_training=False,
        resume_training=None,
        pre_train=None,
        expert_buffer_path=None,
        load_policy=args.load_policy_siemens,
        load_encoder=None,
        max_episodes=args.max_episodes_siemens,
        success_threshold=args.success_threshold_siemens,
        no_ft_success_threshold=args.no_ft_success_threshold,
        no_ft_sensor=args.no_ft_sensor,
        use_pose_estimation=True,
        pe_align_gripper=False,
        use_6dof_grasp=False,
        pe_align_6dof=False,
        pose_viz_dir=(str(Path(args.pose_viz_dir) / "siemens") if args.pose_viz_dir else None),
        snap_reinforce=args.snap_reinforce,
        pe_3dof=False,
        gt_target_goal=False,
        grasp_z_offset=None,
        hand=False,
        grasp_z_offset_pe=0.0,
        mp_backend=args.mp_backend,
        run_name=None,
        eval_json=None,
        eval_subdir="eval_combined/siemens",
    )


def _lego_args(args, combined_joints: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        task="lego_3dof_rz_pe",
        eval=True,
        cli_training=False,
        resume_training=None,
        pre_train=None,
        expert_buffer_path=None,
        load_policy=args.load_policy_lego,
        load_encoder=None,
        max_episodes=args.max_episodes_lego,
        success_threshold=args.success_threshold_lego,
        no_ft_success_threshold=args.no_ft_success_threshold,
        no_ft_sensor=args.no_ft_sensor,
        use_pose_estimation=False,  # Lego PE wrapper handles its own PE; this flag is for s1_pe selection only
        pe_align_gripper=False,
        use_6dof_grasp=False,
        pe_align_6dof=False,
        pose_viz_dir=(str(Path(args.pose_viz_dir) / "lego") if args.pose_viz_dir else None),
        snap_reinforce=args.snap_reinforce,
        pe_3dof=True,  # --3dof in the original CLI
        gt_target_goal=False,
        grasp_z_offset=args.grasp_z_offset,
        hand=args.hand,
        grasp_z_offset_pe=args.grasp_z_offset_pe,
        brick_size=args.brick_size,
        mp_backend=args.mp_backend,
        run_name=None,
        eval_json=None,
        eval_subdir="eval_combined/lego",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_eval_json(task_subdir: str, load_policy: str) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    policy_tag = (load_policy or "policy").replace("/", "__")
    return os.path.join(
        "rollout_data", task_subdir, f"eval_results_{policy_tag}_{ts}.json"
    )


def _build_reward_fn(config: Config):
    return make_rew.create_real_reward_fn(
        config,
        event_reward_map={
            "E_SUCCESS": 0.1 / (1 - config.gamma) * 3,
            "E_FAIL": -0.1 / (1 - config.gamma) / 2 * 3,
            "E_SAFETY_BOX_VIOLATION": 0.0,
            "E_ROLLOUT_UNUSABLE": -100,
            "E_CONTROLLER_ISSUE": -100,
            "E_TORQUE": -100,
        },
    )


def _home_to_combined(base_env, combined_joints: np.ndarray) -> None:
    print(f"[combined] Homing to shared first-PE joints {np.round(combined_joints, 4)}")
    base_env.home(home_config=combined_joints.tolist(), blocking=True)


def _prompt_for_lego_setup() -> bool:
    """Block on stdin. Return True to proceed, False to abort."""
    msg = (
        "\n[combined] Siemens phase complete.\n"
        "  - Place the Lego brick on its grasp stand.\n"
        "  - Confirm target (yellow) brick is in the wrist-camera frame.\n"
        "  - Press Enter to start the Lego phase, or 'q'+Enter to abort.\n"
        "> "
    )
    try:
        answer = input(msg).strip().lower()
    except EOFError:
        return False
    return answer != "q"


def _run_siemens_phase(base_env, args, combined_joints, combined_cart):
    print("\n" + "=" * 70)
    print("[combined] === SIEMENS PHASE ===")
    print("=" * 70)
    config = _config_for_task("siemens")
    task_args = _siemens_args(args, combined_joints)
    task_args.eval_json = _resolve_eval_json("eval_combined/siemens", task_args.load_policy)

    env_config = SiemensConfigFull()
    # Pin BOTH home joint configs to the shared overview PE pose so:
    #   - the first-time joint home (used once at process start) lands at the
    #     overview view (env_config.custom_first_home_position),
    #   - every subsequent per-episode pre-PE home lands there too
    #     (env_config.custom_home_position_pe; also consumed by snap_push as
    #     self.home_config so the post-snap home returns to the overview).
    env_config.custom_first_home_position = combined_joints.copy()
    env_config.custom_home_position_pe = combined_joints.copy()

    env = make_env.create_real_env_s1_full(
        alg_config=config,
        env_config=env_config,
        args=task_args,
        base_env=base_env,
    )

    rew_fn = _build_reward_fn(config)
    run_name = f"{config.env_name}__combined_siemens__{time.strftime('%Y%m%d-%H%M%S')}"
    actor = EvalSACActor(
        task_args,
        config,
        parameters_queue=None,  # eval mode skips parameter sync
        run_name=run_name,
        env=env,
        rew_fn=rew_fn,
        eval_json_path=task_args.eval_json,
        prompt_user=False,
        save_parquet=False,
        parquet_dir=None,
        parquet_task_name="combined_siemens",
    )

    # EvalSACActor.run_eval()'s finally always calls self.close(), and the
    # inherited SACActor.close() calls env.close() + rclpy.shutdown() — which
    # would kill the ROS context before the Lego phase can run. Patch the
    # instance's close() to do only the harmless part of the cleanup
    # (env.reset(last_reset=True)) so the base env + ROS stay alive.
    import types as _types

    def _partial_close(self):
        logging.info("[combined] Siemens partial close (skipping env.close + rclpy.shutdown).")
        try:
            self.env.reset(options={"last_reset": True})
        except Exception as _e:
            logging.warning(f"[combined] Siemens partial-close last_reset failed: {_e}")

    actor.close = _types.MethodType(_partial_close, actor)

    actor.run_eval(data_queue=None)
    return env  # caller keeps a reference for explicit teardown ordering


def _run_lego_phase(base_env, args, combined_joints, combined_cart):
    print("\n" + "=" * 70)
    print("[combined] === LEGO PHASE ===")
    print("=" * 70)
    config = _config_for_task("lego_3dof_rz_pe")
    task_args = _lego_args(args, combined_joints)
    task_args.eval_json = _resolve_eval_json("eval_combined/lego", task_args.load_policy)

    cfg_cls = LEGO_BRICK_CONFIGS.get(args.brick_size, LegoConfig)
    env_config = cfg_cls()
    # Override the wide-PE pose with the shared viewpoint. The Lego PE wrapper
    # consumes demo_goal_pose_estimation_euler as wide_pe_pose_euler.
    env_config.demo_goal_pose_estimation_euler = combined_cart.copy()

    env = make_env.create_real_env_v4_3dof_rz_pe(
        alg_config=config,
        env_config=env_config,
        args=task_args,
        brick_size=args.brick_size,
        base_env=base_env,
    )

    rew_fn = _build_reward_fn(config)
    run_name = f"{config.env_name}__combined_lego__{time.strftime('%Y%m%d-%H%M%S')}"
    actor = EvalSACActor(
        task_args,
        config,
        parameters_queue=None,
        run_name=run_name,
        env=env,
        rew_fn=rew_fn,
        eval_json_path=task_args.eval_json,
        prompt_user=False,
        save_parquet=False,
        parquet_dir=None,
        parquet_task_name="combined_lego",
    )
    try:
        actor.run_eval(data_queue=None)
    finally:
        # Lego phase is the final task — close the env + shut down rclpy now.
        try:
            actor.close()
        except Exception as e:
            logging.warning(f"[combined] Lego actor close failed: {e}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser(
        description="Sequential Siemens-then-Lego eval in one process.",
        allow_abbrev=False,
    )
    parser.add_argument("--load_policy_siemens", type=str, required=True)
    parser.add_argument("--load_policy_lego", type=str, required=True)
    parser.add_argument(
        "--max_episodes_siemens",
        type=int,
        default=1,
        help="Number of Siemens insertion episodes before transitioning to Lego. "
        "Default 1: one insertion, snap_push, home to overview, prompt operator "
        "for Lego setup. Multi-episode runs require manual lid re-placement "
        "between episodes since the inserted lid is left in the goal slot.",
    )
    parser.add_argument("--max_episodes_lego", type=int, default=25)
    parser.add_argument("--success_threshold_siemens", type=float, default=7.0)
    parser.add_argument("--success_threshold_lego", type=float, default=7.0)
    parser.add_argument("--no_ft_success_threshold", type=float, default=4.0)
    parser.add_argument("--no_ft_sensor", action="store_true")
    parser.add_argument("--snap_reinforce", action="store_true")
    parser.add_argument("--brick_size", type=str, default="2x4", choices=list(LEGO_BRICK_CONFIGS.keys()))
    parser.add_argument(
        "--pose_viz_dir",
        type=str,
        default=None,
        help="Root dir; subdirs 'siemens' and 'lego' are created per task.",
    )
    parser.add_argument("--mp_backend", type=str, default="quintic")

    # ---- Lego 3DOF PE Z-control flags (forwarded to InsertionWrapper3DoFRotZPE) ----
    parser.add_argument(
        "--hand",
        action="store_true",
        help="Use PE-estimated Z for the Lego grasp (instead of "
        "LegoConfig.grasp_position_ground_truth[2] + --grasp_z_offset). "
        "Mirrors the standalone Lego eval flag.",
    )
    parser.add_argument(
        "--grasp_z_offset_pe",
        type=float,
        default=0.028,
        help="Additive Z offset (m) on top of the PE-estimated brick top when "
        "--hand is set. Default 0.028 m matches the standalone Lego eval default.",
    )
    parser.add_argument(
        "--grasp_z_offset",
        type=float,
        default=None,
        help="Additive Z offset (m) on top of grasp_position_ground_truth[2] "
        "when --hand is NOT set. None -> wrapper default (2.8 mm in eval).",
    )

    parser.add_argument(
        "--skip_lego",
        action="store_true",
        help="Run only the Siemens phase. Useful for debugging the orchestrator.",
    )
    parser.add_argument(
        "--skip_siemens",
        action="store_true",
        help="Skip the Siemens phase and go directly to Lego. Useful for "
        "debugging the Lego stack alone via this orchestrator.",
    )
    args = parser.parse_args()

    if args.pose_viz_dir:
        Path(args.pose_viz_dir, "siemens").mkdir(parents=True, exist_ok=True)
        Path(args.pose_viz_dir, "lego").mkdir(parents=True, exist_ok=True)

    combined_joints, combined_cart = _load_combined_first_pe()
    print(f"[combined] COMBINED_FIRST_PE_POSITION  = {np.round(combined_joints, 5)}")
    print(f"[combined] COMBINED_FIRST_PE_CARTESIAN = {np.round(combined_cart, 5)}")

    rclpy.init()
    try:
        # Build the base env ONCE. Subsequent task wrappers reuse it.
        base_env = make_env.make_env("my_env_v4_no_ft" if args.no_ft_sensor else "my_env_v4")
        print("[combined] Base env created.")
        base_env.wait_until_ready()
        install_stop_handler()

        # Pre-task: home to the shared viewpoint with both objects in frame.
        _home_to_combined(base_env, combined_joints)

        siemens_env = None
        if not args.skip_siemens:
            siemens_env = _run_siemens_phase(base_env, args, combined_joints, combined_cart)
        else:
            print("[combined] --skip_siemens set; jumping to Lego.")

        if args.skip_lego:
            print("[combined] --skip_lego set; done.")
            # If Siemens ran, env reset already returned home; just close it.
            if siemens_env is not None:
                try:
                    siemens_env.close()
                except Exception:
                    pass
            return

        if not args.skip_siemens:
            if not _prompt_for_lego_setup():
                print("[combined] Operator aborted at transition. Closing.")
                try:
                    if siemens_env is not None:
                        siemens_env.close()
                except Exception:
                    pass
                return
            # Drop the Siemens wrapper chain (base_env survives).
            siemens_env = None  # noqa: F841

        # Re-home to the shared viewpoint before Lego phase.
        _home_to_combined(base_env, combined_joints)

        _run_lego_phase(base_env, args, combined_joints, combined_cart)

    except KeyboardInterrupt:
        logging.info("Ctrl+C — terminating combined eval.")
    finally:
        with _no_interrupts():
            if rclpy.ok():  # pyright: ignore[reportPrivateImportUsage]
                rclpy.shutdown()
            logging.info("[combined] All done.")


if __name__ == "__main__":
    main()
