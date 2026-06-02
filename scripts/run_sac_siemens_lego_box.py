"""Sequential Siemens → Lego → Box eval in a single process.

Drives three pretrained SAC policies back-to-back on the same robot:
  1. Move to a shared first-PE viewpoint (calibrated via calibrate_siemens_full.py 'C').
  2. Run the Siemens insertion policy for --max_episodes_siemens episodes.
  3. Prompt operator to place the Lego brick.
  4. Run the Lego 3DoF-rz policy for --max_episodes_lego episodes.
  5. Prompt operator to place the yellow box on its grasp stand + blue receptacle on shelf.
  6. Run the Box (b1_pe) policy for --max_episodes_box episodes.

Example:
    python scripts/run_sac_siemens_lego_box.py \\
        --load_policy_siemens siemens_1b_pretrain_v1.0_300/pretrain_4 \\
        --load_policy_lego    lego_3dof_rz_pretrain_v1/pretrain_4 \\
        --load_policy_box     box/pretrain_3 \\
        --max_episodes_lego   25 \\
        --max_episodes_box    25 \\
        --success_threshold_siemens 7.0 \\
        --success_threshold_lego    7.0 \\
        --success_threshold_box     7.0 \\
        --snap_reinforce \\
        --pose_viz_dir pe_viz/triple_$(date +%Y%m%d_%H%M%S)
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
    ShelfBoxConfig,
    SiemensConfigFull,
)
from crisp_drl.agents.shared.insertion_wrapper import install_stop_handler
from crisp_drl.envs import make_env, make_rew


# ---------------------------------------------------------------------------
# Per-task Config override
# ---------------------------------------------------------------------------

def _config_for_task(task: str) -> Config:
    config = Config()
    if task in ("lego_3dof_rz", "lego_3dof_rz_pe"):
        config.actor_output_dim = 3
        config.actor_nonvision_input_dim = 18
        config.max_action = np.array([0.00025, 0.00025, np.deg2rad(0.5)])
    # b1_pe uses default Config() dims (2D plane action, same as siemens)
    return config


@contextmanager
def _no_interrupts():
    old = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old)


# ---------------------------------------------------------------------------
# Load shared PE viewpoint
# ---------------------------------------------------------------------------

def _load_combined_first_pe() -> tuple[np.ndarray, np.ndarray]:
    try:
        from crisp_drl.agents.shared import siemens_config_demo as _scd  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "siemens_config_demo.py missing. Run calibrate_siemens_full.py, "
            "press 'C' with all objects visible, then 'P'."
        ) from e
    joints = getattr(_scd, "COMBINED_FIRST_PE_POSITION", None)
    cart = getattr(_scd, "COMBINED_FIRST_PE_CARTESIAN", None)
    if joints is None or cart is None:
        raise SystemExit(
            "siemens_config_demo.py lacks COMBINED_FIRST_PE_POSITION / "
            "COMBINED_FIRST_PE_CARTESIAN. Re-run calibrate_siemens_full.py and press 'C'."
        )
    joints = np.asarray(joints, dtype=np.float64).reshape(-1)
    cart = np.asarray(cart, dtype=np.float64).reshape(-1)
    if joints.size != 7 or cart.size != 6:
        raise SystemExit(
            f"Combined PE shapes wrong: joints={joints.shape}, cart={cart.shape}"
        )
    return joints, cart


# ---------------------------------------------------------------------------
# Args -> per-task namespaces
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
        use_pose_estimation=False,
        pe_align_gripper=False,
        use_6dof_grasp=False,
        pe_align_6dof=False,
        pose_viz_dir=(str(Path(args.pose_viz_dir) / "lego") if args.pose_viz_dir else None),
        snap_reinforce=args.snap_reinforce,
        pe_3dof=True,
        gt_target_goal=False,
        grasp_z_offset=args.grasp_z_offset,
        hand=args.hand,
        grasp_z_offset_pe=args.grasp_z_offset_pe,
        brick_size=args.brick_size,
        mp_backend=args.mp_backend,
        run_name=None,
        eval_json=None,
        eval_subdir="eval_combined/lego",
        stop_after_first_success=False,
    )


def _box_args(args, combined_joints: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        task="b1_pe",
        eval=True,
        cli_training=False,
        resume_training=None,
        pre_train=None,
        expert_buffer_path=None,
        load_policy=args.load_policy_box,
        load_encoder=None,
        max_episodes=args.max_episodes_box,
        success_threshold=args.success_threshold_box,
        no_ft_success_threshold=args.no_ft_success_threshold,
        no_ft_sensor=args.no_ft_sensor,
        use_pose_estimation=True,
        pe_align_gripper=False,
        use_6dof_grasp=False,
        pe_align_6dof=False,
        pose_viz_dir=(str(Path(args.pose_viz_dir) / "box") if args.pose_viz_dir else None),
        snap_reinforce=args.snap_reinforce,
        pe_3dof=False,
        gt_target_goal=False,
        grasp_z_offset=None,
        hand=False,
        grasp_z_offset_pe=0.0,
        mp_backend=args.mp_backend,
        run_name=None,
        eval_json=None,
        eval_subdir="eval_combined/box",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_eval_json(task_subdir: str, load_policy: str) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    policy_tag = (load_policy or "policy").replace("/", "__")
    return os.path.join("rollout_data", task_subdir, f"eval_results_{policy_tag}_{ts}.json")


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
    print(f"[combined] Homing to shared PE joints {np.round(combined_joints, 4)}")
    base_env.home(home_config=combined_joints.tolist(), blocking=True)


def _prompt_for_phase(phase_name: str, items: list[str]) -> bool:
    """Print a between-phase prompt. Return True to proceed, False to abort."""
    bullet_lines = "".join(f"  - {item}\n" for item in items)
    msg = (
        f"\n[combined] Previous phase complete.\n"
        f"{bullet_lines}"
        f"  Press Enter to start the {phase_name} phase, or 'q'+Enter to abort.\n> "
    )
    try:
        answer = input(msg).strip().lower()
    except EOFError:
        return False
    return answer != "q"


def _make_partial_close(label: str):
    """Return a close() replacement that skips env.close() + rclpy.shutdown()."""
    import types as _types

    def _partial_close(self):
        logging.info(
            f"[combined] {label} partial close (skipping env.close + rclpy.shutdown + reset)."
        )

    return _partial_close


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def _run_siemens_phase(base_env, args, combined_joints, combined_cart):
    print("\n" + "=" * 70)
    print("[combined] === SIEMENS PHASE ===")
    print("=" * 70)
    config = _config_for_task("siemens")
    task_args = _siemens_args(args, combined_joints)
    task_args.eval_json = _resolve_eval_json("eval_combined/siemens", task_args.load_policy)

    env_config = SiemensConfigFull()
    env_config.custom_first_home_position = combined_joints.copy()
    env_config.custom_home_position_pe = combined_joints.copy()

    env = make_env.create_real_env_s1_full(
        alg_config=config, env_config=env_config, args=task_args, base_env=base_env
    )

    rew_fn = _build_reward_fn(config)
    run_name = f"{config.env_name}__combined_siemens__{time.strftime('%Y%m%d-%H%M%S')}"
    actor = EvalSACActor(
        task_args, config, parameters_queue=None, run_name=run_name,
        env=env, rew_fn=rew_fn, eval_json_path=task_args.eval_json,
        prompt_user=False, save_parquet=False, parquet_dir=None,
        parquet_task_name="combined_siemens",
    )
    import types as _types
    actor.close = _types.MethodType(_make_partial_close("Siemens"), actor)
    actor.run_eval(data_queue=None)
    return env


def _run_lego_phase(base_env, args, combined_joints, combined_cart):
    print("\n" + "=" * 70)
    print("[combined] === LEGO PHASE ===")
    print("=" * 70)
    config = _config_for_task("lego_3dof_rz_pe")
    task_args = _lego_args(args, combined_joints)
    task_args.eval_json = _resolve_eval_json("eval_combined/lego", task_args.load_policy)
    # In the combined flow, stop Lego after the first successful insertion so
    # the next task can start immediately after snap_push finishes.
    task_args.stop_after_first_success = True

    cfg_cls = LEGO_BRICK_CONFIGS.get(args.brick_size, LegoConfig)
    env_config = cfg_cls()
    env_config.demo_goal_pose_estimation_euler = combined_cart.copy()

    env = make_env.create_real_env_v4_3dof_rz_pe(
        alg_config=config, env_config=env_config, args=task_args,
        brick_size=args.brick_size, base_env=base_env,
    )

    rew_fn = _build_reward_fn(config)
    run_name = f"{config.env_name}__combined_lego__{time.strftime('%Y%m%d-%H%M%S')}"
    actor = EvalSACActor(
        task_args, config, parameters_queue=None, run_name=run_name,
        env=env, rew_fn=rew_fn, eval_json_path=task_args.eval_json,
        prompt_user=False, save_parquet=False, parquet_dir=None,
        parquet_task_name="combined_lego",
    )
    import types as _types
    actor.close = _types.MethodType(_make_partial_close("Lego"), actor)
    actor.run_eval(data_queue=None)
    return env


def _run_box_phase(base_env, args, combined_joints, combined_cart):
    print("\n" + "=" * 70)
    print("[combined] === BOX PHASE ===")
    print("=" * 70)
    config = _config_for_task("b1_pe")
    task_args = _box_args(args, combined_joints)
    task_args.eval_json = _resolve_eval_json("eval_combined/box", task_args.load_policy)

    env_config = ShelfBoxConfig()
    env_config.custom_first_home_position = combined_joints.copy()
    env_config.custom_home_position_pe = combined_joints.copy()
    env_config.custom_home_position = combined_joints.copy()

    env = make_env.create_real_env_b1_pe(
        alg_config=config, env_config=env_config, args=task_args, base_env=base_env
    )

    rew_fn = _build_reward_fn(config)
    run_name = f"{config.env_name}__combined_box__{time.strftime('%Y%m%d-%H%M%S')}"
    actor = EvalSACActor(
        task_args, config, parameters_queue=None, run_name=run_name,
        env=env, rew_fn=rew_fn, eval_json_path=task_args.eval_json,
        prompt_user=False, save_parquet=False, parquet_dir=None,
        parquet_task_name="combined_box",
    )
    # Box is the last phase — let it close normally (env.close + rclpy.shutdown).
    try:
        actor.run_eval(data_queue=None)
    finally:
        try:
            actor.close()
        except Exception as e:
            logging.warning(f"[combined] Box actor close failed: {e}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser(
        description="Sequential Siemens → Lego → Box eval in one process.",
        allow_abbrev=False,
    )
    parser.add_argument("--load_policy_siemens", type=str, required=True)
    parser.add_argument("--load_policy_lego", type=str, required=True)
    parser.add_argument("--load_policy_box", type=str, required=True)

    parser.add_argument("--max_episodes_siemens", type=int, default=1)
    parser.add_argument("--max_episodes_lego", type=int, default=25)
    parser.add_argument("--max_episodes_box", type=int, default=25)

    parser.add_argument("--success_threshold_siemens", type=float, default=7.0)
    parser.add_argument("--success_threshold_lego", type=float, default=7.0)
    parser.add_argument("--success_threshold_box", type=float, default=7.0)

    parser.add_argument("--no_ft_success_threshold", type=float, default=4.0)
    parser.add_argument("--no_ft_sensor", action="store_true")
    parser.add_argument("--snap_reinforce", action="store_true")
    parser.add_argument(
        "--brick_size", type=str, default="2x4", choices=list(LEGO_BRICK_CONFIGS.keys())
    )
    parser.add_argument(
        "--pose_viz_dir", type=str, default=None,
        help="Root dir; subdirs 'siemens', 'lego', 'box' are created per task.",
    )
    parser.add_argument("--mp_backend", type=str, default="quintic")

    parser.add_argument("--hand", action="store_true")
    parser.add_argument("--grasp_z_offset_pe", type=float, default=0.028)
    parser.add_argument("--grasp_z_offset", type=float, default=None)

    parser.add_argument("--skip_siemens", action="store_true")
    parser.add_argument("--skip_lego", action="store_true")
    parser.add_argument("--skip_box", action="store_true")

    args = parser.parse_args()

    if args.pose_viz_dir:
        for sub in ("siemens", "lego", "box"):
            Path(args.pose_viz_dir, sub).mkdir(parents=True, exist_ok=True)

    combined_joints, combined_cart = _load_combined_first_pe()
    print(f"[combined] COMBINED_FIRST_PE_POSITION  = {np.round(combined_joints, 5)}")
    print(f"[combined] COMBINED_FIRST_PE_CARTESIAN = {np.round(combined_cart, 5)}")

    rclpy.init()
    try:
        base_env = make_env.make_env("my_env_v4_no_ft" if args.no_ft_sensor else "my_env_v4")
        print("[combined] Base env created.")
        base_env.wait_until_ready()
        install_stop_handler()

        _home_to_combined(base_env, combined_joints)

        # --- Siemens ---
        if not args.skip_siemens:
            _run_siemens_phase(base_env, args, combined_joints, combined_cart)
        else:
            print("[combined] --skip_siemens set.")

        # --- Lego ---
        if not args.skip_lego:
            if not args.skip_siemens:
                print(
                    "[combined] Siemens phase complete. "
                    "Switching to Lego automatically; place the Lego brick now."
                )
            _home_to_combined(base_env, combined_joints)
            _run_lego_phase(base_env, args, combined_joints, combined_cart)
        else:
            print("[combined] --skip_lego set.")

        # --- Box ---
        if not args.skip_box:
            if not (args.skip_siemens and args.skip_lego):
                print(
                    "[combined] Lego phase complete. "
                    "Switching to Box automatically; place the box setup now."
                )
            _home_to_combined(base_env, combined_joints)
            _run_box_phase(base_env, args, combined_joints, combined_cart)
        else:
            print("[combined] --skip_box set.")

    except KeyboardInterrupt:
        logging.info("Ctrl+C — terminating combined eval.")
    finally:
        with _no_interrupts():
            if rclpy.ok():  # pyright: ignore[reportPrivateImportUsage]
                rclpy.shutdown()
            logging.info("[combined] All done.")


if __name__ == "__main__":
    main()
