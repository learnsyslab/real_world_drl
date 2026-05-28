"""Evaluation runner for trained SAC policies.

Mirrors `scripts/run_sac.py` but is eval-only: never spawns the learner,
runs a bounded number of episodes (default 50), records per-episode forces
(N), torques (Nm), and cycle time (s), and prompts the operator after every
episode for a ground-truth success label. Results are streamed to a JSON
file (atomic write each episode) for crash safety.

Example
-------
python scripts/run_sac_eval.py \\
    --load_policy lego_3dof_rz_pretrain_v5.1_501/pretrain_4 \\
    --task lego_3dof_rz \\
    --success_threshold 7.8 \\
    --snap_reinforce \\
    --max_episodes 50
"""

import logging
import os
import signal
import time
from argparse import ArgumentParser
from contextlib import contextmanager

import numpy as np
import rclpy

from crisp_drl.agents.shared.algorithm_config import Config
from crisp_drl.agents.shared.eval_actor import EvalSACActor
from crisp_drl.agents.shared.insertion_env_config import SiemensConfig
from crisp_drl.agents.shared.insertion_wrapper import install_stop_handler
from crisp_drl.envs import make_env, make_rew


def _override_config_for_task(args, config):
    """Same per-task overrides as scripts/run_sac.py:_override_config_for_task."""
    if args is None:
        return config
    task = getattr(args, "task", "siemens")
    if task in ("lego_3dof_rz", "lego_3dof_rz_pe"):
        config.actor_output_dim = 3
        config.actor_nonvision_input_dim = 18
        config.max_action = np.array(
            [
                0.00025,
                0.00025,
                np.deg2rad(0.5),
            ]
        )
    if getattr(args, "use_6dof_grasp", False):
        if task != "siemens":
            raise ValueError(
                "--use_6dof_grasp is currently only supported for --task siemens."
            )
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


@contextmanager
def no_interrupts():
    old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old_handler)


def launch_actor(args, config, data_queue, parameters_queue, run_name):
    logging.basicConfig(level=logging.INFO)
    try:
        task = getattr(args, "task", "siemens")
        if task == "lego_3dof_rz":
            env = make_env.create_real_env_v4_3dof_rz(config, args=args)
        elif task == "lego_3dof_rz_pe":
            env = make_env.create_real_env_v4_3dof_rz_pe(config, args=args)
        elif task == "lego":
            env = make_env.create_real_env_v4(config, args=args)
        else:
            env = (
                make_env.create_real_env_s1(
                    config, env_config=SiemensConfig(), args=args
                )
                if not args.use_pose_estimation
                else make_env.create_real_env_s1_pe(
                    alg_config=config, env_config=SiemensConfig(), args=args
                )
            )
        install_stop_handler()
        rew_fn = make_rew.create_real_reward_fn(
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
        actor = EvalSACActor(
            args,
            config,
            parameters_queue,
            run_name,
            env,
            rew_fn,
            eval_json_path=args.eval_json,
            prompt_user=not args.no_user_prompt,
            save_parquet=args.save_parquet,
            parquet_dir=args.parquet_dir,
            parquet_task_name=args.parquet_task_name,
        )
    except Exception as e:
        logging.error(f"Failed to initialize EvalSACActor: {e}", exc_info=True)
        return

    try:
        actor.run_eval(data_queue)
    finally:
        # actor.close() is already invoked in run_eval's finally; calling again
        # mirrors launch_actor's pattern in run_sac.py and is harmless once the
        # env has been closed.
        try:
            actor.close()
        except Exception:
            pass


def launch_processes(args, config):
    """Run the eval actor inline in this process.

    Unlike run_sac.py, eval has no learner to spawn — and the spawned-child
    actor in run_sac.py inherits a closed stdin, which would break the
    `input()`-based success prompt. So we just initialise ROS and call the
    actor directly here.
    """
    rclpy.init()
    config = Config()
    config = _override_config_for_task(args, config)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    algo_name = str(os.path.dirname(__file__).split("/")[-1])
    run_name = args.run_name or f"{config.env_name}__{algo_name}_eval__{timestamp}"

    try:
        launch_actor(args, config, None, None, run_name)
    except KeyboardInterrupt:
        logging.info("Ctrl+C — terminating eval actor.")
    finally:
        with no_interrupts():
            if rclpy.ok():  # pyright: ignore[reportPrivateImportUsage]
                rclpy.shutdown()
            logging.info("Eval run finished.")


def _resolve_eval_json(args) -> str:
    if args.eval_json:
        return args.eval_json
    subdir = getattr(args, "eval_subdir", "eval") or "eval"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if args.load_policy:
        policy_tag = args.load_policy.replace("/", "__")
        return os.path.join(
            "rollout_data", subdir, f"eval_results_{policy_tag}_{timestamp}.json"
        )
    return os.path.join("rollout_data", subdir, f"eval_results_{timestamp}.json")


def main():
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser(
        description="Evaluate a trained SAC policy on the real robot.",
        allow_abbrev=False,
    )

    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--eval",
        action="store_true",
        help="No-op (this script is always eval). Accepted so run_sac.py command "
        "lines can be reused.",
    )
    parser.add_argument(
        "--load_policy",
        type=str,
        required=True,
        help="Checkpoint name of policy to evaluate (e.g. "
        "'lego_3dof_rz_pretrain_v5.1_501/pretrain_4').",
    )
    parser.add_argument(
        "--load_encoder",
        type=str,
        default=None,
        help="Optional: load encoder separately. Usually unused for eval.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=50,
        help="Number of valid episodes to evaluate (E_ROLLOUT_UNUSABLE rollouts "
        "are skipped and not counted).",
    )
    parser.add_argument(
        "--eval_json",
        type=str,
        default=None,
        help="Output JSON path. Defaults to "
        "rollout_data/<eval_subdir>/eval_results_<load_policy>_<YYYYmmdd-HHMMSS>.json.",
    )
    parser.add_argument(
        "--eval_subdir",
        type=str,
        default="eval",
        help="Subdirectory under rollout_data/ for both the JSON results and "
        "the parquet dataset. Default 'eval'; pass 'eval_ood' (or any other "
        "name) to isolate runs without colliding with the main eval folder.",
    )
    parser.add_argument(
        "--no_user_prompt",
        action="store_true",
        help="Skip the s/u prompt at episode end. user_success is recorded as "
        "null; success_rate_user is computed only over labelled episodes.",
    )
    parser.add_argument(
        "--save_parquet",
        action="store_true",
        help="Save each eval rollout as a LeRobotDataset episode (parquet).",
    )
    parser.add_argument(
        "--parquet_dir",
        type=str,
        default=None,
        help="Override the parquet output dir. Default: "
        "rollout_data/eval/<load_policy>_<YYYYmmdd-HHMMSS>.",
    )
    parser.add_argument(
        "--parquet_task_name",
        type=str,
        default="eval_data",
        help="LeRobot dataset 'task' field stored per frame.",
    )

    # --- mirrored from scripts/run_sac.py (task + safety + PE flags) ---
    parser.add_argument(
        "--task",
        type=str,
        choices=["siemens", "lego", "lego_3dof_rz", "lego_3dof_rz_pe"],
        default="siemens",
    )
    parser.add_argument(
        "--success_threshold",
        type=float,
        default=9.3,
        help="Threshold on mean Q(s, pi(s)) for appending E_SUCCESS and terminating.",
    )
    parser.add_argument(
        "--no_ft_sensor",
        action="store_true",
        help="Run without the BOTA force-torque sensor (zero FT features).",
    )
    parser.add_argument(
        "--no_ft_success_threshold",
        type=float,
        default=4.0,
    )
    parser.add_argument("--use_pose_estimation", action="store_true")
    parser.add_argument("--pe_align_gripper", action="store_true")
    parser.add_argument("--use_6dof_grasp", action="store_true")
    parser.add_argument(
        "--pe_align_6dof",
        action="store_true",
        help="Use 6DoF PE alignment for the grasp (wrist tilts to grab a "
        "tilted brick) while keeping the policy action space at 2D. After "
        "grasp the standard undo step rotates the wrist back to demo-neutral, "
        "so a 2D-action policy sees the same observation distribution. "
        "Useful with a 2D-trained policy when bricks aren't flat. Implied by "
        "--use_6dof_grasp.",
    )
    parser.add_argument("--pose_viz_dir", type=str, default=None)
    parser.add_argument("--snap_reinforce", action="store_true")
    parser.add_argument(
        "--3dof",
        action="store_true",
        dest="pe_3dof",
        help="PE 3DOF mode: flat-lego assumption.",
    )
    parser.add_argument("--gt_target_goal", action="store_true")
    parser.add_argument(
        "--grasp_z_offset",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--hand",
        action="store_true",
        help="Use PE-estimated Z (from wrist-cam FoundationPose) for the grasp "
        "instead of the fixed grasp_position_ground_truth[2]. Flat-laying "
        "brick assumption is preserved via pe_3dof projection. Only effective "
        "when --3dof is also set.",
    )
    parser.add_argument(
        "--grasp_z_offset_pe",
        type=float,
        default=0.028,
        help="Additive Z offset (m) applied on top of the PE-estimated brick Z "
        "when --hand is set. Default 0.0 (use raw PE Z as TCP grasp Z).",
    )

    args = parser.parse_args()

    # Eval-only: shim flags that SACActor.__init__ / wrappers read.
    args.eval = True  # always on in this script (overrides the no-op flag above)
    args.cli_training = False
    args.resume_training = None
    args.pre_train = None
    args.expert_buffer_path = None

    args.eval_json = _resolve_eval_json(args)
    logging.info(f"[run_sac_eval] eval_json → {args.eval_json}")

    config = Config()
    config = _override_config_for_task(args, config)
    launch_processes(args, config)


if __name__ == "__main__":
    main()
