"""Drive the full SAC env pipeline (cameras, pose estimation, classifier, CLI)
with the motion planner replacing the SAC policy.

Same init as scripts/run_sac.py (siemens PE + no-ft), but instead of sampling
actions from the learned policy, we compute the target EE pose from the
pose-estimation goal stashed on the InsertionWrapperSiemensPE
(`self.goal_position`) and call env.plan_and_execute(target).

After MP reaches, we step the env with zero actions to let the
SuccessClassificationWrapper evaluate Q(s, pi(s)) and terminate the episode
on success (same classifier threshold path as run_sac.py).

Usage:

    pixi run -e jazzy python scripts/run_motion_planner.py \\
        --use_pose_estimation --no_ft_sensor \\
        --max_linear_vel 0.02 --max_episodes 3

Prereqs (same as run_sac.py --use_pose_estimation):
  - franka stack running,
  - wrist camera + depth publishing,
  - Isaac ROS FoundationPose / SAM3 pose_estimator node alive,
  - BOTA disconnected (the --no_ft_sensor path skips the FT subscription).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np

from crisp_py.utils.geometry import Pose


def _get_attr_across_wrappers(env, name: str):
    """Walk .env chain to find the first wrapper carrying attribute `name`."""
    cur = env
    while cur is not None:
        if hasattr(cur, name) and name in cur.__dict__:
            return getattr(cur, name)
        cur = getattr(cur, "env", None)
    raise AttributeError(f"No wrapper in the chain exposes attribute {name!r}.")


def _run_episode(env, args, ep_idx: int, logger: logging.Logger) -> dict:
    obs, info = env.reset()
    logger.info("Episode %d: reset complete.", ep_idx)

    try:
        goal_xyz = np.asarray(_get_attr_across_wrappers(env, "goal_position"), dtype=np.float64)
    except AttributeError:
        goal_xyz = None

    robot = env.unwrapped.robot
    current = robot.end_effector_pose
    logger.info("Episode %d: current EE pos=%s", ep_idx, current.position)

    if goal_xyz is None:
        err = np.asarray(obs.get("observation.error.cartesian", np.zeros(3)))[:3]
        target_xyz = current.position + err
        logger.warning(
            "No goal_position attr found; falling back to obs.error.cartesian: err=%s",
            err,
        )
    else:
        target_xyz = goal_xyz.copy()
        if args.lock_x:
            target_xyz[0] = current.position[0]
        logger.info("Episode %d: goal pos=%s (lock_x=%s)", ep_idx, target_xyz, args.lock_x)

    target_pose = Pose(target_xyz, current.orientation)

    res = env.plan_and_execute(
        target_pose,
        max_linear_vel=args.max_linear_vel,
        max_angular_vel=args.max_angular_vel,
        pos_tol=args.pos_tol,
        rot_tol=args.rot_tol,
        timeout=args.mp_timeout,
        workspace_box=((0.2, 0.85), (-0.4, 0.4), (-0.1, 0.7)),
    )
    logger.info(
        "Episode %d: MP reached=%s reason=%s elapsed=%.2fs pos_err=%.2fmm",
        ep_idx,
        res.reached,
        res.reason,
        res.elapsed,
        res.final_pose_error_m * 1e3,
    )

    action_dim = int(np.prod(env.action_space.shape))
    zero_action = np.zeros(action_dim, dtype=np.float32)

    terminated = truncated = False
    info_last = {}
    n_settle = 0
    for k in range(args.max_settle_steps):
        obs, reward, terminated, truncated, info_last = env.step(zero_action)
        n_settle += 1
        if terminated or truncated:
            break

    logger.info(
        "Episode %d: settle steps=%d terminated=%s truncated=%s info_keys=%s",
        ep_idx,
        n_settle,
        terminated,
        truncated,
        sorted(k for k in info_last.keys() if not k.startswith("observation")),
    )

    return {
        "mp_reached": res.reached,
        "mp_pos_err_m": res.final_pose_error_m,
        "mp_elapsed": res.elapsed,
        "settle_steps": n_settle,
        "terminated": terminated,
        "truncated": truncated,
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("run_mp")

    p = argparse.ArgumentParser()
    p.add_argument(
        "--task", type=str, choices=["siemens", "lego"], default="siemens"
    )
    p.add_argument("--use_pose_estimation", action="store_true", default=True)
    p.add_argument("--no_ft_sensor", action="store_true", default=True)
    p.add_argument("--eval", action="store_true", default=True)
    p.add_argument("--max_episodes", type=int, default=3)

    p.add_argument("--max_linear_vel", type=float, default=0.02)
    p.add_argument("--max_angular_vel", type=float, default=0.3)
    p.add_argument("--pos_tol", type=float, default=0.0035)
    p.add_argument("--rot_tol", type=float, default=0.05)
    p.add_argument("--mp_timeout", type=float, default=20.0)
    p.add_argument(
        "--lock_x",
        action="store_true",
        default=True,
        help="Keep current x when commanding MP target (policy also never moved x; x was FT-controlled or fixed at 0).",
    )
    p.add_argument("--max_settle_steps", type=int, default=30)

    p.add_argument(
        "--success_threshold",
        type=float,
        default=9.3,
        help="Classifier threshold on mean Q(s, pi(s)); forwarded to env.",
    )
    p.add_argument("--no_ft_success_threshold", type=float, default=4.0)

    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--pre_train", type=str, default=None)
    p.add_argument("--expert_buffer_path", type=str, default=None)
    p.add_argument(
        "--load_policy",
        type=str,
        required=True,
        help="Checkpoint subdir under checkpoints/ — same value you'd pass to "
        "run_sac.py --eval. SuccessClassificationWrapper loads "
        "shared_encoder/actor/qf*_state_dict.pth from this dir to compute "
        "mean Q(s,pi(s)) as the termination signal.",
    )
    p.add_argument("--resume_training", type=str, default=None)
    p.add_argument("--load_encoder", type=str, default=None)
    p.add_argument("--cli_training", action="store_true", default=False)

    args = p.parse_args()

    import rclpy

    from crisp_drl.agents.shared.algorithm_config import Config
    from crisp_drl.agents.shared.insertion_env_config import SiemensConfig
    from crisp_drl.envs import make_env

    if not rclpy.ok():
        rclpy.init()

    cfg = Config()
    if args.task == "lego":
        env = make_env.create_real_env_v4(cfg, args=args)
    else:
        if args.use_pose_estimation:
            env = make_env.create_real_env_s1_pe(
                alg_config=cfg, env_config=SiemensConfig(), args=args
            )
        else:
            env = make_env.create_real_env_s1(
                cfg, env_config=SiemensConfig(), args=args
            )

    if not hasattr(env, "plan_and_execute"):
        logger.error(
            "env has no plan_and_execute — MotionPlannerWrapper was not attached. "
            "Make sure --no_ft_sensor is set so make_env wraps the env."
        )
        return 1

    summaries = []
    try:
        for i in range(args.max_episodes):
            logger.info("=== episode %d/%d ===", i + 1, args.max_episodes)
            summaries.append(_run_episode(env, args, i + 1, logger))
            time.sleep(0.5)
    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    reached = sum(1 for s in summaries if s["mp_reached"])
    terminated = sum(1 for s in summaries if s["terminated"])
    logger.info(
        "Summary: %d/%d MP reached, %d/%d classifier-terminated",
        reached,
        len(summaries),
        terminated,
        len(summaries),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
