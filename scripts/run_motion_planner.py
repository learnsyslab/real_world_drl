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


WORKSPACE_BOX = ((0.2, 0.85), (-0.4, 0.4), (-0.1, 0.7))


def _inside_box(pos: np.ndarray, box) -> bool:
    return all(lo <= v <= hi for v, (lo, hi) in zip(pos, box))


def _build_spline_waypoints(
    *,
    current_pose: Pose,
    target_xyz: np.ndarray,
    mode: str,
    lift_m: float,
    box,
) -> list[Pose]:
    """Build spline waypoints for one episode with conservative geometry."""
    ori = current_pose.orientation
    cur = np.asarray(current_pose.position, dtype=np.float64)
    tgt = np.asarray(target_xyz, dtype=np.float64)

    if mode == "single":
        waypoints = [Pose(tgt, ori)]
    elif mode == "lift_xy_then_descend":
        z_lift = max(cur[2], tgt[2]) + max(lift_m, 0.0)
        wp1 = np.array([cur[0], cur[1], z_lift], dtype=np.float64)
        wp2 = np.array([tgt[0], tgt[1], z_lift], dtype=np.float64)
        wp3 = tgt.copy()
        waypoints = [Pose(wp1, ori), Pose(wp2, ori), Pose(wp3, ori)]
    else:
        raise ValueError(f"Unknown spline waypoint mode: {mode!r}")

    # Remove near-duplicate consecutive waypoints so spline knots stay clean.
    filtered: list[Pose] = []
    for wp in waypoints:
        if not filtered:
            filtered.append(wp)
            continue
        if np.linalg.norm(wp.position - filtered[-1].position) > 1e-9:
            filtered.append(wp)

    for i, wp in enumerate(filtered):
        if not _inside_box(wp.position, box):
            raise ValueError(
                f"Spline waypoint {i} outside workspace box: {wp.position}"
            )

    return filtered


def _load_policy(load_policy_subdir: str, cfg, logger: logging.Logger):
    """Load shared_encoder + actor from checkpoints/{subdir}/. Inline duplicate
    of the SACActor init path — we skip learner/writer/queue wiring entirely."""
    import torch
    from crisp_drl.agents.shared.networks_cleanrl import Actor, SharedEncoder

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")
    actor = Actor(cfg).to(device)
    actor.load_state_dict(
        torch.load(f"checkpoints/{load_policy_subdir}/actor_state_dict.pth")
    )
    actor.eval()

    shared_encoder = SharedEncoder(cfg).to(device)
    shared_encoder.load_state_dict(
        torch.load(
            f"checkpoints/{load_policy_subdir}/shared_encoder_state_dict.pth"
        )
    )
    shared_encoder.eval()

    logger.info("Loaded policy from checkpoints/%s on %s", load_policy_subdir, device)
    return actor, shared_encoder, device


def _run_policy_rollout_from_obs(
    env, obs_dict, actor, shared_encoder, cfg, max_steps
):
    """Per-tick policy loop starting from a given obs dict. Mirrors
    SACActor.run_pe's inner loop but without reward_fn / tensorboard / queue."""
    import torch
    from crisp_drl.data import utils as drl_utils

    terminated = truncated = False
    info_last: dict = {}
    n_steps = 0
    obs_feat = obs_dict["observation.formatted"]
    with torch.no_grad():
        for _ in range(max_steps):
            obs_input = drl_utils.shared_encode(
                shared_encoder,
                obs_feat.view(1, -1),
                cfg.shared_encoder_gradient,
            )
            action, _ = actor(obs_input)
            action = action.view(-1).detach().cpu().numpy()
            obs_dict, _, terminated, truncated, info_last = env.step(action)
            obs_feat = obs_dict["observation.formatted"]
            n_steps += 1
            if terminated or truncated:
                break
    return n_steps, terminated, truncated, info_last


def _get_attr_across_wrappers(env, name: str):
    """Walk .env chain to find the first wrapper carrying attribute `name`."""
    cur = env
    while cur is not None:
        if hasattr(cur, name) and name in cur.__dict__:
            return getattr(cur, name)
        cur = getattr(cur, "env", None)
    raise AttributeError(f"No wrapper in the chain exposes attribute {name!r}.")


def _run_episode(
    env,
    args,
    ep_idx: int,
    logger: logging.Logger,
    *,
    policy_bundle=None,
) -> dict:
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

    mp_kwargs = dict(
        max_linear_vel=args.max_linear_vel,
        max_angular_vel=args.max_angular_vel,
        pos_tol=args.pos_tol,
        rot_tol=args.rot_tol,
        timeout=args.mp_timeout,
        workspace_box=WORKSPACE_BOX,
    )
    if args.mp_backend == "ruckig":
        mp_kwargs["max_linear_acc"] = args.max_linear_acc
        mp_kwargs["max_linear_jerk"] = args.max_linear_jerk
    if args.mp_backend == "spline":
        if not hasattr(env, "plan_and_execute_spline"):
            raise AttributeError(
                "mp_backend=spline requested but env has no plan_and_execute_spline"
            )
        spline_waypoints = _build_spline_waypoints(
            current_pose=current,
            target_xyz=target_xyz,
            mode=args.spline_waypoint_mode,
            lift_m=args.spline_lift_m,
            box=WORKSPACE_BOX,
        )
        logger.info(
            "Episode %d: spline mode=%s waypoints=%d positions=%s",
            ep_idx,
            args.spline_waypoint_mode,
            len(spline_waypoints),
            [wp.position.tolist() for wp in spline_waypoints],
        )
        res = env.plan_and_execute_spline(
            spline_waypoints,
            max_linear_acc=args.max_linear_acc,
            max_linear_jerk=args.max_linear_jerk,
            **mp_kwargs,
        )
    else:
        res = env.plan_and_execute(target_pose, **mp_kwargs)
    logger.info(
        "Episode %d: MP reached=%s reason=%s elapsed=%.2fs pos_err=%.2fmm",
        ep_idx,
        res.reached,
        res.reason,
        res.elapsed,
        res.final_pose_error_m * 1e3,
    )

    terminated = truncated = False
    info_last = {}
    n_settle = 0
    if args.no_ft_sensor or policy_bundle is None:
        action_dim = int(np.prod(env.action_space.shape))
        zero_action = np.zeros(action_dim, dtype=np.float32)
        for _ in range(args.max_settle_steps):
            obs, _reward, terminated, truncated, info_last = env.step(zero_action)
            n_settle += 1
            if terminated or truncated:
                break
    else:
        actor, shared_encoder, cfg = policy_bundle
        n_settle, terminated, truncated, info_last = _run_policy_rollout_from_obs(
            env, obs, actor, shared_encoder, cfg, args.max_policy_steps
        )

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
    p.add_argument(
        "--use_6dof_grasp",
        action="store_true",
        default=False,
        help="Enable the Siemens 6DoF grasp/action path (5D policy action + FT-controlled x).",
    )
    p.add_argument("--no_ft_sensor", action="store_true", default=False)
    p.add_argument("--eval", action="store_true", default=True)
    p.add_argument("--max_episodes", type=int, default=3)

    p.add_argument(
        "--mp_backend",
        type=str,
        choices=["quintic", "ruckig", "spline"],
        default="ruckig",
        help="Motion-planner backend inside MotionPlannerWrapper.",
    )
    p.add_argument(
        "--spline_waypoint_mode",
        type=str,
        choices=["single", "lift_xy_then_descend"],
        default="single",
        help="How spline backend constructs waypoint list per episode.",
    )
    p.add_argument(
        "--spline_lift_m",
        type=float,
        default=0.02,
        help="Lift distance (m) used by spline lift_xy_then_descend mode.",
    )
    p.add_argument("--max_linear_vel", type=float, default=0.02)
    p.add_argument("--max_linear_acc", type=float, default=0.5)
    p.add_argument("--max_linear_jerk", type=float, default=5.0)
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
        "--max_policy_steps",
        type=int,
        default=200,
        help="Max env.step() ticks of RL policy rollout per episode when FT "
        "sensor is connected (i.e. --no_ft_sensor not set).",
    )

    p.add_argument(
        "--success_threshold",
        type=float,
        default=9.3,
        help="Classifier threshold on mean Q(s, pi(s)); forwarded to env.",
    )
    p.add_argument("--no_ft_success_threshold", type=float, default=4.0)

    p.add_argument(
        "--pose_viz_dir",
        type=str,
        default=None,
        help="If set, save per-reset pose-estimation overlays (PNG) + raw NPZ "
        "artifacts into this directory for offline inspection.",
    )
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
    force_zero_settle = False
    if args.use_6dof_grasp:
        if args.task != "siemens":
            raise ValueError("--use_6dof_grasp is currently only supported for --task siemens.")
        if not args.no_ft_sensor:
            logger.warning(
                "6DoF grasp mode with legacy 2D checkpoints cannot run post-MP RL policy rollout. "
                "Keeping FT enabled, but forcing zero-action settle path."
            )
            force_zero_settle = True

    logger.info(
        "Config: task=%s use_pose_estimation=%s use_6dof_grasp=%s mp_backend=%s",
        args.task,
        args.use_pose_estimation,
        args.use_6dof_grasp,
        args.mp_backend,
    )

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
            "env has no plan_and_execute — MotionPlannerWrapper was not attached."
        )
        return 1

    policy_bundle = None
    if not args.no_ft_sensor and not force_zero_settle:
        actor, shared_encoder, _device = _load_policy(args.load_policy, cfg, logger)
        policy_bundle = (actor, shared_encoder, cfg)
        logger.info(
            "FT sensor path: post-MP rollout uses RL policy (max_policy_steps=%d).",
            args.max_policy_steps,
        )
    elif force_zero_settle:
        logger.info(
            "FT sensor path: using zero-action settle (max_settle_steps=%d) "
            "because checkpoint action dim is incompatible with 6DoF policy rollout.",
            args.max_settle_steps,
        )
    else:
        logger.info(
            "no_ft_sensor path: post-MP rollout uses zero-action settle "
            "(max_settle_steps=%d).",
            args.max_settle_steps,
        )

    summaries = []
    try:
        for i in range(args.max_episodes):
            logger.info("=== episode %d/%d ===", i + 1, args.max_episodes)
            summaries.append(
                _run_episode(env, args, i + 1, logger, policy_bundle=policy_bundle)
            )
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
