#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from crisp_drl.agents.shared.env_wrappers import (
    ActionTimeStampWrapper,
    LastObservationWrapper,
)
from crisp_drl.envs.make_env import make_env
from crisp_drl.envs.sam3_pe import PoseEstimationHelper


INITIAL_HOME_CONFIG = np.array(
    [
        -0.16125268,
        -0.19791204,
        -0.32023647,
        -2.4182775,
        -0.08884472,
        2.2305312,
        0.36234158,
    ],
    dtype=np.float32,
)

REPEAT_HOME_CONFIG = np.array(
    [
        -0.29983243,
        0.38195968,
        -0.15804778,
        -2.2734513,
        0.08058647,
        2.640415,
        0.26108682,
    ],
    dtype=np.float32,
)

TARGET_XYZ = np.array([0.475, -0.242, 0.085], dtype=np.float32)
RELATIVE_XYZ = np.array([-0.005, 0.0, 0.02], dtype=np.float32)
RELATIVE_EULER = np.array([0.0, 0.0, -np.deg2rad(1.0)], dtype=np.float32)


def action_dim_from_env(env) -> int:
    action_shape = getattr(env.action_space, "shape", None)
    if not action_shape:
        return 3
    return int(action_shape[0])


def wrap_angle_to_half_turn(angle_rad: float) -> float:
    return float((angle_rad + np.pi / 2.0) % np.pi - np.pi / 2.0)


def move_to_target(
    env,
    current_obs: dict[str, np.ndarray],
    target_xyz: np.ndarray,
    target_euler_rad: np.ndarray | None = None,
    tolerance: float = 0.001,
    orientation_tolerance_rad: float = 0.01,
    settle_steps: int = 3,
    max_settle_iterations: int = 250,
) -> dict[str, np.ndarray]:
    action_dim = action_dim_from_env(env)
    action = np.zeros(action_dim, dtype=np.float32)
    current_state = current_obs["observation.state.cartesian"]
    action[:3] = target_xyz - current_state[:3]

    if target_euler_rad is not None and action_dim > 5:
        action[3:6] = (
            np.asarray(target_euler_rad, dtype=np.float32) - current_state[3:6]
        )

    obs, *_ = env.step(action)

    for _ in range(max_settle_iterations):
        current_xyz = obs["observation.state.cartesian"][:3]
        current_euler = obs["observation.state.cartesian"][3:6]
        position_error = float(np.linalg.norm(target_xyz - current_xyz))
        velocity_error = float(
            np.linalg.norm(obs["observation.velocity.cartesian"][:3])
        )
        orientation_error = 0.0

        if target_euler_rad is not None:
            orientation_error = float(
                np.linalg.norm(
                    np.asarray(target_euler_rad, dtype=np.float32) - current_euler
                )
            )

        if (
            position_error <= tolerance
            and velocity_error <= tolerance / 2.0
            and orientation_error <= orientation_tolerance_rad
        ):
            break

        if velocity_error <= tolerance / 2.0 and position_error > tolerance:
            action = np.zeros(action_dim, dtype=np.float32)
            action[:3] = target_xyz - current_xyz
            if target_euler_rad is not None and action_dim > 5:
                action[3:6] = (
                    np.asarray(target_euler_rad, dtype=np.float32) - current_euler
                )
            obs, *_ = env.step(action)
        else:
            obs, *_ = env.step(np.zeros(action_dim, dtype=np.float32))

    for _ in range(settle_steps):
        obs, *_ = env.step(np.zeros(action_dim, dtype=np.float32))

    return obs


def move_relative(
    env,
    obs: dict[str, np.ndarray],
    delta_xyz: np.ndarray,
    delta_euler_rad: np.ndarray | None = None,
    settle_steps: int = 3,
) -> dict[str, np.ndarray]:
    action_dim = action_dim_from_env(env)
    action = np.zeros(action_dim, dtype=np.float32)
    action[:3] = np.asarray(delta_xyz, dtype=np.float32)
    if delta_euler_rad is not None and action_dim > 5:
        action[3:6] = np.asarray(delta_euler_rad, dtype=np.float32)

    obs, *_ = env.step(action)
    for _ in range(settle_steps):
        obs, *_ = env.step(np.zeros(action_dim, dtype=np.float32))
    return obs


def estimate_barcode_poses_tcp(
    pose_helper: PoseEstimationHelper, obs: dict[str, np.ndarray]
) -> list[np.ndarray]:
    return pose_helper.estimate_barcodes_tcp(
        (
            obs["observation.images.wrist_camera"],
            obs["observation.images.wrist_depth_camera"],
        )
    )


def write_jsonl(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record) + "\n")
    handle.flush()


def mean_pose_vector(poses: list[np.ndarray]) -> np.ndarray:
    return np.mean(
        np.stack([np.asarray(pose, dtype=np.float32) for pose in poses]), axis=0
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Home the robot, repeat a grasp and TCP pose-estimation sequence, and "
            "save all yellow barcode TCP pose estimates to a JSONL log."
        )
    )
    parser.add_argument("--exp-name", required=True, help="Name used for the log file")
    parser.add_argument(
        "--cycles",
        type=int,
        default=10,
        help="How many times to repeat the grasp-and-estimate sequence",
    )
    parser.add_argument(
        "--env-name",
        default="my_env_v4",
        help="Environment name passed to make_env",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional explicit output path for the JSONL log",
    )
    args = parser.parse_args()

    output_path = (
        args.output_path or Path("rollout_data") / "tests" / f"{args.exp_name}.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pose_helper = PoseEstimationHelper(
        centroid_method="mean_pixel_depth",
        full_mask_prompt="yellow cardboard box",
        width_m=0.03,
        height_m=0.04,
        filter_depth_outliers=True,
    )

    env = make_env(args.env_name)
    print(f"Env created: {args.env_name}")
    env.wait_until_ready()
    print("Env ready.")

    env = ActionTimeStampWrapper(env)
    env = LastObservationWrapper(env)

    all_tcp_poses: list[np.ndarray] = []

    try:
        with output_path.open("w", encoding="utf-8") as handle:
            base_env: Any = env.unwrapped

            print("Initial homing...")
            base_env.home(home_config=INITIAL_HOME_CONFIG.tolist())
            base_env.gripper.open()
            obs, *_ = env.reset()
            obs, *_ = env.step(np.zeros(action_dim_from_env(env), dtype=np.float32))

            for cycle_idx in range(args.cycles):
                print(
                    f"\n[Cycle {cycle_idx + 1}/{args.cycles}] Homing to repeat pose..."
                )
                base_env.home(home_config=REPEAT_HOME_CONFIG.tolist())
                base_env.gripper.open()
                obs, *_ = env.reset()
                obs, *_ = env.step(np.zeros(action_dim_from_env(env), dtype=np.float32))

                print(f"  Moving to target xyz: {TARGET_XYZ.tolist()}")
                obs = move_to_target(env, obs, TARGET_XYZ)

                print("  Grasping...")
                base_env.gripper.set_target(0.4)
                time.sleep(1.0)
                obs, *_ = env.step(np.zeros(action_dim_from_env(env), dtype=np.float32))

                print(f"  Moving relative xyz: {RELATIVE_XYZ.tolist()}")
                obs = move_relative(env, obs, RELATIVE_XYZ)

                print(f"  Moving relative euler: {RELATIVE_EULER.tolist()}")
                obs = move_relative(
                    env, obs, np.zeros(3, dtype=np.float32), RELATIVE_EULER
                )

                print("  Estimating yellow barcode poses in tcp frame...")
                tcp_poses = estimate_barcode_poses_tcp(pose_helper, obs)
                if not tcp_poses:
                    raise RuntimeError("No yellow barcode TCP poses detected")

                cycle_mean = mean_pose_vector(tcp_poses)
                all_tcp_poses.extend(tcp_poses)

                write_jsonl(
                    handle,
                    {
                        "type": "cycle_estimates",
                        "cycle": cycle_idx,
                        "poses_tcp": [
                            np.asarray(pose, dtype=np.float32).tolist()
                            for pose in tcp_poses
                        ],
                        "mean_pose_tcp": cycle_mean.astype(np.float32).tolist(),
                        "count": len(tcp_poses),
                    },
                )

                print("  Opening gripper...")
                base_env.gripper.open()
                time.sleep(5.0)
                obs, *_ = env.step(np.zeros(action_dim_from_env(env), dtype=np.float32))

            run_mean = mean_pose_vector(all_tcp_poses)
            write_jsonl(
                handle,
                {
                    "type": "run_summary",
                    "cycles": args.cycles,
                    "total_estimates": len(all_tcp_poses),
                    "mean_pose_tcp": run_mean.astype(np.float32).tolist(),
                },
            )

        print(f"Saved TCP pose estimates to {output_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
