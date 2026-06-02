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


def action_dim_from_env(env) -> int:
    action_shape = getattr(env.action_space, "shape", None)
    if not action_shape:
        return 3
    return int(action_shape[0])


def wrap_angle_to_half_turn(angle_rad: float) -> float:
    return float((angle_rad + np.pi / 2.0) % np.pi - np.pi / 2.0)


def object_frame_z_rotation_mod_180_rad(pose_world: np.ndarray) -> float:
    yaw_rad = float(np.arctan2(pose_world[1, 0], pose_world[0, 0]))
    return wrap_angle_to_half_turn(yaw_rad)


def barcode_offset_position(
    pose_world: np.ndarray,
    offset_local_m: np.ndarray,
    local: bool,
) -> np.ndarray:
    if local:
        return pose_world[:3, 3] + pose_world[:3, :3] @ offset_local_m
    return pose_world[:3, 3] + offset_local_m


def estimate_barcode_poses(
    pose_helper: PoseEstimationHelper, obs: dict[str, np.ndarray], label: str
) -> list[np.ndarray]:
    poses = pose_helper.estimate_barcodes(
        (
            obs["observation.images.wrist_camera"],
            obs["observation.images.wrist_depth_camera"],
        ),
        obs["observation.state.cartesian"],
    )
    print(f"Received the following {label} barcode pose estimates (world frame):")
    for i, pose in enumerate(poses):
        print(f"  {label} barcode {i + 1}: {pose[:3, 3]}")
    return poses


def write_pose_line(
    handle,
    cycle_idx: int,
    stage: str,
    poses_world: list[np.ndarray],
) -> None:
    record = {
        "cycle": cycle_idx,
        "stage": stage,
        "poses_world": [pose.astype(np.float32).tolist() for pose in poses_world],
    }
    handle.write(json.dumps(record) + "\n")
    handle.flush()


def move_to_target(
    env,
    current_obs: dict[str, np.ndarray],
    target_xyz: np.ndarray,
    target_euler_rad: np.ndarray | None = None,
    target_z_rotation_rad: float | None = None,
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
    elif target_z_rotation_rad is not None and action_dim > 5:
        action[5] = wrap_angle_to_half_turn(
            target_z_rotation_rad - float(current_state[5])
        )

    obs, *_ = env.step(action)
    last_i_term = False

    for _ in range(max_settle_iterations):
        current_xyz = obs["observation.state.cartesian"][:3]
        current_euler = obs["observation.state.cartesian"][3:6]
        delta = target_xyz - current_xyz
        position_error = float(np.linalg.norm(delta))
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
        elif target_z_rotation_rad is not None:
            orientation_error = float(
                abs(
                    wrap_angle_to_half_turn(
                        target_z_rotation_rad - float(current_euler[2])
                    )
                )
            )

        if (
            position_error <= tolerance
            and velocity_error <= tolerance / 2
            and orientation_error <= orientation_tolerance_rad
        ):
            break

        if (
            velocity_error <= tolerance / 2
            and (
                position_error > tolerance
                or orientation_error > orientation_tolerance_rad
            )
            and not last_i_term
        ):
            action = np.zeros(action_dim, dtype=np.float32)
            action[:3] = delta
            if target_euler_rad is not None and action_dim > 5:
                action[3:6] = (
                    np.asarray(target_euler_rad, dtype=np.float32) - current_euler
                )
            elif target_z_rotation_rad is not None and action_dim > 5:
                action[5] = wrap_angle_to_half_turn(
                    target_z_rotation_rad - float(current_euler[2])
                )
            obs, *_ = env.step(action)
            last_i_term = True
        else:
            obs, *_ = env.step(np.zeros(action_dim, dtype=np.float32))
            last_i_term = False

    for _ in range(settle_steps):
        obs, *_ = env.step(np.zeros(action_dim, dtype=np.float32))

    return obs


def rotate_in_place(
    env, obs: dict[str, np.ndarray], pitch_rad: float
) -> dict[str, np.ndarray]:
    current_state = obs["observation.state.cartesian"]
    target_xyz = current_state[:3].copy()
    target_euler = current_state[3:6].copy()
    target_euler[1] += pitch_rad
    return move_to_target(env, obs, target_xyz, target_euler_rad=target_euler)


def descend_along_axis_until_force(
    env,
    obs: dict[str, np.ndarray],
    axis: np.ndarray,
    target_force_delta: float = 1.0,
    step_m: float = 0.001,
    max_contact_force: float = 3.0,
    correction_gain: float = 0.5,
    max_correction_step: float = 0.0005,
    max_iterations: int = 2000,
) -> tuple[dict[str, np.ndarray], bool]:
    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / float(np.linalg.norm(axis))
    start_pos = obs["observation.state.cartesian"][:3].copy()
    initial_z = float(obs["observation.state.sensors_bota_ft_sensor"][2])
    traveled = 0.0
    action_dim = action_dim_from_env(env)

    for _ in range(max_iterations):
        desired_traveled = traveled + step_m
        desired_pos = start_pos + axis * desired_traveled

        current_pos = obs["observation.state.cartesian"][:3]
        error = desired_pos - current_pos

        # correction perpendicular to axis
        error_along = np.dot(error, axis) * axis
        perp_error = error - error_along
        if np.linalg.norm(perp_error) > 0.0:
            corr_step = correction_gain * perp_error
            corr_norm = float(np.linalg.norm(corr_step))
            if corr_norm > max_correction_step:
                corr_step = corr_step / corr_norm * max_correction_step
        else:
            corr_step = np.zeros(3, dtype=np.float32)

        step_vec = axis * step_m + corr_step
        action = np.zeros(action_dim, dtype=np.float32)
        action[:3] = step_vec
        obs, *_ = env.step(action)
        traveled = float(
            np.linalg.norm((obs["observation.state.cartesian"][:3] - start_pos))
        )

        current_z = float(obs["observation.state.sensors_bota_ft_sensor"][2])
        if abs(current_z - initial_z) >= target_force_delta:
            return obs, True

        # safety: if measured z force magnitude exceeds max_contact_force, abort
        if abs(current_z) > max_contact_force:
            print(
                f"Aborting descent: z-force {current_z:.3f} N exceeds safety {max_contact_force} N"
            )
            return obs, False

    print("Descent did not reach target force within max iterations")
    return obs, False


def move_along_axis_carefully(
    env,
    obs: dict[str, np.ndarray],
    axis: np.ndarray,
    distance_m: float,
    step_m: float = 0.001,
    correction_gain: float = 0.5,
    max_correction_step: float = 0.0005,
    max_iterations: int = 2000,
    settle_steps: int = 3,
) -> dict[str, np.ndarray]:
    axis = np.asarray(axis, dtype=np.float32)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm == 0.0:
        raise ValueError("Axis must be non-zero")

    direction = axis / axis_norm
    if distance_m < 0:
        direction = -direction
    target_distance = abs(float(distance_m))

    start_pos = obs["observation.state.cartesian"][:3].copy()
    traveled = 0.0
    action_dim = action_dim_from_env(env)

    for _ in range(max_iterations):
        if traveled >= target_distance:
            break

        desired_traveled = min(target_distance, traveled + step_m)
        desired_pos = start_pos + direction * desired_traveled

        current_pos = obs["observation.state.cartesian"][:3]
        error = desired_pos - current_pos
        error_along = np.dot(error, direction) * direction
        perp_error = error - error_along

        if np.linalg.norm(perp_error) > 0.0:
            corr_step = correction_gain * perp_error
            corr_norm = float(np.linalg.norm(corr_step))
            if corr_norm > max_correction_step:
                corr_step = corr_step / corr_norm * max_correction_step
        else:
            corr_step = np.zeros(3, dtype=np.float32)

        action = np.zeros(action_dim, dtype=np.float32)
        action[:3] = direction * step_m + corr_step
        obs, *_ = env.step(action)
        traveled = float(
            np.dot(obs["observation.state.cartesian"][:3] - start_pos, direction)
        )

    for _ in range(settle_steps):
        obs, *_ = env.step(np.zeros(action_dim, dtype=np.float32))

    return obs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a dual-barcode SAM3 rollout: estimate blue and yellow barcodes, "
            "refine both, then grasp and reposition toward the blue target."
        )
    )
    parser.add_argument(
        "--exp-name",
        required=True,
        help="Experiment name used to derive rollout output paths",
    )
    blue_mode_group = parser.add_mutually_exclusive_group()
    blue_mode_group.add_argument(
        "--blue-initial-only",
        action="store_true",
        help=(
            "Only run the initial blue estimate, then skip blue refinement and "
            "continue with the regular yellow estimation path"
        ),
    )
    blue_mode_group.add_argument(
        "--blue-above-only",
        action="store_true",
        help=(
            "Run the initial blue estimate and the blue above-pose estimate, "
            "then skip the remaining blue refinement and continue with yellow"
        ),
    )
    args = parser.parse_args()

    env_name = "my_env_v4"
    output_path = Path("rollout_data") / "tests" / f"{args.exp_name}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # blue_initial_offset_m = np.asarray((0.0, 0.004, 0.115), dtype=np.float32)
    blue_initial_offset_m = np.asarray((0.0, 0.004, 0.14), dtype=np.float32)
    blue_final_offset_m = np.asarray((0.0, 0.004, 0.065), dtype=np.float32)
    yellow_initial_offset_m = np.asarray((0.05, 0.0, 0.1), dtype=np.float32)
    yellow_final_offset_m = np.asarray((0.05, 0.00, -0.02), dtype=np.float32)
    blue_return_offset_m = np.asarray((0.017, 0.048, 0.07), dtype=np.float32)

    blue_helper = PoseEstimationHelper(
        centroid_method="mean_pixel_depth",
        full_mask_prompt="blue cardboard box",
        width_m=0.026,
        height_m=0.0375,
        filter_depth_outliers=True,
    )
    yellow_helper = PoseEstimationHelper(
        centroid_method="mean_pixel_depth",
        full_mask_prompt="yellow cardboard box",
        width_m=0.03,
        height_m=0.04,
        filter_depth_outliers=True,
        processor=blue_helper.processor,
    )

    env = make_env(env_name)
    print(f"Env created: {env_name}")
    env.wait_until_ready()
    print("Env ready.")

    env = ActionTimeStampWrapper(env)
    env = LastObservationWrapper(env)

    try:
        with output_path.open("w", encoding="utf-8") as handle:
            base_env: Any = env.unwrapped
            for cycle_idx in range(10):
                print(f"\n[Cycle {cycle_idx + 1}/10] Homing and opening gripper...")
                base_env.home(
                    # home_config=[
                    #     -0.16125268,
                    #     -0.19791204,
                    #     -0.32023647,
                    #     -2.4182775,
                    #     -0.08884472,
                    #     2.2305312,
                    #     0.36234158,
                    # ]
                    home_config=[
                        -1.6601315e-01,
                        7.1060816e-03,
                        -2.8422508e-01,
                        -2.0143490e00,
                        1.0296288e-03,
                        2.0213308e00,
                        3.2737154e-01,
                    ]
                )
                base_env.gripper.open()

                obs, *_ = env.reset()
                obs, *_ = env.step(np.zeros(action_dim_from_env(env), dtype=np.float32))

                print("  Estimating initial blue barcode poses...")
                blue_initial_poses = estimate_barcode_poses(blue_helper, obs, "blue")
                if not blue_initial_poses:
                    raise RuntimeError(
                        "No blue barcode poses detected in initial estimate"
                    )
                write_pose_line(handle, cycle_idx, "initial_blue", blue_initial_poses)

                print("  Estimating initial yellow barcode poses...")
                yellow_initial_poses = estimate_barcode_poses(
                    yellow_helper, obs, "yellow"
                )
                if not yellow_initial_poses:
                    raise RuntimeError(
                        "No yellow barcode poses detected in initial estimate"
                    )
                write_pose_line(
                    handle, cycle_idx, "initial_yellow", yellow_initial_poses
                )

                if args.blue_initial_only:
                    print(
                        "  Skipping blue refinement after initial estimation "
                        "(--blue-initial-only)"
                    )
                    blue_return_pose_source = blue_initial_poses
                else:
                    blue_initial_pose = blue_initial_poses[0]
                    blue_above_target = barcode_offset_position(
                        blue_initial_pose, blue_initial_offset_m, local=False
                    )
                    print(f"  Moving to blue above-target pose: {blue_above_target}")
                    obs = move_to_target(env, obs, blue_above_target)

                    print("  Estimating blue pose above target...")
                    blue_above_poses = estimate_barcode_poses(blue_helper, obs, "blue")
                    if not blue_above_poses:
                        raise RuntimeError(
                            "No blue barcode poses detected above target"
                        )
                    write_pose_line(handle, cycle_idx, "blue_above", blue_above_poses)

                    if args.blue_above_only:
                        print(
                            "  Skipping blue final refinement after above-pose "
                            "estimation (--blue-above-only)"
                        )
                        blue_return_pose_source = blue_above_poses
                    else:
                        blue_above_pose = blue_above_poses[0]
                        blue_final_target = barcode_offset_position(
                            blue_above_pose, blue_final_offset_m, local=False
                        )
                        print(f"  Moving to blue final target: {blue_final_target}")
                        obs = move_to_target(env, obs, blue_final_target)

                        print("  Estimating blue final pose...")
                        blue_final_poses = estimate_barcode_poses(
                            blue_helper, obs, "blue"
                        )
                        if not blue_final_poses:
                            raise RuntimeError(
                                "No blue barcode poses detected at final target"
                            )
                        write_pose_line(
                            handle, cycle_idx, "blue_final", blue_final_poses
                        )
                        blue_return_pose_source = blue_final_poses

                yellow_initial_pose = yellow_initial_poses[0]
                yellow_above_target = barcode_offset_position(
                    yellow_initial_pose, yellow_initial_offset_m, local=True
                )
                yellow_above_rotation_rad = object_frame_z_rotation_mod_180_rad(
                    yellow_initial_pose
                )
                print(f"  Moving to yellow above-target pose: {yellow_above_target}")
                obs = move_to_target(
                    env,
                    obs,
                    yellow_above_target,
                    target_z_rotation_rad=yellow_above_rotation_rad,
                )

                print("  Estimating yellow pose above target...")
                yellow_above_poses = estimate_barcode_poses(
                    yellow_helper, obs, "yellow"
                )
                if not yellow_above_poses:
                    raise RuntimeError("No yellow barcode poses detected above target")
                write_pose_line(handle, cycle_idx, "yellow_above", yellow_above_poses)

                yellow_above_pose = yellow_above_poses[0]
                yellow_final_target = barcode_offset_position(
                    yellow_above_pose, yellow_final_offset_m, local=True
                )
                yellow_final_rotation_rad = object_frame_z_rotation_mod_180_rad(
                    yellow_above_pose
                )
                print(f"  Moving to yellow final target: {yellow_final_target}")
                obs = move_to_target(
                    env,
                    obs,
                    yellow_final_target,
                    target_z_rotation_rad=yellow_final_rotation_rad,
                )

                print("  Estimating yellow final pose...")
                yellow_final_poses = estimate_barcode_poses(
                    yellow_helper, obs, "yellow"
                )
                if not yellow_final_poses:
                    raise RuntimeError(
                        "No yellow barcode poses detected at final target"
                    )
                write_pose_line(handle, cycle_idx, "yellow_final", yellow_final_poses)

                yellow_final_target = barcode_offset_position(
                    yellow_final_poses[0], yellow_final_offset_m, local=True
                )
                yellow_final_rotation_rad = object_frame_z_rotation_mod_180_rad(
                    yellow_final_poses[0]
                )

                print("  Moving once more to yellow final target after estimation...")
                obs = move_to_target(
                    env,
                    obs,
                    yellow_final_target,
                    target_z_rotation_rad=yellow_final_rotation_rad,
                )

                print("  Closing gripper to 0.4 and waiting 2 seconds...")
                base_env.gripper.set_target(0.6)
                time.sleep(1.0)
                base_env.gripper.set_target(0.4)
                time.sleep(1.0)
                obs, *_ = env.step(np.zeros(action_dim_from_env(env), dtype=np.float32))
                lift_target = obs["observation.state.cartesian"][:3] + np.array(
                    [0.0, 0.0, 0.1], dtype=np.float32
                )
                print(f"  Moving up 10 cm: {lift_target}")
                obs = move_to_target(env, obs, lift_target)

                print("  Undoing z-rotation to align grasped object and re-estimate...")
                current_state = obs["observation.state.cartesian"]
                current_xyz = current_state[:3].copy()
                desired_z = float(current_state[5]) - yellow_final_rotation_rad
                obs = move_to_target(
                    env, obs, current_xyz, target_z_rotation_rad=desired_z
                )

                print("  Estimating grasped yellow barcode pose after undo rotation...")
                yellow_grasped_poses = estimate_barcode_poses(
                    yellow_helper, obs, "yellow_grasped"
                )
                if not yellow_grasped_poses:
                    print("Warning: No yellow barcode poses detected after grasp")
                else:
                    write_pose_line(
                        handle, cycle_idx, "yellow_grasped", yellow_grasped_poses
                    )

                print("  Rotating by +30 deg around the y-axis...")
                grasp_reference_state = obs["observation.state.cartesian"].copy()
                target_xyz = grasp_reference_state[:3].copy()
                target_euler = grasp_reference_state[3:6].copy()
                target_euler[1] += np.deg2rad(10.0)
                obs = move_to_target(
                    env,
                    obs,
                    target_xyz,
                    target_euler_rad=target_euler,
                    orientation_tolerance_rad=np.deg2rad(3.0),
                    max_settle_iterations=5,
                )
                target_euler[1] += np.deg2rad(10.0)
                obs = move_to_target(
                    env,
                    obs,
                    target_xyz,
                    target_euler_rad=target_euler,
                    orientation_tolerance_rad=np.deg2rad(3.0),
                    max_settle_iterations=5,
                )
                target_euler[1] += np.deg2rad(10.0)
                obs = move_to_target(
                    env,
                    obs,
                    target_xyz,
                    target_euler_rad=target_euler,
                    orientation_tolerance_rad=np.deg2rad(3.0),
                )

                blue_return_target = barcode_offset_position(
                    blue_return_pose_source[0], blue_return_offset_m, local=False
                )
                print(
                    "  Moving back to refined blue position with global offset "
                    f"{blue_return_offset_m.tolist()}: {blue_return_target}"
                )
                obs = move_to_target(env, obs, blue_return_target)

                descend_axis = np.asarray([-0.5, 0.0, -0.7071], dtype=np.float32)
                print("  Descending along axis to establish contact force delta...")
                obs, descended = descend_along_axis_until_force(
                    env,
                    obs,
                    axis=descend_axis,
                    target_force_delta=1.0,
                    step_m=0.0008,
                    max_contact_force=20.0,
                    correction_gain=0.6,
                    max_correction_step=0.0006,
                    max_iterations=2000,
                )
                if descended:
                    print("  Contact force delta reached during descent.")
                    time.sleep(1.0)
                else:
                    raise RuntimeError(
                        "Contact force was not established during descent"
                    )

                print("  Retreating 5 cm along the same axis carefully...")
                obs = move_along_axis_carefully(
                    env,
                    obs,
                    axis=-descend_axis,
                    distance_m=0.05,
                    step_m=0.0008,
                    correction_gain=0.6,
                    max_correction_step=0.0006,
                    max_iterations=2000,
                )

                grasp_above_target = grasp_reference_state[:3].copy()
                grasp_above_target[2] += 0.07
                print(
                    f"  Moving quickly to 7 cm above grasp position: {grasp_above_target}"
                )
                obs = move_to_target(
                    env,
                    obs,
                    grasp_above_target,
                    settle_steps=1,
                    max_settle_iterations=120,
                )

                print("  Rotating back 30 deg and moving 14 cm down...")
                current_state = obs["observation.state.cartesian"]
                return_euler = current_state[3:6].copy()
                return_euler[1] -= np.deg2rad(30.0)
                down_target = grasp_above_target.copy()
                down_target[2] -= 0.14
                obs = move_to_target(
                    env,
                    obs,
                    down_target,
                    target_euler_rad=return_euler,
                    settle_steps=1,
                    max_settle_iterations=120,
                    orientation_tolerance_rad=np.deg2rad(3.0),
                )

                print("  Applying random z rotation before opening gripper...")
                z_random_rotation_rad = np.deg2rad(np.random.uniform(-30.0, 30.0))
                current_state = obs["observation.state.cartesian"]
                current_xyz = current_state[:3].copy()
                current_euler = current_state[3:6].copy()
                current_euler[2] += z_random_rotation_rad
                obs = move_to_target(
                    env,
                    obs,
                    current_xyz,
                    target_euler_rad=current_euler,
                    orientation_tolerance_rad=np.deg2rad(3.0),
                    max_settle_iterations=120,
                )

                print("  Opening gripper...")
                base_env.gripper.open()
                time.sleep(0.5)
                obs, *_ = env.step(np.zeros(action_dim_from_env(env), dtype=np.float32))

                print(f"  Cycle {cycle_idx + 1}/10 complete.")

        print(f"Saved barcode pose estimates to {output_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
