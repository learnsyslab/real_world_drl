import json
import os
from pathlib import Path
import time
import imageio
import numpy as np
import pickle

from crisp_drl.agents.shared.algorithm_config import Config
from crisp_drl.envs import make_env, make_rew
import sys
# import debugpy


# Alternate between going to the goal position and moving randomly with probaility p
ps = [0.8]
N_ROLLOUTS = 25
max_random_action_magnitude = 0.3e-3
perfect_action_magnitude = 0.00025

base_exp_name = ""
if len(sys.argv) > 1:
    base_exp_name = sys.argv[1]

config = Config()


env = make_env.create_real_env_v3_timo(config)


reward_fn = make_rew.create_real_reward_fn(
    config,
    event_reward_map={
        "E_SUCCESS": 0.1 / (1 - config.gamma) * 3,
        "E_FAIL": -0.1 / (1 - config.gamma) / 2 * 3,
        "E_SAFETY_BOX_VIOLATION": 0.0,  # -0.05,
        "E_ROLLOUT_UNUSABLE": -100,
        "E_CONTROLLER_ISSUE": -100,
        "E_TORQUE": -100,
    },
)


for p in ps:
    if base_exp_name == "":
        exp_name = str(p)
    else:
        exp_name = f"{base_exp_name}_{p}"

    current_date = time.strftime("%Y%m%d-%H%M%S")
    data_dir = Path(f"rollout_data/collect_data_real/{current_date}_{exp_name}")
    data_dir.mkdir(parents=True, exist_ok=True)
    n_success = 0
    total_successful_length = 0
    i = 0
    while i < N_ROLLOUTS:
        # new reset wrapper: automatically place brick back and pick it up again
        obs, reset_info = env.reset()
        # compute goal position
        actual_grasp_pos = reset_info["reset.grasped.position"]
        goal_pos = config.goal_position_ground_truth[:2].copy()
        goal_pos[0] += actual_grasp_pos[0] - config.grasp_position_ground_truth[0]

        all_actions = []
        all_rewards = []
        all_perfect_actions = []
        all_observations = [obs.cpu().numpy()]
        all_infos = [reset_info]
        info = reset_info

        while True:
            perfect_action = (
                goal_pos - info["observation"]["observation.state.cartesian"][:2]
            )
            perfect_action = (
                perfect_action
                / np.linalg.norm(perfect_action)
                * perfect_action_magnitude
            )
            all_perfect_actions.append(perfect_action)

            if np.random.rand() < p:
                action = np.random.uniform(
                    -max_random_action_magnitude,
                    max_random_action_magnitude,
                    size=(2,),
                )
            else:
                action = perfect_action
            all_actions.append(action.copy())

            obs, reward, terminated, truncated, info = env.step(action)
            all_observations.append(obs.cpu().numpy())
            all_infos.append(info)
            all_rewards.append(reward)

            if terminated or truncated:
                break

        # compute rewards
        all_actions, all_observations, all_rewards, all_infos = reward_fn(
            all_actions,
            all_observations,
            all_rewards,
            all_infos,
            actual_grasp_pos_xy=actual_grasp_pos[:2],
        )
        last_custom_events = map(lambda x: x[1], all_infos[-1].get("custom_events", []))
        if (
            "E_ROLLOUT_UNUSABLE" in last_custom_events
            or "E_CONTROLLER_ISSUE" in last_custom_events
            or "E_TORQUE" in last_custom_events
        ):
            print("Rollout unusable, skipping saving this rollout.")
            continue

        if terminated:
            n_success += 1
            total_successful_length += len(all_actions)
        print(
            f"Episode finished: Length = {len(all_actions)}, Success: {terminated}, success rate = {n_success / (i + 1) * 100:.2f}%; i = {i}"
        )
        # save data
        np.savez_compressed(
            data_dir / f"rollout_{i:03d}.npz",
            observations=all_observations,
            actions=all_actions,
            rewards=all_rewards,
            perfect_actions=all_perfect_actions,
            terminated=terminated,
        )
        all_wrist_cam_images = [
            info["observation"]["observation.images.wrist_camera_raw"]
            for info in all_infos
        ]
        imageio.mimwrite(
            os.path.join(data_dir / f"rollout_{i:03d}_info_wrist_camera.mp4"),
            all_wrist_cam_images,
            format="mp4",  # pyright: ignore[reportArgumentType]
            fps=config.control_frequency,
            codec="libx264",
        )  # type: ignore

        for info in all_infos:
            # delete large images to save space
            if (
                "observation" in info
                and "observation.images.wrist_camera_raw" in info["observation"]
            ):
                info["observation"].pop("observation.images.wrist_camera_raw")
            if (
                "observation" in info
                and "observation.images.wrist_depth_camera_raw" in info["observation"]
            ):
                info["observation"].pop("observation.images.wrist_depth_camera_raw")

        if (
            "reset.pose_estimation.purple" in reset_info
            and "reset.pose_estimation.lavender" in reset_info
            and "reset.pose_estimation.joint_state" in reset_info
            and "reset.pose_estimation.cartesian" in reset_info
        ):
            with open(data_dir / f"rollout_{i:03d}_pose_estimation.json", "w") as f:
                json.dump(
                    {
                        "purple": reset_info["reset.pose_estimation.purple"].tolist(),
                        "lavender": reset_info[
                            "reset.pose_estimation.lavender"
                        ].tolist(),
                        "joint_state": reset_info[
                            "reset.pose_estimation.joint_state"
                        ].tolist(),
                        "cartesian": reset_info[
                            "reset.pose_estimation.cartesian"
                        ].tolist(),
                    },
                    f,
                )

        with open(data_dir / f"rollout_{i:03d}_info.pkl", "wb") as f:
            pickle.dump(all_infos, f)

        i += 1

    # rename data_dir to include success rate
    success_rate = n_success / N_ROLLOUTS * 100
    new_data_dir = data_dir.parent / f"{data_dir.name}_{int(success_rate):02d}"
    data_dir.rename(new_data_dir)
    print(
        f"average successful length: {total_successful_length / n_success if n_success > 0 else 0:.2f}"
    )
    print(f"Data directory renamed to include success rate: {new_data_dir}")

env.close()

# average successful length: 91.50
# Data directory renamed to include success rate: rollout_data/collect_data_real/20260105-182707_1.0_40

# average successful length: 81.50
# Data directory renamed to include success rate: rollout_data/collect_data_real/20260105-183306_0.85_80
