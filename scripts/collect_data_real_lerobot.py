from pathlib import Path
import threading
import time
import numpy as np

from crisp_drl.agents.shared.config import Config
from crisp_drl.envs import make_env, make_rew

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Alternate between going to the goal position and moving randomly with probability p
p = 0.8
N_ROLLOUTS = 30
max_random_action_magnitude = 0.3e-3
perfect_action_magnitude = 0.00025

base_exp_name = "env_v0"
exp_name = f"{base_exp_name}_{p}"

repo_id = f"collect_data_real/{exp_name}"
data_dir = Path("rollout_data") / repo_id

config = Config()

env = make_env.create_real_env_v3(config)

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


def create_or_load_lerobot_dataset():
    """Create or load a LeRobotDataset for recording data."""

    # Define features based on the observation and action structure
    features = {
        "observation.state.cartesian": {
            "shape": (6,),
            "dtype": "float32",
            "names": ["x", "y", "z", "roll", "pitch", "yaw"],
        },
        "observation.state.joints": {
            "shape": (7,),
            "dtype": "float32",
        },
        "observation.velocity.cartesian": {
            "shape": (3,),
            "dtype": "float32",
            "names": ["vx", "vy", "vz"],
        },
        "observation.error.cartesian": {
            "shape": (3,),
            "dtype": "float32",
            "names": ["ex", "ey", "ez"],
        },
        "observation.previous.action": {
            "shape": (3,),
            "dtype": "float32",
            "names": ["prev_dx", "prev_dy", "prev_dz"],
        },
        "observation.previous.error.cartesian": {
            "shape": (3,),
            "dtype": "float32",
            "names": ["prev_ex", "prev_ey", "prev_ez"],
        },
        "action": {
            "shape": (2,),
            "dtype": "float32",
            "names": ["dx", "dy"],
        },
        "reward": {
            "shape": (1,),
            "dtype": "float32",
        },
        "is_terminal": {
            "shape": (1,),
            "dtype": "bool",
        },
        "pose_estimation.purple": {
            "shape": (4, 4),
            "dtype": "float32",
        },
        "pose_estimation.lavender": {
            "shape": (4, 4),
            "dtype": "float32",
        },
        "pose_estimation.joint_state": {
            "shape": (7,),
            "dtype": "float32",
        },
        "pose_estimation.cartesian": {
            "shape": (6,),
            "dtype": "float32",
        },
        "observation.features.wrist_camera": {
            "shape": (config.vision_head_input_dim,),
            "dtype": "float32",
        },
        "observation.formatted": {
            "shape": (config.vision_head_input_dim + config.actor_nonvision_input_dim,),
            "dtype": "float32",
        },
        # "observation.images.wrist_camera": {
        #     "shape": (480, 848, 3),
        #     "dtype": "image",
        # },
    }

    # Check if dataset already exists
    dataset_path = data_dir / "meta" / "info.json"
    if dataset_path.exists():
        print(f"Loading existing dataset from {data_dir}")
        dataset = LeRobotDataset(repo_id=repo_id, root=data_dir)
    else:
        print(f"Creating new dataset at {data_dir}")
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=config.control_frequency,
            features=features,
            robot_type="franka",
            root=data_dir,
            use_videos=True,
            image_writer_threads=4,
            image_writer_processes=1,
        )

    return dataset


# Create or load LeRobotDataset
dataset = create_or_load_lerobot_dataset()
save_thread = None


n_success = 0
total_successful_length = 0
i = 0
try:
    while i < N_ROLLOUTS:
        # new reset wrapper: automatically place brick back and pick it up again
        obs, reset_info = env.reset()

        # If the previous episode is still being saved, wait now (after reset)
        if save_thread is not None:
            save_thread.join()
            save_thread = None

        # compute goal position
        actual_grasp_pos = reset_info["reset.grasped.position"]
        goal_pos = config.goal_position_ground_truth[:2].copy()
        goal_pos[0] += actual_grasp_pos[0] - config.grasp_position_ground_truth[0]

        all_actions = []
        all_rewards = []
        all_perfect_actions = []
        all_observations = [obs]
        all_infos = [reset_info]
        info = reset_info

        is_first = True
        episode_start_time = time.time()
        frame_index = 0

        while True:
            perfect_action = goal_pos - obs["observation.state.cartesian"][:2]
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
            all_observations.append(obs)
            all_infos.append(info)
            all_rewards.append(float(reward))

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

        all_actions.append(np.full(2, np.nan, dtype=np.float32))  # for last frame
        all_rewards = [float("nan")] + all_rewards  # shift rewards for last frame

        # Add frames to the lerobot dataset

        # Save the episode in a separate thread
        def _save_episode_async(ep_idx: int, started_at: float):
            for frame_idx in range(len(all_observations)):
                # Prepare frame data
                frame_data = {
                    "observation.state.cartesian": all_observations[frame_idx][
                        "observation.state.cartesian"
                    ].astype(np.float32),
                    "action": all_actions[frame_idx].astype(np.float32),
                    "reward": np.array([all_rewards[frame_idx]], dtype=np.float32),
                    "is_terminal": np.array(
                        [terminated and frame_idx == len(all_observations) - 1],
                        dtype=bool,
                    ),
                    "observation.state.joints": all_observations[frame_idx][
                        "observation.state.joints"
                    ].astype(np.float32),
                    "observation.velocity.cartesian": all_observations[frame_idx][
                        "observation.velocity.cartesian"
                    ].astype(np.float32),
                    "observation.error.cartesian": all_observations[frame_idx][
                        "observation.error.cartesian"
                    ].astype(np.float32),
                    "observation.previous.action": all_observations[frame_idx][
                        "observation.previous.action"
                    ].astype(np.float32),
                    "observation.previous.error.cartesian": all_observations[frame_idx][
                        "observation.previous.error.cartesian"
                    ].astype(np.float32),
                    "observation.features.wrist_camera": np.array(
                        all_observations[frame_idx]["observation.features.wrist_camera"]
                    ).astype(np.float32),
                    "observation.formatted": all_observations[frame_idx][
                        "observation.formatted"
                    ]
                    .cpu()
                    .numpy()
                    .astype(np.float32),
                    # "observation.images.wrist_camera": all_observations[frame_idx][
                    #     "observation.images.wrist_camera"
                    # ].astype(np.uint8),
                    "pose_estimation.purple": reset_info[
                        "reset.pose_estimation.purple"
                    ].astype(np.float32)
                    if "reset.pose_estimation.purple" in reset_info and frame_idx == 0
                    else np.full((4, 4), np.nan, dtype=np.float32),
                    "pose_estimation.lavender": reset_info[
                        "reset.pose_estimation.lavender"
                    ].astype(np.float32)
                    if "reset.pose_estimation.lavender" in reset_info and frame_idx == 0
                    else np.full((4, 4), np.nan, dtype=np.float32),
                    "pose_estimation.joint_state": reset_info[
                        "reset.pose_estimation.joint_state"
                    ].astype(np.float32)
                    if "reset.pose_estimation.joint_state" in reset_info
                    and frame_idx == 0
                    else np.full((7,), np.nan, dtype=np.float32),
                    "pose_estimation.cartesian": reset_info[
                        "reset.pose_estimation.cartesian"
                    ].astype(np.float32)
                    if "reset.pose_estimation.cartesian" in reset_info
                    and frame_idx == 0
                    else np.full((6,), np.nan, dtype=np.float32),
                }

                # Add frame to dataset
                dataset.add_frame(
                    frame=frame_data,
                    task="collect_data",
                    timestamp=frame_idx / config.control_frequency,
                )
            dataset.save_episode()
            elapsed = time.time() - started_at
            print(f"Saved episode {ep_idx} in {elapsed:.2f} seconds.")

        save_thread = threading.Thread(
            target=_save_episode_async, args=(i + 1, time.time())
        )
        save_thread.start()

        if terminated:
            n_success += 1
            total_successful_length += len(all_actions)

        print(
            f"Episode finished: Length = {len(all_actions)}, Success: {terminated}, "
            f"success rate = {n_success / (i + 1) * 100:.2f}%; i = {i}"
        )

        i += 1
except KeyboardInterrupt:
    print("Data collection interrupted by user.")
finally:
    if save_thread is not None:
        save_thread.join()
        save_thread = None
    # Compute and print statistics
    success_rate = n_success / N_ROLLOUTS * 100
    print(
        f"average successful length: {total_successful_length / n_success if n_success > 0 else 0:.2f}"
    )
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Dataset saved at: {data_dir}")
    dataset._wait_image_writer()
    dataset.stop_image_writer()

    print("Image writer stopped.")

    env.close()
