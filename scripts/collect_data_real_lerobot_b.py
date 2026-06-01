from pathlib import Path
import threading
import time

import numpy as np

from crisp_drl.agents.shared.algorithm_config import Config
from crisp_drl.agents.shared.insertion_env_config import ShelfBoxConfig
from crisp_drl.envs import make_env, make_rew

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# Alternate between going to the goal position and moving randomly with probability p.
p = 0.82
N_ROLLOUTS = 2
max_random_action_magnitude = 0.25e-3
perfect_action_magnitude = 0.25e-3
GRASP_DELTA_ROTATION_DEG = 30.0
INSERTION_AXIS = np.array([-0.5, 0.0, -0.7071], dtype=float)

base_exp_name = "run_b_2605_19_25"
exp_name = f"{base_exp_name}_{p}"

repo_id = f"collect_data_real/{exp_name}"
data_dir = Path("rollout_data") / repo_id

config = Config()
env_config = ShelfBoxConfig()

env = make_env.create_real_env_b1(config, env_config)

reward_fn = make_rew.create_real_reward_fn(
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


def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


def orthogonal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array([0.0, 1.0, 0.0], dtype=float)
    if abs(float(np.dot(axis, reference))) > 0.95:
        reference = np.array([1.0, 0.0, 0.0], dtype=float)
    first = normalize(np.cross(axis, reference))
    second = normalize(np.cross(axis, first))
    return first, second


def rotate_about_y(vec: np.ndarray, angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    rotation = np.array(
        [
            [cos_angle, 0.0, sin_angle],
            [0.0, 1.0, 0.0],
            [-sin_angle, 0.0, cos_angle],
        ],
        dtype=float,
    )
    return rotation @ vec


def world_delta_to_plane_action(delta_world: np.ndarray) -> np.ndarray:
    axis = normalize(INSERTION_AXIS)
    plane_axis_1, plane_axis_2 = orthogonal_basis(axis)
    plane_delta = delta_world - np.dot(delta_world, axis) * axis
    return np.array(
        [
            float(np.dot(plane_delta, plane_axis_1)),
            float(np.dot(plane_delta, plane_axis_2)),
        ],
        dtype=float,
    )


def scale_to_magnitude(vector: np.ndarray, magnitude: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return np.zeros_like(vector)
    return min(1.0, magnitude / norm) * vector


def create_or_load_lerobot_dataset():
    """Create or load a LeRobotDataset for recording box-task rollouts."""

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
        "observation.rotated.previous.action": {
            "shape": (6,),
            "dtype": "float32",
            "names": [
                "prev_dx",
                "prev_dy",
                "prev_dz",
                "prev_droll",
                "prev_dpitch",
                "prev_dyaw",
            ],
        },
        "observation.rotated.previous.error.cartesian": {
            "shape": (3,),
            "dtype": "float32",
            "names": ["prev_ex", "prev_ey", "prev_ez"],
        },
        "observation.rotated.velocity.cartesian": {
            "shape": (3,),
            "dtype": "float32",
            "names": ["vx", "vy", "vz"],
        },
        "observation.rotated.error.cartesian": {
            "shape": (3,),
            "dtype": "float32",
            "names": ["ex", "ey", "ez"],
        },
        "observation.state.sensors_bota_ft_sensor": {
            "shape": (6,),
            "dtype": "float32",
            "names": ["fx", "fy", "fz", "mx", "my", "mz"],
        },
        "action": {
            "shape": (2,),
            "dtype": "float32",
            "names": ["dy", "dz"],
        },
        "perfect_action": {
            "shape": (2,),
            "dtype": "float32",
            "names": ["dy", "dz"],
        },
        "grasp_delta": {
            "shape": (3,),
            "dtype": "float32",
            "names": ["grasp_dx", "grasp_dy", "grasp_dz"],
        },
        "goal_position_delta": {
            "shape": (3,),
            "dtype": "float32",
            "names": ["goal_dx", "goal_dy", "goal_dz"],
        },
        "reward": {
            "shape": (1,),
            "dtype": "float32",
        },
        "success": {
            "shape": (1,),
            "dtype": "bool",
        },
        "is_terminal": {
            "shape": (1,),
            "dtype": "bool",
        },
        "observation.features.wrist_camera": {
            "shape": (config.vision_head_input_dim,),
            "dtype": "float32",
        },
        "observation.formatted": {
            "shape": (
                config.vision_head_input_dim * env_config.n_cameras
                + config.actor_nonvision_input_dim,
            ),
            "dtype": "float32",
        },
    }

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
        )

    return dataset


dataset = create_or_load_lerobot_dataset()
save_thread = None

n_success = 0
total_successful_length = 0
i = 0
try:
    while i < N_ROLLOUTS:
        obs, reset_info = env.reset()

        if save_thread is not None:
            save_thread.join()
            save_thread = None

        grasp_delta = np.asarray(reset_info["reset.grasped.delta"], dtype=float)
        rotated_grasp_delta = rotate_about_y(grasp_delta, GRASP_DELTA_ROTATION_DEG)
        goal_position_delta = np.asarray(
            reset_info["reset.goal_position.offset"], dtype=float
        )
        shifted_goal_pos = env_config.goal_position_ground_truth + rotated_grasp_delta

        all_actions = []
        all_rewards = []
        all_perfect_actions = []
        all_observations = [obs]
        all_infos = [reset_info]

        while True:
            current_pos = np.asarray(
                obs["observation.state.cartesian"][:3], dtype=float
            )
            perfect_action_world = shifted_goal_pos - current_pos
            perfect_action_plane = world_delta_to_plane_action(perfect_action_world)
            perfect_action_plane = scale_to_magnitude(
                perfect_action_plane, perfect_action_magnitude
            )
            all_perfect_actions.append(perfect_action_plane)

            if np.random.rand() < p:
                action = np.random.uniform(
                    -max_random_action_magnitude,
                    max_random_action_magnitude,
                    size=(2,),
                )
            else:
                action = perfect_action_plane
            all_actions.append(action.copy())

            obs, reward, terminated, truncated, info = env.step(action)
            all_observations.append(obs)
            all_infos.append(info)
            all_rewards.append(float(reward))

            if terminated or truncated:
                break

        current_pos = np.asarray(obs["observation.state.cartesian"][:3], dtype=float)
        perfect_action_world = shifted_goal_pos - current_pos
        perfect_action_plane = world_delta_to_plane_action(perfect_action_world)
        perfect_action_plane = scale_to_magnitude(
            perfect_action_plane, perfect_action_magnitude
        )
        all_perfect_actions.append(perfect_action_plane)

        all_actions, all_observations, all_rewards, all_infos = reward_fn(
            all_actions,
            all_observations,
            all_rewards,
            all_infos,
        )

        last_custom_events = list(
            map(lambda x: x[1], all_infos[-1].get("custom_events", []))
        )
        if (
            "E_ROLLOUT_UNUSABLE" in last_custom_events
            or "E_CONTROLLER_ISSUE" in last_custom_events
            or "E_TORQUE" in last_custom_events
        ):
            print("Rollout unusable, skipping saving this rollout.")
            input("Press Enter to continue...")
            continue

        successful = "E_SUCCESS" in last_custom_events

        all_actions.append(np.full(2, np.nan, dtype=np.float32))
        all_rewards = [float("nan")] + all_rewards

        def _save_episode_async(ep_idx: int, started_at: float):
            for frame_idx in range(len(all_observations)):
                frame_data = {
                    "observation.state.cartesian": all_observations[frame_idx][
                        "observation.state.cartesian"
                    ].astype(np.float32),
                    "observation.state.joints": all_observations[frame_idx][
                        "observation.state.joints"
                    ].astype(np.float32),
                    "observation.rotated.previous.action": np.asarray(
                        all_observations[frame_idx][
                            "observation.rotated.previous.action"
                        ]
                    ).astype(np.float32),
                    "observation.rotated.previous.error.cartesian": np.asarray(
                        all_observations[frame_idx][
                            "observation.rotated.previous.error.cartesian"
                        ]
                    ).astype(np.float32),
                    "observation.rotated.velocity.cartesian": np.asarray(
                        all_observations[frame_idx][
                            "observation.rotated.velocity.cartesian"
                        ]
                    ).astype(np.float32),
                    "observation.rotated.error.cartesian": np.asarray(
                        all_observations[frame_idx][
                            "observation.rotated.error.cartesian"
                        ]
                    ).astype(np.float32),
                    "observation.state.sensors_bota_ft_sensor": all_observations[
                        frame_idx
                    ]["observation.state.sensors_bota_ft_sensor"].astype(np.float32),
                    "action": all_actions[frame_idx].astype(np.float32),
                    "perfect_action": all_perfect_actions[frame_idx].astype(np.float32),
                    "grasp_delta": grasp_delta.astype(np.float32),
                    "goal_position_delta": goal_position_delta.astype(np.float32),
                    "reward": np.array([all_rewards[frame_idx]], dtype=np.float32),
                    "success": np.array(
                        [successful and frame_idx == len(all_observations) - 1],
                        dtype=bool,
                    ),
                    "is_terminal": np.array(
                        [terminated and frame_idx == len(all_observations) - 1],
                        dtype=bool,
                    ),
                    "observation.features.wrist_camera": np.array(
                        all_observations[frame_idx]["observation.features.wrist_camera"]
                    ).astype(np.float32),
                    "observation.formatted": all_observations[frame_idx][
                        "observation.formatted"
                    ]
                    .cpu()
                    .numpy()
                    .astype(np.float32),
                }

                dataset.add_frame(
                    frame=frame_data,
                    task="collect_data_box",
                    timestamp=frame_idx / config.control_frequency,
                )
            dataset.save_episode()
            elapsed = time.time() - started_at
            print(f"Saved episode {ep_idx} in {elapsed:.2f} seconds.")

        save_thread = threading.Thread(
            target=_save_episode_async, args=(i + 1, time.time())
        )
        save_thread.start()

        if successful:
            n_success += 1
            total_successful_length += len(all_actions)

        print(
            f"Episode finished: Length = {len(all_actions)}, Success: {successful}, "
            f"success rate = {n_success / (i + 1) * 100:.2f}%; i = {i}"
        )

        i += 1
except KeyboardInterrupt:
    print("Data collection interrupted by user.")
finally:
    if save_thread is not None:
        save_thread.join()
        save_thread = None

    success_rate = n_success / i * 100 if i > 0 else 0.0
    print(
        f"average successful length: {total_successful_length / n_success if n_success > 0 else 0:.2f}"
    )
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Dataset saved at: {data_dir}")
    dataset._wait_image_writer()
    dataset.stop_image_writer()

    print("Image writer stopped.")

    env.reset(options={"last_reset": True})
    env.close()
