"""Real-world 3-DoF + Rot-Z data collector (LEGO insertion).

Mirrors ``scripts/collect_data_real_lerobot.py`` but emits 3-D agent actions
``[dx, dy, drz]`` for ``InsertionWrapper3DoFRotZ``.

Angle convention: ``observation.state.cartesian[3:6]`` is rotvec
(axis*angle, rad). With rx≈ry≈0 at home, ``cartesian[5]`` is pure yaw.
The ideal goal yaw is 0 (brick fixtured square with the world).
"""

from pathlib import Path
import threading
import time
from datetime import datetime
import numpy as np

from crisp_drl.agents.shared.algorithm_config import Config
from crisp_drl.envs import make_env, make_rew

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# Alternate between perfect goal-seeking action (with prob 1-p) and uniform random (p)
p = 0.70 # 0.70 to start with
# p = 0.0 # perfect action test
N_ROLLOUTS = 30 # 10 to start with

# Translation magnitudes
max_random_action_magnitude = 0.25e-3   # m / step
perfect_action_magnitude = 0.25e-3      # m / step

# Rotation magnitudes — scale rotation step relative to translation by the
# pose-estimation-accuracy ratio (matches sim collector); 2 deg ↔ 1 mm.
pe_accuracy_ratio_rot_trans = np.deg2rad(2) / 0.001
max_random_action_magnitude_rz = max_random_action_magnitude * pe_accuracy_ratio_rot_trans
perfect_action_magnitude_rz = perfect_action_magnitude * pe_accuracy_ratio_rot_trans

ideal_goal_pos_rz = 0.0  # absolute world yaw target

base_exp_name = "run_4_3dof_rz"
start_time_tag = datetime.now().strftime("%m%d_%H_%M")
exp_name = f"{base_exp_name}_{p}_{start_time_tag}"

# Continue from folder, eg. "run_4_3dof_rz_0.7_0501_10_57"
exp_name = "run_4_3dof_rz_0.7_0501_11_48"

repo_id = f"collect_data_real/{exp_name}"
data_dir = Path("rollout_data") / repo_id

config = Config()
# Override dims for 3-DoF + RotZ before instantiating env.
config.actor_output_dim = 3
config.actor_nonvision_input_dim = 18
config.max_action = np.array(
    [
        0.00025,
        0.00025,
        np.deg2rad(0.5),
    ]
)

env = make_env.create_real_env_v4_3dof_rz(config)

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


def create_or_load_lerobot_dataset():
    features = {
        "observation.state.cartesian": {
            "shape": (6,),
            "dtype": "float32",
            "names": ["x", "y", "z", "rvx", "rvy", "rvz"],
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
        "observation.velocity.angular": {
            "shape": (3,),
            "dtype": "float32",
            "names": ["wx", "wy", "wz"],
        },
        "observation.error.cartesian": {
            "shape": (3,),
            "dtype": "float32",
            "names": ["ex", "ey", "ez"],
        },
        "observation.error.angular": {
            "shape": (3,),
            "dtype": "float32",
            "names": ["erx", "ery", "erz"],
        },
        "observation.previous.action": {
            "shape": (6,),
            "dtype": "float32",
            "names": ["prev_dx", "prev_dy", "prev_dz", "prev_drx", "prev_dry", "prev_drz"],
        },
        "observation.previous.error.cartesian": {
            "shape": (3,),
            "dtype": "float32",
            "names": ["prev_ex", "prev_ey", "prev_ez"],
        },
        "observation.previous.error.angular": {
            "shape": (3,),
            "dtype": "float32",
            "names": ["prev_erx", "prev_ery", "prev_erz"],
        },
        "observation.state.sensors_bota_ft_sensor": {
            "shape": (6,),
            "dtype": "float32",
            "names": ["fx", "fy", "fz", "mx", "my", "mz"],
        },
        "action": {
            "shape": (3,),
            "dtype": "float32",
            "names": ["dx", "dy", "drz"],
        },
        "perfect_action": {
            "shape": (3,),
            "dtype": "float32",
            "names": ["dx", "dy", "drz"],
        },
        "grasp_delta": {
            "shape": (3,),
            "dtype": "float32",
            "names": ["grasp_dx", "grasp_dy", "grasp_dz"],
        },
        "goal_position_delta": {
            "shape": (2,),
            "dtype": "float32",
            "names": ["goal_dx", "goal_dy"],
        },
        "goal_rotation_z": {
            "shape": (1,),
            "dtype": "float32",
        },
        "start_rotation_z": {
            "shape": (1,),
            "dtype": "float32",
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
        # "observation.images.wrist_camera": {
        #     "shape": (480, 848, 3),
        #     "dtype": "image",
        # },
        "observation.formatted": {
            "shape": (config.vision_head_input_dim + config.actor_nonvision_input_dim,),
            "dtype": "float32",
        },
    }

    dataset_path = data_dir / "meta" / "tasks.jsonl"
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


def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


n_success = 0
total_successful_length = 0
i = 0
try:
    while i < N_ROLLOUTS:
        obs, reset_info = env.reset()

        if save_thread is not None:
            save_thread.join()
            save_thread = None

        # XY goal: ground-truth goal + grasp delta (matches v4 reset semantics)
        grasp_delta = reset_info["reset.grasped.delta_estimated"]
        goal_pos = config.goal_position_ground_truth[:2].copy()
        goal_pos[0] += grasp_delta[0]
        goal_position_delta = reset_info["reset.goal_position.offset"][:2]

        # Wrapper-side noisy goal yaw (used for safety-box centering inside env)
        goal_rotation_z = float(reset_info["reset.goal_orientation.rotation_z"])
        start_rotation_z = float(reset_info.get("reset.start_orientation.rotation_z", 0.0))
        # Perfect-action target — drive to the IDEAL physical yaw (=0), mirroring
        # how XY perfect uses goal_position_ground_truth (not the randomised wrapper goal).
        perfect_target_rz = ideal_goal_pos_rz

        all_actions = []
        all_rewards = []
        all_perfect_actions = []
        all_observations = [obs]
        all_infos = [reset_info]
        info = reset_info

        while True:
            cartesian = obs["observation.state.cartesian"]

            # Translation perfect action
            pa_xy = goal_pos - cartesian[:2]
            norm_xy = float(np.linalg.norm(pa_xy))
            if norm_xy > 1e-9:
                pa_xy = pa_xy / norm_xy * perfect_action_magnitude
            else:
                pa_xy = np.zeros(2)

            # Yaw perfect action — wrap [-pi, pi] then clip magnitude
            pa_rz = _wrap_pi(perfect_target_rz - float(cartesian[5]))
            if abs(pa_rz) > 1e-6:
                pa_rz = float(np.sign(pa_rz)) * min(abs(pa_rz), perfect_action_magnitude_rz)
            else:
                pa_rz = 0.0

            perfect_action = np.array([pa_xy[0], pa_xy[1], pa_rz], dtype=np.float32)
            all_perfect_actions.append(perfect_action)

            if np.random.rand() < p:
                action = np.array(
                    [
                        np.random.uniform(
                            -max_random_action_magnitude, max_random_action_magnitude
                        ),
                        np.random.uniform(
                            -max_random_action_magnitude, max_random_action_magnitude
                        ),
                        np.random.uniform(
                            -max_random_action_magnitude_rz,
                            max_random_action_magnitude_rz,
                        ),
                    ],
                    dtype=np.float32,
                )
            else:
                action = perfect_action.copy()
            all_actions.append(action.copy())

            obs, reward, terminated, truncated, info = env.step(action)
            all_observations.append(obs)
            all_infos.append(info)
            all_rewards.append(float(reward))

            if terminated or truncated:
                break

        # Final perfect action for the terminal frame (mirrors 2-DoF script)
        cartesian = obs["observation.state.cartesian"]
        pa_xy = goal_pos - cartesian[:2]
        norm_xy = float(np.linalg.norm(pa_xy))
        if norm_xy > 1e-9:
            pa_xy = pa_xy / norm_xy * perfect_action_magnitude
        else:
            pa_xy = np.zeros(2)
        pa_rz = _wrap_pi(perfect_target_rz - float(cartesian[5]))
        if abs(pa_rz) > 1e-6:
            pa_rz = float(np.sign(pa_rz)) * min(abs(pa_rz), perfect_action_magnitude_rz)
        else:
            pa_rz = 0.0
        all_perfect_actions.append(np.array([pa_xy[0], pa_xy[1], pa_rz], dtype=np.float32))

        # Reward labelling (sparse events -> per-step rewards)
        all_actions, all_observations, all_rewards, all_infos = reward_fn(
            all_actions, all_observations, all_rewards, all_infos,
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

        # Pad action and reward for the terminal frame (no-op action / nan reward).
        all_actions.append(np.full(3, np.nan, dtype=np.float32))
        all_rewards = [float("nan")] + all_rewards

        def _save_episode_async(ep_idx: int, started_at: float):
            for frame_idx in range(len(all_observations)):
                obs_f = all_observations[frame_idx]
                frame_data = {
                    "observation.state.cartesian": obs_f["observation.state.cartesian"].astype(np.float32),
                    "observation.state.joints": obs_f["observation.state.joints"].astype(np.float32),
                    "observation.velocity.cartesian": obs_f["observation.velocity.cartesian"].astype(np.float32),
                    "observation.velocity.angular": obs_f["observation.velocity.angular"].astype(np.float32),
                    "observation.error.cartesian": obs_f["observation.error.cartesian"].astype(np.float32),
                    "observation.error.angular": obs_f["observation.error.angular"].astype(np.float32),
                    "observation.previous.action": obs_f["observation.previous.action"].astype(np.float32),
                    "observation.previous.error.cartesian": obs_f["observation.previous.error.cartesian"].astype(np.float32),
                    "observation.previous.error.angular": obs_f["observation.previous.error.angular"].astype(np.float32),
                    "observation.state.sensors_bota_ft_sensor": obs_f["observation.state.sensors_bota_ft_sensor"].astype(np.float32),
                    "observation.features.wrist_camera": np.array(obs_f["observation.features.wrist_camera"]).astype(np.float32),
                    "observation.formatted": obs_f["observation.formatted"].cpu().numpy().astype(np.float32),
                    "action": all_actions[frame_idx].astype(np.float32),
                    "perfect_action": all_perfect_actions[frame_idx].astype(np.float32),
                    "reward": np.array([all_rewards[frame_idx]], dtype=np.float32),
                    "success": np.array(
                        [successful and frame_idx == len(all_observations) - 1], dtype=bool,
                    ),
                    "is_terminal": np.array(
                        [terminated and frame_idx == len(all_observations) - 1], dtype=bool,
                    ),
                    "grasp_delta": grasp_delta.astype(np.float32),
                    "goal_position_delta": goal_position_delta.astype(np.float32),
                    "goal_rotation_z": np.array([goal_rotation_z], dtype=np.float32),
                    "start_rotation_z": np.array([start_rotation_z], dtype=np.float32),
                    # "observation.images.wrist_camera": obs_f["observation.images.wrist_camera"].astype(np.uint8),
                }
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

        if successful:
            n_success += 1
            total_successful_length += len(all_actions)

        print(
            f"Episode finished: Length = {len(all_actions)}, Success: {successful}, "
            f"goal_rz = {np.rad2deg(goal_rotation_z):.2f} deg, "
            f"start_rz = {np.rad2deg(start_rotation_z):.2f} deg, "
            f"success rate = {n_success / (i + 1) * 100:.2f}%; i = {i}"
        )

        i += 1
except KeyboardInterrupt:
    print("Data collection interrupted by user.")
finally:
    if save_thread is not None:
        save_thread.join()
        save_thread = None
    success_rate = n_success / (i + 1) * 100 if i >= 0 else 0.0
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
