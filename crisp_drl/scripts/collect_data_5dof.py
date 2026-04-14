from pathlib import Path
import time
import numpy as np
import pickle
from crisp_drl.agents.shared.rewards import (
    sparse_event_reward,
    xy_action_magnitude_dense_reward,
    xy_dense_delta_place_reward,
    xy_dense_simple_place_reward,
)
from crisp_drl.agents.shared.algorithm_config import Config
from crisp_drl.envs import make_env, make_rew
import sys

# export MUJOCO_GL=egl
# pixi shell -e jazzy
# python crisp_drl/scripts/collect_data.py

# Alternate between going to the goal position and moving randomly with probaility p

decoupled_action = False
N_ROLLOUTS = 2000
pe_accuracy_ratio_rot_trans = np.deg2rad(1) / 0.001
max_random_action_magnitude = 0.00025
max_random_action_magnitude_rotation = (
    max_random_action_magnitude * pe_accuracy_ratio_rot_trans
)
perfect_action_magnitude = 0.00025
perfect_action_magnitude_rot = perfect_action_magnitude * pe_accuracy_ratio_rot_trans
ideal_goal_pos = np.array([0.6, 0.0])
ideal_goal_orientation = np.zeros(3)
ideal_grasp_pos = np.array([0.0, 0.0])

base_exp_name = ""
if len(sys.argv) > 1:
    pe_accuracy = float(sys.argv[1])
else:
    pe_accuracy = 0.0015

if len(sys.argv) > 2:
    episode_length = int(sys.argv[2])
else:
    episode_length = 150

if len(sys.argv) > 3:
    p = float(sys.argv[3])
else:
    p = 0.8

if len(sys.argv) > 4:
    n = int(sys.argv[4])
else:
    n = 3

ps = [p for _ in range(n)]
config = Config(episode_length=episode_length)
env = make_env.create_simulated_env_5dof(
    {
        "initial_keyframe": 2,
        "live_view": False,
    },
    pe_accuracy=pe_accuracy,
    pe_accuracy_angular=pe_accuracy * pe_accuracy_ratio_rot_trans,
    sac_config=config,
)


reward_fn = make_rew.create_sim_reward_fn(
    config,
    ideal_goal_pos_xy=np.array([0.6, 0.0]),
    ideal_grasp_pos_xy=np.array([0.0, 0.0]),
    event_reward_map={
        "E_SUCCESS": 0.1 / (1 - config.gamma) * 3,
        "E_FAIL": -0.1 / (1 - config.gamma) / 2 * 3,
        "E_SAFETY_BOX_VIOLATION": 0.0,  # -0.05,
    },
)


for p in ps:
    print(
        f"Collecting data with p={p} (perfect action probability), pe_accuracy={pe_accuracy}, episode_length={episode_length}..."
    )
    if base_exp_name == "":
        exp_name = str(p)
    else:
        exp_name = f"{base_exp_name}_{p}"

    current_date = time.strftime("%Y%m%d-%H%M%S")
    data_dir = Path(f"rollout_data/collect_data/{current_date}_{exp_name}")
    data_dir.mkdir(parents=True, exist_ok=True)
    n_success = 0
    total_successful_length = 0
    for i in range(N_ROLLOUTS):
        # new reset wrapper: automatically place brick back and pick it up again
        obs, reset_info = env.reset()
        # compute goal position
        actual_grasp_pos = reset_info["reset.grasped.position"]
        goal_pos = ideal_goal_pos.copy()
        goal_pos[0] -= actual_grasp_pos[0]

        all_actions = []
        all_rewards = []
        all_perfect_actions = []
        all_observations = [obs["observation.formatted"].cpu().numpy()]
        all_infos = [reset_info]
        info = reset_info
        while True:
            perfect_action_xy = goal_pos - obs["observation.state.cartesian"][:2]
            perfect_action_xy = (
                perfect_action_xy
                / np.linalg.norm(perfect_action_xy)
                * perfect_action_magnitude
            )
            perfect_action_rotation = -obs["observation.state.cartesian"][3:6]
            perfect_action_rotation = (
                perfect_action_rotation
                / np.linalg.norm(perfect_action_rotation)
                * perfect_action_magnitude_rot
                if np.linalg.norm(perfect_action_rotation) > 1e-6
                else np.zeros(3)
            )
            perfect_action = np.array(
                [
                    perfect_action_xy[0],
                    perfect_action_xy[1],
                    perfect_action_rotation[0],
                    perfect_action_rotation[1],
                    perfect_action_rotation[2],
                ]
            )
            all_perfect_actions.append(perfect_action)

            if np.random.rand() < p:
                action_xy = np.random.uniform(
                    -max_random_action_magnitude,
                    max_random_action_magnitude,
                    size=(2,),
                )
                action_rotation = np.random.uniform(
                    -max_random_action_magnitude_rotation,
                    max_random_action_magnitude_rotation,
                    size=(3,),
                )
                action = np.array(
                    [
                        action_xy[0],
                        action_xy[1],
                        action_rotation[0],
                        action_rotation[1],
                        action_rotation[2],
                    ]
                )
            else:
                action = perfect_action
            all_actions.append(action.copy())

            obs, reward, terminated, truncated, info = env.step(action)
            all_observations.append(obs["observation.formatted"].cpu().numpy())
            all_infos.append(info)
            all_rewards.append(reward)

            if terminated or truncated:
                last_custom_events = list(
                    map(lambda x: x[1], all_infos[-1].get("custom_events", []))
                )
                successful = "E_SUCCESS" in last_custom_events
                if successful:
                    n_success += 1
                    total_successful_length += len(all_actions)
                print(
                    f"[{int((i + 1) / N_ROLLOUTS * 100)}%] Episode finished: Length = {len(all_actions)}, Success: {successful}, success rate = {n_success / (i + 1) * 100:.2f}%"
                )

                break

        # compute rewards
        all_actions, all_observations, all_rewards, all_infos = reward_fn(
            all_actions,
            all_observations,
            all_rewards,
            all_infos,
            actual_grasp_pos_xy=actual_grasp_pos[:2],
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
        with open(data_dir / f"rollout_{i:03d}_info.pkl", "wb") as f:
            pickle.dump(all_infos, f)

    # rename data_dir to include success rate
    success_rate = n_success / N_ROLLOUTS * 100
    new_data_dir = data_dir.parent / f"{data_dir.name}_{int(success_rate):02d}"
    data_dir.rename(new_data_dir)
    print(
        f"average successful length: {total_successful_length / n_success if n_success > 0 else 0:.2f}"
    )
    print(f"Data directory renamed to include success rate: {new_data_dir}")

env.close()
