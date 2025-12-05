from pathlib import Path
import time
import numpy as np
import pickle
from crisp_drl.agents.rlpd.rewards import (
    xy_action_magnitude_dense_reward,
    xy_dense_place_reward,
)
from crisp_drl.envs import make
import sys

exp_name = ""
if len(sys.argv) > 1:
    exp_name = sys.argv[1]

current_date = time.strftime("%Y%m%d-%H%M%S")
data_dir = Path(f"rollout_data/collect_data/{current_date}_{exp_name}")
data_dir.mkdir(parents=True, exist_ok=True)

env = make.create_simulated_env(
    {
        "initial_keyframe": 2,
        "lego_shift_range": (-0.002, 0.002),
        "initial_position_range": (
            np.array([-1.0, -1.0, -2.0]) * 1e-3,  # np.zeros(3),
            np.array([1.0, 1.0, -1.0]) * 1e-3,  # np.zeros(3),
        ),
        "live_view": True,
    }
)


# Alternate between going to the goal position and moving randomly with probaility p
p = 0.999
max_random_action_magnitude = 0.3e-3
perfect_action_magnitude = 0.00025
N_ROLLOUTS = 100
ideal_goal_pos = np.array([0.6, 0.0])
ideal_grasp_pos = np.array([0.0, 0.0])
n_success = 0

for i in range(N_ROLLOUTS):
    # new reset wrapper: automatically place brick back and pick it up again
    obs, reset_info = env.reset()
    # compute goal position
    actual_grasp_pos = reset_info["reset.grasped.position"]
    goal_pos = ideal_goal_pos.copy()
    goal_pos[0] += actual_grasp_pos[0] - ideal_grasp_pos[0]

    all_actions = []
    all_perfect_actions = []
    all_observations = [obs.cpu().numpy()]
    all_infos = [reset_info]
    info = reset_info
    while True:
        perfect_action = (
            goal_pos - info["observation"]["observation.state.cartesian"][:2]
        )
        perfect_action = (
            perfect_action / np.linalg.norm(perfect_action) * perfect_action_magnitude
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

        if terminated or truncated:
            if terminated:
                n_success += 1
            print(
                f"Episode finished: Length = {len(all_actions)}, Success: {terminated}, success rate = {n_success / (i + 1) * 100:.2f}%"
            )

            break

    # compute rewards
    all_rewards = xy_dense_place_reward(
        all_actions,
        all_observations,
        all_infos,
        ideal_goal_pos_xy=goal_pos,
        ideal_grasp_pos_xy=ideal_grasp_pos,
        actual_grasp_pos_xy=actual_grasp_pos,
    )
    all_rewards = xy_action_magnitude_dense_reward(
        all_rewards,
        all_actions,
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

env.close()
print(f"Data collection complete. Data saved to {data_dir}")

# rename data_dir to include success rate
success_rate = n_success / N_ROLLOUTS * 100
new_data_dir = data_dir.parent / f"{data_dir.name}_{int(success_rate):02d}"
data_dir.rename(new_data_dir)
print(f"Data directory renamed to include success rate: {new_data_dir}")
