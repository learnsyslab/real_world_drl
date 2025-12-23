import pickle
import os
import sys

import numpy as np
from crisp_drl.agents.shared.rewards import (
    dense_place_reward,
    xy_dense_delta_place_reward,
    xy_dense_simple_place_reward,
)
from crisp_drl.data.buffers_cleanrl import ReplayBuffer, ReplayBufferGpu
from gymnasium import spaces


def main():
    rollout_directory = sys.argv[1]
    n_cameras = 2
    all_actions = []
    all_observations = []
    all_rewards = []
    total_len = 0
    for file in os.listdir(os.path.join("rollouts", rollout_directory)):
        if not file.endswith(".pkl"):
            continue
        with open(os.path.join("rollouts", rollout_directory, file), "rb") as f:
            data = pickle.load(f)

        actions = data["actions"]
        acts = []
        for action in actions:
            # action = np.concatenate((action[:3],action[6:7]))
            action = action[:2]
            acts.append(action)
        actions = acts

        observations = data["observations"]
        rewards = data["rewards"]

        infos = data["infos"]
        actual_grasp_pos = data["reset_info"]["reset.grasped.position"]
        rewards = xy_dense_delta_place_reward(
            actions,
            observations,
            rewards,
            [data["reset_info"]] + infos,
            max_rew=0.01,
            max_action_magnitude=0.0008,
            ideal_goal_pos_xy=np.array([0.53975, -0.033]),
            ideal_grasp_pos_xy=np.array([0.58833, -0.13817]),
            actual_grasp_pos_xy=actual_grasp_pos[:2],
        )

        all_actions.append(actions)
        all_observations.append(observations)
        all_rewards.append(rewards)
        total_len += len(actions)

    rb = ReplayBufferGpu(
        total_len,
        all_observations[0][0].shape[0],
        spaces.Box(-np.inf, np.inf, shape=all_actions[0][0].shape, dtype=np.float32),
    )

    print("Converting dataset to replay buffer...")
    for i in range(len(all_actions)):
        rb.add_rollout(
            np.array(all_observations[i]),
            np.array(all_actions[i]),
            np.array(all_rewards[i]),
            True,  # all trajectories are successful therefore terminated
        )
    rb.save_buffer(os.path.join("expert_buffers", f"{rollout_directory}.joblib"))
    print("Done.")


if __name__ == "__main__":
    main()
