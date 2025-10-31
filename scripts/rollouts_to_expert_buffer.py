
import pickle
import os
import sys

import numpy as np
from crisp_drl.agents.rlpd.rewards import dense_place_reward
from crisp_drl.data.buffers_cleanrl import ReplayBuffer
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
            action = action[:3]
            acts.append(action)
        actions = acts

        observations = data["observations"]

        infos = data["infos"]
        actual_grasp_pos = data["actual_grasped_pos"]
        rewards = dense_place_reward(actions, observations, infos, {"E_TORQUE": -10.0, "E_FAR_AWAY": -3.0, "E_SUCCESS": 20.0, "E_FAIL": -1.0, "E_BAD_BEHAVIOR": -5.0}, 
                                     max_rew=0.01, ideal_goal_pos=[0.53975, -0.033,  0.05], ideal_grasp_pos=np.array([0.58833, -0.13817,  0.04229]), actual_grasp_pos=actual_grasp_pos, k_xy=0.005, k_z=0.002)

        all_actions.append(actions)
        all_observations.append(observations)
        all_rewards.append(rewards)
        total_len += len(actions)    


    rb = ReplayBuffer(
        total_len,
        spaces.Box(-np.inf, np.inf, shape=all_observations[0][0].shape, dtype=np.float32),
        [None for _ in range(n_cameras)],
        spaces.Box(-np.inf, np.inf, shape=all_actions[0][0].shape, dtype=np.float32),
    )

    print("Converting dataset to replay buffer...")
    for i in range(len(all_actions)):
        rb.add_rollout(
            np.array(all_observations[i]),
            np.array(all_actions[i]),
            np.array(all_rewards[i]),
            True, # all trajectories are successful therefore terminated
        )
    rb.save_buffer(os.path.join("expert_buffers", f"{rollout_directory}.joblib"))
    print("Done.")

if __name__ == "__main__":
    main()