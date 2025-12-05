base_folder = "/home/linus/uni/master/code/real_world_drl/rollout_data/collect_data"

import os
import numpy as np
from pathlib import Path
import pickle
from gymnasium import spaces
import torch
from crisp_drl.data.buffers_cleanrl import ReplayBufferGpuWithPerfectActions

# Combine per-subfolder data and cross-folder into single rollout buffers with additional field perfect_action

exp_folders_to_skip = {
    "20251204-095036_0.95_45",
    "20251204-095627_0.33_99",
    "20251204-100213_0.999_26",
    "20251204-101910_0.9_46",
    "20251204-102434_0.8_92",
}

# 0) Load all data from subfolders
all_run_data = {}
for exp_folder in os.listdir(base_folder):
    exp_path = Path(base_folder) / exp_folder
    if not exp_path.is_dir() or exp_folder in exp_folders_to_skip:
        continue

    all_run_data[exp_folder] = []
    for run_folder in os.listdir(exp_path):
        run_file = exp_path / run_folder
        if not run_file.is_file() or run_folder.endswith("_info.pkl"):
            continue

        run_data = np.load(run_file, allow_pickle=True)
        all_run_data[exp_folder].append(run_data)

    # separate image encoders from buffer and move into actor?
    #  => decide in actor whether to train image encoder or not
    # online VAE training? -> probably does not do much, no fundamentally new data
    #  => maybe continue training with policy gradient, similar to no pre-training
    # Pipeline:
    #    [1) VAE vs no VAE]
    #     2) Actor+Critic [image encoder continue-training vs fixed]
    #     3) RL [image encoder continue-training vs fixed]

# 1) compute buffer sizes
buffer_sizes = {}
for exp_folder, run_datas in all_run_data.items():
    total_size = 0
    for run_data in run_datas:
        total_size += len(run_data["observations"])
    buffer_sizes[exp_folder] = total_size
global_buffer_size = sum(buffer_sizes.values())

# 2) create buffers
buffers = {}
for exp_folder, size in buffer_sizes.items():
    buffers[exp_folder] = ReplayBufferGpuWithPerfectActions(
        action_space=spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
        n_step_return=1,
        gamma=0.99,
        device="cuda",
        buffer_size=size,
        observation_dim=384 + 11,
    )
global_buffer = ReplayBufferGpuWithPerfectActions(
    action_space=spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
    n_step_return=1,
    gamma=0.99,
    device="cuda",
    buffer_size=global_buffer_size,
    observation_dim=384 + 11,
)

# 3) fill buffers
for exp_folder, run_datas in all_run_data.items():
    for run_data in run_datas:
        all_observations = run_data["observations"]
        all_actions = run_data["actions"]
        all_rewards = run_data["rewards"]
        all_perfect_actions = run_data["perfect_actions"]
        terminated = bool(run_data["terminated"])

        obs = list(map(torch.from_numpy, all_observations))
        act = list(map(torch.from_numpy, all_actions))
        perf_act = list(map(torch.from_numpy, all_perfect_actions))

        global_buffer.add_rollout(
            obs=obs,
            action=act,
            reward=all_rewards,
            perfect_action=perf_act,
            terminated=terminated,
        )
        buffers[exp_folder].add_rollout(
            obs=obs,
            action=act,
            reward=all_rewards,
            perfect_action=perf_act,
            terminated=terminated,
        )
# 4) save buffers
for exp_folder, buffer in buffers.items():
    save_path = Path(base_folder) / exp_folder
    buffer.save_buffer(save_path)
global_save_path = Path(base_folder)
global_buffer.save_buffer(global_save_path)
