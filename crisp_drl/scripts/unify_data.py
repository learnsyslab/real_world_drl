import os
import numpy as np
from pathlib import Path
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
    "20251205-083116_0.33_100",
    "20251205-084554_0.8_92",
    "20251205-085351_0.9_56",
    "20251205-090548_0.95_39",
    "20251205-091208_0.999_24",
    "20251212-092812_0.999_18",
    "20251212-093216_0.33_100",
    "20251212-095949_0.33_100",
    "20251212-100257_0.999_22",
    "20251212-115108_0.999_23",
    "20251212-115351_0.33_100",
    "20251212-130027_0.33_100",  # sparse + delta place + action magnitude
    "20251212-130201_0.999_24",  # sparse + delta place + action magnitude
    "20251215-105714_0.999_30",  # sparse only
    "20251215-110132_0.33_100",  # sparse only
    "20251215-112134_0.33_100",  # sparse + delta place
    "20251215-112238_0.999_17",  # sparse + delta place
    "20251215-120456_0.999_15",  # sparse + delta place + action magnitude
    "20251215-120723_0.33_100",  # sparse + delta place + action magnitude
    "20251215-134429_0.33_99",  # sparse + delta simple + action magnitude
    "20251215-134537_0.999_24",  # sparse + delta simple + action magnitude
    "20251215-144507_0.33_100",  # sparse only
    "20251215-144631_0.999_28",  # sparse only
    "20251215-151319_0.5_100",  # sparse only
    "20251215-152854_0.999_23",  # sparse only
    "20251215-161436_0.999_18",
    "20251215-161550_0.9_48",
    "20251215-161657_0.8_94",
    "20251215-161813_0.33_100",
    # start of 25-batch runs
    "20251222-121602_0.33_100",
    "20251222-121615_0.8_92",
    "20251222-121639_0.9_60",
    "20251222-121709_0.999_12",
    "20251222-133328_0.33_100",
    "20251222-133341_0.8_88",
    "20251222-133401_0.9_68",
    "20251222-133425_0.999_16",
    "20251222-145841_0.33_100",
    "20251222-145854_0.33_100",
    "20251222-145907_0.999_24",
    "20251222-145938_0.999_16",
    # batch 100 runs
    "20251223-120851_0.33_100",
    "20251223-120945_0.8_79",
    "20251223-121121_0.9_47",
    "20251223-121317_0.95_45",
    "20251223-121502_0.999_24",
    # batch 500 runs
    "20251223-132427_0.0_100",
    "20251223-135234_0.33_99",
    "20251223-135857_0.8_84",  # l=42.37
    "20251223-140641_0.9_55",  # l=49.34
    "20251223-141653_0.95_37",  # l=49.59
    "20251223-142728_0.999_21",  # l=45.76
    # batch 300 runs
    "20251226-001530_0.75_94",  # l=38.76
    "20251226-001826_0.85_73",  # l=48.12
    # batch 500, extened safety box
    "20251226-181302_0.85_73",  # l=48.65
    # batch 400, length 150
    # "20251230-182128_0.85_87",  # l=63.82
    # batch 400, length 200
    "20251231-103020_0.9_77",  # l=77.66
    "20251231-135935_0.88_82",  # l= 71.63
}
base_folder = "rollout_data/collect_data"
global_buffer_file_name = "replay_buffer_120_85_el2.joblib"
SUBSET_SIZE = 120
N_SKIP = 120
# N_COMPLETION_TARGET = 180

# 0) Load all data from subfolders
all_run_data = {}
for exp_folder in os.listdir(base_folder):
    exp_path = Path(base_folder) / exp_folder
    if not exp_path.is_dir() or exp_folder in exp_folders_to_skip:
        continue
    print(f"Processing folder: {exp_folder}")

    all_run_data[exp_folder] = []
    i = 0
    # n_completed = 0
    for run_path in sorted(os.listdir(exp_path)):
        run_file = exp_path / run_path
        if (
            not run_file.is_file()
            or run_path.endswith("_info.pkl")
            or run_path.endswith(".joblib")
        ):
            continue

        run_data = np.load(run_file, allow_pickle=True)
        # if n_completed >= N_COMPLETION_TARGET and run_data["terminated"]:
        #     continue
        # if (
        #     len(all_run_data[exp_folder]) - n_completed
        #     >= (SUBSET_SIZE - N_COMPLETION_TARGET)
        #     and not run_data["terminated"]
        # ):
        #     continue
        # if run_data["terminated"]:
        #     n_completed += 1

        i += 1
        if i <= N_SKIP:
            continue

        all_run_data[exp_folder].append(run_data)
        if len(all_run_data[exp_folder]) >= SUBSET_SIZE:
            break

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
    for run_data in run_datas:  # [: len(run_datas) // 2]:
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
    for run_data in run_datas:  # [: len(run_datas) // 2]:
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
# for exp_folder, buffer in buffers.items():
#     save_path = Path(base_folder) / exp_folder
#     buffer.save_buffer(save_path)
global_save_path = Path(base_folder) / global_buffer_file_name
global_buffer.save_buffer(global_save_path)
