from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import jsonlines
import numpy as np

base_exp_name = "env_v0"
exp_name = f"{base_exp_name}_{0.8}"

repo_id = f"collect_data_real/{exp_name}"
data_dir = Path("rollout_data") / repo_id

dataset = LeRobotDataset(repo_id=repo_id, root=data_dir)

with open(data_dir / "meta" / "episodes.jsonl", "r") as f:
    meta_episodes = list(jsonlines.Reader(f))

n_episodes_meta = len(meta_episodes)
episode_lengths_meta = [m["length"] for m in meta_episodes]
n_frames_meta = sum(episode_lengths_meta)

dataset_length = len(dataset)

print(
    f"Dataset has {dataset_length} frames according to LeRobotDataset, and {n_frames_meta} "
    f"frames according to episodes.jsonl ({n_episodes_meta} episodes)."
)

# count how ofthen the reward is 10 at last index and second to last index
# same with is_terminal
reward_at_last_index = 0
reward_at_second_last_index = 0
is_terminal_at_last_index = 0
is_terminal_at_second_last_index = 0

for i in range(n_episodes_meta):
    start_idx = sum(episode_lengths_meta[:i])
    end_idx = start_idx + episode_lengths_meta[i]
    # Verify that the terminal flag is set correctly
    all_episode_rewards = [
        dataset[j]["reward"].squeeze().item() for j in range(start_idx, end_idx)
    ]
    if dataset[start_idx]["reward"].squeeze().item() is float("nan"):
        print(f"Discrepancy in episode {i} (missing nan at start)")
        print(
            "All rewards for the episode: ",
            all_episode_rewards,
        )
    if any(np.isnan(r) for r in all_episode_rewards[1:]):
        print(f"Discrepancy in episode {i} (nan in middle of episode)")
        print(
            "All rewards for the episode: ",
            all_episode_rewards,
        )

    if all_episode_rewards[-1] == 10.0:
        reward_at_last_index += 1
    if all_episode_rewards[-2] == 10.0:
        reward_at_second_last_index += 1
    all_episode_is_terminals = [
        dataset[j]["is_terminal"].squeeze().item() for j in range(start_idx, end_idx)
    ]
    if all_episode_is_terminals[-1]:
        is_terminal_at_last_index += 1
    if all_episode_is_terminals[-2]:
        is_terminal_at_second_last_index += 1

print(
    f"Reward is 10.0 at last index in {reward_at_last_index} episodes, "
    f"and at second to last index in {reward_at_second_last_index} episodes."
)
print(
    f"Is_terminal is True at last index in {is_terminal_at_last_index} episodes, "
    f"and at second to last index in {is_terminal_at_second_last_index} episodes."
)
