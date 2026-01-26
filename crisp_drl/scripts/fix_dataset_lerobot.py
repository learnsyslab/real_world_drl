from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import jsonlines
import pyarrow.parquet as pq
import pyarrow as pa
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

# Track which episodes need reward modification
episodes_to_fix = []

for i in range(n_episodes_meta):
    start_idx = sum(episode_lengths_meta[:i])
    end_idx = start_idx + episode_lengths_meta[i]
    # Verify that the terminal flag is set correctly
    all_episode_rewards = [
        dataset[j]["reward"].squeeze().item() for j in range(start_idx, end_idx)
    ]

    if all_episode_rewards[-2] == 10.0:
        episodes_to_fix.append(i)
        print(f"Episode {i}: Will shift reward from second-to-last to last frame")

# Modify the parquet files (each episode has its own file)
for episode_idx in episodes_to_fix:
    parquet_path = (
        data_dir / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
    )
    table = pq.read_table(parquet_path)

    # Get the reward column and convert to numpy for modification
    reward_col = table.column("reward").to_numpy()
    reward_col = np.array(reward_col, dtype=np.float32)

    # Modify: set second-to-last to 0, last to 10
    reward_col[-2] = 0.0
    reward_col[-1] = 10.0

    # Replace the reward column in the table
    col_idx = table.schema.get_field_index("reward")
    new_table = table.set_column(col_idx, "reward", pa.array(reward_col))

    pq.write_table(new_table, parquet_path)
    print(f"Modified episode {episode_idx} and saved to {parquet_path}")

print(f"\nTotal modified: {len(episodes_to_fix)} episodes")
