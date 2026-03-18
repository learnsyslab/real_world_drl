"""Push a local LeRobot dataset to Hugging Face Hub."""

from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

LOCAL_ROOT = Path(
    "/home/linusschwarz/repos/real_world_drl/rollout_data/collect_data_real/run_4_0.75"
)
HUB_REPO_ID = "LSY-lab/lego_stacking_generated_v0"

# Load the dataset from local disk
dataset = LeRobotDataset(
    repo_id=HUB_REPO_ID,
    root=LOCAL_ROOT,
)

print(f"Loaded dataset: {dataset.meta.info}")
print(f"Total episodes: {dataset.meta.total_episodes}")
print(f"Total frames: {dataset.meta.total_frames}")

# Push to Hugging Face Hub
dataset.push_to_hub(
    tags=["lego", "stacking", "franka", "real-world"],
    private=False,
)

print(f"Dataset pushed to https://huggingface.co/datasets/{HUB_REPO_ID}")
