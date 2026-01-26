import numpy as np

data = np.load(
    "rollout_data/collect_data_real/run_2_0.8/20260108-142328_0.8/rollout_000.npz",
    allow_pickle=True,
)

rewards = data["rewards"]

print("Rewards:", rewards)
