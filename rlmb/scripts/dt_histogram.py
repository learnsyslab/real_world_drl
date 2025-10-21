import pickle
import matplotlib.pyplot as plt

# Path to your pickle file
pickle_path = "/home/linusschwarz/real_world_drl/rollouts/PreProgrammedPolicy__20251021-094115/inf_0.pkl"

# Load the data
with open(pickle_path, "rb") as f:
    data = pickle.load(f)[1:]

# Collect all dt keys
dt_keys = [key for key in data[0] if key.startswith("dt_")]

# Prepare data for each dt key
dt_values = {key: [entry[key] for entry in data] for key in dt_keys}

# Plot histograms
fig, axs = plt.subplots(len(dt_keys), 1, figsize=(8, 2 * len(dt_keys)), sharex=True)
if len(dt_keys) == 1:
    axs = [axs]  # Ensure axs is iterable

for ax, key in zip(axs, dt_keys):
    ax.hist(dt_values[key], bins=30, alpha=0.7)
    ax.set_title(f"Histogram of {key}")
    ax.set_ylabel("Count")

plt.xlabel("Value")
plt.tight_layout()
plt.show()