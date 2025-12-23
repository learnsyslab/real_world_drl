# first 11 dimensions are non-vision -> only look at those

import joblib

from crisp_drl.data.buffers_cleanrl import ReplayBufferGpu


buffer: ReplayBufferGpu = joblib.load("rollout_data/collect_data/replay_buffer.joblib")


mean = buffer.observations[:, :11].mean(dim=0).cpu().numpy()
std = buffer.observations[:, :11].std(dim=0).cpu().numpy()

print("Feature means (non-vision):", mean)
print("Feature stds (non-vision):", std)

print(f"Scale factors (non-vision): {1.0 / std}")
