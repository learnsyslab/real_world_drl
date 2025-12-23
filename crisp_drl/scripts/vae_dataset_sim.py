import joblib
import argparse

import numpy as np
from crisp_drl.data.buffers_cleanrl import ReplayBufferGpu

parser = argparse.ArgumentParser()
parser.add_argument("--buffer_path", type=str, help="Path to the buffer file")
parser.add_argument(
    "--obs_dim", type=int, help="Dimension of the observation to extract"
)
args = parser.parse_args()

buffer: ReplayBufferGpu = joblib.load(args.buffer_path)
print(f"Loaded buffer from {args.buffer_path} with {buffer.size()} transitions.")

N, D_OBS = buffer.observations.shape
dataset = buffer.observations[:, D_OBS - args.obs_dim :].cpu().numpy()
output_path = args.buffer_path.replace(".joblib", ".npz")
np.savez_compressed(output_path, features=dataset)
