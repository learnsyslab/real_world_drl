import os
from pathlib import Path

import joblib
import numpy as np

from crisp_drl.data.buffers_cleanrl import ReplayBufferGpuWithPerfectActions


base_folder = "rollout_data/collect_data"
buffer_file_name_1 = "replay_buffer_120_85_el1.joblib"
buffer_file_name_2 = "replay_buffer_120_85_el2.joblib"
N_ROLLOUTS = 120
TRUNCATION_LENGTH = 150

buffer_1: ReplayBufferGpuWithPerfectActions = joblib.load(
    Path(base_folder) / buffer_file_name_1
)
buffer_2: ReplayBufferGpuWithPerfectActions = joblib.load(
    Path(base_folder) / buffer_file_name_2
)

n_success_1 = int(round(buffer_1.terminateds.sum().item()))
n_success_2 = int(round(buffer_2.terminateds.sum().item()))


total_len_1 = buffer_1.pos_acts
total_len_2 = buffer_2.pos_acts


def print_buffer_stats(buffer_id: int, n_success: int, total_len: int):
    print(f"Buffer {buffer_id}: {total_len / N_ROLLOUTS} average rollout length")
    print(f"Buffer {buffer_id}: {n_success / N_ROLLOUTS} success rate")
    print(
        f"Buffer {buffer_id}: {(total_len - TRUNCATION_LENGTH * (N_ROLLOUTS - n_success)) / n_success} average successful length"
    )


print_buffer_stats(1, n_success_1, total_len_1)
print_buffer_stats(2, n_success_2, total_len_2)
