import json
import time

import numpy as np
from crisp_drl.agents.shared.algorithm_config import Config
from crisp_drl.agents.shared.env_wrappers import (
    ActionTimeStampWrapper,
    LastObservationWrapper,
    NoRotationNoGripperActionWrapper,
    PoseEstimationEvalWrapper,
)
from crisp_drl.envs.make_env import make_env

config = Config()

env = make_env("my_env_v3")
print("Env created.")
env.wait_until_ready()
print("Env ready.")

env = ActionTimeStampWrapper(env)
env = NoRotationNoGripperActionWrapper(env)
env = LastObservationWrapper(env)
env = PoseEstimationEvalWrapper(
    env,
    config=config,
)

for i in range(100):
    obs, info = env.reset()
    # Convert reset info numpy arrays to lists for JSON serialization
    reset_info_serializable = {
        k: v.tolist() if isinstance(v, (np.ndarray,)) else v for k, v in info.items()
    }
    reset_info_serializable["datetime"] = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime()
    )
    with open("eval/pose_estimation/samples.jsonl", "a") as f:
        f.write(json.dumps(reset_info_serializable) + "\n")

    if (i + 1) % 10 == 0:
        print(f"Completed {i + 1} episodes.")
