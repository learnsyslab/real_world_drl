import mujoco
import numpy as np
from crisp_drl.envs import make_env


env = make_env.create_simulated_env(
    {
        "initial_keyframe": 2,
        "lego_shift_range": (-0.002, 0.002),
        "initial_position_range": (
            np.array([-1.0, -1.0, -2.0]) * 1e-3,  # np.zeros(3),
            np.array([1.0, 1.0, -1.0]) * 1e-3,  # np.zeros(3),
        ),
        "live_view": True,
    }
)

for _ in range(100):
    obs, info = env.reset()
    reset_grasp_pos = info["reset.grasped.position"]
    done = False
    while not done:
        action = np.array([0.0, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        data: mujoco.MjData = env.data  # type: ignore
        # print("Top wall pos", data.geom("wall_top").xpos)
        # print("Main box pos", data.geom("main_box").xpos)
        # print("Target:", info["observation"]["observation.state.target"])
        # print("Error:", info["observation"]["observation.error.cartesian"])

env.close()
