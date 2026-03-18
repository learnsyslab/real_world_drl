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
from crisp_drl.envs.pose_estimation_helper import PoseEstimationHelper

config = Config()

env = make_env("my_env_v4")
print("Env created.")
env.wait_until_ready()
print("Env ready.")


env = ActionTimeStampWrapper(env)
env = NoRotationNoGripperActionWrapper(env)
env = LastObservationWrapper(env)

pe_helper = PoseEstimationHelper(
    assumed_orientation=config.pose_estimation_assumed_orientation
)

for i in range(20):
    # Home
    env.unwrapped.home(home_config=config.custom_home_position)  # type: ignore
    obs, info = env.reset()
    current_pos = obs["observation.state.cartesian"][:3]
    # input("Homed. Press Enter to continue to randomized position...")

    # Go to randomized position near demo pose
    randomized_goal_pos = current_pos + np.random.uniform(
        low=-0.002, high=0.002, size=(3,)
    )
    for _ in range(15):
        action = np.zeros(3)
        action[:3] = np.clip(randomized_goal_pos - current_pos, -0.0005, 0.0005)
        obs, reward, terminated, truncated, info = env.step(action)
        current_pos = obs["observation.state.cartesian"][:3]
    # input("Reached randomized position. Press Enter to continue to pose estimation...")

    obs, reward, terminated, truncated, info = env.step(np.zeros(3))
    # Estimate lavender pose
    starting_pos = obs["observation.state.cartesian"][:3]
    estimation_pose_euler = obs["observation.state.cartesian"]
    lavender_pose, purple_pose = pe_helper.estimate_two_lego_bricks_absolute(
        obs["observation.images.wrist_camera"],
        obs["observation.images.wrist_depth_camera"],
        estimation_pose_euler,
    )

    # Measure error
    estimation_error = lavender_pose[:3, 3] - config.demo_grasped_pose_lavender[:3, 3]
    print(
        f"Episode {i + 1}: Lavender position: {lavender_pose[:3, 3]}, Grasp position error (mm): {estimation_error * 1000}, norm: {np.linalg.norm(estimation_error) * 1000:.2f} mm"
    )

    # Move to estimated grasp pose
    estimation_position = (
        lavender_pose[:3, 3]
        - config.demo_grasped_pose_lavender[:3, 3]
        + config.demo_grasp_pose_estimation_pose_euler[:3]
    )
    for _ in range(15):
        action = np.zeros(3)
        action[:3] = np.clip(
            estimation_position - current_pos,
            -0.0005,
            0.0005,
        )
        obs, reward, terminated, truncated, info = env.step(action)
        current_pos = obs["observation.state.cartesian"][:3]

    # input("Reached estimated grasp position. Press Enter to continue...")

    controller_error = estimation_position - obs["observation.state.cartesian"][:3]

    total_error_gripper_open = (
        obs["observation.state.cartesian"][:3]
        - config.demo_grasp_pose_estimation_pose_euler[:3]
    )

    env.unwrapped.gripper.set_target(0.2)  # type: ignore
    time.sleep(1.0)  # wait for gripper to close
    obs, *_ = env.step(np.zeros(3))

    total_error_gripper_closed = (
        obs["observation.state.cartesian"][:3] - config.grasp_position_ground_truth[:3]
    )
    print(
        f"Episode {i + 1}: total position error with gripper closed (mm): {total_error_gripper_closed * 1000}, norm: {np.linalg.norm(total_error_gripper_closed) * 1000:.2f} mm"
    )

    # Convert info numpy arrays to lists for JSON serialization
    reset_info_serializable = {
        "estimation_error": estimation_error.tolist(),
        "controller_error": controller_error.tolist(),
        "total_error_gripper_open": total_error_gripper_open.tolist(),
        "total_error_gripper_closed": total_error_gripper_closed.tolist(),  # -> xz error matters most (determines randomisation bounds)
        "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    with open("eval/pose_estimation_grasping/samples.jsonl", "a") as f:
        f.write(json.dumps(reset_info_serializable) + "\n")

    if (i + 1) % 10 == 0:
        print(f"Completed {i + 1} episodes.")
