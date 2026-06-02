import numpy as np
import threading

from pynput import keyboard

from crisp_drl.agents.shared.algorithm_config import Config
from crisp_drl.agents.shared.insertion_env_config import ShelfBoxConfig
from crisp_drl.agents.shared.insertion_wrapper_s import printoptions
from crisp_drl.envs.make_env import create_real_env_b1


class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


GRASP_DELTA_ROTATION_DEG = 30.0
INSERTION_AXIS = np.array([-0.5, 0.0, -0.7071], dtype=float)


def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


def orthogonal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array([0.0, 1.0, 0.0], dtype=float)
    if abs(float(np.dot(axis, reference))) > 0.95:
        reference = np.array([1.0, 0.0, 0.0], dtype=float)
    first = normalize(np.cross(axis, reference))
    second = normalize(np.cross(axis, first))
    return first, second


def rotate_about_y(vec: np.ndarray, angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    rotation = np.array(
        [
            [cos_angle, 0.0, sin_angle],
            [0.0, 1.0, 0.0],
            [-sin_angle, 0.0, cos_angle],
        ],
        dtype=float,
    )
    return rotation @ vec


def world_delta_to_plane_action(delta_world: np.ndarray) -> np.ndarray:
    axis = normalize(INSERTION_AXIS)
    plane_axis_1, plane_axis_2 = orthogonal_basis(axis)
    plane_delta = delta_world - np.dot(delta_world, axis) * axis
    return np.array(
        [
            float(np.dot(plane_delta, plane_axis_1)),
            float(np.dot(plane_delta, plane_axis_2)),
        ],
        dtype=float,
    )


alg_config = Config()
env_config = ShelfBoxConfig()
args = Dict2Class({"eval": False, "use_pose_estimation": False})
env = create_real_env_b1(alg_config=alg_config, env_config=env_config, args=args)
reset_requested = threading.Event()


def on_press(key):
    if getattr(key, "char", None) == "s":
        reset_requested.set()


listener = keyboard.Listener(on_press=on_press)
listener.start()


try:
    while True:
        if reset_requested.is_set():
            print("Reset requested. Resetting environment.")
            reset_requested.clear()

        obs, reset_info = env.reset()

        grasp_delta = np.asarray(reset_info["reset.grasped.delta"], dtype=float)
        rotated_grasp_delta = rotate_about_y(grasp_delta, GRASP_DELTA_ROTATION_DEG)

        shifted_goal_pos = env_config.goal_position_ground_truth + rotated_grasp_delta

        done = False
        while not done:
            if reset_requested.is_set():
                print("Reset requested. Resetting environment.")
                reset_requested.clear()
                break

            current_pos = np.asarray(
                obs["observation.state.cartesian"][:3], dtype=float
            )
            perfect_action_world = shifted_goal_pos - current_pos
            perfect_action_plane = world_delta_to_plane_action(perfect_action_world)
            perfect_action_norm = float(np.linalg.norm(perfect_action_plane))
            if perfect_action_norm > 1e-8:
                perfect_action_plane = (
                    min(1.0, 0.00025 / perfect_action_norm) * perfect_action_plane
                )
            with printoptions(precision=4):
                print(
                    "goal_gt:",
                    env_config.goal_position_ground_truth,
                    "grasp_offset:",
                    grasp_delta,
                    "adjusted_goal:",
                    shifted_goal_pos,
                    "perfect_action_raw:",
                    perfect_action_world,
                    "perfect_action_plane:",
                    perfect_action_plane,
                )
            obs, _, terminated, truncated, info = env.step(perfect_action_plane)
            done = terminated or truncated

        with printoptions(precision=4):
            print(f"Ended at {obs['observation.state.cartesian']}")

except KeyboardInterrupt:
    pass
finally:
    listener.stop()
    env.close()
