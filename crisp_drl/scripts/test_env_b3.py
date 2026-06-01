import numpy as np

from crisp_drl.agents.shared.algorithm_config import Config
from crisp_drl.agents.shared.insertion_env_config import ShelfBoxConfig
from crisp_drl.agents.shared.insertion_wrapper_s import printoptions
from crisp_drl.envs.make_env import create_real_env_b3


class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


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


def wrap_angle_to_pi(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


alg_config = Config()
env_config = ShelfBoxConfig()
args = Dict2Class({"eval": False, "use_pose_estimation": False})
env = create_real_env_b3(alg_config=alg_config, env_config=env_config, args=args)


try:
    while True:
        obs, reset_info = env.reset()

        target_xyz = (
            env_config.goal_position_ground_truth
            + np.asarray(reset_info["reset.grasped.delta"], dtype=float)
            + np.asarray(reset_info["reset.goal_position.offset"], dtype=float)
        )
        target_rotation_z = float(reset_info["reset.rotation.goal"])

        done = False
        while not done:
            current_pos = np.asarray(
                obs["observation.state.cartesian"][:3], dtype=float
            )
            current_rotation_z = float(obs["observation.state.cartesian"][5])

            action = np.zeros(3, dtype=float)
            perfect_action_world = target_xyz - current_pos
            action[:2] = world_delta_to_plane_action(perfect_action_world)
            action[2] = wrap_angle_to_pi(target_rotation_z - current_rotation_z)

            if np.linalg.norm(action[:2]) > 0.00025:
                action[:2] = action[:2] / np.linalg.norm(action[:2]) * 0.00025
            action[2] = float(np.clip(action[2], -np.deg2rad(5.0), np.deg2rad(5.0)))

            with printoptions(precision=4):
                print(
                    "target_xyz:",
                    target_xyz,
                    "current_xyz:",
                    current_pos,
                    "target_rot_z:",
                    target_rotation_z,
                    "current_rot_z:",
                    current_rotation_z,
                    "action:",
                    action,
                )

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        with printoptions(precision=4):
            print(f"Ended at {obs['observation.state.cartesian']}")

except KeyboardInterrupt:
    env.close()
