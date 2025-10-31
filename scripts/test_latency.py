"""Draw a circle using the ManipulatorCartesianEnv."""

import functools
import time
import numpy as np
import multiprocessing as mp

from crisp_drl.agents.rlpd.env_wrappers import ActionTimeStampWrapper, BelowZTerminationWrapper, CLIWrapper, DictObservationToInfoMover, ContainerWatcherWrapper, FarAwayTerminationWrapper, ImageEncoderWrapper, InsertionResetWrapper, LastObservationWrapper, NaiveToGoalPositionWrapper, NoRotationActionWrapper, NoRotationNoGripperActionWrapper, ObservationFormatterWrapper, TimeMeasurementWrapper, observation_has_z_pressure

from crisp_gym.manipulator_env import make_env
from crisp_gym.util.rl_utils import load_actions_safe
# %% === Circle Parameters ===
RADIUS = 0.1  # [m]
CENTER = np.array([0.5, -0.15, 0.3])
CTRL_FREQ = 15  # control frequency in Hz
SIN_FREQ = 0.1  # frequency of circular motion in Hz


# %% === Environment Setup ===
env = make_env("my_env")
env.wait_until_ready()

# env = InsertionResetWrapper(env, initial_pos=np.array([0.200, -0.020, -0.200]), grasp_randomization_bounds=(np.array([-0.005, -0.005, -0.001]), np.array([0.005, 0.005, 0.002])), 
#                             insert_randomization_bounds=(np.array([-0.01, -0.01, 0.0]), np.array([0.01, 0.01, 0.005])), action_sequence_to_grasp=load_actions_safe("v3_go_to_pick.json"), action_sequence_after_grasp=load_actions_safe("v3_after_pick.json"))
env = ActionTimeStampWrapper(env)
env = LastObservationWrapper(env)
# env = BelowZTerminationWrapper(env, min_z=0.0475)
# env = FarAwayTerminationWrapper(env, approximate_goal_pos=np.array([539.75, -33,  50]) * 0.001, max_distance=0.055)
# env = ContainerWatcherWrapper(env, ctx=mp.get_context("spawn"))

# env = CLIWrapper(env, termination_fn = functools.partial(observation_has_z_pressure, error_threshold=0.005, previous_error_threshold=0.003, min_z_height=0.055))
# env = NaiveToGoalPositionWrapper(env, step_size=0.001, base_goal_position=np.array([0.541, -0.034,  0.0435]), ideal_grasp_position=np.array([0.58833, -0.13817,  0.04229]))
# env = ImageEncoderWrapper(env, n_cameras=2, image_size=(256, 256))
env = DictObservationToInfoMover(env)
# env = ObservationFormatterWrapper(env, keys_ranges_scales=[('observation.previous.action', (0,3), 10.0), ('observation.previous.action', (6,7), 20.0), ('observation.velocity.cartesian', (0, 3), 100.0), ('observation.error.cartesian', (0, 3), 10.0), ('observation.velocity.gripper', (0, 1), 20.0),
#                                         ('observation.error.gripper', (0, 1), 20.0), ('observation.state.gripper', (0, 1), 1.0), ('observation.target.gripper', (0, 1), 1.0), ('observation.images.wrist_camera', (0, 512), 1.0), ('observation.images.side_camera', (0, 512), 1.0)])
env = ObservationFormatterWrapper(env, keys_ranges_scales=[('observation.previous.action', (0,3), 10.0), ('observation.previous.error.cartesian', (0,3), 10.0), ('observation.velocity.cartesian', (0, 3), 100.0), ('observation.error.cartesian', (0, 3), 10.0), 
 ]) # ('observation.images.wrist_camera', (0, 512), 1.0), ('observation.images.side_camera', (0, 512), 1.0)]) # 268 or 1036
env = NoRotationNoGripperActionWrapper(env)

# %% === Move to Starting Point ===
start_position = CENTER + [0, RADIUS, 0]
print(f"Moving to start position: {start_position}")
env.move_to(position=start_position, speed=0.15)
time.sleep(1.0)
env.gripper.open()
obs, _ = env.reset()


# %%=== Generate Circle Trajectory ===
steps = int(CTRL_FREQ / SIN_FREQ)

angles = np.linspace(0, 2 * np.pi, steps, endpoint=False)
x = RADIUS * np.sin(angles)
y = RADIUS * np.cos(angles)

# Velocity (finite difference)
dx = np.diff(np.concatenate([[x[-1]], x]))
dy = np.diff(np.concatenate([[y[-1]], y]))


# %% === Execute Circular Motion ===
try:
    while True:
        t0 = time.time()
        ts = [t0]

        for t in range(steps):
            action = np.zeros((3))

            action[0] = dx[t]
            action[1] = dy[t]

            obs, _, _, _, _ = env.step(action, block=True)
            ts.append(time.time())

        delta = time.time() - t0
        diffs = np.diff(ts)
        std = np.std(diffs)
        print(f"Completed circle of {steps=} in {delta:.2f} seconds, effective freq: {steps / delta:.2f} Hz, std dev: {std*1000:.2f} ms")
        obs, _ = env.reset()

except KeyboardInterrupt:
    print("\nCircle drawing interrupted by user.")
    print("Going back home.")
    env.home()

    env.close()






