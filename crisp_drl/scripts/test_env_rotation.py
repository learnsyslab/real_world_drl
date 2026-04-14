import os
import sys
import time
import cv2
import tifffile
from crisp_gym.envs.manipulator_env import ManipulatorCartesianEnv, make_env
import numpy as np
from pynput import keyboard
import imageio

env = make_env("my_env_v4_rotation")
print("Env created.")
env.wait_until_ready()
print("Env ready.")
env.gripper.open()
env.home()
env.reset()
print("Env reset.")
env.gripper.close()
time.sleep(3)

input("Enter to quit.")
env.close()

# home: -0.0263873 ,  0.34892041, -0.02547229, -2.3921653 ,  0.01901308, 2.74496494,  0.71526465
# grasp: 5.1047003e-01 -2.9486790e-02  4.1392598e-02
# place: 5.4190356e-01 -2.9769510e-02  5.1701453e-02


# Keyboard listener active. Press 'o' to open gripper, 'c' to close, 'r' to print_and_write state.
# if[('task', ''), ('observation.state.cartesian', array([ 0.51002705, -0.03166126,  0.07771762,  3.1398149 , -0.00529997,
#         0.00330537], dtype=float32)), ('observation.state.gripper', array([0.48168105], dtype=float32)), ('observation.target.gripper', array([0.5], dtype=float32)), ('observation.state.joints', array([-0.02646138,  0.35014075, -0.03229037, -2.3585236 ,  0.0231909 ,
#         2.713794  ,  0.70426035], dtype=float32)), ('dt_camera', 1.4479737728834152e-06), ('observation.state.target', array([ 3.08386065e-01, -1.66665085e-04,  4.87584074e-01, -3.13918706e+00,
#        -4.33491024e-03,  2.47853464e-03]))]
# ['observation.images.wrist_camera', 'observation.images.wrist_depth_camera']
# Executed: gripper close
# cr[ 0.5100313  -0.03166383  0.07772372  3.1398158  -0.00531182  0.00329709]
# ir[ 0.52480197 -0.03154222  0.0966146  -3.1412485  -0.00335749  0.00383465]
# [ 0.5421958  -0.03224613  0.08956918 -3.14106    -0.00318236  0.00329792]
