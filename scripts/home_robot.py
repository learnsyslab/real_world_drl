"""Homing the robot."""

# %%
import matplotlib.pyplot as plt
import numpy as np

from crisp_py.robot import make_robot

left_arm = make_robot("fr3")
print(left_arm._current_joint)
left_arm.wait_until_ready()

# %%
print(left_arm.end_effector_pose)
print(left_arm.joint_values)

# %%
print("Going to home position...")
left_arm.home()
homing_pose = left_arm.end_effector_pose.copy()
