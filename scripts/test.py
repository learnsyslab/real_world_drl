from crisp_py.robot import Robot

robot = Robot()
robot.wait_until_ready()

# %%

print(robot.controller_switcher_client.get_controller_list())
print(robot.cartesian_controller_parameters_client.list_parameters())
robot.cartesian_controller_parameters_client.set_parameters([("task.k_pos_z", 0.0)])
robot.shutdown()
