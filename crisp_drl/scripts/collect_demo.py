from crisp_gym.envs.manipulator_env import ManipulatorCartesianEnv, make_env
import numpy as np
from pynput import keyboard

env = make_env("my_env_v3_grav_comp_xyz")
print("Env created.")
env.wait_until_ready()
print("Env ready.")
env.gripper.open()
env.home()
env.reset()
print("Env reset.")


def on_press(key):
    """Handle keyboard events in background."""
    try:
        if hasattr(key, "char"):
            if key.char == "o":
                env.step(np.array([0, 0, 0, 0, 0, 0, 0.2]))
                print("Executed: gripper open")
            elif key.char == "c":
                env.step(np.array([0, 0, 0, 0, 0, 0, -0.2]))
                print("Executed: gripper close")
            elif key.char == "r":
                obs, *_ = env.step(np.zeros(7))
                print(obs["observation.state.cartesian"])
            elif key.char == "f":
                obs, *_ = env.step(np.zeros(7))
                print([(k, obs[k]) for k in obs if "image" not in k])
    except Exception as e:
        print(f"Error handling key press: {e}")


# Start keyboard listener in background thread
listener = keyboard.Listener(on_press=on_press)
listener.start()

print(
    "Keyboard listener active. Press 'o' to open gripper, 'c' to close, 'r' to print state."
)

input()

# Stop listener and close environment
listener.stop()
env.close()

# home: -0.0263873 ,  0.34892041, -0.02547229, -2.3921653 ,  0.01901308, 2.74496494,  0.71526465
# grasp: 5.1047003e-01 -2.9486790e-02  4.1392598e-02
# place: 5.4190356e-01 -2.9769510e-02  5.1701453e-02
