import os
import argparse
import cv2
import tifffile
from crisp_gym.envs.manipulator_env import ManipulatorCartesianEnv, make_env
import numpy as np
from pynput import keyboard
import imageio

parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp_name", required=True, help="Experiment name used for saved outputs"
)
parser.add_argument(
    "--save", action="store_true", help="Write demo events to rollout_data/demos"
)
args = parser.parse_args()

env = make_env("my_env_v3_grav_comp")
# env = make_env("my_env_v3_grav_comp_xyz")
# env = make_env("my_env_v4")
print("Env created.")
env.wait_until_ready()
print("Env ready.")
env.gripper.open()
env.home()
env.reset()
print("Env reset.")

i_demo_img = 0

exp_name = args.exp_name
rot_deg = 1
rot_deg_z = 0.25
rot_deg_x = 5
trans = 0.001

save_to_file = args.save


def print_and_write(line):
    line = str(line)
    if save_to_file and "Executed:" not in line:
        fd = os.open(
            os.path.join("rollout_data/demos", f"{exp_name}.jsonl"),
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o644,
        )
        os.write(fd, (line.replace("\n", "") + "\n").encode())
        os.close(fd)
    print(line)


def on_press(key):
    global i_demo_img
    """Handle keyboard events in background."""
    if hasattr(key, "char"):
        if key.char == "o":
            env.step(np.array([0, 0, 0, 0, 0, 0, 0.2]))
            print_and_write("Executed: gripper open")
        if key.char == "y":
            env.step(np.array([0, 0, 0, 0, np.deg2rad(rot_deg), 0, 0.0]))
            print_and_write(f"Executed: rotate +y {rot_deg}°")
        if key.char == "x":
            env.step(np.array([0, 0, 0, 0, np.deg2rad(-rot_deg), 0, 0.0]))
            print_and_write(f"Executed: rotate -y {rot_deg}°")
        if key.char == "k":
            env.step(np.array([0, 0, 0, 0, 0, np.deg2rad(rot_deg_z), 0.0]))
            print_and_write(f"Executed: rotate +z {rot_deg_z:.2f}°")
        if key.char == "l":
            env.step(np.array([0, 0, 0, 0, 0, np.deg2rad(-rot_deg_z), 0.0]))
            print_and_write(f"Executed: rotate -z {rot_deg_z:.2f}°")
        if key.char == "1":
            env.step(np.array([0, 0, -trans, 0, 0, 0, 0.0]))
            print_and_write(f"Executed: down -z {-trans:.2f}°")
        if key.char == "2":
            env.step(np.array([0, 0, +trans, 0, 0, 0, 0.0]))
            print_and_write(f"Executed: up +z {trans:.2f}°")
        if key.char == "3":
            env.step(np.array([0, -trans, 0, 0, 0, 0, 0.0]))
            print_and_write(f"Executed: left -y {-trans:.2f}°")
        if key.char == "4":
            env.step(np.array([0, +trans, 0, 0, 0, 0, 0.0]))
            print_and_write(f"Executed: right +y {trans:.2f}°")
        if key.char == "5":
            env.step(np.array([-trans, 0, 0, 0, 0, 0, 0.0]))
            print_and_write(f"Executed: back -x {-trans:.2f}°")
        if key.char == "6":
            env.step(np.array([+trans, 0, 0, 0, 0, 0, 0.0]))
            print_and_write(f"Executed: front +x {trans:.2f}°")
        if key.char == "m":
            env.step(np.array([0, 0, 0, np.deg2rad(rot_deg_x), 0, 0, 0.0]))
            print_and_write(f"Executed: rotate +x {rot_deg_x:.2f}°")
        if key.char == "n":
            env.step(np.array([0, 0, 0, np.deg2rad(-rot_deg_x), 0, 0, 0.0]))
            print_and_write(f"Executed: rotate -x {rot_deg_x:.2f}°")
        elif key.char == "c":
            env.step(np.array([0, 0, 0, 0, 0, 0, -0.2]))
            print_and_write("Executed: gripper close")
        elif key.char == "r":
            obs, *_ = env.step(np.zeros(7))
            print_and_write(obs["observation.state.cartesian"])
        elif key.char == "f":
            obs, *_ = env.step(np.zeros(7))
            print_and_write([(k, obs[k]) for k in obs if "image" not in k])
            print_and_write([k for k in obs if "image" in k])
        elif key.char == "i":
            obs, *_ = env.step(np.zeros(7))
            # print(obs["observation.images.wrist_camera"])
            tifffile.imwrite(
                f"test_images/demo_img_color_{i_demo_img}_{exp_name}.tiff",
                obs["observation.images.wrist_camera"],
            )
            tifffile.imwrite(
                f"test_images/demo_img_depth_{i_demo_img}_{exp_name}.tiff",
                obs["observation.images.wrist_depth_camera"],
            )
            # cv2.imwrite(
            #     "test_images/demo_img_color_resized.png",
            #     cv2.resize(
            #         obs["observation.images.wrist_camera"],
            #         (768, 576),
            #         interpolation=cv2.INTER_AREA,
            #     ),
            # )
            tifffile.imwrite(
                f"test_images/demo_img_color_resized_cropped_{i_demo_img}_{exp_name}.tiff",
                cv2.resize(
                    obs["observation.images.wrist_camera"][
                        175 : 175 + 224, 346 : 346 + 224
                    ],
                    (336, 336),
                    interpolation=cv2.INTER_AREA,
                )[56 : 56 + 224, 56 : 56 + 224],
            )

            # imageio.imwrite(
            #     "test_images/demo_img_color_cropped.png",
            #     obs["observation.images.wrist_camera"][
            #         153 : 153 + 224, 243 : 243 + 224
            #     ],
            # )
            # imageio.imwrite(
            #     "test_images/demo_img_color_cropped.png",
            #     obs["observation.images.wrist_camera"][
            #         337 : 337 + 224, 571 : 571 + 224
            #     ],
            # )
            tifffile.imwrite(
                f"test_images/demo_img_color_cropped_{i_demo_img}_{exp_name}.tiff",
                obs["observation.images.wrist_camera"][
                    175 : 175 + 224, 346 : 346 + 224
                ],
            )

            i_demo_img += 1


# 346, 175

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