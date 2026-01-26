import cv2
import tifffile
from crisp_gym.envs.manipulator_env import ManipulatorCartesianEnv, make_env
import numpy as np
from pynput import keyboard
import imageio

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
            print([k for k in obs if "image" in k])
        elif key.char == "i":
            obs, *_ = env.step(np.zeros(7))
            tifffile.imwrite(
                "test_images/demo_img_color.tiff",
                obs["observation.images.wrist_camera"],
            )
            tifffile.imwrite(
                "test_images/demo_img_depth.tiff",
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
                "test_images/demo_img_color_resized_cropped.tiff",
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
                "test_images/demo_img_color_cropped.tiff",
                obs["observation.images.wrist_camera"][
                    175 : 175 + 224, 346 : 346 + 224
                ],
            )


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
