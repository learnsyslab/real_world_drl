"""Teleop + PE/grasp-demo capture (extends the original collect_demo).

Combines the original free-jog/image-capture flow with the Procedure B PE
calibration workflow from crisp_drl/scripts/calibrate_lego_2x4.py:

  - jog: w/s/a/d/q/e (translation), y/x/k/l/m/n (rotation), o/c (gripper)
  - capture: i = save RGB/depth + record TCP for PE
  - PE: t = servo to refined-PE hover; p = run PE on last capture and print a
        paste-ready demo_grasped_pose_lavender block
  - state: r = print TCP, f = print obs keys
  - step size: +/- to scale translation step
  - Enter to quit
"""

import argparse
import os
import sys

import cv2
import imageio  # noqa: F401  (kept for parity with the original file)
import numpy as np
import tifffile
from pynput import keyboard

from crisp_gym.envs.manipulator_env import ManipulatorCartesianEnv, make_env  # noqa: F401
from crisp_drl.envs.pose_estimation_helper import PoseEstimationHelper
from crisp_drl.agents.shared.insertion_env_config import LegoConfig2x4
from crisp_drl.envs.pose_visualizer import PoseOverlayRenderer, dump_raw_npz


# ---------------------------------------------------------------------- #
# CLI                                                                    #
# ---------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true", help="Append jog log to jsonl.")
parser.add_argument("--brick-color", default="lavender")
parser.add_argument("--brick-size", default="2x4")
parser.add_argument(
    "--exp-name",
    default="angled_bracket_grasped_test_pe_g",
    help="Used in image / jsonl filenames.",
)
parser.add_argument(
    "--env-name",
    default="my_env_v3_grav_comp_xyz",
    help="manipulator_env name (preserves original default).",
)
parser.add_argument(
    "--viz-dir",
    default="pe_viz/collect_demo",
    help="Directory for PE overlay PNGs + raw npz dumps (per attempt).",
)
parser.add_argument(
    "--no-viz",
    action="store_true",
    help="Disable PE overlay rendering (skip mesh load).",
)
parser.add_argument(
    "--camera-info-json",
    default="camera_parameters/realsense_d405_single.json",
    help="Camera intrinsics JSON for PE overlay projection.",
)
parser.add_argument(
    "--mesh-path",
    default=None,
    help="Override .obj mesh path. Default: "
         "/workspaces/isaac_ros-dev/lego_assets/lego_<size>_<color>_up.obj.",
)
# Allow legacy `--save` style as positional flag too.
if len(sys.argv) > 1 and sys.argv[1] == "--save" and "--save" not in sys.argv[2:]:
    pass  # already handled by argparse
ARGS = parser.parse_args()


# ---------------------------------------------------------------------- #
# Env                                                                    #
# ---------------------------------------------------------------------- #
env = make_env(ARGS.env_name)
print(f"Env created ({ARGS.env_name}).")
env.wait_until_ready()
print("Env ready.")
env.gripper.open()
env.home()
env.reset()
print("Env reset.")


# ---------------------------------------------------------------------- #
# PE config (Procedure B: use grasp_position_ground_truth as TCP reference)
# ---------------------------------------------------------------------- #
_cfg = LegoConfig2x4()
GRASP_POS_REF = np.array(_cfg.grasp_position_ground_truth, dtype=np.float64)
GRASP_Z = float(GRASP_POS_REF[2])
REFINED_PE_HOVER = np.array([GRASP_POS_REF[0], GRASP_POS_REF[1], GRASP_Z + 0.015])
print(f"[calib] Config.grasp_position_ground_truth = {GRASP_POS_REF}")
print(f"[calib] REFINED_PE_HOVER                   = {REFINED_PE_HOVER}")

pe_helper = PoseEstimationHelper(
    assumed_orientation=np.array([]),
    lock_orientation=False,
    brick_size=ARGS.brick_size,
    use_tracker=False,
)
print(f"[pe] helper ready (brick_size={ARGS.brick_size}, color={ARGS.brick_color})")

# ---------------------------------------------------------------------- #
# PE overlay renderer (optional)                                         #
# ---------------------------------------------------------------------- #
renderer: PoseOverlayRenderer | None = None
if not ARGS.no_viz:
    mesh_candidates = []
    if ARGS.mesh_path:
        mesh_candidates.append(ARGS.mesh_path)
    mesh_candidates.extend(
        [
            f"/workspaces/isaac_ros-dev/lego_assets/"
            f"lego_{ARGS.brick_size}_{ARGS.brick_color}_up.obj",
            os.path.expanduser(
                f"~/workspaces/isaac_ros-dev/lego_assets/"
                f"lego_{ARGS.brick_size}_{ARGS.brick_color}_up.obj"
            ),
        ]
    )
    mesh_path = next((p for p in mesh_candidates if p and os.path.exists(p)), None)
    try:
        renderer = PoseOverlayRenderer(
            camera_info_json_path=ARGS.camera_info_json,
            mesh_path=mesh_path,
            use_default_mesh_fallback=False,
        )
        os.makedirs(ARGS.viz_dir, exist_ok=True)
        print(f"[viz] renderer ready  mesh={mesh_path}  out={ARGS.viz_dir}")
    except Exception as exc:  # noqa: BLE001
        print(f"[viz] renderer unavailable ({exc}) — continuing without overlays.")
        renderer = None


# ---------------------------------------------------------------------- #
# Runtime state                                                          #
# ---------------------------------------------------------------------- #
exp_name = ARGS.exp_name
save_to_file = ARGS.save
rot_deg = 1
rot_deg_z = 0.25
rot_deg_x = 0.25
step_mm = 3.0

i_demo_img = 0
last_pe_tcp: np.ndarray | None = None
last_color_path: str | None = None
last_depth_path: str | None = None

os.makedirs("test_images", exist_ok=True)
if save_to_file:
    os.makedirs("rollout_data/demos", exist_ok=True)


def print_and_write(line):
    line = str(line)
    if save_to_file:
        fd = os.open(
            os.path.join("rollout_data/demos", f"{exp_name}.jsonl"),
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o644,
        )
        os.write(fd, (line + "\n").encode())
        os.close(fd)
    print(line)


# ---------------------------------------------------------------------- #
# Robot helpers                                                          #
# ---------------------------------------------------------------------- #
def servo_to_xyz(target_xyz, step_size=0.004, tol=0.001, max_steps=600):
    obs, *_ = env.step(np.zeros(7))
    current = obs["observation.state.cartesian"][:3]
    n = 0
    while np.linalg.norm(target_xyz - current) > tol and n < max_steps:
        delta = target_xyz - current
        norm = np.linalg.norm(delta)
        step = delta / norm * min(step_size, norm)
        obs, *_ = env.step(np.array([step[0], step[1], step[2], 0, 0, 0, 0]))
        current = obs["observation.state.cartesian"][:3]
        n += 1
    res_mm = np.linalg.norm(target_xyz - current) * 1000
    print(f"[servo] done: {current}  residual={res_mm:.1f} mm  steps={n}")


# ---------------------------------------------------------------------- #
# PE: load last captured pair, run segmentation + FoundationPose         #
# ---------------------------------------------------------------------- #
def run_pe():
    """Procedure B: load last RGB/depth, run PE, print paste-ready block.

    Math:
        o_T_o = lav_R.T @ (grasp_position_ground_truth − lav_t)
    where lav_R, lav_t come from the PE result (brick world pose).
    """
    if last_color_path is None or last_pe_tcp is None:
        print("[PE] No image — press 'i' first.")
        return

    print(f"[PE] Loading {last_color_path} ...")
    image = tifffile.imread(last_color_path)
    depth = tifffile.imread(last_depth_path)
    depth_m = depth.astype(np.float32) / 1000.0

    print("[PE] Running SAM3 segmentation ...")
    masks = pe_helper.segmenter.segment_lego(image, colors=("lavender", "yellow"))
    print(f"[PE] Detected masks: {list(masks.keys())}")
    if ARGS.brick_color not in masks:
        print(f"[PE] ERROR: '{ARGS.brick_color}' not detected.")
        return

    print("[PE] Running FoundationPose ...")
    cam_pose = pe_helper.pose_estimator.estimate_lego(
        image, depth_m, masks[ARGS.brick_color], ARGS.brick_color
    )
    print(f"[PE] camera-frame pose:\n{cam_pose}")

    world_pose = pe_helper._compute_pose_in_world_frame(cam_pose, last_pe_tcp)
    print(f"[PE] world-frame pose:\n{world_pose}")

    lav_R = world_pose[:3, :3]
    lav_t = world_pose[:3, 3]
    o_T_o = lav_R.T @ (GRASP_POS_REF - lav_t)

    print(f"[PE] grasp_position_ground_truth (Config): {GRASP_POS_REF}")
    print(f"[PE] brick centroid in world:              {lav_t}")
    print(f"[PE] grasp_pos − lav_t (world frame mm):   "
          f"{np.round((GRASP_POS_REF - lav_t) * 1000, 2)}")
    print(f"[PE] o_T_o_tcpgrasp_{ARGS.brick_color} (brick mm):   "
          f"{np.round(o_T_o * 1000, 2)}")

    # ---- Overlay + raw-npz dump --------------------------------------
    if renderer is not None:
        ep_idx = max(0, i_demo_img - 1)
        try:
            overlay_path = renderer.save_single(
                out_dir=ARGS.viz_dir,
                episode_idx=ep_idx,
                rgb=image,
                pose_cam_obj=cam_pose,
                mask=masks[ARGS.brick_color],
                suffix=f"{ARGS.brick_color}_demo_pose",
                label=f"{ARGS.brick_color} ep{ep_idx} demo PE  "
                      f"o_T_o={np.round(o_T_o * 1000, 1)} mm",
            )
            print(f"[viz] overlay → {overlay_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[viz] overlay save failed: {exc}")
        try:
            npz_path = dump_raw_npz(
                ARGS.viz_dir,
                episode_idx=ep_idx,
                rgb=image,
                depth=depth,
                mask=masks[ARGS.brick_color],
                tcp_cart=last_pe_tcp,
                cam_pose=cam_pose,
                world_pose=world_pose,
                grasp_pos_ref=GRASP_POS_REF,
                o_T_o=o_T_o,
            )
            print(f"[viz] raw npz → {npz_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[viz] npz save failed: {exc}")

    R, t = lav_R, lav_t
    print(
        f"""
=== Config.grasp_position_ground_truth UNCHANGED (kept at training reference) ===

=== Paste into LegoConfig2x4 in crisp_drl/agents/shared/insertion_env_config.py ===

    demo_grasped_pose_{ARGS.brick_color}: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [{R[0,0]:.8f}, {R[0,1]:.8f}, {R[0,2]:.8f}, {t[0]:.8f}],
                [{R[1,0]:.8f}, {R[1,1]:.8f}, {R[1,2]:.8f}, {t[1]:.8f}],
                [{R[2,0]:.8f}, {R[2,1]:.8f}, {R[2,2]:.8f}, {t[2]:.8f}],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )

=== End paste ===

Sanity (typical 2x4 grasp): a few mm xy, ~25 mm z.
    grasp_pos − lav_t = {np.round((GRASP_POS_REF - t) * 1000, 2)} mm  (world)
    o_T_o             = {np.round(o_T_o * 1000, 2)} mm  (brick)
Repeat i+p with the brick at different yaws — brick-frame o_T_o should stay
stable to within ~2 mm. Larger drift means PE yaw extraction is jittering.
"""
    )


# ---------------------------------------------------------------------- #
# Keyboard handler                                                       #
# ---------------------------------------------------------------------- #
def on_press(key):
    """Handle keyboard events in background."""
    global i_demo_img, last_pe_tcp, last_color_path, last_depth_path, step_mm
    if not hasattr(key, "char") or key.char is None:
        return
    c = key.char
    s = step_mm / 1000.0  # mm → m

    # ---- gripper -------------------------------------------------------
    if c == "o":
        env.step(np.array([0, 0, 0, 0, 0, 0, 0.2]))
        print_and_write("Executed: gripper open")
        return
    if c == "c":
        env.step(np.array([0, 0, 0, 0, 0, 0, -0.4]))
        print_and_write("Executed: gripper close")
        return

    # ---- translation (w/s/a/d/q/e) ------------------------------------
    if c == "w":
        env.step(np.array([+s, 0, 0, 0, 0, 0, 0]))
        return
    if c == "s":
        env.step(np.array([-s, 0, 0, 0, 0, 0, 0]))
        return
    if c == "a":
        env.step(np.array([0, +s, 0, 0, 0, 0, 0]))
        return
    if c == "d":
        env.step(np.array([0, -s, 0, 0, 0, 0, 0]))
        return
    if c == "q":
        env.step(np.array([0, 0, +s, 0, 0, 0, 0]))
        return
    if c == "e":
        env.step(np.array([0, 0, -s, 0, 0, 0, 0]))
        return
    if c == "+":
        step_mm = min(step_mm * 2, 20.0)
        print(f"step = {step_mm:.1f} mm")
        return
    if c == "-":
        step_mm = max(step_mm / 2, 0.5)
        print(f"step = {step_mm:.1f} mm")
        return

    # ---- rotation (preserve original mapping) -------------------------
    if c == "y":
        env.step(np.array([0, 0, 0, 0, np.deg2rad(rot_deg), 0, 0.0]))
        print_and_write(f"Executed: rotate +y {rot_deg}°")
        return
    if c == "x":
        env.step(np.array([0, 0, 0, 0, np.deg2rad(-rot_deg), 0, 0.0]))
        print_and_write(f"Executed: rotate -y {rot_deg}°")
        return
    if c == "k":
        env.step(np.array([0, 0, 0, 0, 0, np.deg2rad(rot_deg_z), 0.0]))
        print_and_write(f"Executed: rotate +z {rot_deg_z:.2f}°")
        return
    if c == "l":
        env.step(np.array([0, 0, 0, 0, 0, np.deg2rad(-rot_deg_z), 0.0]))
        print_and_write(f"Executed: rotate -z {rot_deg_z:.2f}°")
        return
    if c == "m":
        env.step(np.array([0, 0, 0, np.deg2rad(rot_deg_x), 0, 0, 0.0]))
        print_and_write(f"Executed: rotate +x {rot_deg_x:.2f}°")
        return
    if c == "n":
        env.step(np.array([0, 0, 0, np.deg2rad(-rot_deg_x), 0, 0, 0.0]))
        print_and_write(f"Executed: rotate -x {rot_deg_x:.2f}°")
        return

    # ---- state inspection ---------------------------------------------
    if c == "r":
        obs, *_ = env.step(np.zeros(7))
        print_and_write(obs["observation.state.cartesian"])
        return
    if c == "f":
        obs, *_ = env.step(np.zeros(7))
        print_and_write([(k, obs[k]) for k in obs if "image" not in k])
        print_and_write([k for k in obs if "image" in k])
        return

    # ---- PE workflow --------------------------------------------------
    if c == "t":
        print(f"[servo] moving to REFINED_PE_HOVER {REFINED_PE_HOVER} ...")
        servo_to_xyz(REFINED_PE_HOVER)
        return
    if c == "p":
        try:
            run_pe()
        except Exception as exc:  # noqa: BLE001
            print(f"[PE] FAILED: {exc}")
        return

    # ---- capture (also records TCP for subsequent PE) -----------------
    if c == "i":
        obs, *_ = env.step(np.zeros(7))
        tcp = obs["observation.state.cartesian"]
        last_pe_tcp = tcp.copy()

        color_path = f"test_images/demo_img_color_{i_demo_img}_{exp_name}.tiff"
        depth_path = f"test_images/demo_img_depth_{i_demo_img}_{exp_name}.tiff"
        tifffile.imwrite(color_path, obs["observation.images.wrist_camera"])
        tifffile.imwrite(depth_path, obs["observation.images.wrist_depth_camera"])
        last_color_path = color_path
        last_depth_path = depth_path

        # Resized + cropped variants kept from the original collect_demo.
        tifffile.imwrite(
            f"test_images/demo_img_color_resized_cropped_{i_demo_img}_{exp_name}.tiff",
            cv2.resize(
                obs["observation.images.wrist_camera"][175 : 175 + 224, 346 : 346 + 224],
                (336, 336),
                interpolation=cv2.INTER_AREA,
            )[56 : 56 + 224, 56 : 56 + 224],
        )
        tifffile.imwrite(
            f"test_images/demo_img_color_cropped_{i_demo_img}_{exp_name}.tiff",
            obs["observation.images.wrist_camera"][175 : 175 + 224, 346 : 346 + 224],
        )
        print(f"Saved {color_path}  PE-TCP: {tcp}")
        i_demo_img += 1
        return


# 346, 175

# Start keyboard listener in background thread
listener = keyboard.Listener(on_press=on_press)
listener.start()

print(
    "\nKeys:\n"
    "  Translation:  w/s = ±X   a/d = ±Y   q/e = ±Z   (+/- to scale step_mm)\n"
    "  Rotation:     y/x = ±Ry  k/l = ±Rz  m/n = ±Rx\n"
    "  Gripper:      o = open   c = close\n"
    "  State:        r = print TCP   f = print obs keys\n"
    "  Capture:      i = save RGB+depth + record TCP for PE\n"
    "  PE:           t = servo to REFINED_PE_HOVER   p = run PE + print paste\n"
    f"                viz: {'ON → ' + ARGS.viz_dir if renderer is not None else 'OFF'}\n"
    f"  Step:         {step_mm:.1f} mm\n"
    "  Enter = quit\n"
    "\nProcedure B workflow: t → i → p   (repeat i+p with brick at different yaws)\n"
)

input()

# Stop listener and close environment
listener.stop()
env.close()

# home: -0.0263873 ,  0.34892041, -0.02547229, -2.3921653 ,  0.01901308, 2.74496494,  0.71526465
# grasp: 5.1047003e-01 -2.9486790e-02  4.1392598e-02
# place: 5.4190356e-01 -2.9769510e-02  5.1701453e-02
