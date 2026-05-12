"""
Calibration of demo_grasped_pose_lavender for LegoConfig2x4 — Procedure B.

Procedure B uses Config.grasp_position_ground_truth directly as the reference
TCP (does NOT measure a real grasp). This keeps Config.grasp_position_ground_truth
unchanged so the trained policy sees the same physical reference at eval as
during data collection; only the PE-side calibration is updated.

Workflow:
  1. Place a lavender 2x4 brick on the table where you usually place it during
     data collection. Square it up (yaw approximately 0) — the brick's
     orientation matters because lav_R rotates the offset into brick-frame.
  2. Press 't' — robot servos to REFINED_PE_HOVER (camera viewpoint).
  3. Press 'i' — capture RGB + depth image with the brick on the table.
  4. Press 'p' (Isaac ROS / FoundationPose running) — run PE, compute
     o_T_o using Config.grasp_position_ground_truth, print paste-block.
  5. Repeat 'i' + 'p' a few times with the brick at different yaws to verify
     o_T_o is stable; if it drifts >2 mm, PE is unreliable here.

Translation keys (mm per press, adjustable with +/-):
  w/s  — +/-X     a/d  — +/-Y     q/e  — +/-Z

Rotation keys:
  y/x  — +/-Ry    k/l  — +/-Rz    m/n  — +/-Rx

Other:
  o/c  — gripper open/close
  r    — print TCP state
  t    — servo to REFINED_PE_HOVER
  i    — capture image (brick on table, no grasp)
  p    — run PE + print constants
  Enter — quit
"""

import tifffile
import numpy as np
from pynput import keyboard
from crisp_gym.envs.manipulator_env import make_env
from crisp_drl.envs.pose_estimation_helper import PoseEstimationHelper
from crisp_drl.agents.shared.algorithm_config import Config

# XYZ the runtime system hovers at for refined PE.
# = [wide_pe_lavender_X, wide_pe_lavender_Y, grasp_z + 0.015]
# Update from the "wide PE lavender pos" line in a real run.
REFINED_PE_HOVER = np.array([0.51669, -0.0275, 0.093])

env = make_env("my_env_v4")
print("Env created.")
env.wait_until_ready()
print("Env ready.")
env.gripper.open()
env.home()
env.reset()
print("Env reset.")

exp_name = "lego_2x4_calib"
rot_deg = 1.0
rot_deg_z = 0.25
rot_deg_x = 0.25
step_mm = 3.0  # translation step size in mm

# Procedure B: use Config.grasp_position_ground_truth directly as the grasp TCP.
# Keeps the policy's coordinate system anchored at the data-collection reference.
_cfg = Config()
GRASP_POS_REF = np.array(_cfg.grasp_position_ground_truth, dtype=np.float64)
print(f"[calib] Using Config.grasp_position_ground_truth = {GRASP_POS_REF}")

i_demo_img = 0
last_pe_tcp = None
last_color_path = None
last_depth_path = None


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
    residual = np.linalg.norm(target_xyz - current) * 1000
    print(f"[servo] done: {current}  residual={residual:.1f} mm  steps={n}")


def run_pe():
    if last_color_path is None or last_pe_tcp is None:
        print("[PE] No image — press 'i' first.")
        return

    print(f"[PE] Loading {last_color_path} ...")
    image = tifffile.imread(last_color_path)
    depth = tifffile.imread(last_depth_path)
    depth_m = depth.astype(np.float32) / 1000.0

    pe = PoseEstimationHelper(
        assumed_orientation=np.array([]),
        lock_orientation=False,
        brick_size="2x4",
        use_tracker=False,
    )

    print("[PE] Running SAM3 segmentation ...")
    masks = pe.segmenter.segment_lego(image, colors=("lavender", "yellow"))
    print(f"[PE] Detected masks: {list(masks.keys())}")

    if "lavender" not in masks:
        print("[PE] ERROR: 'lavender' not detected.")
        return

    print("[PE] Running FoundationPose (requires Isaac ROS) ...")
    cam_pose = pe.pose_estimator.estimate_lego(image, depth_m, masks["lavender"], "lavender")
    print(f"[PE] Camera-frame pose:\n{cam_pose}")

    world_pose = pe._compute_pose_in_world_frame(cam_pose, last_pe_tcp)
    print(f"[PE] World-frame pose:\n{world_pose}")

    lav_R = world_pose[:3, :3]
    lav_t = world_pose[:3, 3]
    grasp_pos = GRASP_POS_REF  # Procedure B: use Config reference, not a measured TCP
    o_T_o = lav_R.T @ (grasp_pos - lav_t)

    print(f"[PE] grasp_position_ground_truth (from Config): {grasp_pos}")
    print(f"[PE] brick centroid in world:                   {lav_t}")
    print(f"[PE] grasp_pos - lav_t (world frame):           {np.round((grasp_pos - lav_t) * 1000, 2)} mm")
    print(f"[PE] o_T_o_tcpgrasp_lavender (brick frame):     {np.round(o_T_o * 1000, 2)} mm")

    R, t = world_pose[:3, :3], world_pose[:3, 3]
    print(f"""
=== Config.grasp_position_ground_truth is UNCHANGED (kept at training reference) ===

=== Paste into LegoConfig2x4 in crisp_drl/agents/shared/insertion_env_config.py ===

    demo_grasped_pose_lavender: np.ndarray = field(
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

Sanity check — the wrapper will compute o_T_o = lav_R.T @ (grasp_pos - lav_t)
at runtime, so verify these magnitudes look right (a few mm xy, ~25 mm z is
typical for a 2x4 brick grasped at the studs):
    grasp_pos - lav_t = {np.round((grasp_pos - t) * 1000, 2)} mm  (world frame)
    o_T_o_tcpgrasp_lavender = {np.round(o_T_o * 1000, 2)} mm  (brick frame)
Repeat the i+p capture a few times with the brick at different yaws — the
brick-frame o_T_o should stay stable to within ~2 mm. Large drift means PE
yaw extraction is jittering.
""")


def on_press(key):
    global i_demo_img, last_pe_tcp, last_color_path, last_depth_path, step_mm
    if not hasattr(key, "char") or key.char is None:
        return
    c = key.char

    s = step_mm / 1000.0
    if c == "o":
        env.step(np.array([0, 0, 0, 0, 0, 0, 0.2]))
        print("gripper open")
    elif c == "c":
        env.step(np.array([0, 0, 0, 0, 0, 0, -0.4]))
        print("gripper close")
    elif c == "w":
        env.step(np.array([s, 0, 0, 0, 0, 0, 0]))
    elif c == "s":
        env.step(np.array([-s, 0, 0, 0, 0, 0, 0]))
    elif c == "a":
        env.step(np.array([0, s, 0, 0, 0, 0, 0]))
    elif c == "d":
        env.step(np.array([0, -s, 0, 0, 0, 0, 0]))
    elif c == "q":
        env.step(np.array([0, 0, s, 0, 0, 0, 0]))
    elif c == "e":
        env.step(np.array([0, 0, -s, 0, 0, 0, 0]))
    elif c == "y":
        env.step(np.array([0, 0, 0, 0, np.deg2rad(rot_deg), 0, 0]))
    elif c == "x":
        env.step(np.array([0, 0, 0, 0, np.deg2rad(-rot_deg), 0, 0]))
    elif c == "k":
        env.step(np.array([0, 0, 0, 0, 0, np.deg2rad(rot_deg_z), 0]))
    elif c == "l":
        env.step(np.array([0, 0, 0, 0, 0, np.deg2rad(-rot_deg_z), 0]))
    elif c == "m":
        env.step(np.array([0, 0, 0, np.deg2rad(rot_deg_x), 0, 0, 0]))
    elif c == "n":
        env.step(np.array([0, 0, 0, np.deg2rad(-rot_deg_x), 0, 0, 0]))
    elif c == "+":
        step_mm = min(step_mm * 2, 20.0)
        print(f"step = {step_mm:.1f} mm")
    elif c == "-":
        step_mm = max(step_mm / 2, 0.5)
        print(f"step = {step_mm:.1f} mm")
    elif c == "r":
        obs, *_ = env.step(np.zeros(7))
        print(obs["observation.state.cartesian"])
    elif c == "t":
        print(f"[servo] moving to REFINED_PE_HOVER {REFINED_PE_HOVER} ...")
        servo_to_xyz(REFINED_PE_HOVER)
    elif c == "i":
        obs, *_ = env.step(np.zeros(7))
        tcp = obs["observation.state.cartesian"]
        last_pe_tcp = tcp.copy()
        color_path = f"test_images/demo_img_color_{i_demo_img}_{exp_name}.tiff"
        depth_path = f"test_images/demo_img_depth_{i_demo_img}_{exp_name}.tiff"
        tifffile.imwrite(color_path, obs["observation.images.wrist_camera"])
        tifffile.imwrite(depth_path, obs["observation.images.wrist_depth_camera"])
        last_color_path = color_path
        last_depth_path = depth_path
        print(f"Saved {color_path}  PE-TCP: {tcp}")
        i_demo_img += 1
    elif c == "p":
        run_pe()


listener = keyboard.Listener(on_press=on_press)
listener.start()

print(
    f"\nControls:\n"
    f"  w/s/a/d/q/e = X/Y/Z steps ({step_mm:.0f} mm, +/- to adjust)\n"
    f"  y/x/k/l/m/n = rotate   o/c = gripper   r = print state\n"
    f"  t = servo to REFINED_PE_HOVER {REFINED_PE_HOVER}\n"
    f"  i = capture image (brick on table)   p = run PE + print paste-block\n"
    f"\nWorkflow (Procedure B, no grasp): t → i → p   "
    f"(repeat i+p with different brick yaws to sanity-check stability)\n"
)

input()

listener.stop()
env.close()
