"""
Full SiemensConfig recalibration tool — keyboard jog + snapshot capture.

Recalibrates every position-dependent field of SiemensConfig for a NEW
data-collection layout (new grasp spot, new insertion spot, new waypoints,
new camera viewpoints). Procedure B is used for demo_w_D_w_o / demo_t_D_t_o
(PE on the lid alone, no real grasp), so grasp_position_ground_truth is
captured by jogging the TCP to the physical grasp pose and pressing 'g'.

Mesh used by FoundationPose for this object:
  /home/gabor/workspaces/isaac_ros-dev/foundation_pose_meshes/SiemensLid_centered.obj
(Inside the Isaac ROS container this is mounted at
 /workspaces/isaac_ros-dev/lego_assets/SiemensLid_centered.obj — see
 pose_estimator._set_mesh_file_siemens.)

Recommended workflow:
  1. Jog the robot to the desired FIRST-HOME joint config, press 'u' (capture
     custom_first_home_position).
  2. Jog to general HOME, press 'h' (capture custom_home_position).
  3. Jog to PE-HOME viewpoint, press 'j' (capture custom_home_position_pe).
     Use this same viewpoint for the PE step below.
  4. Jog the gripper down to the physical grasp pose on the lid (touching
     the lid the way the demo will grasp it). Press 'g' (capture
     grasp_position_ground_truth + grasp_orientation_ground_truth_euler).
  5. With the lid still in place, lift back to the PE-home viewpoint and:
       press 'i' to capture an RGB+depth frame, then
       press 'p' to run FoundationPose. The print-block will include
       demo_w_D_w_o and demo_t_D_t_o.
     Repeat 'i'+'p' at a few brick yaws to sanity-check ≤2 mm drift.
  6. Jog to the goal (insertion target) pose. Press 'f' (capture
     goal_position_ground_truth + goal_orientation_ground_truth_euler).
  7. Jog along the desired post-grasp trajectory. At each waypoint press
     'v' (appends to waypoints_after_grasp; rotations stored as delta from
     the grasp orientation captured in step 4, matching the wrapper's
     convention). 'z' undoes the last waypoint if you misclick.
  8. Jog to the dropoff (failure / abort) pose, press 'b' (capture
     dropoff_point).
  9. Press 'P' to print the full SiemensConfig paste-block. Paste it into
     crisp_drl/agents/shared/insertion_env_config.py (SiemensConfig).

Translation keys (mm per press, adjustable with +/-):
  w/s  — +/-X     a/d  — +/-Y     q/e  — +/-Z

Rotation keys:
  y/x  — +/-Ry    k/l  — +/-Rz    m/n  — +/-Rx
  Y    — +3° about Y in one shot (matches waypoint[1] tilt)

Capture keys (record current pose into snapshot):
  g  — grasp_position_ground_truth + grasp_orientation_ground_truth_euler
  f  — goal_position_ground_truth  + goal_orientation_ground_truth_euler
  h  — custom_home_position        (joints)
  j  — custom_home_position_pe     (joints)
  u  — custom_first_home_position  (joints)
  v  — append current TCP as waypoint_after_grasp (rot delta from grasp)
  b  — dropoff_point (XY+Z)
  z  — undo last appended waypoint
  P  — print full SiemensConfig paste-block

Other:
  o/c  — gripper open/close
  r    — print TCP + joint state
  t    — servo to REFINED_PE_HOVER
  i    — capture RGB+depth image (lid on table, no grasp)
  p    — run PE + print demo_w_D_w_o / demo_t_D_t_o
  Enter — quit
"""

import tifffile
import numpy as np
from pynput import keyboard
from crisp_gym.envs.manipulator_env import make_env
from crisp_drl.envs.pose_estimation_helper import PoseEstimationHelper
from crisp_drl.agents.shared.insertion_env_config import SiemensConfig

# XYZ the runtime system hovers at for refined PE.
# = [wide_pe_siemens_X, wide_pe_siemens_Y, grasp_z + 0.015]
# Update from the "wide PE siemens pos" line in a real run.
_cfg = SiemensConfig()
GRASP_POS_REF = np.array(_cfg.grasp_position_ground_truth, dtype=np.float64)
REFINED_PE_HOVER = np.array(
    [GRASP_POS_REF[0], GRASP_POS_REF[1], GRASP_POS_REF[2] + 0.015]
)

env = make_env("my_env_v4")
print("Env created.")
env.wait_until_ready()
print("Env ready.")
env.gripper.open()
# env.home()
env.reset()
print("Env reset.")

exp_name = "siemens_calib"
rot_deg = 1.0
rot_deg_z = 0.25
rot_deg_x = 0.25
step_mm = 3.0  # translation step size in mm

# Procedure B: use SiemensConfig.grasp_position_ground_truth directly as the
# grasp TCP. Keeps the policy's coordinate system anchored at the data-collection
# reference.
print(f"[calib] Using SiemensConfig.grasp_position_ground_truth = {GRASP_POS_REF}")
print(f"[calib] REFINED_PE_HOVER = {REFINED_PE_HOVER}")

i_demo_img = 0
last_pe_tcp = None
last_color_path = None
last_depth_path = None

# Snapshot of captured SiemensConfig fields. Filled by g/f/h/j/u/v/b keys.
# Printed by 'P'.
snapshot: dict = {
    "grasp_pos": None,
    "grasp_euler": None,
    "goal_pos": None,
    "goal_euler": None,
    "home_joints": None,
    "home_pe_joints": None,
    "first_home_joints": None,
    "waypoints": [],  # list of ([x,y,z,rx_rel,ry_rel,rz_rel], tol)
    "dropoff": None,
    "demo_w_D_w_o": None,  # filled by run_pe()
    "demo_t_D_t_o": None,  # filled by run_pe()
}

# Default tolerance (m) used when appending a waypoint via 'v'.
# Edit per-row in the printed paste-block if you want tighter/looser tols.
WAYPOINT_TOL_DEFAULT = 0.001


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
        brick_size="2x2",
        use_tracker=False,
    )

    print("[PE] Running SAM3 segmentation ('black cover with circular grille') ...")
    mask = pe.segmenter.segment_siemens(image)
    print(f"[PE] Mask coverage: {int(mask.sum() / 255)} px")

    print("[PE] Running FoundationPose (requires Isaac ROS) ...")
    cam_pose = pe.pose_estimator.estimate_siemens(image, depth_m, mask)
    print(f"[PE] Camera-frame pose:\n{cam_pose}")

    world_pose = pe._compute_pose_in_world_frame(cam_pose, last_pe_tcp)
    print(f"[PE] World-frame pose:\n{world_pose}")

    obj_R = world_pose[:3, :3]
    obj_t = world_pose[:3, 3]
    grasp_pos = GRASP_POS_REF  # Procedure B: use Config reference, not a measured TCP
    o_T_o = obj_R.T @ (grasp_pos - obj_t)

    print(f"[PE] grasp_position_ground_truth (from Config): {grasp_pos}")
    print(f"[PE] siemens centroid in world:                 {obj_t}")
    print(f"[PE] grasp_pos - obj_t (world frame):           {np.round((grasp_pos - obj_t) * 1000, 2)} mm")
    print(f"[PE] o_T_o_tcpgrasp_siemens (object frame):     {np.round(o_T_o * 1000, 2)} mm")

    R, t = world_pose[:3, :3], world_pose[:3, 3]
    print(f"""
=== SiemensConfig.grasp_position_ground_truth is UNCHANGED (kept at training reference) ===

=== Paste into SiemensConfig in crisp_drl/agents/shared/insertion_env_config.py ===

    demo_w_D_w_o: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [{R[0,0]:.8f}, {R[0,1]:.8f}, {R[0,2]:.8f}, {t[0]:.8f}],
                [{R[1,0]:.8f}, {R[1,1]:.8f}, {R[1,2]:.8f}, {t[1]:.8f}],
                [{R[2,0]:.8f}, {R[2,1]:.8f}, {R[2,2]:.8f}, {t[2]:.8f}],
            ]
        )
    )

=== End paste ===

Sanity check — the wrapper will compute
  o_T_o = demo_w_D_w_o[:3,:3].T @ (grasp_pos - demo_w_D_w_o[:3,3])
at runtime (see insertion_wrapper_s.py), so verify these magnitudes look
right (a few mm xy, ~25 mm z is typical when grasped at the top):
    grasp_pos - obj_t = {np.round((grasp_pos - t) * 1000, 2)} mm  (world frame)
    o_T_o_tcpgrasp_siemens = {np.round(o_T_o * 1000, 2)} mm  (object frame)
Repeat the i+p capture a few times with the lid at different yaws — the
object-frame o_T_o should stay stable to within ~2 mm. Large drift means PE
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
    elif c == "Y":
        # +3° about Y in one shot (matches the +3° tilt baked into
        # waypoints_after_grasp[1] in SiemensConfig).
        env.step(np.array([0, 0, 0, 0, np.deg2rad(3.0), 0, 0]))
        print("[rot] applied +3° about Y")
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
        tcp = np.asarray(obs["observation.state.cartesian"], dtype=np.float64)
        joints = np.asarray(obs.get("observation.state.joints"), dtype=np.float64)
        print(f"[state] cartesian (x y z rx ry rz) = {tcp}")
        if joints.ndim == 1 and joints.size >= 7:
            print(f"[state] joints (q1..q{joints.size})    = {joints}")
            # Paste-block for SiemensConfig home configs (custom_first_home_position /
            # custom_home_position / custom_home_position_pe — all 7 joint values, rad).
            print(
                "    np.array(\n"
                "        [\n"
                + "".join(f"            {q:.8f},\n" for q in joints[:7])
                + "        ]\n"
                "    )"
            )
        else:
            print("[state] joints obs not available (key 'observation.state.joints' missing)")
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
    f"  y/x/k/l/m/n = rotate   Y = +3° about Y (one-shot)\n"
    f"  o/c = gripper   r = print state\n"
    f"  t = servo to REFINED_PE_HOVER {REFINED_PE_HOVER}\n"
    f"  i = capture image (lid on table)   p = run PE + print paste-block\n"
    f"\nWorkflow (Procedure B, no grasp): t → i → p   "
    f"(repeat i+p with different lid yaws to sanity-check stability)\n"
)

input()

listener.stop()
env.close()
