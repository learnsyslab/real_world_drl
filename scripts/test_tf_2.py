import json
import numpy as np
import pyvista as pv


def r_x(phi):
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)],
        ]
    )


def r_y(phi):
    return np.array(
        [
            [np.cos(phi), 0, np.sin(phi)],
            [0, 1, 0],
            [-np.sin(phi), 0, np.cos(phi)],
        ]
    )


def r_z(phi):
    return np.array(
        [
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi), np.cos(phi), 0],
            [0, 0, 1],
        ]
    )


r_z_negative_pi_2 = np.array(
    [
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ]
)


def euler_to_rot_matrix(roll, pitch, yaw):
    return r_z(yaw) @ r_y(pitch) @ r_x(roll)


cam_R_tcp = np.eye(3)
tcp_T_tcp_cam = np.zeros(3)
# offset to cam center
tcp_T_tcp_cam += np.array(
    ##[0.0824748 - 0.0013, 0.0, 0.1034 - 0.0095955 - 0.035 + 0.04]
    [0.0824748 - 0.0013, 0.0, -0.1034 + 0.0095955]
)  # ? -35mm-z for camera adapter? + ? for aloha gripper instead of franka
# aloha tcp is 1.3mm in front (x) of gripper center
# rotate around Y by 155 degrees
##Ry = r_y(np.deg2rad(155))
Ry = r_y(np.deg2rad(25))
cam_R_tcp = Ry @ cam_R_tcp
##Rz = r_z(np.deg2rad(90))
Rz = r_z(np.deg2rad(-90))
cam_R_tcp = Rz @ cam_R_tcp
# # offset cam center to pinhole
tcp_T_tcp_cam += cam_R_tcp.T @ np.array([-0.009, 0.0, -0.0037])
# # camera has flipped x-axis
# cam_R_tcp[0, :] *= -1
tcp_R_cam = np.linalg.inv(cam_R_tcp)


# goal: worl_T_world_obj = world_T_world_tcp + world_R_tcp @ tcp_T_tcp_cam + world_R_tcp @ tcp_R_cam @ cam_T_cam_obj
# 1) load poses
estimated_poses_grp_cam_path = "/home/linusschwarz/workspaces/isaac_ros-dev/own_samples/1/fpt_grp_d0.05-1.0_r5_up_prior_p4.json"
with open(estimated_poses_grp_cam_path, "r") as f:
    estimated_poses_grp_cam = [np.array(x) for x in json.load(f)]
estimated_poses_plc_cam_path = "/home/linusschwarz/workspaces/isaac_ros-dev/own_samples/1/fpt_plc_d0.05-1.0_r5_up_prior_p4.json"
with open(estimated_poses_plc_cam_path, "r") as f:
    estimated_poses_plc_cam = [np.array(x) for x in json.load(f)]
tcp_in_world_poses_path = "/home/linusschwarz/workspaces/isaac_ros-dev/own_samples/1/positions.json"  # xzy, euler-xyz radians
with open(tcp_in_world_poses_path, "r") as f:
    tcp_in_world_poses_euler = json.load(f)
obj_path = (
    "/home/linusschwarz/workspaces/isaac_ros-dev/lego_assets/lego_2x2_lavender_up.obj"
)
mesh_scale = 1.0  # scale the loaded mesh if it's too large/small
base_mesh = pv.read(obj_path)
base_mesh.points = base_mesh.points * mesh_scale

world_T_world_tcp_list = [np.array(p[:3]) for p in tcp_in_world_poses_euler]
world_R_tcp_list = [
    euler_to_rot_matrix(p[3], p[4], p[5]) for p in tcp_in_world_poses_euler
]

cam_T_cam_grp_list = [np.array(pose[:3, 3]) for pose in estimated_poses_grp_cam]
cam_R_grp_list = [np.array(pose[:3, :3]) for pose in estimated_poses_grp_cam]

cam_T_cam_plc_list = [np.array(pose[:3, 3]) for pose in estimated_poses_plc_cam]
cam_R_plc_list = [np.array(pose[:3, :3]) for pose in estimated_poses_plc_cam]

subset_len = 80
for subset_start in [0]:
    plotter = pv.Plotter()
    mean_dir = np.zeros(3)
    for i in range(
        len(estimated_poses_grp_cam[subset_start : subset_start + subset_len])
    ):
        world_T_world_tcp = world_T_world_tcp_list[i]
        world_R_tcp = world_R_tcp_list[i]
        cam_T_cam_grp = cam_T_cam_grp_list[i]
        cam_R_grp = cam_R_grp_list[i]
        cam_T_cam_plc = cam_T_cam_plc_list[i]
        cam_R_plc = cam_R_plc_list[i]

        def plot_axes(R, t):
            arrow_x = pv.Arrow(
                start=t,
                direction=R[:, 0],
                scale=0.01 * mesh_scale,
            )
            plotter.add_mesh(arrow_x, color="red")
            arrow_y = pv.Arrow(
                start=t,
                direction=R[:, 1],
                scale=0.01 * mesh_scale,
            )
            plotter.add_mesh(arrow_y, color="green")
            arrow_z = pv.Arrow(
                start=t,
                direction=R[:, 2],
                scale=0.01 * mesh_scale,
            )
            plotter.add_mesh(arrow_z, color="blue")

        def make_mesh(cam_T_cam_obj, cam_R_obj, color, axes=False):
            # compute world_T_world_obj
            world_T_world_obj = (
                world_T_world_tcp
                + world_R_tcp @ tcp_T_tcp_cam
                + world_R_tcp @ tcp_R_cam @ cam_T_cam_obj
            )
            world_R_obj = world_R_tcp @ tcp_R_cam @ cam_R_obj

            obj_pose_world = np.eye(4)
            obj_pose_world[:3, :3] = world_R_obj
            obj_pose_world[:3, 3] = world_T_world_obj

            mesh_obj = base_mesh.copy(deep=True)
            R = obj_pose_world[:3, :3]
            t = obj_pose_world[:3, 3]
            mesh_obj.points = (R @ mesh_obj.points.T).T + t

            plotter.add_mesh(mesh_obj, opacity=0.6, color=color)
            if axes:
                plot_axes(world_R_obj, world_T_world_obj)
            return world_R_obj, world_T_world_obj

        _, world_T_world_grp = make_mesh(cam_T_cam_grp, cam_R_grp, "red")
        _, world_T_world_plc = make_mesh(cam_T_cam_plc, cam_R_plc, "blue", axes=True)

        # also plot small spheres for tcp positions and 3 arrows for tcp frame
        sphere = pv.Sphere(radius=0.001 * mesh_scale, center=world_T_world_tcp)
        plotter.add_mesh(sphere, color="black")

        plot_axes(world_R_tcp, world_T_world_tcp)

        world_T_plc_grp = world_T_world_grp - world_T_world_plc
        # print(f"world_T_plc_grp: {world_T_plc_grp}")

        # dir_vec = (
        #     world_T_world_tcp - world_T_world_obj
        # )  # vector from tcp to obj in world frame
        # mean_dir += dir_vec

    # compute mean vector from tcp positions to obj positions to indicate direction
    mean_dir /= subset_len
    print(f"mean direction vector tcp->obj: {mean_dir}")

    # also show origin
    m_origin = base_mesh.copy(deep=True)
    plotter.add_mesh(m_origin, opacity=0.3, color="gray")
    arrow = pv.Arrow(start=[0, 0, 0], direction=[0, 0, 1], scale=0.01 * mesh_scale)
    plotter.add_mesh(arrow, color="gray")

    plotter.add_axes()  # type: ignore
    plotter.show()
