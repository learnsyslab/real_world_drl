import os
from typing import Dict, cast

import numpy as np
import pyvista as pv

MESH_FILE_NAME = "_centered.obj"
ESTIMATED_POSES_FILE = "test_images/gear_world_estimates.txt"
MESH_PATH_LOCAL = os.path.expanduser(f"~/repos/foundation_pose_meshes/{MESH_FILE_NAME}")


def load_estimated_poses(path: str) -> Dict[int, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Estimated poses file not found: {path}")

    data = np.loadtxt(path, comments="#")
    if data.size == 0:
        return {}
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != 17:
        raise ValueError(
            f"Expected 17 columns in {path} (idx + 16 pose values), got {data.shape[1]}"
        )

    poses_world: Dict[int, np.ndarray] = {}
    for row in data:
        idx = int(row[0])
        w_pose = row[1:].reshape(4, 4).astype(np.float32)
        poses_world[idx] = w_pose
    return poses_world


def show_3d(poses_world: Dict[int, np.ndarray], mesh_path: str) -> None:
    plotter = pv.Plotter()
    base_mesh = pv.read(mesh_path)
    base_mesh.points = base_mesh.points.copy()

    n = len(poses_world)
    cmap = pv.LookupTable("coolwarm", n_values=max(n, 2))

    for i, (idx, w_pose) in enumerate(sorted(poses_world.items())):
        R = w_pose[:3, :3]
        t = w_pose[:3, 3]
        mesh_obj = cast(pv.DataSet, base_mesh.copy(deep=True))
        mesh_obj.points = (R @ mesh_obj.points.T).T + t

        color = cmap.map_value(i / max(n - 1, 1))[:3]
        color_f = [c for c in color]

        plotter.add_mesh(  # pyright: ignore[reportArgumentType]
            mesh_obj, opacity=0.7, color=color_f, label=f"img {idx}"
        )
        plotter.add_mesh(pv.Arrow(start=t, direction=R[:, 0], scale=0.01), color="red")
        plotter.add_mesh(
            pv.Arrow(start=t, direction=R[:, 1], scale=0.01), color="green"
        )
        plotter.add_mesh(pv.Arrow(start=t, direction=R[:, 2], scale=0.01), color="blue")

    plotter.add_axes()  # pyright: ignore[reportCallIssue]
    plotter.add_legend()  # pyright: ignore[reportCallIssue]
    plotter.show()


def main() -> None:
    poses_world = load_estimated_poses(ESTIMATED_POSES_FILE)
    if not poses_world:
        print("No estimated poses loaded from", ESTIMATED_POSES_FILE)
        return

    print(f"Loaded {len(poses_world)} estimated poses from {ESTIMATED_POSES_FILE}")
    show_3d(poses_world, MESH_PATH_LOCAL)


if __name__ == "__main__":
    main()
