"""Per-reset pose-estimation visualization + raw-artifact dump.

Renders the estimated object pose on top of the wrist-camera RGB using
OpenCV projection (no external 3D renderer). Intended for offline inspection
of FoundationPose quality per episode.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional
from datetime import datetime

import cv2
import numpy as np


DEFAULT_MESH_PATH = (
    "/workspaces/isaac_ros-dev/lego_assets/SiemensLid_centered.obj"
)
DEFAULT_MESH_PATH_HOST_FALLBACK = (
    os.path.expanduser("~/workspaces/isaac_ros-dev/lego_assets/SiemensLid_centered.obj")
)


def _load_obj_vertices(path: str) -> np.ndarray:
    """Parse `v x y z` lines from a Wavefront .obj. Returns (N, 3) float32."""
    verts = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not verts:
        raise RuntimeError(f"No vertices parsed from {path}")
    return np.asarray(verts, dtype=np.float32)


def _load_intrinsics(camera_info_json_path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(camera_info_json_path) as f:
        data = json.load(f)
    K = np.asarray(data["k"], dtype=np.float64).reshape(3, 3)
    D = np.asarray(data["d"], dtype=np.float64).reshape(-1)
    return K, D


def _project_points(
    points_obj: np.ndarray,
    pose_cam_obj: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project obj-frame points to pixel coords given pose^cam_obj (4x4).

    Returns (uv, in_front) where uv is (N, 2) float32 pixel coords and
    in_front is (N,) bool for z > 0 in camera frame.
    """
    R = pose_cam_obj[:3, :3]
    t = pose_cam_obj[:3, 3]
    points_cam = (R @ points_obj.T + t[:, None]).T  # (N, 3)
    in_front = points_cam[:, 2] > 1e-4
    rvec = cv2.Rodrigues(np.eye(3))[0]
    tvec = np.zeros((3, 1), dtype=np.float64)
    uv, _ = cv2.projectPoints(
        points_cam.astype(np.float64), rvec, tvec, K.astype(np.float64), D
    )
    return uv.reshape(-1, 2).astype(np.float32), in_front


class PoseOverlayRenderer:
    """Overlay estimated object pose on wrist-camera RGB.

    Draws (optionally) a sparse vertex silhouette of the mesh at the estimated
    pose, the SAM3 mask contour, a projected 3D bounding box, and an XYZ
    axis triad rooted at the pose origin.
    """

    def __init__(
        self,
        camera_info_json_path: str,
        mesh_path: Optional[str] = None,
        axis_length_m: float = 0.03,
        vertex_stride: int = 20,
    ):
        self.K, self.D = _load_intrinsics(camera_info_json_path)
        self.axis_length_m = axis_length_m
        self.vertex_stride = max(1, int(vertex_stride))
        self.mesh_verts: Optional[np.ndarray] = None
        self.bbox_corners: Optional[np.ndarray] = None

        resolved_mesh = None
        for p in (mesh_path, DEFAULT_MESH_PATH, DEFAULT_MESH_PATH_HOST_FALLBACK):
            if p and Path(p).exists():
                resolved_mesh = p
                break
        if resolved_mesh is None:
            print(
                "[PoseOverlayRenderer] mesh not found; overlay will show axes + "
                "mask contour only. Tried:",
                mesh_path,
                DEFAULT_MESH_PATH,
                DEFAULT_MESH_PATH_HOST_FALLBACK,
            )
            return

        verts = _load_obj_vertices(resolved_mesh)
        self.mesh_verts = verts
        mn = verts.min(axis=0)
        mx = verts.max(axis=0)
        self.bbox_corners = np.array(
            [
                [mn[0], mn[1], mn[2]],
                [mx[0], mn[1], mn[2]],
                [mx[0], mx[1], mn[2]],
                [mn[0], mx[1], mn[2]],
                [mn[0], mn[1], mx[2]],
                [mx[0], mn[1], mx[2]],
                [mx[0], mx[1], mx[2]],
                [mn[0], mx[1], mx[2]],
            ],
            dtype=np.float32,
        )
        print(
            f"[PoseOverlayRenderer] loaded mesh from {resolved_mesh} "
            f"({len(verts)} verts, stride={self.vertex_stride})"
        )

    def render(
        self,
        rgb: np.ndarray,
        pose_cam_obj: np.ndarray,
        mask: Optional[np.ndarray] = None,
        label: str = "",
        silhouette_color: tuple[int, int, int] = (255, 150, 0),
    ) -> np.ndarray:
        """Return a BGR image with the overlay drawn on top of `rgb`."""
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected HxWx3 RGB, got {rgb.shape}")
        canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        h, w = canvas.shape[:2]

        # mask contour (green)
        if mask is not None and mask.size > 0:
            m8 = (mask > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(canvas, contours, -1, (0, 255, 0), 1)

        # vertex silhouette
        if self.mesh_verts is not None:
            sub = self.mesh_verts[:: self.vertex_stride]
            uv, in_front = _project_points(sub, pose_cam_obj, self.K, self.D)
            overlay = canvas.copy()
            for (u, v), ok in zip(uv, in_front):
                if not ok:
                    continue
                iu, iv = int(round(u)), int(round(v))
                if 0 <= iu < w and 0 <= iv < h:
                    overlay[iv, iu] = silhouette_color
            canvas = cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0)

        # projected bounding box edges
        if self.bbox_corners is not None:
            uv, in_front = _project_points(
                self.bbox_corners, pose_cam_obj, self.K, self.D
            )
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ]
            for a, b in edges:
                if in_front[a] and in_front[b]:
                    pa = tuple(int(x) for x in uv[a])
                    pb = tuple(int(x) for x in uv[b])
                    cv2.line(canvas, pa, pb, silhouette_color, 1, cv2.LINE_AA)

        # XYZ axes at pose origin
        axis_pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [self.axis_length_m, 0.0, 0.0],
                [0.0, self.axis_length_m, 0.0],
                [0.0, 0.0, self.axis_length_m],
            ],
            dtype=np.float32,
        )
        uv_ax, in_front_ax = _project_points(axis_pts, pose_cam_obj, self.K, self.D)
        if in_front_ax[0]:
            origin = tuple(int(x) for x in uv_ax[0])
            for idx, color in (
                (1, (0, 0, 255)),    # X red
                (2, (0, 255, 0)),    # Y green
                (3, (255, 0, 0)),    # Z blue
            ):
                if in_front_ax[idx]:
                    tip = tuple(int(x) for x in uv_ax[idx])
                    cv2.line(canvas, origin, tip, color, 2, cv2.LINE_AA)

        if label:
            cv2.putText(
                canvas,
                label,
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                label,
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        return canvas

    def save_pair(
        self,
        out_dir: str,
        episode_idx: int,
        rgb: np.ndarray,
        mask: Optional[np.ndarray],
        pose_cam_coarse: np.ndarray,
        pose_cam_refined: Optional[np.ndarray] = None,
        rgb_refined: Optional[np.ndarray] = None,
        mask_refined: Optional[np.ndarray] = None,
    ) -> dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        paths: dict[str, str] = {}
        timestamp = datetime.now().strftime("%m-%d-%H-%M")
        coarse_img = self.render(rgb, pose_cam_coarse, mask=mask, label=f"ep{episode_idx} coarse")
        p_coarse = os.path.join(
            out_dir, f"ep{episode_idx:03d}_coarse_{timestamp}.png"
        )
        cv2.imwrite(p_coarse, coarse_img)
        paths["coarse"] = p_coarse

        if pose_cam_refined is not None:
            refined_rgb = rgb_refined if rgb_refined is not None else rgb
            refined_mask = mask_refined if mask_refined is not None else mask
            refined_img = self.render(
                refined_rgb, pose_cam_refined, mask=refined_mask, label=f"ep{episode_idx} refined"
            )
            p_refined = os.path.join(
                out_dir, f"ep{episode_idx:03d}_refined_{timestamp}.png"
            )
            cv2.imwrite(p_refined, refined_img)
            paths["refined"] = p_refined

            if refined_img.shape == coarse_img.shape:
                side = np.concatenate([coarse_img, refined_img], axis=1)
                p_side = os.path.join(
                    out_dir, f"ep{episode_idx:03d}_sidebyside_{timestamp}.png"
                )
                cv2.imwrite(p_side, side)
                paths["sidebyside"] = p_side

        return paths

    def save_single(
        self,
        out_dir: str,
        episode_idx: int,
        rgb: np.ndarray,
        pose_cam_obj: np.ndarray,
        mask: Optional[np.ndarray] = None,
        suffix: str = "single",
        label: Optional[str] = None,
    ) -> str:
        os.makedirs(out_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%m-%d-%H-%M")
        draw_label = label if label is not None else f"ep{episode_idx} {suffix}"
        img = self.render(rgb, pose_cam_obj, mask=mask, label=draw_label)
        path = os.path.join(
            out_dir, f"ep{episode_idx:03d}_{suffix}_{timestamp}.png"
        )
        cv2.imwrite(path, img)
        return path

    def save_rgb(
        self,
        out_dir: str,
        episode_idx: int,
        rgb: np.ndarray,
        suffix: str = "rgb",
    ) -> str:
        os.makedirs(out_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%m-%d-%H-%M")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        path = os.path.join(
            out_dir, f"ep{episode_idx:03d}_{suffix}_{timestamp}.png"
        )
        cv2.imwrite(path, bgr)
        return path


def dump_raw_npz(out_dir: str, episode_idx: int, **arrays) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ep{episode_idx:03d}_raw.npz")
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items() if v is not None})
    return path
