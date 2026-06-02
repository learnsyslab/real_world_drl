from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Optional

import cv2  # type: ignore[import-not-found]
import numpy as np
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def r_x(phi: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(phi), -np.sin(phi)],
            [0.0, np.sin(phi), np.cos(phi)],
        ],
        dtype=np.float32,
    )


def r_y(phi: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(phi), 0.0, np.sin(phi)],
            [0.0, 1.0, 0.0],
            [-np.sin(phi), 0.0, np.cos(phi)],
        ],
        dtype=np.float32,
    )


def r_z(phi: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(phi), -np.sin(phi), 0.0],
            [np.sin(phi), np.cos(phi), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def euler_xyz_to_rot_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return r_z(yaw) @ r_y(pitch) @ r_x(roll)


def get_t_R_c_and_t_T_t_c() -> tuple[np.ndarray, np.ndarray]:
    c_R_t = np.eye(3, dtype=np.float32)
    t_T_t_c = np.array(
        [0.0824748 - 0.0013, 0.0 - 0.0047, -0.1034 + 0.0095955],
        dtype=np.float32,
    )

    c_R_t = r_y(np.deg2rad(25.0)) @ c_R_t
    c_R_t = r_z(np.deg2rad(-90.0)) @ c_R_t

    c_T_c_pinhole = np.array([-0.009, 0.0, -0.0037], dtype=np.float32)
    t_T_t_c = t_T_t_c + c_R_t.T @ c_T_c_pinhole

    t_R_c = c_R_t.T
    return t_R_c, t_T_t_c


def _default_bpe_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "sam3"
        / "sam3"
        / "assets"
        / "bpe_simple_vocab_16e6.txt.gz"
    )


def load_camera_matrix(camera_json_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with camera_json_path.open("r", encoding="utf-8") as f:
        params = json.load(f)

    if "k" not in params or len(params["k"]) != 9:
        raise ValueError(
            f"Camera JSON at {camera_json_path} must contain a flattened 3x3 'k' matrix"
        )

    camera_matrix = np.asarray(params["k"], dtype=np.float32).reshape(3, 3)
    dist_coeffs = np.asarray(params.get("d", []), dtype=np.float32).reshape(-1, 1)
    if dist_coeffs.size == 0:
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    return camera_matrix, dist_coeffs


def depth_to_meters(depth_map: np.ndarray) -> np.ndarray:
    depth_m = depth_map.astype(np.float32)
    finite_depth = depth_m[np.isfinite(depth_m)]
    if finite_depth.size > 0 and float(np.nanmax(finite_depth)) > 10.0:
        depth_m = depth_m / 1000.0
    return depth_m


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return vector / norm


def _unpack_rgbd(rgbd: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(rgbd, dict):
        image = rgbd.get("rgb")
        if image is None:
            image = rgbd.get("image")
        if image is None:
            image = rgbd.get("color")
        depth = rgbd.get("depth")
        if image is None or depth is None:
            raise ValueError("rgbd dict must contain image/rgb/color and depth entries")
        return np.asarray(image), np.asarray(depth)

    if isinstance(rgbd, (tuple, list)) and len(rgbd) == 2:
        return np.asarray(rgbd[0]), np.asarray(rgbd[1])

    raise TypeError("rgbd must be a (rgb, depth) pair or a dict with image and depth")


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def extract_rectangle_corners(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("No contour found in mask.")

    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx_corners = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx_corners) != 4:
        rect = cv2.minAreaRect(largest_contour)
        approx_corners = cv2.boxPoints(rect).reshape(-1, 1, 2)

    img_pts = approx_corners.reshape(4, 2).astype(np.float32)
    return order_points(img_pts)


def build_object_points(width_m: float, height_m: float) -> np.ndarray:
    return np.array(
        [
            [-width_m / 2.0, -height_m / 2.0, 0.0],
            [width_m / 2.0, -height_m / 2.0, 0.0],
            [width_m / 2.0, height_m / 2.0, 0.0],
            [-width_m / 2.0, height_m / 2.0, 0.0],
        ],
        dtype=np.float32,
    )


def median_mask_depth(depth_map: np.ndarray, mask: np.ndarray) -> float:
    valid_depths = depth_map[mask > 0]
    valid_depths = valid_depths[valid_depths > 0]
    if valid_depths.size == 0:
        raise ValueError("No valid depth values found inside the mask.")
    return float(np.median(valid_depths).astype(np.float32))


def masked_points_3d(
    depth_map_m: np.ndarray,
    mask: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    if depth_map_m.shape != mask.shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match depth shape {depth_map_m.shape}"
        )

    valid = mask & (depth_map_m > 0)
    pixel_y, pixel_x = np.nonzero(valid)
    if pixel_x.size == 0:
        raise ValueError("No valid masked pixels with positive depth were found")

    depth_values = depth_map_m[pixel_y, pixel_x]
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    x = (pixel_x.astype(np.float32) - cx) * depth_values / fx
    y = (pixel_y.astype(np.float32) - cy) * depth_values / fy
    z = depth_values
    return np.stack([x, y, z], axis=1)


def centroid_from_mask(
    depth_map_m: np.ndarray,
    mask: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    points_3d = masked_points_3d(depth_map_m, mask, camera_matrix)
    return np.mean(points_3d, axis=0)


def centroid_from_mean_mask_pixel_and_depth(
    depth_map_m: np.ndarray,
    mask: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    valid = mask & (depth_map_m > 0)
    pixel_y, pixel_x = np.nonzero(valid)
    if pixel_x.size == 0:
        raise ValueError("No valid masked pixels with positive depth were found")

    mean_x = float(np.mean(pixel_x.astype(np.float32)))
    mean_y = float(np.mean(pixel_y.astype(np.float32)))
    mean_depth = float(np.mean(depth_map_m[pixel_y, pixel_x].astype(np.float32)))

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    x = (mean_x - cx) * mean_depth / fx
    y = (mean_y - cy) * mean_depth / fy
    z = mean_depth
    return np.array([x, y, z], dtype=np.float32)


def compute_centroid(
    depth_map_m: np.ndarray,
    mask: np.ndarray,
    camera_matrix: np.ndarray,
    method: str,
) -> np.ndarray:
    if method == "point_cloud_mean":
        return centroid_from_mask(depth_map_m, mask, camera_matrix)
    if method == "mean_pixel_depth":
        return centroid_from_mean_mask_pixel_and_depth(depth_map_m, mask, camera_matrix)
    raise ValueError(f"Unknown centroid method: {method}")


def select_masks(
    output: dict,
    max_masks: int = 3,
    score_drop_threshold: float = 0.2,
    containment_mask: Optional[np.ndarray] = None,
) -> tuple[list[np.ndarray], list[float]]:
    masks = output["masks"]
    scores = output["scores"]

    if len(masks) == 0:
        raise ValueError("No masks found")

    scores_np = scores.cpu().numpy()
    ranked_indices = np.argsort(scores_np)[::-1]
    if ranked_indices.size == 0:
        return [], []

    best_score = float(scores_np[ranked_indices[0]])

    containment_binary: Optional[np.ndarray] = None
    if containment_mask is not None:
        containment_binary = (np.asarray(containment_mask) > 0).astype(bool)
        # validate containment shape if possible
        h = int(masks[0].shape[-2])
        w = int(masks[0].shape[-1])
        if containment_binary.shape != (h, w):
            raise ValueError(
                f"containment_mask shape {containment_binary.shape} does not match mask shape {(h, w)}"
            )

    candidate_indices: list[int] = []
    for idx in ranked_indices:
        if len(candidate_indices) >= max_masks:
            break

        score = float(scores_np[idx])
        if best_score - score > score_drop_threshold:
            break

        if containment_binary is not None:
            mask_np = (
                masks[int(idx)]
                .cpu()
                .numpy()
                .reshape(masks[0].shape[-2], masks[0].shape[-1])
            )
            mask_binary = mask_np > 0
            if np.any(mask_binary & ~containment_binary):
                # not fully contained, skip
                continue

        candidate_indices.append(int(idx))

    print(
        f"Found {len(candidate_indices)} masks with scores: {scores_np[candidate_indices].tolist()}"
    )

    selected_masks = [
        masks[idx].cpu().numpy().reshape(masks[0].shape[-2], masks[0].shape[-1]) * 255
        for idx in candidate_indices
    ]
    selected_scores = [float(scores_np[idx]) for idx in candidate_indices]
    return selected_masks, selected_scores


def select_best_mask(output: dict) -> np.ndarray:
    masks = output.get("masks", [])
    scores = output["scores"]

    if len(masks) == 0:
        raise ValueError("No masks found")

    scores_np = scores.cpu().numpy()
    best_idx = int(np.argmax(scores_np))
    print(
        f"Found {len(masks)} masks with top 5 scores: {scores_np[np.argsort(scores_np)[::-1][:5]].tolist()}"
    )
    return (
        masks[best_idx].cpu().numpy().reshape(masks[0].shape[-2], masks[0].shape[-1])
        * 255
    )


def run_sam3_masks(
    processor: Sam3Processor,
    image: np.ndarray,
    prompt: str,
    max_masks: int = 3,
    score_drop_threshold: float = 0.2,
    containment_mask: Optional[np.ndarray] = None,
) -> tuple[list[np.ndarray], list[float]]:
    inference_state = processor.set_image(Image.fromarray(image).convert("RGB"))
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
    return select_masks(
        output,
        max_masks=max_masks,
        score_drop_threshold=score_drop_threshold,
        containment_mask=containment_mask,
    )


def run_sam3_best_mask(
    processor: Sam3Processor,
    image: np.ndarray,
    prompt: str,
) -> np.ndarray:
    inference_state = processor.set_image(Image.fromarray(image).convert("RGB"))
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
    return select_best_mask(output)


def union_masks(masks: list[np.ndarray]) -> np.ndarray:
    if not masks:
        raise ValueError("No masks available")
    return np.any(np.stack([mask > 0 for mask in masks], axis=0), axis=0)


def build_coordinate_frame(
    best: dict, full_object_centroid_camera_m: np.ndarray
) -> dict:
    origin = best["tvec"].reshape(3).astype(np.float32)
    rotation = best["R"].astype(np.float32)
    short_axis_index = 0 if best["width_m"] <= best["height_m"] else 1

    z_axis = normalize_vector(rotation[:, 2])
    if z_axis[2] > 0:
        z_axis = -z_axis

    x_axis = rotation[:, short_axis_index]
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    x_axis = normalize_vector(x_axis)

    to_full_centroid = full_object_centroid_camera_m - origin
    to_full_centroid = to_full_centroid - np.dot(to_full_centroid, z_axis) * z_axis
    if np.dot(to_full_centroid, x_axis) < 0:
        x_axis = -x_axis

    y_axis = normalize_vector(np.cross(z_axis, x_axis))
    x_axis = normalize_vector(np.cross(y_axis, z_axis))

    return {
        "origin_camera_m": origin,
        "x_axis_camera": x_axis,
        "y_axis_camera": y_axis,
        "z_axis_camera": z_axis,
        "rotation_camera": np.column_stack([x_axis, y_axis, z_axis]),
    }


def solve_rectangle_pose(
    img_pts: np.ndarray,
    width_m: float,
    height_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    observed_z_m: float,
) -> dict:
    obj_pts = build_object_points(width_m, height_m)

    success, rvecs, tvecs, reproj_errors = cv2.solvePnPGeneric(
        obj_pts,
        img_pts,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not success or len(rvecs) == 0:
        raise ValueError("solvePnPGeneric failed to find a pose solution.")

    best = None
    for idx, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        projected, _ = cv2.projectPoints(
            obj_pts, rvec, tvec, camera_matrix, dist_coeffs
        )
        projected = projected.reshape(-1, 2)
        reproj_error = float(np.mean(np.linalg.norm(projected - img_pts, axis=1)))
        predicted_z = float(tvec[2, 0])
        z_error = abs(predicted_z - observed_z_m)
        score = reproj_error + z_error

        candidate = {
            "solution_index": idx,
            "rvec": rvec,
            "tvec": tvec,
            "R": cv2.Rodrigues(rvec)[0],
            "reprojection_error_px": reproj_error,
            "predicted_z_m": predicted_z,
            "observed_z_m": observed_z_m,
            "z_error_m": z_error,
            "score": score,
            "solver_reproj_error": float(reproj_errors[idx][0])
            if reproj_errors is not None and len(reproj_errors) > idx
            else None,
        }

        if best is None or candidate["score"] < best["score"]:
            best = candidate

    if best is None:
        raise RuntimeError("No valid pose candidate found.")

    return best


def to_world_pose(
    c_D_c_o: np.ndarray,
    w_R_t: np.ndarray,
    w_T_t: np.ndarray,
    t_R_c: np.ndarray,
    t_T_t_c: np.ndarray,
) -> np.ndarray:
    c_R_o = c_D_c_o[:3, :3]
    c_T_c_o = c_D_c_o[:3, 3]

    w_R_o = w_R_t @ t_R_c @ c_R_o
    w_T_w_o = w_T_t + w_R_t @ t_T_t_c + w_R_t @ t_R_c @ c_T_c_o

    w_D_w_o = np.eye(4, dtype=np.float32)
    w_D_w_o[:3, :3] = w_R_o
    w_D_w_o[:3, 3] = w_T_w_o
    return w_D_w_o


def to_tcp_pose(
    c_D_c_o: np.ndarray,
    t_R_c: np.ndarray,
    t_T_t_c: np.ndarray,
) -> np.ndarray:
    c_R_o = c_D_c_o[:3, :3]
    c_T_c_o = c_D_c_o[:3, 3]

    t_R_o = t_R_c @ c_R_o
    t_T_t_o = t_T_t_c + t_R_c @ c_T_c_o

    t_D_t_o = np.eye(4, dtype=np.float32)
    t_D_t_o[:3, :3] = t_R_o
    t_D_t_o[:3, 3] = t_T_t_o
    return t_D_t_o


def world_z_alignment_score(pose_world: np.ndarray) -> float:
    z_axis_world = normalize_vector(pose_world[:3, 2])
    return float(z_axis_world[2])


class PoseEstimationHelper:
    def __init__(
        self,
        camera_info_json_path: str = "camera_parameters/realsense_d405_single.json",
        sam3_prompt: str = "white rectangle with barcode",
        full_mask_prompt: str = "yellow cardboard box",
        width_m: float = 0.03,
        height_m: float = 0.04,
        centroid_method: str = "point_cloud_mean",
        confidence_threshold: float = 0.01,
        filter_depth_outliers: bool = False,
        bpe_path: str | Path | None = None,
        processor: Sam3Processor | None = None,
    ):
        camera_path = Path(camera_info_json_path)
        if not camera_path.exists():
            raise FileNotFoundError(f"Camera JSON not found: {camera_path}")

        self.camera_matrix, self.dist_coeffs = load_camera_matrix(camera_path)
        self.sam3_prompt = sam3_prompt
        self.full_mask_prompt = full_mask_prompt
        self.width_m = width_m
        self.height_m = height_m
        self.centroid_method = centroid_method
        self.filter_depth_outliers = filter_depth_outliers
        self.t_R_c, self.t_T_t_c = get_t_R_c_and_t_T_t_c()
        self.bpe_path = Path(bpe_path) if bpe_path is not None else _default_bpe_path()

        if processor is not None:
            # Use the provided processor (shared instance) and don't reload model
            self.processor = processor
        else:
            if not self.bpe_path.exists():
                raise FileNotFoundError(f"SAM3 BPE file not found: {self.bpe_path}")

            model = build_sam3_image_model(bpe_path=str(self.bpe_path))
            self.processor = Sam3Processor(
                model, confidence_threshold=confidence_threshold
            )

        self.last_mask: np.ndarray | None = None
        self.last_masks: list[np.ndarray] | None = None
        self.last_mask_scores: list[float] | None = None
        self.last_camera_poses: list[np.ndarray] | None = None
        self.last_full_mask: np.ndarray | None = None
        self.last_camera_pose: np.ndarray | None = None
        self.last_tcp_pose: np.ndarray | None = None
        self.last_tcp_poses: list[np.ndarray] | None = None

    def _maybe_filter_depth_outliers(
        self,
        depth_m: np.ndarray,
        mask: np.ndarray,
        mask_name: str,
    ) -> np.ndarray:
        if not self.filter_depth_outliers:
            return depth_m

        valid = mask & np.isfinite(depth_m) & (depth_m > 0)
        valid_depths = depth_m[valid]
        if valid_depths.size == 0:
            raise ValueError(f"No valid depth values found inside the {mask_name}.")

        median_depth = float(np.median(valid_depths))
        lower_bound = 0.75 * median_depth
        upper_bound = 1.25 * median_depth
        keep = valid & (depth_m >= lower_bound) & (depth_m <= upper_bound)

        valid_count = int(valid.sum())
        removed_count = valid_count - int(keep.sum())
        removed_fraction = removed_count / valid_count
        if removed_fraction > 0.25:
            warnings.warn(
                (
                    f"Removed {removed_count}/{valid_count} depth pixels "
                    f"({removed_fraction:.1%}) outside ±25% of the median for {mask_name}."
                ),
                RuntimeWarning,
                stacklevel=2,
            )

        filtered_depth = np.array(depth_m, copy=True)
        filtered_depth[valid & ~keep] = 0.0
        return filtered_depth

    def _estimate_camera_pose(
        self,
        image: np.ndarray,
        depth_m: np.ndarray,
        barcode_mask: np.ndarray,
        full_object_centroid_camera_m: np.ndarray,
    ) -> np.ndarray:
        barcode_mask = barcode_mask > 0
        filtered_depth_m = self._maybe_filter_depth_outliers(
            depth_m,
            barcode_mask,
            "barcode mask",
        )
        img_pts = extract_rectangle_corners(barcode_mask)
        observed_z_m = median_mask_depth(filtered_depth_m, barcode_mask)

        candidates = []
        for width_m, height_m, label in (
            (self.width_m, self.height_m, "width x height"),
            (self.height_m, self.width_m, "height x width"),
        ):
            candidate = solve_rectangle_pose(
                img_pts=img_pts,
                width_m=width_m,
                height_m=height_m,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                observed_z_m=observed_z_m,
            )
            candidate["label"] = label
            candidate["width_m"] = width_m
            candidate["height_m"] = height_m
            candidates.append(candidate)

        best = min(candidates, key=lambda item: item["score"])
        coordinate_frame = build_coordinate_frame(best, full_object_centroid_camera_m)

        pose_cam = np.eye(4, dtype=np.float32)
        pose_cam[:3, :3] = coordinate_frame["rotation_camera"]
        pose_cam[:3, 3] = coordinate_frame["origin_camera_m"]

        self.last_mask = barcode_mask
        self.last_camera_pose = pose_cam
        return pose_cam

    def _ee_pose_to_world_transform(
        self, ee_pose: np.ndarray | list[float] | tuple[float, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        ee_pose_arr = np.asarray(ee_pose, dtype=np.float32)
        if ee_pose_arr.shape == (4, 4):
            return ee_pose_arr[:3, :3], ee_pose_arr[:3, 3]

        if ee_pose_arr.ndim != 1 or ee_pose_arr.size < 6:
            raise ValueError(
                "ee_pose must be a 6D cartesian state or a 4x4 transform matrix"
            )

        w_T_t = ee_pose_arr[:3]
        roll, pitch, yaw = ee_pose_arr[3:6]
        w_R_t = euler_xyz_to_rot_matrix(roll, pitch, yaw)
        return w_R_t, w_T_t

    def estimate_barcodes(
        self,
        rgbd: Any,
        ee_pose: np.ndarray | list[float] | tuple[float, ...],
    ) -> list[np.ndarray]:
        image, depth = _unpack_rgbd(rgbd)
        depth_m = depth_to_meters(depth)
        full_mask = run_sam3_best_mask(self.processor, image, self.full_mask_prompt)
        barcode_masks, barcode_scores = run_sam3_masks(
            self.processor, image, self.sam3_prompt, containment_mask=full_mask
        )
        full_object_depth_m = self._maybe_filter_depth_outliers(
            depth_m,
            full_mask > 0,
            "full object mask",
        )
        full_object_centroid_camera_m = compute_centroid(
            full_object_depth_m,
            full_mask,
            self.camera_matrix,
            self.centroid_method,
        )

        pose_records: list[dict[str, Any]] = []
        full_binary = full_mask > 0
        for barcode_mask, barcode_score in zip(barcode_masks, barcode_scores):
            barcode_binary = barcode_mask > 0
            if np.any(barcode_binary & ~full_binary):
                print(
                    f"Skipping barcode mask with score {barcode_score:.3f}: not fully inside full object mask"
                )
                continue

            try:
                camera_pose = self._estimate_camera_pose(
                    image,
                    depth_m,
                    barcode_mask,
                    full_object_centroid_camera_m,
                )
            except Exception as exc:
                print(f"Skipping barcode mask with score {barcode_score:.3f}: {exc}")
                continue
            pose_records.append(
                {
                    "mask": barcode_mask,
                    "score": barcode_score,
                    "camera_pose": camera_pose,
                }
            )

        w_R_t, w_T_t = self._ee_pose_to_world_transform(ee_pose)
        for record in pose_records:
            record["world_pose"] = to_world_pose(
                record["camera_pose"],
                w_R_t,
                w_T_t,
                self.t_R_c,
                self.t_T_t_c,
            )

        if not pose_records:
            raise RuntimeError("No valid barcode poses found")

        pose_records = sorted(
            pose_records,
            key=lambda record: world_z_alignment_score(record["world_pose"]),
            reverse=True,
        )
        world_poses = [record["world_pose"] for record in pose_records]

        self.last_masks = [record["mask"] for record in pose_records]
        self.last_mask_scores = [record["score"] for record in pose_records]
        self.last_camera_poses = [record["camera_pose"] for record in pose_records]
        self.last_full_mask = full_mask
        self.last_tcp_poses = world_poses
        self.last_tcp_pose = world_poses[0] if world_poses else None
        return world_poses

    def estimate_barcodes_tcp(
        self,
        rgbd: Any,
    ) -> list[np.ndarray]:
        image, depth = _unpack_rgbd(rgbd)
        depth_m = depth_to_meters(depth)
        full_mask = run_sam3_best_mask(self.processor, image, self.full_mask_prompt)
        barcode_masks, barcode_scores = run_sam3_masks(
            self.processor, image, self.sam3_prompt, containment_mask=full_mask
        )
        full_object_depth_m = self._maybe_filter_depth_outliers(
            depth_m,
            full_mask > 0,
            "full object mask",
        )
        full_object_centroid_camera_m = compute_centroid(
            full_object_depth_m,
            full_mask,
            self.camera_matrix,
            self.centroid_method,
        )

        pose_records: list[dict[str, Any]] = []
        full_binary = full_mask > 0
        for barcode_mask, barcode_score in zip(barcode_masks, barcode_scores):
            barcode_binary = barcode_mask > 0
            if np.any(barcode_binary & ~full_binary):
                print(
                    f"Skipping barcode mask with score {barcode_score:.3f}: not fully inside full object mask"
                )
                continue

            try:
                camera_pose = self._estimate_camera_pose(
                    image,
                    depth_m,
                    barcode_mask,
                    full_object_centroid_camera_m,
                )
            except Exception as exc:
                print(f"Skipping barcode mask with score {barcode_score:.3f}: {exc}")
                continue
            pose_records.append(
                {
                    "mask": barcode_mask,
                    "score": barcode_score,
                    "camera_pose": camera_pose,
                }
            )

        for record in pose_records:
            record["tcp_pose"] = to_tcp_pose(
                record["camera_pose"],
                self.t_R_c,
                self.t_T_t_c,
            )

        if not pose_records:
            raise RuntimeError("No valid barcode poses found")

        pose_records = sorted(
            pose_records,
            key=lambda record: world_z_alignment_score(record["tcp_pose"]),
            reverse=True,
        )
        tcp_poses = [record["tcp_pose"] for record in pose_records]

        self.last_masks = [record["mask"] for record in pose_records]
        self.last_mask_scores = [record["score"] for record in pose_records]
        self.last_camera_poses = [record["camera_pose"] for record in pose_records]
        self.last_full_mask = full_mask
        self.last_tcp_poses = tcp_poses
        self.last_tcp_pose = tcp_poses[0] if tcp_poses else None
        return tcp_poses

    def estimate_barcode(
        self,
        rgbd: Any,
        ee_pose: np.ndarray | list[float] | tuple[float, ...],
    ) -> list[np.ndarray]:
        return self.estimate_barcodes(rgbd, ee_pose)
