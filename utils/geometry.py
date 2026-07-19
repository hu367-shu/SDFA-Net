"""
Geometric Utilities for Coordinate Transforms, Projection, and BEV Operations.

Covers:
    - Point cloud transformation (homogeneous 4x4)
    - LiDAR → Camera → Image projection
    - BEV ↔ World coordinate conversion
    - BEV box IoU computation
    - BEV Non-Maximum Suppression
"""

import torch
import numpy as np
from typing import Tuple, List, Optional


# ── Point Transform ──────────────────────────────────────────────

def transform_points(
    points: torch.Tensor,  # [N, 3]
    T: torch.Tensor,       # [4, 4]
) -> torch.Tensor:
    """
    Transform 3D points by a 4x4 homogeneous matrix.

    Args:
        points: [N, 3] or [B, N, 3] xyz coordinates.
        T: [4, 4] or [B, 4, 4] transformation matrix.

    Returns:
        transformed: Same shape as points.
    """
    if points.dim() == 2:
        N = points.shape[0]
        ones = torch.ones(N, 1, dtype=points.dtype, device=points.device)
        pts_h = torch.cat([points, ones], dim=1)         # [N, 4]
        pts_t = (T @ pts_h.T).T[:, :3]                    # [N, 3]
        return pts_t
    else:
        B, N, _ = points.shape
        ones = torch.ones(B, N, 1, dtype=points.dtype, device=points.device)
        pts_h = torch.cat([points, ones], dim=2)          # [B, N, 4]
        pts_t = torch.bmm(T, pts_h.transpose(1, 2)).transpose(1, 2)[:, :, :3]
        return pts_t


def transform_points_np(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """NumPy version of transform_points."""
    N = points.shape[0]
    ones = np.ones((N, 1), dtype=points.dtype)
    pts_h = np.hstack([points[:, :3], ones])
    pts_t = (T @ pts_h.T).T[:, :3]
    return pts_t


# ── Camera Projection ────────────────────────────────────────────

def project_lidar_to_image(
    points_lidar: torch.Tensor,    # [N, 3]
    T_cam_lidar: torch.Tensor,     # [4, 4] LiDAR → Camera
    K: torch.Tensor,               # [3, 3] camera intrinsic
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project LiDAR points to image pixel coordinates.

    Args:
        points_lidar: [N, 3] points in LiDAR coordinate frame.
        T_cam_lidar: [4, 4] LiDAR → Camera extrinsic matrix.
        K: [3, 3] camera intrinsic matrix.

    Returns:
        uv: [N, 2] pixel coordinates (u, v).
        depths: [N] camera-frame depths.
    """
    N = points_lidar.shape[0]
    device = points_lidar.device

    # Transform to camera frame
    ones = torch.ones(N, 1, device=device)
    pts_h = torch.cat([points_lidar, ones], dim=1)  # [N, 4]
    pts_cam = (T_cam_lidar @ pts_h.T).T[:, :3]       # [N, 3]

    depths = pts_cam[:, 2]

    # Project to image
    pts_img_h = (K @ pts_cam.T).T  # [N, 3]
    u = pts_img_h[:, 0] / (pts_img_h[:, 2] + 1e-8)
    v = pts_img_h[:, 1] / (pts_img_h[:, 2] + 1e-8)

    uv = torch.stack([u, v], dim=1)
    return uv, depths


# ── BEV ↔ World ──────────────────────────────────────────────────

def bev_coords_to_world(
    u: torch.Tensor,
    v: torch.Tensor,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    bev_w: int,
    bev_h: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert BEV pixel grid indices to world coordinates (LiDAR frame).

    Args:
        u: BEV column indices (can be fractional for sub-pixel).
        v: BEV row indices.
        x_range: (x_min, x_max) in meters.
        y_range: (y_min, y_max) in meters.
        bev_w: BEV grid width.
        bev_h: BEV grid height.

    Returns:
        x_world, y_world: World coordinates in meters.
    """
    x_res = (x_range[1] - x_range[0]) / bev_w
    y_res = (y_range[1] - y_range[0]) / bev_h

    x_world = u * x_res + x_range[0]
    y_world = v * y_res + y_range[0]

    return x_world, y_world


def world_coords_to_bev(
    x: torch.Tensor,
    y: torch.Tensor,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    bev_w: int,
    bev_h: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert world coordinates to BEV pixel indices.

    Returns:
        u, v: BEV column and row indices (continuous, not rounded).
    """
    x_res = (x_range[1] - x_range[0]) / bev_w
    y_res = (y_range[1] - y_range[0]) / bev_h

    u = (x - x_range[0]) / x_res
    v = (y - y_range[0]) / y_res

    return u, v


# ── IoU Computation ──────────────────────────────────────────────

def compute_iou_bev(
    boxes1: torch.Tensor,  # [N, 4] (cx, cy, w, l)
    boxes2: torch.Tensor,  # [M, 4] (cx, cy, w, l)
) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of axis-aligned BEV boxes.

    Returns:
        iou: [N, M]
    """
    # Convert (cx, cy, w, l) → (x1, y1, x2, y2)
    b1 = torch.stack([
        boxes1[:, 0] - boxes1[:, 2] / 2,
        boxes1[:, 1] - boxes1[:, 3] / 2,
        boxes1[:, 0] + boxes1[:, 2] / 2,
        boxes1[:, 1] + boxes1[:, 3] / 2,
    ], dim=1)

    b2 = torch.stack([
        boxes2[:, 0] - boxes2[:, 2] / 2,
        boxes2[:, 1] - boxes2[:, 3] / 2,
        boxes2[:, 0] + boxes2[:, 2] / 2,
        boxes2[:, 1] + boxes2[:, 3] / 2,
    ], dim=1)

    # Intersection
    inter_x1 = torch.max(b1[:, None, 0], b2[None, :, 0])
    inter_y1 = torch.max(b1[:, None, 1], b2[None, :, 1])
    inter_x2 = torch.min(b1[:, None, 2], b2[None, :, 2])
    inter_y2 = torch.min(b1[:, None, 3], b2[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Area
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]

    union = area1[:, None] + area2[None, :] - inter_area

    return inter_area / (union + 1e-7)


# ── BEV NMS ──────────────────────────────────────────────────────

def nms_bev(
    boxes: torch.Tensor,        # [N, 4] (cx, cy, w, l)
    scores: torch.Tensor,       # [N]
    iou_threshold: float = 0.45,
) -> torch.Tensor:
    """
    BEV Non-Maximum Suppression.

    Args:
        boxes: [N, 4] axis-aligned BEV boxes.
        scores: [N] confidence scores.
        iou_threshold: IoU threshold for suppression.

    Returns:
        keep: Indices of kept detections.
    """
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    # Sort by score descending
    _, order = scores.sort(descending=True)
    boxes = boxes[order]

    keep = []
    suppressed = torch.zeros(boxes.shape[0], dtype=torch.bool, device=boxes.device)

    # Convert to (x1, y1, x2, y2)
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    areas = (x2 - x1) * (y2 - y1)

    for i in range(boxes.shape[0]):
        if suppressed[i]:
            continue

        keep.append(order[i].item())

        # Compute IoU with remaining boxes
        inter_x1 = torch.max(x1[i], x1[i + 1:])
        inter_y1 = torch.max(y1[i], y1[i + 1:])
        inter_x2 = torch.min(x2[i], x2[i + 1:])
        inter_y2 = torch.min(y2[i], y2[i + 1:])

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter = inter_w * inter_h

        iou = inter / (areas[i] + areas[i + 1:] - inter + 1e-7)

        suppressed[i + 1:][iou > iou_threshold] = True

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)
