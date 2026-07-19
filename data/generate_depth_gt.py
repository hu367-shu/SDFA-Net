"""
Semi-Dense Depth Ground-Truth Generation.

Projects aggregated LiDAR point cloud onto the rectified image plane to create
pixel-wise depth bin labels. Only pixels with a valid LiDAR projection receive
supervision (semi-dense).

This module is used both during offline dataset preparation and online in the
training loop (if GT is generated on-the-fly).
"""

import numpy as np
import torch
from typing import Tuple, Optional, List


class DepthGTGenerator:
    """
    Generates semi-dense depth ground truth by projecting LiDAR points
    onto the rectified (virtual pinhole) image plane.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (384, 640),
        depth_bins: int = 64,
        depth_min: float = 0.5,
        depth_max: float = 30.0,
    ):
        """
        Args:
            image_size: (H, W) of the rectified image.
            depth_bins: Number of discrete depth bins.
            depth_min: Minimum depth in meters.
            depth_max: Maximum depth in meters.
        """
        self.H, self.W = image_size
        self.depth_bins = depth_bins
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.log_min = np.log(depth_min)
        self.log_max = np.log(depth_max)

    def _discretize_depth(self, depth: float) -> int:
        """
        Convert a continuous depth value to a discrete bin index
        using logarithmic spacing (LID: Linear-Increasing Discretization).
        """
        if depth <= self.depth_min:
            return 0
        if depth >= self.depth_max:
            return self.depth_bins - 1

        log_d = np.log(depth)
        bin_idx = int(
            (log_d - self.log_min) / (self.log_max - self.log_min) * self.depth_bins
        )
        return np.clip(bin_idx, 0, self.depth_bins - 1)

    def _undiscretize_depth(self, bin_idx: int) -> float:
        """Convert a depth bin index back to a continuous depth value (center of bin)."""
        log_d = self.log_min + (bin_idx + 0.5) / self.depth_bins * (self.log_max - self.log_min)
        return float(np.exp(log_d))

    def generate(
        self,
        agg_points: np.ndarray,
        T_cam_lidar: np.ndarray,
        K_virtual: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate semi-dense depth GT from aggregated LiDAR points.

        Args:
            agg_points: [N, 3] aggregated point cloud in LiDAR coordinates.
            T_cam_lidar: [4, 4] transformation from LiDAR to camera frame.
            K_virtual: [3, 3] virtual pinhole camera intrinsic matrix.

        Returns:
            depth_gt: [H, W] int64, depth bin index for each pixel (0 where invalid).
            depth_mask: [H, W] float32, 1.0 where a LiDAR point projects, 0 elsewhere.
        """
        depth_gt = np.zeros((self.H, self.W), dtype=np.int64)
        depth_mask = np.zeros((self.H, self.W), dtype=np.float32)

        if agg_points.shape[0] == 0:
            return depth_gt, depth_mask

        # Transform LiDAR points to camera frame
        N = agg_points.shape[0]
        points_h = np.hstack([agg_points[:, :3], np.ones((N, 1), dtype=agg_points.dtype)])
        points_cam = (T_cam_lidar @ points_h.T).T  # (N, 4)

        # Filter points behind the camera
        valid_depth = points_cam[:, 2] > 0
        points_cam = points_cam[valid_depth]

        if points_cam.shape[0] == 0:
            return depth_gt, depth_mask

        # Project to image plane
        pts_3d = points_cam[:, :3]
        pts_2d_h = (K_virtual @ pts_3d.T).T  # (M, 3)
        u = (pts_2d_h[:, 0] / pts_2d_h[:, 2]).astype(np.int32)
        v = (pts_2d_h[:, 1] / pts_2d_h[:, 2]).astype(np.int32)
        depths = pts_3d[:, 2]

        # Filter points within image bounds
        valid_u = (u >= 0) & (u < self.W)
        valid_v = (v >= 0) & (v < self.H)
        valid_img = valid_u & valid_v

        u_valid = u[valid_img]
        v_valid = v[valid_img]
        d_valid = depths[valid_img]

        # Assign depth bin indices (keep closest depth per pixel)
        for px, py, d in zip(u_valid, v_valid, d_valid):
            bin_id = self._discretize_depth(d)
            if depth_mask[py, px] == 0 or d < self._undiscretize_depth(depth_gt[py, px]):
                depth_gt[py, px] = bin_id
                depth_mask[py, px] = 1.0

        return depth_gt, depth_mask

    def generate_batch(
        self,
        agg_points_list: List[np.ndarray],
        T_cam_lidar_list: List[np.ndarray],
        K_virtual_list: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch version of generate().

        Returns:
            depth_gt: [B, H, W] int64
            depth_mask: [B, H, W] float32
        """
        B = len(agg_points_list)
        depth_gt_batch = np.zeros((B, self.H, self.W), dtype=np.int64)
        depth_mask_batch = np.zeros((B, self.H, self.W), dtype=np.float32)

        for i in range(B):
            gt, mask = self.generate(
                agg_points_list[i],
                T_cam_lidar_list[i],
                K_virtual_list[i],
            )
            depth_gt_batch[i] = gt
            depth_mask_batch[i] = mask

        return depth_gt_batch, depth_mask_batch

    def get_depth_bin_edges(self) -> np.ndarray:
        """Return the depth bin boundaries (in meters)."""
        log_edges = np.linspace(self.log_min, self.log_max, self.depth_bins + 1)
        return np.exp(log_edges)


def generate_depth_gt(
    agg_points: np.ndarray,
    calib: dict,
    image_size: Tuple[int, int],
    depth_bins: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function matching the paper's pseudocode signature.

    Args:
        agg_points: [N, 3] aggregated LiDAR points.
        calib: Dict with keys 'T_lidar_cam' [4,4] and 'K_virtual' [3,3].
        image_size: (H, W).
        depth_bins: Number of depth bins.

    Returns:
        depth_gt: [H, W] depth bin indices.
        depth_mask: [H, W] binary mask.
    """
    generator = DepthGTGenerator(
        image_size=image_size,
        depth_bins=depth_bins,
    )
    return generator.generate(
        agg_points,
        calib["T_lidar_cam"],
        calib["K_virtual"],
    )
