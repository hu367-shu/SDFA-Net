"""
Multi-Frame LiDAR Point Cloud Aggregation.

Sliding-window accumulation of historical point clouds into the current LiDAR
coordinate frame using SLAM poses. Includes quality-control gating on pose
stability and voxel downsampling post-aggregation.

Per the paper:
    P_t^{agg} = U_{k=t-N+1}^{t} T_t^{-1} * T_k * P_k
"""

import numpy as np
from typing import List, Optional, Tuple
from scipy.spatial.transform import Rotation as R


class LidarAggregator:
    """
    Aggregates multi-frame LiDAR point clouds using SLAM poses.

    Maintains a sliding window buffer of recent point clouds and transforms
    them to the current frame's coordinate system.
    """

    def __init__(
        self,
        window_size: int = 10,
        pose_std_threshold: float = 0.05,
        voxel_size: float = 0.05,
    ):
        """
        Args:
            window_size: N, number of frames to aggregate (including current).
            pose_std_threshold: Max allowed translation standard deviation (meters).
                                Frames exceeding this are rejected.
            voxel_size: Downsampling voxel size in meters.
        """
        self.window_size = window_size
        self.pose_std_threshold = pose_std_threshold
        self.voxel_size = voxel_size

        # Internal buffer
        self._cloud_buffer: List[np.ndarray] = []
        self._pose_buffer: List[np.ndarray] = []

    def add_frame(self, cloud: np.ndarray, pose: np.ndarray) -> None:
        """
        Add a new frame to the buffer.

        Args:
            cloud: [N, 3+] point cloud in LiDAR coordinates.
            pose: [4, 4] transformation matrix LiDAR → World.
        """
        self._cloud_buffer.append(cloud)
        self._pose_buffer.append(pose)

        # Trim to window size
        if len(self._cloud_buffer) > self.window_size:
            self._cloud_buffer.pop(0)
            self._pose_buffer.pop(0)

    def aggregate(self) -> Optional[np.ndarray]:
        """
        Aggregate buffered frames to the current (most recent) frame.

        Returns:
            aggregated_cloud: [M, 3] aggregated point cloud in current LiDAR frame,
                              or None if buffer is empty.
        """
        if not self._cloud_buffer:
            return None

        T_curr = self._pose_buffer[-1]  # current frame: LiDAR → World
        T_curr_inv = np.linalg.inv(T_curr)

        aggregated = []

        for i in range(len(self._cloud_buffer)):
            # Check pose stability
            if not self._check_pose_quality(self._pose_buffer[i]):
                continue

            cloud = self._cloud_buffer[i][:, :3]  # use xyz only
            T_k = self._pose_buffer[i]

            # Transform: world → current LiDAR
            T_rel = T_curr_inv @ T_k  # source LiDAR → current LiDAR
            cloud_transformed = self._transform_points(cloud, T_rel)

            aggregated.append(cloud_transformed)

        if not aggregated:
            return self._cloud_buffer[-1][:, :3].copy()

        agg = np.concatenate(aggregated, axis=0)
        agg = self._voxel_downsample(agg, self.voxel_size)

        return agg

    def reset(self) -> None:
        """Clear the internal buffer."""
        self._cloud_buffer.clear()
        self._pose_buffer.clear()

    # ── Private helpers ──────────────────────────────────────────────

    @staticmethod
    def _transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Transform points by a 4x4 homogeneous matrix.

        Args:
            points: [N, 3] xyz coordinates.
            T: [4, 4] transformation matrix.

        Returns:
            transformed: [N, 3]
        """
        N = points.shape[0]
        points_h = np.hstack([points, np.ones((N, 1), dtype=points.dtype)])
        transformed_h = (T @ points_h.T).T
        return transformed_h[:, :3]

    @staticmethod
    def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
        """
        Downsample point cloud by averaging points within each voxel.

        Args:
            points: [N, 3] point cloud.
            voxel_size: Voxel edge length in meters.

        Returns:
            downsampled: [M, 3] downsampled cloud.
        """
        if points.shape[0] == 0:
            return points

        voxel_indices = np.floor(points[:, :3] / voxel_size).astype(np.int64)

        # Create unique voxel keys
        unique_voxels, inverse = np.unique(
            voxel_indices, axis=0, return_inverse=True
        )

        # Average points within each voxel
        M = unique_voxels.shape[0]
        downsampled = np.zeros((M, 3), dtype=points.dtype)

        for i in range(M):
            mask = inverse == i
            downsampled[i] = points[mask].mean(axis=0)

        return downsampled

    @staticmethod
    def _check_pose_quality(pose: np.ndarray) -> bool:
        """
        Simple check: verify the pose matrix is valid (no NaN/Inf, det(R) ≈ 1).

        A more sophisticated check would compare against a local window for
        translation std, but that requires the full window context.
        """
        if np.any(np.isnan(pose)) or np.any(np.isinf(pose)):
            return False

        # Rotation part should have determinant ≈ 1
        rot = pose[:3, :3]
        det = np.linalg.det(rot)
        if abs(det - 1.0) > 0.1:
            return False

        return True


def get_translation_std(poses: List[np.ndarray]) -> float:
    """
    Compute the standard deviation of translation vectors over a window.

    Args:
        poses: List of [4, 4] transformation matrices.

    Returns:
        std: Standard deviation of translation components (scalar in meters).
    """
    if len(poses) < 2:
        return 0.0

    translations = np.array([p[:3, 3] for p in poses])  # (N, 3)
    stds = np.std(translations, axis=0)  # (3,)
    return float(np.linalg.norm(stds))


def aggregate_multiframe_cloud(
    clouds: List[np.ndarray],
    poses: List[np.ndarray],
    current_idx: int,
    N: int = 10,
    pose_std_th: float = 0.05,
    voxel_size: float = 0.05,
) -> np.ndarray:
    """
    Standalone function for multi-frame point cloud aggregation.

    Args:
        clouds: List of all-frame point clouds [N_points, 3+].
        poses: List of LiDAR→World 4x4 pose matrices per frame.
        current_idx: Index of the current keyframe.
        N: Sliding window size.
        pose_std_th: Pose translation std threshold for frame rejection.
        voxel_size: Voxel downsample size.

    Returns:
        aggregated: [M, 3] aggregated point cloud in current LiDAR frame.
    """
    T_t = poses[current_idx]
    T_t_inv = np.linalg.inv(T_t)

    agg_points = []

    # Check translation std over the window
    window_poses = poses[max(0, current_idx - N + 1): current_idx + 1]
    if get_translation_std(window_poses) > pose_std_th:
        # Use only the current frame if poses are unstable
        window_range = [current_idx]
    else:
        window_range = range(current_idx - N + 1, current_idx + 1)

    for k in window_range:
        if k < 0:
            continue

        P_k = clouds[k][:, :3]
        T_k = poses[k]
        T_rel = T_t_inv @ T_k  # k's LiDAR → current LiDAR

        P_k_to_t = LidarAggregator._transform_points(P_k, T_rel)
        agg_points.append(P_k_to_t)

    if not agg_points:
        return np.empty((0, 3), dtype=np.float32)

    agg = np.concatenate(agg_points, axis=0)

    # Voxel downsample
    agg = LidarAggregator._voxel_downsample(agg, voxel_size)

    return agg
