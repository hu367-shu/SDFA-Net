"""
Voxel Utilities for Point Cloud Downsampling.

Provides efficient voxel grid filtering used in:
    - Post-aggregation point cloud downsampling (0.05 m voxels)
    - Point cloud preprocessing for Pillar Feature Net
"""

import torch
import numpy as np
from typing import Optional


def voxel_downsample(
    points: torch.Tensor,
    voxel_size: float,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Downsample a point cloud using voxel grid filtering.

    Args:
        points: [N, 3+] point cloud (xyz + optional features).
        voxel_size: Voxel edge length in meters.
        reduction: 'mean' (average points per voxel) or 'first' (keep first point).

    Returns:
        downsampled: [M, 3+] reduced point cloud.
    """
    if points.shape[0] == 0:
        return points

    device = points.device
    dtype = points.dtype

    # Quantize coordinates to voxel indices
    voxel_idx = torch.floor(points[:, :3] / voxel_size).long()  # [N, 3]

    # Create unique voxel keys
    # Hash: z * (grid_x * grid_y) + y * grid_x + x
    # For simplicity, use unique on the concatenated index
    unique_voxels, inverse = torch.unique(voxel_idx, dim=0, return_inverse=True)
    num_voxels = unique_voxels.shape[0]

    # Allocate output
    downsampled = torch.zeros(num_voxels, points.shape[1], dtype=dtype, device=device)

    for i in range(num_voxels):
        mask = inverse == i
        if reduction == "mean":
            downsampled[i] = points[mask].mean(dim=0)
        else:
            downsampled[i] = points[mask][0]

    return downsampled


def voxel_grid_mean(
    points: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """
    NumPy version: average points within each voxel.

    Args:
        points: [N, 3] numpy array.
        voxel_size: Voxel size in meters.

    Returns:
        downsampled: [M, 3] voxel centers with averaged coordinates.
    """
    if points.shape[0] == 0:
        return points

    voxel_indices = np.floor(points[:, :3] / voxel_size).astype(np.int64)
    unique_voxels, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
    M = unique_voxels.shape[0]

    downsampled = np.zeros((M, points.shape[1]), dtype=points.dtype)
    for i in range(M):
        mask = inverse == i
        downsampled[i] = points[mask].mean(axis=0)

    return downsampled


def voxel_downsample_np(
    points: np.ndarray,
    voxel_size: float = 0.05,
) -> np.ndarray:
    """Alias for voxel_grid_mean with standard parameters."""
    return voxel_grid_mean(points, voxel_size)
