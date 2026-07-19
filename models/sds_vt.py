"""
SDS-VT: Semi-Dense Depth-Supervised View Transform.

This is the core module that transforms image features into the BEV plane
using predicted depth distributions supervised by semi-dense LiDAR projections.

Pipeline:
    1. DepthHead: Predict per-pixel depth logits → softmax → depth_prob [B, D, H, W]
    2. Compute depth entropy for FA-Fusion confidence gating
    3. Lift: outer product image_feat ⊗ depth_prob → 3D frustum [B, C, D, H, W]
    4. Splat: sum-pool frustum features into BEV grid based on camera geometry

Key references:
    - LSS (Philion & Fidler, ECCV 2020)
    - BEVDepth (Li et al., 2022)
    - SDFA-Net paper equations (1)-(5)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DepthHead(nn.Module):
    """
    Predicts per-pixel categorical depth distribution.

    Output: [B, D, H, W] logits before softmax.
    Paper formula (1): P_d(u,v) = Softmax(Z_d(u,v))
    """

    def __init__(
        self,
        in_channels: int,
        depth_bins: int = 64,
    ):
        super().__init__()
        self.depth_bins = depth_bins

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, depth_bins, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] image features.

        Returns:
            depth_prob: [B, D, H, W] softmax-normalized depth probabilities.
        """
        logits = self.conv(x)  # [B, D, H, W]
        depth_prob = F.softmax(logits, dim=1)
        return depth_prob


class CameraGeometry(nn.Module):
    """
    Precomputes the 3D frustum grid that maps each (depth_bin, pixel_v, pixel_u)
    to a LiDAR-frame coordinate.

    This is the geometric backbone of Lift-Splat: given a virtual pinhole camera
    model and a set of depth bins, we precompute a grid of (x, y, z) LiDAR
    coordinates for each frustum cell.
    """

    def __init__(
        self,
        image_h: int,
        image_w: int,
        depth_bins: int,
        depth_min: float,
        depth_max: float,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        bev_h: int,
        bev_w: int,
    ):
        super().__init__()
        self.image_h = image_h
        self.image_w = image_w
        self.depth_bins = depth_bins
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.bev_h = bev_h
        self.bev_w = bev_w

        # Register depth bin centers as buffer
        depth_bin_edges = self._get_depth_bins()
        self.register_buffer("depth_bins_tensor", depth_bin_edges, persistent=False)

        # Precompute pixel grid
        u_coords = torch.arange(image_w, dtype=torch.float32)
        v_coords = torch.arange(image_h, dtype=torch.float32)
        self.register_buffer("u_grid", u_coords, persistent=False)
        self.register_buffer("v_grid", v_coords, persistent=False)

    def _get_depth_bins(self) -> torch.Tensor:
        """Logarithmically spaced depth bins (LID discretization)."""
        log_min = np.log(self.depth_min)
        log_max = np.log(self.depth_max)
        log_edges = np.linspace(log_min, log_max, self.depth_bins)
        return torch.from_numpy(np.exp(log_edges)).float()

    def get_frustum_coords(
        self,
        K: torch.Tensor,      # [B, 3, 3]
        T_lidar_cam: torch.Tensor,  # [B, 4, 4] LiDAR → Camera
    ) -> torch.Tensor:
        """
        Compute LiDAR-frame (x,y,z) for each (d, h, w) frustum cell.

        Args:
            K: Virtual pinhole intrinsic matrices [B, 3, 3].
            T_lidar_cam: LiDAR→Camera transforms [B, 4, 4].

        Returns:
            coords: [B, D, H, W, 3] LiDAR-frame coordinates.
        """
        B = K.shape[0]
        D, H, W = self.depth_bins, self.image_h, self.image_w
        device = K.device

        # Pixel grid (u, v)
        u = self.u_grid.to(device)  # [W]
        v = self.v_grid.to(device)  # [H]
        uu, vv = torch.meshgrid(u, v, indexing="xy")  # [H, W] each

        # Homogeneous pixel coordinates
        ones = torch.ones_like(uu)
        pixels_h = torch.stack([uu, vv, ones], dim=0)  # [3, H, W]

        # K inverse: pixel → camera ray direction (unnormalized)
        # For each batch
        coords_list = []

        for b in range(B):
            K_b = K[b]                             # [3, 3]
            K_inv = torch.inverse(K_b)             # [3, 3]

            # Direction vectors in camera frame (unit depth)
            rays_cam = K_inv @ pixels_h.view(3, -1)  # [3, H*W]
            rays_cam = rays_cam.view(3, H, W)        # [3, H, W]

            # Scale by depth bins → 3D points in camera frame
            depths = self.depth_bins_tensor.to(device)  # [D]
            pts_cam = rays_cam.unsqueeze(0) * depths.view(D, 1, 1, 1)
            # [D, 3, H, W]

            # Transform to LiDAR frame
            T_cam_lidar = torch.inverse(T_lidar_cam[b])  # Camera → LiDAR

            pts_cam_h = torch.cat([
                pts_cam,
                torch.ones(D, 1, H, W, device=device),
            ], dim=1)  # [D, 4, H, W]

            pts_cam_flat = pts_cam_h.permute(0, 2, 3, 1).reshape(-1, 4)  # [D*H*W, 4]
            pts_lidar_flat = (T_cam_lidar @ pts_cam_flat.T).T  # [D*H*W, 4]
            pts_lidar = pts_lidar_flat[:, :3].view(D, H, W, 3)  # [D, H, W, 3]

            coords_list.append(pts_lidar)

        return torch.stack(coords_list, dim=0)  # [B, D, H, W, 3]


class SDSVT(nn.Module):
    """
    Semi-Dense Depth-Supervised View Transform.

    Full pipeline:
        img_feat [B, C, H, W] → DepthHead → depth_prob [B, D, H, W]
        img_feat ⊗ depth_prob → frustum_feat [B, C, D, H, W]
        frustum_feat + geometry → sum-pool → cam_bev_feat [B, C, H_bev, W_bev]
    """

    def __init__(
        self,
        in_channels: int = 256,
        depth_bins: int = 64,
        depth_min: float = 0.5,
        depth_max: float = 30.0,
        image_h: int = 384,
        image_w: int = 640,
        x_range: Tuple[float, float] = (-25.6, 25.6),
        y_range: Tuple[float, float] = (-12.8, 38.4),
        bev_h: int = 256,
        bev_w: int = 256,
    ):
        """
        Args:
            in_channels: Input image feature channels.
            depth_bins: Number of discrete depth bins.
            depth_min, depth_max: Depth range in meters.
            image_h, image_w: Rectified image dimensions.
            x_range, y_range: BEV coordinate ranges in meters.
            bev_h, bev_w: BEV grid dimensions in pixels.
        """
        super().__init__()
        self.in_channels = in_channels
        self.depth_bins = depth_bins
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.x_res = (self.x_max - self.x_min) / bev_w
        self.y_res = (self.y_max - self.y_min) / bev_h

        # Depth prediction head
        self.depth_head = DepthHead(
            in_channels=in_channels,
            depth_bins=depth_bins,
        )

        # Frustum → BEV scatter geometry
        self.geometry = CameraGeometry(
            image_h=image_h,
            image_w=image_w,
            depth_bins=depth_bins,
            depth_min=depth_min,
            depth_max=depth_max,
            x_range=x_range,
            y_range=y_range,
            bev_h=bev_h,
            bev_w=bev_w,
        )

        # Post-BEV refinement convolution
        self.bev_refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        img_feat: torch.Tensor,
        K: torch.Tensor,
        T_lidar_cam: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            img_feat: [B, C, H, W] image features from backbone.
            K: [B, 3, 3] virtual pinhole camera intrinsic matrices.
            T_lidar_cam: [B, 4, 4] LiDAR → Camera transforms.

        Returns:
            cam_bev: [B, C, H_bev, W_bev] camera-view BEV features.
            depth_prob: [B, D, H, W] depth probability distributions.
            entropy_bev: [B, 1, H_bev, W_bev] depth entropy projected to BEV.
        """
        B, C, H, W = img_feat.shape

        # 1. Predict depth distribution
        depth_prob = self.depth_head(img_feat)  # [B, D, H, W]

        # 2. Compute depth entropy (for FA-Fusion gating)
        entropy = self._compute_entropy(depth_prob)  # [B, H, W]

        # 3. Lift: outer product → frustum features
        frustum = img_feat.unsqueeze(2) * depth_prob.unsqueeze(1)
        # [B, C, D, H, W]

        # 4. Get frustum geometry
        coords_lidar = self.geometry.get_frustum_coords(K, T_lidar_cam)
        # [B, D, H, W, 3]

        # 5. Splat: frustum → BEV via sum pooling
        cam_bev = self._splat_to_bev(frustum, coords_lidar)
        # [B, C, H_bev, W_bev]

        # 6. Project entropy to BEV for gating
        entropy_bev = self._splat_entropy_to_bev(entropy, coords_lidar)
        # [B, 1, H_bev, W_bev]

        # 7. Refine BEV features
        cam_bev = self.bev_refine(cam_bev)

        return cam_bev, depth_prob, entropy_bev

    @staticmethod
    def _compute_entropy(depth_prob: torch.Tensor) -> torch.Tensor:
        """
        Compute depth prediction entropy.
        Paper formula (5): H(u,v) = -sum_d P_d(u,v) * log(P_d(u,v) + ε)

        Args:
            depth_prob: [B, D, H, W]

        Returns:
            entropy: [B, H, W] normalized entropy.
        """
        eps = 1e-6
        log_prob = torch.log(depth_prob + eps)
        entropy = -torch.sum(depth_prob * log_prob, dim=1)  # [B, H, W]

        # Normalize by max entropy (uniform distribution = log(D))
        max_entropy = np.log(depth_prob.shape[1])
        entropy = entropy / max_entropy

        return entropy

    def _splat_to_bev(
        self,
        frustum: torch.Tensor,        # [B, C, D, H, W]
        coords_lidar: torch.Tensor,   # [B, D, H, W, 3]
    ) -> torch.Tensor:
        """
        Sum-pool frustum features into BEV grid.
        Paper formula (4): F_cam^{BEV}(i,j) = sum_{q∈Ω_{ij}} F_frustum(q)
        """
        B, C, D, H, W = frustum.shape
        device = frustum.device

        bev = torch.zeros(B, C, self.bev_h, self.bev_w, device=device)

        for b in range(B):
            # BEV grid indices
            x_coords = coords_lidar[b, :, :, :, 0]  # [D, H, W]
            y_coords = coords_lidar[b, :, :, :, 1]  # [D, H, W]

            x_idx = ((x_coords - self.x_min) / self.x_res).long()
            y_idx = ((y_coords - self.y_min) / self.y_res).long()

            valid = (
                (x_idx >= 0) & (x_idx < self.bev_w) &
                (y_idx >= 0) & (y_idx < self.bev_h)
            )

            # Flatten valid frustum cells
            x_valid = x_idx[valid]
            y_valid = y_idx[valid]
            feats_valid = frustum[b, :, :, :, :][:, valid]  # [C, N_valid]

            if feats_valid.shape[1] == 0:
                continue

            # Scatter add to BEV
            linear_idx = y_valid * self.bev_w + x_valid  # [N_valid]
            bev_flat = bev[b].view(C, -1)  # [C, H*W]
            idx_expanded = linear_idx.unsqueeze(0).expand(C, -1)
            bev_flat.scatter_add_(1, idx_expanded, feats_valid)

        return bev

    def _splat_entropy_to_bev(
        self,
        entropy: torch.Tensor,           # [B, H, W]
        coords_lidar: torch.Tensor,      # [B, D, H, W, 3]
    ) -> torch.Tensor:
        """
        Project depth entropy into BEV by averaging over depth dimension.
        """
        B, H, W = entropy.shape
        D = coords_lidar.shape[1]
        device = entropy.device

        entropy_bev = torch.zeros(B, 1, self.bev_h, self.bev_w, device=device)
        count_bev = torch.zeros(B, 1, self.bev_h, self.bev_w, device=device)

        for b in range(B):
            x_coords = coords_lidar[b, :, :, :, 0]  # [D, H, W]
            y_coords = coords_lidar[b, :, :, :, 1]  # [D, H, W]

            x_idx = ((x_coords - self.x_min) / self.x_res).long()
            y_idx = ((y_coords - self.y_min) / self.y_res).long()

            valid = (
                (x_idx >= 0) & (x_idx < self.bev_w) &
                (y_idx >= 0) & (y_idx < self.bev_h)
            )

            x_valid = x_idx[valid]
            y_valid = y_idx[valid]

            # Expand entropy to depth dimension and gather valid cells
            ent_valid = entropy[b].unsqueeze(0).expand(D, -1, -1)[valid]  # [N_valid]

            linear_idx = y_valid * self.bev_w + x_valid

            ent_bev_flat = entropy_bev[b].view(-1)
            cnt_bev_flat = count_bev[b].view(-1)

            ent_bev_flat.scatter_add_(0, linear_idx, ent_valid)
            cnt_bev_flat.scatter_add_(0, linear_idx, torch.ones_like(ent_valid))

        # Average
        entropy_bev = entropy_bev / count_bev.clamp(min=1)

        return entropy_bev
