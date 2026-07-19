"""
PointPillars Backbone for LiDAR BEV Feature Extraction.

Implements the PointPillars pipeline:
    1. Point cloud → Pillar voxelization
    2. Pillar Feature Net (PFN): per-pillar point feature extraction
    3. Scatter: pillar features → 2D pseudo-image (BEV)
    4. 2D CNN backbone + FPN for BEV feature extraction
    5. Density map computation for FA-Fusion

Reference: Lang et al., "PointPillars: Fast Encoders for Object Detection
from Point Clouds", CVPR 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class PillarFeatureNet(nn.Module):
    """
    Converts raw points within each pillar into a fixed-length feature vector.

    For each point in a pillar:
        - Compute offset from pillar center: (dx, dy, dz)
        - Compute offset from point mean: (x - x_mean, y - y_mean, z - z_mean)
        - Augment with original (x, y, z, intensity if available)
        → Total per-point features: 9 (or 10 with intensity)
        → Linear → BN → ReLU → Max-pool over points → pillar feature
    """

    def __init__(
        self,
        in_channels: int = 9,
        out_channels: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        voxel_features: torch.Tensor,      # [M, max_pts, C]
        voxel_num_points: torch.Tensor,     # [M]
        voxel_coords: torch.Tensor,         # [M, 4] (batch, z, y, x)
    ) -> torch.Tensor:
        """
        Args:
            voxel_features: Per-point features in each pillar.
            voxel_num_points: Number of valid points per pillar.
            voxel_coords: Pillar indices [batch_idx, z_idx, y_idx, x_idx].

        Returns:
            pillar_features: [M, out_channels]
        """
        # Find distance to mean (robust to empty pillars)
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / \
            voxel_num_points.float().view(-1, 1, 1).clamp(min=1)

        # Offset from pillar center (normalized by voxel size)
        voxel_centers = voxel_coords[:, [3, 2, 1]].float()  # x, y, z
        points_centered = voxel_features[:, :, :3] - voxel_centers.unsqueeze(1)

        # Offset from point mean
        points_offset_mean = voxel_features[:, :, :3] - points_mean

        # Concatenate augmented features
        augmented = torch.cat([
            voxel_features[:, :, :3],       # x, y, z
            points_centered,                 # dx_c, dy_c, dz_c
            points_offset_mean,              # dx_m, dy_m, dz_m
        ], dim=-1)

        # If intensity is present
        if voxel_features.shape[-1] > 3:
            augmented = torch.cat([
                augmented,
                voxel_features[:, :, 3:],
            ], dim=-1)

        # Linear transform
        B, N, C = augmented.shape
        x = augmented.view(-1, C)
        x = self.linear(x)                   # [B*N, out_c]
        x = x.view(B, N, -1)                 # [B, N, out_c]

        # Max-pool over points
        x = x.max(dim=1)[0]                  # [B, out_c]

        return x


class PointPillarsScatter(nn.Module):
    """
    Scatters pillar features back to a 2D BEV pseudo-image grid.

    Each pillar maps to one cell in the BEV grid. Features are placed via
    scatter operation at (batch, y_idx, x_idx).
    """

    def __init__(
        self,
        bev_h: int,
        bev_w: int,
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w

    def forward(
        self,
        pillar_features: torch.Tensor,       # [M, C]
        voxel_coords: torch.Tensor,          # [M, 4]
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pillar_features: Per-pillar features.
            voxel_coords: [batch, z, y, x] pillar indices.
            batch_size: B.

        Returns:
            bev: [B, C, H, W] BEV feature map.
            density: [B, 1, H, W] pillar density map (point count per cell).
        """
        C = pillar_features.shape[1]
        device = pillar_features.device

        bev = torch.zeros(batch_size, C, self.bev_h, self.bev_w, device=device)
        density = torch.zeros(batch_size, 1, self.bev_h, self.bev_w, device=device)

        for b in range(batch_size):
            batch_mask = voxel_coords[:, 0] == b
            feats = pillar_features[batch_mask]     # [M_b, C]
            coords = voxel_coords[batch_mask]        # [M_b, 4]

            y_idx = coords[:, 2].long()
            x_idx = coords[:, 3].long()

            valid = (
                (x_idx >= 0) & (x_idx < self.bev_w) &
                (y_idx >= 0) & (y_idx < self.bev_h)
            )
            feats = feats[valid]
            x_idx = x_idx[valid]
            y_idx = y_idx[valid]

            # Scatter (use scatter_mean for overlapping pillars)
            idx = y_idx * self.bev_w + x_idx
            ones = torch.ones(feats.shape[0], 1, device=device)

            bev_flat = bev[b].view(C, -1)           # [C, H*W]
            density_flat = density[b].view(1, -1)    # [1, H*W]

            idx_expanded = idx.unsqueeze(0).expand(C, -1)
            idx_density = idx.unsqueeze(0).expand(1, -1)

            bev_flat.scatter_add_(1, idx_expanded, feats.T)
            density_flat.scatter_add_(1, idx_density, ones.T)

        # Normalize BEV features by density (mean)
        density_safe = density.clamp(min=1)
        bev = bev / density_safe

        return bev, density


class PointPillarsBackbone(nn.Module):
    """
    Full PointPillars pipeline: pillarization → PFN → scatter → 2D CNN → BEV features.

    Also returns the density map for use in FA-Fusion's physical confidence gate.
    """

    def __init__(
        self,
        point_cloud_range: Tuple[float, ...] = (-25.6, -25.6, -2.0, 25.6, 25.6, 4.0),
        voxel_size: Tuple[float, ...] = (0.2, 0.2, 6.0),
        max_points_per_voxel: int = 32,
        max_voxels: int = 16000,
        pillar_channels: int = 64,
        bev_out_channels: int = 256,
    ):
        super().__init__()
        self.x_min, self.y_min, self.z_min, self.x_max, self.y_max, self.z_max = point_cloud_range
        self.vx, self.vy, self.vz = voxel_size
        self.max_points = max_points_per_voxel
        self.max_voxels = max_voxels
        self.pillar_channels = pillar_channels
        self.bev_out_channels = bev_out_channels

        # Compute BEV grid dimensions
        self.bev_w = int((self.x_max - self.x_min) / self.vx)
        self.bev_h = int((self.y_max - self.y_min) / self.vy)

        # Pillar Feature Net
        self.pfn = PillarFeatureNet(
            in_channels=9,  # x,y,z, dx_c,dy_c,dz_c, dx_m,dy_m,dz_m
            out_channels=pillar_channels,
        )

        # Scatter
        self.scatter = PointPillarsScatter(
            bev_h=self.bev_h,
            bev_w=self.bev_w,
        )

        # 2D CNN backbone for BEV feature extraction
        self.bev_cnn = self._build_bev_cnn(
            in_channels=pillar_channels,
            out_channels=bev_out_channels,
        )

    def _build_bev_cnn(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Build a simple 2D CNN for processing the scattered BEV pseudo-image.
        Uses a down-sample block followed by several residual blocks.
        """
        return nn.Sequential(
            # Block 1: conv + downsample
            nn.Conv2d(in_channels, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 2
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 3
            nn.Conv2d(128, out_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # Upsample back to full BEV resolution
            nn.ConvTranspose2d(out_channels, out_channels, 2, 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, 2, 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, 2, 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def voxelize(
        self, points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert raw point cloud to pillars.

        Args:
            points: [B, N_max, 3+] padded point cloud.

        Returns:
            voxel_features: [M, max_pts, C] point features per pillar.
            voxel_coords: [M, 4] [batch, z, y, x] pillar indices.
            voxel_num_points: [M] number of valid points per pillar.
        """
        B = points.shape[0]
        device = points.device

        all_features = []
        all_coords = []
        all_num_pts = []

        for b in range(B):
            pts = points[b]  # [N, 3+]
            valid = pts[:, 0] != 0  # rough validity check (non-zero position)

            # If all points are zero, skip
            if not valid.any():
                continue

            pts_valid = pts[valid]

            # Compute voxel indices
            x_idx = ((pts_valid[:, 0] - self.x_min) / self.vx).floor().long()
            y_idx = ((pts_valid[:, 1] - self.y_min) / self.vy).floor().long()
            z_idx = ((pts_valid[:, 2] - self.z_min) / self.vz).floor().long()

            # Filter out-of-range points
            mask = (
                (x_idx >= 0) & (x_idx < self.bev_w) &
                (y_idx >= 0) & (y_idx < self.bev_h) &
                (z_idx >= 0)
            )
            x_idx = x_idx[mask]
            y_idx = y_idx[mask]
            z_idx = z_idx[mask]
            pts_filtered = pts_valid[mask]

            if pts_filtered.shape[0] == 0:
                continue

            # Create unique pillar keys
            # Combine batch, z, y, x
            pillar_key = (
                z_idx * (self.bev_h * self.bev_w * 10) +
                y_idx * (self.bev_w * 10) +
                x_idx
            )

            unique_keys, inverse, counts = torch.unique(
                pillar_key, return_inverse=True, return_counts=True
            )
            num_pillars = unique_keys.shape[0]

            if num_pillars > self.max_voxels:
                # Subsample pillars
                perm = torch.randperm(num_pillars, device=device)[:self.max_voxels]
                keep_mask = torch.isin(inverse, perm)
                inverse = inverse[keep_mask]
                pts_filtered = pts_filtered[keep_mask]
                x_idx = x_idx[keep_mask]
                y_idx = y_idx[keep_mask]
                z_idx = z_idx[keep_mask]
                # Recompute
                pillar_key = pillar_key[keep_mask]
                unique_keys, inverse = torch.unique(pillar_key, return_inverse=True)
                num_pillars = unique_keys.shape[0]

            # Allocate pillar tensor
            pillar_feats = torch.zeros(
                num_pillars, self.max_points, pts_filtered.shape[1],
                device=device,
            )
            pillar_num = torch.zeros(num_pillars, dtype=torch.long, device=device)
            pillar_coords = torch.zeros(num_pillars, 4, dtype=torch.long, device=device)

            for p in range(num_pillars):
                p_mask = inverse == p
                p_pts = pts_filtered[p_mask]
                n_pts = min(p_pts.shape[0], self.max_points)
                pillar_feats[p, :n_pts] = p_pts[:n_pts]
                pillar_num[p] = n_pts

                # Get representative coordinates
                ref_idx = p_mask.nonzero()[0]
                pillar_coords[p, 0] = b
                pillar_coords[p, 1] = int(z_idx[ref_idx])
                pillar_coords[p, 2] = int(y_idx[ref_idx])
                pillar_coords[p, 3] = int(x_idx[ref_idx])

            all_features.append(pillar_feats)
            all_coords.append(pillar_coords)
            all_num_pts.append(pillar_num)

        if not all_features:
            # Empty batch: return dummy
            return (
                torch.zeros(1, self.max_points, 3, device=device),
                torch.zeros(1, 4, dtype=torch.long, device=device),
                torch.zeros(1, dtype=torch.long, device=device),
            )

        return (
            torch.cat(all_features, dim=0),
            torch.cat(all_coords, dim=0),
            torch.cat(all_num_pts, dim=0),
        )

    def forward(
        self, points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full PointPillars forward pass.

        Args:
            points: [B, N_max, 3+] padded point cloud.

        Returns:
            bev_feat: [B, bev_out_channels, bev_h, bev_w] BEV features.
            density_map: [B, 1, bev_h, bev_w] pillar density map.
        """
        B = points.shape[0]

        # 1. Voxelize
        voxel_feats, voxel_coords, voxel_num = self.voxelize(points)

        # 2. Pillar Feature Net
        pillar_feats = self.pfn(voxel_feats, voxel_num, voxel_coords)

        # 3. Scatter to BEV
        bev_pseudo, density = self.scatter(pillar_feats, voxel_coords, B)

        # 4. 2D CNN
        bev_feat = self.bev_cnn(bev_pseudo)

        # 5. Upsample density map to match BEV resolution if needed
        if density.shape[-2:] != bev_feat.shape[-2:]:
            density = F.interpolate(
                density,
                size=bev_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return bev_feat, density
