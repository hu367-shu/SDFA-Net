"""
FA-Fusion: Fidelity-Aware Dynamic Fusion.

Fuses camera BEV and LiDAR BEV features using a physical prior gate
followed by channel + spatial attention. The confidence gate is built from
LiDAR point density and depth prediction entropy, so the model dynamically
decides which modality to trust at each BEV location.

Paper formulas (5)-(9):
    (5)  H = -Σ P_d log P_d              → depth entropy
    (6)  C = σ(λρ - (1-λ)H)             → physical confidence gate
    (7)  A_c = σ(MLP(GAP([F_cam^gate, F_lidar])))  → channel attention
    (8)  A_s = σ(Conv(F_c))              → spatial attention
    (9)  F_fuse = A_s ⊙ F_c              → final fused feature
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ChannelAttention2D(nn.Module):
    """
    Squeeze-and-Excitation style channel attention.
    Paper formula (7): A_c = σ(MLP(GAP(x)))
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]

        Returns:
            weighted: [B, C, H, W] channel-weighted features.
        """
        B, C, H, W = x.shape
        gap = F.adaptive_avg_pool2d(x, 1).view(B, C)   # [B, C]
        weight = self.sigmoid(self.mlp(gap)).view(B, C, 1, 1)
        return x * weight


class SpatialAttention2D(nn.Module):
    """
    Convolution-based spatial attention.
    Paper formula (8): A_s = σ(Conv(x))
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]

        Returns:
            weighted: [B, C, H, W] spatially-weighted features.
        """
        # Compute avg and max along channel dim
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out = torch.max(x, dim=1, keepdim=True)[0]  # [B, 1, H, W]
        combined = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]

        weight = self.sigmoid(self.conv(combined))  # [B, 1, H, W]
        return x * weight


class FAFusion(nn.Module):
    """
    Fidelity-Aware Fusion module.

    Combines camera BEV and LiDAR BEV features with:
        1. Physical confidence gating (LiDAR density + depth entropy)
        2. Channel attention
        3. Spatial attention
    """

    def __init__(
        self,
        c_cam: int = 256,
        c_lidar: int = 256,
        c_out: int = 256,
        lambda_density: float = 0.7,
    ):
        """
        Args:
            c_cam: Camera BEV feature channels.
            c_lidar: LiDAR BEV feature channels.
            c_out: Fused output channels.
            lambda_density: Balance weight λ between density and entropy
                            in the physical confidence map (0.5–1.0).
        """
        super().__init__()
        self.lambda_density = lambda_density

        total_channels = c_cam + c_lidar

        # 1x1 conv to reduce concat dimension
        self.reduce = nn.Conv2d(total_channels, c_out, kernel_size=1)

        # Channel attention (operates on concatenated features)
        self.channel_attn = ChannelAttention2D(total_channels)

        # Spatial attention (operates on channel-attended features)
        self.spatial_attn = SpatialAttention2D(kernel_size=7)

        # Output projection
        self.out_conv = nn.Sequential(
            nn.Conv2d(total_channels, c_out, kernel_size=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

        # Learnable gate scaling
        self.gate_scale = nn.Parameter(torch.ones(1))

    def forward(
        self,
        cam_bev: torch.Tensor,
        lidar_bev: torch.Tensor,
        density_map: torch.Tensor,
        entropy_bev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cam_bev: [B, c_cam, H, W] camera BEV features from SDS-VT.
            lidar_bev: [B, c_lidar, H, W] LiDAR BEV features from PointPillars.
            density_map: [B, 1, H, W] LiDAR point density per BEV cell.
            entropy_bev: [B, 1, H, W] depth prediction entropy in BEV.

        Returns:
            fused: [B, c_out, H, W] fused BEV features.
        """
        # ── 1. Physical confidence map ──────────────────────────
        # Normalize density and entropy to [0, 1] per batch
        density_norm = self._normalize_map(density_map)
        entropy_norm = self._normalize_map(entropy_bev)

        # C = σ(λ * ρ - (1-λ) * H + gate_scale)
        conf_raw = (
            self.lambda_density * density_norm
            - (1.0 - self.lambda_density) * entropy_norm
        )
        confidence = torch.sigmoid(conf_raw * self.gate_scale)  # [B, 1, H, W]

        # ── 2. Physical gating ──────────────────────────────────
        cam_gated = cam_bev * confidence  # [B, c_cam, H, W]

        # ── 3. Concatenate ──────────────────────────────────────
        x = torch.cat([cam_gated, lidar_bev], dim=1)  # [B, c_cam + c_lidar, H, W]

        # ── 4. Channel attention ────────────────────────────────
        x_c = self.channel_attn(x)  # [B, total_c, H, W]

        # ── 5. Spatial attention ────────────────────────────────
        x_s = self.spatial_attn(x_c)  # [B, total_c, H, W]

        # ── 6. Output projection ────────────────────────────────
        fused = self.out_conv(x_s)  # [B, c_out, H, W]

        return fused

    @staticmethod
    def _normalize_map(x: torch.Tensor) -> torch.Tensor:
        """
        Min-max normalize per batch to [0, 1].

        Args:
            x: [B, 1, H, W]

        Returns:
            normalized: [B, 1, H, W] in [0, 1].
        """
        B = x.shape[0]
        x_flat = x.view(B, -1)
        x_min = x_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        x_max = x_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)

        denom = (x_max - x_min).clamp(min=1e-8)
        return (x - x_min) / denom


def build_confidence_map(
    density_map: torch.Tensor,
    entropy_map: torch.Tensor,
    lambda_density: float = 0.7,
) -> torch.Tensor:
    """
    Standalone function matching paper pseudocode.

    C = σ(λ * ρ_norm - (1-λ) * H_norm)

    Args:
        density_map: [B, 1, H, W]
        entropy_map: [B, 1, H, W]
        lambda_density: Balance weight in [0.5, 1.0].

    Returns:
        confidence: [B, 1, H, W] physical confidence map in [0, 1].
    """
    density_norm = FAFusion._normalize_map(density_map)
    entropy_norm = FAFusion._normalize_map(entropy_map)

    conf = torch.sigmoid(
        lambda_density * density_norm
        - (1.0 - lambda_density) * entropy_norm
    )
    return conf
