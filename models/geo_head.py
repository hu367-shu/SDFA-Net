"""
Geo-Head: Geometry-Aware Anchor-Free Detection Head.

An anchor-free, decoupled detection head that outputs:
    1. Center heatmap (per-class Gaussian peaks)
    2. Sub-pixel center offset (dx, dy)
    3. Bounding box size (width, length) in meters
    4. Relative elevation offset Δz (manhole cover → road surface)

The head operates on fused BEV features and directly predicts all 8 output
channels (5-class heatmap + 2 offset + 2 size + 1 delta_z).

Paper formula (11): Y(x,y) = exp(-(d²)/(2σ²)) for Gaussian heatmap.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class GeoHead(nn.Module):
    """
    Geometry-aware anchor-free detection head.

    Three parallel branches after a shared convolution:
        - Heatmap head: predicts class-wise Gaussian center maps
        - Offset head: sub-pixel center refinement
        - Size head: bounding box width & length (meters)
        - Height head: relative elevation Δz

    All heads operate at full BEV resolution.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 5,
    ):
        """
        Args:
            in_channels: Input fused BEV feature channels.
            num_classes: Number of manhole cover disease classes.
                         Default 5: good, sunken, raised, broken, missing.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Shared feature refinement
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # Heatmap branch → [num_classes, H, W]
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1),
        )

        # Offset branch → [2, H, W] (dx, dy in pixel units)
        self.offset_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, 1),
        )

        # Size branch → [2, H, W] (width, length in meters)
        self.size_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, 1),
        )

        # Height branch → [1, H, W] (Δz in meters)
        self.height_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize head weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Bias initialization for the final conv of heatmap head
        # Set to logit for prior probability ~0.01 (helps with convergences)
        prior_prob = 0.01
        bias_init = -torch.log(torch.tensor((1.0 - prior_prob) / prior_prob))
        nn.init.constant_(self.heatmap_head[-1].bias, bias_init)

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, C, H_bev, W_bev] fused BEV features.

        Returns:
            outputs dict:
                'heatmap': [B, num_classes, H, W] center probability (sigmoid-activated).
                'offset':  [B, 2, H, W] sub-pixel offset (dx_px, dy_px).
                'size':    [B, 2, H, W] bounding box size (w_meters, l_meters).
                'delta_z': [B, 1, H, W] relative elevation offset in meters.
        """
        # Shared refinement
        feat = self.shared_conv(x)

        # Heatmap: sigmoid to get [0, 1] per-class probabilities
        heatmap = torch.sigmoid(self.heatmap_head(feat))

        # Offset: unconstrained (can be positive or negative in pixel units)
        offset = self.offset_head(feat)

        # Size: ReLU ensures non-negative dimensions
        size = F.relu(self.size_head(feat))

        # Height: unconstrained (can be positive or negative in meters)
        delta_z = self.height_head(feat)

        return {
            "heatmap": heatmap,
            "offset": offset,
            "size": size,
            "delta_z": delta_z,
        }


def draw_gaussian_heatmap(
    heatmap: torch.Tensor,
    center_x: float,
    center_y: float,
    cls_id: int,
    sigma: float = 1.5,
) -> torch.Tensor:
    """
    Draw a Gaussian peak on a single-class heatmap.

    Paper formula (11):
        Y(x,y) = exp(-((x - x_c)² + (y - y_c)²) / (2σ²))

    This function modifies the heatmap in-place for efficiency.

    Args:
        heatmap: [H, W] single-class heatmap to draw on (modified in place).
        center_x, center_y: Center position in BEV pixel coordinates.
        cls_id: Class index (unused here, for API consistency).
        sigma: Gaussian standard deviation in pixels (default 1.5).

    Returns:
        heatmap: Modified heatmap (same tensor, for chaining).
    """
    H, W = heatmap.shape
    sigma_t = max(sigma, 1.0)
    radius = int(min(3 * sigma_t, max(H, W)))

    x_min = max(0, int(center_x - radius))
    x_max = min(W, int(center_x + radius) + 1)
    y_min = max(0, int(center_y - radius))
    y_max = min(H, int(center_y + radius) + 1)

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
            value = torch.exp(-dist_sq / (2 * sigma_t ** 2))
            heatmap[y, x] = max(heatmap[y, x], value)

    return heatmap
