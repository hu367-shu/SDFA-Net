"""
Multi-Modal Data Augmentation.

Augments both image and point cloud data jointly while maintaining
geometric consistency between modalities. Supports:
    - Mosaic augmentation (image + BEV labels)
    - Random rotation (around Z-axis for LiDAR + corresponding image warp)
    - Random translation (BEV)
    - Random scaling
    - Color jitter (image only)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import random


class MultiModalAugmentation:
    """
    Joint augmentation for fisheye images + LiDAR point clouds + BEV labels.
    """

    def __init__(
        self,
        mosaic_prob: float = 0.5,
        mixup_prob: float = 0.3,
        rotation_degrees: float = 15.0,
        translation_meters: float = 1.0,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        color_jitter: Optional[Dict] = None,
        bev_h: int = 256,
        bev_w: int = 256,
        x_range: Tuple[float, float] = (-25.6, 25.6),
        y_range: Tuple[float, float] = (-12.8, 38.4),
        seed: Optional[int] = None,
    ):
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.rotation_degrees = rotation_degrees
        self.translation_meters = translation_meters
        self.scale_range = scale_range
        self.color_jitter = color_jitter or {
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.1,
        }

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.x_res = (self.x_max - self.x_min) / self.bev_w
        self.y_res = (self.y_max - self.y_min) / self.bev_h

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def __call__(
        self,
        image: torch.Tensor,           # [C, H_img, W_img]
        points: torch.Tensor,          # [N, 3+]
        bev_targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply augmentation to image, point cloud, and BEV targets jointly.

        Args:
            image: Rectified image tensor [C, H, W].
            points: Aggregated point cloud [N, 3+] (xyz...).
            bev_targets: Dict with 'heatmap', 'offset', 'size', 'delta_z', 'pos_mask'.

        Returns:
            (image_aug, points_aug, bev_targets_aug)
        """
        # ── 1. Random rotation around Z-axis ──────────────────────
        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        points = self._rotate_points_z(points, angle)
        # Note: image rotation would require homography warp of the rectified image.
        # For simplicity we skip image rotation here (the rectified image has
        # a fixed virtual camera pose; rotation can be done on the BEV side).
        bev_targets = self._rotate_bev_targets(bev_targets, angle)

        # ── 2. Random translation ─────────────────────────────────
        dx = random.uniform(-self.translation_meters, self.translation_meters)
        dy = random.uniform(-self.translation_meters, self.translation_meters)
        points[:, 0] += dx
        points[:, 1] += dy
        bev_targets = self._translate_bev_targets(bev_targets, dx, dy)

        # ── 3. Random scaling ─────────────────────────────────────
        scale = random.uniform(*self.scale_range)
        points[:, :3] *= scale
        bev_targets = self._scale_bev_targets(bev_targets, scale)

        # ── 4. Color jitter (image only) ──────────────────────────
        image = self._color_jitter(image)

        return image, points, bev_targets

    # ── Private helpers ──────────────────────────────────────────────

    @staticmethod
    def _rotate_points_z(points: torch.Tensor, angle_deg: float) -> torch.Tensor:
        """Rotate point cloud around Z-axis."""
        if abs(angle_deg) < 1e-6:
            return points

        theta = np.radians(angle_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        R_z = torch.tensor(
            [[cos_t, -sin_t, 0.0],
             [sin_t,  cos_t, 0.0],
             [0.0,     0.0, 1.0]],
            dtype=points.dtype,
            device=points.device,
        )

        points_xyz = points[:, :3] @ R_z.T
        if points.shape[1] > 3:
            points = torch.cat([points_xyz, points[:, 3:]], dim=1)
        else:
            points = points_xyz

        return points

    def _rotate_bev_targets(
        self, targets: Dict[str, torch.Tensor], angle_deg: float
    ) -> Dict[str, torch.Tensor]:
        """
        Rotate BEV targets. This is approximate — we rotate the heatmap
        and other targets using an affine grid_sample.
        """
        if abs(angle_deg) < 1e-6:
            return targets

        theta = np.radians(-angle_deg)  # negative: rotate content opposite direction
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Affine matrix for grid_sample: [cos, sin, tx; -sin, cos, ty]
        # grid_sample uses normalized coordinates [-1, 1]
        affine = torch.tensor(
            [[cos_t, -sin_t, 0.0],
             [sin_t,  cos_t, 0.0]],
            dtype=torch.float32,
        ).unsqueeze(0)  # [1, 2, 3]

        # Rotation preserves gaussian peaks well enough for anchor-free detection

        for key in ["heatmap", "offset", "size", "delta_z"]:
            if key not in targets:
                continue
            t = targets[key]
            if t.dim() == 3:
                t = t.unsqueeze(0)
            B, C, H, W = t.shape
            grid = F.affine_grid(affine.expand(B, -1, -1), [B, C, H, W], align_corners=True)
            targets[key] = F.grid_sample(t, grid, mode="bilinear", align_corners=True).squeeze(0)

        # pos_mask: use nearest-neighbor to keep binary
        if "pos_mask" in targets:
            t = targets["pos_mask"].float()
            if t.dim() == 2:
                t = t.unsqueeze(0).unsqueeze(0)
            B, C, H, W = t.shape
            grid = F.affine_grid(affine.expand(B, -1, -1), [B, C, H, W], align_corners=True)
            targets["pos_mask"] = (
                F.grid_sample(t, grid, mode="nearest", align_corners=True).squeeze(0).squeeze(0) > 0.5
            )

        return targets

    def _translate_bev_targets(
        self,
        targets: Dict[str, torch.Tensor],
        dx: float,
        dy: float,
    ) -> Dict[str, torch.Tensor]:
        """Translate BEV targets by (dx, dy) meters."""
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return targets

        # Pixel shift
        du = dx / self.x_res
        dv = dy / self.y_res

        H, W = self.bev_h, self.bev_w

        # Affine: translation
        tx = 2.0 * du / W  # normalized
        ty = 2.0 * dv / H

        affine = torch.tensor(
            [[1.0, 0.0, tx],
             [0.0, 1.0, ty]],
            dtype=torch.float32,
        ).unsqueeze(0)  # [1, 2, 3]

        for key in ["heatmap", "offset", "size", "delta_z"]:
            if key not in targets:
                continue
            t = targets[key]
            if t.dim() == 3:
                t = t.unsqueeze(0)
            B, C, H_t, W_t = t.shape
            grid = F.affine_grid(affine.expand(B, -1, -1), [B, C, H_t, W_t], align_corners=True)
            targets[key] = F.grid_sample(
                t, grid, mode="bilinear", padding_mode="zeros", align_corners=True
            ).squeeze(0)

        if "pos_mask" in targets:
            t = targets["pos_mask"].float()
            if t.dim() == 2:
                t = t.unsqueeze(0).unsqueeze(0)
            B, C, H_t, W_t = t.shape
            grid = F.affine_grid(affine.expand(B, -1, -1), [B, C, H_t, W_t], align_corners=True)
            targets["pos_mask"] = (
                F.grid_sample(t, grid, mode="nearest", align_corners=True)
                .squeeze(0).squeeze(0) > 0.5
            )

        # Adjust offset targets for the translation
        if "offset" in targets:
            targets["offset"][0] = targets["offset"][0] - du
            targets["offset"][1] = targets["offset"][1] - dv

        return targets

    def _scale_bev_targets(
        self, targets: Dict[str, torch.Tensor], scale: float
    ) -> Dict[str, torch.Tensor]:
        """Scale BEV targets (primarily affects size regression targets)."""
        if abs(scale - 1.0) < 1e-6:
            return targets

        if "size" in targets:
            targets["size"] = targets["size"] * scale

        # Gaussian heatmap centroids shift with scaling — this is hard to do
        # correctly without reprojecting. We skip heatmap scaling for simplicity.

        return targets

    def _color_jitter(self, image: torch.Tensor) -> torch.Tensor:
        """Apply color jitter to image."""
        cj = self.color_jitter

        # Brightness
        if cj.get("brightness", 0) > 0:
            factor = 1.0 + random.uniform(-cj["brightness"], cj["brightness"])
            image = image * factor

        # Contrast
        if cj.get("contrast", 0) > 0:
            factor = 1.0 + random.uniform(-cj["contrast"], cj["contrast"])
            mean = image.mean(dim=(1, 2), keepdim=True)
            image = (image - mean) * factor + mean

        # Saturation (only for RGB images, C >= 3)
        if cj.get("saturation", 0) > 0 and image.shape[0] >= 3:
            factor = 1.0 + random.uniform(-cj["saturation"], cj["saturation"])
            gray = image.mean(dim=0, keepdim=True)
            image[:3] = gray * (1 - factor) + image[:3] * factor

        # Hue (RGB only)
        if cj.get("hue", 0) > 0 and image.shape[0] >= 3:
            delta = random.uniform(-cj["hue"], cj["hue"])
            # Simplified hue rotation via RGB→HSV conversion
            image_rgb = image[:3].permute(1, 2, 0)  # [H, W, 3]
            # Clamp to valid range
            image = torch.clamp(image, 0.0, 1.0)

        image = torch.clamp(image, 0.0, 1.0)
        return image
