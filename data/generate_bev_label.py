"""
BEV Ground-Truth Label Generation.

Converts 3D annotations (center position, dimensions, elevation offset,
disease class) into BEV-plane training targets:
    - Gaussian heatmap per class
    - Center offset (sub-pixel refinement)
    - Bounding box size (width, length)
    - Relative elevation delta_z
    - Positive pixel mask for regression losses
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional


class BEVLabelGenerator:
    """
    Generates BEV training targets from 3D manhole cover annotations.
    """

    def __init__(
        self,
        bev_h: int = 256,
        bev_w: int = 256,
        x_range: Tuple[float, float] = (-25.6, 25.6),
        y_range: Tuple[float, float] = (-12.8, 38.4),
        num_classes: int = 5,
        gaussian_sigma: float = 1.5,
        positive_radius: int = 1,
    ):
        """
        Args:
            bev_h: BEV grid height in pixels.
            bev_w: BEV grid width in pixels.
            x_range: (x_min, x_max) BEV X range in meters.
            y_range: (y_min, y_max) BEV Y range in meters.
            num_classes: Number of disease classes (default 5).
            gaussian_sigma: σ for Gaussian heatmap radius (in pixels).
            positive_radius: Radius (in pixels) around each center that counts as positive
                             for offset/size/height regression supervision.
        """
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.num_classes = num_classes
        self.gaussian_sigma = gaussian_sigma
        self.positive_radius = positive_radius

        self.x_res = (self.x_max - self.x_min) / self.bev_w
        self.y_res = (self.y_max - self.y_min) / self.bev_h

    def world_to_bev(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates (meters) to BEV pixel indices.

        Args:
            x, y: World coordinates in meters (LiDAR frame).

        Returns:
            (u, v): BEV pixel coordinates (column, row).
        """
        u = int(np.floor((x - self.x_min) / self.x_res))
        v = int(np.floor((y - self.y_min) / self.y_res))
        return u, v

    def bev_to_world(self, u: float, v: float) -> Tuple[float, float]:
        """
        Convert BEV pixel indices to world coordinates (grid center).

        Args:
            u, v: BEV pixel coordinates.

        Returns:
            (x, y): World coordinates in meters.
        """
        x = (u + 0.5) * self.x_res + self.x_min
        y = (v + 0.5) * self.y_res + self.y_min
        return x, y

    def generate(
        self,
        boxes: np.ndarray,
        classes: np.ndarray,
        delta_zs: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Generate BEV targets for one sample.

        Args:
            boxes: [M, 4] (cx, cy, w, l) in meters (LiDAR frame).
            classes: [M] int, disease class indices (0-4).
            delta_zs: [M] float, relative elevation offsets from road surface.

        Returns:
            targets dict:
                'heatmap':   [num_classes, H, W] float32 Gaussian heatmap
                'offset':    [2, H, W] float32 center sub-pixel offset
                'size':      [2, H, W] float32 box width & length
                'delta_z':   [1, H, W] float32 elevation offset
                'pos_mask':  [H, W] bool, positive sample mask
        """
        heatmap = np.zeros((self.num_classes, self.bev_h, self.bev_w), dtype=np.float32)
        offset = np.zeros((2, self.bev_h, self.bev_w), dtype=np.float32)
        size = np.zeros((2, self.bev_h, self.bev_w), dtype=np.float32)
        delta_z = np.zeros((1, self.bev_h, self.bev_w), dtype=np.float32)
        pos_mask = np.zeros((self.bev_h, self.bev_w), dtype=bool)

        for i in range(len(boxes)):
            cx, cy, w, l = boxes[i]
            cls_id = int(classes[i])
            dz = float(delta_zs[i])

            # Convert to BEV pixel
            u_center, v_center = self.world_to_bev(cx, cy)

            # Skip objects outside BEV
            if not self._inside_bev(u_center, v_center):
                continue

            # Draw Gaussian heatmap
            self._draw_gaussian(heatmap[cls_id], u_center, v_center)

            # For each pixel within positive_radius, set regression targets
            u_int = int(np.floor(u_center))
            v_int = int(np.floor(v_center))

            for dv in range(-self.positive_radius, self.positive_radius + 1):
                for du in range(-self.positive_radius, self.positive_radius + 1):
                    pu = u_int + du
                    pv = v_int + dv
                    if not self._inside_bev(pu, pv):
                        continue

                    # Offset: fractional part of center position in pixels
                    offset[0, pv, pu] = u_center - pu
                    offset[1, pv, pu] = v_center - pv

                    # Size in meters (normalized by grid resolution for loss stability)
                    size[0, pv, pu] = w
                    size[1, pv, pu] = l

                    # Elevation offset
                    delta_z[0, pv, pu] = dz

                    pos_mask[pv, pu] = True

        return {
            "heatmap": heatmap,
            "offset": offset,
            "size": size,
            "delta_z": delta_z,
            "pos_mask": pos_mask,
        }

    def generate_batch(
        self,
        boxes_list: List[np.ndarray],
        classes_list: List[np.ndarray],
        delta_zs_list: List[np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Batch version of generate().

        Returns:
            targets dict with batch dimension [B, ...].
        """
        B = len(boxes_list)
        heatmap_b = np.zeros((B, self.num_classes, self.bev_h, self.bev_w), dtype=np.float32)
        offset_b = np.zeros((B, 2, self.bev_h, self.bev_w), dtype=np.float32)
        size_b = np.zeros((B, 2, self.bev_h, self.bev_w), dtype=np.float32)
        delta_z_b = np.zeros((B, 1, self.bev_h, self.bev_w), dtype=np.float32)
        pos_mask_b = np.zeros((B, self.bev_h, self.bev_w), dtype=bool)

        for i in range(B):
            targets = self.generate(
                boxes_list[i],
                classes_list[i],
                delta_zs_list[i],
            )
            heatmap_b[i] = targets["heatmap"]
            offset_b[i] = targets["offset"]
            size_b[i] = targets["size"]
            delta_z_b[i] = targets["delta_z"]
            pos_mask_b[i] = targets["pos_mask"]

        return {
            "heatmap": heatmap_b,
            "offset": offset_b,
            "size": size_b,
            "delta_z": delta_z_b,
            "pos_mask": pos_mask_b,
        }

    # ── Private helpers ──────────────────────────────────────────────

    def _inside_bev(self, u: int, v: int) -> bool:
        return 0 <= u < self.bev_w and 0 <= v < self.bev_h

    def _draw_gaussian(
        self,
        heatmap: np.ndarray,
        u_center: float,
        v_center: float,
    ) -> None:
        """
        Draw a 2D Gaussian on a single-class heatmap.

        Paper formula (11): Y(x,y) = exp(-((x-xc)² + (y-yc)²) / (2σ²))

        Args:
            heatmap: [H, W] heatmap for one class.
            u_center, v_center: Center position in BEV pixel coordinates.
        """
        H, W = heatmap.shape
        sigma = self.gaussian_sigma
        radius = int(np.ceil(3 * sigma))  # 3σ covers ~99.7%

        u_min = max(0, int(np.floor(u_center - radius)))
        u_max = min(W, int(np.ceil(u_center + radius)) + 1)
        v_min = max(0, int(np.floor(v_center - radius)))
        v_max = min(H, int(np.ceil(v_center + radius)) + 1)

        for v in range(v_min, v_max):
            for u in range(u_min, u_max):
                val = np.exp(
                    -((u - u_center) ** 2 + (v - v_center) ** 2) / (2 * sigma**2)
                )
                if val > heatmap[v, u]:
                    heatmap[v, u] = val
