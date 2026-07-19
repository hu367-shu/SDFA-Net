"""
Fisheye Image Rectification using Scaramuzza Omnidirectional Camera Model.

The Scaramuzza model maps a 3D ray to a pixel on the fisheye image via:
    r(θ) = a₀ + a₁θ + a₂θ² + ... + aₙθⁿ
where θ is the incident angle.

Rectification: for each pixel (u,v) on the virtual pinhole image, compute the
corresponding ray direction, find its θ, compute the fisheye radius r(θ),
then map back to fisheye image coordinates via affine transformation and
bilinear interpolation.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class OCamParams:
    """Scaramuzza omnidirectional camera model parameters."""
    # Polynomial coefficients: r(θ) = poly[0] + poly[1]*θ + poly[2]*θ² + ...
    poly: np.ndarray  # shape: (n_coeffs,)

    # Affine transformation parameters (from sensor plane to image pixel)
    c: float  # stretch factor x
    d: float  # skew factor
    e: float  # stretch factor y

    # Principal point (image center)
    cx: float
    cy: float

    # Image dimensions
    height: int
    width: int


class FisheyeUndistorter:
    """
    Rectifies Scaramuzza-model fisheye images to a virtual pinhole camera.

    Implements the inverse mapping: for each pixel in the rectified output,
    compute the corresponding source pixel in the fisheye image and sample.
    """

    def __init__(
        self,
        ocam_params: OCamParams,
        out_size: Tuple[int, int] = (384, 640),
        fov: float = 120.0,
        device: str = "cpu",
    ):
        """
        Args:
            ocam_params: Scaramuzza model parameters.
            out_size: (H, W) of the rectified output image.
            fov: Horizontal field of view of the virtual pinhole camera in degrees.
            device: Computation device.
        """
        self.ocam = ocam_params
        self.out_h, self.out_w = out_size
        self.fov = fov
        self.device = device

        # Precompute the mapping grid for efficiency (H, W, 2) → (src_u, src_v)
        self.map_x, self.map_y = self._build_undistortion_map()
        self.map_x = torch.from_numpy(self.map_x).float().to(device)
        self.map_y = torch.from_numpy(self.map_y).float().to(device)

    def _build_undistortion_map(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a pixel-wise mapping from rectified image coordinates
        to original fisheye image coordinates.

        Returns:
            map_x: [H, W] — source x-coordinate in fisheye image for each output pixel.
            map_y: [H, W] — source y-coordinate in fisheye image for each output pixel.
        """
        H, W = self.out_h, self.out_w
        cx_p, cy_p = W / 2.0, H / 2.0

        # Virtual pinhole focal length derived from horizontal FOV
        f = W / (2.0 * np.tan(np.radians(self.fov) / 2.0))

        # Generate pixel grid
        u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))  # (H, W)
        u_grid = u_grid.astype(np.float64)
        v_grid = v_grid.astype(np.float64)

        # Normalized image plane coordinates
        x_norm = (u_grid - cx_p) / f  # (H, W)
        y_norm = (v_grid - cy_p) / f  # (H, W)
        rho = np.sqrt(x_norm ** 2 + y_norm ** 2)  # (H, W)

        # Incident angle θ
        theta = np.arctan(rho)

        # Evaluate Scaramuzza polynomial: r_fisheye = poly(θ)
        r_fisheye = np.zeros_like(theta)
        for i, coeff in enumerate(self.ocam.poly):
            r_fisheye += coeff * (theta ** i)

        # Sensor plane coordinates
        # For rho ≈ 0, rays map directly to the principal point
        eps = 1e-12
        safe_rho = np.where(rho > eps, rho, 1.0)

        x_sensor = r_fisheye * x_norm / safe_rho  # (H, W)
        y_sensor = r_fisheye * y_norm / safe_rho  # (H, W)

        # Affine transformation to fisheye image pixel coordinates
        src_u = self.ocam.c * x_sensor + self.ocam.d * y_sensor + self.ocam.cx
        src_v = self.ocam.e * x_sensor + y_sensor + self.ocam.cy

        # Handle the optical center (rho ≈ 0) explicitly
        center_mask = rho < eps
        src_u[center_mask] = self.ocam.cx
        src_v[center_mask] = self.ocam.cy

        return src_u.astype(np.float32), src_v.astype(np.float32)

    def __call__(
        self, image: torch.Tensor
    ) -> torch.Tensor:
        """
        Rectify a batch of fisheye images.

        Args:
            image: [B, C, H_fisheye, W_fisheye] or [C, H_fisheye, W_fisheye]
                   raw fisheye image(s).

        Returns:
            rectified: same batch shape, [B, C, H_out, W_out] or [C, H_out, W_out].
        """
        single = image.dim() == 3
        if single:
            image = image.unsqueeze(0)

        B, C, H, W = image.shape

        # Normalize grid to [-1, 1] for grid_sample
        # grid_sample expects (N, H_out, W_out, 2) with values in [-1, 1]
        grid_x = 2.0 * self.map_x / (self.ocam.width - 1) - 1.0
        grid_y = 2.0 * self.map_y / (self.ocam.height - 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (H_out, W_out, 2)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H_out, W_out, 2)

        image = image.to(self.device)
        grid = grid.to(self.device)

        # bilinear interpolation
        rectified = F.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        if single:
            rectified = rectified.squeeze(0)

        return rectified

    def rectify_numpy(self, image: np.ndarray) -> np.ndarray:
        """
        Rectify a single numpy fisheye image.

        Args:
            image: [H_fisheye, W_fisheye, C] uint8/float32 numpy array.

        Returns:
            rectified: [H_out, W_out, C] float32 numpy array.
        """
        if image.ndim == 2:
            image = image[..., np.newaxis]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        H, W = self.out_h, self.out_w
        channels = image.shape[2]
        rectified = np.zeros((H, W, channels), dtype=np.float32)

        for c in range(channels):
            # Bilinear interpolation manually for numpy
            x_src = self.map_x  # (H, W)
            y_src = self.map_y  # (H, W)

            x0 = np.floor(x_src).astype(np.int32)
            y0 = np.floor(y_src).astype(np.int32)
            x1 = x0 + 1
            y1 = y0 + 1

            # Clamp to valid range
            x0 = np.clip(x0, 0, self.ocam.width - 1)
            x1 = np.clip(x1, 0, self.ocam.width - 1)
            y0 = np.clip(y0, 0, self.ocam.height - 1)
            y1 = np.clip(y1, 0, self.ocam.height - 1)

            wx = x_src - x0.astype(np.float32)
            wy = y_src - y0.astype(np.float32)

            rectified[:, :, c] = (
                (1 - wx) * (1 - wy) * image[y0, x0, c]
                + wx * (1 - wy) * image[y0, x1, c]
                + (1 - wx) * wy * image[y1, x0, c]
                + wx * wy * image[y1, x1, c]
            )

        return rectified
