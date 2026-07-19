"""
SDFA-Net: Complete Multi-Modal Manhole Cover Detection & Diagnosis Network.

Architecture overview:
    ┌──────────────────────────────────────────────────────────────┐
    │  Fisheye Image          LiDAR Point Cloud                    │
    │      │                       │                                │
    │  [Rectification]      [Multi-frame Aggregation]              │
    │      │                       │                                │
    │  YOLOv11 Backbone     PointPillars Backbone                  │
    │      │                       │                                │
    │  Image Features        LiDAR BEV Features + Density Map      │
    │      │                       │                                │
    │      └───────┬───────────────┘                                │
    │              │                                                 │
    │         SDS-VT (View Transform)                               │
    │              │                                                 │
    │         Camera BEV Features + Depth Prob + Entropy            │
    │              │                                                 │
    │              └───────┬───────────────┘                        │
    │                      │                                         │
    │                 FA-Fusion                                     │
    │                      │                                         │
    │                 Fused BEV                                     │
    │                      │                                         │
    │                  Geo-Head                                     │
    │                   ╱  ╱  ╲  ╲                                 │
    │            Heatmap Offset Size Δz                             │
    └──────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .image_backbone import YOLOv11Backbone, ResNetBackbone
from .pointpillars import PointPillarsBackbone
from .sds_vt import SDSVT
from .fa_fusion import FAFusion
from .geo_head import GeoHead


class SDFANet(nn.Module):
    """
    SDFA-Net: Fisheye Image + LiDAR Point Cloud Multi-Modal
    Manhole Cover Detection and Disease Diagnosis Network.

    Key components:
        - SDS-VT: Semi-Dense Depth-Supervised View Transform
        - FA-Fusion: Fidelity-Aware Dynamic Fusion
        - Geo-Head: Geometry-Aware Anchor-Free Detection Head
    """

    def __init__(self, cfg: dict):
        """
        Args:
            cfg: Full configuration dict (typically loaded from sdfanet.yaml).
        """
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.get("model", {})

        # ── Image Backbone ────────────────────────────────────
        image_cfg = model_cfg.get("image", {})
        self.image_backbone = YOLOv11Backbone(
            model_name=image_cfg.get("backbone", "yolov11n"),
            pretrained=image_cfg.get("pretrained", True),
            out_channels=image_cfg.get("out_channels", [128, 256, 512]),
            neck_channels=image_cfg.get("neck_channels", 256),
        )

        # ── LiDAR Backbone ────────────────────────────────────
        lidar_cfg = model_cfg.get("lidar", {})
        self.lidar_backbone = PointPillarsBackbone(
            point_cloud_range=tuple(lidar_cfg.get("point_cloud_range",
                [-25.6, -25.6, -2.0, 25.6, 25.6, 4.0])),
            voxel_size=tuple(lidar_cfg.get("voxel_size", [0.2, 0.2, 6.0])),
            max_points_per_voxel=lidar_cfg.get("max_points_per_voxel", 32),
            max_voxels=lidar_cfg.get("max_voxels", 16000),
            pillar_channels=lidar_cfg.get("pillar_channels", 64),
            bev_out_channels=lidar_cfg.get("bev_out_channels", 256),
        )

        # ── SDS-VT ────────────────────────────────────────────
        sds_cfg = model_cfg.get("sds_vt", {})
        self.sds_vt = SDSVT(
            in_channels=image_cfg.get("neck_channels", 256),
            depth_bins=sds_cfg.get("depth_bins", 64),
            depth_min=sds_cfg.get("depth_min", 0.5),
            depth_max=sds_cfg.get("depth_max", 30.0),
            image_h=sds_cfg.get("image_size", [384, 640])[0],
            image_w=sds_cfg.get("image_size", [384, 640])[1],
            x_range=tuple(sds_cfg.get("x_range", [-25.6, 25.6])),
            y_range=tuple(sds_cfg.get("y_range", [-12.8, 38.4])),
            bev_h=sds_cfg.get("bev_h", 256),
            bev_w=sds_cfg.get("bev_w", 256),
        )

        # ── FA-Fusion ─────────────────────────────────────────
        fusion_cfg = model_cfg.get("fusion", {})
        self.fa_fusion = FAFusion(
            c_cam=fusion_cfg.get("c_cam", 256),
            c_lidar=fusion_cfg.get("c_lidar", 256),
            c_out=fusion_cfg.get("c_fuse", 256),
            lambda_density=fusion_cfg.get("lambda_density", 0.7),
        )

        # ── Geo-Head ──────────────────────────────────────────
        geo_cfg = model_cfg.get("geo_head", {})
        self.geo_head = GeoHead(
            in_channels=fusion_cfg.get("c_fuse", 256),
            num_classes=geo_cfg.get("num_classes", 5),
        )

        # ── Store BEV geometry params for decoding ────────────
        self.bev_h = sds_cfg.get("bev_h", 256)
        self.bev_w = sds_cfg.get("bev_w", 256)
        self.x_range = tuple(sds_cfg.get("x_range", [-25.6, 25.6]))
        self.y_range = tuple(sds_cfg.get("y_range", [-12.8, 38.4]))

    def forward(
        self,
        image: torch.Tensor,
        points: torch.Tensor,
        calib: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Full SDFA-Net forward pass.

        Args:
            image: [B, 3, H_img, W_img] rectified fisheye images.
            points: [B, N_max, 3+] padded LiDAR point clouds.
            calib: Dict with:
                - 'K_virtual': [B, 3, 3] virtual pinhole intrinsic.
                - 'T_lidar_cam': [B, 4, 4] LiDAR → Camera transform.

        Returns:
            outputs dict:
                'heatmap':   [B, num_cls, H_bev, W_bev]
                'offset':    [B, 2, H_bev, W_bev]
                'size':      [B, 2, H_bev, W_bev]
                'delta_z':   [B, 1, H_bev, W_bev]
                'depth_prob': [B, D, H_img, W_img]  (for loss computation)
        """
        K = calib["K_virtual"]
        T_lidar_cam = calib["T_lidar_cam"]

        # ── 1. Image branch ──────────────────────────────────
        img_feat = self.image_backbone(image)
        # [B, C_img, H_feat, W_feat]

        # ── 2. LiDAR branch ──────────────────────────────────
        lidar_bev, density_map = self.lidar_backbone(points)
        # lidar_bev: [B, C_lidar, H_bev, W_bev]
        # density_map: [B, 1, H_bev, W_bev]

        # ── 3. SDS-VT: image → BEV ──────────────────────────
        cam_bev, depth_prob, entropy_bev = self.sds_vt(
            img_feat, K, T_lidar_cam,
        )
        # cam_bev: [B, C_cam, H_bev, W_bev]
        # depth_prob: [B, D, H_img, W_img]
        # entropy_bev: [B, 1, H_bev, W_bev]

        # ── 4. FA-Fusion ────────────────────────────────────
        fused = self.fa_fusion(
            cam_bev, lidar_bev, density_map, entropy_bev,
        )
        # [B, C_fuse, H_bev, W_bev]

        # ── 5. Geo-Head ─────────────────────────────────────
        detections = self.geo_head(fused)

        # ── 6. Attach depth_prob for loss computation ───────
        detections["depth_prob"] = depth_prob

        return detections

    def get_bev_params(self) -> Dict:
        """Return BEV grid parameters for decoding detections."""
        return {
            "bev_h": self.bev_h,
            "bev_w": self.bev_w,
            "x_min": self.x_range[0],
            "x_max": self.x_range[1],
            "y_min": self.y_range[0],
            "y_max": self.y_range[1],
            "x_res": (self.x_range[1] - self.x_range[0]) / self.bev_w,
            "y_res": (self.y_range[1] - self.y_range[0]) / self.bev_h,
        }

    @classmethod
    def from_config_file(cls, config_path: str) -> "SDFANet":
        """
        Build SDFANet from a YAML config file.

        Args:
            config_path: Path to sdfanet.yaml.

        Returns:
            SDFANet instance.
        """
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cls(cfg)


def build_sdfanet(cfg: dict) -> SDFANet:
    """
    Factory function to build SDFA-Net from a configuration dict.

    Args:
        cfg: Configuration dictionary.

    Returns:
        SDFANet instance.
    """
    return SDFANet(cfg)
