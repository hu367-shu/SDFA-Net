"""
SDFA-Net Total Loss.

End-to-end joint training loss:
    L_total = L_hm + λ_off * L_off + λ_size * L_size + λ_z * L_z + λ_d * L_depth

where:
    - L_hm: Focal loss on center heatmaps
    - L_off: Smooth L1 on center offset regression
    - L_size: Smooth L1 on box width/length regression
    - L_z: Smooth L1 on elevation offset Δz regression
    - L_depth: Semi-dense depth cross-entropy (SDS-VT)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .depth_loss import SemiDenseDepthLoss, semi_dense_depth_loss
from .detection_loss import (
    FocalLoss,
    SmoothL1LossMasked,
    heatmap_loss,
    offset_loss,
    size_loss,
    height_loss,
)


class SDFANetLoss(nn.Module):
    """
    Combined multi-task loss for SDFA-Net end-to-end training.

    Supports the full training objective:
        - Detection: heatmap + offset + size + height regression
        - View Transform: semi-dense depth supervision
    """

    def __init__(
        self,
        loss_weights: Optional[Dict[str, float]] = None,
        focal_alpha: float = 2.0,
        focal_beta: float = 4.0,
    ):
        """
        Args:
            loss_weights: Dict mapping loss component names to weights.
                          Default: hm=1.0, offset=1.0, size=1.0, height=2.0, depth=0.5.
            focal_alpha: Focal loss α parameter (penalty for hard positives).
            focal_beta: Focal loss β parameter (penalty for easy negatives).
        """
        super().__init__()

        self.loss_weights = loss_weights or {
            "hm": 1.0,
            "offset": 1.0,
            "size": 1.0,
            "height": 2.0,
            "depth": 0.5,
        }

        # Loss modules
        self.heatmap_loss_fn = FocalLoss(alpha=focal_alpha, beta=focal_beta)
        self.offset_loss_fn = SmoothL1LossMasked(beta=1.0 / 9.0)
        self.size_loss_fn = SmoothL1LossMasked(beta=0.1)
        self.height_loss_fn = SmoothL1LossMasked(beta=0.1)
        self.depth_loss_fn = SemiDenseDepthLoss(reduction="mean")

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss.

        Args:
            predictions: Model outputs:
                - 'heatmap': [B, num_cls, H_bev, W_bev] sigmoid-activated
                - 'offset': [B, 2, H_bev, W_bev]
                - 'size': [B, 2, H_bev, W_bev]
                - 'delta_z': [B, 1, H_bev, W_bev]
                - 'depth_prob': [B, D, H_img, W_img]
            targets: Ground-truth targets:
                - 'heatmap': [B, num_cls, H_bev, W_bev] Gaussian
                - 'offset': [B, 2, H_bev, W_bev]
                - 'size': [B, 2, H_bev, W_bev]
                - 'delta_z': [B, 1, H_bev, W_bev]
                - 'pos_mask': [B, H_bev, W_bev] bool
                - 'depth_gt': [B, H_img, W_img] long
                - 'depth_mask': [B, H_img, W_img] float

        Returns:
            total_loss: Scalar tensor.
            loss_dict: Dict of individual loss values (for logging).
        """
        # ── 1. Heatmap loss ─────────────────────────────────
        loss_hm = self.heatmap_loss_fn(
            predictions["heatmap"],
            targets["heatmap"],
        )

        # ── 2. Offset loss (positive locations only) ────────
        pos_mask = targets["pos_mask"]
        loss_offset = self.offset_loss_fn(
            predictions["offset"],
            targets["offset"],
            pos_mask,
        )

        # ── 3. Size loss (positive locations only) ──────────
        loss_size = self.size_loss_fn(
            predictions["size"],
            targets["size"],
            pos_mask,
        )

        # ── 4. Height (Δz) loss (positive locations only) ───
        loss_height = self.height_loss_fn(
            predictions["delta_z"],
            targets["delta_z"],
            pos_mask,
        )

        # ── 5. Depth supervision loss (SDS-VT) ──────────────
        loss_depth = self.depth_loss_fn(
            predictions["depth_prob"],
            targets["depth_gt"],
            targets["depth_mask"],
        )

        # ── 6. Weighted total ───────────────────────────────
        total = (
            self.loss_weights["hm"] * loss_hm
            + self.loss_weights["offset"] * loss_offset
            + self.loss_weights["size"] * loss_size
            + self.loss_weights["height"] * loss_height
            + self.loss_weights["depth"] * loss_depth
        )

        loss_dict = {
            "loss_total": total.item() if isinstance(total, torch.Tensor) else total,
            "loss_hm": loss_hm.item(),
            "loss_offset": loss_offset.item(),
            "loss_size": loss_size.item(),
            "loss_height": loss_height.item(),
            "loss_depth": loss_depth.item(),
        }

        return total, loss_dict


def total_loss(
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    loss_weights: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Standalone function matching the paper's loss computation pseudocode.

    Args:
        pred: Prediction dict from SDFANet.forward().
        target: Target dict from dataset.
        loss_weights: Optional loss component weights.

    Returns:
        (total_loss, loss_dict)
    """
    criterion = SDFANetLoss(loss_weights=loss_weights)
    return criterion(pred, target)
