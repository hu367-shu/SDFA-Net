"""
Detection Loss Functions for SDFA-Net Geo-Head.

Losses:
    - Heatmap loss: Modified focal loss for anchor-free center detection
    - Offset loss: Smooth L1 on center sub-pixel offsets
    - Size loss: Smooth L1 on bounding box width/length
    - Height loss: Smooth L1 on elevation offset Δz

All regression losses are only computed at positive-pixel locations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Modified focal loss for heatmap supervision (CenterNet-style).

    L_hm = -1/N * Σ [
        (1 - Y_pred)^α * log(Y_pred)          if Y_gt == 1
        (1 - Y_gt)^β * (Y_pred)^α * log(1-Y_pred)   if Y_gt < 1
    ]

    where α=2, β=4 by default (from CornerNet/CenterNet).
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        pred: torch.Tensor,   # [B, C, H, W] predicted heatmap (sigmoid-activated)
        target: torch.Tensor,  # [B, C, H, W] ground-truth Gaussian heatmap
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted heatmap, values in [0, 1].
            target: GT Gaussian heatmap, values in [0, 1].

        Returns:
            loss: Scalar heatmap focal loss.
        """
        # Positive locations (GT > 0)
        pos_mask = target.eq(1).float()  # hard positive
        neg_mask = target.lt(1).float()

        # Positive loss: -(1 - pred)^α * log(pred)
        pos_loss = -torch.log(pred + 1e-6) * ((1 - pred) ** self.alpha)
        pos_loss = pos_loss * pos_mask

        # Negative loss: -(1 - target)^β * pred^α * log(1 - pred)
        neg_loss = -torch.log(1.0 - pred + 1e-6) * (pred ** self.alpha)
        neg_loss = neg_loss * ((1.0 - target) ** self.beta)
        neg_loss = neg_loss * neg_mask

        # Total
        num_pos = max(1, pos_mask.sum().item())
        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

        return loss


class SmoothL1LossMasked(nn.Module):
    """
    Smooth L1 loss computed only at positive (masked) locations.
    Used for offset, size, and height regression.
    """

    def __init__(self, beta: float = 1.0):
        """
        Args:
            beta: Smooth L1 transition threshold (default 1.0).
        """
        super().__init__()
        self.beta = beta
        self.loss_fn = nn.SmoothL1Loss(beta=beta, reduction="none")

    def forward(
        self,
        pred: torch.Tensor,     # [B, C, H, W]
        target: torch.Tensor,   # [B, C, H, W]
        mask: torch.Tensor,     # [B, H, W] bool, positive sample locations
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted values.
            target: Ground-truth values.
            mask: Boolean mask, True at positive pixel locations.

        Returns:
            loss: Scalar (mean over positive locations).
        """
        B, C, H, W = pred.shape

        # Expand mask to channel dimension
        mask_exp = mask.unsqueeze(1).expand_as(pred)  # [B, C, H, W]

        loss = self.loss_fn(pred, target)  # [B, C, H, W]
        loss = loss * mask_exp.float()

        num_pos = mask_exp.sum() + 1e-6
        return loss.sum() / num_pos


# ── Convenience functions ──────────────────────────────────────────

def heatmap_loss(
    pred_heatmap: torch.Tensor,
    gt_heatmap: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """Compute focal loss for heatmap."""
    criterion = FocalLoss(alpha=alpha, beta=beta)
    return criterion(pred_heatmap, gt_heatmap)


def offset_loss(
    pred_offset: torch.Tensor,
    gt_offset: torch.Tensor,
    pos_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute Smooth L1 loss for center offset regression."""
    criterion = SmoothL1LossMasked(beta=1.0 / 9.0)  # standard beta for offset
    return criterion(pred_offset, gt_offset, pos_mask)


def size_loss(
    pred_size: torch.Tensor,
    gt_size: torch.Tensor,
    pos_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute Smooth L1 loss for box size regression."""
    criterion = SmoothL1LossMasked(beta=0.1)
    return criterion(pred_size, gt_size, pos_mask)


def height_loss(
    pred_delta_z: torch.Tensor,
    gt_delta_z: torch.Tensor,
    pos_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute Smooth L1 loss for elevation offset regression."""
    criterion = SmoothL1LossMasked(beta=0.1)
    return criterion(pred_delta_z, gt_delta_z, pos_mask)
