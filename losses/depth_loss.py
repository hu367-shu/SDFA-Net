"""
Semi-Dense Depth Supervision Loss.

Paper formula (2):
    L_depth = -(1 / ΣM) * Σ_{u,v} M(u,v) * Σ_d Y_d(u,v) * log(P_d(u,v))

This is masked cross-entropy loss: only pixels with a valid LiDAR point
projection (M=1) contribute to the loss. Y_d is the one-hot depth GT.

The loss is "semi-dense" because typically only ~5-15% of image pixels
receive LiDAR depth supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SemiDenseDepthLoss(nn.Module):
    """
    Masked cross-entropy loss for semi-dense depth supervision.

    Loss is computed only at pixels where depth_mask == 1 (valid LiDAR projection).
    """

    def __init__(self, reduction: str = "mean"):
        """
        Args:
            reduction: 'mean' (default) or 'sum'.
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        depth_prob: torch.Tensor,   # [B, D, H, W]
        depth_gt: torch.Tensor,     # [B, H, W] long, bin indices
        depth_mask: torch.Tensor,   # [B, H, W] float, 1 where valid
    ) -> torch.Tensor:
        """
        Compute the semi-dense depth loss.

        Args:
            depth_prob: Predicted depth probability distribution (softmax-normalized).
            depth_gt: Ground-truth depth bin indices.
            depth_mask: Binary mask (1 = valid LiDAR projection).

        Returns:
            loss: Scalar loss value.
        """
        B = depth_prob.shape[0]

        # Cross-entropy per pixel
        log_prob = torch.log(depth_prob + 1e-6)  # [B, D, H, W]

        # NLL loss (no reduction)
        nll = F.nll_loss(
            log_prob,
            depth_gt,
            reduction="none",
        )  # [B, H, W]

        # Mask out pixels without LiDAR depth
        nll = nll * depth_mask

        # Normalize by total number of valid pixels
        total_valid = depth_mask.sum() + 1e-6

        if self.reduction == "mean":
            loss = nll.sum() / total_valid
        else:
            loss = nll.sum()

        return loss


def semi_dense_depth_loss(
    depth_prob: torch.Tensor,
    depth_gt: torch.Tensor,
    depth_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Standalone function for semi-dense depth supervision loss.

    Paper formula (2):
        L_depth = -(1/ΣM) Σ_{u,v} M(u,v) Σ_d Y_d(u,v) log(P_d(u,v))

    Args:
        depth_prob: [B, D, H, W] softmax-normalized depth probabilities.
        depth_gt: [B, H, W] discrete depth bin indices (long).
        depth_mask: [B, H, W] binary mask (1 where LiDAR projects).

    Returns:
        loss: Scalar tensor.
    """
    log_prob = torch.log(depth_prob + 1e-6)

    loss = F.nll_loss(
        log_prob,
        depth_gt,
        reduction="none",
    )  # [B, H, W]

    loss = loss * depth_mask
    loss = loss.sum() / (depth_mask.sum() + 1e-6)

    return loss
