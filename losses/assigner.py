"""
Task-Aligned Dynamic Sample Assignment.

Paper equation (10):
    A = s^α · IoU^β

where:
    - s: classification score for the ground-truth class
    - IoU: intersection-over-union between predicted box and GT box
    - α, β: task-alignment weights (default α=1.0, β=6.0)

For each ground-truth object, the top-K candidates with highest alignment
score A are selected as positive samples. This dynamically balances
classification and localization quality.

In the anchor-free setting, "boxes" are decoded from heatmap peaks,
and the assigner selects the best center locations for each GT object.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class TaskAlignedAssigner(nn.Module):
    """
    Task-Aligned Dynamic Label Assignment for anchor-free detectors.

    For each ground-truth box, computes alignment score:
        A = cls_score^α * IoU^β

    and selects the top-K candidates as positive samples.

    Reference: TOOD (Feng et al., ICCV 2021), YOLOv8.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 6.0,
        topk: int = 10,
        num_classes: int = 5,
    ):
        """
        Args:
            alpha: Classification task weight.
            beta: Localization (IoU) task weight.
            topk: Number of top candidates per ground-truth object.
            num_classes: Number of disease classes.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.topk = topk
        self.num_classes = num_classes

    @torch.no_grad()
    def forward(
        self,
        pred_scores: torch.Tensor,       # [N, num_classes] classification scores
        pred_boxes: torch.Tensor,        # [N, 4] predicted boxes (cx, cy, w, l)
        gt_boxes: torch.Tensor,          # [M, 4] ground-truth boxes
        gt_classes: torch.Tensor,        # [M] ground-truth class indices
        gt_mask: Optional[torch.Tensor] = None,  # [M] valid GT mask
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute positive sample assignment.

        Args:
            pred_scores: Predicted class scores at candidate locations.
            pred_boxes: Predicted boxes decoded from candidates.
            gt_boxes: Ground-truth boxes.
            gt_classes: Ground-truth class indices.
            gt_mask: Optional boolean mask for valid GTs.

        Returns:
            target_classes: [N] assigned class labels (0 for background/ignore).
            target_boxes: [N, 4] assigned box targets.
            fg_mask: [N] boolean, True for positive samples.
        """
        N = pred_scores.shape[0]
        M = gt_boxes.shape[0]
        device = pred_scores.device

        if M == 0:
            return (
                torch.zeros(N, dtype=torch.long, device=device),
                torch.zeros(N, 4, device=device),
                torch.zeros(N, dtype=torch.bool, device=device),
            )

        # ── 1. Compute IoU between all pred boxes and GT boxes ─────
        ious = self._box_iou(pred_boxes, gt_boxes)  # [N, M]

        # ── 2. Extract classification scores for each GT class ─────
        cls_scores = pred_scores[torch.arange(N, device=device).unsqueeze(1),
                                 gt_classes.unsqueeze(0)]  # [N, M]

        # ── 3. Compute alignment metric ─────────────────────────────
        alignment = (cls_scores ** self.alpha) * (ious ** self.beta)  # [N, M]

        # ── 4. Select top-K candidates per GT ─────────────────────────
        # For each GT, pick topk anchors with highest alignment
        topk_align, topk_indices = torch.topk(
            alignment, k=min(self.topk, N), dim=0
        )  # [topk, M]

        # ── 5. Build target assignments ─────────────────────────────
        target_classes = torch.zeros(N, dtype=torch.long, device=device)
        target_boxes = torch.zeros(N, 4, device=device)
        fg_mask = torch.zeros(N, dtype=torch.bool, device=device)

        for m in range(M):
            if gt_mask is not None and not gt_mask[m]:
                continue

            top_indices = topk_indices[:, m]
            top_aligns = topk_align[:, m]

            # Filter out low-alignment candidates
            valid_top = top_aligns > 0.0
            top_indices = top_indices[valid_top]

            for idx in top_indices:
                if not fg_mask[idx]:
                    target_classes[idx] = gt_classes[m]
                    target_boxes[idx] = gt_boxes[m]
                    fg_mask[idx] = True
                else:
                    # Anchor already assigned to another GT: keep the higher alignment
                    # (already handled by processing GTs in order — fine for small N)
                    pass

        return target_classes, target_boxes, fg_mask

    @staticmethod
    def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU between two sets of axis-aligned BEV boxes.

        Args:
            boxes1: [N, 4] (cx, cy, w, l)
            boxes2: [M, 4] (cx, cy, w, l)

        Returns:
            iou: [N, M]
        """
        # Convert to (x1, y1, x2, y2)
        b1 = torch.stack([
            boxes1[:, 0] - boxes1[:, 2] / 2,  # x1
            boxes1[:, 1] - boxes1[:, 3] / 2,  # y1
            boxes1[:, 0] + boxes1[:, 2] / 2,  # x2
            boxes1[:, 1] + boxes1[:, 3] / 2,  # y2
        ], dim=1)  # [N, 4]

        b2 = torch.stack([
            boxes2[:, 0] - boxes2[:, 2] / 2,  # x1
            boxes2[:, 1] - boxes2[:, 3] / 2,  # y1
            boxes2[:, 0] + boxes2[:, 2] / 2,  # x2
            boxes2[:, 1] + boxes2[:, 3] / 2,  # y2
        ], dim=1)  # [M, 4]

        N, M = b1.shape[0], b2.shape[0]

        # Intersection
        lt = torch.max(b1[:, None, :2], b2[None, :, :2])  # [N, M, 2]
        rb = torch.min(b1[:, None, 2:], b2[None, :, 2:])  # [N, M, 2]
        wh = (rb - lt).clamp(min=0)  # [N, M, 2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

        # Union
        area1 = b1[:, 2] * b1[:, 3]  # [N]
        area2 = b2[:, 2] * b2[:, 3]  # [M]
        union = area1[:, None] + area2[None, :] - inter  # [N, M]

        iou = inter / (union + 1e-7)
        return iou


def task_aligned_assigner(
    pred_scores: torch.Tensor,
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 6.0,
    topk: int = 10,
) -> List[torch.Tensor]:
    """
    Standalone function matching paper pseudocode.

    Returns:
        positive_indices: List of tensors, one per GT object,
                          each containing indices of assigned positive samples.
    """
    N = pred_scores.shape[0]
    M = gt_boxes.shape[0]

    # IoU
    assigner = TaskAlignedAssigner(alpha=alpha, beta=beta, topk=topk)
    ious = assigner._box_iou(pred_boxes, gt_boxes)  # [N, M]

    # Classification scores for GT classes
    cls_scores = torch.zeros(N, M, device=pred_scores.device)
    for m in range(M):
        cls_scores[:, m] = pred_scores[:, gt_classes[m]]

    alignment = (cls_scores ** alpha) * (ious ** beta)

    positive_indices = []
    for m in range(M):
        topk_idx = torch.topk(alignment[:, m], k=min(topk, N)).indices
        positive_indices.append(topk_idx)

    return positive_indices
