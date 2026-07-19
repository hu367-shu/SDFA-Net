"""
Evaluation Metrics for SDFA-Net.

Metrics:
    - mAP (mean Average Precision) at IoU thresholds (0.5, 0.5:0.95)
    - Recall
    - MAE / RMSE for Δz elevation prediction
    - Classification accuracy (mAcc) for disease status
    - Per-class breakdown
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# ── Average Precision ────────────────────────────────────────────

def compute_ap(
    recalls: np.ndarray,
    precisions: np.ndarray,
) -> float:
    """
    Compute Average Precision using 11-point interpolation (VOC2007)
    or all-point interpolation (VOC2010).

    Args:
        recalls: Sorted recall values.
        precisions: Corresponding precision values.

    Returns:
        AP: Average precision score.
    """
    # All-point interpolation
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Compute AP as area under PR curve
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum(
        (recalls[indices + 1] - recalls[indices]) * precisions[indices + 1]
    )
    return float(ap)


def compute_map(
    detections: List[List[Dict]],
    ground_truths: List[List[Dict]],
    iou_threshold: float = 0.5,
    num_classes: int = 5,
) -> Dict[str, float]:
    """
    Compute mAP (mean Average Precision) for BEV detection.

    Args:
        detections: Per-frame list of detection dicts, each with:
            'cx', 'cy', 'width', 'length', 'class_id', 'score'.
        ground_truths: Per-frame list of GT dicts, each with:
            'cx', 'cy', 'width', 'length', 'class_id'.
        iou_threshold: IoU threshold for positive match.
        num_classes: Number of disease classes.

    Returns:
        metrics: Dict with 'mAP', 'mAP@50', and per-class APs.
    """
    aps = []

    for cls_id in range(num_classes):
        # Collect all detections and GTs for this class
        all_dets = []
        all_gts_per_frame = []

        for frame_dets, frame_gts in zip(detections, ground_truths):
            cls_dets = [d for d in frame_dets if d["class_id"] == cls_id]
            cls_gts = [g for g in frame_gts if g["class_id"] == cls_id]
            all_dets.append(cls_dets)
            all_gts_per_frame.append(cls_gts)

        ap = _compute_ap_per_class(all_dets, all_gts_per_frame, iou_threshold)
        aps.append(ap)

    mAP = float(np.mean([ap for ap in aps if ap is not None]))

    metrics = {"mAP": mAP, f"mAP@{int(iou_threshold * 100)}": mAP}

    for cls_id, ap in enumerate(aps):
        if ap is not None:
            metrics[f"AP_cls_{cls_id}"] = ap

    return metrics


def _compute_ap_per_class(
    frame_dets: List[List[Dict]],
    frame_gts: List[List[Dict]],
    iou_threshold: float,
) -> Optional[float]:
    """Compute AP for a single class."""
    # Flatten all detections across frames with frame index
    all_scores = []
    all_matches = []
    total_gt = 0

    for frame_idx in range(len(frame_dets)):
        dets = frame_dets[frame_idx]
        gts = frame_gts[frame_idx]
        total_gt += len(gts)

        if len(dets) == 0:
            continue

        # Sort detections by score descending
        dets_sorted = sorted(dets, key=lambda d: d["score"], reverse=True)

        gt_matched = [False] * len(gts)

        for det in dets_sorted:
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gts):
                if gt_matched[gt_idx]:
                    continue

                iou = _box_iou_single(det, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                gt_matched[best_gt_idx] = True
                all_matches.append(True)
            else:
                all_matches.append(False)

            all_scores.append(det["score"])

    if total_gt == 0:
        return None

    if len(all_scores) == 0:
        return 0.0

    # Sort by score
    indices = np.argsort(all_scores)[::-1]
    all_matches = np.array(all_matches)[indices]

    # Cumulative TP, FP
    tp = np.cumsum(all_matches)
    fp = np.cumsum(~all_matches)

    recalls = tp / total_gt
    precisions = tp / (tp + fp + 1e-7)

    return compute_ap(recalls, precisions)


def _box_iou_single(box1: Dict, box2: Dict) -> float:
    """IoU between two BEV boxes."""
    x1_min = box1["cx"] - box1["width"] / 2
    x1_max = box1["cx"] + box1["width"] / 2
    y1_min = box1["cy"] - box1["length"] / 2
    y1_max = box1["cy"] + box1["length"] / 2

    x2_min = box2["cx"] - box2["width"] / 2
    x2_max = box2["cx"] + box2["width"] / 2
    y2_min = box2["cy"] - box2["length"] / 2
    y2_max = box2["cy"] + box2["length"] / 2

    inter_w = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_h = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter = inter_w * inter_h

    area1 = box1["width"] * box1["length"]
    area2 = box2["width"] * box2["length"]
    union = area1 + area2 - inter

    return inter / (union + 1e-7)


# ── Regression Metrics ────────────────────────────────────────────

def compute_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(pred - target)))


def compute_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root Mean Square Error."""
    return float(np.sqrt(np.mean((pred - target) ** 2)))


# ── Classification Metrics ────────────────────────────────────────

def compute_classification_accuracy(
    pred_classes: np.ndarray,
    target_classes: np.ndarray,
    num_classes: int = 5,
) -> Dict[str, float]:
    """
    Compute classification accuracy metrics.

    Args:
        pred_classes: [N] predicted class indices.
        target_classes: [N] ground-truth class indices.
        num_classes: Number of classes.

    Returns:
        Dict with 'mAcc' (mean per-class accuracy) and 'overall_acc'.
    """
    # Overall accuracy
    overall_acc = float(np.mean(pred_classes == target_classes))

    # Per-class accuracy
    class_accs = []
    for c in range(num_classes):
        mask = target_classes == c
        if mask.sum() > 0:
            acc = (pred_classes[mask] == c).mean()
            class_accs.append(float(acc))
        else:
            class_accs.append(0.0)

    mAcc = float(np.mean(class_accs)) if class_accs else 0.0

    return {
        "overall_accuracy": overall_acc,
        "mean_class_accuracy": mAcc,
        "per_class_accuracy": class_accs,
    }


# ── Full Evaluation ───────────────────────────────────────────────

def evaluate_sdfanet(
    model,
    dataloader,
    bev_params: Dict,
    class_names: Optional[List[str]] = None,
    score_threshold: float = 0.3,
    iou_threshold: float = 0.5,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Run full evaluation of SDFA-Net on a dataset.

    Computes mAP, recall, Δz MAE/RMSE, and classification accuracy.

    Args:
        model: Trained SDFANet model.
        dataloader: Validation or test dataloader.
        bev_params: BEV grid parameters for decoding.
        class_names: List of class name strings.
        score_threshold: Minimum detection score.
        iou_threshold: IoU threshold for mAP.
        device: 'cuda' or 'cpu'.

    Returns:
        metrics: Dict of evaluation metrics.
    """
    from ..utils.decode import decode_predictions_batch

    model.eval()
    model.to(device)

    all_detections = []
    all_ground_truths = []
    all_pred_dz = []
    all_gt_dz = []
    all_pred_classes = []
    all_gt_classes = []

    with torch.no_grad():
        for batch in dataloader:
            image = batch["image"].to(device)
            points = batch["points"].to(device)
            calib = {k: v.to(device) for k, v in batch["calib"].items()}
            targets = batch["targets"]

            # Forward
            outputs = model(image, points, calib)

            # Decode detections
            batch_dets = decode_predictions_batch(
                outputs,
                bev_params,
                score_threshold=score_threshold,
                class_names=class_names,
            )
            all_detections.extend(batch_dets)

            # Collect GTs
            B = image.shape[0]
            for b in range(B):
                gt_boxes_t = targets["boxes"][b]  # [M, 4]
                gt_classes_t = targets["classes"][b]  # [M]
                gt_dz_t = targets["delta_zs"][b]  # [M]

                frame_gts = []
                for m in range(len(gt_classes_t)):
                    frame_gts.append({
                        "cx": float(gt_boxes_t[m, 0]),
                        "cy": float(gt_boxes_t[m, 1]),
                        "width": float(gt_boxes_t[m, 2]),
                        "length": float(gt_boxes_t[m, 3]),
                        "class_id": int(gt_classes_t[m]),
                        "delta_z": float(gt_dz_t[m]),
                    })
                all_ground_truths.append(frame_gts)

    # ── Compute metrics ────────────────────────────────────────
    num_classes = len(class_names) if class_names else 5
    num_classes = max(num_classes, 5)

    detection_metrics = compute_map(
        all_detections, all_ground_truths,
        iou_threshold=iou_threshold, num_classes=num_classes,
    )

    return detection_metrics
