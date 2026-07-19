"""
Geo-Head Output Decoding.

Converts Geo-Head raw outputs (heatmap, offset, size, delta_z) into
structured detections with:
    - World coordinates (meters)
    - Bounding box dimensions (meters)
    - Elevation offset Δz (meters)
    - Disease class and confidence score
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


def extract_peaks_from_heatmap(
    heatmap: torch.Tensor,
    score_threshold: float = 0.3,
    max_detections: int = 100,
    min_distance: int = 3,
) -> List[Tuple[int, int, int, float]]:
    """
    Extract local maxima (peaks) from the heatmap using 3×3 max pooling.

    Performs NMS in the heatmap space by comparing each pixel to its
    8-connected neighbors.

    Args:
        heatmap: [num_classes, H, W] sigmoid-activated heatmap.
        score_threshold: Minimum score to consider a detection.
        max_detections: Maximum number of detections to return.
        min_distance: Minimum pixel distance between peaks (3x3 NMS by default).

    Returns:
        peaks: List of (cls_id, y, x, score) tuples.
    """
    num_cls, H, W = heatmap.shape
    device = heatmap.device

    # 3×3 max pooling for local maxima detection
    pad = nn_max_pool2d(heatmap.unsqueeze(0), 3, 1, 1)
    # pad shape: (1, C, H, W)

    # A pixel is a peak if it equals the max in its 3x3 neighborhood
    keep = (heatmap == pad.squeeze(0))  # [C, H, W]

    all_peaks = []

    for cls_id in range(num_cls):
        cls_heatmap = heatmap[cls_id]
        cls_keep = keep[cls_id]
        valid = cls_keep & (cls_heatmap > score_threshold)

        ys, xs = torch.nonzero(valid, as_tuple=True)
        scores = cls_heatmap[ys, xs]

        for y, x, s in zip(ys, xs, scores):
            all_peaks.append((cls_id, int(y), int(x), float(s)))

    # Sort by score and limit
    all_peaks.sort(key=lambda p: p[3], reverse=True)
    all_peaks = all_peaks[:max_detections]

    return all_peaks


def nn_max_pool2d(x: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    """Lightweight max pool fallback (avoids import issues)."""
    import torch.nn.functional as F
    return F.max_pool2d(x, kernel_size, stride=stride, padding=padding)


def decode_predictions(
    outputs: Dict[str, torch.Tensor],
    bev_params: Dict,
    score_threshold: float = 0.3,
    max_detections: int = 100,
    class_names: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Decode Geo-Head raw outputs into structured detections.

    Args:
        outputs: Geo-Head output dict:
            - 'heatmap': [1, num_classes, H, W]
            - 'offset': [1, 2, H, W]
            - 'size': [1, 2, H, W]
            - 'delta_z': [1, 1, H, W]
        bev_params: BEV grid parameters:
            - 'x_min', 'y_min', 'x_res', 'y_res', 'bev_w', 'bev_h'
        score_threshold: Minimum heatmap score.
        max_detections: Max number of detections to return.
        class_names: Optional list of class name strings.

    Returns:
        detections: List of dicts, each with:
            'cx', 'cy': world coordinates (meters)
            'width', 'length': box dimensions (meters)
            'delta_z': elevation offset (meters)
            'class_id': integer class index
            'class_name': string class name
            'score': detection confidence
            'condition': string condition label
    """
    if class_names is None:
        class_names = ["good", "sunken", "raised", "broken", "missing"]

    heatmap = outputs["heatmap"][0]     # [C, H, W]
    offset = outputs["offset"][0]        # [2, H, W]
    size = outputs["size"][0]            # [2, H, W]
    delta_z = outputs["delta_z"][0]      # [1, H, W]

    x_min = bev_params["x_min"]
    y_min = bev_params["y_min"]
    x_res = bev_params["x_res"]
    y_res = bev_params["y_res"]

    # Extract peaks
    peaks = extract_peaks_from_heatmap(
        heatmap,
        score_threshold=score_threshold,
        max_detections=max_detections,
    )

    detections = []

    for cls_id, py, px, score in peaks:
        # Sub-pixel offset
        dx = offset[0, py, px].item()
        dy = offset[1, py, px].item()

        # Box size (in meters)
        w = size[0, py, px].item()
        l = size[1, py, px].item()

        # Elevation offset
        dz = delta_z[0, py, px].item()

        # BEV → World
        cx = (px + dx) * x_res + x_min
        cy = (py + dy) * y_res + y_min

        # Classify condition
        from .ransac_plane import classify_condition
        condition = classify_condition(dz)

        detections.append({
            "cx": cx,
            "cy": cy,
            "width": w,
            "length": l,
            "delta_z": dz,
            "class_id": cls_id,
            "class_name": class_names[cls_id] if cls_id < len(class_names) else "unknown",
            "score": score,
            "condition": condition,
        })

    return detections


def decode_predictions_batch(
    outputs: Dict[str, torch.Tensor],
    bev_params: Dict,
    score_threshold: float = 0.3,
    max_detections: int = 100,
    class_names: Optional[List[str]] = None,
) -> List[List[Dict]]:
    """
    Batch version of decode_predictions.

    Args:
        outputs: Dict with batch dimension [B, ...].

    Returns:
        batch_detections: List of detection lists, one per batch element.
    """
    B = outputs["heatmap"].shape[0]
    batch_detections = []

    for b in range(B):
        single_output = {
            "heatmap": outputs["heatmap"][b:b + 1],
            "offset": outputs["offset"][b:b + 1],
            "size": outputs["size"][b:b + 1],
            "delta_z": outputs["delta_z"][b:b + 1],
        }
        dets = decode_predictions(
            single_output,
            bev_params,
            score_threshold=score_threshold,
            max_detections=max_detections,
            class_names=class_names,
        )
        batch_detections.append(dets)

    return batch_detections
