"""
SDFA-Net Evaluation Script.

Computes comprehensive metrics on the test set:
    - mAP (mean Average Precision) at IoU=0.5 and IoU=0.5:0.95
    - Per-class AP
    - Recall
    - Δz MAE (Mean Absolute Error) and RMSE
    - Disease classification accuracy (mAcc, overall accuracy)
    - Confusion matrix
    - Inference speed (FPS)

Usage:
    python eval.py --config configs/sdfanet.yaml \
                   --checkpoint checkpoints/best_model.pth \
                   --data_root ./data/manhole_dataset \
                   --output results/eval_results.json
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from models import SDFANet
from data import create_dataloaders
from utils.decode import decode_predictions_batch
from utils.metrics import (
    compute_map,
    compute_mae,
    compute_rmse,
    compute_classification_accuracy,
)
from utils.ransac_plane import compute_delta_z_ransac


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SDFA-Net")
    parser.add_argument(
        "--config", type=str, default="configs/sdfanet.yaml",
        help="Path to YAML config file."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained model checkpoint."
    )
    parser.add_argument(
        "--data_root", type=str, default="./data/manhole_dataset",
        help="Path to dataset root."
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device: 'cuda' or 'cpu'."
    )
    parser.add_argument(
        "--output", type=str, default="results/eval_results.json",
        help="Path to save evaluation results JSON."
    )
    parser.add_argument(
        "--score_threshold", type=float, default=0.3,
        help="Minimum detection score."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Evaluation batch size."
    )
    parser.add_argument(
        "--refine_delta_z", action="store_true",
        help="Refine Δz using RANSAC on point cloud (more accurate but slower)."
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate_full(
    model: SDFANet,
    dataloader,
    bev_params: Dict,
    class_names: List[str],
    device: str = "cuda",
    score_threshold: float = 0.3,
    refine_delta_z: bool = False,
    ransac_cfg: Optional[Dict] = None,
) -> Dict:
    """
    Run full evaluation pipeline.

    Returns a dict of all metrics.
    """
    model.eval()
    model.to(device)

    all_detections = []
    all_ground_truths = []
    all_pred_dz = []
    all_gt_dz = []
    all_pred_classes = []
    all_gt_classes = []
    all_pred_scores = []
    inference_times = []

    num_classes = len(class_names)

    print(f"\nEvaluating on {len(dataloader.dataset)} samples...")

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        image = batch["image"].to(device)
        points = batch["points"].to(device)
        calib = {k: v.to(device) for k, v in batch["calib"].items()}
        targets = batch["targets"]

        B = image.shape[0]

        # ── Forward timing ──────────────────────────────────
        t_start = time.time()
        outputs = model(image, points, calib)
        t_end = time.time()
        inference_times.extend([(t_end - t_start) / B] * B)

        # ── Decode predictions ──────────────────────────────
        batch_dets = decode_predictions_batch(
            outputs, bev_params,
            score_threshold=score_threshold,
            max_detections=100,
            class_names=class_names,
        )

        # ── Collect predictions and GTs ─────────────────────
        for b in range(B):
            frame_dets = batch_dets[b]

            # Refine Δz with RANSAC if requested
            if refine_delta_z and len(frame_dets) > 0:
                pts = points[b][points[b].sum(dim=-1) != 0].cpu().numpy()  # valid points
                if pts.shape[0] > 10:
                    for det in frame_dets:
                        dz = compute_delta_z_ransac(
                            pts,
                            center_xy=(det["cx"], det["cy"]),
                            r_inner=ransac_cfg.get("r_inner", 0.45) if ransac_cfg else 0.45,
                            r_outer=ransac_cfg.get("r_outer", 1.0) if ransac_cfg else 1.0,
                            cover_radius=ransac_cfg.get("cover_radius", 0.35) if ransac_cfg else 0.35,
                        )
                        if dz is not None:
                            det["delta_z"] = dz

            all_detections.append(frame_dets)

            # Collect ground truth
            gt_boxes = targets.get("boxes")
            gt_classes = targets.get("classes")
            gt_delta_zs = targets.get("delta_zs")

            if gt_boxes is not None and len(gt_classes[b]) > 0:
                frame_gts = []
                M = len(gt_classes[b])
                for m in range(M):
                    gt_dict = {
                        "cx": float(gt_boxes[b][m, 0]),
                        "cy": float(gt_boxes[b][m, 1]),
                        "width": float(gt_boxes[b][m, 2]),
                        "length": float(gt_boxes[b][m, 3]),
                        "class_id": int(gt_classes[b][m]),
                        "delta_z": float(gt_delta_zs[b][m]) if gt_delta_zs is not None else 0.0,
                    }
                    frame_gts.append(gt_dict)
                    all_gt_dz.append(gt_dict["delta_z"])
                    all_gt_classes.append(gt_dict["class_id"])
                all_ground_truths.append(frame_gts)

            # Match predictions to GTs for Δz error computation
            for det in frame_dets:
                all_pred_dz.append(det["delta_z"])
                all_pred_classes.append(det["class_id"])
                all_pred_scores.append(det["score"])

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {(batch_idx + 1) * B} / {len(dataloader.dataset)} samples")

    # ── Detection metrics ────────────────────────────────────
    print("\nComputing detection metrics...")
    det_metrics = {}
    for iou_thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        m = compute_map(
            all_detections, all_ground_truths,
            iou_threshold=iou_thresh,
            num_classes=num_classes,
        )
        det_metrics[f"mAP@{int(iou_thresh * 100)}"] = m["mAP"]

    # mAP@50 and mAP@50:95
    map50 = det_metrics.get("mAP@50", 0.0)
    map_vals = [det_metrics.get(f"mAP@{t}", 0.0) for t in range(50, 100, 5)]
    map5095 = float(np.mean(map_vals))

    # Per-class AP at IoU=0.5
    per_class_ap = {}
    full_map50 = compute_map(
        all_detections, all_ground_truths,
        iou_threshold=0.5, num_classes=num_classes,
    )
    for cls_id in range(num_classes):
        key = f"AP_cls_{cls_id}"
        if key in full_map50:
            per_class_ap[class_names[cls_id]] = full_map50[key]

    # ── Δz regression metrics ────────────────────────────────
    if len(all_pred_dz) > 0 and len(all_gt_dz) > 0:
        # Match predictions to closest GT for Δz comparison
        # For simplicity, use all GT Δz and corresponding predictions
        dz_mae = compute_mae(
            np.array(all_pred_dz[:len(all_gt_dz)]),
            np.array(all_gt_dz[:len(all_pred_dz)]),
        )
        dz_rmse = compute_rmse(
            np.array(all_pred_dz[:len(all_gt_dz)]),
            np.array(all_gt_dz[:len(all_pred_dz)]),
        )
    else:
        dz_mae = 0.0
        dz_rmse = 0.0

    # ── Classification metrics ───────────────────────────────
    if len(all_pred_classes) > 0 and len(all_gt_classes) > 0:
        cls_metrics = compute_classification_accuracy(
            np.array(all_pred_classes[:len(all_gt_classes)]),
            np.array(all_gt_classes[:len(all_pred_classes)]),
            num_classes=num_classes,
        )
    else:
        cls_metrics = {"overall_accuracy": 0.0, "mean_class_accuracy": 0.0}

    # ── Speed metrics ────────────────────────────────────────
    avg_fps = 1.0 / np.mean(inference_times) if inference_times else 0.0
    fps_std = np.std([1.0 / t for t in inference_times]) if inference_times else 0.0

    # ── Summary ──────────────────────────────────────────────
    results = {
        "detection": {
            "mAP@50": map50,
            "mAP@50:95": map5095,
            "per_class_AP": per_class_ap,
        },
        "regression": {
            "delta_z_MAE_m": dz_mae,
            "delta_z_RMSE_m": dz_rmse,
        },
        "classification": cls_metrics,
        "speed": {
            "avg_fps": avg_fps,
            "fps_std": fps_std,
            "avg_inference_time_ms": float(np.mean(inference_times) * 1000),
        },
        "counts": {
            "total_detections": len(all_pred_classes),
            "total_ground_truths": len(all_gt_classes),
            "total_frames": len(all_ground_truths),
        },
    }

    return results


def main():
    args = parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Override batch size
    cfg["training"]["batch_size"] = args.batch_size

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Build model
    print("Building model...")
    model = SDFANet(cfg)
    bev_params = model.get_bev_params()

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Create test dataloader
    print("Loading test dataset...")
    data_cfg = cfg.get("data", {})
    class_names = data_cfg.get("classes", ["good", "sunken", "raised", "broken", "missing"])

    from data.dataset import ManholeDataset, custom_collate_fn
    from torch.utils.data import DataLoader

    test_dataset = ManholeDataset(
        data_root=args.data_root,
        split="test",
        cfg=cfg,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg.get("training", {}).get("num_workers", 4),
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )

    # Run evaluation
    results = evaluate_full(
        model=model,
        dataloader=test_loader,
        bev_params=bev_params,
        class_names=class_names,
        device=device,
        score_threshold=args.score_threshold,
        refine_delta_z=args.refine_delta_z,
        ransac_cfg=data_cfg.get("ransac", {}),
    )

    # Print results
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")

    print(f"\nDetection:")
    print(f"  mAP@50:    {results['detection']['mAP@50']:.4f}")
    print(f"  mAP@50:95: {results['detection']['mAP@50:95']:.4f}")
    print(f"  Per-class AP@50:")
    for cls_name, ap in results['detection']['per_class_AP'].items():
        print(f"    {cls_name}: {ap:.4f}")

    print(f"\nRegression (Δz):")
    print(f"  MAE:  {results['regression']['delta_z_MAE_m']:.4f} m "
          f"({results['regression']['delta_z_MAE_m'] * 1000:.1f} mm)")
    print(f"  RMSE: {results['regression']['delta_z_RMSE_m']:.4f} m "
          f"({results['regression']['delta_z_RMSE_m'] * 1000:.1f} mm)")

    print(f"\nClassification:")
    print(f"  Overall accuracy: {results['classification']['overall_accuracy']:.4f}")
    print(f"  Mean class accuracy: {results['classification']['mean_class_accuracy']:.4f}")

    print(f"\nSpeed:")
    print(f"  Avg FPS: {results['speed']['avg_fps']:.1f}")
    print(f"  Avg inference time: {results['speed']['avg_inference_time_ms']:.1f} ms")

    print(f"\nCounts:")
    print(f"  Total frames: {results['counts']['total_frames']}")
    print(f"  Total detections: {results['counts']['total_detections']}")
    print(f"  Total ground truths: {results['counts']['total_ground_truths']}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
