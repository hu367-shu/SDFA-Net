"""
SDFA-Net Inference Script.

Runs inference on new data: loads a trained checkpoint, processes input
(fisheye image + LiDAR point cloud), and outputs structured detections
with manhole cover positions, dimensions, elevation offsets, and disease
condition classifications.

Usage:
    python infer.py \
        --config configs/sdfanet.yaml \
        --checkpoint checkpoints/best_model.pth \
        --image_path data/test/image.png \
        --cloud_path data/test/cloud.npy \
        --calib_path data/test/calib.json \
        --output results.json

Or run on a directory:
    python infer.py \
        --config configs/sdfanet.yaml \
        --checkpoint checkpoints/best_model.pth \
        --input_dir data/test_frames/ \
        --output_dir results/
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
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from models import SDFANet
from data import FisheyeUndistorter, LidarAggregator
from data.generate_depth_gt import DepthGTGenerator
from data.preprocess_fisheye import OCamParams
from utils.decode import decode_predictions
from utils.ransac_plane import compute_delta_z_ransac, classify_condition


class SDFANetInference:
    """
    Inference pipeline for SDFA-Net.

    Handles full preprocessing → model forward → postprocessing → output.
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda",
    ):
        """
        Args:
            config_path: Path to sdfanet.yaml.
            checkpoint_path: Path to trained model checkpoint.
            device: 'cuda' or 'cpu'.
        """
        self.device = device if torch.cuda.is_available() else "cpu"

        # Load config
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # Build model
        self.model = SDFANet(self.cfg)
        self.model.to(self.device)
        self.model.eval()

        # Load checkpoint
        self._load_checkpoint(checkpoint_path)

        # Get BEV params
        self.bev_params = self.model.get_bev_params()

        # Extract config parameters
        model_cfg = self.cfg.get("model", {})
        sds_cfg = model_cfg.get("sds_vt", {})
        data_cfg = self.cfg.get("data", {})

        self.image_size = tuple(sds_cfg.get("image_size", [384, 640]))
        self.fov = sds_cfg.get("fov", 120.0)
        self.depth_bins = sds_cfg.get("depth_bins", 64)
        self.depth_min = sds_cfg.get("depth_min", 0.5)
        self.depth_max = sds_cfg.get("depth_max", 30.0)
        self.multi_frame_window = data_cfg.get("multi_frame_window", 10)
        self.ransac_cfg = data_cfg.get("ransac", {})

        # Initialize processors
        self.depth_gt_gen = DepthGTGenerator(
            image_size=self.image_size,
            depth_bins=self.depth_bins,
            depth_min=self.depth_min,
            depth_max=self.depth_max,
        )

        # Initialize LiDAR aggregator
        self.aggregator = LidarAggregator(
            window_size=self.multi_frame_window,
            voxel_size=data_cfg.get("agg_voxel_size", 0.05),
        )

        self.class_names = data_cfg.get("classes",
            ["good", "sunken", "raised", "broken", "missing"])

        print(f"[Inference] Model loaded on {self.device}")
        print(f"  BEV: {self.bev_params['bev_h']}x{self.bev_params['bev_w']}")
        print(f"  Image: {self.image_size}")

    def _load_checkpoint(self, path: str):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict, strict=True)
        print(f"[Inference] Loaded checkpoint from {path}")

    def predict_single(
        self,
        image: np.ndarray,
        points: np.ndarray,
        calib: dict,
        score_threshold: float = 0.3,
        refine_delta_z: bool = True,
    ) -> List[Dict]:
        """
        Run inference on a single frame.

        Args:
            image: Rectified fisheye image [H, W, 3] float32 or uint8.
            points: Aggregated LiDAR point cloud [N, 3+].
            calib: Dict with 'K_virtual' [3,3] and 'T_lidar_cam' [4,4].
            score_threshold: Minimum detection confidence.
            refine_delta_z: If True, recompute Δz via RANSAC for each detection.

        Returns:
            detections: List of detection dicts.
        """
        # ── Prepare tensors ──────────────────────────────────
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[-1] == 3:
            image_t = torch.from_numpy(image).permute(2, 0, 1).float()
        else:
            image_t = torch.from_numpy(image).float()

        points_t = torch.from_numpy(points[:, :3]).float()

        K_t = torch.from_numpy(calib["K_virtual"]).float()
        T_t = torch.from_numpy(calib["T_lidar_cam"]).float()

        # Batch dimension
        image_t = image_t.unsqueeze(0).to(self.device)
        points_t = points_t.unsqueeze(0).to(self.device)
        K_t = K_t.unsqueeze(0).to(self.device)
        T_t = T_t.unsqueeze(0).to(self.device)

        # ── Forward ──────────────────────────────────────────
        with torch.no_grad():
            outputs = self.model(
                image_t,
                points_t,
                {"K_virtual": K_t, "T_lidar_cam": T_t},
            )

        # ── Decode ───────────────────────────────────────────
        detections = decode_predictions(
            outputs,
            self.bev_params,
            score_threshold=score_threshold,
            max_detections=100,
            class_names=self.class_names,
        )

        # ── Refine Δz using RANSAC ───────────────────────────
        if refine_delta_z and len(detections) > 0:
            for det in detections:
                dz = compute_delta_z_ransac(
                    points,
                    center_xy=(det["cx"], det["cy"]),
                    r_inner=self.ransac_cfg.get("r_inner", 0.45),
                    r_outer=self.ransac_cfg.get("r_outer", 1.0),
                    cover_radius=self.ransac_cfg.get("cover_radius", 0.35),
                )
                if dz is not None:
                    det["delta_z"] = dz
                    det["condition"] = classify_condition(
                        dz,
                        threshold=self.ransac_cfg.get("z_threshold", 0.015),
                    )

        return detections

    def predict_from_files(
        self,
        image_path: str,
        cloud_path: str,
        calib_path: str,
        score_threshold: float = 0.3,
        refine_delta_z: bool = True,
    ) -> List[Dict]:
        """
        Run inference from file paths.

        Args:
            image_path: Path to rectified image file.
            cloud_path: Path to .npy LiDAR point cloud file.
            calib_path: Path to .json calibration file.
            score_threshold: Minimum detection confidence.
            refine_delta_z: Whether to refine Δz via RANSAC.

        Returns:
            detections: List of detection dicts.
        """
        # Load image
        img = np.array(Image.open(image_path).convert("RGB"))

        # Load point cloud
        cloud = np.load(cloud_path)

        # Load calibration
        with open(calib_path, "r") as f:
            calib = json.load(f)
        calib_parsed = {
            "K_virtual": np.array(calib.get("K_virtual", np.eye(3)), dtype=np.float32),
            "T_lidar_cam": np.array(calib.get("T_lidar_cam", np.eye(4)), dtype=np.float32),
        }

        return self.predict_single(
            img, cloud, calib_parsed,
            score_threshold=score_threshold,
            refine_delta_z=refine_delta_z,
        )

    def predict_batch_from_dir(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        score_threshold: float = 0.3,
    ) -> List[Dict]:
        """
        Run inference on all frames in a directory.

        Expected directory structure:
            input_dir/
                images/    (rectified .png/.jpg)
                clouds/    (.npy point clouds)
                calib/     (.json calibration files)

        Args:
            input_dir: Path to input directory.
            output_dir: If provided, save JSON results per frame.
            score_threshold: Minimum detection confidence.

        Returns:
            all_results: Dict mapping frame_id → detections list.
        """
        input_dir = Path(input_dir)
        img_dir = input_dir / "images"
        cloud_dir = input_dir / "clouds"
        calib_dir = input_dir / "calib"

        image_files = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
        all_results = {}

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        for img_path in image_files:
            frame_id = img_path.stem

            cloud_path = cloud_dir / f"{frame_id}.npy"
            calib_path = calib_dir / f"{frame_id}.json"

            if not cloud_path.exists():
                print(f"  Skipping {frame_id}: no point cloud file")
                continue
            if not calib_path.exists():
                print(f"  Skipping {frame_id}: no calibration file")
                continue

            t_start = time.time()

            detections = self.predict_from_files(
                str(img_path),
                str(cloud_path),
                str(calib_path),
                score_threshold=score_threshold,
            )

            elapsed = time.time() - t_start
            all_results[frame_id] = detections

            print(f"  [{frame_id}] {len(detections)} detections in {elapsed:.2f}s")
            for det in detections:
                print(f"    {det['class_name']} ({det['condition']}) "
                      f"score={det['score']:.3f} Δz={det['delta_z']:.3f}m "
                      f"@ ({det['cx']:.2f}, {det['cy']:.2f})")

            # Save individual result
            if output_dir:
                result_path = output_dir / f"{frame_id}.json"
                with open(result_path, "w") as f:
                    json.dump(detections, f, indent=2, default=float)

        # Save consolidated results
        if output_dir:
            consolidated_path = output_dir / "all_results.json"
            with open(consolidated_path, "w") as f:
                json.dump(all_results, f, indent=2, default=float)

        return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="SDFA-Net Inference")
    parser.add_argument(
        "--config", type=str, default="configs/sdfanet.yaml",
        help="Path to YAML config file."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained model checkpoint."
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device: 'cuda' or 'cpu'."
    )

    # Single-file mode
    parser.add_argument("--image_path", type=str, help="Path to rectified image.")
    parser.add_argument("--cloud_path", type=str, help="Path to .npy point cloud.")
    parser.add_argument("--calib_path", type=str, help="Path to .json calibration.")
    parser.add_argument("--output", type=str, help="Path to save JSON result.")

    # Directory mode
    parser.add_argument("--input_dir", type=str, help="Input directory with images/, clouds/, calib/.")
    parser.add_argument("--output_dir", type=str, help="Output directory for results.")

    # Parameters
    parser.add_argument("--score_threshold", type=float, default=0.3,
                        help="Minimum detection confidence.")
    parser.add_argument("--no_ransac_refine", action="store_true",
                        help="Disable RANSAC Δz refinement.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize inference pipeline
    infer = SDFANetInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    if args.input_dir:
        # Batch directory mode
        results = infer.predict_batch_from_dir(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            score_threshold=args.score_threshold,
        )
        print(f"\nProcessed {len(results)} frames.")

    elif args.image_path and args.cloud_path and args.calib_path:
        # Single file mode
        detections = infer.predict_from_files(
            image_path=args.image_path,
            cloud_path=args.cloud_path,
            calib_path=args.calib_path,
            score_threshold=args.score_threshold,
            refine_delta_z=not args.no_ransac_refine,
        )

        print(f"\nDetections ({len(detections)}):")
        for i, det in enumerate(detections):
            print(f"  [{i}] {det['class_name']} | {det['condition']} | "
                  f"score={det['score']:.3f} | Δz={det['delta_z']:.4f}m | "
                  f"pos=({det['cx']:.2f}, {det['cy']:.2f}) | "
                  f"size=({det['width']:.2f}x{det['length']:.2f})m")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(detections, f, indent=2, default=float)
            print(f"\nResults saved to {args.output}")

    else:
        print("Error: Provide either --input_dir or (--image_path + --cloud_path + --calib_path)")
        sys.exit(1)


if __name__ == "__main__":
    main()
