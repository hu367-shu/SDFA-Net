"""
Multi-Modal Manhole Cover Dataset for SDFA-Net.

Handles loading and preprocessing of:
    - Fisheye images (raw → rectified via Scaramuzza model)
    - Multi-frame aggregated LiDAR point clouds
    - BEV labels (heatmap, offset, size, delta_z)
    - Semi-dense depth ground truth
    - Camera-LiDAR calibration and SLAM poses

Data source: NavVis VLX mobile mapping system.
Dataset: 3500 frames, 7:1:2 train/val/test split.
Classes: good, sunken, raised, broken, missing.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path

from .preprocess_fisheye import FisheyeUndistorter, OCamParams
from .aggregate_lidar import LidarAggregator, aggregate_multiframe_cloud
from .generate_depth_gt import DepthGTGenerator, generate_depth_gt
from .generate_bev_label import BEVLabelGenerator
from .augment import MultiModalAugmentation


class ManholeDataset(Dataset):
    """
    Multi-modal dataset for manhole cover detection and disease diagnosis.

    Each sample contains:
        - Rectified fisheye image [3, H, W]
        - Aggregated LiDAR point cloud [N, 3]
        - BEV labels {heatmap, offset, size, delta_z, pos_mask}
        - Depth GT {depth_gt, depth_mask} for SDS-VT supervision
        - Calibration parameters
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        cfg: Optional[dict] = None,
        transform: Optional[Callable] = None,
        preload_depth_gt: bool = False,
    ):
        """
        Args:
            data_root: Root directory of the dataset.
            split: 'train', 'val', or 'test'.
            cfg: Configuration dictionary (from sdfanet.yaml).
            transform: Optional additional transforms.
            preload_depth_gt: If True, pre-generate and cache depth GT during init.
        """
        self.data_root = Path(data_root)
        self.split = split
        self.cfg = cfg or {}
        self.transform = transform
        self.preload_depth_gt = preload_depth_gt

        # ── Load configuration ─────────────────────────────────────
        self._init_from_cfg()

        # ── Load dataset index ─────────────────────────────────────
        self.samples = self._load_split()

        # ── Initialize processors ───────────────────────────────────
        self._init_processors()

        # ── Preloaded depth GT cache ────────────────────────────────
        self._depth_gt_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        if preload_depth_gt and split != "test":
            self._preload_all_depth_gt()

    def _init_from_cfg(self):
        """Extract parameters from config."""
        model_cfg = self.cfg.get("model", {})
        data_cfg = self.cfg.get("data", {})
        aug_cfg = self.cfg.get("augmentation", {})

        # Image
        self.image_size = tuple(model_cfg.get("sds_vt", {}).get("image_size", [384, 640]))

        # BEV
        sds_cfg = model_cfg.get("sds_vt", {})
        self.bev_h = sds_cfg.get("bev_h", 256)
        self.bev_w = sds_cfg.get("bev_w", 256)
        self.x_range = tuple(sds_cfg.get("x_range", [-25.6, 25.6]))
        self.y_range = tuple(sds_cfg.get("y_range", [-12.8, 38.4]))
        self.fov = sds_cfg.get("fov", 120.0)

        # Depth
        self.depth_bins = sds_cfg.get("depth_bins", 64)
        self.depth_min = sds_cfg.get("depth_min", 0.5)
        self.depth_max = sds_cfg.get("depth_max", 30.0)

        # Geo-Head
        geo_cfg = model_cfg.get("geo_head", {})
        self.num_classes = geo_cfg.get("num_classes", 5)
        self.gaussian_sigma = geo_cfg.get("gaussian_sigma", 1.5)

        # Data
        self.multi_frame_window = data_cfg.get("multi_frame_window", 10)
        self.pose_std_threshold = data_cfg.get("pose_std_threshold", 0.05)
        self.agg_voxel_size = data_cfg.get("agg_voxel_size", 0.05)
        self.class_names = data_cfg.get("classes",
            ["good", "sunken", "raised", "broken", "missing"])

        # Augmentation
        self.aug_cfg = aug_cfg

    def _init_processors(self):
        """Initialize all preprocessing modules."""
        # BEV label generator
        self.bev_label_gen = BEVLabelGenerator(
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            x_range=self.x_range,
            y_range=self.y_range,
            num_classes=self.num_classes,
            gaussian_sigma=self.gaussian_sigma,
        )

        # Depth GT generator
        self.depth_gt_gen = DepthGTGenerator(
            image_size=self.image_size,
            depth_bins=self.depth_bins,
            depth_min=self.depth_min,
            depth_max=self.depth_max,
        )

        # Augmentation (training only)
        if self.split == "train":
            self.augmenter = MultiModalAugmentation(
                mosaic_prob=self.aug_cfg.get("mosaic_prob", 0.5),
                mixup_prob=self.aug_cfg.get("mixup_prob", 0.3),
                rotation_degrees=self.aug_cfg.get("rotation_degrees", 15.0),
                translation_meters=self.aug_cfg.get("translation_meters", 1.0),
                scale_range=tuple(self.aug_cfg.get("scale_range", [0.9, 1.1])),
                color_jitter=self.aug_cfg.get("color_jitter"),
                bev_h=self.bev_h,
                bev_w=self.bev_w,
                x_range=self.x_range,
                y_range=self.y_range,
            )
        else:
            self.augmenter = None

    def _load_split(self) -> List[dict]:
        """
        Load the dataset split index.

        Expected structure:
            {data_root}/
                images/
                pointclouds/
                poses/
                calib/
                labels/
                    train.json / val.json / test.json

        Each label JSON entry:
            {
                "frame_id": "000001",
                "boxes": [[cx, cy, w, l], ...],    # meters in LiDAR frame
                "classes": [0, 1, 2, 3, 4],
                "delta_z": [0.005, -0.025, ...],    # meters
            }
        """
        label_file = self.data_root / "labels" / f"{self.split}.json"

        if label_file.exists():
            with open(label_file, "r") as f:
                samples = json.load(f)
            return samples if isinstance(samples, list) else samples.get("frames", [])

        # Fallback: scan image directory
        img_dir = self.data_root / "images"
        if img_dir.exists():
            img_files = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
            return [{"frame_id": f.stem} for f in img_files]

        return []

    def _preload_all_depth_gt(self):
        """Precompute depth GT for all samples (offline preprocessing)."""
        for idx in range(len(self.samples)):
            try:
                sample = self.samples[idx]
                frame_id = sample["frame_id"]

                # Load aggregated point cloud
                agg_points = self._load_and_aggregate_cloud(frame_id)

                # Load calibration
                calib = self._load_calib(frame_id)

                depth_gt, depth_mask = self.depth_gt_gen.generate(
                    agg_points,
                    calib["T_lidar_cam"],
                    calib["K_virtual"],
                )
                self._depth_gt_cache[idx] = (depth_gt, depth_mask)
            except Exception:
                continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return one training/validation sample."""
        sample = self.samples[idx]
        frame_id = sample["frame_id"]

        # ── 1. Load and rectify fisheye image ──────────────────────
        image = self._load_image(frame_id)
        image = self._rectify_image(image, frame_id)

        # ── 2. Load and aggregate point cloud ──────────────────────
        agg_points = self._load_and_aggregate_cloud(frame_id)

        # ── 3. Load calibration ────────────────────────────────────
        calib = self._load_calib(frame_id)

        # ── 4. Generate BEV labels ─────────────────────────────────
        if "boxes" in sample and sample["boxes"]:
            boxes = np.array(sample["boxes"], dtype=np.float32)
            classes = np.array(sample["classes"], dtype=np.int64)
            delta_zs = np.array(sample["delta_z"], dtype=np.float32)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            classes = np.zeros((0,), dtype=np.int64)
            delta_zs = np.zeros((0,), dtype=np.float32)

        bev_targets = self.bev_label_gen.generate(boxes, classes, delta_zs)

        # ── 5. Generate depth GT (from cache or on-the-fly) ────────
        if idx in self._depth_gt_cache:
            depth_gt, depth_mask = self._depth_gt_cache[idx]
        else:
            depth_gt, depth_mask = self.depth_gt_gen.generate(
                agg_points, calib["T_lidar_cam"], calib["K_virtual"]
            )

        # ── 6. Convert to tensors ──────────────────────────────────
        image_t = torch.from_numpy(np.asarray(image)).float()
        if image_t.dim() == 3 and image_t.shape[-1] == 3:
            image_t = image_t.permute(2, 0, 1)  # HWC → CHW

        points_t = torch.from_numpy(agg_points).float()
        heatmap_t = torch.from_numpy(bev_targets["heatmap"]).float()
        offset_t = torch.from_numpy(bev_targets["offset"]).float()
        size_t = torch.from_numpy(bev_targets["size"]).float()
        delta_z_t = torch.from_numpy(bev_targets["delta_z"]).float()
        pos_mask_t = torch.from_numpy(bev_targets["pos_mask"]).bool()

        depth_gt_t = torch.from_numpy(depth_gt).long()
        depth_mask_t = torch.from_numpy(depth_mask).float()

        # ── 7. Apply augmentation (if training) ────────────────────
        if self.augmenter is not None:
            bev_targets_t = {
                "heatmap": heatmap_t,
                "offset": offset_t,
                "size": size_t,
                "delta_z": delta_z_t,
                "pos_mask": pos_mask_t,
            }
            image_t, points_t, bev_targets_t = self.augmenter(
                image_t, points_t, bev_targets_t
            )
            heatmap_t = bev_targets_t["heatmap"]
            offset_t = bev_targets_t["offset"]
            size_t = bev_targets_t["size"]
            delta_z_t = bev_targets_t["delta_z"]
            pos_mask_t = bev_targets_t["pos_mask"]

        # ── 8. Build final sample dict ─────────────────────────────
        return {
            "frame_id": frame_id,
            "image": image_t,                     # [3, H_img, W_img]
            "points": points_t,                   # [N, 3]
            "calib": {
                "T_lidar_cam": torch.from_numpy(calib["T_lidar_cam"]).float(),
                "K_virtual": torch.from_numpy(calib["K_virtual"]).float(),
            },
            "targets": {
                "heatmap": heatmap_t,             # [num_classes, H_bev, W_bev]
                "offset": offset_t,               # [2, H_bev, W_bev]
                "size": size_t,                   # [2, H_bev, W_bev]
                "delta_z": delta_z_t,             # [1, H_bev, W_bev]
                "pos_mask": pos_mask_t,           # [H_bev, W_bev]
                "depth_gt": depth_gt_t,           # [H_img, W_img]
                "depth_mask": depth_mask_t,       # [H_img, W_img]
                "boxes": torch.from_numpy(boxes).float(),  # [M, 4]
                "classes": torch.from_numpy(classes).long(),  # [M]
                "delta_zs": torch.from_numpy(delta_zs).float(),  # [M]
            },
        }

    # ── Data loading helpers (override these for your data format) ──

    def _load_image(self, frame_id: str) -> np.ndarray:
        """
        Load raw fisheye image. Override for your file format.

        Returns: numpy array [H_raw, W_raw, 3] uint8 or float32.
        """
        img_path = self.data_root / "images" / f"{frame_id}.png"
        if not img_path.exists():
            img_path = self.data_root / "images" / f"{frame_id}.jpg"

        if img_path.exists():
            from PIL import Image
            img = Image.open(img_path)
            return np.array(img, dtype=np.float32) / 255.0

        # Placeholder: random image for demo
        H, W = self.image_size
        return np.random.rand(H, W, 3).astype(np.float32)

    def _rectify_image(
        self, image: np.ndarray, frame_id: str
    ) -> np.ndarray:
        """
        Rectify fisheye image. The undistorter is created per sample
        since different cameras may have different Scaramuzza params.

        Override to load actual OCam params from calibration files.
        """
        calib = self._load_calib(frame_id)

        if "ocam_params" in calib and calib["ocam_params"] is not None:
            undistorter = FisheyeUndistorter(
                ocam_params=calib["ocam_params"],
                out_size=self.image_size,
                fov=self.fov,
            )
            return undistorter.rectify_numpy(image)

        # Fallback: simple center crop + resize as placeholder
        H_raw, W_raw = image.shape[:2]
        crop_size = min(H_raw, W_raw)
        start_h = (H_raw - crop_size) // 2
        start_w = (W_raw - crop_size) // 2
        cropped = image[start_h:start_h + crop_size, start_w:start_w + crop_size]

        from PIL import Image
        H_out, W_out = self.image_size
        img_pil = Image.fromarray((cropped * 255).astype(np.uint8))
        img_pil = img_pil.resize((W_out, H_out), Image.BILINEAR)
        return np.array(img_pil, dtype=np.float32) / 255.0

    def _load_and_aggregate_cloud(self, frame_id: str) -> np.ndarray:
        """
        Load raw LiDAR point cloud and perform multi-frame aggregation.

        Override for your file format.
        Returns: [M, 3] xyz numpy array.
        """
        cloud_path = self.data_root / "pointclouds" / f"{frame_id}.npy"
        if cloud_path.exists():
            cloud = np.load(cloud_path)
            return cloud[:, :3].astype(np.float32)

        # Placeholder: random point cloud for demo
        return np.random.randn(5000, 3).astype(np.float32) * 10.0

    def _load_calib(self, frame_id: str) -> dict:
        """
        Load calibration parameters.

        Returns dict with keys:
            - 'T_lidar_cam': [4, 4] LiDAR → Camera transform
            - 'K_virtual': [3, 3] virtual pinhole camera intrinsic
            - 'ocam_params': OCamParams (optional, may be None)
        """
        calib_path = self.data_root / "calib" / f"{frame_id}.json"
        if calib_path.exists():
            with open(calib_path, "r") as f:
                calib_data = json.load(f)

            T_lidar_cam = np.array(calib_data.get("T_lidar_cam", np.eye(4)), dtype=np.float32)
            K_virtual = np.array(calib_data.get("K_virtual", np.eye(3)), dtype=np.float32)

            ocam_params = None
            if "ocam" in calib_data:
                ocam_data = calib_data["ocam"]
                ocam_params = OCamParams(
                    poly=np.array(ocam_data["poly"], dtype=np.float64),
                    c=ocam_data.get("c", 1.0),
                    d=ocam_data.get("d", 0.0),
                    e=ocam_data.get("e", 1.0),
                    cx=ocam_data.get("cx", 0.0),
                    cy=ocam_data.get("cy", 0.0),
                    height=ocam_data.get("height", self.image_size[0]),
                    width=ocam_data.get("width", self.image_size[1]),
                )

            return {
                "T_lidar_cam": T_lidar_cam,
                "K_virtual": K_virtual,
                "ocam_params": ocam_params,
            }

        # Default calibration
        # Virtual pinhole camera (from FOV and image size)
        H, W = self.image_size
        f = W / (2.0 * np.tan(np.radians(self.fov) / 2.0))
        K_virtual = np.array([
            [f, 0, W / 2],
            [0, f, H / 2],
            [0, 0, 1],
        ], dtype=np.float32)

        return {
            "T_lidar_cam": np.eye(4, dtype=np.float32),
            "K_virtual": K_virtual,
            "ocam_params": None,
        }

    def _load_poses(self, frame_id: str) -> Optional[np.ndarray]:
        """Load SLAM pose for a frame. Returns [4, 4] LiDAR→World matrix."""
        pose_path = self.data_root / "poses" / f"{frame_id}.txt"
        if pose_path.exists():
            return np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4)
        return np.eye(4, dtype=np.float32)


def custom_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function that handles variable-sized point clouds
    and nested dict structures.

    Point clouds are padded to the max size in the batch.
    """
    B = len(batch)
    if B == 0:
        return {}

    # ── Collect fixed-size tensors ───────────────────────────────
    images = torch.stack([item["image"] for item in batch], dim=0)

    heatmaps = torch.stack([item["targets"]["heatmap"] for item in batch], dim=0)
    offsets = torch.stack([item["targets"]["offset"] for item in batch], dim=0)
    sizes = torch.stack([item["targets"]["size"] for item in batch], dim=0)
    delta_zs = torch.stack([item["targets"]["delta_z"] for item in batch], dim=0)
    pos_masks = torch.stack([item["targets"]["pos_mask"] for item in batch], dim=0)
    depth_gts = torch.stack([item["targets"]["depth_gt"] for item in batch], dim=0)
    depth_masks = torch.stack([item["targets"]["depth_mask"] for item in batch], dim=0)

    # ── Collect calibration ──────────────────────────────────────
    T_lidar_cams = torch.stack([item["calib"]["T_lidar_cam"] for item in batch], dim=0)
    K_virtuals = torch.stack([item["calib"]["K_virtual"] for item in batch], dim=0)

    # ── Pad point clouds ─────────────────────────────────────────
    point_list = [item["points"] for item in batch]
    max_pts = max(p.shape[0] for p in point_list)
    C_pts = point_list[0].shape[1]

    points_padded = torch.zeros(B, max_pts, C_pts)
    point_masks = torch.zeros(B, max_pts, dtype=torch.bool)

    for i, pts in enumerate(point_list):
        n = pts.shape[0]
        points_padded[i, :n] = pts
        point_masks[i, :n] = True

    # ── Collect frame IDs ────────────────────────────────────────
    frame_ids = [item["frame_id"] for item in batch]

    return {
        "frame_id": frame_ids,
        "image": images,
        "points": points_padded,
        "point_mask": point_masks,
        "calib": {
            "T_lidar_cam": T_lidar_cams,
            "K_virtual": K_virtuals,
        },
        "targets": {
            "heatmap": heatmaps,
            "offset": offsets,
            "size": sizes,
            "delta_z": delta_zs,
            "pos_mask": pos_masks,
            "depth_gt": depth_gts,
            "depth_mask": depth_masks,
        },
    }


def create_dataloaders(
    data_root: str,
    cfg: dict,
) -> Dict[str, DataLoader]:
    """
    Create train/val/test dataloaders.

    Args:
        data_root: Path to dataset root.
        cfg: Full configuration dict.

    Returns:
        Dict mapping split name to DataLoader.
    """
    batch_size = cfg.get("training", {}).get("batch_size", 8)
    num_workers = cfg.get("training", {}).get("num_workers", 4)

    dataloaders = {}

    for split in ["train", "val", "test"]:
        dataset = ManholeDataset(
            data_root=data_root,
            split=split,
            cfg=cfg,
            preload_depth_gt=(split != "train"),
        )

        shuffle = (split == "train")
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            drop_last=(split == "train"),
        )

    return dataloaders
