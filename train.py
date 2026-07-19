"""
SDFA-Net Training Script.

End-to-end training of SDFA-Net for manhole cover detection and disease diagnosis.

Usage:
    python train.py --config configs/sdfanet.yaml --data_root ./data/manhole_dataset

Training configuration (from paper):
    - Optimizer: AdamW
    - Learning rate: 1e-4 (initial), cosine annealing
    - Batch size: 8
    - Max epochs: 200
    - Warmup: 5 epochs
    - Early stopping: 30 epochs patience
    - AMP (automatic mixed precision): enabled
"""

import os
import sys
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models import SDFANet
from losses import SDFANetLoss
from data import create_dataloaders
from utils.metrics import evaluate_sdfanet
from utils.decode import decode_predictions_batch


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train SDFA-Net")
    parser.add_argument(
        "--config", type=str, default="configs/sdfanet.yaml",
        help="Path to YAML configuration file."
    )
    parser.add_argument(
        "--data_root", type=str, default="./data/manhole_dataset",
        help="Path to dataset root directory."
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from."
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use: 'cuda' or 'cpu'."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed."
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs",
        help="Directory for TensorBoard logs."
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="./checkpoints",
        help="Directory for model checkpoints."
    )
    parser.add_argument(
        "--eval_only", action="store_true",
        help="Run evaluation only (no training)."
    )
    return parser.parse_args()


class Trainer:
    """SDFA-Net training loop with AMP, logging, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        train_loader,
        val_loader,
        cfg: dict,
        device: str = "cuda",
        checkpoint_dir: str = "./checkpoints",
        log_dir: str = "./logs",
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=log_dir)

        # AMP
        self.use_amp = cfg.get("training", {}).get("amp", True)
        self.scaler = GradScaler(enabled=self.use_amp)

        # Training config
        train_cfg = cfg.get("training", {})
        self.max_epochs = train_cfg.get("epochs", 200)
        self.grad_clip_norm = train_cfg.get("grad_clip_norm", 10.0)
        self.early_stop_patience = train_cfg.get("early_stop_patience", 30)
        self.eval_interval = train_cfg.get("eval_interval", 5)
        self.log_interval = train_cfg.get("log_interval", 50)
        self.save_top_k = train_cfg.get("save_top_k", 3)
        self.warmup_epochs = train_cfg.get("warmup_epochs", 5)
        self.warmup_lr = train_cfg.get("warmup_lr", 1e-6)
        self.base_lr = train_cfg.get("learning_rate", 1e-4)

        # Metrics tracking
        self.best_map = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.global_step = 0
        self.current_epoch = 0

        # BEV params for decoding
        self.bev_params = model.get_bev_params()

        # Resume
        self.start_epoch = 0

    def train(self):
        """Run full training loop."""
        print(f"\n{'='*60}")
        print(f"SDFA-Net Training")
        print(f"  Device: {self.device}")
        print(f"  AMP: {self.use_amp}")
        print(f"  Max epochs: {self.max_epochs}")
        print(f"  Train samples: {len(self.train_loader.dataset)}")
        print(f"  Val samples: {len(self.val_loader.dataset)}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
        print(f"{'='*60}\n")

        for epoch in range(self.start_epoch, self.max_epochs):
            self.current_epoch = epoch

            # Train one epoch
            train_losses = self.train_one_epoch(epoch)

            # Log training losses
            for name, val in train_losses.items():
                self.writer.add_scalar(f"train/{name}", val, epoch)

            # Evaluate
            if (epoch + 1) % self.eval_interval == 0 or epoch == self.max_epochs - 1:
                val_metrics = self.validate(epoch)

                # Log validation metrics
                for name, val in val_metrics.items():
                    self.writer.add_scalar(f"val/{name}", val, epoch)

                # Check for best model
                current_map = val_metrics.get("mAP", 0.0)
                if current_map > self.best_map:
                    self.best_map = current_map
                    self.best_epoch = epoch
                    self.epochs_no_improve = 0
                    self.save_checkpoint("best_model.pth", epoch, val_metrics)
                    print(f"  ✓ New best mAP: {current_map:.4f}")
                else:
                    self.epochs_no_improve += 1

                # Save periodic checkpoint
                if (epoch + 1) % 20 == 0:
                    self.save_checkpoint(f"epoch_{epoch + 1}.pth", epoch, val_metrics)

            # Learning rate scheduling
            if self.scheduler is not None and epoch >= self.warmup_epochs:
                self.scheduler.step()

            # Early stopping
            if self.epochs_no_improve >= self.early_stop_patience:
                print(f"\n  Early stopping at epoch {epoch + 1} "
                      f"(no improvement for {self.early_stop_patience} epochs)")
                break

        print(f"\n{'='*60}")
        print(f"Training complete. Best mAP: {self.best_map:.4f} at epoch {self.best_epoch + 1}")
        print(f"{'='*60}")

        self.writer.close()

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch. Returns average loss dict."""
        self.model.train()
        epoch_losses = {}
        epoch_start = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            batch = self._to_device(batch)

            # Warmup LR
            if epoch < self.warmup_epochs:
                lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * (epoch * len(self.train_loader) + batch_idx) / (self.warmup_epochs * len(self.train_loader))
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

            # Forward + loss
            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    batch["image"],
                    batch["points"],
                    batch["calib"],
                )
                total_loss, loss_dict = self.loss_fn(outputs, batch["targets"])

            # Backward
            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_norm
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate losses
            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v

            self.global_step += 1

            # Log
            if batch_idx % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - epoch_start
                msg = (
                    f"Epoch [{epoch + 1}/{self.max_epochs}] "
                    f"[{batch_idx}/{len(self.train_loader)}] "
                    f"lr: {current_lr:.6f} | "
                    f"loss: {total_loss.item():.4f} | "
                )
                for k, v in loss_dict.items():
                    msg += f"{k}: {v:.4f} | "
                msg += f"time: {elapsed:.1f}s"
                print(msg)

                self.writer.add_scalar("train/lr", current_lr, self.global_step)

        # Average over epoch
        n_batches = len(self.train_loader)
        for k in epoch_losses:
            epoch_losses[k] /= n_batches

        return epoch_losses

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Run validation and compute metrics."""
        print(f"\n  Validating epoch {epoch + 1}...")
        self.model.eval()

        all_detections = []
        all_ground_truths = []
        val_losses = {}

        for batch in self.val_loader:
            batch = self._to_device(batch)

            outputs = self.model(
                batch["image"],
                batch["points"],
                batch["calib"],
            )

            _, loss_dict = self.loss_fn(outputs, batch["targets"])
            for k, v in loss_dict.items():
                val_losses[k] = val_losses.get(k, 0.0) + v

            # Decode predictions
            batch_dets = decode_predictions_batch(
                outputs,
                self.bev_params,
                score_threshold=0.3,
                max_detections=100,
            )
            all_detections.extend(batch_dets)

            # Collect GT
            targets = batch["targets"]
            B = batch["image"].shape[0]
            for b in range(B):
                boxes = targets.get("boxes")
                classes = targets.get("classes")
                delta_zs = targets.get("delta_zs")

                if boxes is not None:
                    frame_gts = []
                    for m in range(len(classes[b])):
                        frame_gts.append({
                            "cx": float(boxes[b][m, 0]),
                            "cy": float(boxes[b][m, 1]),
                            "width": float(boxes[b][m, 2]),
                            "length": float(boxes[b][m, 3]),
                            "class_id": int(classes[b][m]),
                            "delta_z": float(delta_zs[b][m]) if delta_zs is not None else 0.0,
                        })
                    all_ground_truths.append(frame_gts)

        # Average val losses
        n_batches = len(self.val_loader)
        for k in val_losses:
            val_losses[k] /= n_batches

        # Compute detection metrics
        from utils.metrics import compute_map
        num_classes = self.cfg.get("model", {}).get("geo_head", {}).get("num_classes", 5)
        det_metrics = compute_map(
            all_detections, all_ground_truths,
            iou_threshold=0.5, num_classes=num_classes,
        )

        metrics = {**val_losses, **det_metrics}

        print(f"  Val | loss: {val_losses.get('loss_total', 0):.4f} | "
              f"mAP: {det_metrics.get('mAP', 0):.4f}")

        return metrics

    def save_checkpoint(
        self, filename: str, epoch: int, metrics: Dict[str, float]
    ):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            "metrics": metrics,
            "best_map": self.best_map,
            "cfg": self.cfg,
        }
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint for resuming training."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.use_amp and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_map = checkpoint.get("best_map", 0.0)
        print(f"  Resumed from {path}, starting epoch {self.start_epoch + 1}")

    def _to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to the target device."""
        moved = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(self.device)
            elif isinstance(v, dict):
                moved[k] = self._to_device(v)
            elif isinstance(v, list):
                moved[k] = v
            else:
                moved[k] = v
        return moved


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Override log/checkpoint dirs
    cfg["logging"] = cfg.get("logging", {})
    cfg["logging"]["log_dir"] = args.log_dir
    cfg["logging"]["checkpoint_dir"] = args.checkpoint_dir

    # Create dataloaders
    print("Creating dataloaders...")
    dataloaders = create_dataloaders(
        data_root=args.data_root,
        cfg=cfg,
    )
    train_loader = dataloaders.get("train")
    val_loader = dataloaders.get("val")

    # Build model
    print("Building SDFA-Net model...")
    model = SDFANet(cfg)

    # Build loss
    loss_cfg = cfg.get("loss", {})
    loss_weights = {
        "hm": loss_cfg.get("hm", 1.0),
        "offset": loss_cfg.get("offset", 1.0),
        "size": loss_cfg.get("size", 1.0),
        "height": loss_cfg.get("height", 2.0),
        "depth": loss_cfg.get("depth", 0.5),
    }
    loss_fn = SDFANetLoss(loss_weights=loss_weights)

    # Build optimizer
    train_cfg = cfg.get("training", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    # Build scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg.get("epochs", 200) - train_cfg.get("warmup_epochs", 5),
        eta_min=train_cfg.get("learning_rate", 1e-4) * 0.01,
    )

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=args.device if torch.cuda.is_available() else "cpu",
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    if args.eval_only:
        metrics = trainer.validate(0)
        print("\nEvaluation metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    else:
        trainer.train()


if __name__ == "__main__":
    main()
