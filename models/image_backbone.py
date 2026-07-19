"""
Image Backbone — YOLOv11-based feature extractor.

Uses a lightweight YOLOv11 variant as the image encoder for the rectified
fisheye images. Outputs multi-scale feature maps that are fed into the
SDS-VT view transform module.

If ultralytics is not available, falls back to a ResNet-18 backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict


# ── ResNet Fallback (when ultralytics is not installed) ──────────

class _ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNetBackbone(nn.Module):
    """
    Simple ResNet-18-like backbone as a fallback.
    Outputs multi-scale feature maps.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: List[int] = [128, 256, 512],
    ):
        super().__init__()
        self.out_channels = out_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # FPN-style neck to produce consistent multi-scale outputs
        self._build_neck()

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [_ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(_ResidualBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def _build_neck(self):
        c2, c3, c4 = 128, 256, 512
        out_c = self.out_channels[0]

        self.lateral2 = nn.Conv2d(128, out_c, 1)
        self.lateral3 = nn.Conv2d(256, out_c, 1)
        self.lateral4 = nn.Conv2d(512, out_c, 1)

        self.smooth2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.smooth3 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.smooth4 = nn.Conv2d(out_c, out_c, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.conv1(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)     # /4
        c3 = self.layer2(c2)    # /8
        c4 = self.layer3(c3)    # /16
        c5 = self.layer4(c4)    # /32

        # FPN top-down
        p5 = self.lateral4(c5)
        p4 = self.lateral3(c4) + F.interpolate(p5, size=c4.shape[2:], mode="nearest")
        p3 = self.lateral2(c3) + F.interpolate(p4, size=c3.shape[2:], mode="nearest")

        p3 = self.smooth2(p3)
        p4 = self.smooth3(p4)
        p5 = self.smooth4(p5)

        return [p3, p4, p5]


# ── YOLOv11 Backbone (wrapper around ultralytics) ─────────────────

class YOLOv11Backbone(nn.Module):
    """
    YOLOv11 backbone + FPN neck for image feature extraction.

    Wraps ultralytics YOLO models and extracts multi-scale feature maps
    from different stages of the network.

    If ultralytics is not installed, falls back to ResNetBackbone.
    """

    def __init__(
        self,
        model_name: str = "yolov11n",
        pretrained: bool = True,
        out_channels: List[int] = [128, 256, 512],
        neck_channels: int = 256,
    ):
        super().__init__()
        self.model_name = model_name
        self.out_channels = out_channels
        self.neck_channels = neck_channels

        self._use_yolo = False
        try:
            from ultralytics import YOLO
            self._use_yolo = True
        except ImportError:
            pass

        if self._use_yolo:
            self._init_yolo_backbone(model_name, pretrained)
        else:
            print(f"[YOLOv11Backbone] ultralytics not found, using ResNet fallback.")
            self.resnet = ResNetBackbone(
                in_channels=3,
                out_channels=out_channels,
            )

    def _init_yolo_backbone(self, model_name: str, pretrained: bool):
        """
        Initialize YOLOv11 backbone.

        We construct a YOLOv11-like backbone manually so it's self-contained
        and doesn't depend on the ultralytics model structure.
        """
        # Build a YOLOv11-nano equivalent backbone manually
        c0, c1, c2, c3 = 16, 32, 64, 128

        self.stem = nn.Sequential(
            nn.Conv2d(3, c0, 3, 2, 1, bias=False),
            nn.BatchNorm2d(c0),
            nn.SiLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            nn.Conv2d(c0, c1, 3, 2, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            _C2f(c1, c1, 1),
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
            _C2f(c2, c2, 2),
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, 2, 1, bias=False),
            nn.BatchNorm2d(c3),
            nn.SiLU(inplace=True),
            _C2f(c3, c3, 2),
        )

        self.stage4 = nn.Sequential(
            nn.Conv2d(c3, c3 * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(c3 * 2),
            nn.SiLU(inplace=True),
            _C2f(c3 * 2, c3 * 2, 2),
        )

        # SPPF
        self.sppf = _SPPF(c3 * 2, c3 * 2)

        # Build FPN neck
        out_c = self.out_channels[0]
        self._build_neck(c1, c2, c3, c3 * 2, out_c)

    def _build_neck(self, c1, c2, c3, c4, out_c):
        """Lightweight FPN neck."""
        self.lat1 = nn.Conv2d(c1, out_c, 1)
        self.lat2 = nn.Conv2d(c2, out_c, 1)
        self.lat3 = nn.Conv2d(c3, out_c, 1)
        self.lat4 = nn.Conv2d(c4 * 2, out_c, 1)  # after SPPF doubles channels

        self.smooth1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.smooth2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.smooth3 = nn.Conv2d(out_c, out_c, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] rectified fisheye image.

        Returns:
            feat: [B, neck_channels, H/8, W/8] single-scale feature map
                  ready for SDS-VT.
        """
        if not self._use_yolo:
            # ResNet returns multi-scale; use the middle scale for SDS-VT
            feats = self.resnet(x)
            return feats[1]  # typically 1/8 resolution

        # YOLOv11 forward
        s0 = self.stem(x)            # /2
        s1 = self.stage1(s0)         # /4
        s2 = self.stage2(s1)         # /8
        s3 = self.stage3(s2)         # /16
        s4 = self.stage4(s3)         # /32
        s4_sppf = self.sppf(s4)      # /32

        # FPN top-down
        p4 = self.lat4(s4_sppf)
        p3 = self.lat3(s3) + F.interpolate(p4, size=s3.shape[2:], mode="nearest")
        p2 = self.lat2(s2) + F.interpolate(p3, size=s2.shape[2:], mode="nearest")

        p2 = self.smooth2(p2)

        return p2  # [B, neck_c, H/8, W/8]


# ── YOLOv11 building blocks ──────────────────────────────────────

class _Bottleneck(nn.Module):
    """Standard YOLO bottleneck with residual."""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_hidden = int(c2 * e)
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c_hidden, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_hidden),
            nn.SiLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c_hidden, c2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return x + out if self.add else out


class _C2f(nn.Module):
    """YOLOv8/v11 C2f module."""
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False),
            nn.BatchNorm2d(2 * self.c),
            nn.SiLU(inplace=True),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d((2 + n) * self.c, c2, 1, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.m = nn.ModuleList(
            [_Bottleneck(self.c, self.c, shortcut, 1.0) for _ in range(n)]
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class _SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (YOLOv5/v8/v11)."""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c_, 1, 1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_ * 4, c2, 1, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))
