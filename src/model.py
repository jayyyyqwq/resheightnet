"""ResHeightNet: ResNet34 encoder + skip-connected decoder for single-image
height regression.

Adapted from Amirkolaee & Arefi (2019), "Height estimation from single
aerial images using a deep convolutional encoder-decoder network",
ISPRS J. Photogramm. Remote Sens. 149, 50-66.

Deviation from the paper (disclosed): the encoder is an ImageNet-
pretrained torchvision ResNet34 rather than a residual encoder trained
from scratch. See PROJECT_PLAN.md Section 2/15 for the full rationale.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResHeightNet(nn.Module):
    """ResNet34 encoder (ImageNet-pretrained) + skip-connected decoder.

    Args:
        pretrained: load ImageNet weights for the encoder.
        freeze_stem: if True, sets requires_grad=False on the stem
            (conv1/bn1/relu) parameters. NOTE: this does NOT put the
            stem's BatchNorm in eval mode, so running_mean/running_var
            still update during model.train() even when frozen -- see
            the module-level warning below. The project default is
            freeze_stem=False (fine-tune everything, matching
            DepthWizard's recipe) specifically to avoid this trap;
            freeze_stem=True is kept only for ablation experiments.
    """

    def __init__(self, pretrained: bool = True, freeze_stem: bool = False):
        super().__init__()
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet34(weights=weights)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # H/2, 64ch
        self.pool = resnet.maxpool  # H/4
        self.layer1 = resnet.layer1  # H/4, 64ch
        self.layer2 = resnet.layer2  # H/8, 128ch
        self.layer3 = resnet.layer3  # H/16, 256ch
        self.layer4 = resnet.layer4  # H/32, 512ch

        self.freeze_stem = freeze_stem
        if freeze_stem:
            for p in self.stem.parameters():
                p.requires_grad = False

        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 32)

        self.final_up = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.head = nn.Conv2d(16, 1, kernel_size=1)

        n_trainable = sum(p.requires_grad for p in self.parameters())
        assert n_trainable > 0, (
            "ResHeightNet has zero trainable parameters -- freeze_stem "
            "combined with an optimizer filter on requires_grad would "
            "silently receive an empty parameter list."
        )

    def train(self, mode: bool = True):
        """Override train() so a frozen stem's BatchNorm stays in eval mode.

        Setting requires_grad=False alone does not stop BatchNorm from
        updating running_mean/running_var during forward passes in
        train() mode. If freeze_stem=True, this keeps the stem's BN
        layers in eval() regardless of the outer mode, so "frozen"
        actually means frozen and the run stays reproducible.
        """
        super().train(mode)
        if self.freeze_stem:
            self.stem.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)  # H/2
        p0 = self.pool(s0)  # H/4
        s1 = self.layer1(p0)  # H/4
        s2 = self.layer2(s1)  # H/8
        s3 = self.layer3(s2)  # H/16
        s4 = self.layer4(s3)  # H/32

        d4 = self.dec4(s4, s3)
        d3 = self.dec3(d4, s2)
        d2 = self.dec2(d3, s1)
        d1 = self.dec1(d2, s0)

        out = self.final_up(d1)
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(out)
