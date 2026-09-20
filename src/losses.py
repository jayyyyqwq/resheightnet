"""Masked regression losses for height estimation.

There is exactly one correct way to reduce a masked per-pixel loss.
Two tempting alternatives are wrong in different ways:

    F.l1_loss(pred * mask, target * mask)   # WRONG: divides by ALL
                                             # pixels including masked-out
                                             # ones, so gradient magnitude
                                             # silently drifts with the
                                             # per-crop nodata fraction.
    F.l1_loss(pred[mask], target[mask])     # correct value, but raises/
                                             # NaNs when mask is empty.

The correct form used here:

    ((pred - target).abs() * mask).sum() / mask.sum().clamp(min=1.0)

`.clamp(min=1.0)` on the denominator turns an all-invalid crop (which
would otherwise be 0/0 -> NaN -> a permanently dead model, since nodata
is spatially clustered at tile edges/water and an all-invalid 512^2
crop is a real event across ~15,000 training samples) into loss 0.0
with a finite (zero) gradient instead.

Under AMP on a T4, the *sum* inside this reduction must not be computed
in fp16: a 512x512 crop's masked abs-diff sum is ~262,144 * ~3m =
~786,000, and fp16 saturates at 65,504 -> inf -> NaN gradients. This
cannot be reproduced on local CPU (fp16 sums there don't overflow the
same way) which is exactly why it is guarded explicitly here rather
than left to be "found" on Colab.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over valid (mask==True) pixels only.

    pred, target, mask: same shape, e.g. (B, 1, H, W). mask may be bool
    or numeric (compared truthy).
    """
    pred_f = pred.float()
    target_f = target.float()
    mask_f = mask.float()

    diff = (pred_f - target_f).abs() * mask_f
    denom = mask_f.sum().clamp(min=1.0)
    return diff.sum() / denom


def masked_smooth_l1_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, beta: float = 1.0
) -> torch.Tensor:
    """Masked SmoothL1 (Huber), kept as a documented ablation.

    Not the headline loss: with beta=1.0 in metres, SmoothL1 squares
    sub-metre errors, de-emphasising exactly the regime the DepthWizard
    reference's median absolute error (1.685 m) lives in, while the
    project is scored on MAE. Plain L1 (masked_l1_loss) is the aligned
    objective and is what train.py uses by default.
    """
    pred_f = pred.float()
    target_f = target.float()
    mask_f = mask.float()

    per_pixel = F.smooth_l1_loss(pred_f, target_f, beta=beta, reduction="none")
    diff = per_pixel * mask_f
    denom = mask_f.sum().clamp(min=1.0)
    return diff.sum() / denom
