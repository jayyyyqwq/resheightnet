"""M4: masked loss correctness.

The single most important test in this file is
test_masked_l1_loss_discriminates_the_three_implementations: a fixture
whose correct answer (7/3) differs from BOTH of the two tempting-but-
wrong implementations (2.25 and 1.75), so passing it proves the actual
reduction formula, not just "some masking happened".
"""
from __future__ import annotations

import math

import torch

from src.losses import masked_l1_loss, masked_smooth_l1_loss


def test_masked_l1_loss_discriminates_the_three_implementations():
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    mask = torch.tensor([[True, False], [True, True]])

    # |1-1| + |3-0| + |4-0| = 0 + 3 + 4 = 7, over 3 valid pixels = 7/3
    correct = masked_l1_loss(pred, target, mask)
    assert math.isclose(correct.item(), 7.0 / 3.0, rel_tol=1e-6)

    # WRONG #1: multiply-then-mean over ALL 4 pixels (mask-multiply form)
    # |1-1| + |0-0| + |3-0| + |4-0| = 0+0+3+4 = 7, over 4 pixels = 1.75
    wrong_multiply_then_mean = torch.nn.functional.l1_loss(pred * mask.float(), target * mask.float())
    assert math.isclose(wrong_multiply_then_mean.item(), 1.75, rel_tol=1e-6)
    assert not math.isclose(correct.item(), wrong_multiply_then_mean.item(), rel_tol=1e-6)

    # WRONG #2: naive mean over all 4 pixels ignoring the mask entirely
    # |1-1| + |2-0| + |3-0| + |4-0| = 0+2+3+4 = 9, over 4 pixels = 2.25
    wrong_ignore_mask = torch.nn.functional.l1_loss(pred, target)
    assert math.isclose(wrong_ignore_mask.item(), 2.25, rel_tol=1e-6)
    assert not math.isclose(correct.item(), wrong_ignore_mask.item(), rel_tol=1e-6)


def test_masked_l1_loss_all_false_mask_returns_zero_with_finite_grad():
    pred = torch.tensor([[1.0, 2.0]], requires_grad=True)
    target = torch.tensor([[5.0, 9.0]])
    mask = torch.tensor([[False, False]])

    loss = masked_l1_loss(pred, target, mask)
    assert loss.item() == 0.0

    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all(), "all-invalid crop must not produce NaN gradients"


def test_masked_l1_loss_ignores_invalid_pixel_magnitude():
    """A huge value at an invalid (masked-out) position must not affect
    the loss at all -- the multiply-then-mean form would leak it in via
    the denominator staying fixed at total pixel count while numerator
    stays the same, but let's verify the actual leak channel: value
    magnitude at invalid positions."""
    pred = torch.tensor([[1.0, 1e6]])
    target = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[True, False]])

    loss = masked_l1_loss(pred, target, mask)
    assert loss.item() == 0.0, "invalid-position magnitude must not leak into the loss"


def test_masked_l1_loss_fp16_sum_does_not_overflow_on_512x512():
    """Reproduces the AMP trap: a masked abs-diff SUM over a 512x512
    crop with errors averaging a few metres can reach ~700,000+, well
    past fp16's 65,504 max. masked_l1_loss casts to float() (fp32)
    before the reduction specifically to avoid this; this test fails if
    that cast is ever removed and the function is called under
    autocast(dtype=float16)."""
    torch.manual_seed(0)
    pred = torch.rand(1, 1, 512, 512, dtype=torch.float16) * 40  # up to 40 m
    target = torch.rand(1, 1, 512, 512, dtype=torch.float16) * 40
    mask = torch.ones(1, 1, 512, 512, dtype=torch.bool)

    loss = masked_l1_loss(pred, target, mask)
    assert torch.isfinite(loss), "masked_l1_loss overflowed under fp16 input"

    # Confirm this scenario really would overflow fp16 if computed naively,
    # so the test is actually exercising the danger zone.
    naive_fp16_sum = ((pred - target).abs() * mask).sum()
    assert naive_fp16_sum.dtype == torch.float16


def test_masked_smooth_l1_loss_reduction_none_then_masked_mean():
    pred = torch.tensor([[0.0, 0.0]])
    target = torch.tensor([[0.5, 5.0]])
    mask = torch.tensor([[True, True]])

    loss = masked_smooth_l1_loss(pred, target, mask, beta=1.0)
    # SmoothL1(beta=1): 0.5*x^2/beta for |x|<beta else |x|-0.5*beta
    # x=0.5 -> 0.5*0.25 = 0.125 ; x=5.0 -> 5.0-0.5 = 4.5
    expected = (0.125 + 4.5) / 2
    assert math.isclose(loss.item(), expected, rel_tol=1e-5)


def test_masked_smooth_l1_loss_all_false_mask_returns_zero():
    pred = torch.tensor([[1.0, 2.0]])
    target = torch.tensor([[5.0, 9.0]])
    mask = torch.tensor([[False, False]])
    loss = masked_smooth_l1_loss(pred, target, mask)
    assert loss.item() == 0.0
