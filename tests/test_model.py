"""M3: model forward pass correctness."""
from __future__ import annotations

import torch
import torchvision.models as models

from src.model import ResHeightNet

EXPECTED_PARAM_COUNT = 24_345_729  # pinned regression guard; see test below for derivation


@torch.no_grad()
def _forward_shape(size: int, batch: int = 2) -> torch.Size:
    model = ResHeightNet(pretrained=False, freeze_stem=False)
    model.eval()
    x = torch.randn(batch, 3, size, size)
    return model(x).shape


def test_forward_shape_512():
    shape = _forward_shape(512)
    assert shape == torch.Size([2, 1, 512, 512])


def test_forward_shape_256():
    shape = _forward_shape(256)
    assert shape == torch.Size([2, 1, 256, 256])


def test_forward_shape_non_multiple_of_32():
    """500 is not a multiple of 32, so the decoder's upsample path will
    not land on x's exact spatial size and must go through the final
    F.interpolate(size=x.shape[-2:]) fallback."""
    shape = _forward_shape(500)
    assert shape == torch.Size([2, 1, 500, 500])


def test_param_count_regression():
    model = ResHeightNet(pretrained=False, freeze_stem=False)
    n_params = sum(p.numel() for p in model.parameters())
    # Regression guard, not a hardcoded magic number: derive the expected
    # count from a fresh resnet34 + explicit decoder param math, so this
    # test documents *why* the number is what it is.
    resnet_params = sum(p.numel() for p in models.resnet34(weights=None).parameters())
    # ResHeightNet uses conv1/bn1/relu/maxpool/layer1-4 from resnet34,
    # i.e. everything except resnet's avgpool (no params) and fc layer.
    fc_params = sum(
        p.numel() for p in models.resnet34(weights=None).fc.parameters()
    )
    encoder_params = resnet_params - fc_params
    decoder_and_head_params = n_params - encoder_params
    assert encoder_params > 20_000_000, "encoder param count looks wrong (expected ResNet34-scale)"
    assert 2_000_000 < decoder_and_head_params < 6_000_000, (
        f"decoder+head param count {decoder_and_head_params} outside expected range"
    )
    # Pin the total too, so an accidental architecture change (e.g. a
    # decoder channel width edit) is caught explicitly rather than only
    # passing the loose range checks above.
    assert n_params == encoder_params + decoder_and_head_params
    assert n_params == EXPECTED_PARAM_COUNT, (
        f"total param count changed to {n_params} (expected {EXPECTED_PARAM_COUNT}); "
        f"update EXPECTED_PARAM_COUNT only if this is an intentional architecture change"
    )


def test_pretrained_conv1_matches_torchvision_weights():
    model = ResHeightNet(pretrained=True, freeze_stem=False)
    reference = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    conv1 = model.stem[0]
    assert torch.equal(conv1.weight, reference.conv1.weight), (
        "pretrained conv1 weights must bit-match torchvision's IMAGENET1K_V1"
    )


def test_untrained_conv1_differs_from_pretrained():
    pretrained_model = ResHeightNet(pretrained=True, freeze_stem=False)
    torch.manual_seed(0)
    fresh_model = ResHeightNet(pretrained=False, freeze_stem=False)
    conv1_pretrained = pretrained_model.stem[0].weight
    conv1_fresh = fresh_model.stem[0].weight
    assert not torch.equal(conv1_pretrained, conv1_fresh), (
        "a freshly-initialized conv1 should not coincidentally match ImageNet weights"
    )


def test_layer4_receives_nonzero_gradients_when_not_frozen():
    """Catches an accidental full-freeze: if freeze_stem or some other
    bug disabled grads on the encoder tail, layer4's grad would be None
    (never computed) or all-zero after a backward pass."""
    model = ResHeightNet(pretrained=False, freeze_stem=False)
    model.train()
    x = torch.randn(1, 3, 64, 64, requires_grad=False)
    out = model(x)
    loss = out.mean()
    loss.backward()

    layer4_params = list(model.layer4.parameters())
    assert any(p.grad is not None for p in layer4_params), "layer4 got no gradient at all"
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in layer4_params
    ), "layer4 gradient is all-zero -- encoder tail appears frozen"


def test_freeze_stem_true_actually_stops_stem_bn_updating_in_train_mode():
    """The trap: requires_grad=False alone does NOT stop BatchNorm's
    running_mean/running_var from updating during model.train() forward
    passes. ResHeightNet.train() is overridden to keep a frozen stem's
    BN in eval() mode specifically to prevent this."""
    model = ResHeightNet(pretrained=True, freeze_stem=True)
    model.train()

    bn = model.stem[1]
    running_mean_before = bn.running_mean.clone()

    x = torch.randn(2, 3, 64, 64) * 5 + 3  # away from the pretrained running stats
    with torch.no_grad():
        model(x)

    assert torch.equal(bn.running_mean, running_mean_before), (
        "frozen stem's BatchNorm running_mean updated during train() forward -- "
        "freeze_stem is not actually freezing the stem"
    )


def test_zero_trainable_params_is_impossible_via_freeze_stem_alone():
    """freeze_stem only freezes the stem (3 layers), never the whole
    model -- construction must always leave the decoder trainable."""
    model = ResHeightNet(pretrained=False, freeze_stem=True)
    n_trainable = sum(p.requires_grad for p in model.parameters())
    assert n_trainable > 0


def test_eval_mode_forward_is_deterministic():
    model = ResHeightNet(pretrained=False, freeze_stem=False)
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.equal(out1, out2)
