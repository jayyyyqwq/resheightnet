"""M2: dataset + mask correctness.

All synthetic -- no real GAMUS data required. Covers the correctness
traps identified during planning: RGB layout normalization, CLS dtype
consistency, nodata masking without height mutation, crop/augmentation
alignment across all four arrays (rgb/height/cls/mask), contiguity
after flips/rotations, and the numpy-RNG DataLoader-worker trap.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.dataset import (
    GamusHeightDataset,
    MIN_CROP_VALID_FRACTION,
    _hflip_all,
    _normalize_rgb_layout,
    _rot90_all,
    _vflip_all,
    compute_valid_mask,
    parse_split_file,
)
from src.utils import IMAGENET_MEAN, IMAGENET_STD
from tests.helpers import make_synthetic_tile, write_split_file


# ---------------------------------------------------------------------------
# RGB layout normalization
# ---------------------------------------------------------------------------


def test_normalize_rgb_layout_channel_last_unchanged():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[0, 0] = [1, 2, 3]
    out = _normalize_rgb_layout(rgb)
    assert out.shape == (8, 8, 3)
    assert np.array_equal(out[0, 0], [1, 2, 3])


def test_normalize_rgb_layout_channel_first_transposed_correctly():
    rgb_chw = np.zeros((3, 8, 8), dtype=np.uint8)
    rgb_chw[:, 2, 5] = [10, 20, 30]  # pixel (row=2, col=5)
    out = _normalize_rgb_layout(rgb_chw)
    assert out.shape == (8, 8, 3)
    assert np.array_equal(out[2, 5], [10, 20, 30]), "pixel identity must survive the transpose"


def test_normalize_rgb_layout_rejects_bad_shape():
    with pytest.raises(ValueError):
        _normalize_rgb_layout(np.zeros((8, 8), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Masking: the nodata-killer invariant
# ---------------------------------------------------------------------------


def test_valid_mask_excludes_negative_height():
    height = np.array([[5.0, -5.0], [10.0, -0.01]], dtype=np.float32)
    cls = np.zeros((2, 2), dtype=np.uint8)
    mask = compute_valid_mask(height, cls)
    assert mask.tolist() == [[True, False], [True, False]]


def test_valid_mask_excludes_nan_and_inf():
    height = np.array([[5.0, np.nan], [np.inf, -np.inf]], dtype=np.float32)
    cls = np.zeros((2, 2), dtype=np.uint8)
    mask = compute_valid_mask(height, cls)
    assert mask.tolist() == [[True, False], [False, False]]


def test_valid_mask_excludes_out_of_range_class():
    height = np.full((2, 2), 5.0, dtype=np.float32)
    cls = np.array([[0, 7], [-1, 6]])
    mask = compute_valid_mask(height, cls)
    assert mask.tolist() == [[True, False], [False, True]]


def test_cls_dtype_float_and_uint8_give_identical_masks():
    height = np.full((16, 16), 5.0, dtype=np.float32)
    rng = np.random.default_rng(0)
    cls_uint8 = rng.integers(0, 9, size=(16, 16)).astype(np.uint8)  # includes some 7/8 (invalid)
    cls_float = cls_uint8.astype(np.float32)

    mask_uint8 = compute_valid_mask(height, cls_uint8)
    mask_float = compute_valid_mask(height, cls_float)
    assert np.array_equal(mask_uint8, mask_float), "mask must be bit-identical across cls dtypes"


def test_dataset_never_mutates_height_array(tmp_path):
    root = str(tmp_path / "gamus")
    tile = make_synthetic_tile(root, "DC_1_1", size=128, nodata_fraction=0.05, seed=1)
    split_file = tmp_path / "split.txt"
    write_split_file(str(split_file), [tile])

    ds = GamusHeightDataset(str(split_file), data_root=root, patch_size=64, train=False)
    sample = ds[0]
    height = sample["height"].numpy()
    # The synthetic tile was written with 5% nodata pixels at -5.0; if the
    # loader ever "cleaned" the data (clamp/nan_to_num), this would be
    # False for a large enough crop given the DepthWizard nodata density.
    assert (height < 0).any(), (
        "expected negative (nodata) values to survive into the loaded "
        "sample untouched -- height must never be modified, only masked"
    )


# ---------------------------------------------------------------------------
# Normalization order
# ---------------------------------------------------------------------------


def test_normalization_order_matches_manual_computation(tmp_path):
    root = str(tmp_path / "gamus")
    size = 64
    # Constant uint8 127 RGB tile.
    from tests.helpers import write_h5

    write_h5(f"{root}/images/train/A_RGB.h5", np.full((size, size, 3), 127, dtype=np.uint8))
    write_h5(f"{root}/heights/train/A_AGL.h5", np.full((size, size), 5.0, dtype=np.float32))
    write_h5(f"{root}/classes/train/A_CLS.h5", np.zeros((size, size), dtype=np.uint8))

    split_file = tmp_path / "split.txt"
    write_split_file(
        str(split_file),
        [
            {
                "sample_id": "A",
                "rgb_relpath": "images/train/A_RGB.h5",
                "height_relpath": "heights/train/A_AGL.h5",
                "cls_relpath": "classes/train/A_CLS.h5",
            }
        ],
    )

    ds = GamusHeightDataset(str(split_file), data_root=root, patch_size=32, train=False)
    sample = ds[0]
    rgb = sample["rgb"].numpy()  # (3, H, W)

    expected = (127.0 / 255.0 - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    for c in range(3):
        assert np.allclose(rgb[c], expected[c], atol=1e-6), (
            f"channel {c}: normalization order (/255 then ImageNet mean/std) not matched"
        )


# ---------------------------------------------------------------------------
# Crop / augmentation alignment (the ramp trick)
# ---------------------------------------------------------------------------


def _make_ramp_arrays(size: int):
    """rgb channel0=row%256, channel1=col%256; height and cls encode the
    same (row, col) index so post-transform alignment can be verified by
    decoding the RGB channels and checking height/cls/mask agree at the
    same output pixel."""
    rows = np.arange(size).reshape(-1, 1)
    cols = np.arange(size).reshape(1, -1)
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[:, :, 0] = (rows % 256).astype(np.uint8)
    rgb[:, :, 1] = (cols % 256).astype(np.uint8)
    rgb[:, :, 2] = 0

    # height encodes row*1000+col (as a positive, valid value) so it can
    # be checked against the decoded RGB row/col.
    height = (rows * 1000 + cols).astype(np.float32)
    cls = np.zeros((size, size), dtype=np.uint8)
    mask = np.ones((size, size), dtype=bool)
    return rgb, height, cls, mask


def test_ramp_alignment_survives_hflip():
    rgb, height, cls, mask = _make_ramp_arrays(16)
    rgb2, height2, cls2, mask2 = _hflip_all(rgb, height, cls, mask)
    _assert_ramp_consistent(rgb2, height2)


def test_ramp_alignment_survives_vflip():
    rgb, height, cls, mask = _make_ramp_arrays(16)
    rgb2, height2, cls2, mask2 = _vflip_all(rgb, height, cls, mask)
    _assert_ramp_consistent(rgb2, height2)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_ramp_alignment_survives_rot90(k):
    rgb, height, cls, mask = _make_ramp_arrays(16)
    rgb2, height2, cls2, mask2 = _rot90_all(rgb, height, cls, mask, k)
    _assert_ramp_consistent(rgb2, height2)


def _assert_ramp_consistent(rgb: np.ndarray, height: np.ndarray):
    """Decode row/col from the RGB ramp channels and check height at the
    SAME output pixel encodes the SAME original (row, col) -- this fails
    immediately if rgb and height were cropped/flipped independently."""
    decoded_row = rgb[:, :, 0].astype(np.int64)
    decoded_col = rgb[:, :, 1].astype(np.int64)
    expected_height = decoded_row * 1000 + decoded_col
    assert np.array_equal(height.astype(np.int64), expected_height), (
        "height array desynchronized from RGB after transform -- "
        "crop/augmentation alignment is broken"
    )


def test_contiguous_after_transforms():
    rgb, height, cls, mask = _make_ramp_arrays(16)
    rgb2, height2, cls2, mask2 = _rot90_all(rgb, height, cls, mask, 1)
    for arr in (rgb2, height2, cls2, mask2):
        assert arr.flags["C_CONTIGUOUS"], "torch.from_numpy requires non-negative strides"
    # This is the actual failure mode being guarded against: negative
    # strides raise inside torch.from_numpy.
    torch.from_numpy(rgb2)
    torch.from_numpy(height2)


def test_dataset_random_crop_alignment_end_to_end(tmp_path):
    """Full pipeline (dataset __getitem__ including augmentation) must
    keep rgb/height/cls/mask aligned at every output pixel."""
    root = str(tmp_path / "gamus")
    size = 128
    rgb, height, cls, mask = _make_ramp_arrays(size)
    from tests.helpers import write_h5

    write_h5(f"{root}/images/train/A_RGB.h5", rgb)
    write_h5(f"{root}/heights/train/A_AGL.h5", height)
    write_h5(f"{root}/classes/train/A_CLS.h5", cls)

    split_file = tmp_path / "split.txt"
    write_split_file(
        str(split_file),
        [
            {
                "sample_id": "A",
                "rgb_relpath": "images/train/A_RGB.h5",
                "height_relpath": "heights/train/A_AGL.h5",
                "cls_relpath": "classes/train/A_CLS.h5",
            }
        ],
    )

    # min_tile_valid_fraction=0 since this synthetic tile's height values
    # (row*1000+col) are all >= 0 and valid by construction; the check
    # here is about alignment, not nodata handling.
    ds = GamusHeightDataset(
        str(split_file), data_root=root, patch_size=32, train=True, min_tile_valid_fraction=0.0
    )

    # Undo the ImageNet normalization to recover raw RGB values for decoding.
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)

    for _ in range(20):
        sample = ds[0]
        rgb_t = sample["rgb"].numpy().transpose(1, 2, 0)  # (H, W, 3)
        raw = np.clip(np.round((rgb_t * std + mean) * 255.0), 0, 255).astype(np.int64)
        height_out = sample["height"].numpy()[0]  # (H, W)

        decoded_row = raw[:, :, 0]
        decoded_col = raw[:, :, 1]
        expected_height = decoded_row * 1000 + decoded_col
        assert np.array_equal(height_out.astype(np.int64), expected_height), (
            "end-to-end dataset pipeline desynchronized rgb/height for some crop+augmentation"
        )


# ---------------------------------------------------------------------------
# Center-crop determinism for eval mode
# ---------------------------------------------------------------------------


def test_val_mode_crops_at_exact_center(tmp_path):
    root = str(tmp_path / "gamus")
    tile = make_synthetic_tile(root, "DC_1_1", size=1024, nodata_fraction=0.0, seed=2)
    split_file = tmp_path / "split.txt"
    write_split_file(str(split_file), [tile])

    ds = GamusHeightDataset(str(split_file), data_root=root, patch_size=512, train=False)
    sample = ds[0]
    height_out = sample["height"].numpy()[0]
    expected = tile["height"][256:768, 256:768]
    assert np.array_equal(height_out, expected), "val crop must be exactly top=left=256 for 1024->512"


def test_val_mode_deterministic_across_calls(tmp_path):
    root = str(tmp_path / "gamus")
    tile = make_synthetic_tile(root, "DC_1_1", size=256, nodata_fraction=0.0, seed=3)
    split_file = tmp_path / "split.txt"
    write_split_file(str(split_file), [tile])

    ds = GamusHeightDataset(str(split_file), data_root=root, patch_size=128, train=False)
    s1 = ds[0]["height"].numpy()
    s2 = ds[0]["height"].numpy()
    assert np.array_equal(s1, s2)


def test_train_mode_yields_multiple_distinct_crops(tmp_path):
    root = str(tmp_path / "gamus")
    tile = make_synthetic_tile(root, "DC_1_1", size=256, nodata_fraction=0.0, seed=4)
    split_file = tmp_path / "split.txt"
    write_split_file(str(split_file), [tile])

    ds = GamusHeightDataset(
        str(split_file), data_root=root, patch_size=64, train=True, min_tile_valid_fraction=0.0
    )
    seen = set()
    for _ in range(20):
        h = ds[0]["height"].numpy().tobytes()
        seen.add(h)
    assert len(seen) > 1, "random crop + augmentation should not always produce the same output"


# ---------------------------------------------------------------------------
# Whole-tile filtering (train only, never for val)
# ---------------------------------------------------------------------------


def test_train_drops_mostly_nodata_tiles(tmp_path):
    root = str(tmp_path / "gamus")
    bad_tile = make_synthetic_tile(root, "DC_1_1", size=64, nodata_fraction=0.99, seed=5)
    good_tile = make_synthetic_tile(root, "DC_1_2", size=64, nodata_fraction=0.01, seed=6)
    split_file = tmp_path / "split.txt"
    write_split_file(str(split_file), [bad_tile, good_tile])

    ds = GamusHeightDataset(
        str(split_file), data_root=root, patch_size=32, train=True, min_tile_valid_fraction=0.05
    )
    assert len(ds) == 1
    assert ds.dropped_tiles == ["DC_1_1"]


def test_val_never_drops_tiles_regardless_of_nodata(tmp_path):
    root = str(tmp_path / "gamus")
    bad_tile = make_synthetic_tile(root, "DC_1_1", size=64, nodata_fraction=0.99, seed=7)
    split_file = tmp_path / "split.txt"
    write_split_file(str(split_file), [bad_tile])

    ds = GamusHeightDataset(str(split_file), data_root=root, patch_size=32, train=False)
    assert len(ds) == 1, "val must never drop tiles -- count must match the split file exactly"


# ---------------------------------------------------------------------------
# DataLoader worker RNG (numpy is NOT seeded per-worker by PyTorch)
# ---------------------------------------------------------------------------


def test_augmentation_uses_random_module_not_numpy():
    """Guards the specific bug: if augmentation used np.random instead of
    Python's random module, every DataLoader worker would produce
    IDENTICAL crops/flips (since PyTorch's default worker_init seeds
    torch and random per-worker, but not numpy), silently halving
    effective training diversity whenever num_workers > 0."""
    import inspect

    from src import dataset as dataset_module

    source = inspect.getsource(dataset_module.GamusHeightDataset._augment)
    assert "np.random" not in source, (
        "augmentation must use Python's random module (seeded per-worker "
        "by PyTorch) rather than numpy.random (NOT reseeded per-worker)"
    )


def test_no_duplicate_crops_across_simulated_workers(tmp_path):
    """Simulates two DataLoader workers by seeding torch/random
    differently (as PyTorch's worker_init_fn does) and checking they
    produce different crops -- this is the actual DataLoader-worker
    trap, exercised without needing num_workers>0 (which is unreliable
    inside a test process on some platforms)."""
    import random as random_module

    root = str(tmp_path / "gamus")
    tile = make_synthetic_tile(root, "DC_1_1", size=256, nodata_fraction=0.0, seed=8)
    split_file = tmp_path / "split.txt"
    write_split_file(str(split_file), [tile])

    ds = GamusHeightDataset(
        str(split_file), data_root=root, patch_size=64, train=True, min_tile_valid_fraction=0.0
    )

    random_module.seed(100)
    torch.manual_seed(100)
    out_worker0 = ds[0]["height"].numpy().copy()

    random_module.seed(101)
    torch.manual_seed(101)
    out_worker1 = ds[0]["height"].numpy().copy()

    assert not np.array_equal(out_worker0, out_worker1), (
        "different per-worker random seeds should (almost always) yield "
        "different crops; identical output suggests augmentation is not "
        "actually consuming the random module's state"
    )
