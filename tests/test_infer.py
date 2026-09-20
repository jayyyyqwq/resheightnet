"""Tests for src/infer.py: reads .h5 (not .tif -- the PROJECT_PLAN's
original rasterio-based skeleton was wrong about the file format), uses
a fixed 0-40m colour scale, and resolves sample_id -> file path across
the images/{train,val,test}/ layout including the NYC _IMG.h5 naming."""
from __future__ import annotations

import os

import numpy as np
import torch

from src.infer import (
    load_rgb_h5,
    preprocess,
    resolve_sample_path,
    run_inference,
    save_height_map,
)
from src.utils import IMAGENET_MEAN, IMAGENET_STD
from tests.helpers import write_h5


def test_load_rgb_h5_normalizes_channel_first(tmp_path):
    path = str(tmp_path / "a.h5")
    chw = np.zeros((3, 8, 8), dtype=np.uint8)
    chw[:, 1, 2] = [9, 8, 7]
    write_h5(path, chw)

    rgb = load_rgb_h5(path)
    assert rgb.shape == (8, 8, 3)
    assert np.array_equal(rgb[1, 2], [9, 8, 7])


def test_preprocess_matches_manual_normalization():
    rgb = np.full((16, 16, 3), 200, dtype=np.uint8)
    tensor = preprocess(rgb)
    assert tensor.shape == (1, 3, 16, 16)

    expected = (200.0 / 255.0 - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    for c in range(3):
        assert torch.allclose(
            tensor[0, c], torch.full((16, 16), float(expected[c])), atol=1e-5
        )


def test_run_inference_no_checkpoint_produces_finite_output(tmp_path):
    path = str(tmp_path / "tile_RGB.h5")
    write_h5(path, np.random.default_rng(0).integers(0, 255, size=(32, 32, 3)).astype(np.uint8))

    pred = run_inference(path, checkpoint_path=None, device="cpu")
    assert pred.shape == (32, 32)
    assert np.isfinite(pred).all()


def test_save_height_map_writes_a_file(tmp_path):
    pred = np.random.default_rng(0).uniform(0, 40, size=(16, 16)).astype(np.float32)
    out_path = str(tmp_path / "out" / "demo.png")
    save_height_map(pred, out_path)
    assert os.path.isfile(out_path)
    assert os.path.getsize(out_path) > 0


def test_resolve_sample_path_finds_rgb_file(tmp_path):
    root = str(tmp_path / "gamus")
    write_h5(os.path.join(root, "images", "val", "DC_1_1_RGB.h5"), np.zeros((4, 4, 3), dtype=np.uint8))

    resolved = resolve_sample_path("DC_1_1", data_root=root)
    assert resolved == os.path.join(root, "images", "val", "DC_1_1_RGB.h5")


def test_resolve_sample_path_finds_nyc_img_suffix(tmp_path):
    root = str(tmp_path / "gamus")
    write_h5(
        os.path.join(root, "images", "test", "NYC_00735_IMG.h5"),
        np.zeros((4, 4, 3), dtype=np.uint8),
    )
    resolved = resolve_sample_path("NYC_00735", data_root=root)
    assert resolved == os.path.join(root, "images", "test", "NYC_00735_IMG.h5")


def test_resolve_sample_path_raises_when_missing(tmp_path):
    root = str(tmp_path / "gamus")
    os.makedirs(os.path.join(root, "images", "train"), exist_ok=True)
    try:
        resolve_sample_path("NOPE_0_0", data_root=root)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
