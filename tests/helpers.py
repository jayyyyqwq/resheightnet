"""Shared synthetic-data helpers for tests. No real GAMUS data required."""
from __future__ import annotations

import os

import h5py
import numpy as np


def write_h5(path: str, array: np.ndarray, key: str = "image") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset(key, data=array)


def make_synthetic_tile(
    root: str,
    sample_id: str,
    split_dir: str = "train",
    size: int = 128,
    channel_first_rgb: bool = False,
    cls_dtype: str = "uint8",
    nodata_fraction: float = 0.02,
    seed: int = 0,
    key: str = "image",
) -> dict:
    """Write a synthetic RGB/AGL/CLS triplet and return their relpaths
    (HF-style, forward-slash, relative to `root`) plus the ground-truth
    arrays actually written, so a test can assert against them.

    RGB and height are generated from a SHARED blockwise-random field
    (RGB brightness correlates with height, plus per-pixel texture
    noise), not independently. This mirrors why a CNN can overfit real
    GAMUS tiles quickly: real RGB imagery carries local texture cues
    that actually predict height (buildings/roads/trees look visually
    distinct). A position-only RGB encoding (e.g. a plain row/col ramp)
    with an RGB-independent height target has no learnable function
    for a translation-equivariant CNN to exploit -- it would need to
    memorize an arbitrary position-to-value lookup, which is a much
    harder and unrepresentative optimization problem. Dedicated
    crop/flip/rotate alignment tests in test_dataset.py use their own
    purpose-built ramp fixture (_make_ramp_arrays) instead of this one.
    """
    rng = np.random.default_rng(seed)

    # Blockwise-smooth field: real GAMUS height maps have strong spatial
    # autocorrelation (a whole building shares roughly one height).
    block = max(1, size // 16)
    coarse_h = (size + block - 1) // block
    coarse_height = rng.uniform(0.0, 30.0, size=(coarse_h, coarse_h)).astype(np.float32)
    height = np.kron(coarse_height, np.ones((block, block), dtype=np.float32))[:size, :size]
    height = height + rng.normal(0.0, 0.5, size=(size, size)).astype(np.float32)
    height = np.clip(height, 0.0, None)

    # RGB brightness is a (noisy, nonlinear) function of the SAME coarse
    # height field, giving the model a genuine local-texture cue to
    # learn from, plus per-pixel noise so it isn't a trivial constant.
    coarse_rgb_base = np.clip(coarse_height / 30.0 * 200.0 + 20.0, 0, 255).astype(np.float32)
    rgb_base = np.kron(coarse_rgb_base, np.ones((block, block), dtype=np.float32))[:size, :size]
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    for c in range(3):
        channel_noise = rng.normal(0.0, 8.0, size=(size, size)).astype(np.float32)
        rgb[:, :, c] = np.clip(rgb_base + channel_noise, 0, 255).astype(np.uint8)
    n_nodata = int(size * size * nodata_fraction)
    if n_nodata > 0:
        flat_idx = rng.choice(size * size, size=n_nodata, replace=False)
        height_flat = height.reshape(-1)
        height_flat[flat_idx] = -5.0
        height = height_flat.reshape(size, size)

    cls = rng.integers(0, 7, size=(size, size))
    if cls_dtype == "float32":
        cls = cls.astype(np.float32)
    else:
        cls = cls.astype(np.uint8)

    rgb_write = np.transpose(rgb, (2, 0, 1)) if channel_first_rgb else rgb

    rgb_rel = f"images/{split_dir}/{sample_id}_RGB.h5"
    height_rel = f"heights/{split_dir}/{sample_id}_AGL.h5"
    cls_rel = f"classes/{split_dir}/{sample_id}_CLS.h5"

    write_h5(os.path.join(root, *rgb_rel.split("/")), rgb_write, key=key)
    write_h5(os.path.join(root, *height_rel.split("/")), height, key=key)
    write_h5(os.path.join(root, *cls_rel.split("/")), cls, key=key)

    return {
        "sample_id": sample_id,
        "rgb_relpath": rgb_rel,
        "height_relpath": height_rel,
        "cls_relpath": cls_rel,
        "rgb": rgb,
        "height": height,
        "cls": cls,
    }


def write_split_file(path: str, rows: list[dict], hf_folder: str = "train") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(
                "\t".join(
                    [
                        row["sample_id"],
                        hf_folder,
                        row["rgb_relpath"],
                        row["height_relpath"],
                        row["cls_relpath"],
                    ]
                )
                + "\n"
            )
