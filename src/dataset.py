"""GAMUS dataset loader for ResHeightNet.

Reads the DepthWizard-format split manifests (TAB-separated, 5 columns,
no header) and HDF5 triplets (RGB uint8, AGL float32 metres, CLS
int/float class map). Mirrors DepthWizard's dataset invariants exactly
so evaluation numbers are comparable:

  - Height is NEVER modified. Validity is carried as a separate boolean
    mask; every consumer (loss, metric) re-checks it.
  - valid = isfinite(h) & (h >= 0.0) & (cls >= 0) & (cls <= 6)
  - GAMUS nodata is encoded as a negative height (~ -5 m), not NaN/-9999.

See PROJECT_PLAN.md and the fail-proof execution plan for the full
rationale behind each design choice below.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import IMAGENET_MEAN, IMAGENET_STD, resolve_data_root

TILE_SIZE = 1024
NUM_CLASSES = 7  # classes 0..6 inclusive are valid semantic labels
MIN_TILE_VALID_FRACTION = 0.05  # whole-tile filter applied at index-build time
MIN_CROP_VALID_FRACTION = 0.10  # per-crop resample threshold
MAX_CROP_RESAMPLES = 10


@dataclass(frozen=True)
class SplitRow:
    sample_id: str
    hf_folder: str
    rgb_relpath: str
    height_relpath: str
    cls_relpath: str


def parse_split_file(split_path: str) -> list[SplitRow]:
    """Parse a TAB-separated, 5-column, headerless GAMUS split manifest.

    Column 2 (hf_folder) is the upstream Hugging Face repo folder and is
    INDEPENDENT of the experiment split -- a *_val.txt file legitimately
    contains rows whose hf_folder is "train". Never use column 2 as the
    experiment split; split membership is which FILE the row came from,
    not this column.
    """
    rows: list[SplitRow] = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 5:
                raise ValueError(
                    f"{split_path}:{line_num}: expected 5 tab-separated fields, "
                    f"got {len(parts)}: {line!r}"
                )
            rows.append(SplitRow(*parts))
    return rows


def _relpath_to_fs(root: str, relpath: str) -> str:
    """Join a forward-slash HF-style relpath onto a local root, OS-correctly."""
    return os.path.join(root, *relpath.split("/"))


def _read_h5_array(path: str) -> np.ndarray:
    """Read the sole dataset out of an HDF5 file, key-agnostic.

    GAMUS files use the key 'image' for all three modalities, but we
    read defensively via list(f.keys())[0] to match DepthWizard's more
    robust loader rather than hardcoding the key.
    """
    with h5py.File(path, "r") as h5f:
        keys = list(h5f.keys())
        if not keys:
            raise ValueError(f"Empty H5 file: {path}")
        return np.asarray(h5f[keys[0]])


def _read_h5_crop(path: str, top: int, left: int, size: int) -> np.ndarray:
    """Read only a spatial crop out of an HDF5 file via hyperslab slicing.

    GAMUS files are uncompressed, so this reads ~1/4 of the bytes a full
    read would (512^2 vs 1024^2), which matters most on Colab where I/O
    is the training bottleneck. Not currently wired into the Dataset
    (full-tile read + in-memory crop is used instead, since augmentation
    resampling needs the whole tile); kept as an optimization hook for
    scripts/cache_val_crops.py style precomputation.
    """
    with h5py.File(path, "r") as h5f:
        keys = list(h5f.keys())
        if not keys:
            raise ValueError(f"Empty H5 file: {path}")
        dataset = h5f[keys[0]]
        if dataset.ndim == 2:
            return np.asarray(dataset[top : top + size, left : left + size])
        if dataset.ndim == 3 and dataset.shape[0] in (1, 3):
            return np.asarray(dataset[:, top : top + size, left : left + size])
        if dataset.ndim == 3:
            return np.asarray(dataset[top : top + size, left : left + size, :])
        raise ValueError(f"Unsupported H5 array shape {dataset.shape}: {path}")


def _normalize_rgb_layout(rgb: np.ndarray) -> np.ndarray:
    """Standardize RGB to (H, W, 3) uint8, handling channel-first input."""
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB shape (*, *, 3) after layout fix, got {rgb.shape}")
    return rgb


def compute_valid_mask(height: np.ndarray, cls: np.ndarray) -> np.ndarray:
    """valid = isfinite(h) & (h >= 0) & (0 <= cls <= 6).

    cls is cast to int64 BEFORE the range comparison: GAMUS stores cls as
    float32 for DC tiles and uint8 for PHL/NYC tiles. Casting first
    guarantees bit-identical masks across both source dtypes.

    Height is read but never written to -- nodata pixels stay in the
    array at their raw (negative) value; only the mask excludes them.
    """
    cls_i = cls.astype(np.int64)
    return np.isfinite(height) & (height >= 0.0) & (cls_i >= 0) & (cls_i <= NUM_CLASSES - 1)


def _rot90_all(rgb, height, cls, mask, k):
    rgb = np.ascontiguousarray(np.rot90(rgb, k, axes=(0, 1)))
    height = np.ascontiguousarray(np.rot90(height, k, axes=(0, 1)))
    cls = np.ascontiguousarray(np.rot90(cls, k, axes=(0, 1)))
    mask = np.ascontiguousarray(np.rot90(mask, k, axes=(0, 1)))
    return rgb, height, cls, mask


def _hflip_all(rgb, height, cls, mask):
    rgb = np.ascontiguousarray(rgb[:, ::-1, :])
    height = np.ascontiguousarray(height[:, ::-1])
    cls = np.ascontiguousarray(cls[:, ::-1])
    mask = np.ascontiguousarray(mask[:, ::-1])
    return rgb, height, cls, mask


def _vflip_all(rgb, height, cls, mask):
    rgb = np.ascontiguousarray(rgb[::-1, :, :])
    height = np.ascontiguousarray(height[::-1, :])
    cls = np.ascontiguousarray(cls[::-1, :])
    mask = np.ascontiguousarray(mask[::-1, :])
    return rgb, height, cls, mask


class GamusHeightDataset(Dataset):
    """GAMUS RGB -> height regression dataset.

    Reuses DepthWizard's deterministic stage_a1 splits and masking
    invariant exactly, so metrics computed on this dataset are
    comparable to DepthWizard's reported numbers.

    Args:
        split_path: path to a TAB-separated 5-column split manifest.
        data_root: filesystem root containing images/, heights/,
            classes/. If None, resolved via resolve_data_root().
        patch_size: crop size in pixels (512 to match DepthWizard).
        train: if True, random crop + augmentation. If False
            (validation/test), deterministic center crop, no
            augmentation.
        limit: optionally truncate the manifest to the first N rows
            (used for the local 20-tile sanity subset).
        min_tile_valid_fraction: whole-tile valid-pixel fraction below
            which a TRAIN tile is dropped entirely at index-build time.
            Never applied in eval mode -- val must retain every tile for
            parity with DepthWizard's 200-tile protocol.
    """

    def __init__(
        self,
        split_path: str,
        data_root: str | None = None,
        patch_size: int = 512,
        train: bool = True,
        limit: int | None = None,
        min_tile_valid_fraction: float = MIN_TILE_VALID_FRACTION,
    ):
        self.data_root = data_root or resolve_data_root()
        self.patch_size = patch_size
        self.train = train
        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        self.std = np.array(IMAGENET_STD, dtype=np.float32)

        rows = parse_split_file(split_path)
        if limit is not None:
            rows = rows[:limit]

        self.dropped_tiles: list[str] = []
        if train:
            kept = []
            for row in rows:
                height_path = _relpath_to_fs(self.data_root, row.height_relpath)
                cls_path = _relpath_to_fs(self.data_root, row.cls_relpath)
                height = _read_h5_array(height_path).astype(np.float32)
                cls = _read_h5_array(cls_path)
                mask = compute_valid_mask(height, cls)
                if mask.mean() < min_tile_valid_fraction:
                    self.dropped_tiles.append(row.sample_id)
                    continue
                kept.append(row)
            self.rows = kept
        else:
            # Validation/test: never drop tiles. Count must stay exactly
            # what the split file specifies for parity with DepthWizard.
            self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def _load_tile(self, row: SplitRow):
        rgb_path = _relpath_to_fs(self.data_root, row.rgb_relpath)
        height_path = _relpath_to_fs(self.data_root, row.height_relpath)
        cls_path = _relpath_to_fs(self.data_root, row.cls_relpath)

        rgb = _normalize_rgb_layout(_read_h5_array(rgb_path))
        height = _read_h5_array(height_path).astype(np.float32)
        cls = _read_h5_array(cls_path)
        return rgb, height, cls

    def _crop(self, rgb, height, cls):
        h, w = height.shape
        p = self.patch_size

        if not self.train:
            # Deterministic center crop. For 1024 tiles and patch_size=512
            # this is exactly top = left = 256, matching DepthWizard's
            # evaluation protocol.
            top = (h - p) // 2
            left = (w - p) // 2
            rgb_c = rgb[top : top + p, left : left + p, :]
            height_c = height[top : top + p, left : left + p]
            cls_c = cls[top : top + p, left : left + p]
            mask_c = compute_valid_mask(height_c, cls_c)
            return rgb_c, height_c, cls_c, mask_c

        # Train: random crop, resampled up to MAX_CROP_RESAMPLES times to
        # avoid landing on an (almost) all-nodata region. Falls back to
        # the single best-seen crop, then center crop, if no resample
        # clears the threshold -- this keeps __getitem__ total, never
        # raising.
        best = None
        for _ in range(MAX_CROP_RESAMPLES):
            top = random.randint(0, h - p)
            left = random.randint(0, w - p)
            rgb_c = rgb[top : top + p, left : left + p, :]
            height_c = height[top : top + p, left : left + p]
            cls_c = cls[top : top + p, left : left + p]
            mask_c = compute_valid_mask(height_c, cls_c)
            frac = float(mask_c.mean())
            if best is None or frac > best[1]:
                best = ((rgb_c, height_c, cls_c, mask_c), frac)
            if frac >= MIN_CROP_VALID_FRACTION:
                return rgb_c, height_c, cls_c, mask_c

        if best is not None:
            return best[0]
        top = (h - p) // 2
        left = (w - p) // 2
        rgb_c = rgb[top : top + p, left : left + p, :]
        height_c = height[top : top + p, left : left + p]
        cls_c = cls[top : top + p, left : left + p]
        mask_c = compute_valid_mask(height_c, cls_c)
        return rgb_c, height_c, cls_c, mask_c

    def _augment(self, rgb, height, cls, mask):
        """Apply the SAME flip/rotation to all four arrays.

        Uses Python's `random` module, not numpy, so that PyTorch's
        per-worker seeding (which covers torch/random but NOT numpy)
        actually diversifies augmentation across DataLoader workers.
        """
        if random.random() > 0.5:
            rgb, height, cls, mask = _hflip_all(rgb, height, cls, mask)
        if random.random() > 0.5:
            rgb, height, cls, mask = _vflip_all(rgb, height, cls, mask)
        k = random.choice([0, 1, 2, 3])
        if k != 0:
            rgb, height, cls, mask = _rot90_all(rgb, height, cls, mask, k)
        return rgb, height, cls, mask

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        rgb, height, cls = self._load_tile(row)
        rgb_c, height_c, cls_c, mask_c = self._crop(rgb, height, cls)

        if self.train:
            rgb_c, height_c, cls_c, mask_c = self._augment(rgb_c, height_c, cls_c, mask_c)

        # Normalize AFTER cropping/augmentation, manually (never
        # transforms.ToTensor(), which silently skips the /255 step on
        # already-float input): /255 first, then ImageNet mean/std.
        rgb_f = rgb_c.astype(np.float32) / 255.0
        rgb_f = (rgb_f - self.mean) / self.std
        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb_f.transpose(2, 0, 1)))

        height_t = torch.from_numpy(np.ascontiguousarray(height_c)).unsqueeze(0)
        mask_t = torch.from_numpy(np.ascontiguousarray(mask_c)).unsqueeze(0)
        cls_t = torch.from_numpy(np.ascontiguousarray(cls_c.astype(np.int64)))

        return {
            "rgb": rgb_t,
            "height": height_t,
            "mask": mask_t,
            "cls": cls_t,
            "sample_id": row.sample_id,
        }
