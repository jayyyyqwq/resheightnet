"""Evaluation entry point for ResHeightNet, matching DepthWizard's M1
protocol exactly:

  - 200 val tiles, 512x512 CENTER crop at top=left=(1024-512)//2=256
  - valid mask = isfinite(h) & (h>=0) & (0<=cls<=6)
  - metrics pooled per-pixel across ALL tiles (not per-tile-averaged)
  - predictions NOT clamped to >= 0

Arrays are preallocated (not accumulated via list-append + concatenate,
which would peak at ~2x memory) so peak memory for the full 200-tile
val set (52,428,800 pixels) stays around ~472 MB, well under this
machine's ~3.2 GB free.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import GamusHeightDataset
from src.metrics import compute_height_metrics, constant_predictor_metrics
from src.model import ResHeightNet
from src.utils import load_checkpoint, set_seed

DEFAULT_VAL_SPLIT = "data/splits/stage_a1_val.txt"


def evaluate_split(
    split_path: str,
    data_root: str | None = None,
    checkpoint: str | None = None,
    limit: int | None = None,
    patch_size: int = 512,
    batch_size: int = 4,
    device: str | None = None,
    constant_baseline: bool = False,
) -> dict:
    """Run the val-parity evaluation protocol and return a metrics dict.

    If checkpoint is None, a freshly-initialized (ImageNet-pretrained
    encoder, random decoder) model is evaluated -- useful for smoke
    testing the harness itself, not for a real result.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds = GamusHeightDataset(
        split_path, data_root=data_root, patch_size=patch_size, train=False, limit=limit
    )
    n_tiles = len(ds)
    if n_tiles == 0:
        raise ValueError(f"No tiles loaded from {split_path} (limit={limit})")

    n_pixels = n_tiles * patch_size * patch_size
    pred_buf = np.empty(n_pixels, dtype=np.float32)
    tgt_buf = np.empty(n_pixels, dtype=np.float32)
    mask_buf = np.empty(n_pixels, dtype=np.uint8)
    cls_buf = np.empty(n_pixels, dtype=np.int64)

    if constant_baseline:
        offset = 0
        for i in range(n_tiles):
            sample = ds[i]
            height = sample["height"].numpy().reshape(-1)
            mask = sample["mask"].numpy().reshape(-1)
            cls = sample["cls"].numpy().reshape(-1)
            n = height.size
            tgt_buf[offset : offset + n] = height
            mask_buf[offset : offset + n] = mask.astype(np.uint8)
            cls_buf[offset : offset + n] = cls
            offset += n
        result = constant_predictor_metrics(tgt_buf[:offset], mask_buf[:offset])
        result["n_tiles"] = n_tiles
        result["protocol"] = "512_center_crop_pooled"
        return result

    model = ResHeightNet(pretrained=checkpoint is None, freeze_stem=False).to(device)
    if checkpoint:
        load_checkpoint(checkpoint, model=model, device=device)
    model.eval()

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    offset = 0
    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            pred = model(rgb).cpu().numpy()  # (B, 1, H, W)
            height = batch["height"].numpy()
            mask = batch["mask"].numpy()
            cls = batch["cls"].numpy()

            b = pred.shape[0]
            n = b * patch_size * patch_size
            pred_buf[offset : offset + n] = pred.reshape(-1)
            tgt_buf[offset : offset + n] = height.reshape(-1)
            mask_buf[offset : offset + n] = mask.reshape(-1).astype(np.uint8)
            cls_buf[offset : offset + n] = cls.reshape(-1)
            offset += n

    result = compute_height_metrics(
        pred_buf[:offset], tgt_buf[:offset], mask_buf[:offset], cls_buf[:offset]
    )
    result["n_tiles"] = n_tiles
    result["protocol"] = "512_center_crop_pooled"

    valid_mask_count = int((mask_buf[:offset] > 0).sum())
    if valid_mask_count > 0:
        coverage = result["n_valid"] / valid_mask_count
        result["pred_finite_coverage"] = coverage
        if coverage < 0.99:
            print(
                f"[WARNING] only {coverage:.4%} of mask-valid pixels had a "
                f"finite prediction -- the reported metrics are computed on "
                f"a subset smaller than the intended mask. Investigate NaN "
                f"predictions before trusting this number."
            )
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate ResHeightNet")
    parser.add_argument("--split", default=DEFAULT_VAL_SPLIT)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--constant-baseline", action="store_true")
    parser.add_argument("--out", default=None, help="Optional path to write metrics as JSON")
    args = parser.parse_args()

    set_seed(args.seed)
    result = evaluate_split(
        args.split,
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        limit=args.limit,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        constant_baseline=args.constant_baseline,
    )

    print(json.dumps({k: v for k, v in result.items() if k not in ("class_mae", "bucket_mae")}, indent=2))
    print("class_mae:", json.dumps(result.get("class_mae", {}), indent=2))
    print("bucket_mae:", json.dumps(result.get("bucket_mae", {}), indent=2))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote metrics to {args.out}")


if __name__ == "__main__":
    main()
