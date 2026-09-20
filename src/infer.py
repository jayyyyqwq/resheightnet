"""Inference / demo script for ResHeightNet.

Reads a GAMUS HDF5 RGB tile directly (NOT .tif -- GAMUS data is HDF5;
see PROJECT_PLAN.md's original rasterio-based skeleton, which was wrong
about the file format). Renders the predicted height map with a FIXED
0-40 m colour scale and a metre-labelled colorbar -- never min-max
normalized per-image, which would discard the absolute scale that is
the entire point of a metric-height model.
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset import _normalize_rgb_layout, _read_h5_array
from src.model import ResHeightNet
from src.utils import IMAGENET_MEAN, IMAGENET_STD, load_checkpoint, resolve_data_root

FIXED_VMIN = 0.0
FIXED_VMAX = 40.0


def load_rgb_h5(path: str) -> np.ndarray:
    """Read + layout-normalize a GAMUS RGB .h5 tile to (H, W, 3) uint8."""
    return _normalize_rgb_layout(_read_h5_array(path))


def preprocess(rgb: np.ndarray) -> torch.Tensor:
    """/255 then ImageNet normalize -- identical order to dataset.py,
    NOT transforms.ToTensor() (which silently skips /255 on float input)."""
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    rgb_f = rgb.astype(np.float32) / 255.0
    rgb_f = (rgb_f - mean) / std
    tensor = torch.from_numpy(np.ascontiguousarray(rgb_f.transpose(2, 0, 1)))
    return tensor.unsqueeze(0)


def run_inference(image_path: str, checkpoint_path: str, device: str | None = None) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rgb = load_rgb_h5(image_path)
    tensor = preprocess(rgb).to(device)

    model = ResHeightNet(pretrained=checkpoint_path is None, freeze_stem=False).to(device)
    if checkpoint_path:
        load_checkpoint(checkpoint_path, model=model, device=device)
    model.eval()

    with torch.no_grad():
        pred = model(tensor).squeeze().cpu().numpy()
    return pred


def save_height_map(pred: np.ndarray, out_path: str) -> None:
    """Fixed 0-40 m colour scale, metre-labelled colorbar -- absolute
    scale is preserved, unlike a per-image min-max normalization."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(pred, cmap="viridis", vmin=FIXED_VMIN, vmax=FIXED_VMAX)
    ax.set_title(
        f"Predicted height (m)\nmin={pred.min():.2f}  mean={pred.mean():.2f}  max={pred.max():.2f}"
    )
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Height (m)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def resolve_sample_path(sample_id: str, data_root: str | None = None) -> str:
    """Locate a GAMUS RGB .h5 file by sample_id under the resolved data
    root, searching the standard images/{train,val,test}/ subfolders."""
    root = data_root or resolve_data_root()
    for split_dir in ("train", "val", "test"):
        candidate = os.path.join(root, "images", split_dir, f"{sample_id}_RGB.h5")
        if os.path.exists(candidate):
            return candidate
        candidate_nyc = os.path.join(root, "images", split_dir, f"{sample_id}_IMG.h5")
        if os.path.exists(candidate_nyc):
            return candidate_nyc
    raise FileNotFoundError(f"No RGB .h5 found for sample_id={sample_id!r} under {root}")


def main():
    parser = argparse.ArgumentParser(description="Run ResHeightNet inference on a GAMUS tile")
    parser.add_argument("--image", default=None, help="Path to a GAMUS RGB .h5 file")
    parser.add_argument("--sample", default=None, help="Sample id to resolve under --data-root")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", default="results/checkpoints/best.pt")
    parser.add_argument("--out", default="results/sample_outputs/out.png")
    args = parser.parse_args()

    if not args.image and not args.sample:
        parser.error("Provide either --image PATH or --sample SAMPLE_ID")

    image_path = args.image or resolve_sample_path(args.sample, args.data_root)
    checkpoint = args.checkpoint if os.path.exists(args.checkpoint) else None
    if checkpoint is None:
        print(
            f"[WARNING] checkpoint not found at {args.checkpoint!r} -- running with "
            f"randomly-initialized decoder weights. This output is NOT a trained prediction."
        )

    pred = run_inference(image_path, checkpoint)
    save_height_map(pred, args.out)
    print(f"Saved height map to {args.out}")
    print(f"min={pred.min():.2f} m  mean={pred.mean():.2f} m  max={pred.max():.2f} m")


if __name__ == "__main__":
    main()
