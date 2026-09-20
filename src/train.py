"""Training entry point for ResHeightNet.

Two modes:
  1. Overfit gate (M6): `--overfit N --steps S` trains on N fixed tiles
     with a fixed center crop, augmentation OFF, AMP OFF, and prints the
     loss trajectory plus the L0 baseline and the two controls
     (constant-image, 1-tile) needed to make the gate falsifiable. This
     is the fast, local, CPU-only check that must pass before any Colab
     time is spent.
  2. Full training (M7): standard train loop, matching DepthWizard's
     recipe (AdamW lr=1e-4 wd=1e-4, CosineAnnealingLR, 15 epochs,
     seed=42, batch=8 @ 512x512). Intended to run on Colab's T4; will
     run on CPU here too, just far slower -- CPU is fine for the
     overfit gate, not for a real 1000-tile run.

Checkpoints are written via src.utils.save_checkpoint (plain-dict
config only, weights_only=True-safe).
"""
from __future__ import annotations

import argparse
import copy
import os
import time

import torch
from torch.utils.data import DataLoader

from src.dataset import GamusHeightDataset
from src.losses import masked_l1_loss
from src.metrics import compute_height_metrics
from src.model import ResHeightNet
from src.utils import load_checkpoint, save_checkpoint, set_seed

DEFAULT_TRAIN_SPLIT = "data/splits/stage_a1_train.txt"
DEFAULT_VAL_SPLIT = "data/splits/stage_a1_val.txt"


def masked_mean_abs(height: torch.Tensor, mask: torch.Tensor) -> float:
    """mean(|h|) over valid pixels -- the loss a zero-predicting model
    achieves. Used as the L0 baseline for the overfit gate."""
    mask_f = mask.float()
    denom = mask_f.sum().clamp(min=1.0)
    return float(((height.abs()) * mask_f).sum() / denom)


def run_overfit_gate(
    split_path: str,
    data_root: str | None,
    n_tiles: int,
    steps: int,
    patch_size: int = 512,
    lr: float = 3e-3,
    seed: int = 42,
    device: str | None = None,
    constant_image_control: bool = False,
    log_every: int = 10,
) -> dict:
    """M6 overfit gate. Fixed center crop, augmentation OFF, AMP OFF,
    fp32.

    NOTE on lr: this default (3e-3) is deliberately 10x train_full's
    production lr (1e-4, matching DepthWizard's recipe). The overfit
    gate's purpose is fast convergence on a handful of tiles with a
    freshly-initialized decoder, not reproducing the production
    training dynamics -- empirically, 3e-4 left loss barely moving
    after 150 steps on a synthetic fixture with a genuinely learnable
    RGB->height relationship (see tests/test_integration.py), while
    3e-3 converged in well under 300. Using the production lr here
    would risk the gate reporting a false FAIL on a correctly-wired
    pipeline simply because it hadn't converged yet.

    Returns a dict with the loss trajectory and L0 so the caller
    (or a test) can apply the pass/fail thresholds:
        final < 0.30 m  AND  final / L0 < 0.25
    """
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds = GamusHeightDataset(
        split_path, data_root=data_root, patch_size=patch_size, train=False, limit=n_tiles
    )
    if len(ds) == 0:
        raise ValueError(f"No tiles loaded from {split_path} (limit={n_tiles})")

    # Materialize fixed batches once (center crop, no augmentation).
    samples = [ds[i] for i in range(len(ds))]
    rgb = torch.stack([s["rgb"] for s in samples]).to(device)
    height = torch.stack([s["height"] for s in samples]).to(device)
    mask = torch.stack([s["mask"] for s in samples]).to(device)

    if constant_image_control:
        # Replace every RGB tile with the same gray image; targets
        # unchanged. If the model can still drive loss down, something
        # other than the RGB content is identifying the sample (e.g. a
        # batch-order / index leak).
        gray = torch.zeros_like(rgb[0:1])
        rgb = gray.expand_as(rgb).clone()

    l0 = masked_mean_abs(height, mask)

    model = ResHeightNet(pretrained=True, freeze_stem=False).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        pred = model(rgb)
        loss = masked_l1_loss(pred, height, mask)
        loss.backward()
        optimizer.step()
        loss_val = float(loss.item())
        losses.append(loss_val)
        if step % log_every == 0 or step == steps - 1:
            print(f"step {step:04d}  loss={loss_val:.4f} m  (L0={l0:.4f} m)")

    final = losses[-1]
    ratio = final / l0 if l0 > 0 else float("inf")
    return {
        "losses": losses,
        "l0": l0,
        "final": final,
        "final_over_l0": ratio,
        "n_tiles": len(ds),
    }


def build_dataloaders(
    train_split: str,
    val_split: str,
    data_root: str | None,
    patch_size: int,
    batch_size: int,
    num_workers: int,
):
    train_ds = GamusHeightDataset(
        train_split, data_root=data_root, patch_size=patch_size, train=True
    )
    val_ds = GamusHeightDataset(val_split, data_root=data_root, patch_size=patch_size, train=False)

    if train_ds.dropped_tiles:
        print(
            f"[data] dropped {len(train_ds.dropped_tiles)} train tiles "
            f"below min valid fraction: {train_ds.dropped_tiles[:10]}"
            f"{'...' if len(train_ds.dropped_tiles) > 10 else ''}"
        )

    if len(train_ds) == 0:
        # DataLoader(shuffle=True) on an empty dataset fails deep inside
        # torch's RandomSampler with a cryptic "num_samples should be a
        # positive integer" error. Fail here instead, with a message
        # that actually explains what happened (every train tile fell
        # below min_tile_valid_fraction) rather than requiring the
        # caller to trace a torch internal.
        raise ValueError(
            f"All {len(train_ds.dropped_tiles)} train tiles were dropped for having "
            f"too few valid pixels (below GamusHeightDataset's min_tile_valid_fraction). "
            f"Dropped: {train_ds.dropped_tiles}. Check the data root and split file."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, batch_size // 2),
        shuffle=False,
        num_workers=max(0, num_workers // 2),
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def train_full(
    train_split: str = DEFAULT_TRAIN_SPLIT,
    val_split: str = DEFAULT_VAL_SPLIT,
    data_root: str | None = None,
    epochs: int = 15,
    batch_size: int = 8,
    patch_size: int = 512,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
    num_workers: int = 0,
    checkpoint_dir: str = "results/checkpoints",
    amp: bool = True,
    resume_from: str | None = None,
) -> None:
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = amp and device == "cuda"

    os.makedirs(checkpoint_dir, exist_ok=True)

    train_loader, val_loader = build_dataloaders(
        train_split, val_split, data_root, patch_size, batch_size, num_workers
    )

    model = ResHeightNet(pretrained=True, freeze_stem=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * steps_per_epoch, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    best_val = float("inf")
    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "patch_size": patch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "seed": seed,
    }

    if resume_from and os.path.exists(resume_from):
        state = load_checkpoint(
            resume_from, model=model, optimizer=optimizer, scaler=scaler, device=device,
            restore_rng=True,
        )
        start_epoch = state["epoch"] + 1
        best_val = state["best_val"]
        print(f"[resume] resuming from epoch {start_epoch}, best_val={best_val:.4f}")

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            height = batch["height"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                pred = model(rgb)
            # Loss reduction deliberately OUTSIDE autocast and always in
            # fp32: the masked abs-diff sum over a 512x512 crop can reach
            # ~786,000, well past fp16's 65,504 max. See src/losses.py.
            loss = masked_l1_loss(pred, height, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += float(loss.item())
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)

        model.eval()
        val_running = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch["rgb"].to(device, non_blocking=True)
                height = batch["height"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                pred = model(rgb)
                loss = masked_l1_loss(pred, height, mask)
                val_running += float(loss.item())
                val_batches += 1
        val_loss = val_running / max(1, val_batches)

        elapsed = time.time() - t0
        print(
            f"epoch {epoch + 1:02d}/{epochs}  train_L1={train_loss:.4f} m  "
            f"val_L1={val_loss:.4f} m  ({elapsed:.1f}s)"
        )

        last_path = os.path.join(checkpoint_dir, "last.pt")
        save_checkpoint(
            last_path, epoch=epoch, model=model, optimizer=optimizer, scaler=scaler,
            best_val=min(best_val, val_loss), config=config,
        )
        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(checkpoint_dir, "best.pt")
            save_checkpoint(
                best_path, epoch=epoch, model=model, optimizer=optimizer, scaler=scaler,
                best_val=best_val, config=config,
            )
            print(f"  [checkpoint] new best val_L1={best_val:.4f} m -> {best_path}")


def main():
    parser = argparse.ArgumentParser(description="Train ResHeightNet")
    parser.add_argument("--train-split", default=DEFAULT_TRAIN_SPLIT)
    parser.add_argument("--val-split", default=DEFAULT_VAL_SPLIT)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="results/checkpoints")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume-from", default=None)

    # Overfit gate mode
    parser.add_argument("--overfit", type=int, default=None, help="Run M6 overfit gate on N tiles")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--constant-image-control", action="store_true")

    args = parser.parse_args()

    if args.overfit is not None:
        result = run_overfit_gate(
            args.val_split,
            args.data_root,
            n_tiles=args.overfit,
            steps=args.steps,
            patch_size=args.patch_size,
            seed=args.seed,
            constant_image_control=args.constant_image_control,
        )
        print(
            f"\n[overfit gate] n_tiles={result['n_tiles']} L0={result['l0']:.4f} m "
            f"final={result['final']:.4f} m  final/L0={result['final_over_l0']:.4f}"
        )
        passed = result["final"] < 0.30 and result["final_over_l0"] < 0.25
        print(f"[overfit gate] {'PASS' if passed else 'FAIL'}")
        return

    train_full(
        train_split=args.train_split,
        val_split=args.val_split,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        amp=not args.no_amp,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
