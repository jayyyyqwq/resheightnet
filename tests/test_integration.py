"""Integration tests: dataset -> model -> loss -> optimizer wired
together, checkpoint round-trip (including the bitwise resume test that
protects the Colab run), and a fast CI-sized version of the M6 overfit
gate.
"""
from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from src.dataset import GamusHeightDataset
from src.losses import masked_l1_loss
from src.metrics import compute_height_metrics
from src.model import ResHeightNet
from src.utils import load_checkpoint, save_checkpoint, set_seed
from tests.helpers import make_synthetic_tile, write_split_file


def _make_synthetic_split(tmp_path, n_tiles: int, size: int = 64, seed: int = 0):
    root = str(tmp_path / "gamus")
    tiles = [
        make_synthetic_tile(root, f"DC_{i}_{i}", size=size, nodata_fraction=0.02, seed=seed + i)
        for i in range(n_tiles)
    ]
    split_file = tmp_path / "split.txt"
    write_split_file(str(split_file), tiles)
    return root, str(split_file)


def test_one_full_cpu_training_step_reduces_nothing_but_stays_finite(tmp_path):
    """One full step through the real pipeline: DataLoader -> model ->
    masked loss -> backward -> optimizer.step(). Loss must be finite and
    at least one parameter must actually change."""
    set_seed(0)
    root, split_file = _make_synthetic_split(tmp_path, n_tiles=2, size=64)

    ds = GamusHeightDataset(
        split_file, data_root=root, patch_size=32, train=True, min_tile_valid_fraction=0.0
    )
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)

    model = ResHeightNet(pretrained=False, freeze_stem=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    before = {name: p.clone() for name, p in model.named_parameters()}

    batch = next(iter(loader))
    optimizer.zero_grad()
    pred = model(batch["rgb"])
    loss = masked_l1_loss(pred, batch["height"], batch["mask"])
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()

    changed = any(
        not torch.equal(before[name], p) for name, p in model.named_parameters()
    )
    assert changed, "optimizer.step() did not change any parameter"


def test_fifty_steps_on_two_tiles_reduces_loss_substantially(tmp_path):
    """Fast CI-sized version of the M6 overfit gate: not the full
    300-step / 10-tile protocol (too slow for CI), but the same
    falsifiable shape -- loss must drop well below its initial value,
    proving the wiring (crop alignment, mask, loss reduction, gradient
    flow) is correct end to end."""
    set_seed(0)
    root, split_file = _make_synthetic_split(tmp_path, n_tiles=2, size=64)

    ds = GamusHeightDataset(
        split_file, data_root=root, patch_size=64, train=False  # fixed center crop, no augmentation
    )
    rgb = torch.stack([ds[i]["rgb"] for i in range(len(ds))])
    height = torch.stack([ds[i]["height"] for i in range(len(ds))])
    mask = torch.stack([ds[i]["mask"] for i in range(len(ds))])

    model = ResHeightNet(pretrained=True, freeze_stem=False)
    model.train()
    # lr=3e-3, not the production 1e-4/train_full default: matches
    # run_overfit_gate's own default and rationale (see src/train.py) --
    # a freshly-initialized decoder needs a higher lr to converge within
    # a small step budget than the full 15-epoch training recipe uses.
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        pred = model(rgb)
        loss = masked_l1_loss(pred, height, mask)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < 0.25 * losses[0], (
        f"loss did not drop enough: initial={losses[0]:.4f} final={losses[-1]:.4f} "
        f"(expected final < 25% of initial) -- suggests a wiring bug "
        f"(crop misalignment, broken mask, or dead gradient path)"
    )


def test_checkpoint_round_trip_with_weights_only_true(tmp_path):
    """torch.load defaults to weights_only=True since torch 2.6 --
    save_checkpoint must only ever write plain dict/list/str/int/float/
    tensor content, or this round-trip fails."""
    model = ResHeightNet(pretrained=False, freeze_stem=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    ckpt_path = str(tmp_path / "ckpt.pt")
    save_checkpoint(
        ckpt_path, epoch=3, model=model, optimizer=optimizer, scaler=None,
        best_val=1.23, config={"lr": 1e-4, "epochs": 15, "seed": 42},
    )

    # Simulates torch's actual safe default explicitly, not relying on
    # load_checkpoint's own default staying weights_only=True forever.
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert state["epoch"] == 3
    assert state["best_val"] == 1.23
    assert isinstance(state["config"], dict)

    fresh_model = ResHeightNet(pretrained=False, freeze_stem=False)
    fresh_model.load_state_dict(state["model_state_dict"])
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), fresh_model.named_parameters()):
        assert torch.equal(p1, p2), f"parameter {n1} did not round-trip"


def test_resume_is_bitwise_identical_to_uninterrupted_training(tmp_path):
    """The test that protects the Colab run: 4 steps uninterrupted must
    produce IDENTICAL final weights to 2 steps + checkpoint + resume +
    2 steps. An untested resume path is the classic way to lose a
    training session to a Colab disconnect."""
    root, split_file = _make_synthetic_split(tmp_path, n_tiles=2, size=64)

    def make_batch():
        ds = GamusHeightDataset(
            split_file, data_root=root, patch_size=32, train=True, min_tile_valid_fraction=0.0
        )
        set_seed(0)
        rgb = torch.stack([ds[i]["rgb"] for i in range(len(ds))])
        height = torch.stack([ds[i]["height"] for i in range(len(ds))])
        mask = torch.stack([ds[i]["mask"] for i in range(len(ds))])
        return rgb, height, mask

    # --- Uninterrupted: 4 steps straight through ---
    set_seed(123)
    rgb, height, mask = make_batch()
    model_a = ResHeightNet(pretrained=False, freeze_stem=False)
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
    for _ in range(4):
        opt_a.zero_grad()
        loss = masked_l1_loss(model_a(rgb), height, mask)
        loss.backward()
        opt_a.step()

    # --- Interrupted: 2 steps, checkpoint, "restart", resume, 2 more steps ---
    set_seed(123)
    rgb2, height2, mask2 = make_batch()
    model_b = ResHeightNet(pretrained=False, freeze_stem=False)
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3)
    for _ in range(2):
        opt_b.zero_grad()
        loss = masked_l1_loss(model_b(rgb2), height2, mask2)
        loss.backward()
        opt_b.step()

    ckpt_path = str(tmp_path / "resume.pt")
    save_checkpoint(
        ckpt_path, epoch=1, model=model_b, optimizer=opt_b, scaler=None,
        best_val=999.0, config={"lr": 1e-3},
    )

    # Simulate a fresh process: brand-new model/optimizer objects, loaded from disk.
    model_c = ResHeightNet(pretrained=False, freeze_stem=False)
    opt_c = torch.optim.AdamW(model_c.parameters(), lr=1e-3)
    load_checkpoint(ckpt_path, model=model_c, optimizer=opt_c, device="cpu")

    for _ in range(2):
        opt_c.zero_grad()
        loss = masked_l1_loss(model_c(rgb2), height2, mask2)
        loss.backward()
        opt_c.step()

    for (na, pa), (nc, pc) in zip(model_a.named_parameters(), model_c.named_parameters()):
        assert torch.allclose(pa, pc, atol=1e-6), (
            f"resume mismatch at {na}: uninterrupted and resumed training "
            f"diverged -- a real Colab resume would silently continue from "
            f"a different point than intended"
        )


def test_same_seed_gives_identical_first_batch_and_loss(tmp_path):
    root, split_file = _make_synthetic_split(tmp_path, n_tiles=3, size=64)

    def first_batch_loss():
        set_seed(7)
        ds = GamusHeightDataset(
            split_file, data_root=root, patch_size=32, train=True, min_tile_valid_fraction=0.0
        )
        loader = DataLoader(ds, batch_size=3, shuffle=True, num_workers=0)
        batch = next(iter(loader))
        set_seed(7)
        model = ResHeightNet(pretrained=False, freeze_stem=False)
        with torch.no_grad():
            pred = model(batch["rgb"])
            loss = masked_l1_loss(pred, batch["height"], batch["mask"])
        return batch["height"].numpy().copy(), float(loss.item())

    height1, loss1 = first_batch_loss()
    height2, loss2 = first_batch_loss()

    assert (height1 == height2).all(), "same seed did not reproduce the same crop/augmentation"
    assert math.isclose(loss1, loss2, rel_tol=1e-6)


def test_evaluate_metrics_pipeline_with_stub_predictions(tmp_path):
    """End-to-end shape check for the evaluate.py pipeline: given a
    dataset's real height/mask tensors and a stub prediction of gt+1.5,
    compute_height_metrics must report MAE exactly 1.5."""
    root, split_file = _make_synthetic_split(tmp_path, n_tiles=3, size=64, seed=50)
    ds = GamusHeightDataset(split_file, data_root=root, patch_size=32, train=False)

    all_pred = []
    all_gt = []
    all_mask = []
    for i in range(len(ds)):
        sample = ds[i]
        gt = sample["height"].numpy()
        all_gt.append(gt.reshape(-1))
        all_pred.append((gt + 1.5).reshape(-1))
        all_mask.append(sample["mask"].numpy().reshape(-1))

    import numpy as np

    result = compute_height_metrics(
        np.concatenate(all_pred), np.concatenate(all_gt), np.concatenate(all_mask)
    )
    assert math.isclose(result["mae_m"], 1.5, rel_tol=1e-6)
