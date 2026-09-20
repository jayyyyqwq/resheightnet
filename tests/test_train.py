"""Tests for src/train.py: the M6 overfit gate and the M7 training loop
wiring (checkpointing, resume, dataloader construction), on tiny
synthetic data so these run in seconds, not minutes."""
from __future__ import annotations

import os

import torch

from src.train import build_dataloaders, main as train_main, run_overfit_gate, train_full
from tests.helpers import make_synthetic_tile, write_split_file


def _make_split(tmp_path, n_tiles, size=64, nodata_fraction=0.02, seed=0, prefix="split"):
    root = str(tmp_path / "gamus")
    # sample_id includes `prefix` so multiple _make_split calls sharing
    # the same tmp_path/root (e.g. one for train, one for val) never
    # collide on identical .h5 filenames and silently overwrite each
    # other's data -- this bit a test in development (see
    # test_build_dataloaders_reports_dropped_tiles's history).
    tiles = [
        make_synthetic_tile(root, f"DC_{prefix}{i}_{prefix}{i}", size=size,
                             nodata_fraction=nodata_fraction, seed=seed + i)
        for i in range(n_tiles)
    ]
    split_file = tmp_path / f"{prefix}.txt"
    write_split_file(str(split_file), tiles)
    return root, str(split_file)


def test_run_overfit_gate_returns_expected_keys_and_converges(tmp_path):
    root, split_file = _make_split(tmp_path, n_tiles=2, size=64)
    result = run_overfit_gate(
        split_file, root, n_tiles=2, steps=60, patch_size=32, lr=3e-3, seed=0, log_every=100
    )
    assert set(result.keys()) == {"losses", "l0", "final", "final_over_l0", "n_tiles"}
    assert len(result["losses"]) == 60
    assert result["n_tiles"] == 2
    assert result["final"] < result["losses"][0], "loss should decrease from its initial value"


def test_run_overfit_gate_constant_image_control_learns_less(tmp_path):
    """The control: feeding every sample the same gray image (targets
    unchanged) should converge WORSE (or at least not obviously better)
    than the real-RGB run, since the model can no longer distinguish
    samples by content."""
    root, split_file = _make_split(tmp_path, n_tiles=2, size=64)

    real = run_overfit_gate(
        split_file, root, n_tiles=2, steps=60, patch_size=32, lr=3e-3, seed=0, log_every=100
    )
    control = run_overfit_gate(
        split_file, root, n_tiles=2, steps=60, patch_size=32, lr=3e-3, seed=0, log_every=100,
        constant_image_control=True,
    )
    assert control["final"] >= real["final"] - 1e-6, (
        "constant-image control should not converge better than the real run"
    )


def test_build_dataloaders_reports_dropped_tiles(tmp_path, capsys):
    """One bad (99% nodata, dropped) tile alongside one good tile: the
    drop must be reported AND training must still proceed on the
    surviving tile."""
    root = str(tmp_path / "gamus")
    good = make_synthetic_tile(root, "DC_good_good", size=64, nodata_fraction=0.0, seed=0)
    bad = make_synthetic_tile(root, "DC_bad_bad", size=64, nodata_fraction=0.99, seed=1)
    split_file = tmp_path / "mixed.txt"
    write_split_file(str(split_file), [good, bad])

    val_root, val_split = _make_split(tmp_path, n_tiles=2, size=64, nodata_fraction=0.0, prefix="val")

    train_loader, val_loader = build_dataloaders(
        str(split_file), val_split, root, patch_size=32, batch_size=2, num_workers=0
    )
    assert len(train_loader.dataset) == 1, "only the 99%-nodata tile should be dropped"
    assert len(val_loader.dataset) == 2, "val must never drop tiles"

    captured = capsys.readouterr()
    assert "dropped" in captured.out
    assert "DC_bad_bad" in captured.out


def test_build_dataloaders_raises_clear_error_when_all_train_tiles_dropped(tmp_path):
    """If every train tile is dropped, DataLoader(shuffle=True) on an
    empty dataset would otherwise fail deep inside torch's
    RandomSampler with a cryptic 'num_samples should be a positive
    integer' error -- build_dataloaders must fail fast with a message
    that actually explains what happened."""
    root, split_file = _make_split(tmp_path, n_tiles=2, size=64, nodata_fraction=0.99)
    val_root, val_split = _make_split(tmp_path, n_tiles=2, size=64, nodata_fraction=0.0, prefix="val")

    try:
        build_dataloaders(split_file, val_split, root, patch_size=32, batch_size=2, num_workers=0)
        assert False, "expected ValueError when all train tiles are dropped"
    except ValueError as exc:
        assert "dropped" in str(exc).lower()


def test_train_full_runs_one_epoch_and_writes_checkpoints(tmp_path):
    # _make_split always shares one root ("gamus") across calls, and its
    # sample_id encodes `prefix`, so train and val tiles coexist under
    # the same data_root without filename collisions.
    root, train_split = _make_split(tmp_path, n_tiles=2, size=64, nodata_fraction=0.0, prefix="train")
    _, val_split = _make_split(tmp_path, n_tiles=2, size=64, nodata_fraction=0.0, seed=10, prefix="val")

    ckpt_dir = str(tmp_path / "checkpoints")
    train_full(
        train_split=train_split,
        val_split=val_split,
        data_root=root,
        epochs=1,
        batch_size=2,
        patch_size=32,
        lr=1e-4,
        weight_decay=1e-4,
        seed=0,
        num_workers=0,
        checkpoint_dir=ckpt_dir,
        amp=False,
    )

    assert os.path.isfile(os.path.join(ckpt_dir, "last.pt"))
    assert os.path.isfile(os.path.join(ckpt_dir, "best.pt"))


def test_train_full_resume_continues_from_saved_epoch(tmp_path):
    root, train_split = _make_split(tmp_path, n_tiles=2, size=64, nodata_fraction=0.0, prefix="train")
    val_split = train_split  # reuse for a fast, self-contained test

    ckpt_dir = str(tmp_path / "checkpoints")
    train_full(
        train_split=train_split, val_split=val_split, data_root=root,
        epochs=1, batch_size=2, patch_size=32, seed=0, num_workers=0,
        checkpoint_dir=ckpt_dir, amp=False,
    )
    last_path = os.path.join(ckpt_dir, "last.pt")
    assert os.path.isfile(last_path)

    # Resume for one more epoch from the saved checkpoint -- must not
    # raise, and must advance past epoch 0.
    train_full(
        train_split=train_split, val_split=val_split, data_root=root,
        epochs=2, batch_size=2, patch_size=32, seed=0, num_workers=0,
        checkpoint_dir=ckpt_dir, amp=False, resume_from=last_path,
    )
    state = torch.load(last_path, map_location="cpu", weights_only=True)
    assert state["epoch"] == 1


def test_cli_overfit_mode_runs_and_prints_verdict(tmp_path, monkeypatch, capsys):
    root, split_file = _make_split(tmp_path, n_tiles=2, size=64)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--val-split", split_file,
            "--data-root", root,
            "--overfit", "2",
            "--steps", "10",
            "--patch-size", "32",
        ],
    )
    train_main()
    captured = capsys.readouterr()
    assert "overfit gate" in captured.out
    assert ("PASS" in captured.out) or ("FAIL" in captured.out)


def test_cli_train_mode_runs_one_epoch(tmp_path, monkeypatch):
    root, train_split = _make_split(tmp_path, n_tiles=2, size=64, nodata_fraction=0.0, prefix="train")
    val_split = train_split
    ckpt_dir = str(tmp_path / "cli_checkpoints")

    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--train-split", train_split,
            "--val-split", val_split,
            "--data-root", root,
            "--epochs", "1",
            "--batch-size", "2",
            "--patch-size", "32",
            "--num-workers", "0",
            "--checkpoint-dir", ckpt_dir,
            "--no-amp",
        ],
    )
    train_main()
    assert os.path.isfile(os.path.join(ckpt_dir, "last.pt"))
