"""Deterministic seeding and small shared helpers for ResHeightNet.

Seeding covers torch, numpy, and Python's random module, plus cuDNN
determinism flags. Called once at the start of any entry point
(train.py, evaluate.py, infer.py) before any model/data object is built.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int = 42) -> None:
    """Seed torch, numpy, and random; force cuDNN determinism.

    NOTE: does not seed per-DataLoader-worker numpy RNGs — PyTorch's
    worker seeding covers torch/random per worker but NOT numpy. Any
    numpy-based randomness inside Dataset.__getitem__ must be replaced
    with torch/random calls, or workers will emit duplicate augmentations.
    See tests/test_dataset.py::test_no_duplicate_crops_across_workers.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_data_root() -> str:
    """Locate the GAMUS dataset root, trying env var then known platform paths.

    Probes for an `images/` subdirectory rather than just directory
    existence, so a half-populated or wrong folder fails loudly instead
    of silently yielding an empty dataset.
    """
    candidates = [
        os.environ.get("GAMUS_ROOT"),
        "/content/gamus",
        r"D:\RP_implementation\data\gamus",
    ]
    for root in candidates:
        if root and os.path.isdir(os.path.join(root, "images")):
            return root
    raise FileNotFoundError(
        "GAMUS data root not found. Set the GAMUS_ROOT environment variable, "
        "or place data at /content/gamus (Colab) or "
        r"D:\RP_implementation\data\gamus (local), "
        "each containing images/, heights/, classes/ subdirectories. "
        "Run scripts/fetch_gamus.py first."
    )


def _numpy_rng_state_to_plain(state: tuple) -> dict:
    """np.random.get_state() returns (str, ndarray[624] uint32, int, int,
    float). The ndarray is NOT safe-unpicklable under torch's
    weights_only=True default (it requires allowlisting
    numpy._core.multiarray._reconstruct) -- converting it to a plain
    list here keeps the whole checkpoint loadable with the safe
    default. Discovered by tests/test_integration.py's
    checkpoint-round-trip test, which failed with exactly this error
    before this conversion was added."""
    algo, keys, pos, has_gauss, cached_gaussian = state
    return {
        "algo": algo,
        "keys": keys.tolist(),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _numpy_rng_state_from_plain(plain: dict) -> tuple:
    return (
        plain["algo"],
        np.array(plain["keys"], dtype=np.uint32),
        plain["pos"],
        plain["has_gauss"],
        plain["cached_gaussian"],
    )


def save_checkpoint(path: str, *, epoch: int, model, optimizer=None, scaler=None,
                     best_val: float, config: dict) -> None:
    """Write a checkpoint containing only plain-Python/tensor state.

    torch.load defaults to weights_only=True since torch 2.6. Saving an
    argparse.Namespace, pathlib.Path, or other non-tensor object here
    will produce a checkpoint that cannot be loaded back with the safe
    default. `config` MUST be a plain dict. numpy's RNG state is also
    NOT weights_only-safe as-is (see _numpy_rng_state_to_plain) and is
    converted before saving.
    """
    if not isinstance(config, dict):
        raise TypeError(f"config must be a plain dict, got {type(config)!r}")
    py_rng = random.getstate()
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_val": best_val,
        "config": config,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": _numpy_rng_state_to_plain(np.random.get_state()),
        # random.getstate() is (int, tuple[int, ...], float|None) -- all
        # plain types already, but the middle element is a tuple; store
        # it as a list so the whole structure is list/dict/str/int/float
        # only, matching what weights_only=True allowlists by default.
        "python_rng_state": [py_rng[0], list(py_rng[1]), py_rng[2]],
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    torch.save(state, path)


def load_checkpoint(path: str, *, model, optimizer=None, scaler=None, device="cpu",
                     restore_rng: bool = False) -> dict:
    """Load a checkpoint written by save_checkpoint, using the safe default.

    weights_only=True is left at its torch-2.6 default deliberately: it is
    the guard that catches an accidental non-tensor object in the
    checkpoint (a config that isn't a plain dict) before it becomes a
    Colab-side failure hours into training.
    """
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in state:
        scaler.load_state_dict(state["scaler_state_dict"])
    if restore_rng:
        torch.set_rng_state(state["torch_rng_state"])
        np.random.set_state(_numpy_rng_state_from_plain(state["numpy_rng_state"]))
        py_rng = state["python_rng_state"]
        random.setstate((py_rng[0], tuple(py_rng[1]), py_rng[2]))
    return state
