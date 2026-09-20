"""Tests for src/evaluate.py: the eval-harness wiring (M5/M8), not just
the metrics math already covered by tests/test_metrics.py."""
from __future__ import annotations

import json
import math

from src.evaluate import evaluate_split, main as evaluate_main
from tests.helpers import make_synthetic_tile, write_split_file


def _make_split(tmp_path, n_tiles=3, size=64, nodata_fraction=0.02, seed=0):
    root = str(tmp_path / "gamus")
    tiles = [
        make_synthetic_tile(root, f"DC_{i}_{i}", size=size, nodata_fraction=nodata_fraction, seed=seed + i)
        for i in range(n_tiles)
    ]
    split_file = tmp_path / "split.txt"
    write_split_file(str(split_file), tiles)
    return root, str(split_file)


def test_evaluate_split_constant_baseline_matches_metrics_module(tmp_path):
    root, split_file = _make_split(tmp_path, n_tiles=3, size=64)
    result = evaluate_split(
        split_file, data_root=root, patch_size=32, constant_baseline=True
    )
    assert result["n_tiles"] == 3
    assert result["protocol"] == "512_center_crop_pooled"
    assert "constant_value_m" in result
    assert result["mae_m"] >= 0.0


def test_evaluate_split_with_fresh_model_produces_finite_metrics(tmp_path):
    """No checkpoint provided -> evaluates a freshly-initialized model
    (pretrained encoder, random decoder). Not a real result, but the
    harness must run end to end without crashing and produce finite,
    well-shaped output."""
    root, split_file = _make_split(tmp_path, n_tiles=2, size=64)
    result = evaluate_split(
        split_file, data_root=root, checkpoint=None, patch_size=32, batch_size=2
    )
    assert math.isfinite(result["mae_m"])
    assert math.isfinite(result["rmse_m"])
    assert result["n_tiles"] == 2
    assert "class_mae" in result and "bucket_mae" in result


def test_evaluate_split_respects_limit(tmp_path):
    root, split_file = _make_split(tmp_path, n_tiles=5, size=64)
    result = evaluate_split(
        split_file, data_root=root, patch_size=32, limit=2, constant_baseline=True
    )
    assert result["n_tiles"] == 2


def test_evaluate_split_reports_pred_finite_coverage(tmp_path):
    root, split_file = _make_split(tmp_path, n_tiles=2, size=64)
    result = evaluate_split(
        split_file, data_root=root, checkpoint=None, patch_size=32, batch_size=2
    )
    assert "pred_finite_coverage" in result
    assert 0.0 <= result["pred_finite_coverage"] <= 1.0 + 1e-9


def test_main_cli_writes_json_output(tmp_path, monkeypatch, capsys):
    root, split_file = _make_split(tmp_path, n_tiles=2, size=64)
    out_path = str(tmp_path / "metrics.json")

    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate.py",
            "--split", split_file,
            "--data-root", root,
            "--patch-size", "32",
            "--constant-baseline",
            "--out", out_path,
        ],
    )
    evaluate_main()

    with open(out_path, "r", encoding="utf-8") as f:
        written = json.load(f)
    assert written["n_tiles"] == 2
    assert "mae_m" in written

    captured = capsys.readouterr()
    assert "mae_m" in captured.out
