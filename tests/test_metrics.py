"""M5: eval harness + metric parity.

Includes the parity test against tests/reference/depthwizard_metrics.py
(vendored verbatim from DepthWizard's src/training/metrics.py) -- the
test that actually proves ResHeightNet's numbers are comparable to the
published 3.308/6.275/0.68 baseline, not just internally self-consistent.
"""
from __future__ import annotations

import math
import tracemalloc

import numpy as np
import pytest

from src.metrics import compute_height_metrics, constant_predictor_metrics, pearson_chunked
from tests.reference.depthwizard_metrics import compute_height_metrics as reference_compute_height_metrics


def test_basic_metrics_pred_equals_gt_plus_constant():
    rng = np.random.default_rng(0)
    gt = rng.uniform(0, 30, size=1000).astype(np.float32)
    pred = gt + 2.0
    mask = np.ones_like(gt, dtype=bool)

    result = compute_height_metrics(pred, gt, mask)
    assert math.isclose(result["mae_m"], 2.0, rel_tol=1e-5)
    assert math.isclose(result["rmse_m"], 2.0, rel_tol=1e-5)
    assert math.isclose(result["pearson_r"], 1.0, rel_tol=1e-5)


def test_nodata_pixels_do_not_affect_metrics_at_all():
    """The nodata-killer test: adding extreme gt=-5, pred=1000 pairs
    (i.e. exactly GAMUS's nodata encoding, fed a wildly wrong prediction)
    must leave all three headline metrics bit-identical -- if the mask
    logic has any leak, this is where it would show up."""
    rng = np.random.default_rng(1)
    gt = rng.uniform(0, 30, size=500).astype(np.float32)
    pred = gt + 1.5
    mask = np.ones_like(gt, dtype=bool)
    baseline = compute_height_metrics(pred, gt, mask)

    nodata_gt = np.concatenate([gt, np.full(500, -5.0, dtype=np.float32)])
    nodata_pred = np.concatenate([pred, np.full(500, 1000.0, dtype=np.float32)])
    nodata_mask = np.concatenate([mask, np.ones(500, dtype=bool)])
    with_nodata = compute_height_metrics(nodata_pred, nodata_gt, nodata_mask)

    assert with_nodata["mae_m"] == baseline["mae_m"]
    assert with_nodata["rmse_m"] == baseline["rmse_m"]
    assert with_nodata["pearson_r"] == baseline["pearson_r"]
    assert with_nodata["n_valid"] == baseline["n_valid"]


def test_predictions_are_not_clamped_to_nonnegative():
    """The M1 parity path does NOT clamp predictions (unlike DepthWizard's
    M2/M3 paths) -- a negative prediction must flow straight into MAE/RMSE
    unmodified."""
    gt = np.array([5.0, 5.0], dtype=np.float32)
    pred = np.array([-3.0, 5.0], dtype=np.float32)
    mask = np.ones(2, dtype=bool)

    result = compute_height_metrics(pred, gt, mask)
    # |-3-5| + |5-5| = 8 + 0, mean = 4.0. If clamped to 0, would be
    # |0-5| + |0-5| = 5+0=... different value entirely (2.5) -- so this
    # distinguishes clamped from unclamped behavior.
    assert math.isclose(result["mae_m"], 4.0, rel_tol=1e-6)


def test_pooled_across_tiles_differs_from_mean_of_per_tile_metrics():
    """Pins the exact protocol: metrics pooled per-pixel across ALL
    tiles must NOT equal the mean of independently-computed per-tile
    metrics when tiles have different valid-pixel counts."""
    tile_a_gt = np.array([10.0] * 100, dtype=np.float32)
    tile_a_pred = tile_a_gt + 1.0  # MAE 1.0, 100 valid pixels
    tile_a_mask = np.ones(100, dtype=bool)

    tile_b_gt = np.array([10.0] * 10, dtype=np.float32)
    tile_b_pred = tile_b_gt + 5.0  # MAE 5.0, only 10 valid pixels
    tile_b_mask = np.ones(10, dtype=bool)

    per_tile_mean = (1.0 + 5.0) / 2  # naive per-tile average = 3.0

    pooled_gt = np.concatenate([tile_a_gt, tile_b_gt])
    pooled_pred = np.concatenate([tile_a_pred, tile_b_pred])
    pooled_mask = np.concatenate([tile_a_mask, tile_b_mask])
    pooled_result = compute_height_metrics(pooled_pred, pooled_gt, pooled_mask)

    # Weighted-by-pixel-count: (100*1.0 + 10*5.0) / 110 = 150/110 = 1.3636...
    expected_pooled = (100 * 1.0 + 10 * 5.0) / 110
    assert math.isclose(pooled_result["mae_m"], expected_pooled, rel_tol=1e-5)
    assert not math.isclose(pooled_result["mae_m"], per_tile_mean, rel_tol=1e-2)


def test_empty_mask_returns_reference_zero_dict():
    gt = np.array([1.0, 2.0], dtype=np.float32)
    pred = np.array([1.0, 2.0], dtype=np.float32)
    mask = np.zeros(2, dtype=bool)
    result = compute_height_metrics(pred, gt, mask)
    assert result["mae_m"] == 0.0
    assert result["rmse_m"] == 0.0
    assert result["pearson_r"] == 0.0
    assert result["n_valid"] == 0


# ---------------------------------------------------------------------------
# Chunked Pearson: exactness and memory bound
# ---------------------------------------------------------------------------


def test_pearson_chunked_matches_numpy_corrcoef():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 30, size=8_000_000).astype(np.float32)
    y = x * 0.7 + rng.normal(0, 2, size=8_000_000).astype(np.float32)

    chunked = pearson_chunked(x, y, chunk_size=1_000_000)
    reference = float(np.corrcoef(x.astype(np.float64), y.astype(np.float64))[0, 1])

    assert abs(chunked - reference) < 1e-10, f"chunked={chunked} reference={reference}"


def test_pearson_chunked_memory_bound():
    """tracemalloc-based peak memory guard: the chunked implementation
    must not blow past a modest budget regardless of input size, which
    is the entire reason it exists (the naive scipy.pearsonr on 52.4M
    elements peaks at 2.5-3.5 GB on this machine's ~3.2 GB free RAM)."""
    rng = np.random.default_rng(3)
    n = 20_000_000
    x = rng.uniform(0, 30, size=n).astype(np.float32)
    y = rng.uniform(0, 30, size=n).astype(np.float32)

    tracemalloc.start()
    pearson_chunked(x, y, chunk_size=4_000_000)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)
    assert peak_mb < 300, f"pearson_chunked peak memory {peak_mb:.1f} MB exceeds 300 MB budget"


def test_pearson_chunked_empty_input_returns_zero():
    assert pearson_chunked(np.array([], dtype=np.float32), np.array([], dtype=np.float32)) == 0.0


# ---------------------------------------------------------------------------
# Parity vs. vendored DepthWizard reference implementation
# ---------------------------------------------------------------------------


def test_parity_against_vendored_depthwizard_metrics():
    """The comparability test: on random masked data, this project's
    compute_height_metrics must match DepthWizard's own
    src/training/metrics.py (vendored verbatim in tests/reference/) to
    numerical precision, differing only in that ours uses a chunked
    Pearson (exact, just computed differently) instead of scipy.pearsonr
    directly."""
    rng = np.random.default_rng(4)
    n = 5000
    gt = rng.uniform(-2, 30, size=n).astype(np.float32)  # includes some nodata-like negatives
    pred = gt + rng.normal(0, 3, size=n).astype(np.float32)
    mask = rng.random(n) > 0.1  # ~10% masked out, independent of nodata
    semantic = rng.integers(0, 7, size=n)

    ours = compute_height_metrics(pred, gt, mask, semantic)
    reference = reference_compute_height_metrics(pred, gt, mask, semantic)

    assert math.isclose(ours["mae_m"], reference["mae_m"], rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(ours["rmse_m"], reference["rmse_m"], rel_tol=1e-6, abs_tol=1e-9)
    # Pearson tolerance is looser (1e-6, not 1e-9): pearson_chunked and
    # scipy.stats.pearsonr are two different (both exact) floating-point
    # reduction orders over the same float32 data, so they agree to
    # ~1e-8 relative, not bit-for-bit. MAE/RMSE above use simple
    # sum/mean and do agree to 1e-9.
    assert math.isclose(ours["pearson_r"], reference["pearson_r"], rel_tol=1e-6, abs_tol=1e-9)
    assert ours["n_valid"] == reference["valid_pixels"]


# ---------------------------------------------------------------------------
# Constant-predictor baseline
# ---------------------------------------------------------------------------


def test_constant_predictor_metrics_uses_median():
    gt = np.array([1.0, 2.0, 3.0, 4.0, 100.0], dtype=np.float32)  # median = 3.0
    mask = np.ones(5, dtype=bool)
    result = constant_predictor_metrics(gt, mask)
    assert math.isclose(result["constant_value_m"], 3.0, rel_tol=1e-6)
    # MAE of predicting the median: |1-3|+|2-3|+|3-3|+|4-3|+|100-3| / 5
    expected_mae = (2 + 1 + 0 + 1 + 97) / 5
    assert math.isclose(result["mae_m"], expected_mae, rel_tol=1e-5)


def test_evaluate_split_stub_model_yields_exact_mae():
    """evaluate.py integration point: a stub model that always returns
    gt+1.5 must yield MAE exactly 1.5 through the full evaluate_split
    pipeline. See tests/test_integration.py for the actual wiring test;
    this test exercises just the metrics math with the same fixture
    shape evaluate_split produces."""
    gt = np.full(4096, 10.0, dtype=np.float32)
    pred = gt + 1.5
    mask = np.ones(4096, dtype=bool)
    result = compute_height_metrics(pred, gt, mask)
    assert math.isclose(result["mae_m"], 1.5, rel_tol=1e-6)
