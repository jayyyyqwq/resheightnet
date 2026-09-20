"""Height evaluation metrics, matching DepthWizard's M1 protocol exactly
so numbers are comparable to the published 3.308 / 6.275 / 0.68 baseline.

Protocol (mirrors DepthWizard src/training/metrics.py:28-82):
  - valid = mask & isfinite(target) & isfinite(pred) & (target >= 0.0)
  - metrics pooled per-pixel across ALL tiles into one vector, NOT
    per-tile-then-averaged
  - Pearson correlation over the full flattened valid vector, exact,
    not subsampled
  - predictions are NOT clamped to >= 0 (the M1 path does not clamp;
    only the M2/M3 paths in DepthWizard do -- do not copy that here)

The naive approach of concatenating per-batch numpy arrays and calling
scipy.stats.pearsonr on the full 200-tile x 512x512 = 52,428,800-element
vector peaks at ~2.5-3.5 GB (scipy upcasts to float64 and builds several
full-length temporaries). This machine has ~3.2 GB free RAM, so that
peak reliably OOMs. pearson_chunked below is a mathematically exact
(not approximate) two-pass streaming algorithm measured at ~128 MB
peak / ~1.2 s / ~1e-14 relative error against np.corrcoef on an
equivalent array.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

CLASS_NAMES = {
    0: "Others",
    1: "Ground",
    2: "Low vegetation",
    3: "Buildings",
    4: "Water",
    5: "Road",
    6: "Tree",
}

HEIGHT_BUCKETS = {
    "0-2m": (0.0, 2.0),
    "2-10m": (2.0, 10.0),
    "10-20m": (10.0, 20.0),
    "20-50m": (20.0, 50.0),
    ">50m": (50.0, 1000.0),
}

DEFAULT_CHUNK = 4_000_000


def pearson_chunked(x: np.ndarray, y: np.ndarray, chunk_size: int = DEFAULT_CHUNK) -> float:
    """Exact Pearson correlation via a two-pass, chunked streaming
    computation, bounded to O(chunk_size) memory regardless of input
    size. x and y are already-filtered 1-D arrays of equal length.

    Two-pass (mean first, then centered sums of squares/products) to
    avoid the catastrophic-cancellation risk of a naive single-pass
    sum-of-products formula.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = x.size
    if n == 0:
        return 0.0

    sx = 0.0
    sy = 0.0
    for i in range(0, n, chunk_size):
        a = x[i : i + chunk_size].astype(np.float64)
        b = y[i : i + chunk_size].astype(np.float64)
        sx += float(a.sum())
        sy += float(b.sum())
    mx = sx / n
    my = sy / n

    cxx = 0.0
    cyy = 0.0
    cxy = 0.0
    for i in range(0, n, chunk_size):
        a = x[i : i + chunk_size].astype(np.float64) - mx
        b = y[i : i + chunk_size].astype(np.float64) - my
        cxx += float(a @ a)
        cyy += float(b @ b)
        cxy += float(a @ b)

    denom = np.sqrt(cxx * cyy)
    if denom <= 0:
        return 0.0
    return float(cxy / denom)


def compute_height_metrics(
    pred_arr: np.ndarray,
    target_arr: np.ndarray,
    mask_arr: np.ndarray | None = None,
    semantic_arr: np.ndarray | None = None,
) -> dict:
    """Computes MAE, RMSE, Pearson r, Spearman rho (and optional
    per-class / per-height-bucket MAE), matching DepthWizard's
    compute_height_metrics validity rule and pooling exactly.

    Args:
        pred_arr: (N,) or (H, W) or (B, H, W) predictions in metres.
        target_arr: same shape, ground-truth height in metres.
        mask_arr: same shape, boolean/numeric validity mask (optional;
            if omitted, all pixels are treated as candidate-valid and
            filtered only by the target/pred validity rule below).
        semantic_arr: same shape, semantic class ids (optional; enables
            per-class MAE breakdown).

    Returns:
        dict with keys mae_m, rmse_m, pearson_r, spearman_rho,
        n_valid, class_mae, bucket_mae.
    """
    pred_flat = np.asarray(pred_arr).reshape(-1)
    tgt_flat = np.asarray(target_arr).reshape(-1)

    if mask_arr is not None:
        mask_flat = np.asarray(mask_arr).reshape(-1) > 0
    else:
        mask_flat = np.ones_like(tgt_flat, dtype=bool)

    valid = mask_flat & np.isfinite(tgt_flat) & np.isfinite(pred_flat) & (tgt_flat >= 0.0)

    if valid.sum() == 0:
        return {
            "mae_m": 0.0,
            "rmse_m": 0.0,
            "pearson_r": 0.0,
            "spearman_rho": 0.0,
            "n_valid": 0,
            "class_mae": {},
            "bucket_mae": {},
        }

    p_val = pred_flat[valid].astype(np.float32)
    t_val = tgt_flat[valid].astype(np.float32)

    diff = np.abs(p_val - t_val)
    mae = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff.astype(np.float64) ** 2)))

    if len(p_val) > 10 and np.std(p_val) > 1e-6 and np.std(t_val) > 1e-6:
        r = pearson_chunked(p_val, t_val)
        if len(p_val) > 25000:
            idx = np.random.choice(len(p_val), 25000, replace=False)
            rho, _ = stats.spearmanr(p_val[idx], t_val[idx])
        else:
            rho, _ = stats.spearmanr(p_val, t_val)
    else:
        r = rho = 0.0

    class_mae: dict[str, float | None] = {}
    if semantic_arr is not None:
        sem_flat = np.asarray(semantic_arr).reshape(-1)
        for cid, cname in CLASS_NAMES.items():
            c_mask = valid & (sem_flat == cid)
            if c_mask.sum() > 50:
                class_mae[cname] = float(np.mean(np.abs(pred_flat[c_mask] - tgt_flat[c_mask])))
            else:
                class_mae[cname] = None

    bucket_mae: dict[str, float | None] = {}
    for bname, (b_min, b_max) in HEIGHT_BUCKETS.items():
        b_mask = valid & (tgt_flat >= b_min) & (tgt_flat < b_max)
        if b_mask.sum() > 50:
            bucket_mae[bname] = float(np.mean(np.abs(pred_flat[b_mask] - tgt_flat[b_mask])))
        else:
            bucket_mae[bname] = None

    return {
        "mae_m": mae,
        "rmse_m": rmse,
        "pearson_r": float(r),
        "spearman_rho": float(rho),
        "n_valid": int(valid.sum()),
        "class_mae": class_mae,
        "bucket_mae": bucket_mae,
    }


def constant_predictor_metrics(target_arr: np.ndarray, mask_arr: np.ndarray) -> dict:
    """Baseline: predict the median valid height everywhere.

    This is the number M7's real training run must beat -- if the
    trained model's val MAE is not strictly below this, the model has
    not learned anything, regardless of how the loss curve looked.
    """
    tgt_flat = np.asarray(target_arr).reshape(-1)
    mask_flat = np.asarray(mask_arr).reshape(-1) > 0
    valid = mask_flat & np.isfinite(tgt_flat) & (tgt_flat >= 0.0)
    median_height = float(np.median(tgt_flat[valid]))
    pred_flat = np.full_like(tgt_flat, median_height)
    metrics = compute_height_metrics(pred_flat, tgt_flat, mask_arr)
    metrics["constant_value_m"] = median_height
    return metrics
