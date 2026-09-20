"""
DepthWizard (SIH26175) — Metric Height Evaluation Metrics
Player 1: AI/ML Lead
"""

import numpy as np
import scipy.stats as stats
import torch

CLASS_NAMES = {
    0: 'Others',
    1: 'Ground',
    2: 'Low vegetation',
    3: 'Buildings',
    4: 'Water',
    5: 'Road',
    6: 'Tree'
}

HEIGHT_BUCKETS = {
    '0-2m': (0.0, 2.0),
    '2-10m': (2.0, 10.0),
    '10-20m': (10.0, 20.0),
    '20-50m': (20.0, 50.0),
    '>50m': (50.0, 1000.0)
}

def compute_height_metrics(pred_arr, target_arr, mask_arr=None, semantic_arr=None):
    """
    Computes rigorous physical height evaluation metrics in metres.
    Args:
        pred_arr: np.ndarray (H, W) or (N,)
        target_arr: np.ndarray (H, W) or (N,)
        mask_arr: np.ndarray (H, W) or (N,) bool
        semantic_arr: np.ndarray (H, W) or (N,) int (optional)
    Returns:
        dict of metric values
    """
    if isinstance(pred_arr, torch.Tensor):
        pred_arr = pred_arr.detach().cpu().numpy()
    if isinstance(target_arr, torch.Tensor):
        target_arr = target_arr.detach().cpu().numpy()
    if isinstance(mask_arr, torch.Tensor):
        mask_arr = mask_arr.detach().cpu().numpy()
    if isinstance(semantic_arr, torch.Tensor):
        semantic_arr = semantic_arr.detach().cpu().numpy()

    pred_flat = pred_arr.reshape(-1)
    tgt_flat = target_arr.reshape(-1)
    
    if mask_arr is not None:
        mask_flat = mask_arr.reshape(-1) > 0
    else:
        mask_flat = np.ones_like(tgt_flat, dtype=bool)

    valid = mask_flat & np.isfinite(tgt_flat) & np.isfinite(pred_flat) & (tgt_flat >= 0.0)
    
    if valid.sum() == 0:
        return {
            'mae_m': 0.0, 'rmse_m': 0.0, 'pearson_r': 0.0, 'spearman_rho': 0.0,
            'class_mae': {}, 'bucket_mae': {}
        }

    p_val = pred_flat[valid]
    t_val = tgt_flat[valid]

    # Global Metrics
    diff = np.abs(p_val - t_val)
    mae = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean((p_val - t_val) ** 2)))

    # Pearson Correlation
    if len(p_val) > 10 and np.std(p_val) > 1e-6 and np.std(t_val) > 1e-6:
        r, _ = stats.pearsonr(p_val, t_val)
        # Subsample for Spearman rank if large
        if len(p_val) > 25000:
            idx = np.random.choice(len(p_val), 25000, replace=False)
            rho, _ = stats.spearmanr(p_val[idx], t_val[idx])
        else:
            rho, _ = stats.spearmanr(p_val, t_val)
    else:
        r = rho = 0.0

    # Semantic Class-Wise MAE
    class_mae = {}
    if semantic_arr is not None:
        sem_flat = semantic_arr.reshape(-1)
        for cid, cname in CLASS_NAMES.items():
            c_mask = valid & (sem_flat == cid)
            if c_mask.sum() > 50:
                c_diff = np.abs(pred_flat[c_mask] - tgt_flat[c_mask])
                class_mae[cname] = float(np.mean(c_diff))
            else:
                class_mae[cname] = None

    # Height-Bucket MAE
    bucket_mae = {}
    for bname, (b_min, b_max) in HEIGHT_BUCKETS.items():
        b_mask = valid & (tgt_flat >= b_min) & (tgt_flat < b_max)
        if b_mask.sum() > 50:
            b_diff = np.abs(pred_flat[b_mask] - tgt_flat[b_mask])
            bucket_mae[bname] = float(np.mean(b_diff))
        else:
            bucket_mae[bname] = None

    return {
        'mae_m': mae,
        'rmse_m': rmse,
        'pearson_r': float(r),
        'spearman_rho': float(rho),
        'valid_pixels': int(valid.sum()),
        'class_mae': class_mae,
        'bucket_mae': bucket_mae
    }
