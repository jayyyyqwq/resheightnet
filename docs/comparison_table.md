# ResHeightNet vs. DepthWizard — GAMUS stage_a1 comparison

Evaluation protocol (identical for every row): 200 `stage_a1_val` tiles,
512x512 **center crop** at `top=left=256`, valid mask =
`isfinite(h) & (h>=0) & (0<=cls<=6)`, metrics **pooled per-pixel across
all 200 tiles** (not per-tile-averaged), predictions **not** clamped to
`>= 0`. See `docs/limitations.md` for what this protocol does and does
not control for (notably: train/val spatial adjacency within DC).

| Model | MAE (m) | RMSE (m) | Pearson r | Source |
|---|---|---|---|---|
| Constant predictor (median height) | TBD | TBD | TBD | `src/evaluate.py --constant-baseline` (M5) |
| DepthWizard M1 *(reported, not re-run)* | 3.308 | 6.275 | 0.68 | `docs/evidence/product_scientific_audit_summary.json` → `global_metrics_phl_dc_val`, in the DepthWizard repo. **No script in that repository regenerates this JSON** — it is a recorded evaluation result, not one this project can independently reproduce. |
| ResHeightNet (ours) | TBD | TBD | TBD | `src/evaluate.py --checkpoint results/checkpoints/best.pt` (M8, after Colab training) |

**Never write an estimated number into this table.** A row stays `TBD`
until the corresponding command has actually been run and its output
copied here.

## How to fill in the TBD rows

```powershell
# Constant-predictor baseline (M5) -- run once the 20-tile local subset
# or the full val split is fetched:
D:\RP_implementation\.venv\Scripts\python.exe -m src.evaluate --split data\splits\stage_a1_val.txt --constant-baseline

# ResHeightNet (M8) -- after training on Colab and downloading
# results/checkpoints/best.pt locally:
D:\RP_implementation\.venv\Scripts\python.exe -m src.evaluate --split data\splits\stage_a1_val.txt --checkpoint results\checkpoints\best.pt --out docs\resheightnet_metrics.json
```
