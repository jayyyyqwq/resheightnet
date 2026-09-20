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
| **ResHeightNet (ours)** | **2.169** | **4.331** | **0.844** | Real run: 15 epochs on `stage_a1_train` (1000 tiles), ResNet34 encoder (ImageNet-pretrained, fully fine-tuned), AdamW lr=1e-4 wd=1e-4, seed=42, Colab T4. Evaluated on all 200 `stage_a1_val` tiles, `n_valid=51,283,929` px (97.8% of the 512²×200 pixel budget, consistent with GAMUS's ~1.4-2% nodata rate), `pred_finite_coverage=1.0` (no NaN predictions). |

**Never write an estimated number into this table.** A row stays `TBD`
until the corresponding command has actually been run and its output
copied here.

## Per-class and per-height-bucket MAE (ResHeightNet)

Not part of the headline comparison (DepthWizard's reported baseline
doesn't publish this breakdown), but useful for understanding where
the model is weaker:

| Semantic class | MAE (m) | | Height bucket | MAE (m) |
|---|---|---|---|---|
| Others | 1.032 | | 0-2m | 0.579 |
| Ground | 0.479 | | 2-10m | 2.423 |
| Low vegetation | 0.862 | | 10-20m | 5.053 |
| Buildings | 3.154 | | 20-50m | 9.556 |
| Water | 0.216 | | >50m | 60.762 |
| Road | 1.275 | | | |
| Tree | 4.364 | | | |

Error grows sharply with height and is worst on trees and tall
buildings — expected for a single-RGB-image model with no explicit
elevation prior (unlike DepthWizard, whose RDAH architecture takes a
frozen Depth Anything V2 depth prior as an additional input). The
`>50m` bucket is thin (few pixels that tall in the val set at all,
so a handful of large misses dominate that number) — treat it as
noisy, not representative.

## How to fill in the remaining TBD row

The constant-predictor row must be computed on the same full 200-tile
`stage_a1_val` set as the two rows above it (not the 20-tile local
sanity subset, which gives a different, non-comparable number) — run
inside the Colab session, where the full val split is already on disk:

```python
!python -m src.evaluate --split data/splits/stage_a1_val.txt --data-root /content/gamus --constant-baseline --out /content/constant_baseline_metrics.json
```
