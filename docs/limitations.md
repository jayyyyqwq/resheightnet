# Known Limitations

This document is maintained alongside `docs/comparison_table.md` and is
intended to be read together with it. Nothing here invalidates the
ResHeightNet vs. DepthWizard comparison — both systems use the
identical `stage_a1` split — but each item below affects how the
*absolute* numbers should be interpreted.

## Spatial adjacency between train and val tiles (DC)

GAMUS DC tiles are named `DC_<row>_<col>`, encoding their position in a
regular spatial grid. Checking `data/splits/stage_a1_train.txt` against
`data/splits/stage_a1_val.txt` (see `tests/test_splits.py::test_spatial_adjacency_disclosed_in_limitations`
for the exact, re-runnable check):

- **57 of the 100 DC validation tiles are edge-adjacent** (row/col
  differ by exactly 1 in one axis) to at least one DC training tile.
- **85 of the 100 DC validation tiles are adjacent including
  diagonals** (row/col differ by at most 1 in both axes).
- PHL sample IDs (`PHL_<n>`) do not encode a 2D grid position in this
  split format, so this check could not be extended to PHL tiles.

**Implication:** a model that partly memorizes local texture/context
near a tile boundary gets an easier validation set than a genuinely
geographically disjoint holdout would provide. The reported MAE/RMSE
for both DepthWizard M1 and ResHeightNet on `stage_a1_val` should be
read as **an apples-to-apples comparison on this split**, not as an
estimate of accuracy on unseen geography in general. DepthWizard's own
NYC-holdout numbers (fully disjoint city, not used by this project)
show a substantially worse MAE (4.912 m vs. 3.308 m for M1), which is
consistent with this effect.

**Why this project did not re-derive a spatially disjoint split:**
doing so would break comparability with DepthWizard's reported
baseline, which is the entire point of reusing `stage_a1` verbatim (see
`PROJECT_PLAN.md` Section 3). Disclosing the adjacency here, rather
than silently presenting the number as a clean generalization estimate,
is the tradeoff this project makes instead.

## DepthWizard's baseline numbers are reported, not re-run

The comparison figures (MAE 3.308 m / RMSE 6.275 m / Pearson r 0.68)
come from `docs/evidence/product_scientific_audit_summary.json` in the
DepthWizard repository (field `global_metrics_phl_dc_val`). No script in
that repository regenerates this JSON — it is a recorded evaluation
result, not a number this project (or anyone, from the DepthWizard repo
as it currently exists) can independently reproduce. See
`docs/comparison_table.md` for the full citation and the constant-
predictor baseline included alongside it for calibration.

## M6 overfit-gate learning rate differs from production training

`src/train.py::run_overfit_gate` (M6) uses `lr=3e-3` by default, not the
`lr=1e-4` used by `train_full` (M7, matching DepthWizard's recipe). This
was discovered empirically while building the test suite: on a
synthetic fixture with a genuinely learnable RGB→height relationship
(see `tests/test_integration.py`), a freshly-initialized decoder at
`lr=3e-4` (10x lower) had barely moved off its initial loss after 150
steps, while `lr=3e-3` converged well within a 300-step budget. The
overfit gate's purpose is a fast, falsifiable wiring check on a
handful of tiles, not a reproduction of the full training dynamics —
using the production lr here risks a false FAIL on a correctly-wired
pipeline simply because it has not converged yet within the step
budget. `train_full` (the real M7 run whose output is what gets
evaluated in `docs/comparison_table.md`) is unaffected and still uses
`lr=1e-4` to match DepthWizard exactly.

**Update after running on real data (Colab T4, real `stage_a1_val`
tiles):** at `lr=3e-3`/300 steps, the gate reached `final/L0=0.084`
(well under the 0.25 ratio threshold) but `final=0.4474m` (just above
the 0.30m absolute threshold) — a smooth, monotonically-decreasing
curve with no plateau, NaN, or instability at step 299. Real GAMUS
tiles carry more per-pixel texture/detail than the synthetic fixture
this default was tuned on, so 300 steps was a genuine step-budget
shortfall, not a wiring bug — the ratio criterion (which is
self-calibrating against that same tile set's L0) already confirmed
correct wiring. The notebook default is now `--steps 800`, which a T4
runs in a few extra minutes.

## GAMUS class-band dtype is inconsistent upstream

The `classes/*_CLS.h5` semantic label band is stored as `float32` for
DC tiles and `uint8` for PHL/NYC tiles. `src/dataset.py::compute_valid_mask`
casts to `int64` before the range comparison specifically to make the
resulting valid-pixel mask bit-identical regardless of source dtype
(see `tests/test_dataset.py::test_cls_dtype_float_and_uint8_give_identical_masks`).
