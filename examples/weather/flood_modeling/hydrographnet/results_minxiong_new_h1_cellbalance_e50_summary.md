# Minxiong New H1 Formal Cell-Balance E50 Summary

Date: 2026-06-03

## Scope

This experiment uses the corrected single-event H1 dataset exported from
`/mnt/8tb_hdd2/joyce/Minxiong/Minxiong_new`.

Training data:
`/mnt/8tb_hdd2/joyce/Minxiong/Minxiong_new/hgn_extract_workspace/outputs_native_30min_area_extrap/HGN_dataset`

Formal HDF budget source:
`/mnt/8tb_hdd2/joyce/Minxiong/Minxiong_new/hgn_extract_workspace/outputs_native/planH*/Minxiong.p01.hdf`

The dataset uses area-above-table extrapolated `M80_V`, and the formal loss
uses native HEC-RAS `Cell Flow Balance` integrated over each HGN interval plus
native precipitation volume. No fitted scale factor is used.

## Checkpoints

| Checkpoint | Loss branch | Epochs |
|---|---|---:|
| `checkpoints_minxiong_new_h1_noface_e50` | baseline noface | 50 |
| `checkpoints_minxiong_new_h1_cellbalance_w1e9_e50` | formal cell-balance, weight `1e-9`, zone mode `all` | 50 |

## Training-End Metrics

| Metric | noface E50 | cell-balance E50 |
|---|---:|---:|
| Final total loss | `1.6696e-03` | `2.1803e-03` |
| Final MSE loss | `3.4212e-04` | `2.6576e-04` |
| Final physics loss | `1.3275e-03` | `1.3750e-03` |
| Final formal cell-balance loss | N/A | `5.3959e+05` |
| Zone 3 volume RMSE | `1.3474e-02` | `1.1245e-02` |
| Zone 3 water-depth RMSE | `2.3055e-02` | `2.2585e-02` |

The formal branch reduced its own raw `hecras_cell_balance_loss` from
`8.0855e+07` at epoch 0 to `5.3959e+05` at epoch 49.

## One-Step H1 Evaluation

Result file:
`results_minxiong_new_h1_cellbalance_e50_eval.csv`

| Metric | noface E50 | cell-balance E50 |
|---|---:|---:|
| One-step MSE | `3.9273e-04` | `3.4993e-04` |
| Zone 0 volume RMSE | `2.0615e-02` | `1.2553e-02` |
| Zone 1 volume RMSE | `1.3654e-02` | `1.5345e-02` |
| Zone 2 volume RMSE | `1.4008e-02` | `1.5549e-02` |
| Zone 3 volume RMSE | `1.2326e-02` | `1.5613e-02` |

This evaluation is on the same H1 event because only H1 currently has a
budget-ready event HDF in `Minxiong_new`. Treat it as a proof-of-pipeline
comparison, not as held-out generalization evidence.

## Formal Budget Residual Evaluation

Result file:
`results_minxiong_new_h1_cellbalance_e50_budget.csv`

| Metric | noface E50 | cell-balance E50 |
|---|---:|---:|
| Samples | `39` | `39` |
| Prediction MSE | `3.4177e-04` | `2.8560e-04` |
| Cell-balance RMSE | `984.7777 ft^3` | `842.7720 ft^3` |
| Cell-balance MAE | `765.6130 ft^3` | `669.2209 ft^3` |
| Relative cell-balance RMSE | `1.8812` | `1.6099` |
| Zone 3 cell-balance RMSE | `913.2072 ft^3` | `890.9314 ft^3` |

The formal cell-balance branch improved the overall HEC-RAS budget residual
while preserving or improving the one-step prediction MSE on this H1 training
event. Zone 3 budget residual improved only modestly, so multi-event validation
is still needed before making a strong claim.

## Current Limitation

Only H1 in `Minxiong_new` currently has the required budget-ready HDF with
30-second native output and native `Cell Flow Balance`. The older
`/mnt/8tb_hdd2/joyce/Minxiong/outputs/planH1-H6` HDFs are not suitable for
formal local conservation because they store 30-minute output and do not contain
native `Face Flow` or `Cell Flow Balance`.

Next step: generate/export budget-ready H2-H6 and held-out test events with the
same HEC-RAS output settings, then repeat this comparison on multi-event train
and held-out test splits.
