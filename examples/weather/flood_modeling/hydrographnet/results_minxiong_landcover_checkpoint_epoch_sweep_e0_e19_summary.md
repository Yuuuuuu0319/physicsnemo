# Minxiong LandCover Checkpoint Epoch Sweep

Date: 2026-06-06

## Purpose

This sweep checks whether the 20-epoch results are an artifact of using the
latest checkpoint. It evaluates every saved epoch from 0 to 19 for the current
three core ablation groups:

- `noface`
- `cellbalance_all_w3e9`
- `cellbalance_high_w3e9`

The evaluator reports one-step zone metrics, formal HEC-RAS native
cell-balance residuals, and 25-step rollout zone metrics in one table.

Source CSV:

`results_minxiong_landcover_h13h15_epoch_sweep_e0_e19.csv`

Smoke check CSV:

`results_minxiong_landcover_h13h15_epoch_sweep_smoke_e19.csv`

One-step rerun check:

`results_minxiong_landcover_h13h15_zone_eval_rerun_check.csv`

## Best Epoch Across All Groups

| Metric | Best checkpoint | Epoch | Value |
|---|---|---:|---:|
| One-step MSE | `cellbalance_all_w3e9` | 19 | `8.358352e-05` |
| One-step Zone 3 volume RMSE | `cellbalance_high_w3e9` | 19 | `3.339335e-03` |
| One-step Zone 3 WD RMSE | `cellbalance_all_w3e9` | 19 | `1.516730e-02` |
| Global cell-balance RMSE ft3 | `cellbalance_all_w3e9` | 19 | `443.137` |
| Zone 3 cell-balance RMSE ft3 | `cellbalance_high_w3e9` | 19 | `299.977` |
| Rollout25 overall RMSE | `cellbalance_all_w3e9` | 19 | `0.103100` |
| Rollout25 volume RMSE | `cellbalance_all_w3e9` | 15 | `0.062836` |
| Rollout25 WD RMSE | `cellbalance_high_w3e9` | 19 | `0.126067` |
| Zone 3 rollout25 volume RMSE | `cellbalance_high_w3e9` | 19 | `0.039164` |
| Zone 3 rollout25 WD RMSE | `cellbalance_all_w3e9` | 17 | `0.166884` |

## Best Epoch Per Checkpoint

### `noface`

| Metric | Best epoch | Value |
|---|---:|---:|
| One-step MSE | 19 | `1.286266e-04` |
| Zone 3 cell-balance RMSE ft3 | 19 | `600.861` |
| Rollout25 overall RMSE | 16 | `0.124654` |
| Zone 3 rollout25 volume RMSE | 19 | `0.084840` |

### `cellbalance_all_w3e9`

| Metric | Best epoch | Value |
|---|---:|---:|
| One-step MSE | 19 | `8.358352e-05` |
| Global cell-balance RMSE ft3 | 19 | `443.137` |
| Zone 3 cell-balance RMSE ft3 | 19 | `431.563` |
| Rollout25 overall RMSE | 19 | `0.103100` |
| Zone 3 rollout25 volume RMSE | 16 | `0.060374` |

### `cellbalance_high_w3e9`

| Metric | Best epoch | Value |
|---|---:|---:|
| One-step MSE | 19 | `1.154251e-04` |
| Global cell-balance RMSE ft3 | 19 | `643.716` |
| Zone 3 cell-balance RMSE ft3 | 19 | `299.977` |
| Rollout25 overall RMSE | 19 | `0.109055` |
| Zone 3 rollout25 volume RMSE | 19 | `0.039164` |

## Interpretation

The final epoch is not an arbitrary weak checkpoint for the current 20-epoch
runs. Most best metrics occur at epoch 19, and the key selective-local result
remains true under epoch selection:

- `cellbalance_all_w3e9` is still the strongest overall/global model.
- `cellbalance_high_w3e9` is still the strongest high-fidelity-zone volume and
  local-budget model.

This supports the current research framing:

Selective high-zone conservation is not the global error minimizer, but it
concentrates physical-budget improvement in the high-fidelity region selected
by Automatic Fidelity Zoning.

## Next Step

The next fair robustness step is longer training with the same three groups,
for example 50 epochs:

- `checkpoints_minxiong_landcover_h1h12_noface_e50`
- `checkpoints_minxiong_landcover_h1h12_cellbalance_all_w3e9_e50`
- `checkpoints_minxiong_landcover_h1h12_cellbalance_high_w3e9_e50`

After 50-epoch checkpoints are available, rerun the same epoch-sweep evaluator
and compare best-epoch metrics rather than latest-only metrics.
