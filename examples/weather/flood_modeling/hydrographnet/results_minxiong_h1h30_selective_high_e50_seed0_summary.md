# Minxiong H1-H30 Selective High-Zone Cell-Balance Result

Date: 2026-06-13

## Dataset

- HEC-RAS working folder:
  `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611`
- HGN dataset:
  `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset`
- Train split: `train_h1h24.txt`
- Test split: `test_h25h30.txt`
- Events: H1-H30
- Nodes: 12715 fixed active cells
- Timesteps: 121 per event
- Evaluation rollout length: 10 steps

## New Run

Added the missing H1-H30 selective high-zone branch:

```text
checkpoints_minxiong_h1h30_cellbalance_high_w3e9_e50_seed0
```

Training settings:

- epochs: `50`
- seed: `0`
- num training samples: `1200`
- HEC-RAS Cell Flow Balance loss weight: `3e-9`
- HEC-RAS Cell Flow Balance zone mode: `high`
- HEC-RAS source:
  `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/planH*/Minxiong.p01.hdf`

Final training epoch 49:

| Metric | Value |
|---|---:|
| Total loss | `1.4155e-03` |
| MSE loss | `4.1090e-05` |
| Physics loss | `1.3430e-03` |
| HEC-RAS cell-balance loss | `1.0485e+04` |
| Zone 3 volume RMSE | `3.2482e-03` |
| Zone 3 WD RMSE | `1.1785e-02` |

## H25-H30 Epoch 49 Three-Way Evaluation

Output CSV:

```text
results_minxiong_h1h30_h25h30_epoch49_threeway_eval_e50_seed0_len10.csv
```

| Metric | noface | all-zone cell-balance | high-zone cell-balance |
|---|---:|---:|---:|
| One-step MSE | `3.584949e-05` | `1.251145e-05` | `5.912313e-05` |
| Global cell-balance RMSE (ft^3) | `384.037` | `185.819` | `520.312` |
| Zone 3 cell-balance RMSE (ft^3) | `337.094` | `245.204` | `91.162` |
| Rollout RMSE | `0.029554` | `0.016024` | `0.038119` |
| Zone 3 rollout WD RMSE | `0.055544` | `0.033510` | `0.058703` |
| Zone 3 rollout volume RMSE | `0.020569` | `0.008648` | `0.014970` |

Interpretation:

- `all-zone cell-balance` is still the strongest overall model on H25-H30 at
  epoch 49.
- `high-zone cell-balance` is not better for global metrics or rollout metrics
  at epoch 49.
- `high-zone cell-balance` gives the strongest formal high-fidelity Zone 3
  local-budget closure:
  - vs noface: `337.094 -> 91.162 ft^3`
  - vs all-zone: `245.204 -> 91.162 ft^3`

This supports a narrower but important claim:

```text
Selective high-zone local conservation can strongly improve local physical
budget consistency inside the high-fidelity zone, but it does not automatically
improve global rollout accuracy.
```

## High-Zone Checkpoint Sweep

Output CSV:

```text
results_minxiong_h1h30_h25h30_high_epoch_sweep_e50_seed0_len10.csv
```

Best high-zone checkpoints:

| Selection metric | Best epoch | Value |
|---|---:|---:|
| One-step MSE | `35` | `4.131872e-05` |
| Global cell-balance RMSE (ft^3) | `35` | `433.685` |
| Zone 3 cell-balance RMSE (ft^3) | `46` | `83.595` |
| Rollout RMSE | `35` | `0.031340` |
| Zone 3 rollout WD RMSE | `43` | `0.049694` |
| Zone 3 rollout volume RMSE | `26` | `0.008063` |

Interpretation:

- If the paper selects checkpoint by overall rollout accuracy, high-zone should
  use epoch `35`, but it still does not beat all-zone epoch 49 rollout RMSE.
- If the paper selects checkpoint by high-fidelity local conservation, high-zone
  should use epoch `46`, because it gives the best Zone 3 formal cell-balance
  RMSE.
- This means checkpoint selection must be reported transparently. The high-zone
  branch should not be described as an overall accuracy winner; it is currently
  a targeted high-fidelity local-conservation branch.

## Research Consequence

The H1-H30 result sharpens the current paper direction:

- The strongest global/local-all model is `all-zone cell-balance`.
- The strongest high-fidelity local-budget model is `high-zone cell-balance`.
- The selective method is promising as a zone-targeted physical regularizer, but
  the present evidence is not enough to claim that selective local conservation
  improves every metric.
- The next paper-level step is still the formal edge-informed pipeline:
  reconstruct HEC-RAS face/edge flux, map it to HGN nodes, validate divergence
  against native Cell Flow Balance, and only then train an edge-informed loss.
