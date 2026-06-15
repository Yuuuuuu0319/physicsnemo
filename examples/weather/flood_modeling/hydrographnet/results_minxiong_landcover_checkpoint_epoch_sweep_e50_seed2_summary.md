# Minxiong LandCover H13-H15 Checkpoint Sweep - Seed 2 / 50 Epochs

## Setup

- Train events: `H1-H12`
- Held-out test events: `H13-H15`
- Epochs: `0-49`
- Rollout length: `25`
- Training seed: `2`
- `num_training_samples`: `240`
- HEC-RAS source:
  `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong/outputs/planH*/Minxiong.p01.hdf`
- Output CSV:
  `results_minxiong_landcover_h13h15_epoch_sweep_e50_seed2.csv`
- Execution note: these seed 2 checkpoints were rerun in the GPU-visible
  environment with `CUDA_VISIBLE_DEVICES=0`. Earlier sandbox/CPU partial
  checkpoints were moved aside and are not used in this summary.

## Checkpoints

| Name | Checkpoint directory | Conservation setting |
|---|---|---|
| `noface_e50_seed2` | `checkpoints_minxiong_landcover_h1h12_noface_e50_seed2` | HydroGraphNet physics loss only |
| `cellbalance_all_w3e9_e50_seed2` | `checkpoints_minxiong_landcover_h1h12_cellbalance_all_w3e9_e50_seed2` | Native HEC-RAS `Cell Flow Balance`, all zones |
| `cellbalance_high_w3e9_e50_seed2` | `checkpoints_minxiong_landcover_h1h12_cellbalance_high_w3e9_e50_seed2` | Native HEC-RAS `Cell Flow Balance`, high-fidelity Zone 3 only |

## Best Metrics Across All Seed 2 Checkpoints

| Metric | Best checkpoint | Epoch | Value |
|---|---:|---:|---:|
| One-step MSE | `cellbalance_all_w3e9_e50_seed2` | 44 | `2.859441e-05` |
| Global formal cell-balance RMSE | `cellbalance_all_w3e9_e50_seed2` | 49 | `266.674 ft^3` |
| Zone 3 formal cell-balance RMSE | `cellbalance_high_w3e9_e50_seed2` | 44 | `167.203 ft^3` |
| 25-step rollout RMSE | `cellbalance_all_w3e9_e50_seed2` | 45 | `0.052871` |
| 25-step rollout volume RMSE | `cellbalance_all_w3e9_e50_seed2` | 45 | `0.036701` |
| 25-step rollout WD RMSE | `cellbalance_all_w3e9_e50_seed2` | 44 | `0.064961` |
| Zone 3 one-step volume RMSE | `cellbalance_high_w3e9_e50_seed2` | 40 | `0.001862` |
| Zone 3 one-step WD RMSE | `noface_e50_seed2` | 27 | `0.012471` |
| Zone 3 rollout volume RMSE | `cellbalance_high_w3e9_e50_seed2` | 40 | `0.022568` |
| Zone 3 rollout WD RMSE | `noface_e50_seed2` | 27 | `0.123689` |

## Best Rollout Checkpoint Per Branch

| Branch | Best epoch | Rollout RMSE | One-step MSE | Global cell-balance RMSE | Zone 3 cell-balance RMSE | Zone 3 rollout volume RMSE | Zone 3 rollout WD RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `noface_e50_seed2` | 47 | `0.081478` | `7.177839e-05` | `587.129 ft^3` | `528.465 ft^3` | `0.078567` | `0.140076` |
| `cellbalance_all_w3e9_e50_seed2` | 45 | `0.052871` | `2.943424e-05` | `272.728 ft^3` | `286.423 ft^3` | `0.029452` | `0.141417` |
| `cellbalance_high_w3e9_e50_seed2` | 44 | `0.076220` | `5.936272e-05` | `510.923 ft^3` | `167.203 ft^3` | `0.030570` | `0.145533` |

## Epoch 49 Comparison

| Branch | One-step MSE | Global cell-balance RMSE | Zone 3 cell-balance RMSE | Rollout RMSE | Rollout volume RMSE | Rollout WD RMSE | Zone 3 rollout volume RMSE | Zone 3 rollout WD RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `noface_e50_seed2` | `7.408559e-05` | `491.021 ft^3` | `318.129 ft^3` | `0.083512` | `0.073354` | `0.061074` | `0.038637` | `0.142661` |
| `cellbalance_all_w3e9_e50_seed2` | `3.589339e-05` | `266.674 ft^3` | `278.843 ft^3` | `0.056562` | `0.045886` | `0.070043` | `0.028024` | `0.146702` |
| `cellbalance_high_w3e9_e50_seed2` | `8.381626e-05` | `624.706 ft^3` | `214.874 ft^3` | `0.085997` | `0.068088` | `0.076379` | `0.026799` | `0.157459` |

## Interpretation

- Seed 2 strengthens the most conservative conclusion:
  `cellbalance_all_w3e9` is the best branch for overall H13-H15 25-step
  rollout, one-step MSE, and global formal cell-balance residual.
- Seed 2 also supports the selective local-conservation mechanism:
  `cellbalance_high_w3e9` is best for Zone 3 formal cell-balance RMSE and
  Zone 3 one-step/rollout volume metrics.
- The water-depth claim must remain cautious:
  unlike seeds 0 and 1, seed 2 has the best Zone 3 rollout WD RMSE in
  `noface_e50_seed2`. This means the current selective high-zone method is
  robust for formal local-budget closure, but not yet robust for every Zone 3
  rollout accuracy metric.
