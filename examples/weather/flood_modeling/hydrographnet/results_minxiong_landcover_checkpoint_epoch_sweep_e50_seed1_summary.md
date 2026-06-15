# Minxiong LandCover H13-H15 Checkpoint Sweep - Seed 1 / 50 Epochs

## Setup

- Train events: `H1-H12`
- Held-out test events: `H13-H15`
- Epochs: `0-49`
- Rollout length: `25`
- Training seed: `1`
- `num_training_samples`: `240`
- HEC-RAS source:
  `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong/outputs/planH*/Minxiong.p01.hdf`
- Output CSV:
  `results_minxiong_landcover_h13h15_epoch_sweep_e50_seed1.csv`

## Checkpoints

| Name | Checkpoint directory | Conservation setting |
|---|---|---|
| `noface_e50_seed1` | `checkpoints_minxiong_landcover_h1h12_noface_e50_seed1` | HydroGraphNet physics loss only |
| `cellbalance_all_w3e9_e50_seed1` | `checkpoints_minxiong_landcover_h1h12_cellbalance_all_w3e9_e50_seed1` | Native HEC-RAS `Cell Flow Balance`, all zones |
| `cellbalance_high_w3e9_e50_seed1` | `checkpoints_minxiong_landcover_h1h12_cellbalance_high_w3e9_e50_seed1` | Native HEC-RAS `Cell Flow Balance`, high-fidelity Zone 3 only |

## Best Metrics Across All Seed 1 Checkpoints

| Metric | Best checkpoint | Epoch | Value |
|---|---:|---:|---:|
| One-step MSE | `cellbalance_all_w3e9_e50_seed1` | 48 | `3.427456e-05` |
| Global formal cell-balance RMSE | `cellbalance_all_w3e9_e50_seed1` | 42 | `300.928 ft^3` |
| Zone 3 formal cell-balance RMSE | `cellbalance_high_w3e9_e50_seed1` | 47 | `136.551 ft^3` |
| 25-step rollout RMSE | `cellbalance_all_w3e9_e50_seed1` | 48 | `0.060910` |
| 25-step rollout volume RMSE | `cellbalance_all_w3e9_e50_seed1` | 42 | `0.048427` |
| 25-step rollout WD RMSE | `cellbalance_high_w3e9_e50_seed1` | 47 | `0.065050` |
| Zone 3 one-step volume RMSE | `cellbalance_all_w3e9_e50_seed1` | 45 | `0.002570` |
| Zone 3 one-step WD RMSE | `cellbalance_high_w3e9_e50_seed1` | 48 | `0.011220` |
| Zone 3 rollout volume RMSE | `cellbalance_all_w3e9_e50_seed1` | 45 | `0.032253` |
| Zone 3 rollout WD RMSE | `cellbalance_high_w3e9_e50_seed1` | 48 | `0.113761` |

## Best Rollout Checkpoint Per Branch

| Branch | Best epoch | Rollout RMSE | One-step MSE | Global cell-balance RMSE | Zone 3 cell-balance RMSE | Zone 3 rollout volume RMSE | Zone 3 rollout WD RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `noface_e50_seed1` | 49 | `0.102112` | `8.811070e-05` | `709.605 ft^3` | `838.479 ft^3` | `0.146871` | `0.165964` |
| `cellbalance_all_w3e9_e50_seed1` | 48 | `0.060910` | `3.427456e-05` | `340.462 ft^3` | `367.413 ft^3` | `0.046748` | `0.132438` |
| `cellbalance_high_w3e9_e50_seed1` | 48 | `0.080271` | `6.480511e-05` | `591.463 ft^3` | `172.433 ft^3` | `0.045106` | `0.113761` |

## Epoch 49 Comparison

| Branch | One-step MSE | Global cell-balance RMSE | Zone 3 cell-balance RMSE | Rollout RMSE | Rollout volume RMSE | Rollout WD RMSE | Zone 3 rollout volume RMSE | Zone 3 rollout WD RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `noface_e50_seed1` | `8.811070e-05` | `709.605 ft^3` | `838.479 ft^3` | `0.102112` | `0.116124` | `0.085786` | `0.146871` | `0.165964` |
| `cellbalance_all_w3e9_e50_seed1` | `4.318166e-05` | `323.111 ft^3` | `356.547 ft^3` | `0.062522` | `0.054293` | `0.069758` | `0.049585` | `0.124289` |
| `cellbalance_high_w3e9_e50_seed1` | `1.023740e-04` | `866.388 ft^3` | `295.155 ft^3` | `0.118629` | `0.140331` | `0.091877` | `0.058222` | `0.127708` |

## Interpretation

- Seed 1 repeats the main seed 0 pattern:
  - `cellbalance_all_w3e9` is the strongest branch for overall accuracy,
    global formal cell-balance residual, and 25-step rollout RMSE.
  - `cellbalance_high_w3e9` is the strongest branch for selective high-zone
    local-budget evidence, especially Zone 3 formal cell-balance RMSE and
    Zone 3 rollout WD RMSE.
- This supports the current paper framing:
  all-zone conservation is the best overall/global baseline, while
  selective high-zone conservation is useful when the research objective is
  local physical consistency in the high-fidelity region identified by
  Automatic Fidelity Zoning.
- The result should still not be written as a universal improvement over
  HydroGraphNet or DUALFloodGNN. It is a robustness check for the current
  zone-aware selective local-conservation claim under a second random seed.
