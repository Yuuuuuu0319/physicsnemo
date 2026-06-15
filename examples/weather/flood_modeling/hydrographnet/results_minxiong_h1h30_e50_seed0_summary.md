# Minxiong H1-H30 E50 Seed0 Summary

Date: 2026-06-12

## Dataset

- HEC-RAS working folder: `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611`
- HGN dataset: `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset`
- Events generated: H1-H30, all successful.
- HGN conversion: 30 events, 12715 fixed cells, 121 timesteps per event, 30 min output interval.
- Split:
  - Train: `train_h1h24.txt`
  - Test: `test_h25h30.txt`
- Static zone files copied from `/mnt/8tb_hdd2/joyce/train`:
  - `zone_label.txt`
  - `zone_weight.txt`
  - `zone_weight_full.txt`
  - `zone_summary.json`

## Training Runs

| Run | Checkpoint | Epochs | Train events | Samples | Notes |
|---|---|---:|---|---:|---|
| noface | `checkpoints_minxiong_h1h30_noface_e50_seed0` | 50 | H1-H24 | 1200 | Baseline with HydroGraphNet global physics loss only. |
| all-zone cell-balance | `checkpoints_minxiong_h1h30_cellbalance_all_w3e9_e50_seed0` | 50 | H1-H24 | 1200 | Adds HEC-RAS Cell Flow Balance loss on all zones, weight `3e-9`. |

## Final Training Metrics

| Run | Epoch | MSE | Physics loss | Cell-balance loss | Total loss | Zone3 volume RMSE | Zone3 WD RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| noface | 49 | 3.1345e-05 | 1.3295e-03 | NA | 1.3608e-03 | NA | NA |
| all-zone cell-balance | 49 | 1.4497e-05 | 1.3720e-03 | 3.4143e+04 | 1.4890e-03 | 1.2944e-03 | 9.0928e-03 |

## H25-H30 Evaluation

Evaluation used `rollout_length=10`. `rollout_length=25` failed because the dataset rollout sampling for H26 did not leave enough remaining timesteps for that rollout length.

| Metric | noface | all-zone cell-balance | Relative change |
|---|---:|---:|---:|
| One-step MSE | 3.633999e-05 | 1.251145e-05 | -65.6% |
| Global cell-balance RMSE (ft^3) | 384.037 | 185.819 | -51.6% |
| Zone3 cell-balance RMSE (ft^3) | 337.094 | 245.204 | -27.3% |
| Rollout RMSE | 0.029554 | 0.016024 | -45.8% |
| Rollout WD RMSE | 0.027771 | 0.018145 | -34.7% |
| Rollout volume RMSE | 0.031226 | 0.013554 | -56.6% |
| Zone3 rollout WD RMSE | 0.055544 | 0.033510 | -39.7% |
| Zone3 rollout volume RMSE | 0.020569 | 0.008648 | -58.0% |

## Interpretation

The new H1-H30 dataset gives a cleaner train/test separation than the earlier H1-H15 setup: H1-H24 are used for training, while H25-H30 are unseen events. On this first seed0 comparison, all-zone Cell Flow Balance training improves both prediction error and conservation-related residuals on the unseen events.

This is a stronger result than the earlier small-event experiments because the test events are no longer part of the original H1-H15 pool. The next required step is to repeat the same H1-H24/H25-H30 evaluation for selective high-zone cell-balance training and additional seeds before treating the conclusion as robust.
