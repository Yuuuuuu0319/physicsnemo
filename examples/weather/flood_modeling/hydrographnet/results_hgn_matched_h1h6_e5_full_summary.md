# H1-H6 Matched HDF 5-Epoch Full-Dataloader Summary

This run extends the previous H1-H6 matched-HDF experiments beyond short
batch-limited training. Both checkpoints use the same H1-H6 split, full
sliding-window dataloader, seed 0, and 5 epochs.

Common setup:

- Train data: `/mnt/8tb_hdd2/joyce/train`
- Hydrograph ids: `train_h1h6.txt`
- HDF glob: `/mnt/8tb_hdd2/joyce/hecras-dataset/origin/Linux_RAS_v66/Minxiong_hgn_h1h6/outputs_seed/planH*/Minxiong.p01.hdf`
- Face graph: `/mnt/8tb_hdd2/joyce/Minxiong/hecras_hgn_face_graph_hgn_h1h6_check.npz`
- Face zone mode: `high`
- Face calibration: enabled

## Training

| Experiment | Face loss weight | Epochs | Final train total loss | Final MSE loss | Final physics loss | Final HEC-RAS face loss | Final Zone 3 RMSE | Final Zone 3 volume RMSE | Final Zone 3 WD RMSE | Checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| h1h6_w1e9_e5_full | 1e-9 | 5 | 1.0657e-04 | 9.0392e-05 | 0.0 | 1.6175e+04 | 1.1192e-02 | 6.3732e-03 | 1.4468e-02 | `checkpoints_hgn_h1h6_hecrasface_w1e9_e5_full` |
| h1h6_noface_e5_full | 0 | 5 | 8.6494e-05 | 8.6494e-05 | 0.0 |  | 1.1081e-02 | 8.2130e-03 | 1.3300e-02 | `checkpoints_hgn_h1h6_noface_e5_full` |

## Matched-HDF Face Residual

Lower is better. These metrics use event-matched HEC-RAS `Face Velocity`.

| Rollout | Experiment | Pred face residual RMSE | Zone 3 pred face residual RMSE | Pred-vs-GT delta RMSE | Zone 3 pred-vs-GT delta RMSE |
| ---: | --- | ---: | ---: | ---: | ---: |
| 10 | h1h6_w1e9_e5_full | 171.384 | 146.728 | 1079.579 | 867.998 |
| 10 | h1h6_noface_e5_full | 186.735 | 186.645 | 1101.474 | 980.392 |
| 30 | h1h6_w1e9_e5_full | 174.998 | 159.459 | 2562.374 | 2287.319 |
| 30 | h1h6_noface_e5_full | 205.027 | 217.120 | 2653.242 | 2590.556 |
| 45 | h1h6_w1e9_e5_full | 173.564 | 162.255 | 3393.139 | 3112.987 |
| 45 | h1h6_noface_e5_full | 210.969 | 228.331 | 3635.216 | 3652.125 |

## Standard Rollout Metrics

These are HydroGraphNet WD/volume rollout metrics on H1-H6. Lower is better.

| Rollout | Experiment | Rollout RMSE | WD RMSE | Volume RMSE | Zone 3 WD RMSE | Zone 3 volume RMSE |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 30 | h1h6_w1e9_e5_full | 0.119769 | 0.130872 | 0.107413 | 0.183305 | 0.095883 |
| 30 | h1h6_noface_e5_full | 0.118525 | 0.125344 | 0.111222 | 0.169548 | 0.108594 |
| 45 | h1h6_w1e9_e5_full | 0.165420 | 0.185499 | 0.142238 | 0.255440 | 0.130494 |
| 45 | h1h6_noface_e5_full | 0.164496 | 0.175702 | 0.152385 | 0.235214 | 0.153094 |

## Interpretation

- Longer training strongly reduces the matched-HDF residual compared with the
  previous b600 run.
- The matched-HDF face-loss model consistently reduces HEC-RAS face residuals
  relative to the no-face baseline at 10, 30, and 45 rollout steps.
- The Zone 3 high-fidelity improvement is especially clear for face residuals.
- Standard WD/volume rollout shows a tradeoff:
  - face loss improves volume RMSE, including Zone 3 volume RMSE;
  - no-face baseline has slightly better WD RMSE.
- This supports the current research framing: the HEC-RAS face loss is helping
  local/volume conservation consistency, but it is not automatically the best
  standalone water-depth predictor. A balanced objective may need a face-loss
  weight sweep or combination with zone-weighted supervised loss.

## Generated Result Files

- `results_hgn_matched_h1h6_face_residual_all_e5_full_len10.csv`
- `results_hgn_matched_h1h6_face_residual_all_e5_full_len30.csv`
- `results_hgn_matched_h1h6_face_residual_all_e5_full_len45.csv`
- `results_hgn_matched_h1h6_zone_rollout_e5_full_len30.csv`
- `results_hgn_matched_h1h6_zone_rollout_e5_full_len45.csv`
