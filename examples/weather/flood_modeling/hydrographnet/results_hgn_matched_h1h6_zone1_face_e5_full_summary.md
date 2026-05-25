# Matched H1-H6 Zone + HEC-RAS Face Loss e5 Full Summary

Date: 2026-05-24

## Setup

- Dataset: `/mnt/8tb_hdd2/joyce/train`
- Hydrograph IDs: `train_h1h6.txt`
- Events: H1-H6
- Epochs: 5
- Samples: full available dataloader target, `num_training_samples=600`
- Device: GPU 1, `CUDA_VISIBLE_DEVICES=1`
- HEC-RAS matched event HDFs: `/mnt/8tb_hdd2/joyce/hecras-dataset/origin/Linux_RAS_v66/Minxiong_hgn_h1h6/outputs_seed/planH*/Minxiong.p01.hdf`
- Face graph: `/mnt/8tb_hdd2/joyce/Minxiong/hecras_hgn_face_graph_hgn_h1h6_check.npz`
- Main new checkpoint: `checkpoints_hgn_h1h6_zone1_hecrasface_w1e9_e5_full`

## Compared Checkpoints

- `h1h6_zone1_w1e9_e5_full`: zone loss enabled, `zone_loss_weight=1.0`, HEC-RAS face loss enabled, `hecras_face_loss_weight=1e-9`
- `h1h6_zone1_e5_full`: zone loss enabled, `zone_loss_weight=1.0`, no HEC-RAS face loss
- `h1h6_w1e9_e5_full`: zone metrics logged but zone loss disabled, HEC-RAS face loss enabled, `hecras_face_loss_weight=1e-9`
- `h1h6_noface_e5_full`: zone metrics logged, no HEC-RAS face loss

## Training Result For Zone + Face

Final epoch 4:

- total loss: `1.4850e-04`
- mse loss: `6.8217e-05`
- physics loss: `0.0000e+00`
- zone loss: `6.8496e-05`
- hecras face loss: `1.1789e+04`
- Zone 3 RMSE: `8.8409e-03`
- Zone 3 volume RMSE: `5.2758e-03`
- Zone 3 water-depth RMSE: `1.1296e-02`

Compared with the earlier e5 full training runs, adding zone loss produced the best final training MSE and the lowest Zone 3 training volume/water-depth errors.

## Training Result For Zone-Only

Final epoch 4:

- total loss: `1.3494e-04`
- mse loss: `6.7056e-05`
- physics loss: `0.0000e+00`
- zone loss: `6.7882e-05`
- Zone 3 RMSE: `8.8424e-03`
- Zone 3 volume RMSE: `6.2986e-03`
- Zone 3 water-depth RMSE: `1.0743e-02`

Zone-only reached the lowest final training MSE and Zone 3 water-depth training error, but its Zone 3 volume training error was worse than zone + face.

## Face Residual Evaluation

Lower is better. The face residual metric evaluates local conservation consistency against matched event-specific HEC-RAS face velocities.

| rollout length | checkpoint | pred face residual RMSE | Zone 3 pred face residual RMSE |
|---:|---|---:|---:|
| 10 | zone + face | 174.9773 | 148.9561 |
| 10 | face-only | 171.3836 | 146.7281 |
| 10 | noface | 186.7349 | 186.6452 |
| 30 | zone + face | 188.8212 | 175.8947 |
| 30 | zone-only | 223.8250 | 232.1034 |
| 30 | face-only | 174.9978 | 159.4593 |
| 30 | noface | 205.0273 | 217.1199 |
| 45 | zone + face | 188.3969 | 180.8951 |
| 45 | zone-only | 240.9693 | 262.4484 |
| 45 | face-only | 173.5640 | 162.2554 |
| 45 | noface | 210.9691 | 228.3314 |

Interpretation: HEC-RAS face loss clearly improves local conservation. Zone-only has the weakest face residual among the four-checkpoint ablation, which confirms that zone supervision alone does not enforce HEC-RAS-style face conservation. The pure face-loss model remains best on the face residual metric, while zone + face keeps much of the conservation gain and improves rollout prediction quality.

## Rollout Evaluation

Lower is better. These are normal HydroGraphNet multi-step rollout prediction metrics on H1-H6.

| rollout length | checkpoint | rollout RMSE | volume RMSE | water-depth RMSE | Zone 3 RMSE | Zone 3 volume RMSE | Zone 3 water-depth RMSE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 30 | zone + face | 0.106580 | 0.096383 | 0.115805 | 0.111646 | 0.075074 | 0.138890 |
| 30 | zone-only | 0.106440 | 0.108519 | 0.104222 | 0.109426 | 0.088261 | 0.126981 |
| 30 | face-only | 0.119769 | 0.107413 | 0.130872 | 0.146291 | 0.095883 | 0.183305 |
| 30 | noface | 0.118525 | 0.111222 | 0.125344 | 0.142424 | 0.108594 | 0.169548 |
| 45 | zone + face | 0.147471 | 0.130277 | 0.162737 | 0.154200 | 0.103062 | 0.192158 |
| 45 | zone-only | 0.146923 | 0.146650 | 0.147044 | 0.153289 | 0.125086 | 0.176873 |
| 45 | face-only | 0.165420 | 0.142238 | 0.185499 | 0.202848 | 0.130494 | 0.255440 |
| 45 | noface | 0.164496 | 0.152385 | 0.175702 | 0.198527 | 0.153094 | 0.235214 |

Interpretation: zone-only is slightly best for global RMSE and water-depth RMSE, while zone + face is best for volume RMSE, especially in Zone 3. Because the research target includes flood-volume consistency and local conservation, zone + face is the stronger current candidate for the physics-aware model. Zone-only remains useful as the accuracy-focused ablation.

## Current Conclusion

The best current configuration for prediction accuracy is:

`zone_loss_weight=1.0`

The best current configuration for volume-aware rollout accuracy is:

`zone_loss_weight=1.0 + hecras_face_loss_weight=1e-9 + matched event-specific HDF face velocities`

The best current configuration for the HEC-RAS face residual metric alone is:

`zone_loss_weight=0.0 + hecras_face_loss_weight=1e-9`

This supports keeping the losses separated in the code and reporting them as different objectives: zone loss improves multi-resolution rollout accuracy, face loss improves local conservation, and the combined model is currently the best candidate when water volume and physics consistency matter.

## Output Files

- `results_hgn_matched_h1h6_face_residual_all_e5_full_zone1_len10.csv`
- `results_hgn_matched_h1h6_face_residual_all_e5_full_zone1_len30.csv`
- `results_hgn_matched_h1h6_face_residual_all_e5_full_zone1_len45.csv`
- `results_hgn_matched_h1h6_zone_rollout_e5_full_zone1_len30.csv`
- `results_hgn_matched_h1h6_zone_rollout_e5_full_zone1_len45.csv`
- `results_hgn_matched_h1h6_face_residual_all_e5_full_ablation_len30.csv`
- `results_hgn_matched_h1h6_face_residual_all_e5_full_ablation_len45.csv`
- `results_hgn_matched_h1h6_zone_rollout_e5_full_ablation_len30.csv`
- `results_hgn_matched_h1h6_zone_rollout_e5_full_ablation_len45.csv`
- `results_hgn_matched_h1h6_zone1_face_e5_full_summary.csv`
