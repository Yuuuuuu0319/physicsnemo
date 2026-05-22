# New Train/Test Source-Term-Aware HydroGraphNet Experiments

Date: 2026-05-22

## Data

- Train dataset: `/mnt/8tb_hdd2/joyce/train`
- Train split file: `train.txt` (`H1`-`H10`)
- Test dataset: `/mnt/8tb_hdd2/joyce/test`
- Test split file: `test.txt` (`H11`-`H15`)
- HEC-RAS face graph: `/mnt/8tb_hdd2/joyce/Minxiong/hecras_hgn_face_graph.npz`

The new dataset now has nonzero precipitation and nonzero infiltration:

| Dataset | Hydrographs | Pr nonzero files | Pr max | IP nonzero nodes | IP max |
|---|---:|---:|---:|---:|---:|
| train | 15 | 15 | 75.930831909 | 12346 | 85.0 |
| test | 5 | 5 | 75.930831909 | 12346 | 85.0 |

Zone labels and weights were regenerated for both datasets using the existing
HEC-RAS multi-resolution area midpoint thresholds: `8200`, `5000`, `2600`.

## Event Matching

After the user replaced `/mnt/8tb_hdd2/joyce/Minxiong/Minxiong.p03.hdf`, the
HEC-RAS face graph was regenerated. Geometry still aligns with HGN node order:

| Item | Value |
|---|---:|
| HGN nodes | 12715 |
| HDF cells | 13429 |
| HDF faces | 29347 |
| Internal HGN faces | 28633 |
| Boundary / ghost faces | 714 |
| Face velocity shape | 239 x 29347 |
| Max coordinate distance | 6.7944e-10 |

The updated `Minxiong.p03.hdf` still does not match the new `train` / `test`
hydrographs by upstream inflow. Best `result_upstreamBC1` match was only
approximately:

| HGN ID | Split | Corr | z-NRMSE | Max abs diff | Exact/close |
|---|---|---:|---:|---:|---:|
| H1 | train | 0.479215 | 1.021e+00 | 1.825e+03 | 0 |

Therefore, HDF face velocity should still not be used as a final event-matched
training target for the new dataset. The safer branch is source-aware
geometry/target-gradient using HGN `Pr` and `IP`, without HDF face velocity.

Files:

- `results_hecras_hgn_event_match_new_train_test.csv`
- `results_hecras_hgn_event_match_new_train_test.md`
- `results_hecras_hgn_event_match_updated_hdf_train_test.csv`
- `results_hecras_hgn_event_match_updated_hdf_train_test.md`

## New Source-Aware Branch

Implemented source-aware true-face modes:

- `hecras_face_geometry_reference_mode=source_smooth`
- `hecras_face_geometry_reference_mode=source_target_gradient`

This mode estimates node-wise source volume rate from:

```text
local_source_rate = precipitation * cell_area * (IP / 100)
```

In `source_smooth`, the HEC-RAS true-face geometry loss subtracts
`local_source_rate * delta_t` from the predicted volume delta before smoothing
the source-corrected transport component. This keeps local rainfall/infiltration
source terms from being regularized as if they were transport.

In `source_target_gradient`, the source term is subtracted from both prediction
and target before matching gradients. Because the same source is subtracted on
both sides, this mode is effectively close to `target_gradient`; the branch is
kept for clarity, but `source_smooth` is the more source-sensitive short-run
test.

This branch remains separate from:

- `use_hecras_face_loss` / HDF face velocity branch.
- `edge_local_proxy` / kNN VX/VY proxy branch.
- node-zone weighted loss branch.

## Short Training

All runs used:

- Seed: `0`
- Train hydrographs: `H1`-`H10`
- Epochs: `1`
- Max train batches: `50`
- Device: GPU 1

| Run | Extra loss | Avg total loss | MSE loss | Physics loss | Extra loss value | Zone 3 WD train RMSE | Checkpoint |
|---|---|---:|---:|---:|---:|---:|---|
| baseline | none | 1.3376e-01 | 1.6281e-02 | 1.1748e-01 | - | 1.0159e-01 | `checkpoints_new_seed0_baseline_h1h10_b50` |
| zone1 | zone loss weight 1.0 | 1.4790e-01 | 1.4549e-02 | 1.1850e-01 | zone loss 1.4852e-02 | 9.3339e-02 | `checkpoints_new_seed0_zone1_h1h10_b50` |
| source_targetgrad | source-aware target gradient 1e-4 | 1.3386e-01 | 1.6280e-02 | 1.1747e-01 | geom loss 1.0304e+00 | 1.0158e-01 | `checkpoints_new_seed0_source_targetgrad_h1h10_b50` |
| source_smooth | source-aware transport smoothing 1e-4 | 1.3385e-01 | 1.6280e-02 | 1.1747e-01 | geom loss 1.0297e+00 | 1.0159e-01 | `checkpoints_new_seed0_source_smooth_h1h10_b50` |

## H11-H15 One-Step Evaluation

| Run | MSE | Zone loss | Zone 3 WD RMSE | Zone 3 Volume RMSE |
|---|---:|---:|---:|---:|
| baseline | 2.2140e-03 | 2.3968e-03 | 5.2549e-02 | 4.5361e-02 |
| zone1 | 2.1541e-03 | 2.1596e-03 | 5.5519e-02 | 3.7848e-02 |
| source_targetgrad | 2.2152e-03 | 2.3939e-03 | 5.2574e-02 | 4.5204e-02 |

File: `results_new_eval_seed0_h11h15.csv`

## H11-H15 10-Step Rollout

| Run | Rollout RMSE | WD RMSE | Volume RMSE | Zone 3 WD RMSE | Zone 3 Volume RMSE |
|---|---:|---:|---:|---:|---:|
| baseline | 2.4839e-01 | 2.8286e-01 | 2.0766e-01 | 2.9401e-01 | 2.3224e-01 |
| zone1 | 2.6151e-01 | 3.1112e-01 | 1.9923e-01 | 3.3315e-01 | 1.8985e-01 |
| source_targetgrad | 2.4845e-01 | 2.8314e-01 | 2.0743e-01 | 2.9425e-01 | 2.3138e-01 |
| source_smooth | 2.4843e-01 | 2.8306e-01 | 2.0748e-01 | 2.9419e-01 | 2.3164e-01 |

File: `results_new_rollout_seed0_h11h15_len10.csv`
Updated file with source_smooth: `results_new_rollout_seed0_h11h15_len10_with_source_smooth.csv`

## Source-Aware Face-Gradient Diagnostic

This diagnostic uses HGN `Pr/IP` and HEC-RAS true faces only. It does not use HDF
face velocity, so it remains valid while HDF event matching is unresolved.

| Run | Pred-vs-GT transport gradient RMSE | High-adjacent pred-vs-GT RMSE | Pred transport gradient RMSE |
|---|---:|---:|---:|
| baseline | 9.1217e-01 | 2.5959e+00 | 1.7738e-01 |
| zone1 | 9.0284e-01 | 2.5161e+00 | 1.7619e-01 |
| source_targetgrad | 9.1020e-01 | 2.5886e+00 | 1.7705e-01 |
| source_smooth | 9.1045e-01 | 2.5898e+00 | 1.7708e-01 |

File: `results_source_face_gradient_residual_seed0_h11h15_len10.csv`

## Interpretation

- The new `Pr/IP` files are usable: both precipitation and infiltration are
  nonzero, and source-aware true-face training runs successfully.
- `Minxiong.p03.hdf` face velocity is still not event-matched to the new data,
  so it remains diagnostic only.
- In this short 50-batch run, `source_targetgrad` and `source_smooth` both
  behave very close to baseline. They slightly improve Zone 3 rollout volume
  RMSE but not Zone 3 WD.
- `zone1` improves one-step MSE and Zone 3 volume, but its 10-step rollout WD
  worsens in this new split. This should not be treated as a final conclusion
  until longer training and multiple seeds are run.
- The source-aware face-gradient diagnostic currently favors `zone1` on
  high-adjacent pred-vs-GT residual, while source branches remain close to
  baseline. This suggests the source-aware branch is wired correctly but needs
  stronger/longer training or a refined loss before it can be claimed as an
  improvement.

## Next Step

- Run longer training for `baseline`, `zone1`, and `source_targetgrad` on
  `H1`-`H10`, then re-evaluate H11-H15.
- Try longer training and a weight sweep for `source_smooth`, for example
  `1e-5`, `1e-4`, `1e-3`, because the current 50-batch run is too small to
  determine the useful scale.
- If a new HDF result is generated for the same H1-H15 events, rerun event
  matching before enabling any HDF face velocity loss.
