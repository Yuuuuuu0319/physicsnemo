# Zone-Aware Fidelity Experiment - Seed 0, H1-H3, 50 Batches

This run is intentionally limited to node-level fidelity zoning. It does not add
edge features, an edge head, or velocity-based local flux terms.

## Setup

- Dataset: `/mnt/8tb_hdd2/joyce/HGN_train`
- Hydrographs: `train_h1_h3.txt` (`H1`, `H2`, `H3`)
- GPU: `CUDA_VISIBLE_DEVICES=1` (NVIDIA RTX A6000)
- Seed: `0`
- Epochs: `1`
- Max train batches: `50`
- Physics loss: enabled
- Zone metrics: enabled

## Fidelity Zones

| Zone | Meaning | Weight | Node Count |
|---:|---|---:|---:|
| 0 | Low fidelity, area >= 8000 | 0.0 | 6292 |
| 1 | 5000 <= area < 8000 | 0.25 | 3083 |
| 2 | 2500 <= area < 5000 | 0.5 | 2125 |
| 3 | High fidelity, area < 2500 | 1.0 | 1215 |

## Results

| Run | zone_loss_weight | total_loss | mse_loss | physics_loss | zone_loss | zone_3_rmse | zone_3_wd_rmse | zone_3_volume_rmse | Checkpoint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline | 0.0 | 1.3408e-01 | 1.3094e-02 | 1.2099e-01 | n/a | 9.4146e-02 | 9.6224e-02 | 8.4447e-02 | `checkpoints_seed0_baseline_h1h3_b50/checkpoint.0.0.pt` |
| Zone 0.25 | 0.25 | 1.3710e-01 | 1.2668e-02 | 1.2095e-01 | 1.3919e-02 | 8.8623e-02 | 8.6373e-02 | 8.4681e-02 | `checkpoints_seed0_zone025_h1h3_b50/checkpoint.0.0.pt` |
| Zone 1.0 | 1.0 | 1.4572e-01 | 1.2154e-02 | 1.2046e-01 | 1.3112e-02 | 8.3596e-02 | 7.6369e-02 | 8.5029e-02 | `checkpoints_seed0_zone1_h1h3_b50/checkpoint.0.0.pt` |

## Early Read

- `zone_loss_weight=1.0` gives the best high-fidelity zone WD RMSE in this short
  training slice.
- Zone-aware loss slightly increases high-fidelity zone volume RMSE in this run.
- Physics loss is similar across all three runs, with a small decrease as zone
  weight increases.
- These are training-batch metrics, not rollout validation metrics. The next
  step should evaluate saved checkpoints on held-out hydrographs or rollout
  sequences.

## Next Candidates

- Run longer with `max_train_batches=150` and compare `zone_loss_weight=0.25`,
  `0.5`, and `1.0`.
- Add a separate evaluation script that loads checkpoints and reports per-zone
  metrics on fixed held-out hydrographs.
- Keep edge-informed local conservation separate from this node-zone branch.

## Held-Out One-Step Evaluation

Evaluation script:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/mnt/8tb_hdd2/joyce/physicsnemo:/mnt/8tb_hdd2/joyce/.local/lib/python3.10/site-packages:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages /mnt/8tb_hdd2/joyce/conda-envs/yutest/bin/python evaluate_zone_checkpoints.py --data-dir /mnt/8tb_hdd2/joyce/HGN_train --eval-ids-file eval_h4_h6.txt --output-csv results_zone_eval_seed0_h4h6.csv --checkpoint baseline checkpoints_seed0_baseline_h1h3_b50 --checkpoint zone025 checkpoints_seed0_zone025_h1h3_b50 --checkpoint zone1 checkpoints_seed0_zone1_h1h3_b50
```

Held-out hydrographs: `H4`, `H5`, `H6`

Output CSV: `results_zone_eval_seed0_h4h6.csv`

| Checkpoint | MSE | Zone Loss | Zone 3 RMSE | Zone 3 WD RMSE | Zone 3 Volume RMSE |
|---|---:|---:|---:|---:|---:|
| baseline | 2.4669e-03 | 2.6295e-03 | 4.3744e-02 | 5.6606e-02 | 2.4951e-02 |
| zone025 | 2.0569e-03 | 2.2800e-03 | 4.2470e-02 | 5.3499e-02 | 2.7289e-02 |
| zone1 | 1.7007e-03 | 1.7713e-03 | 3.8960e-02 | 4.6999e-02 | 2.8743e-02 |

Early held-out read:

- `zone_loss_weight=1.0` improves held-out one-step MSE and high-fidelity zone
  WD RMSE the most among these short runs.
- Zone 3 volume RMSE increases compared with baseline, so the current zone loss
  is improving depth more than volume.
- This is one-step evaluation, not full rollout evaluation.

## Ablation Controls

Additional node-zone ablations:

- `random`: shuffled `zone_weight.txt` with seed 0, preserving the weight
  distribution but destroying spatial/fidelity meaning.
- `full`: all nodes use weight `1.0`, approximating a full-local-everywhere
  node-weighted supervision control.

Generated files:

- `/mnt/8tb_hdd2/joyce/HGN_train/zone_weight_random_seed0.txt`
- `/mnt/8tb_hdd2/joyce/HGN_train/zone_weight_full.txt`
- `/mnt/8tb_hdd2/joyce/HGN_train/zone_ablation_summary_seed0.json`

One-step ablation CSV: `results_zone_ablation_eval_seed0_h4h6.csv`

| Checkpoint | MSE | Zone Loss | Zone 3 RMSE | Zone 3 WD RMSE | Zone 3 Volume RMSE |
|---|---:|---:|---:|---:|---:|
| baseline | 2.4669e-03 | 2.6295e-03 | 4.3744e-02 | 5.6606e-02 | 2.4951e-02 |
| zone025 | 2.0569e-03 | 2.2800e-03 | 4.2470e-02 | 5.3499e-02 | 2.7289e-02 |
| zone1 | 1.7007e-03 | 1.7713e-03 | 3.8960e-02 | 4.6999e-02 | 2.8743e-02 |
| random | 1.6702e-03 | 1.7921e-03 | 3.8745e-02 | 4.7572e-02 | 2.7176e-02 |
| full | 1.6731e-03 | 1.7985e-03 | 3.8777e-02 | 4.7731e-02 | 2.6988e-02 |

One-step read:

- Zone-aware `zone1` gives the best Zone 3 WD RMSE.
- `random` and `full` are competitive and slightly better in one-step MSE and
  Zone 3 volume RMSE.
- This means the current node-weighted loss is not yet sufficient to prove that
  Automatic Fidelity Zoning is better than generic extra supervision.

## Held-Out 10-Step Rollout Evaluation

Rollout CSV: `results_zone_rollout_seed0_h4h6_len10.csv`

| Checkpoint | Rollout RMSE | Rollout WD RMSE | Rollout Volume RMSE | Zone 3 Rollout RMSE | Zone 3 WD RMSE | Zone 3 Volume RMSE |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 2.3772e-01 | 3.0211e-01 | 1.4746e-01 | 2.0196e-01 | 2.5561e-01 | 1.2734e-01 |
| zone025 | 2.1702e-01 | 2.6801e-01 | 1.4956e-01 | 1.8932e-01 | 2.3243e-01 | 1.3285e-01 |
| zone1 | 2.0213e-01 | 2.4035e-01 | 1.5475e-01 | 1.7278e-01 | 2.0272e-01 | 1.3638e-01 |
| random | 2.0084e-01 | 2.4744e-01 | 1.3945e-01 | 1.7852e-01 | 2.1567e-01 | 1.3123e-01 |
| full | 2.0069e-01 | 2.4726e-01 | 1.3934e-01 | 1.7826e-01 | 2.1576e-01 | 1.3037e-01 |

Rollout read:

- `zone1` gives the best Zone 3 rollout WD RMSE, which supports the high-fidelity
  zone targeting idea for water-depth rollout.
- `random` and `full` give slightly better overall rollout RMSE and volume RMSE.
- The current node-zone loss is therefore a useful first implementation, but not
  yet the final research method. The next version needs a more physical local
  residual, likely velocity- or edge-informed, while keeping it separate from
  this node-zone branch.

## VX/VY Pseudo-Local Residual Diagnostic

Diagnostic script:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/mnt/8tb_hdd2/joyce/physicsnemo:/mnt/8tb_hdd2/joyce/.local/lib/python3.10/site-packages:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages /mnt/8tb_hdd2/joyce/conda-envs/yutest/bin/python evaluate_pseudo_local_residual.py --data-dir /mnt/8tb_hdd2/joyce/HGN_train --eval-ids-file eval_h4_h6.txt --rollout-length 10 --k 4 --output-csv results_pseudo_local_residual_seed0_h4h6_len10_k4.csv --checkpoint baseline checkpoints_seed0_baseline_h1h3_b50 --checkpoint zone025 checkpoints_seed0_zone025_h1h3_b50 --checkpoint zone1 checkpoints_seed0_zone1_h1h3_b50 --checkpoint random checkpoints_seed0_randomzone_h1h3_b50 --checkpoint full checkpoints_seed0_fulllocal_h1h3_b50
```

Output CSV: `results_pseudo_local_residual_seed0_h4h6_len10_k4.csv`

| Checkpoint | Pred Proxy Residual | Pred-vs-GT ΔV RMSE | Zone 3 Pred Proxy Residual | Zone 3 Pred-vs-GT ΔV RMSE |
|---|---:|---:|---:|---:|
| baseline | 6.5801e+02 | 3.6459e+03 | 5.5244e+02 | 3.0513e+03 |
| zone025 | 6.5657e+02 | 3.6973e+03 | 5.7546e+02 | 3.2431e+03 |
| zone1 | 6.1290e+02 | 3.6243e+03 | 5.6643e+02 | 3.2969e+03 |
| random | 5.7490e+02 | 3.3500e+03 | 5.4101e+02 | 3.1284e+03 |
| full | 5.7642e+02 | 3.3522e+03 | 5.3920e+02 | 3.1114e+03 |

Diagnostic read:

- The kNN + VX/VY proxy is useful as a diagnostic, but it is not yet a reliable
  final physics loss. The GT proxy residual is large, meaning kNN velocity
  projection does not reconstruct HEC-RAS local volume change well enough.
- `random` and `full` again look competitive in this proxy metric, so the current
  node-weighting branch still does not isolate the value of Automatic Fidelity
  Zoning.
- The next meaningful method step is not to blindly add this proxy to training;
  it is to improve local residual construction, preferably using HEC-RAS face
  connectivity / face flow, or at least a better calibrated edge-flow proxy.

## Separate Edge-Local Proxy Training Branch

This branch is intentionally gated by config:

- `use_edge_local_proxy`
- `edge_local_loss_weight`

It loads `M80_VX_*.txt` / `M80_VY_*.txt` and edge unit vectors into the graph,
but it does not change the model architecture and does not enable node-zone loss.

Training runs:

| Run | edge_local_loss_weight | total_loss | mse_loss | physics_loss | edge_local_proxy_loss | Zone 3 WD RMSE | Checkpoint |
|---|---:|---:|---:|---:|---:|---:|---|
| Edge proxy 1e-6 | 1e-6 | 9.2051e+00 | 1.3263e-02 | 1.0337e-01 | 9.0885e+06 | 9.5490e-02 | `checkpoints_seed0_edgeproxy1e-6_h1h3_b50/checkpoint.0.0.pt` |
| Edge proxy 1e-8 | 1e-8 | 2.2620e-01 | 1.3498e-02 | 1.1669e-01 | 9.6011e+06 | 1.0290e-01 | `checkpoints_seed0_edgeproxy1e-8_h1h3_b50/checkpoint.0.0.pt` |

One-step CSV: `results_edgeproxy_eval_seed0_h4h6.csv`

| Checkpoint | MSE | Zone Loss | Zone 3 RMSE | Zone 3 WD RMSE | Zone 3 Volume RMSE |
|---|---:|---:|---:|---:|---:|
| baseline | 2.4669e-03 | 2.6295e-03 | 4.3744e-02 | 5.6606e-02 | 2.4951e-02 |
| edgeproxy1e-6 | 2.3767e-03 | 2.0952e-03 | 4.2958e-02 | 5.2448e-02 | 3.0647e-02 |
| edgeproxy1e-8 | 3.0182e-03 | 2.5003e-03 | 4.4175e-02 | 5.3641e-02 | 3.2013e-02 |

10-step rollout CSV: `results_edgeproxy_rollout_seed0_h4h6_len10.csv`

| Checkpoint | Rollout RMSE | WD RMSE | Volume RMSE | Zone 3 Rollout RMSE | Zone 3 WD RMSE | Zone 3 Volume RMSE |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 2.3772e-01 | 3.0211e-01 | 1.4746e-01 | 2.0196e-01 | 2.5561e-01 | 1.2734e-01 |
| edgeproxy1e-6 | 2.7619e-01 | 3.3229e-01 | 2.0528e-01 | 2.2922e-01 | 2.6839e-01 | 1.8163e-01 |
| edgeproxy1e-8 | 2.8772e-01 | 3.6154e-01 | 1.8664e-01 | 2.2369e-01 | 2.4883e-01 | 1.9480e-01 |

Edge-proxy read:

- The current kNN + VX/VY proxy should stay separate from the stronger node-zone
  branch. It does not improve overall rollout.
- `edgeproxy1e-6` improves one-step Zone 3 WD but damages rollout and volume.
- `edgeproxy1e-8` gives slightly better Zone 3 WD rollout than baseline, but
  overall rollout is still worse.
- This supports the research direction that the final local conservation term
  should use real HEC-RAS face connectivity / face flow instead of this rough
  kNN transport proxy.

## True HGN_Test Evaluation And HEC-RAS Face Data Update

New data now available:

- Train dataset: `/mnt/8tb_hdd2/joyce/HGN_train` (`H1`-`H40`)
- Test dataset: `/mnt/8tb_hdd2/joyce/HGN_test` (`H41`-`H50`)
- HEC-RAS geometry: `/mnt/8tb_hdd2/joyce/Minxiong/Minxiong.g01`
- HEC-RAS result HDF: `/mnt/8tb_hdd2/joyce/Minxiong/Minxiong.p03.hdf`

The zone generation was updated to use midpoint thresholds between nominal
square-cell areas for the HEC-RAS refinement regions `100`, `80`, `60`, `40`:

| Zone | Nominal resolution | Nominal area | Rule | Weight | Node Count |
|---:|---:|---:|---|---:|---:|
| 0 | 100 | 10000 | `area >= 8200` | 0.0 | 6193 |
| 1 | 80 | 6400 | `5000 <= area < 8200` | 0.25 | 3182 |
| 2 | 60 | 3600 | `2600 <= area < 5000` | 0.5 | 2066 |
| 3 | 40 | 1600 | `area < 2600` | 1.0 | 1274 |

Generated/updated files:

- `/mnt/8tb_hdd2/joyce/HGN_train/zone_label.txt`
- `/mnt/8tb_hdd2/joyce/HGN_train/zone_weight.txt`
- `/mnt/8tb_hdd2/joyce/HGN_train/zone_summary.json`
- `/mnt/8tb_hdd2/joyce/HGN_test/zone_label.txt`
- `/mnt/8tb_hdd2/joyce/HGN_test/zone_weight.txt`
- `/mnt/8tb_hdd2/joyce/HGN_test/zone_summary.json`

The HEC-RAS HDF was also used to extract a separate true-face graph. This is not
mixed into the kNN edge proxy branch.

Face graph files:

- `/mnt/8tb_hdd2/joyce/Minxiong/hecras_hgn_face_graph.npz`
- `/mnt/8tb_hdd2/joyce/Minxiong/hecras_hgn_face_graph_summary.json`

Face graph summary:

| Item | Value |
|---|---:|
| HGN nodes | 12715 |
| HDF cells | 13429 |
| HDF faces | 29347 |
| Internal HGN faces | 28633 |
| Boundary/ghost faces | 714 |
| Face velocity shape | `239 x 29347` |
| Max HGN/HDF coordinate distance | `6.79e-10` |

Important interpretation:

- `HGN_train/M80_XY.txt` and `HGN_test/M80_XY.txt` match the first `12715` HDF
  cell centers exactly enough to use the same cell index order.
- The additional `714` HDF cells appear as boundary/ghost-style cells and are
  kept separate from the internal HGN face graph.
- The HDF contains face velocity, but it is event-specific to `Minxiong.p03.hdf`;
  it should only be used as a face-flow target when the HDF event matches the
  evaluated hydrograph.

### HGN_Test One-Step Evaluation

One-step CSV: `results_zone_eval_hgn_test_seed0_h41h50.csv`

| Checkpoint | MSE | Zone Loss | Zone 3 RMSE | Zone 3 WD RMSE | Zone 3 Volume RMSE |
|---|---:|---:|---:|---:|---:|
| baseline | 2.4745e-03 | 2.6465e-03 | 4.4385e-02 | 5.7550e-02 | 2.5054e-02 |
| zone025 | 2.0643e-03 | 2.2955e-03 | 4.3048e-02 | 5.4389e-02 | 2.7344e-02 |
| zone1 | 1.7078e-03 | 1.7882e-03 | 3.9546e-02 | 4.7989e-02 | 2.8707e-02 |
| random | 1.6774e-03 | 1.8103e-03 | 3.9381e-02 | 4.8614e-02 | 2.7159e-02 |
| full | 1.6803e-03 | 1.8166e-03 | 3.9406e-02 | 4.8757e-02 | 2.6976e-02 |
| edgeproxy1e-6 | 2.3832e-03 | 2.1191e-03 | 4.3586e-02 | 5.3528e-02 | 3.0557e-02 |
| edgeproxy1e-8 | 3.0251e-03 | 2.5153e-03 | 4.4605e-02 | 5.4384e-02 | 3.1955e-02 |

### HGN_Test 10-Step Rollout Evaluation

Rollout CSV: `results_zone_rollout_hgn_test_seed0_h41h50_len10.csv`

| Checkpoint | Rollout RMSE | WD RMSE | Volume RMSE | Zone 3 Rollout RMSE | Zone 3 WD RMSE | Zone 3 Volume RMSE |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 2.3772e-01 | 3.0212e-01 | 1.4746e-01 | 2.0372e-01 | 2.5819e-01 | 1.2777e-01 |
| zone025 | 2.1703e-01 | 2.6802e-01 | 1.4956e-01 | 1.9097e-01 | 2.3493e-01 | 1.3316e-01 |
| zone1 | 2.0214e-01 | 2.4036e-01 | 1.5475e-01 | 1.7427e-01 | 2.0521e-01 | 1.3646e-01 |
| random | 2.0085e-01 | 2.4745e-01 | 1.3945e-01 | 1.8012e-01 | 2.1817e-01 | 1.3148e-01 |
| full | 2.0070e-01 | 2.4728e-01 | 1.3934e-01 | 1.7983e-01 | 2.1819e-01 | 1.3064e-01 |
| edgeproxy1e-6 | 2.7619e-01 | 3.3228e-01 | 2.0528e-01 | 2.3010e-01 | 2.6992e-01 | 1.8157e-01 |
| edgeproxy1e-8 | 2.8773e-01 | 3.6155e-01 | 1.8664e-01 | 2.2412e-01 | 2.4979e-01 | 1.9453e-01 |

HGN_Test read:

- `zone1` remains the best method for high-fidelity Zone 3 water-depth rollout.
- `random` and `full` remain strongest for overall rollout and volume metrics.
- The edge proxy branch remains weaker overall and should not be used as the
  main method.
- The next local-conservation branch should use the extracted HEC-RAS true-face
  graph instead of kNN/VX/VY proxy edges.

### HEC-RAS Face Residual Diagnostic

Face residual CSV: `results_hecras_face_residual_hgn_test_seed0_h41h50_len10.csv`

This diagnostic uses the extracted HEC-RAS face graph and the face velocity in
`/mnt/8tb_hdd2/joyce/Minxiong/Minxiong.p03.hdf`. The command was intentionally
kept separate from the model training and from the previous kNN/VX/VY edge proxy
branch.

Important caveat: `--matched-hdf-event` was not set, so these values should be
treated as alignment/scale diagnostics only. They are not final physics metrics
unless the HDF result event is confirmed to match the evaluated HGN hydrographs.

| Checkpoint | Face proxy scale | GT-face residual RMSE | Pred-face residual RMSE | Pred-vs-GT delta RMSE | Zone 3 pred-face residual RMSE |
|---|---:|---:|---:|---:|---:|
| baseline | 2.6978e-03 | 2.9586e+03 | 6.1169e+02 | 3.4656e+03 | 5.4284e+02 |
| zone1 | 2.7247e-03 | 3.1155e+03 | 6.1999e+02 | 3.6371e+03 | 5.5955e+02 |
| random | 1.9178e-03 | 2.8202e+03 | 5.5714e+02 | 3.2773e+03 | 5.3785e+02 |
| full | 1.9128e-03 | 2.8178e+03 | 5.5699e+02 | 3.2748e+03 | 5.3449e+02 |

Diagnostic read:

- `full` and `random` are closest on the face-residual diagnostic, which matches
  their stronger HGN_Test rollout/volume behavior.
- `zone1` is still the most useful high-fidelity Zone 3 water-depth weighting
  result, but it does not dominate this face-residual diagnostic.
- The next clean branch should consume `hecras_hgn_face_graph.npz` directly and
  add an optional true-face local-conservation loss, without touching the
  existing zone-weight-only branch.

### HEC-RAS True-Face Loss Smoke Branch

Code branch:

- `use_hecras_face_loss`
- `hecras_face_loss_weight`
- `hecras_face_graph_file`
- `hecras_face_velocity_file`
- `hecras_face_zone_mode`
- `hecras_face_calibrate_to_target`

This branch is separate from `edge_local_proxy`. It uses the extracted HEC-RAS
internal face graph and optional HDF face velocity to build a selected-zone
face residual during training. It remains disabled by default because
`Minxiong.p03.hdf` event matching is not yet confirmed.

Short training:

| Checkpoint | Train hydrographs | Batches | Face loss weight | Avg total loss | Avg HEC-RAS face loss | Zone 3 WD train RMSE |
|---|---|---:|---:|---:|---:|---:|
| hecrasface1e-10 | H1-H3 | 50 | 1e-10 | 1.3506e-01 | 9.7852e+06 | 9.6588e-02 |

HGN_Test one-step CSV: `results_hecrasface_eval_hgn_test_seed0_h41h50.csv`

| Checkpoint | MSE | Zone Loss | Zone 3 RMSE | Zone 3 WD RMSE | Zone 3 Volume RMSE |
|---|---:|---:|---:|---:|---:|
| hecrasface1e-10 | 2.4802e-03 | 2.6558e-03 | 4.4483e-02 | 5.7827e-02 | 2.4763e-02 |

HGN_Test rollout CSV: `results_hecrasface_rollout_hgn_test_seed0_h41h50_len10.csv`

| Checkpoint | Rollout RMSE | WD RMSE | Volume RMSE | Zone 3 Rollout RMSE | Zone 3 WD RMSE | Zone 3 Volume RMSE |
|---|---:|---:|---:|---:|---:|---:|
| hecrasface1e-10 | 2.3799e-01 | 3.0327e-01 | 1.4593e-01 | 2.0369e-01 | 2.5886e-01 | 1.2631e-01 |

HEC-RAS face diagnostic CSV:
`results_hecrasface_face_residual_hgn_test_seed0_h41h50_len10.csv`

| Checkpoint | Face proxy scale | GT-face residual RMSE | Pred-face residual RMSE | Pred-vs-GT delta RMSE | Zone 3 pred-face residual RMSE |
|---|---:|---:|---:|---:|---:|
| hecrasface1e-10 | 2.6892e-03 | 2.9304e+03 | 6.0458e+02 | 3.4298e+03 | 5.3681e+02 |

Current read:

- The new true-face training path is technically wired and GPU-trainable.
- With `hecras_face_loss_weight=1e-10`, rollout accuracy is close to baseline
  and does not beat `zone1`.
- Face residual diagnostic improves slightly versus baseline
  (`611.7 -> 604.6`, Zone 3 `542.8 -> 536.8`), but this is still diagnostic,
  not a final physics result.
- The next useful experiment is not simply increasing the loss weight; first
  confirm HDF event matching or switch to a geometry-only residual that does not
  rely on event-specific face velocity.

### HEC-RAS / HGN Event Matching Check

Event match CSV: `results_hecras_hgn_event_match.csv`

Event match report: `results_hecras_hgn_event_match.md`

Tool:

- `match_hecras_hgn_events.py`

Key result:

| Check | Result |
|---|---:|
| HGN hydrographs checked | 50 |
| HGN hydrographs exactly matching HDF `result_upstreamBC1` | 50 |
| Max abs diff for `result_upstreamBC1` | 5.0022e-10 |
| HGN precipitation files checked | 50 |
| HGN precipitation nonzero files | 0 |
| HDF event precipitation max | 1.0 |

Interpretation:

- The HDF result upstream boundary condition `upstreamBC1` matches every HGN
  `M80_US_InF_H*.txt` file exactly within numerical tolerance.
- This means boundary inflow alone cannot identify a unique HGN hydrograph
  event; H1-H50 appear to share the same upstream inflow template.
- HGN precipitation files are all zero, while the HDF event precipitation is
  nonzero. Because of this mismatch, `Minxiong.p03.hdf` face velocity should
  still be treated as a diagnostic target unless the precipitation/source setup
  is reconciled.
- A safer next branch is geometry-only true-face regularization, or obtaining
  event-specific HDF outputs that match each HGN hydrograph's full forcing and
  state trajectory.

### HEC-RAS Geometry-Only True-Face Regularization

Code branch:

- `use_hecras_face_geometry_loss`
- `hecras_face_geometry_loss_weight`
- `hecras_face_geometry_zone_mode`

This branch only uses the extracted HEC-RAS internal face connectivity, face
length, and cell area. It does not use event-specific HDF face velocity, so it
is safer than `use_hecras_face_loss` when HDF precipitation/forcing does not
fully match HGN.

The loss regularizes predicted volume delta per cell area across true HEC-RAS
internal faces, with optional fidelity-zone weighting.

Short training:

| Checkpoint | Train hydrographs | Batches | Geometry loss weight | Device | Avg total loss | Avg geometry loss | Zone 3 WD train RMSE |
|---|---|---:|---:|---|---:|---:|---:|
| hecrasgeom1e-3 | H1-H3 | 50 | 1e-3 | CPU | 1.3441e-01 | 3.7201e-01 | 9.6484e-02 |

HGN_Test one-step CSV: `results_hecrasgeom_eval_hgn_test_seed0_h41h50.csv`

| Checkpoint | MSE | Zone Loss | Zone 3 RMSE | Zone 3 WD RMSE | Zone 3 Volume RMSE |
|---|---:|---:|---:|---:|---:|
| hecrasgeom1e-3 | 2.4759e-03 | 2.6521e-03 | 4.4428e-02 | 5.7661e-02 | 2.4953e-02 |

HGN_Test rollout CSV: `results_hecrasgeom_rollout_hgn_test_seed0_h41h50_len10.csv`

| Checkpoint | Rollout RMSE | WD RMSE | Volume RMSE | Zone 3 Rollout RMSE | Zone 3 WD RMSE | Zone 3 Volume RMSE |
|---|---:|---:|---:|---:|---:|---:|
| hecrasgeom1e-3 | 2.3776e-01 | 3.0237e-01 | 1.4705e-01 | 2.0366e-01 | 2.5833e-01 | 1.2727e-01 |

Current read:

- Geometry-only true-face regularization is wired, trainable, and avoids the
  HDF face-velocity event mismatch.
- With `hecras_face_geometry_loss_weight=1e-3`, HGN_Test rollout remains close
  to baseline and does not beat `zone1`.
- This branch is a safer base for later experiments than event-specific face
  velocity loss, but it still needs stronger design to become a useful local
  conservation method.
- The short training/evaluation was run on CPU because GPU 1 was full during
  this experiment; repeat on GPU when resources are available before treating
  timing as comparable.

### Wet/High-Zone Geometry-Only True-Face Regularization

Motivation:

- The current HGN precipitation (`Pr`) files are all zero and are expected to be
  updated later.
- The current infiltration (`IP`) files are also all zero and are being
  corrected separately.
- Because those source-term inputs are not final, this branch intentionally does
  not use precipitation or infiltration as local conservation terms.
- Instead, it improves the safer geometry-only true-face branch by adding:
  - a wet/dry face mask from current water depth;
  - a `high_adjacent` zone mode that applies only to faces touching Zone 3.

Code/config branch:

- `hecras_face_geometry_zone_mode=high_adjacent`
- `hecras_face_geometry_wet_depth_threshold=0.0`
- `hecras_face_geometry_loss_weight=1e-4`

Short training:

| Checkpoint | Train hydrographs | Batches | Geometry loss weight | Zone mode | Wet depth threshold | Device | Avg total loss | Avg geometry loss | Zone 3 WD train RMSE |
|---|---|---:|---:|---|---:|---|---:|---:|---:|
| hecrasgeom_wet_high1e-4 | H1-H3 | 50 | 1e-4 | high_adjacent | 0.0 | GPU 1 | 1.3417e-01 | 9.4341e-01 | 9.6508e-02 |

HGN_Test one-step CSV: `results_hecrasgeom_wet_high_eval_hgn_test_seed0_h41h50.csv`

| Checkpoint | MSE | Zone Loss | Zone 3 RMSE | Zone 3 WD RMSE | Zone 3 Volume RMSE |
|---|---:|---:|---:|---:|---:|
| hecrasgeom_wet_high1e-4 | 2.4755e-03 | 2.6477e-03 | 4.4369e-02 | 5.7547e-02 | 2.5006e-02 |

HGN_Test rollout CSV: `results_hecrasgeom_wet_high_rollout_hgn_test_seed0_h41h50_len10.csv`

| Checkpoint | Rollout RMSE | WD RMSE | Volume RMSE | Zone 3 Rollout RMSE | Zone 3 WD RMSE | Zone 3 Volume RMSE |
|---|---:|---:|---:|---:|---:|---:|
| hecrasgeom_wet_high1e-4 | 2.3782e-01 | 3.0230e-01 | 1.4738e-01 | 2.0368e-01 | 2.5820e-01 | 1.2759e-01 |

Current read:

- The wet/high-adjacent branch is wired, GPU trainable, and does not depend on
  unfinished `Pr` or `IP` source-term files.
- The HGN_Test rollout is still close to baseline and does not beat `zone1`.
- This is a cleaner experimental base for future selective local conservation
  than the earlier all-face geometry-only branch, but the loss still needs more
  physics before it can become the final method.

### Wet/High-Zone Geometry Sweep

Loss-weight sweep, fixed `hecras_face_geometry_zone_mode=high_adjacent` and
`hecras_face_geometry_wet_depth_threshold=0.0`.

One-step CSV: `results_hecrasgeom_wet_high_weight_sweep_eval_hgn_test_seed0_h41h50.csv`

Rollout CSV: `results_hecrasgeom_wet_high_weight_sweep_rollout_hgn_test_seed0_h41h50_len10.csv`

| Checkpoint | Weight | One-step MSE | One-step Zone 3 WD RMSE | Rollout RMSE | Rollout WD RMSE | Rollout Volume RMSE | Rollout Zone 3 WD RMSE | Rollout Zone 3 Volume RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hecrasgeom_wet_high1e-5 | 1e-5 | 2.4756e-03 | 5.7550e-02 | 2.3777e-01 | 3.0215e-01 | 1.4754e-01 | 2.5821e-01 | 1.2780e-01 |
| hecrasgeom_wet_high1e-4 | 1e-4 | 2.4755e-03 | 5.7547e-02 | 2.3782e-01 | 3.0230e-01 | 1.4738e-01 | 2.5820e-01 | 1.2759e-01 |
| hecrasgeom_wet_high1e-3 | 1e-3 | 2.4813e-03 | 5.7811e-02 | 2.3784e-01 | 3.0286e-01 | 1.4633e-01 | 2.5817e-01 | 1.2659e-01 |

Wet-threshold sweep, fixed `hecras_face_geometry_loss_weight=1e-4` and
`hecras_face_geometry_zone_mode=high_adjacent`.

One-step CSV: `results_hecrasgeom_wet_threshold_sweep_eval_hgn_test_seed0_h41h50.csv`

Rollout CSV: `results_hecrasgeom_wet_threshold_sweep_rollout_hgn_test_seed0_h41h50_len10.csv`

| Checkpoint | Wet threshold | One-step MSE | One-step Zone 3 WD RMSE | Rollout RMSE | Rollout WD RMSE | Rollout Volume RMSE | Rollout Zone 3 WD RMSE | Rollout Zone 3 Volume RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wet0_high1e-4 | 0.0 | 2.4755e-03 | 5.7547e-02 | 2.3782e-01 | 3.0230e-01 | 1.4738e-01 | 2.5820e-01 | 1.2759e-01 |
| wet001_high1e-4 | 0.01 | 2.4747e-03 | 5.7559e-02 | 2.3777e-01 | 3.0226e-01 | 1.4733e-01 | 2.5821e-01 | 1.2759e-01 |
| wet005_high1e-4 | 0.05 | 2.4743e-03 | 5.7557e-02 | 2.3775e-01 | 3.0222e-01 | 1.4735e-01 | 2.5819e-01 | 1.2762e-01 |

Sweep read:

- The sweep confirms this branch is stable across these small weights and wet
  thresholds.
- `1e-3` slightly improves Zone 3 rollout volume but worsens one-step MSE and
  rollout WD, so it is not clearly better.
- `wet_depth_threshold=0.05` gives the best overall rollout among the threshold
  sweep, but the gain is tiny.
- None of the wet/high-adjacent settings beats the existing `zone1` result.
- Until PR/IP files are finalized, this branch should remain a safe diagnostic
  or scaffold for future local conservation rather than the main result.

### Target-Gradient True-Face Geometry Branch

Motivation:

- The earlier geometry-only branch smooths predicted volume delta per area
  across true faces.
- Smoothing is safe but can erase physically meaningful local gradients.
- `hecras_face_geometry_reference_mode=target_gradient` instead matches the
  target cross-face volume-delta gradient from HGN labels. This still avoids
  PR/IP source terms and HDF face velocity, but gives the loss a supervised
  local spatial reference.

Code/config branch:

- `hecras_face_geometry_reference_mode=target_gradient`
- `hecras_face_geometry_zone_mode=high_adjacent`
- `hecras_face_geometry_wet_depth_threshold=0.05`
- `hecras_face_geometry_loss_weight=1e-4`

HGN_Test one-step CSV: `results_hecrasgeom_targetgrad_eval_hgn_test_seed0_h41h50.csv`

HGN_Test rollout CSV: `results_hecrasgeom_targetgrad_rollout_hgn_test_seed0_h41h50_len10.csv`

| Checkpoint | One-step MSE | One-step Zone 3 WD RMSE | Rollout RMSE | Rollout WD RMSE | Rollout Volume RMSE | Rollout Zone 3 WD RMSE | Rollout Zone 3 Volume RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| targetgrad_wet005_high1e-4 | 2.4745e-03 | 5.7556e-02 | 2.3776e-01 | 3.0224e-01 | 1.4732e-01 | 2.5819e-01 | 1.2760e-01 |

Current read:

- Target-gradient loss is wired and trainable.
- Compared with smooth `wet005_high1e-4`, it very slightly improves Zone 3 WD
  rollout but is not meaningfully different overall.
- This is conceptually cleaner than pure smoothing, but still not enough to
  beat `zone1` in the current short-training setup.
