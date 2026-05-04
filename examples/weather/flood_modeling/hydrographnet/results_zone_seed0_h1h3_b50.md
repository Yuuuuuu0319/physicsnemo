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
