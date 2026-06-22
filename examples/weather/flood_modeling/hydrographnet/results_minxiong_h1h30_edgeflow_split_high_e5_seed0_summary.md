# Minxiong H1-H30 Edge-Flow Split Target Smoke/E5 Result

Date: 2026-06-15

## Purpose

This run tests a boundary-aware decomposition of the native HEC-RAS Face Flow
transport target:

- `internal_edge_delta`: transport contribution from HEC-RAS faces whose two
  adjacent cells are both represented by HGN nodes.
- `boundary_source_delta`: the residual contribution needed to recover the
  full local HEC-RAS transport delta, computed as `all_touching - internal`.
- `cell_balance_delta`: native HEC-RAS Cell Flow Balance transport target.

The training loss still supervises the node-level predicted volume transition,
but it now keeps the internal face contribution and the boundary/source
contribution as separate tensors in the dataset. This is an intermediate step
toward a publishable edge-local-conservation formulation where a future model
head can predict internal edge flux explicitly while boundary/source terms are
handled separately.

## Generated Target

Input split:

- Train events: `train_h1h24.txt`
- HDF glob:
  `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/planH*/Minxiong.p01.hdf`
- Face graph:
  `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/hecras_face_graph_h1h30.npz`

Output target:

`/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/hecras_edge_flow_delta_h1h24_internal_plus_boundary_source.npz`

Validation report:

- `physical_budget_results/minxiong_h1h24_edge_flow_split_delta_validation.csv`
- `physical_budget_results/minxiong_h1h24_edge_flow_split_delta_validation.md`

Key validation result:

- `internal` alone does not close the HEC-RAS Cell Flow Balance target
  (relative RMSE is about 44-50 across H1-H24).
- `all_touching` closes the target to about `1.4e-6` to `1.6e-6` relative RMSE.
- Therefore boundary/ghost/source contributions are required for formal local
  closure.

## Training

Checkpoint:

`checkpoints_minxiong_h1h30_edgeflow_split_high_w3e9_e5_seed0`

Training command used the following important overrides:

- `hydrograph_ids_file=train_h1h24.txt`
- `epochs=5`
- `num_training_samples=120`
- `use_physics_loss=true`
- `use_fidelity_zones=true`
- `use_hecras_edge_flow_loss=true`
- `hecras_edge_flow_loss_weight=3e-9`
- `hecras_edge_flow_zone_mode=high`
- `hecras_edge_flow_mode=internal_plus_boundary_source`

Training loss trend:

| Epoch | Total loss | HEC-RAS edge-flow loss | MSE loss | Physics loss | Zone 3 RMSE |
|---:|---:|---:|---:|---:|---:|
| 0 | `1.3916e-01` | `1.0512e+07` | `2.9898e-03` | `1.0464e-01` | `3.8094e-02` |
| 1 | `1.0557e-02` | `1.9908e+06` | `5.8995e-04` | `3.9949e-03` | `2.3376e-02` |
| 2 | `8.8707e-03` | `1.7127e+06` | `3.9807e-04` | `3.3346e-03` | `2.0384e-02` |
| 3 | `8.0252e-03` | `1.5968e+06` | `3.0461e-04` | `2.9303e-03` | `1.8413e-02` |
| 4 | `7.2526e-03` | `1.5144e+06` | `2.3927e-04` | `2.4702e-03` | `1.6807e-02` |

## H25-H30 Held-Out Evaluation

Rollout file:

`results_minxiong_h1h30_h25h30_edgeflow_split_high_e5_seed0_len10.csv`

Formal Cell Flow Balance file:

`results_minxiong_h1h30_h25h30_edgeflow_split_high_e5_cellbalance_seed0.csv`

Main metrics:

| Metric | Value |
|---|---:|
| 10-step rollout RMSE | `0.074358` |
| 10-step rollout volume RMSE | `0.058236` |
| 10-step rollout water-depth RMSE | `0.087503` |
| Zone 3 rollout volume RMSE | `0.031541` |
| Zone 3 rollout water-depth RMSE | `0.104459` |
| Global Cell Flow Balance RMSE | `784.177 ft^3` |
| Zone 3 Cell Flow Balance RMSE | `634.954 ft^3` |
| Relative Cell Flow Balance RMSE | `1.109480` |

## Interpretation

The split target path is operational: it can generate H1-H24 targets, attach the
split tensors through the dataloader, train with GPU, save checkpoints, and run
the existing H25-H30 rollout and formal Cell Flow Balance evaluators.

This E5 run is not a replacement for the existing E50 baselines. It is much
shorter and currently performs worse than the best E50 models:

- Best existing overall model remains
  `cellbalance_all_w3e9_e50_seed0`.
- Best existing Zone 3 formal local-budget model remains
  `cellbalance_high_w3e9_e50_seed0`.
- Existing E50 edge-flow variants also outperform this E5 split run on several
  held-out metrics, so a fair conclusion requires an E50 split run or a future
  model-side edge-flux head.

Research implication:

This result supports the pipeline and physical decomposition, not yet a final
accuracy claim. The next publishable step is to make the model predict internal
edge flux explicitly and compare:

1. node delta supervised by Cell Flow Balance,
2. node delta supervised by all-touching Face Flow,
3. split internal + boundary/source target,
4. model-side internal edge-flux head with boundary/source correction.
