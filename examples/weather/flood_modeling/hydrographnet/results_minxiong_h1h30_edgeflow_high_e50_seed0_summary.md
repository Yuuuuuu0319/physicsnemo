# Minxiong H1-H30 Edge-Flow High-Zone Result

Date: 2026-06-15

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

## Edge-Flow Physical Gate

Before training, native HEC-RAS `Face Flow` was integrated over each HGN time
interval and accumulated into node-level transport deltas. Two face sets were
checked:

- `internal`: faces whose left and right cells are both mapped to HGN nodes.
- `all_touching`: any face that touches a mapped HGN node, including
  boundary/ghost-cell faces.

Validation outputs:

```text
physical_budget_results/minxiong_h1h30_edge_flow_delta_gate_summary.md
physical_budget_results/minxiong_h1h24_edge_flow_delta_validation.csv
physical_budget_results/minxiong_h25h30_edge_flow_delta_validation.csv
```

Aggregate result:

| Split | Face set | Mean relative RMSE | Mean RMSE (ft^3) | Mean Zone 3 RMSE (ft^3) |
|---|---|---:|---:|---:|
| H1-H24 train | all_touching | `1.493391e-06` | `0.002229` | `0.001572` |
| H25-H30 test | all_touching | `1.487763e-06` | `0.002208` | `0.001526` |
| H1-H24 train | internal | `47.018256` | `70193.525377` | `148639.468583` |
| H25-H30 test | internal | `46.648551` | `69271.494774` | `144848.637559` |

Interpretation:

```text
all_touching Face Flow divergence reproduces native Cell Flow Balance transport
delta to numerical precision. Internal-only faces do not close the HEC-RAS
local budget because boundary/ghost faces carry essential flux terms.
```

## New Branch

Added an independent edge-flow target branch instead of mixing it into the
existing cell-balance branch:

- Dataset option: `return_hecras_edge_flow`
- Dataset target file: `hecras_edge_flow_npz`
- Dataset target mode: `hecras_edge_flow_mode`
- Training flag: `use_hecras_edge_flow_loss`
- Training weight: `hecras_edge_flow_loss_weight`
- Zone mode: `hecras_edge_flow_zone_mode`

The production target used for this run was:

```text
/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/hecras_edge_flow_delta_h1h24_all_touching.npz
```

## New Run

Checkpoint:

```text
checkpoints_minxiong_h1h30_edgeflow_high_w3e9_e50_seed0
```

Training settings:

- epochs: `50`
- seed: `0`
- num training samples: `1200`
- physics loss: enabled
- fidelity zones: enabled
- HEC-RAS edge-flow loss weight: `3e-9`
- HEC-RAS edge-flow zone mode: `high`
- HEC-RAS edge-flow mode: `all_touching`

Final training epoch 49:

| Metric | Value |
|---|---:|
| Total loss | `4.7459e-03` |
| MSE loss | `5.5233e-05` |
| Physics loss | `1.3912e-03` |
| HEC-RAS edge-flow loss | `1.0998e+06` |
| Zone 3 volume RMSE | `7.8944e-03` |
| Zone 3 WD RMSE | `1.2685e-02` |

The edge-flow training loss decreased from about `1.0512e+07` at epoch 0 to
about `1.0998e+06` at epoch 49, so the model does learn from the native
Face-Flow-derived target.

## H25-H30 Epoch 49 Four-Way Evaluation

Output CSV:

```text
results_minxiong_h1h30_h25h30_epoch49_fourway_eval_e50_seed0_len10.csv
```

| Metric | noface | all-zone cell-balance | high-zone cell-balance | high-zone edge-flow |
|---|---:|---:|---:|---:|
| One-step MSE | `3.633998e-05` | `1.332539e-05` | `5.983033e-05` | `4.596162e-05` |
| Global cell-balance RMSE (ft^3) | `384.037` | `185.819` | `520.312` | `627.654` |
| Zone 3 cell-balance RMSE (ft^3) | `337.094` | `245.204` | `91.162` | `900.556` |
| Rollout RMSE | `0.029554` | `0.016024` | `0.038119` | `0.031053` |
| Zone 3 rollout WD RMSE | `0.055544` | `0.033510` | `0.058703` | `0.047745` |
| Zone 3 rollout volume RMSE | `0.020569` | `0.008648` | `0.014970` | `0.019134` |

Interpretation:

- `all-zone cell-balance` remains the strongest overall model on H25-H30.
- `high-zone cell-balance` remains the strongest formal Zone 3 local-budget
  model at epoch 49.
- `high-zone edge-flow` trains successfully, but does not yet improve H25-H30
  physical closure or rollout metrics.

## Edge-Flow Checkpoint Sweep

Output CSV:

```text
results_minxiong_h1h30_h25h30_edgeflow_high_epoch_sweep_e50_seed0_len10.csv
```

Best edge-flow checkpoints:

| Selection metric | Best epoch | Value |
|---|---:|---:|
| One-step MSE | `35` | `4.172810e-05` |
| Global cell-balance RMSE (ft^3) | `27` | `562.206` |
| Zone 3 cell-balance RMSE (ft^3) | `7` | `537.077` |
| Rollout RMSE | `41` | `0.030683` |
| Zone 3 rollout WD RMSE | `46` | `0.044454` |
| Zone 3 rollout volume RMSE | `44` | `0.017670` |

Even with best-epoch selection, the current edge-flow branch does not beat the
cell-balance branches. This suggests the limitation is not just final-checkpoint
selection.

## Research Consequence

This result is important but not yet the final paper method:

- The physical extraction is now correct: native HEC-RAS Face Flow can reproduce
  Cell Flow Balance only when boundary/ghost faces are included.
- The software path is working: edge-flow targets can be loaded into
  HydroGraphNet and trained as a separate loss branch.
- The first direct formulation, `all_touching` edge-flow divergence applied only
  to high-fidelity nodes, is not sufficient for improved test performance.

The next paper-level step should reformulate the edge loss instead of merely
training longer:

1. Split edge transport into internal edge divergence plus explicit boundary
   source/sink terms.
2. Add a boundary-node or ghost-face mask so high-zone nodes are not penalized
   by unresolved external flux in the same way as interior nodes.
3. Sweep loss weights below and around `3e-9`, because the current edge-flow
   target magnitude is much larger than the cell-balance target used in the
   previous successful branch.
4. Report the edge-flow branch as a negative/diagnostic result until the
   boundary-aware formulation improves either Zone 3 closure or rollout metrics.

## Boundary-Aware Follow-Up: High-Interior Edge-Flow

Because the edge-flow gate showed that boundary/ghost faces are essential for
exact local closure, a second edge-flow branch was added after the first
`high-zone edge-flow` run. This branch keeps the same precomputed
`all_touching` Face-Flow target, but applies the loss only to Zone 3 nodes that
are not touched by boundary/ghost faces.

Code additions:

- `HydroGraphDataset` now loads `boundary_face_cell_index` from
  `hecras_face_graph_file` when `return_hecras_edge_flow=True`.
- Each graph can receive `hecras_boundary_node_mask`.
- `compute_hecras_edge_flow_loss(..., zone_mode="high_interior")` applies
  weights only where:

```text
zone_label == 3 and hecras_boundary_node_mask == False
```

Checkpoint:

```text
checkpoints_minxiong_h1h30_edgeflow_high_interior_w3e9_e50_seed0
```

Training settings:

- epochs: `50`
- seed: `0`
- num training samples: `1200`
- HEC-RAS edge-flow loss weight: `3e-9`
- HEC-RAS edge-flow zone mode: `high_interior`
- HEC-RAS edge-flow mode: `all_touching`
- HEC-RAS face graph:
  `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/hecras_face_graph_h1h30.npz`

Final training epoch 49:

| Metric | Value |
|---|---:|
| Total loss | `3.6575e-03` |
| MSE loss | `4.5960e-05` |
| Physics loss | `1.3594e-03` |
| HEC-RAS edge-flow loss | `7.5073e+05` |
| Zone 3 volume RMSE | `6.6750e-03` |
| Zone 3 WD RMSE | `1.0375e-02` |

Compared with direct high-zone edge-flow, the final training edge-flow loss
improved from about `1.0998e+06` to `7.5073e+05`.

## H25-H30 Epoch 49 Five-Way Evaluation

Output CSV:

```text
results_minxiong_h1h30_h25h30_epoch49_fiveway_eval_e50_seed0_len10.csv
```

| Metric | noface | all-zone cell-balance | high-zone cell-balance | high-zone edge-flow | high-interior edge-flow |
|---|---:|---:|---:|---:|---:|
| One-step MSE | `3.584949e-05` | `1.251145e-05` | `5.912313e-05` | `4.600320e-05` | `4.600559e-05` |
| Global cell-balance RMSE (ft^3) | `384.037` | `185.819` | `520.312` | `627.654` | `656.206` |
| Zone 3 cell-balance RMSE (ft^3) | `337.094` | `245.204` | `91.162` | `900.556` | `737.459` |
| Rollout RMSE | `0.029554` | `0.016024` | `0.038119` | `0.031053` | `0.033982` |
| Zone 3 rollout WD RMSE | `0.055544` | `0.033510` | `0.058703` | `0.047745` | `0.043951` |
| Zone 3 rollout volume RMSE | `0.020569` | `0.008648` | `0.014970` | `0.019134` | `0.019836` |

Interpretation:

- `high_interior` improves Zone 3 cell-balance residual compared with direct
  high-zone edge-flow:
  `900.556 -> 737.459 ft^3`.
- `high_interior` also improves Zone 3 rollout WD RMSE compared with direct
  high-zone edge-flow:
  `0.047745 -> 0.043951`.
- However, it still does not beat `cellbalance_all` on overall metrics, and it
  is still far from `cellbalance_high` on Zone 3 local-budget closure.

## High-Interior Checkpoint Sweep

Output CSV:

```text
results_minxiong_h1h30_h25h30_edgeflow_high_interior_epoch_sweep_e50_seed0_len10.csv
```

Best high-interior checkpoints:

| Selection metric | Best epoch | Value |
|---|---:|---:|
| One-step MSE | `37` | `3.969664e-05` |
| Global cell-balance RMSE (ft^3) | `23` | `504.670` |
| Zone 3 cell-balance RMSE (ft^3) | `19` | `453.315` |
| Rollout RMSE | `43` | `0.030382` |
| Zone 3 rollout WD RMSE | `49` | `0.043951` |
| Zone 3 rollout volume RMSE | `19` | `0.016700` |

Research interpretation:

- Boundary-aware masking helps the edge-flow branch compared with direct
  high-zone edge-flow, but it is not enough to make edge-flow competitive with
  the cell-balance branches.
- The next edge-level paper step should not be longer training. It should split
  the physical target into:

```text
internal-face divergence + explicit boundary/source/sink terms
```

- A true edge-conservation paper claim likely needs a model-side edge-flux head
  or an edge-message-derived flux quantity, not only a node-volume prediction
  matched to a Face-Flow-derived node target.
