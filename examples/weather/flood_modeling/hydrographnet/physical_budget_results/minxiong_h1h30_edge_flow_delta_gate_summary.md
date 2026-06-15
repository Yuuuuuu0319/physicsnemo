# Minxiong H1-H30 Edge Flow Delta Gate Summary

Date: 2026-06-14

## Purpose

This is the main pre-training gate for a future **edge-informed local
mass-conservation loss**. It checks whether interval-integrated native HEC-RAS
`Face Flow` can reproduce the existing formal `Cell Flow Balance` transport
delta over the exact HGN 30-minute intervals.

This is not yet an edge-loss training result. It is evidence that the edge-flow
target is physically and numerically consistent enough to support the next
implementation step.

## Inputs

- HGN dataset:
  `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset`
- HDF glob:
  `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/planH*/Minxiong.p01.hdf`
- Face topology:
  `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/hecras_face_graph_h1h30.npz`
- Train events: `train_h1h24.txt`
- Test events: `test_h25h30.txt`

## New Validator

Added:

```text
examples/weather/flood_modeling/hydrographnet/validate_hecras_edge_flow_delta.py
```

The validator:

- reads HEC-RAS native `Face Flow`
- builds signed face-to-node divergence using the saved HDF-cell to HGN-node map
- integrates 30-second HEC-RAS face flow over HGN 30-minute intervals
- compares the result to native `Cell Flow Balance` integrated over the same
  intervals
- reports both:
  - `internal`: only faces whose two adjacent cells are both HGN nodes
  - `all_touching`: any face that touches an HGN node, including
    boundary/ghost faces

The first implementation used per-time-step `np.add.at` and was too slow for
30 events. It was replaced by a sparse face-node incidence matrix:

```text
node_flow = face_flow @ incidence
```

This keeps the same sign convention but makes the diagnostic practical.

## Outputs

```text
physical_budget_results/minxiong_h1h24_edge_flow_delta_validation.csv
physical_budget_results/minxiong_h1h24_edge_flow_delta_validation.md
physical_budget_results/minxiong_h25h30_edge_flow_delta_validation.csv
physical_budget_results/minxiong_h25h30_edge_flow_delta_validation.md
```

Additional NPZ created for the held-out test split:

```text
/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/hecras_edge_flow_delta_h25h30_validation.npz
```

The NPZ is diagnostic only for now. For training, a smaller production tensor
should probably store only the selected edge target, not every comparison array.

## Aggregate Results

### Train H1-H24

| Mode | Events | Mean relative RMSE | Max relative RMSE | Mean RMSE (ft^3) | Max RMSE (ft^3) | Mean Zone 3 RMSE (ft^3) | Min correlation |
|---|---:|---:|---:|---:|---:|---:|---:|
| `all_touching` | 24 | `1.493391e-06` | `1.599590e-06` | `0.002229` | `0.002385` | `0.001572` | `1.000000` |
| `internal` | 24 | `4.701826e+01` | `5.047696e+01` | `70193.525377` | `75264.318861` | `148639.468583` | `0.016282` |

### Test H25-H30

| Mode | Events | Mean relative RMSE | Max relative RMSE | Mean RMSE (ft^3) | Max RMSE (ft^3) | Mean Zone 3 RMSE (ft^3) | Min correlation |
|---|---:|---:|---:|---:|---:|---:|---:|
| `all_touching` | 6 | `1.487763e-06` | `1.565293e-06` | `0.002208` | `0.002434` | `0.001526` | `1.000000` |
| `internal` | 6 | `4.664855e+01` | `4.941003e+01` | `69271.494774` | `76838.679811` | `144848.637559` | `0.016427` |

## Interpretation

The result is decisive:

```text
all_touching Face Flow divergence ~= Cell Flow Balance transport delta
```

to numerical precision on both train and held-out test events.

But:

```text
internal-only Face Flow divergence != Cell Flow Balance transport delta
```

because HEC-RAS `Cell Flow Balance` includes boundary/ghost-face contributions
for cells touching the edge of the 2D area. Therefore, a formal edge-informed
local conservation loss cannot blindly use only the internal HGN message-passing
edges and claim complete local mass conservation.

## Consequence For The Paper Method

This resolves a key uncertainty from the earlier planning notes:

- The project can now justify a true edge-informed target from native HEC-RAS
  `Face Flow`.
- The correct exact local-closure graph is not merely the internal HGN graph.
- Boundary/ghost faces are physically required for exact HEC-RAS local closure.

The safest paper wording right now:

```text
We reconstruct a physical conservation graph from HEC-RAS native face-flow
topology. Exact agreement with HEC-RAS Cell Flow Balance requires all faces that
touch modeled HGN cells, including boundary/ghost faces. We therefore keep the
physical conservation graph separate from the HydroGraphNet message-passing
graph and evaluate selective high-fidelity-zone edge constraints with explicit
boundary treatment.
```

## Next Engineering Step

Before training, implement a production edge-flow target loader with one of
these policies:

1. `all_touching` edge target:
   - closest to exact HEC-RAS Cell Flow Balance
   - includes boundary/ghost-face contribution
   - best for formal local conservation evidence

2. `internal_only_with_boundary_mask`:
   - use only internal HGN-HGN physical faces
   - exclude nodes touched by boundary/ghost faces from edge loss
   - useful if the paper wants a strictly graph-internal edge loss

3. `internal_plus_boundary_source`:
   - use internal HGN-HGN faces
   - add boundary/ghost face contribution as a known source/sink term
   - probably the cleanest formulation for a paper

Recommended next branch:

```text
use_hecras_edge_flow_loss=true
hecras_edge_flow_mode=internal_plus_boundary_source
hecras_edge_flow_zone_mode=high
```

This keeps edge-informed conservation separate from the already working
`Cell Flow Balance` branch and preserves the ablation structure:

- noface
- all-zone cell-balance
- high-zone cell-balance
- high-zone edge-flow
- high-zone edge-flow + boundary source

## First Implementation Step Completed

Added an independent precomputed edge-flow branch:

```text
use_hecras_edge_flow_loss
hecras_edge_flow_loss_weight
hecras_edge_flow_zone_mode
hecras_edge_flow_npz
hecras_edge_flow_mode
```

Code paths:

- `physicsnemo/datapipes/gnn/hydrographnet_dataset.py`
  - loads precomputed edge-flow deltas from NPZ
  - attaches `graph.hecras_edge_flow_delta`
  - keeps this separate from `graph.hecras_cell_balance_delta`
- `examples/weather/flood_modeling/hydrographnet/utils.py`
  - adds `compute_hecras_edge_flow_loss`
- `examples/weather/flood_modeling/hydrographnet/train.py`
  - adds edge-flow loss to both normal and pushforward training branches
- `examples/weather/flood_modeling/hydrographnet/conf/config.yaml`
  - adds default config keys for the edge-flow branch

Generated train target:

```text
/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/hecras_edge_flow_delta_h1h24_all_touching.npz
```

Size:

```text
236M
```

This NPZ stores only `H*_all_touching_edge_delta` arrays for H1-H24. It avoids
storing the internal-only comparison arrays in the training target file.

## Smoke Training

Ran a 1-epoch GPU smoke test:

```text
checkpoints_minxiong_h1h30_edgeflow_high_w3e9_smoke_e1_seed0
```

Settings:

- train split: `train_h1h24.txt`
- samples: `48`
- epochs: `1`
- `use_hecras_edge_flow_loss=true`
- `hecras_edge_flow_loss_weight=3e-9`
- `hecras_edge_flow_zone_mode=high`
- `hecras_edge_flow_mode=all_touching`

Epoch 0 result:

| Metric | Value |
|---|---:|
| total loss | `1.3918e-01` |
| MSE loss | `2.9905e-03` |
| physics loss | `1.0465e-01` |
| HEC-RAS edge-flow loss | `1.0513e+07` |
| Zone 3 volume RMSE | `2.2462e-02` |
| Zone 3 WD RMSE | `4.7862e-02` |

Interpretation:

- The edge-flow target loader works.
- `graph.hecras_edge_flow_delta` is attached correctly.
- `compute_hecras_edge_flow_loss` participates in training and logs normally.
- The branch is ready for a real multi-epoch experiment, but the smoke result
  itself should not be reported as model performance.

Updated next experiment recommendation:

```text
H1-H24 train / H25-H30 test
edge-flow high-zone
epochs=50
num_training_samples=1200
weight=3e-9
seed=0
```

Then compare against:

- `noface_e50_seed0`
- `cellbalance_all_w3e9_e50_seed0`
- `cellbalance_high_w3e9_e50_seed0`
