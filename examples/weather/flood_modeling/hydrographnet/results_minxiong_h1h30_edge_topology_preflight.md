# Minxiong H1-H30 Edge Topology Preflight

Date: 2026-06-13

## Purpose

This preflight prepares the first required input for a future formal
edge-informed local mass-conservation loss. It does not train an edge loss yet.
The goal is to verify whether HEC-RAS face topology and native face-flow data can
be aligned to the current HGN nodes without mixing this branch into the existing
cell-balance experiments.

## Inputs

- HDF source:
  `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/planH1/Minxiong.p01.hdf`
- HGN dataset:
  `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset`

## Code Update

Updated:

```text
examples/weather/flood_modeling/hydrographnet/extract_hecras_face_graph.py
```

The extractor now saves:

- `hgn_to_hdf_cell_index`
- `hdf_to_hgn_node_index`
- HEC-RAS internal face indexes
- HEC-RAS boundary/ghost face indexes
- face normals and lengths
- native `Face Flow` path and availability
- `Face Velocity` path as a fallback only

This keeps the future physical conservation graph separate from the
HydroGraphNet message-passing graph.

## Output Files

```text
/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/hecras_face_graph_h1h30.npz
/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/hecras_face_graph_h1h30_summary.json
```

## Preflight Results

| Check | Result |
|---|---:|
| HGN nodes | `12715` |
| HDF cells | `13429` |
| Mapping mode | HGN node index equals HDF cell index for first `12715` cells |
| Max coordinate distance | `6.794358e-10` |
| Mean coordinate distance | `4.639583e-10` |
| HDF faces | `29347` |
| Internal HGN faces | `28633` |
| Boundary or ghost faces | `714` |
| Native `Face Flow` available | `true` |
| Native `Face Flow` shape | `[7201, 29347]` |
| `Face Velocity` available | `true` |
| `Face Velocity` shape | `[7201, 29347]` |

## Interpretation

The H1-H30 dataset passes the coordinate mapping preflight:

```text
HGN node i corresponds to HDF cell i for i < 12715.
```

This is enough to build a physical edge graph for the current H1-H30 dataset.
Native HEC-RAS `Face Flow` is available and should be the primary edge-flux
source. `Face Velocity` should not be used for formal conservation unless it is
converted to a physical flow with geometry.

The area comparison is not exact:

| Area check | Value |
|---|---:|
| Area RMSE | `153.017` |
| Area max absolute difference | `3331.275` |
| Exact count within `1e-3` | `11978` |

This does not block topology extraction, but it means future edge residual
validation should use HEC-RAS native `Face Flow` and native `Cell Flow Balance`
for closure checks, rather than assuming HGN polygon area alone is the correct
physical volume basis.

## Next Gate Before Edge Loss Training

Before adding an edge-informed loss to `train.py`, the next diagnostic should
validate:

```text
integrated signed Face Flow divergence
  ~= native Cell Flow Balance over the same HGN 30-minute interval
```

Required outputs for the next step:

- per-event integrated edge divergence tensor on HGN nodes
- RMSE / correlation against native `Cell Flow Balance`
- Zone 3 residual metrics
- a clear sign-convention decision for face orientation

Only after this gate passes should the project train an `edge-high` or
`zone+edge` loss branch.

## H1 Budget-Term Gate

Ran:

```text
diagnose_hecras_budget_terms.py
```

Outputs:

```text
physical_budget_results/minxiong_h1h30_h1_budget_terms_first24h.csv
physical_budget_results/minxiong_h1h30_h1_budget_terms_first24h.md
```

Important result:

| Check | Value |
|---|---:|
| `Cell Flow Balance` vs signed native `Face Flow` net sum RMSE | `3.451515e-06 cfs` |
| `Cell Flow Balance` vs signed native `Face Flow` correlation | `1.000000` |
| `Cell Flow Balance` vs inward-only face flow RMSE | `121.836361 cfs` |
| Best volume-budget variant | `cell_balance_trapz_plus_precipitation` |
| Best relative RMSE | `0.037181` |
| Best cell RMSE | `82.084297 ft^3` |
| Zone 3 budget residual RMSE | `48.455710 ft^3` |
| Zone 3 relative RMSE | `0.049548` |

Interpretation:

- The face sign convention used by the validator is correct: native
  `Cell Flow Balance` is numerically the signed net sum of native `Face Flow`.
- This is the strongest evidence so far that a true edge-informed conservation
  loss is physically possible for the H1-H30 dataset.
- Inward-only face flow is not a valid conservation definition.
- The correct local-budget target includes precipitation:
  `cell_balance_trapz_plus_precipitation`.

Remaining caveat:

- HGN `M80_V` and HDF reconstructed active-cell volume are not identical in this
  H1-H30 preflight:
  - RMSE: `2.214976e+04 ft^3`
  - max abs: `3.267357e+05 ft^3`
- Therefore the edge-loss branch should not be trained blindly against a
  different volume convention. Before training `edge-high`, define one
  consistent target:
  1. use the current HGN `M80_V` target and accept the export convention, or
  2. export / load an area-extrapolated volume target aligned with the native
     HEC-RAS budget.

Practical next step:

```text
Build an interval-integrated Face Flow divergence tensor on HGN nodes and compare
it directly with the already-used Cell Flow Balance delta for H1-H30 events.
```

If this tensor matches the `Cell Flow Balance` delta at the HGN training step
level, then the project can add `use_hecras_edge_flow_loss` as a separate branch
without mixing it into the existing cell-balance loss.
