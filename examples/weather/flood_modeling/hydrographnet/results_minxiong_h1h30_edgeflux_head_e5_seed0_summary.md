# Minxiong H1-H30 Model-Side Edge-Flux Head E5 Result

Date: 2026-06-15

## Purpose

This experiment is the first model-side step toward true edge local
conservation. It keeps the original HydroGraphNet/MeshGraphKAN node-output
model intact and adds a small experiment-local edge head:

```text
edge_flux = MLP([x_src, x_dst, normalized_face_length])
```

The edge head predicts a signed interval volume flux on the HEC-RAS internal
face graph. Its divergence is used in two terms:

1. `divergence(edge_flux)` should match the native `internal_edge_delta`.
2. `predicted_node_volume_delta - boundary_source_delta - divergence(edge_flux)`
   should close to zero.

This is different from the earlier split node-level loss. The split node-level
loss still supervises the node delta directly against
`internal_edge_delta + boundary_source_delta`; this branch creates an explicit
model-side internal edge-flux variable.

## Code Path

New opt-in config keys:

- `use_hecras_edge_flux_head`
- `hecras_edge_flux_head_loss_weight`
- `hecras_edge_flux_head_zone_mode`
- `hecras_edge_flux_head_divergence_target_weight`
- `hecras_edge_flux_head_hidden_dim`
- `hecras_edge_flux_head_scale`

The default config keeps the branch disabled, so existing HydroGraphNet,
cell-balance, and node-level edge-flow runs are unchanged.

## Training Inputs

- Train split: `train_h1h24.txt`
- Held-out test split: `test_h25h30.txt`
- Face graph:
  `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/hecras_face_graph_h1h30.npz`
- Split edge target:
  `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/hecras_edge_flow_delta_h1h24_internal_plus_boundary_source.npz`

Common overrides:

- `epochs=5`
- `num_training_samples=120`
- `use_physics_loss=true`
- `use_fidelity_zones=true`
- `use_hecras_edge_flow_loss=false`
- `use_hecras_edge_flux_head=true`
- `hecras_edge_flux_head_loss_weight=3e-9`
- `hecras_edge_flux_head_zone_mode=high`
- `hecras_edge_flow_mode=internal_plus_boundary_source`

## Main E5 Run, Scale 1

Checkpoint:

`checkpoints_minxiong_h1h30_edgeflux_head_high_w3e9_e5_seed0`

Training trend:

| Epoch | Total loss | Closure loss | Divergence loss | Edge head loss | Zone 3 volume RMSE |
|---:|---:|---:|---:|---:|---:|
| 0 | `1.4174e-01` | `9.3578e+06` | `2.0968e+06` | `1.1455e+07` | `2.1967e-02` |
| 1 | `1.4320e-02` | `1.0471e+06` | `2.0965e+06` | `3.1437e+06` | `1.1043e-02` |
| 2 | `1.2217e-02` | `7.7756e+05` | `2.0948e+06` | `2.8724e+06` | `8.3432e-03` |
| 3 | `1.1501e-02` | `6.7408e+05` | `2.0912e+06` | `2.7653e+06` | `7.0481e-03` |
| 4 | `1.0802e-02` | `6.0267e+05` | `2.0860e+06` | `2.6886e+06` | `5.9599e-03` |

Held-out H25-H30 metrics:

| Metric | Value |
|---|---:|
| 10-step rollout RMSE | `0.074196` |
| 10-step rollout volume RMSE | `0.063990` |
| 10-step rollout water-depth RMSE | `0.083132` |
| Zone 3 rollout volume RMSE | `0.034340` |
| Zone 3 rollout water-depth RMSE | `0.108861` |
| Global Cell Flow Balance RMSE | `833.606 ft^3` |
| Zone 3 Cell Flow Balance RMSE | `519.801 ft^3` |
| Relative Cell Flow Balance RMSE | `1.179415` |

Files:

- `results_minxiong_h1h30_h25h30_edgeflux_head_high_e5_seed0_len10.csv`
- `results_minxiong_h1h30_h25h30_edgeflux_head_high_e5_cellbalance_seed0.csv`

## Scale Diagnostics

Because the scale-1 edge head improved closure but barely changed the
divergence target, two short scale tests were run.

### Scale 1e5

Checkpoint:

`checkpoints_minxiong_h1h30_edgeflux_head_high_w3e9_scale1e5_e3_seed0`

Result:

- Initial edge-head loss became too large.
- Divergence loss decreased from `4.8989e+07` to `4.9254e+06` by epoch 2, but
  remained worse than scale 1.
- This scale is too aggressive for the current optimizer/loss weight.

### Scale 1e4

Checkpoint directory:

`checkpoints_minxiong_h1h30_edgeflux_head_high_w3e9_scale1e4_e3_seed0`

The directory name contains `e3` because it started as a 3-epoch diagnostic and
was then continued to epoch 4.

Training trend after continuation:

| Epoch | Total loss | Closure loss | Divergence loss | Edge head loss | Zone 3 volume RMSE |
|---:|---:|---:|---:|---:|---:|
| 0 | `1.4474e-01` | `9.8441e+06` | `2.6312e+06` | `1.2475e+07` | `2.2209e-02` |
| 1 | `1.4395e-02` | `1.0761e+06` | `2.0846e+06` | `3.1607e+06` | `1.1328e-02` |
| 2 | `1.1431e-02` | `7.0883e+05` | `1.9949e+06` | `2.7037e+06` | `7.7032e-03` |
| 3 | `1.0475e-02` | `6.2955e+05` | `1.9715e+06` | `2.6010e+06` | `6.6833e-03` |
| 4 | `1.0110e-02` | `5.9223e+05` | `1.9523e+06` | `2.5445e+06` | `6.1526e-03` |

Held-out H25-H30 metrics:

| Metric | Value |
|---|---:|
| 10-step rollout RMSE | `0.069964` |
| 10-step rollout volume RMSE | `0.064178` |
| 10-step rollout water-depth RMSE | `0.075251` |
| Zone 3 rollout volume RMSE | `0.040805` |
| Zone 3 rollout water-depth RMSE | `0.101292` |
| Global Cell Flow Balance RMSE | `825.924 ft^3` |
| Zone 3 Cell Flow Balance RMSE | `582.496 ft^3` |
| Relative Cell Flow Balance RMSE | `1.168546` |

Files:

- `results_minxiong_h1h30_h25h30_edgeflux_head_high_scale1e4_e5_seed0_len10.csv`
- `results_minxiong_h1h30_h25h30_edgeflux_head_high_scale1e4_e5_cellbalance_seed0.csv`

## Interpretation

The model-side edge-flux branch is now operational. It trains, saves both the
main MeshGraphKAN and the edge head, and can be evaluated with the existing
rollout and Cell Flow Balance evaluators.

Current result is mixed:

- Scale 1 gives the best Zone 3 formal Cell Flow Balance result among the E5
  edge-flux-head diagnostics (`519.801 ft^3`).
- Scale 1e4 gives the best E5 rollout RMSE among the model-side edge-flux-head
  diagnostics (`0.069964`) and shows better divergence-loss reduction.
- Neither E5 edge-flux-head run beats the established E50 cell-balance
  baselines.

The important research conclusion is not yet performance superiority. The
conclusion is that the implementation now separates:

1. internal edge transport,
2. boundary/source correction,
3. node storage prediction,
4. model-side edge-flux prediction.

This is the correct structure for a paper-level true edge local conservation
method. The next fair test should use longer training and a fixed evaluation
selection protocol, most likely starting with `hecras_edge_flux_head_scale=1e4`
and comparing it against the E50 baselines.

## E20 Continuation and Epoch Sweep

Both scale 1 and scale 1e4 were continued to epoch 19 and evaluated on H25-H30.
The sweep file is:

`results_minxiong_h1h30_h25h30_edgeflux_head_scale_sweep_e20_seed0_len10.csv`

The checkpoint directories kept their original diagnostic names:

- Scale 1: `checkpoints_minxiong_h1h30_edgeflux_head_high_w3e9_e5_seed0`
- Scale 1e4: `checkpoints_minxiong_h1h30_edgeflux_head_high_w3e9_scale1e4_e3_seed0`

### Latest Epoch 19 Metrics

| Branch | Rollout RMSE | Rollout volume RMSE | Rollout WD RMSE | Zone 3 rollout volume RMSE | Zone 3 rollout WD RMSE | Global Cell Balance RMSE | Zone 3 Cell Balance RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| scale 1 epoch 19 | `0.041046` | `0.040364` | `0.041648` | `0.017015` | `0.068888` | `524.286 ft^3` | `304.393 ft^3` |
| scale 1e4 epoch 19 | `0.039249` | `0.042150` | `0.036094` | `0.028770` | `0.060209` | `567.181 ft^3` | `511.746 ft^3` |

### Best Epochs From Sweep

| Selection metric | Best branch | Epoch | Value |
|---|---|---:|---:|
| Overall rollout RMSE | scale 1e4 | 18 | `0.036035` |
| Rollout volume RMSE | scale 1 | 12 | `0.038401` |
| Rollout water-depth RMSE | scale 1e4 | 18 | `0.031685` |
| Zone 3 rollout volume RMSE | scale 1 | 11 | `0.015283` |
| Zone 3 rollout water-depth RMSE | scale 1e4 | 18 | `0.055773` |
| Global Cell Balance RMSE | scale 1 | 17 | `452.185 ft^3` |
| Zone 3 Cell Balance RMSE | scale 1 | 19 | `304.393 ft^3` |
| One-step MSE | scale 1e4 | 18 | `4.908223e-05` |

### E20 Interpretation

Longer training makes the model-side edge-flux branch substantially stronger.
For scale 1e4, rollout RMSE improved from the E5 value `0.069964` to the best
sweep value `0.036035`. For scale 1, Zone 3 Cell Balance RMSE improved from the
E5 value `519.801 ft^3` to `304.393 ft^3` at epoch 19.

The two scale settings now have a clear tradeoff:

- `scale=1e4` is better for overall rollout and water-depth rollout metrics.
- `scale=1` is better for high-fidelity-zone formal local-conservation metrics.

Compared with the established E50 baselines, the edge-flux head still does not
beat the strongest Cell Flow Balance models:

- `cellbalance_all_w3e9_e50_seed0` remains best for overall rollout and global
  formal Cell Balance.
- `cellbalance_high_w3e9_e50_seed0` remains best for Zone 3 formal Cell Balance.

However, the scale 1 edge-flux head at epoch 19 does improve Zone 3 formal Cell
Balance relative to the no-face E50 baseline (`304.393 ft^3` vs.
`337.094 ft^3`) and improves Zone 3 rollout volume relative to no-face E50
(`0.017015` vs. `0.020569`). This is the first positive evidence that the
model-side edge-flux formulation can provide a selective high-zone benefit,
even though it is not yet the best overall model.

Next fair steps:

1. Keep both selection protocols explicit: best-overall-rollout uses scale 1e4
   epoch 18, while best-Zone-3-conservation uses scale 1 epoch 19.
2. Run a longer E50 comparison only after deciding which claim is being tested.
3. Consider strengthening the internal divergence term or normalizing the edge
   flux target, because scale 1 learns Zone 3 node closure well but still learns
   internal edge divergence slowly.

## Scale 1 E50 Continuation

The scale 1 checkpoint was continued to epoch 49. The checkpoint directory name
is still:

`checkpoints_minxiong_h1h30_edgeflux_head_high_w3e9_e5_seed0`

This name is historical: it began as the E5 run and was then continued. The
directory now contains checkpoints through epoch 49.

Latest epoch 49 evaluation files:

- `results_minxiong_h1h30_h25h30_edgeflux_head_high_scale1_e50_epoch49_seed0_len10.csv`
- `results_minxiong_h1h30_h25h30_edgeflux_head_high_scale1_e50_epoch49_cellbalance_seed0.csv`

Latest epoch 49 held-out H25-H30 metrics:

| Metric | Value |
|---|---:|
| 10-step rollout RMSE | `0.037004` |
| 10-step rollout volume RMSE | `0.039585` |
| 10-step rollout water-depth RMSE | `0.034217` |
| Zone 3 rollout RMSE | `0.050474` |
| Zone 3 rollout volume RMSE | `0.023188` |
| Zone 3 rollout water-depth RMSE | `0.067262` |
| Global Cell Flow Balance RMSE | `468.669 ft^3` |
| Zone 3 Cell Flow Balance RMSE | `481.339 ft^3` |

The final checkpoint improves overall rollout compared with scale 1 epoch 19
but worsens Zone 3 formal Cell Flow Balance. This means the final epoch should
not automatically be treated as the best local-conservation checkpoint.

### Scale 1 Epoch 20-49 Sweep

Sweep file:

`results_minxiong_h1h30_h25h30_edgeflux_head_scale1_epoch_sweep_e20_e50_seed0_len10.csv`

Best epochs from the scale 1 E20-E49 sweep:

| Selection metric | Epoch | Value |
|---|---:|---:|
| One-step MSE | 27 | `5.175874e-05` |
| Overall rollout RMSE | 28 | `0.035555` |
| Rollout volume RMSE | 28 | `0.036399` |
| Rollout water-depth RMSE | 27 | `0.034135` |
| Zone 3 rollout RMSE | 21 | `0.045946` |
| Zone 3 rollout volume RMSE | 21 | `0.016689` |
| Zone 3 rollout water-depth RMSE | 31 | `0.061901` |
| Global Cell Balance RMSE | 21 | `430.678 ft^3` |
| Zone 3 Cell Balance RMSE | 20 | `307.044 ft^3` |

Compared with the earlier scale 1 E20 sweep, extending to E50 gives better
overall rollout and global Cell Flow Balance, but it does not improve the best
Zone 3 Cell Flow Balance. The practical interpretation is:

- For an overall rollout claim, scale 1 epoch 28 is stronger than scale 1 epoch
  19 and is now slightly better than the earlier scale 1e4 epoch 18 result
  (`0.035555` vs. `0.036035`). However, the difference is small enough that both
  should stay in the comparison table until the validation protocol is fixed.
- For a high-fidelity-zone local-conservation claim, scale 1 epoch 20/19 is the
  useful range; later training drifts away from the best Zone 3 Cell Balance.
- The method needs an explicit checkpoint-selection protocol, ideally based on
  validation Zone 3 Cell Balance and validation rollout together, before making
  a paper-level performance claim.

## Edge-Flux-Head Direct Diagnostics

A dedicated evaluator was added:

`evaluate_edge_flux_head.py`

Unlike the rollout evaluator, this script loads both:

1. `MeshGraphKAN`, which predicts node water-depth and volume increments.
2. `HecRasEdgeFluxHead`, which predicts signed interval flux on the HEC-RAS
   internal face graph.

It then evaluates the learned edge variable directly:

```text
divergence_residual = divergence(predicted_edge_flux) - internal_edge_delta
closure_residual = predicted_node_delta - boundary_source_delta
                   - divergence(predicted_edge_flux)
total_node_residual = predicted_node_delta
                      - (internal_edge_delta + boundary_source_delta)
```

The H1-H30 split target was generated so that this diagnostic can run on the
held-out H25-H30 events:

- `physical_budget_results/minxiong_h1h30_edge_flow_split_delta_validation.csv`
- `physical_budget_results/minxiong_h1h30_edge_flow_split_delta_validation.md`
- `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/hecras_edge_flow_delta_h1h30_internal_plus_boundary_source.npz`

Diagnostic output:

`results_minxiong_h1h30_h25h30_edge_flux_head_diagnostics_seed0.csv`

| Checkpoint | Epoch | Scale | Global closure RMSE | Global divergence RMSE | Zone 3 closure RMSE | Zone 3 divergence RMSE | Zone 3 total-node RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| scale 1 | 20 | `1` | `31998.651 ft^3` | `32059.604 ft^3` | `166.664 ft^3` | `835.036 ft^3` | `845.994 ft^3` |
| scale 1 | 28 | `1` | `32003.116 ft^3` | `32064.833 ft^3` | `122.604 ft^3` | `845.933 ft^3` | `847.937 ft^3` |
| scale 1 | 49 | `1` | `32011.375 ft^3` | `32073.452 ft^3` | `108.102 ft^3` | `849.297 ft^3` | `872.318 ft^3` |
| scale 1e4 | 18 | `10000` | `32523.561 ft^3` | `32593.449 ft^3` | `216.731 ft^3` | `874.583 ft^3` | `840.947 ft^3` |

Interpretation:

- The edge-flux-head branch is now measurable as an actual model-side edge
  variable, not just an implicit node-level loss.
- Zone 3 closure improves with longer scale 1 training (`166.664 ft^3` at
  epoch 20 to `108.102 ft^3` at epoch 49).
- Zone 3 internal divergence does not improve in the same way and stays around
  `835-849 ft^3`. This means the current branch is not yet a strong learned
  HEC-RAS internal edge-transport model.
- Therefore, the current honest claim is: the code now supports a true
  edge-variable formulation and exposes direct edge diagnostics, but the
  existing training recipe still needs stronger divergence supervision,
  normalization, or a staged loss schedule before claiming paper-level edge
  local conservation.

## Direct HDF Face-Flow Target Extension

The validation script and dataset loader were extended one step further:

- `validate_hecras_edge_flow_delta.py` now stores
  `{event_id}_internal_face_delta`, aligned to
  `hecras_face_graph["internal_hdf_face_index"]`.
- `HydroGraphDataset` now attaches this per-face target as
  `graph.hecras_internal_face_flow_delta` when it is present in the NPZ.
- `compute_hecras_edge_flux_head_loss` now has an opt-in
  `face_target_weight` term:

```text
face_residual = predicted_internal_face_flux_delta
                - HDF_internal_face_flow_delta
```

The default config keeps this disabled:

```text
hecras_edge_flux_head_face_target_weight: 0.0
```

This matters because the earlier edge-flux-head runs only supervised the edge
head through node-wise divergence. That is an underdetermined target: many edge
flux fields can produce similar node divergence, and the model can improve
closure without learning true HDF face-flow magnitude.

For H25-H30, a test NPZ with direct face targets was generated:

`/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/hecras_edge_flow_delta_h25h30_internal_plus_boundary_source_with_face.npz`

Diagnostic output:

`results_minxiong_h1h30_h25h30_edge_flux_head_face_diagnostics_seed0.csv`

| Checkpoint | Epoch | Scale | Internal face RMSE | Internal face relative RMSE | Internal face target RMS | Edge-flux output RMS |
|---|---:|---:|---:|---:|---:|---:|
| scale 1 | 20 | `1` | `91417.905 ft^3` | `1.000018` | `91415.276 ft^3` | `219.920 ft^3` |
| scale 1 | 28 | `1` | `91418.693 ft^3` | `1.000022` | `91415.276 ft^3` | `298.429 ft^3` |
| scale 1 | 49 | `1` | `91420.036 ft^3` | `1.000032` | `91415.276 ft^3` | `439.164 ft^3` |
| scale 1e4 | 18 | `10000` | `91466.361 ft^3` | `1.000446` | `91415.276 ft^3` | `2264.109 ft^3` |

This confirms the current limitation: the existing divergence-only edge head is
not learning native HEC-RAS face-flow magnitude. It mostly learns a small
correction useful for high-zone closure. The next experiment should train with
`hecras_edge_flux_head_face_target_weight > 0` using a training NPZ that also
contains `internal_face_delta`.

## Direct Face-Supervision Smoke Test

The H1-H24 training NPZ with direct face targets was generated:

`/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/hecras_edge_flow_delta_h1h24_internal_plus_boundary_source_with_face.npz`

Files:

- `physical_budget_results/minxiong_h1h24_edge_flow_split_face_delta_validation.csv`
- `physical_budget_results/minxiong_h1h24_edge_flow_split_face_delta_validation.md`

A short 3-epoch smoke training was run to verify the new direct face target
path:

```text
ckpt_path=checkpoints_minxiong_h1h30_edgeflux_head_face_direct_w1e11_scale1e4_e3_seed0
hecras_edge_flux_head_loss_weight=1e-11
hecras_edge_flux_head_face_target_weight=1.0
hecras_edge_flux_head_scale=10000
```

Training trend:

| Epoch | Total loss | Face loss | Closure loss | Divergence loss | Zone 3 volume RMSE |
|---:|---:|---:|---:|---:|---:|
| 0 | `1.3134e-01` | `2.0739e+09` | `1.1722e+07` | `2.8956e+06` | `3.1152e-02` |
| 1 | `2.5529e-02` | `2.0638e+09` | `3.2744e+06` | `3.1610e+06` | `1.9410e-02` |
| 2 | `2.4439e-02` | `2.0425e+09` | `4.4005e+06` | `4.7894e+06` | `1.5350e-02` |

Held-out H25-H30 direct face diagnostic:

`results_minxiong_h1h30_h25h30_edge_flux_head_face_direct_smoke_e3_seed0.csv`

| Metric | Value |
|---|---:|
| Internal face RMSE | `91669.865 ft^3` |
| Internal face relative RMSE | `1.002814` |
| Internal face target RMS | `91415.276 ft^3` |
| Edge-flux output RMS | `6326.720 ft^3` |
| Zone 3 closure RMSE | `2120.504 ft^3` |
| Zone 3 divergence RMSE | `2175.282 ft^3` |
| Selected face loss | `1.726535e+09` |

Interpretation:

- The direct face target path works mechanically: the training loss sees
  `hecras_edge_flux_face_loss`, saves checkpoints, and the evaluator can measure
  direct face-flow error on held-out H25-H30.
- The edge head output magnitude increased compared with the earlier
  divergence-only scale 1e4 checkpoint (`6326.720 ft^3` vs. `2264.109 ft^3`),
  so the direct face term is affecting the edge head.
- Three epochs are not enough to learn HDF face-flow magnitude, and the node
  closure/divergence metrics became worse. The next experiment should not simply
  increase epochs blindly; it should normalize the face target or use a staged
  schedule, for example:
  1. pretrain the edge head against normalized direct face flow,
  2. then add divergence/closure consistency,
  3. finally fine-tune the node model with rollout and Cell Balance metrics.

## Target-RMS Normalized Face-Supervision Smoke Test

To avoid injecting the raw `~1e9` face-flow MSE directly into the mixed
node/edge loss, the face target loss was extended with:

```text
hecras_edge_flux_head_face_loss_normalization: target_rms
```

This keeps the edge head output in physical `ft^3` units, but divides the
face-flow MSE by the selected target RMS squared before it is added to the
training objective. The raw face loss is still logged separately as
`hecras_edge_flux_raw_face_loss`.

Code changes:

- `compute_hecras_edge_flux_head_loss(..., face_loss_normalization="none")`
- `hecras_edge_flux_head_face_loss_normalization` config key
- `evaluate_edge_flux_head.py --face-loss-normalization`

Smoke checkpoint:

```text
ckpt_path=checkpoints_minxiong_h1h30_edgeflux_head_face_norm_w3e9_scale1e4_e3_seed0
hecras_edge_flux_head_loss_weight=3e-9
hecras_edge_flux_head_face_target_weight=1.0
hecras_edge_flux_head_face_loss_normalization=target_rms
hecras_edge_flux_head_scale=10000
```

Training trend:

| Epoch | Total loss | Normalized face loss | Raw face loss | Closure loss | Divergence loss | Zone 3 volume RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | `1.4476e-01` | `7.0588e+02` | `2.0764e+09` | `9.8450e+06` | `2.6313e+06` | `2.2216e-02` |
| 1 | `1.4409e-02` | `9.9682e+01` | `2.0762e+09` | `1.0779e+06` | `2.0848e+06` | `1.1347e-02` |
| 2 | `1.2012e-02` | `8.0510e+01` | `2.0762e+09` | `8.0307e+05` | `2.0273e+06` | `8.7748e-03` |

Held-out H25-H30 direct face diagnostic:

`results_minxiong_h1h30_h25h30_edge_flux_head_face_norm_smoke_e3_seed0.csv`

| Metric | Value |
|---|---:|
| Internal face RMSE | `91420.961 ft^3` |
| Internal face relative RMSE | `1.000025` |
| Internal face target RMS | `91415.276 ft^3` |
| Edge-flux output RMS | `525.466 ft^3` |
| Zone 3 closure RMSE | `516.584 ft^3` |
| Zone 3 divergence RMSE | `909.076 ft^3` |
| Selected normalized face loss | `100.865` |
| Selected raw face loss | `1.769996e+09` |

Held-out H25-H30 rollout and Cell Balance:

`results_minxiong_h1h30_h25h30_edgeflux_head_face_norm_smoke_e3_rollout_seed0_len10.csv`

| Metric | Value |
|---|---:|
| One-step MSE | `3.868071e-04` |
| 10-step rollout RMSE | `0.095991` |
| 10-step rollout volume RMSE | `0.075845` |
| 10-step rollout water-depth RMSE | `0.112482` |
| Zone 3 rollout volume RMSE | `0.035589` |
| Zone 3 rollout water-depth RMSE | `0.127287` |
| Global Cell Balance RMSE | `999.137 ft^3` |
| Zone 3 Cell Balance RMSE | `582.541 ft^3` |

Interpretation:

- Target-RMS normalization makes the mixed loss numerically safer than raw
  direct face supervision. The unnormalized direct-face smoke produced Zone 3
  closure/divergence errors above `2000 ft^3`; the normalized version reduces
  those to `516.584/909.076 ft^3`.
- The normalized face loss falls quickly on training events, but the raw face
  loss barely changes and the held-out internal-face relative RMSE remains
  about `1.0`. The model is still not learning the native HDF face-flow
  magnitude.
- This confirms that the next paper-level edge-local-conservation experiment
  should be staged, not just longer: first train or pretrain the edge head with
  a direct normalized face target, then introduce divergence/closure and node
  rollout objectives. It may also need stronger edge features than
  `[x_src, x_dst, face_length]`.

## Face-Only Staged Smoke Tests

The loss was extended with an explicit closure weight:

```text
hecras_edge_flux_head_closure_target_weight
```

This makes it possible to run a face-only stage:

```text
closure_target_weight=0
divergence_target_weight=0
face_target_weight=1
face_loss_normalization=target_rms
```

The intent is to test whether the edge head can learn HDF internal Face Flow
magnitude before it is asked to satisfy node closure/divergence consistency.

### Face-Only Without Face Normal

Checkpoint:

`checkpoints_minxiong_h1h30_edgeflux_head_face_only_norm_w1e4_scale1e4_e3_seed0`

Training trend:

| Epoch | Total loss | Normalized face loss | Raw face loss | Zone 3 volume RMSE |
|---:|---:|---:|---:|---:|
| 0 | `2.4755e-01` | `1.3706e+03` | `2.0770e+09` | `3.1499e-02` |
| 1 | `6.2678e-02` | `5.7880e+02` | `2.0765e+09` | `1.9679e-02` |
| 2 | `1.9052e-02` | `1.5127e+02` | `2.0763e+09` | `1.5566e-02` |

Held-out H25-H30 direct face diagnostic:

`results_minxiong_h1h30_h25h30_edge_flux_head_face_only_norm_smoke_e3_seed0.csv`

| Metric | Value |
|---|---:|
| Internal face RMSE | `91418.913 ft^3` |
| Internal face relative RMSE | `1.000015` |
| Edge-flux output RMS | `395.183 ft^3` |
| Zone 3 closure RMSE | `1214.672 ft^3` |
| Zone 3 divergence RMSE | `1126.576 ft^3` |
| Selected normalized face loss | `133.371` |
| Selected raw face loss | `1.770142e+09` |

### Face-Only With HEC-RAS Face Normal

The face graph loader now attaches:

```text
graph.hecras_face_normal
```

The edge head can opt into this geometry input:

```text
hecras_edge_flux_head_use_face_normal=true
```

Checkpoint:

`checkpoints_minxiong_h1h30_edgeflux_head_face_only_norm_normal_w1e4_scale1e4_e3_seed0`

Training trend:

| Epoch | Total loss | Normalized face loss | Raw face loss | Zone 3 volume RMSE |
|---:|---:|---:|---:|---:|
| 0 | `2.1144e-01` | `1.0096e+03` | `2.0764e+09` | `3.1504e-02` |
| 1 | `1.9000e-02` | `1.4197e+02` | `2.0762e+09` | `1.9686e-02` |
| 2 | `1.4605e-02` | `1.0678e+02` | `2.0761e+09` | `1.5568e-02` |

Held-out H25-H30 direct face diagnostic:

`results_minxiong_h1h30_h25h30_edge_flux_head_face_only_norm_normal_smoke_e3_seed0.csv`

| Metric | Value |
|---|---:|
| Internal face RMSE | `91414.854 ft^3` |
| Internal face relative RMSE | `0.999998` |
| Edge-flux output RMS | `252.195 ft^3` |
| Zone 3 closure RMSE | `1161.015 ft^3` |
| Zone 3 divergence RMSE | `1064.213 ft^3` |
| Selected normalized face loss | `111.881` |
| Selected raw face loss | `1.769987e+09` |

Interpretation:

- Face-only staging and face normals improve the selected normalized face loss,
  especially on training events.
- They still do not solve the core magnitude problem. Held-out edge-flux output
  RMS remains only hundreds of `ft^3`, while the HDF internal face-flow target
  RMS is about `91415 ft^3`.
- This suggests the next method change should add physically informative edge
  features, not just train longer. Candidate features include water-surface
  gradient, bed-elevation difference, face length/area scale, current wet/dry
  state on both adjacent cells, and possibly boundary/source context. A pure
  MLP over `[x_src, x_dst, face_length, face_normal]` is still too weak for
  HEC-RAS face-flow magnitude.

## Surface-Feature Edge-Head Smoke

The edge head was extended with another opt-in input block:

```text
hecras_edge_flux_head_use_surface_features=true
```

This appends signed edge deltas for:

- latest normalized water depth,
- latest normalized volume,
- normalized water-surface proxy `elevation + latest_water_depth`,
- graph-RMS-normalized local source rate.

The source-rate normalization was added after an initial quick smoke showed no
benefit; the normalized and unnormalized source-rate results were effectively
identical.

These tests used `max_train_batches=40`, so they are only quick functional
diagnostics, not fair final comparisons.

Checkpoint without source-rate normalization:

`checkpoints_minxiong_h1h30_edgeflux_head_face_only_norm_normal_surface_w1e4_scale1e4_e3_b40_seed0`

Checkpoint with source-rate normalization:

`checkpoints_minxiong_h1h30_edgeflux_head_face_only_norm_normal_surface_srcnorm_w1e4_scale1e4_e3_b40_seed0`

Held-out H25-H30 diagnostics:

| Branch | Internal face RMSE | Internal face relative RMSE | Edge-flux output RMS | Zone 3 closure RMSE | Zone 3 divergence RMSE | Selected face loss |
|---|---:|---:|---:|---:|---:|---:|
| face-only + normal, full E3 | `91414.854 ft^3` | `0.999998` | `252.195 ft^3` | `1161.015 ft^3` | `1064.213 ft^3` | `111.881` |
| + surface features, b40 | `91420.497 ft^3` | `1.000089` | `399.926 ft^3` | `2122.039 ft^3` | `1345.070 ft^3` | `600.500` |
| + surface features + source norm, b40 | `91420.497 ft^3` | `1.000089` | `399.926 ft^3` | `2122.225 ft^3` | `1345.070 ft^3` | `600.500` |

Interpretation:

- The surface-feature code path works, but the quick diagnostic is worse than
  the simpler face-normal run.
- Source-rate scaling is not the limiting factor for this branch.
- This does not justify a longer surface-feature run yet. The next better
  direction is to design physically scaled edge inputs, especially
  denormalized water-surface elevation difference divided by face length and
  signed along the HEC-RAS face normal, rather than feeding raw normalized
  feature differences into a generic MLP.

## Physical Surface-Slope Edge-Head Smoke

The dataset now attaches denormalized current water depth and current
water-surface elevation whenever the HEC-RAS face graph is requested:

```text
graph.current_water_depth_denorm
graph.current_surface_elevation
```

The edge head can opt into a more physical feature block:

```text
hecras_edge_flux_head_use_physical_surface_features=true
```

This adds:

- graph-RMS-normalized water-surface slope across the HEC-RAS face,
- graph-RMS-normalized water-depth slope across the HEC-RAS face,
- source/target wet flags from denormalized water depth.

The intent was to replace the earlier normalized proxy with a more interpretable
hydraulic signal. A quick `max_train_batches=40` smoke was run:

`checkpoints_minxiong_h1h30_edgeflux_head_face_only_norm_normal_physurf_w1e4_scale1e4_e3_b40_seed0`

Held-out H25-H30 diagnostic:

`results_minxiong_h1h30_h25h30_edge_flux_head_face_only_norm_normal_physurf_smoke_e3_b40_seed0.csv`

| Branch | Internal face RMSE | Internal face relative RMSE | Edge-flux output RMS | Zone 3 closure RMSE | Zone 3 divergence RMSE | Selected face loss |
|---|---:|---:|---:|---:|---:|---:|
| face-only + normal, full E3 | `91414.854 ft^3` | `0.999998` | `252.195 ft^3` | `1161.015 ft^3` | `1064.213 ft^3` | `111.881` |
| normalized proxy surface, b40 | `91420.497 ft^3` | `1.000089` | `399.926 ft^3` | `2122.039 ft^3` | `1345.070 ft^3` | `600.500` |
| physical surface-slope, b40 | `91420.230 ft^3` | `1.000084` | `426.900 ft^3` | `2150.387 ft^3` | `1394.268 ft^3` | `556.124` |

Interpretation:

- The physical surface-slope code path works, but the quick diagnostic is still
  worse than the simpler face-normal-only full E3 smoke.
- This negative result suggests that the limiting factor is not simply missing
  water-surface slope features. The current edge head may be too weak, the face
  target may require event/time normalization, or the target may need to be
  learned with a dedicated per-edge architecture before coupling to the node
  rollout model.
- Do not spend a long E20/E50 run on the current surface-feature variant unless
  the architecture or target scaling is changed first.

## Face-Target Distribution and Asinh Target Transform Smoke

The direct HDF Face Flow target is highly multi-scale, so a diagnostic script
was added:

```text
analyze_hecras_face_target_distribution.py
```

It summarizes `*_internal_face_delta` arrays by event and transition, with
optional natural event sorting and optional subset limits for quick checks.

Full H1-H24 train-side target distribution:

```text
results_minxiong_h1h24_face_target_distribution_summary.md
results_minxiong_h1h24_face_target_distribution_by_event.csv
results_minxiong_h1h24_face_target_distribution_by_transition.csv
```

Key H1-H24 distribution facts:

- global internal face target RMS: `307006.149 ft^3`
- median absolute face target: `1254.351 ft^3`
- p90 absolute face target: `98798.212 ft^3`
- p99 absolute face target: `1096146.303 ft^3`
- max absolute face target: `12032229.000 ft^3`
- largest transition: `H18` transition `101`, RMS `503410.457 ft^3`
- Zone 3 touching face RMS: `275590.156 ft^3`
- Zone 3 internal face RMS: `250869.883 ft^3`

This confirms that the face-flow target spans several orders of magnitude.
Therefore the loss now also supports:

```text
hecras_edge_flux_head_face_loss_normalization=asinh_target_rms
hecras_edge_flux_head_face_loss_normalization=signed_log1p_target_rms
```

These transform prediction and target after target-RMS scaling, while
`hecras_edge_flux_raw_face_loss` remains the original untransformed MSE in
`ft^6`-like squared volume units for diagnostics.

GPU smoke:

```text
checkpoints_minxiong_h1h30_edgeflux_head_face_asinh_normal_w1e4_scale1e4_e3_b40_seed0
```

Settings:

- epochs: `3`
- max train batches: `40`
- face target only:
  `closure_target_weight=0`,
  `divergence_target_weight=0`,
  `face_target_weight=1`
- `hecras_edge_flux_head_face_loss_normalization=asinh_target_rms`
- `hecras_edge_flux_head_scale=10000`
- `hecras_edge_flux_head_use_face_normal=true`

Training trend:

| Epoch | asinh face loss | raw face loss | total loss |
|---:|---:|---:|---:|
| 0 | `5.1510` | `2.4912e9` | `1.5380` |
| 1 | `3.4281` | `2.4913e9` | `6.2978e-02` |
| 2 | `2.8888` | `2.4914e9` | `1.8791e-02` |

Held-out H25-H30 diagnostic:

```text
results_minxiong_h1h30_h25h30_edgeflux_head_face_asinh_normal_smoke_e3_b40_seed0.csv
```

| Metric | Value |
|---|---:|
| internal face target RMS | `91415.276 ft^3` |
| internal face RMSE | `91417.397 ft^3` |
| internal face relative RMSE | `1.000023` |
| edge-flux output RMS | `340.147 ft^3` |
| selected asinh face loss | `2.071579` |
| raw face loss | `1.770246e9` |
| divergence RMSE | `32069.886 ft^3` |
| Zone 3 divergence RMSE | `1186.531 ft^3` |
| Zone 3 closure RMSE | `2003.669 ft^3` |

Interpretation:

- The asinh target transform is numerically stable and train/eval both run.
- It decreases the transformed objective more cleanly than raw MSE.
- It still does not solve native face-flow magnitude learning: edge output RMS
  remains only `340 ft^3` against a held-out target RMS of `91415 ft^3`, and
  direct face relative RMSE is still about `1.0`.
- The next step should be stronger edge architecture or explicit target
  standardization/pretraining, not more epochs of the same shallow edge head.

## Per-Face RMS Target Standardization Smoke

Because the H1-H24 target distribution is extremely uneven across individual
faces, a train-set per-face statistics file was generated:

```text
/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/hecras_internal_face_target_stats_h1h24.npz
```

Statistics:

- events: `24`
- transitions: `2880`
- internal faces: `28633`
- mean face RMS: `74729.545 ft^3`
- median face RMS: `7958.847 ft^3`
- max face RMS: `6700508.821 ft^3`

This confirms that per-face scale variation is large; using one global target
RMS can underweight most faces while still being dominated by a small number of
very large-flow faces.

The dataset now optionally attaches:

```text
graph.hecras_internal_face_flow_mean
graph.hecras_internal_face_flow_std
graph.hecras_internal_face_flow_rms
```

and the edge-flux-head loss supports:

```text
hecras_edge_flux_head_face_loss_normalization=per_face_rms
hecras_edge_flux_head_face_loss_normalization=asinh_per_face_rms
hecras_edge_flux_head_face_loss_normalization=signed_log1p_per_face_rms
```

GPU smoke:

```text
checkpoints_minxiong_h1h30_edgeflux_head_face_asinh_perface_normal_w1e4_scale1e4_e3_b40_seed0
```

Settings match the global-asinh smoke except:

```text
hecras_edge_flow_face_stats_npz=.../hecras_internal_face_target_stats_h1h24.npz
hecras_edge_flux_head_face_loss_normalization=asinh_per_face_rms
```

Training trend:

| Epoch | asinh per-face loss | raw face loss | total loss |
|---:|---:|---:|---:|
| 0 | `9.4292` | `2.4911e9` | `1.5385` |
| 1 | `6.1423` | `2.4912e9` | `6.3253e-02` |
| 2 | `5.3278` | `2.4913e9` | `1.9035e-02` |

Held-out H25-H30 diagnostic:

```text
results_minxiong_h1h30_h25h30_edgeflux_head_face_asinh_perface_normal_smoke_e3_b40_seed0.csv
```

| Branch | Internal face relative RMSE | Edge-flux output RMS | Raw face loss | Zone 3 divergence RMSE | Zone 3 closure RMSE |
|---|---:|---:|---:|---:|---:|
| global `asinh_target_rms` | `1.000023` | `340.147 ft^3` | `1.770246e9` | `1186.531 ft^3` | `2003.669 ft^3` |
| per-face `asinh_per_face_rms` | `0.999999` | `282.377 ft^3` | `1.770193e9` | `1076.082 ft^3` | `1995.773 ft^3` |

Interpretation:

- Per-face RMS normalization slightly improves Zone 3 divergence / closure in
  this short smoke.
- It still does not solve direct native face-flow magnitude learning: edge
  output RMS is still only hundreds of `ft^3` versus held-out target RMS
  `91415 ft^3`.
- Per-face standardization is a useful diagnostic and may be part of a final
  formulation, but the current shallow edge head is still too weak by itself.
- Next step: decouple edge-flow learning from node rollout with a dedicated
  edge pretraining objective or stronger edge encoder.

## Dedicated Edge-Head Pretraining Smoke

To separate edge-flow magnitude learning from HydroGraphNet node rollout, an
edge-only pretraining script was added:

```text
pretrain_hecras_edge_flux_head.py
```

This trains only `HecRasEdgeFluxHead` against direct HEC-RAS internal Face Flow
targets. It does not load or train `MeshGraphKAN`, so failure here means the
edge-head inputs / architecture / target transform are insufficient even before
coupling to node prediction.

Smoke setting:

```text
checkpoints_minxiong_h1h30_edgeflux_head_pretrain_asinh_perface_h256_l4_scale1e4_e5_b40_seed0
results_minxiong_h1h30_edgeflux_head_pretrain_asinh_perface_h256_l4_e5_b40_seed0.csv
```

- hidden dim: `256`
- hidden layers: `4`
- scale: `10000`
- face normal: enabled
- loss normalization: `asinh_per_face_rms`
- train: H1-H24
- eval: H25-H30
- epochs: `5`
- max train batches: `40`

Training loop selected loss:

| Epoch | selected loss | raw face loss |
|---:|---:|---:|
| 0 | `3.92126` | `1.93608e9` |
| 1 | `2.42229` | `3.49428e9` |
| 2 | `2.35722` | `2.52119e9` |
| 3 | `2.53956` | `2.46186e9` |
| 4 | `2.54909` | `1.65180e9` |

Held-out H25-H30 edge-only evaluation:

| Epoch | edge output RMS | target RMS | face relative RMSE | Zone 3 output RMS | Zone 3 target RMS | Zone 3 relative RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | `59.402 ft^3` | `91415.276 ft^3` | `1.000010` | `47.143 ft^3` | `25932.506 ft^3` | `1.000375` |
| 1 | `44.835 ft^3` | `91415.276 ft^3` | `1.000008` | `30.289 ft^3` | `25932.506 ft^3` | `1.000174` |
| 2 | `42.959 ft^3` | `91415.276 ft^3` | `1.000005` | `41.583 ft^3` | `25932.506 ft^3` | `1.000308` |
| 3 | `73.350 ft^3` | `91415.276 ft^3` | `1.000007` | `59.472 ft^3` | `25932.506 ft^3` | `1.000343` |
| 4 | `52.149 ft^3` | `91415.276 ft^3` | `1.000006` | `40.651 ft^3` | `25932.506 ft^3` | `1.000351` |

Interpretation:

- The deeper edge-only head trains the transformed objective, but it still does
  not learn native Face Flow magnitude.
- Because the node model is not involved, the main blocker is not HydroGraphNet
  rollout coupling. It is the edge-head input representation / target
  formulation.
- Next research step should move beyond an MLP on current node features:
  include temporal forcing/previous-flow context, use an autoregressive or
  per-face sequence model, predict normalized target and explicitly invert
  scale, or use HEC-RAS hydraulic variables that directly determine face flow.

## Previous Face-Flow Context Smoke

To test whether the edge head mainly failed because it lacked temporal face-flow
context, the dataset and `HecRasEdgeFluxHead` were extended with an optional
teacher-forced feature:

```text
graph.hecras_previous_internal_face_flow_delta
hecras_edge_flux_head_use_previous_face_flow
--edge-head-use-previous-face-flow
```

For transition `t`, this feature provides the HEC-RAS internal Face Flow delta
from transition `t - 1`; transition `0` receives zeros. The feature is scaled by
per-face RMS statistics when available and is only used when explicitly enabled.
This is a diagnostic input, not a final deployment assumption, because future
rollouts would need either observed previous face flow or an autoregressive
prediction of it.

Smoke setting:

```text
checkpoints_minxiong_h1h30_edgeflux_head_pretrain_prevface_asinh_perface_h256_l4_scale1e4_e5_b40_seed0
results_minxiong_h1h30_edgeflux_head_pretrain_prevface_asinh_perface_h256_l4_e5_b40_seed0.csv
```

- hidden dim: `256`
- hidden layers: `4`
- scale: `10000`
- face normal: enabled
- previous internal face-flow feature: enabled
- loss normalization: `asinh_per_face_rms`
- train: H1-H24
- eval: H25-H30
- epochs: `5`
- max train batches: `40`

Training loop selected loss:

| Epoch | selected loss | raw face loss |
|---:|---:|---:|
| 0 | `4.82863` | `2.94591e9` |
| 1 | `2.17092` | `1.06096e9` |
| 2 | `2.22584` | `2.10048e9` |
| 3 | `1.74982` | `1.91394e9` |
| 4 | `2.22062` | `1.83826e9` |

Held-out H25-H30 edge-only evaluation:

| Epoch | edge output RMS | target RMS | face relative RMSE | Zone 3 output RMS | Zone 3 target RMS | Zone 3 relative RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | `70.633 ft^3` | `91415.276 ft^3` | `0.999976` | `33.424 ft^3` | `25932.506 ft^3` | `1.000173` |
| 1 | `54.993 ft^3` | `91415.276 ft^3` | `0.999971` | `35.344 ft^3` | `25932.506 ft^3` | `1.000180` |
| 2 | `58.754 ft^3` | `91415.276 ft^3` | `0.999966` | `37.731 ft^3` | `25932.506 ft^3` | `1.000111` |
| 3 | `46.395 ft^3` | `91415.276 ft^3` | `0.999969` | `26.396 ft^3` | `25932.506 ft^3` | `1.000048` |
| 4 | `40.094 ft^3` | `91415.276 ft^3` | `0.999972` | `24.665 ft^3` | `25932.506 ft^3` | `1.000080` |

Interpretation:

- Previous face-flow context did not solve the native Face Flow magnitude
  problem. The held-out target RMS remains `91415 ft^3`, while predicted output
  RMS remains only tens of `ft^3`.
- The result is slightly different from the no-previous-flow pretraining smoke
  but not meaningfully better. Relative RMSE remains approximately `1.0`.
- This suggests the blocker is not only missing one-step temporal memory. A
  publishable edge-local conservation formulation likely needs a stronger
  hydraulic representation: forcing/time context, wet/dry state, surface-head
  gradients in physical units, roughness/land-cover parameters, or an
  autoregressive edge-state model.

## Edge Feature Audit and Physical-Feature Smoke

An edge-feature audit script was added:

```text
audit_hecras_edge_features.py
```

It writes a per-internal-face table and Markdown summary:

```text
results_minxiong_h1h24_edge_feature_audit.csv
results_minxiong_h1h24_edge_feature_audit.md
```

Audit inputs:

- face graph: `hecras_face_graph_h1h30.npz`
- face target stats: `hecras_internal_face_target_stats_h1h24.npz`
- event IDs: `train_h1h24.txt`
- LandCover lookup: `/mnt/8tb_hdd2/joyce/hecras-dataset/winres/LandCover.hdf`
- dynamic water depth events: H1-H24, `2880` transitions

Important audit findings:

- HEC-RAS internal face topology, face length, face normal, cell elevation,
  cell area, Manning's n, IP, fidelity zones, and dynamic water-surface
  gradient proxy are all available.
- LandCover `Variables` lookup is available and maps Manning's n to Percent
  Impervious. All `12715` HGN cells match the lookup.
- HGN `M80_IP` unique values are `[0.0, 2.0, 10.0, 85.0]`.
- High Face Flow RMS is not concentrated in Zone 3. Zone 3 internal faces have
  much lower median/mean face target RMS than cross-zone faces in this dataset:

| Group | Faces | Median target RMS | Mean target RMS | P99 target RMS | Max target RMS |
|---|---:|---:|---:|---:|---:|
| all | `28633` | `7958.847 ft^3` | `74729.545 ft^3` | `1297282.285 ft^3` | `6700509.000 ft^3` |
| Zone 3 touching | `3564` | `154.072 ft^3` | `49941.136 ft^3` | `1216095.442 ft^3` | `5288590.500 ft^3` |
| Zone 3 internal | `2015` | `126.622 ft^3` | `30446.842 ft^3` | `679036.086 ft^3` | `5288590.500 ft^3` |
| cross-zone | `6170` | `1188.891 ft^3` | `114549.084 ft^3` | `2030621.744 ft^3` | `6700509.000 ft^3` |

The dataset and edge head were then extended with an optional static edge
physical feature block:

```text
graph.hecras_edge_physical_features
hecras_edge_flux_head_use_edge_physical_features
--edge-head-use-edge-physical-features
```

The feature block has 12 columns per internal face:

```text
center_distance, elevation_diff, bed_slope,
manning_mean, manning_diff,
infiltration_mean, infiltration_diff,
area_mean, area_ratio,
same_zone, zone3_touching, zone3_internal
```

These features are denormalized physical values from the HGN static files and
zone labels. The edge head standardizes them across the face graph before
concatenation.

Smoke setting:

```text
checkpoints_minxiong_h1h30_edgeflux_head_pretrain_edgephys_asinh_perface_h256_l4_scale1e4_e5_b40_seed0
results_minxiong_h1h30_edgeflux_head_pretrain_edgephys_asinh_perface_h256_l4_e5_b40_seed0.csv
```

- hidden dim: `256`
- hidden layers: `4`
- scale: `10000`
- face normal: enabled
- static edge physical features: enabled
- loss normalization: `asinh_per_face_rms`
- train: H1-H24
- eval: H25-H30
- epochs: `5`
- max train batches: `40`

Training loop selected loss:

| Epoch | selected loss | raw face loss |
|---:|---:|---:|
| 0 | `3.36454` | `1.85329e9` |
| 1 | `2.41803` | `1.61372e9` |
| 2 | `2.07297` | `2.17019e9` |
| 3 | `1.65021` | `1.92473e9` |
| 4 | `2.67196` | `1.73255e9` |

Held-out H25-H30 edge-only evaluation:

| Epoch | edge output RMS | target RMS | face relative RMSE | Zone 3 output RMS | Zone 3 target RMS | Zone 3 relative RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | `59.570 ft^3` | `91415.276 ft^3` | `1.000002` | `58.521 ft^3` | `25932.506 ft^3` | `1.000301` |
| 1 | `56.088 ft^3` | `91415.276 ft^3` | `1.000009` | `56.243 ft^3` | `25932.506 ft^3` | `1.000422` |
| 2 | `46.400 ft^3` | `91415.276 ft^3` | `1.000008` | `52.110 ft^3` | `25932.506 ft^3` | `1.000416` |
| 3 | `56.850 ft^3` | `91415.276 ft^3` | `1.000012` | `57.400 ft^3` | `25932.506 ft^3` | `1.000503` |
| 4 | `51.211 ft^3` | `91415.276 ft^3` | `1.000010` | `55.045 ft^3` | `25932.506 ft^3` | `1.000368` |

Interpretation:

- Static physical edge features alone do not solve the native Face Flow
  magnitude problem. Held-out output RMS remains tens of `ft^3` while target RMS
  remains `91415 ft^3`.
- The audit is still useful: it proves the dataset now has a clean edge feature
  inventory and identifies which features can be promoted into a stronger
  edge-local model.
- Since no-current-feature, previous-face-flow, and static-edge-physical
  variants all stay near relative RMSE `1.0`, the next publishable step should
  change the target/architecture: predict normalized edge flow with explicit
  inverse scaling, add multi-step forcing/time context, or build a true
  node-edge message-passing module rather than only an auxiliary MLP head.

## Normalized-Output Edge Head Smoke

The edge head was extended with an explicit output-space option:

```text
hecras_edge_flux_head_output_mode
--edge-head-output-mode
```

Supported modes:

```text
raw
per_face_rms
asinh_per_face_rms
signed_log1p_per_face_rms
```

In `asinh_per_face_rms` mode, the MLP predicts a stable normalized value `z`,
then the head converts it back to native HEC-RAS face-flow units:

```text
Q_pred = sinh(z) * face_rms
```

This is different from the earlier loss-only normalization. Earlier runs still
asked the MLP to output raw `ft^3`; only the loss transformed the residual.
This run makes the model output space match the normalized target formulation,
then explicitly returns to raw physical units for diagnostics and future
conservation losses.

### High-Zone Supervision

Smoke setting:

```text
checkpoints_minxiong_h1h30_edgeflux_head_pretrain_edgephys_outasinh_perface_h256_l4_scale1_e5_b40_seed0
results_minxiong_h1h30_edgeflux_head_pretrain_edgephys_outasinh_perface_h256_l4_e5_b40_seed0.csv
```

- output mode: `asinh_per_face_rms`
- scale: `1`
- face normal: enabled
- static edge physical features: enabled
- loss normalization: `asinh_per_face_rms`
- zone mode: `high`
- train: H1-H24
- eval: H25-H30
- epochs: `5`
- max train batches: `40`

Training loop selected loss:

| Epoch | selected loss | raw face loss |
|---:|---:|---:|
| 0 | `0.0351138` | `2.09685e9` |
| 1 | `0.0280583` | `1.58549e9` |
| 2 | `0.0310272` | `1.99046e9` |
| 3 | `0.0205478` | `1.18066e9` |
| 4 | `0.0297315` | `2.36661e9` |

Held-out H25-H30 edge-only evaluation:

| Epoch | edge output RMS | target RMS | face relative RMSE | Zone 3 output RMS | Zone 3 target RMS | Zone 3 relative RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | `10910.756 ft^3` | `91415.276 ft^3` | `1.006526` | `8914.905 ft^3` | `25932.506 ft^3` | `1.163374` |
| 1 | `8733.995 ft^3` | `91415.276 ft^3` | `1.003014` | `8949.867 ft^3` | `25932.506 ft^3` | `1.177464` |
| 2 | `9250.191 ft^3` | `91415.276 ft^3` | `1.000184` | `10560.783 ft^3` | `25932.506 ft^3` | `1.197801` |
| 3 | `9071.657 ft^3` | `91415.276 ft^3` | `0.998891` | `12183.930 ft^3` | `25932.506 ft^3` | `1.218251` |
| 4 | `10960.098 ft^3` | `91415.276 ft^3` | `0.998025` | `15039.507 ft^3` | `25932.506 ft^3` | `1.269005` |

Interpretation:

- This is the first edge-head run that clearly escapes the near-zero-output
  failure mode. Output RMS rises from tens of `ft^3` to roughly
  `9000-11000 ft^3`.
- Global held-out relative RMSE improves slightly below `1.0` by epochs 3-4.
- Zone 3 remains worse than global. This is consistent with the edge-feature
  audit: large internal face-flow targets are not concentrated in Zone 3.
- The result is a useful direction for target formulation, but still not a
  publishable edge-local conservation result. The model still explains only a
  small fraction of native face-flow magnitude and does not produce reliable
  Zone 3 edge flux.

### All-Face Supervision Check

The same normalized-output setup was also run with `zone_mode=all`:

```text
checkpoints_minxiong_h1h30_edgeflux_head_pretrain_edgephys_outasinh_perface_all_h256_l4_scale1_e5_b40_seed0
results_minxiong_h1h30_edgeflux_head_pretrain_edgephys_outasinh_perface_all_h256_l4_e5_b40_seed0.csv
```

Held-out H25-H30 edge-only evaluation:

| Epoch | edge output RMS | target RMS | face relative RMSE | Zone 3 output RMS | Zone 3 target RMS | Zone 3 relative RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | `14901.905 ft^3` | `91415.276 ft^3` | `1.001962` | `7007.083 ft^3` | `25932.506 ft^3` | `1.131462` |
| 1 | `11040.785 ft^3` | `91415.276 ft^3` | `0.999184` | `8745.108 ft^3` | `25932.506 ft^3` | `1.171114` |
| 2 | `28567.977 ft^3` | `91415.276 ft^3` | `1.021722` | `15679.499 ft^3` | `25932.506 ft^3` | `1.351970` |
| 3 | `37011.213 ft^3` | `91415.276 ft^3` | `1.047308` | `24895.480 ft^3` | `25932.506 ft^3` | `1.622734` |
| 4 | `31771.774 ft^3` | `91415.276 ft^3` | `1.007511` | `12140.113 ft^3` | `25932.506 ft^3` | `1.237662` |

Interpretation:

- All-face supervision is not better in this short smoke. It increases output
  magnitude but becomes less stable and does not improve final raw relative
  RMSE.
- The current best edge-head formulation is high-zone supervision with
  `output_mode=asinh_per_face_rms`, but its improvements are still preliminary.
- The next architecture-level step should separate two roles:
  1. learn edge-flow supervision in a stable normalized output space, and
  2. apply local balance/selective conservation through a node-edge message
     passing or autoregressive edge-state model, rather than relying on this
     standalone auxiliary MLP.
