# Minxiong H1-H30 Face Target Scale Normalization Summary

Date: 2026-06-23

## Goal

This diagnostic continues the edge-informed local conservation roadmap.  Earlier
raw-output edge-head experiments showed that a shallow auxiliary MLP tends to
predict near-zero HEC-RAS internal Face Flow deltas.  The goal here is to
separate the Face Flow target magnitude into:

- persistent per-face scale,
- event-level hydrograph scale,
- transition-level flood-stage scale.

This is not yet the final publishable edge local conservation method.  It is a
target-formulation diagnostic before committing to longer GPU training.

## New Code

- `compute_hecras_face_target_scale_stats.py`
  computes global, per-face, per-event, and per-transition RMS statistics from
  `*_internal_face_delta` arrays.
- `HydroGraphDataset`
  can optionally load `hecras_edge_flow_scale_stats_npz` and attach:
  - `graph.hecras_internal_face_flow_global_rms`
  - `graph.hecras_internal_face_flow_event_rms`
  - `graph.hecras_internal_face_flow_transition_rms`
- `HecRasEdgeFluxHead`
  now supports additional diagnostic output modes:
  - `event_rms`
  - `transition_rms`
  - `face_event_rms`
  - `face_transition_rms`
  - corresponding `asinh_*` and `signed_log1p_*` modes
- `pretrain_hecras_edge_flux_head.py`, `evaluate_edge_flux_head.py`, and
  `utils.py` now support the same scale-aware face-loss normalization modes.

## Scale Statistics

Train split H1-H24:

- events: `24`
- transitions: `2880`
- internal faces: `28633`
- global RMS: `307006.148697 ft^3`
- median per-face RMS: `7958.847074 ft^3`
- mean per-face RMS: `74729.544882 ft^3`
- max per-face RMS: `6700508.820826 ft^3`
- largest event RMS: `H18`, `329775.545835 ft^3`
- largest transition RMS: `H18`, transition `101`, `503410.456638 ft^3`

Test split H25-H30:

- events: `6`
- transitions: `720`
- internal faces: `28633`
- global RMS: `302126.432310 ft^3`
- median per-face RMS: `7775.375989 ft^3`
- mean per-face RMS: `73545.827080 ft^3`
- max per-face RMS: `6548338.842875 ft^3`
- largest event RMS: `H28`, `336341.829073 ft^3`
- largest transition RMS: `H28`, transition `103`, `512039.018237 ft^3`

Interpretation:

- The target is extremely long-tailed.
- A single global scale is too coarse.
- Per-face scale alone misses event/time intensity.
- `face_transition_rms` is a useful diagnostic scale because it combines
  persistent per-face magnitude and the current flood-stage magnitude.

## Smoke Experiment

Command family:

```text
pretrain_hecras_edge_flux_head.py
  --edge-head-output-mode asinh_face_transition_rms
  --face-loss-normalization asinh_face_transition_rms
  --edge-head-use-face-normal
  --edge-head-use-edge-physical-features
  --edge-head-hidden-dim 64
  --edge-head-num-hidden-layers 2
  --epochs 1
  --max-train-batches 3
```

Important caveat:

- This smoke used target-aware transition RMS for held-out diagnostics.  That is
  acceptable for diagnosing target formulation, but it is not a final inference
  method because future transition scale would not be known without a predictor.

Results:

| Split | Face relative RMSE | Edge output RMS ft^3 | Target RMS ft^3 | Zone 3 relative RMSE |
|---|---:|---:|---:|---:|
| train eval | `1.008054` | `10828.896` | `92700.006` | `1.166516` |
| H25-H30 eval | `1.008273` | `10892.726` | `91415.275` | `1.189752` |

Compared with previous raw-output diagnostics:

- raw/deeper MLP output RMS was only about `40--52 ft^3`;
- `asinh_per_face_rms` reached about `10960 ft^3` in the high-zone smoke;
- `asinh_face_transition_rms` also reaches about `10893 ft^3` after only a tiny
  3-batch smoke, so the scale-aware path is operational.

However:

- relative face RMSE is still about `1.0`;
- Zone 3 face RMSE is still poor;
- this does not prove formal edge-informed local mass conservation.

## Current Conclusion

The new scale-aware path is useful and should be kept, but the bottleneck is not
fully solved.  The next publishable-method step should not be simply "more
epochs of the same auxiliary MLP".  Better next experiments are:

1. Train `asinh_face_transition_rms` for more batches/epochs only as a diagnostic
   to see whether it leaves the relative-RMSE plateau.
2. Replace target-aware transition RMS with an inference-available scale proxy,
   such as upstream inflow, precipitation, current water volume, or a learned
   event-stage scalar.
3. Move from a detached auxiliary edge head toward a node-edge message-passing
   edge decoder if the diagnostic remains near relative RMSE `1.0`.

## Files

- Train scale NPZ:
  `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/hecras_internal_face_target_scale_stats_h1h24.npz`
- Test scale NPZ:
  `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/hecras_internal_face_target_scale_stats_h25h30.npz`
- Train scale summary:
  `results_minxiong_h1h24_face_target_scale_summary.md`
- Test scale summary:
  `results_minxiong_h25h30_face_target_scale_summary.md`
- Smoke CSV:
  `results_minxiong_h1h30_edgeflux_head_pretrain_facetransition_smoke.csv`

## 2026-06-24 Update: Inference-Available Scale Proxy

The target-aware transition RMS is useful for diagnosis, but it cannot be used
as a final inference method.  A new proxy was fit from HGN-available quantities:

- current upstream inflow
- current downstream outflow
- current precipitation
- total / mean / max current volume
- mean / max / p90 water depth
- wet fractions
- one-step changes in forcing and total volume

New script:

```text
fit_hecras_face_scale_proxy.py
```

Proxy fit result:

| Split | Target RMS | Pred RMS | Relative RMSE | Correlation |
|---|---:|---:|---:|---:|
| train H1-H24 | `307006.148582` | `307059.746241` | `0.031458` | `0.997870` |
| test H25-H30 | `302126.432257` | `304652.486126` | `0.031669` | `0.998080` |

This is important because it suggests that the flood-stage scale of internal
Face Flow can be estimated from inference-available HGN inputs instead of being
taken from the true HEC-RAS Face Flow target.

Generated files:

- proxy model:
  `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/hecras_face_transition_scale_proxy_h1h24_ridge.npz`
- predicted scale stats:
  `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/hecras_face_transition_scale_proxy_h1h30_ridge_predicted.npz`
- proxy prediction report:
  `results_minxiong_h1h30_face_scale_proxy_summary.md`

Proxy-scale edge-head smoke:

```text
edge-head-output-mode = asinh_face_transition_rms
face-loss-normalization = asinh_face_transition_rms
scale stats = hecras_face_transition_scale_proxy_h1h30_ridge_predicted.npz
epochs = 1
max_train_batches = 3
```

| Scale source | H25-H30 face relative RMSE | H25-H30 edge output RMS ft^3 | Zone 3 relative RMSE |
|---|---:|---:|---:|
| target-aware transition scale | `1.008273` | `10892.726` | `1.189752` |
| inference-available proxy scale | `1.003454` | `7160.595` | `1.176843` |

Interpretation:

- The proxy-scale path is operational and slightly improves held-out relative
  face RMSE in the tiny smoke.
- The edge head still has not learned full Face Flow magnitude.
- This is a better path toward publishable edge local conservation because it
  removes direct dependence on target-aware transition RMS.
- Formal edge local conservation still requires a stronger edge decoder or
  longer staged training once CUDA is available.

## 2026-06-24 GPU Follow-up: Longer Proxy-Scale Edge Pretraining

GPU visibility note:

- The normal sandbox could not see `/dev/nvidia*`, so `nvidia-smi` and PyTorch
  reported no CUDA device.
- In the unsandbox / elevated execution environment, both GPUs were visible:
  - GPU0: NVIDIA RTX PRO 5000 Blackwell
  - GPU1: NVIDIA RTX A6000
- GPU1 was heavily occupied, so the following experiments used GPU0.

Longer proxy-scale pretraining:

```text
edge-head-output-mode = asinh_face_transition_rms
face-loss-normalization = asinh_face_transition_rms
scale stats = inference-available ridge proxy scale
hidden_dim = 256
hidden_layers = 4
max_train_batches = 80
```

### High-zone supervision

Checkpoint:

```text
checkpoints_minxiong_h1h30_edgeflux_head_pretrain_proxy_facetransition_h256_l4_e10_b80_seed0
```

Best H25-H30:

- best face relative RMSE: epoch `2`, `0.998353`
- best Zone 3 face relative RMSE: epoch `8`, `1.017190`

Interpretation:

- High-zone supervision alone still struggles to learn global Face Flow
  magnitude.
- It is more stable for selective local metrics, but not enough for full edge
  target learning.

### All-face supervision

Checkpoint:

```text
checkpoints_minxiong_h1h30_edgeflux_head_pretrain_proxy_facetransition_all_h256_l4_e10_b80_seed0
```

CSV files:

```text
results_minxiong_h1h30_edgeflux_head_pretrain_proxy_facetransition_all_h256_l4_e10_b80_seed0.csv
results_minxiong_h1h30_edgeflux_head_pretrain_proxy_facetransition_all_h256_l4_e20_b80_seed0_continue.csv
```

Best / latest H25-H30 face-target results:

| Epoch | Face relative RMSE | Edge output RMS ft^3 | Zone 3 face relative RMSE |
|---:|---:|---:|---:|
| 9 | `0.908524` | `37120.234` | `1.031578` |
| 18 | `0.728222` | `45273.477` | `1.008603` |
| 19 | `0.707966` | `56665.755` | `1.062389` |

This is the strongest Face Flow magnitude result so far.  The edge head is no
longer near-zero and now explains a substantial part of held-out face-flow
magnitude.

### Edge-head-only divergence diagnostic

New evaluator:

```text
evaluate_pretrained_edge_flux_head.py
```

It evaluates a pretrained edge head without a HydroGraphNet node model and
reports whether predicted face flux divergence reconstructs HEC-RAS internal
edge divergence.

Corrected aggregate H25-H30 metrics:

| Checkpoint | Face relative RMSE | Internal divergence relative RMSE | Zone 3 face relative RMSE | Zone 3 divergence relative RMSE |
|---|---:|---:|---:|---:|
| all epoch18 | `0.718506` | `1.932374` | `0.880269` | `14.034999` |
| all epoch19 | `0.695773` | `2.334911` | `0.870377` | `20.399533` |

Important interpretation:

- Face-level magnitude learning has improved substantially.
- But divergence/local-budget closure is not solved.
- Zone 3 face target RMSE can improve while Zone 3 node divergence remains poor,
  because small signed face errors accumulate around cells and the Zone 3
  internal-divergence target RMS is only about `1310.764 ft^3`.
- Therefore the next edge-local conservation step must include an explicit
  divergence/closure loss during edge-head training, not only direct face
  supervision.

Current status:

```text
Face-flow magnitude: improved and now usable as evidence of progress.
Formal edge-local conservation: not yet complete.
Next bottleneck: signed divergence / node local budget closure.
```

## 2026-06-24 Update: Explicit Divergence Loss Fine-tuning

The previous all-face edge-head run learned held-out Face Flow magnitude, but
the signed divergence of those predicted face fluxes did not reconstruct the
node-level HEC-RAS internal volume change.  Two follow-up runs therefore added
an explicit divergence loss:

```text
loss = transformed_face_loss + 0.1 * normalized_divergence_loss
```

where divergence is computed directly from the predicted internal face deltas:

```text
div_i = sum incoming predicted face delta - sum outgoing predicted face delta
```

The target is `graph.hecras_edge_internal_delta`, i.e. the HEC-RAS internal
edge-flow contribution to each node's volume change.  These experiments are
still edge-head-only diagnostics; they are not yet coupled back into the
HydroGraphNet rollout loss.

### All-node divergence loss

Checkpoint:

```text
checkpoints_minxiong_h1h30_edgeflux_head_pretrain_proxy_facetransition_all_h256_l4_div_w01_finetune_seed0
```

Aggregate H25-H30 metrics:

| Checkpoint | Face relative RMSE | Internal divergence relative RMSE | Zone 3 face relative RMSE | Zone 3 divergence relative RMSE |
|---|---:|---:|---:|---:|
| all face-only epoch19 | `0.695773` | `2.334911` | `0.870377` | `20.399533` |
| all-div epoch28 | `0.836630` | `0.465641` | `0.931184` | `3.318986` |
| all-div epoch29 | `0.826326` | `0.466571` | `0.926608` | `3.446900` |

Interpretation:

- This is the first meaningful edge-local conservation improvement.
- Internal divergence relative RMSE improved from `2.334911` to about
  `0.466`, while face relative RMSE stayed below `0.83`.
- Zone 3 divergence also improved strongly compared with face-only, but remains
  too high for a final high-fidelity-zone conservation claim.

### Zone 3 divergence-focused fine-tune

Checkpoint:

```text
checkpoints_minxiong_h1h30_edgeflux_head_pretrain_proxy_facetransition_all_h256_l4_div_w01_highzone_finetune_seed0
```

This run kept all-face face supervision, but changed the divergence loss mask to
`divergence-zone-mode=high`, meaning only `zone_label == 3` nodes contributed
to the divergence loss.

Aggregate H25-H30 metrics:

| Checkpoint | Face relative RMSE | Internal divergence relative RMSE | Zone 3 face relative RMSE | Zone 3 divergence relative RMSE | Edge flux RMS ft^3 |
|---|---:|---:|---:|---:|---:|
| highzone-div epoch33 | `0.992826` | `0.966163` | `1.001446` | `1.816571` | `5490.910` |
| highzone-div epoch34 | `0.992952` | `0.955606` | `1.001558` | `2.050468` | `4880.039` |

Lower high-zone divergence weight:

```text
divergence-loss-weight=0.01
```

| Checkpoint | Face relative RMSE | Internal divergence relative RMSE | Zone 3 face relative RMSE | Zone 3 divergence relative RMSE | Edge flux RMS ft^3 |
|---|---:|---:|---:|---:|---:|
| highzone-div-w001 epoch31 | `0.977827` | `1.004350` | `1.001411` | `1.332608` | `9872.435` |
| highzone-div-w001 epoch32 | `0.969769` | `1.030427` | `1.001246` | `1.303066` | `11431.540` |
| highzone-div-w001 epoch33 | `0.964416` | `1.061119` | `1.000359` | `1.291765` | `12352.244` |

Interpretation:

- Zone 3 divergence relative RMSE improved further, reaching `1.816571` at
  epoch 33 for weight `0.1`.
- Reducing the weight to `0.01` produced a better Zone 3 trade-off, reaching
  Zone 3 divergence relative RMSE `1.291765` while keeping face relative RMSE
  at `0.964416`.
- However, both high-zone-focused runs still reduce edge-flow magnitude compared
  with the all-divergence checkpoint, and all-face/all-node divergence remains
  better for global internal divergence.
- This shows that a high-zone-only divergence loss can reduce local budget
  residuals, but the current loss balance still trades off against physically
  meaningful face-flow magnitude.

Updated status:

```text
Face-flow magnitude: best with face-only/all-face pretraining.
Global internal divergence: best with all-node divergence loss.
Zone 3 local divergence: best current result from high-zone divergence loss
weight 0.01, but with a face-magnitude trade-off.
Formal publishable edge-local conservation: not complete yet.
Next bottleneck: preserve face-flow magnitude while enforcing divergence, likely
through staged weighting, a constrained/projection layer, or a coupled node-edge
HydroGraphNet loss.
```

## 2026-06-25 Update: Sparse Incidence Projection Diagnostic

The divergence-loss fine-tunes showed a clear optimization trade-off:

- face supervision preserves Face Flow magnitude but leaves poor signed
  divergence closure;
- divergence supervision reduces node-budget residuals but can suppress
  physically meaningful face-flow magnitude.

A new diagnostic therefore evaluates a sparse incidence projection after the
edge-head prediction.  The projection solves a regularized least-change problem:

```text
minimize ||q_projected - q_predicted||^2
subject approximately to A_selected q_projected = d_selected
```

where:

- `q_predicted` is the edge head's internal face-flow delta prediction;
- `A_selected` is the HEC-RAS face-to-cell incidence matrix restricted to a
  selected node set;
- `d_selected` is `graph.hecras_edge_internal_delta` on those nodes;
- the closed form is evaluated with sparse conjugate gradient:

```text
q_projected = q_predicted
              - A_selected^T (A_selected A_selected^T + ridge I)^-1
                (A_selected q_predicted - d_selected)
```

New script:

```text
evaluate_projected_edge_flux_head.py
```

Reusable helper module:

```text
hecras_projection.py
```

Compact summary files:

```text
results_minxiong_h25h30_projected_edge_flux_head_compact_summary.csv
results_minxiong_h25h30_projected_edge_flux_head_compact_summary.md
```

This is still a diagnostic, not yet integrated into rollout training.  However,
it directly tests whether local conservation can be enforced while retaining
the learned Face Flow magnitude.

Important caveat:

- The current projection diagnostic uses the HEC-RAS internal delta target
  `graph.hecras_edge_internal_delta`.
- That is useful as an oracle feasibility test, but it is not yet an
  inference-time method.
- A publishable non-oracle projection must use an available target such as
  `predicted node volume delta - boundary/source delta`, then evaluate whether
  the projected edge flux remains close to HEC-RAS Face Flow and improves
  node-budget closure.

### Projection on all-divergence checkpoint

Base checkpoint:

```text
checkpoints_minxiong_h1h30_edgeflux_head_pretrain_proxy_facetransition_all_h256_l4_div_w01_finetune_seed0
epoch 29
```

Original aggregate H25-H30 metrics:

| Metric | Value |
|---|---:|
| Face relative RMSE | `0.826326` |
| Internal divergence relative RMSE | `0.466571` |
| Zone 3 face relative RMSE | `0.926608` |
| Zone 3 divergence relative RMSE | `3.446900` |

High-zone projection (`projection-mode=high`):

| Ridge | Face relative RMSE | Internal divergence relative RMSE | Zone 3 divergence relative RMSE | Projection correction RMS ft^3 |
|---:|---:|---:|---:|---:|
| `0.1` | `0.826306` | `0.465226` | `0.124991` | `466.775` |
| `1` | `0.826308` | `0.464970` | `0.855213` | `355.366` |
| `10` | `0.826318` | `0.465640` | `2.516986` | `121.510` |
| `100` | `0.826324` | `0.466418` | `3.315382` | `16.839` |

All-node projection (`projection-mode=all`):

| Ridge | Face relative RMSE | Internal divergence relative RMSE | Zone 3 divergence relative RMSE | Projection correction RMS ft^3 |
|---:|---:|---:|---:|---:|
| `0.1` | `0.814831` | `0.107415` | `0.719349` | `7857.672` |
| `1` | `0.822575` | `0.209399` | `1.311377` | `3779.566` |
| `10` | `0.825406` | `0.360778` | `2.576357` | `1226.483` |
| `100` | `0.826197` | `0.450333` | `3.317041` | `176.367` |

Interpretation:

- High-zone projection with ridge `0.1` almost perfectly fixes the Zone 3 local
  budget residual while leaving Face Flow accuracy unchanged.
- All-node projection with ridge `0.1` strongly improves global internal
  divergence and even slightly improves face RMSE, but it requires a larger
  correction.
- This supports a two-stage method: let the edge head learn Face Flow magnitude,
  then apply a selective sparse conservation projection to enforce local budget
  closure in fidelity-critical regions.

### Projection on face-only checkpoint

Base checkpoint:

```text
checkpoints_minxiong_h1h30_edgeflux_head_pretrain_proxy_facetransition_all_h256_l4_e10_b80_seed0
epoch 19
```

Original aggregate H25-H30 metrics:

| Metric | Value |
|---|---:|
| Face relative RMSE | `0.695773` |
| Internal divergence relative RMSE | `2.334911` |
| Zone 3 face relative RMSE | `0.870377` |
| Zone 3 divergence relative RMSE | `20.399533` |

Projection results:

| Projection | Ridge | Face relative RMSE | Internal divergence relative RMSE | Zone 3 face relative RMSE | Zone 3 divergence relative RMSE | Projection correction RMS ft^3 |
|---|---:|---:|---:|---:|---:|---:|
| high | `0.1` | `0.695038` | `2.321444` | `0.845481` | `0.641260` | `2610.177` |
| all | `0.1` | `0.628700` | `0.131457` | `0.832957` | `1.240389` | `23905.516` |

Interpretation:

- Face-only training plus projection is currently the strongest combined
  result.
- High-zone projection fixes Zone 3 divergence while preserving the strongest
  Face Flow magnitude.
- All-node projection yields the best global local-conservation metric and also
  improves face RMSE, but with a much larger correction.

Updated method direction:

```text
Most promising publishable path:
1. Train an edge head primarily for HEC-RAS Face Flow magnitude.
2. Add a sparse incidence projection layer or differentiable projection penalty.
3. Apply projection selectively by Automatic Fidelity Zoning for the main
   method, and report all-node projection as a diagnostic/upper-bound variant.
4. Replace oracle HEC-RAS projection targets with inference-available closure
   targets from the HydroGraphNet node model.
5. Couple the projected edge flux back into HydroGraphNet rollout and compare
   noface, face-only, divergence-loss, high-zone projection, and all-node
   projection.
```

## 2026-06-26 Update: Non-oracle Closure-target Projection

The 2026-06-25 projection results used `graph.hecras_edge_internal_delta` as an
oracle projection target.  That test answered whether a sparse projection can
mathematically enforce local conservation without destroying Face Flow
magnitude, but it cannot be used as the final inference-time method.

New script:

```text
evaluate_nonoracle_projected_edge_flux_head.py
```

This evaluator loads a HydroGraphNet node model and a pretrained edge head
separately.  The projection target can be:

- `hecras_internal`: oracle HEC-RAS internal delta, for upper-bound diagnosis.
- `closure`: inference-available target
  `predicted node volume delta - boundary/source delta`.

The `closure` target is the important non-oracle setting because it does not
directly use the HEC-RAS internal edge-flow target during projection.

### Baseline node model + face-only edge head

Node model:

```text
checkpoints_new_seed0_baseline_h1h10_b50
epoch 0
```

Edge head:

```text
checkpoints_minxiong_h1h30_edgeflux_head_pretrain_proxy_facetransition_all_h256_l4_e10_b80_seed0
epoch 19
```

Projection target:

```text
closure = predicted node volume delta - boundary/source delta
```

Aggregate H25-H30 results:

| Projection | Face rel RMSE | Closure rel RMSE | Internal divergence rel RMSE | Zone 3 face rel RMSE | Zone 3 closure rel RMSE | Zone 3 internal divergence rel RMSE | Correction RMS ft^3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| none/original | `0.695773` | `2.330175` | `2.334911` | `0.870377` | `8.036618` | `20.399533` | `0` |
| high closure projection | `0.695077` | `2.317166` | `2.322020` | `0.846824` | `0.367212` | `2.486171` | `2713.898` |
| all closure projection | `0.629796` | `0.140481` | `0.158183` | `0.834233` | `0.793557` | `2.123495` | `23893.720` |

Interpretation:

- High-zone non-oracle projection strongly improves Zone 3 closure, but it does
  not improve global closure because only Zone 3 nodes are constrained.
- All-node non-oracle projection gives a strong formal conservation result:
  global closure relative RMSE drops from `2.330175` to `0.140481`, and internal
  divergence relative RMSE drops from `2.334911` to `0.158183`.
- It also improves Face Flow relative RMSE from `0.695773` to `0.629796`,
  despite using a non-oracle closure target.
- Zone 3 internal divergence remains higher than the oracle projection result,
  which means the baseline node model's closure target is still imperfect in
  the high-fidelity zone.

### Baseline node model + divergence fine-tuned edge head

Edge head:

```text
checkpoints_minxiong_h1h30_edgeflux_head_pretrain_proxy_facetransition_all_h256_l4_div_w01_finetune_seed0
epoch 29
```

All-node closure projection:

| Projection | Face rel RMSE | Closure rel RMSE | Internal divergence rel RMSE | Zone 3 face rel RMSE | Zone 3 closure rel RMSE | Zone 3 internal divergence rel RMSE | Correction RMS ft^3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| none/original | `0.826326` | `0.468794` | `0.466571` | `0.926608` | `1.654363` | `3.446900` | `0` |
| all closure projection | `0.815748` | `0.117445` | `0.138795` | `0.927391` | `0.659498` | `1.881570` | `7835.636` |

Interpretation:

- Divergence fine-tuning provides a better starting closure metric than the
  face-only edge head.
- After non-oracle all-node projection, it reaches slightly better closure and
  internal divergence metrics than the face-only edge head.
- However, its Face Flow accuracy is much worse (`0.815748` versus `0.629796`
  for face-only edge + all closure projection).
- For a prediction-accuracy/conservation trade-off, the current best candidate
  is still face-only edge pretraining followed by non-oracle all-node closure
  projection.  For a conservation-heavy ablation, all-divergence edge
  pretraining followed by all-node closure projection is useful.

Updated status after non-oracle test:

```text
Oracle projection: proves sparse incidence projection can almost eliminate
selected local budget residuals.

Non-oracle closure projection: now works and is the first inference-available
projection evidence.

Best balanced candidate so far:
face-only edge head + all-node closure projection
  face relative RMSE = 0.629796
  closure relative RMSE = 0.140481
  internal divergence relative RMSE = 0.158183

Remaining bottleneck:
the HydroGraphNet node model's closure target still limits Zone 3 local
conservation.  The next publishable step is to train/evaluate a stronger node
model and then rerun non-oracle high-zone and all-node projection.
```

## 2026-06-26 Update: Rollout Projection Diagnostic

New script:

```text
evaluate_rollout_projected_edge_flux.py
```

This moves the projection check from one-step graphs to autoregressive rollout.
The script keeps the HydroGraphNet state rollout unchanged and evaluates, at
each rollout step, whether the edge head can be projected to close the
node-model volume delta.  It is therefore a multi-step conservation diagnostic,
not yet a corrected-state rollout.

Important target distinction:

- `node_delta`: projects internal face divergence directly to node volume
  delta.  This is physically incomplete because the incidence divergence sums
  to zero while total node volume change includes boundary/source terms.
- `hecras_boundary_source`: projects to
  `node volume delta - boundary/source delta`, matching the local conservation
  form used by the one-step non-oracle closure projection.

The current H25-H30 test split has limited sequence length:

| Event | Time steps | Max rollout length |
|---|---:|---:|
| H25 | `27` | `25` |
| H26 | `25` | `23` |
| H27 | `29` | `27` |
| H28 | `27` | `25` |
| H29 | `25` | `23` |
| H30 | `27` | `25` |

Therefore the fair common maximum rollout length for H25-H30 is `23`, not
`30` or `48`.

### Face-only edge head + all-node projection

Node model:

```text
checkpoints_new_seed0_baseline_h1h10_b50
epoch 0
```

Edge head:

```text
checkpoints_minxiong_h1h30_edgeflux_head_pretrain_proxy_facetransition_all_h256_l4_e10_b80_seed0
epoch 19
```

Aggregate H25-H30 rollout results:

| Rollout length | Projection target | Rollout RMSE | WD RMSE | Volume RMSE | Original closure rel RMSE | Projected closure rel RMSE | Zone 3 projected closure rel RMSE | Correction RMS ft^3 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `10` | `node_delta` | `0.244453` | `0.280805` | `0.201273` | `16.019346` | `1.089021` | `0.652210` | `13645.784` |
| `10` | `hecras_boundary_source` | `0.244453` | `0.280805` | `0.201273` | `1.769619` | `0.130243` | `0.652210` | `15851.602` |
| `23` | `hecras_boundary_source` | `0.459265` | `0.509919` | `0.401308` | `1.933633` | `0.133072` | `0.711253` | `20343.921` |

Interpretation:

- The physically incomplete `node_delta` target fails in rollout because it
  ignores boundary/source terms.
- Adding boundary/source terms gives stable multi-step closure: projected
  closure relative RMSE remains about `0.13` at rollout lengths `10` and `23`.
- The rollout state RMSE is unchanged by design in this diagnostic because the
  projected edge flux is not yet fed back into the node state update.
- Zone 3 projected closure remains around `0.65--0.71`, so high-fidelity-zone
  closure still needs either a stronger node model or a high-zone-specific
  projection/training variant.

Updated next step:

```text
1. Keep face-only edge head + boundary/source-aware projection as the main
   inference-available method candidate.
2. Extend the rollout evaluator to optionally feed projected divergence back
   into the volume update, then compare state RMSE with and without correction.
3. Train/evaluate a stronger node model so the projection target is more
   accurate in Zone 3.
4. Re-run rollout diagnostics after more events or longer H25-H30-style test
   sequences are generated, because the current common max rollout is 23.
```

## 2026-06-26 Update: Projected Volume State Feedback

The rollout evaluator now supports:

```text
--state-update original
--state-update projected_volume
```

`original` keeps the HydroGraphNet node-model autoregressive state unchanged and
uses the projected edge flux only as a conservation diagnostic.  `projected_volume`
feeds the projected local-conservation volume update back into the next rollout
state:

```text
projected total volume delta = projected internal divergence + boundary/source delta
```

This is the first diagnostic that tests whether the edge/local-conservation
correction can be coupled back into the HydroGraphNet rollout state, rather than
only evaluated after the node model has already predicted the next state.

Aggregate H25-H30 results with the same node model and face-only edge head:

| Rollout length | State update | Rollout RMSE | WD RMSE | Volume RMSE | Original closure rel RMSE | Projected closure rel RMSE | Zone 3 projected closure rel RMSE | Correction RMS ft^3 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `10` | `original` | `0.244453` | `0.280805` | `0.201273` | `1.769619` | `0.130243` | `0.652210` | `15851.602` |
| `10` | `projected_volume` | `0.271897` | `0.279073` | `0.263331` | `1.732462` | `0.130159` | `0.661915` | `15487.967` |
| `23` | `original` | `0.459265` | `0.509919` | `0.401308` | `1.933633` | `0.133072` | `0.711253` | `20343.921` |
| `23` | `projected_volume` | `0.563433` | `0.508885` | `0.609455` | `2.052550` | `0.128465` | `0.717116` | `20659.978` |

Interpretation:

- Boundary/source-aware projection consistently reduces closure relative RMSE to
  about `0.13`, including when the projected volume is fed back into the rollout.
- Feeding the projected volume state back into the autoregressive rollout
  increases volume/state RMSE for the current weak node checkpoint:
  - rollout length `10`: `0.244453 -> 0.271897`
  - rollout length `23`: `0.459265 -> 0.563433`
- The water-depth RMSE changes little at length `23`
  (`0.509919 -> 0.508885`), while volume RMSE worsens
  (`0.401308 -> 0.609455`).  This suggests the current projected update is
  improving closure but is not yet dynamically calibrated with the node model's
  learned state evolution.
- This is an important paper-level distinction: the method has an
  inference-available edge/local-conservation projection, but the corrected-state
  rollout is not yet a final improved forecasting method.

Updated next step:

```text
1. Treat projected-volume feedback as a required research target, not as solved.
2. Train a stronger node model and repeat projected-state rollout, because the
   current checkpoint is an early baseline checkpoint.
3. Add a loss/training path that aligns node volume deltas with the
   boundary/source-aware projected edge divergence, instead of only projecting at
   inference time.
4. Keep reporting both conservation metrics and rollout prediction metrics; a
   conservation-only improvement is not sufficient for the final paper claim.
```

## 2026-06-26 Update: Stronger H1-H24 Baseline Node Checkpoint

To separate edge/local-conservation behavior from a weak node-model checkpoint,
a cleaner baseline HydroGraphNet node model was trained on the H1-H24 training
split.

Training command summary:

```text
data_dir=/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset
hydrograph_ids_file=train_h1h24.txt
epochs=20
num_training_samples=600
use_physics_loss=true
use_fidelity_zones=false
use_hecras_edge_flux_head=false
use_hecras_edge_flow_loss=false
ckpt_path=./checkpoints_minxiong_h1h24_baseline_node_e20_seed0
```

Final training metrics:

| Epoch | Total loss | MSE loss | Physics loss |
|---:|---:|---:|---:|
| `0` | `1.1048e-01` | `3.1268e-03` | `1.0736e-01` |
| `12` | `1.7327e-03` | `8.3310e-05` | `1.6494e-03` |
| `19` | `1.9815e-03` | `6.9398e-05` | `1.9121e-03` |

There was a temporary physics-loss spike around epochs `13--14`, but the model
returned to the `~2e-03` loss range by epoch `19`.

The epoch-19 checkpoint was then used for H25-H30 rollout projection tests with
the existing face-only edge head:

```text
node checkpoint: checkpoints_minxiong_h1h24_baseline_node_e20_seed0, epoch 19
edge checkpoint: checkpoints_minxiong_h1h30_edgeflux_head_pretrain_proxy_facetransition_all_h256_l4_e10_b80_seed0, epoch 19
projection mode: all
projection target: hecras_boundary_source
projection ridge: 0.1
```

Aggregate H25-H30 results:

| Rollout length | State update | Rollout RMSE | WD RMSE | Volume RMSE | Original closure rel RMSE | Projected closure rel RMSE | Zone 3 projected closure rel RMSE | Correction RMS ft^3 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `10` | `original` | `0.043131` | `0.040755` | `0.045374` | `1.646275` | `0.121125` | `1.683011` | `14921.541` |
| `10` | `projected_volume` | `0.160526` | `0.040859` | `0.223254` | `1.632316` | `0.116503` | `1.670311` | `14665.200` |
| `23` | `original` | `0.084183` | `0.079789` | `0.088327` | `1.640814` | `0.126297` | `2.337306` | `18023.394` |
| `23` | `projected_volume` | `0.385997` | `0.079778` | `0.539867` | `1.790712` | `0.119305` | `2.263103` | `18965.256` |

Interpretation:

- The stronger node checkpoint greatly improves rollout prediction when the
  HydroGraphNet state is left unchanged:
  - previous epoch-0 node, length `23`: rollout RMSE `0.459265`
  - new epoch-19 node, length `23`: rollout RMSE `0.084183`
- Boundary/source-aware projection still gives strong global closure
  improvement:
  - length `10`: `1.646275 -> 0.121125`
  - length `23`: `1.640814 -> 0.126297`
- Directly feeding projected volume back into the autoregressive state still
  hurts the volume/state trajectory:
  - length `10`: rollout RMSE `0.043131 -> 0.160526`
  - length `23`: rollout RMSE `0.084183 -> 0.385997`
- Water-depth RMSE changes very little under projected-volume feedback, while
  volume RMSE worsens strongly.  This suggests the current coupling mainly
  disturbs the volume channel and needs a learned or damped state-correction
  mechanism.
- Zone 3 projected closure is not solved by the all-node projection
  (`~1.7--2.3` relative RMSE in these runs).  A paper-level edge-local
  conservation method still needs a high-fidelity-zone-specific objective or
  projection mode, not only global all-node closure projection.

Updated next step:

```text
1. Keep the epoch-19 H1-H24 baseline node checkpoint as the current rollout
   baseline for H25-H30.
2. Add a damping/blending parameter for projected-volume state feedback:
   volume_next = original_volume_next + alpha * (projected_volume_next - original_volume_next).
3. Sweep alpha values such as 0.05, 0.1, 0.25, 0.5, and 1.0 on H25-H30.
4. Add high-zone projection diagnostics or high-zone-weighted projection to
   directly target the fidelity-zone research claim.
```

## 2026-06-26 Update: Damped Projected-Volume Feedback

The rollout evaluator now supports a damping/blending parameter:

```text
--projected-volume-alpha
```

When `--state-update=projected_volume`, the next rollout volume is updated as:

```text
volume_next = original_volume_next + alpha * (projected_volume_next - original_volume_next)
```

This keeps the edge/local-conservation projection unchanged, but controls how
strongly the projected volume correction is fed back into the autoregressive
HydroGraphNet state.

H25-H30 results using the stronger H1-H24 epoch-19 node checkpoint:

| Rollout length | State update | Alpha | Rollout RMSE | WD RMSE | Volume RMSE | Projected closure rel RMSE | Zone 3 projected closure rel RMSE |
|---:|---|---:|---:|---:|---:|---:|---:|
| `10` | `original` | `0.0` | `0.043131` | `0.040755` | `0.045374` | `0.121125` | `1.683011` |
| `10` | `projected_volume` | `0.1` | `0.045065` | `0.040770` | `0.048975` | `0.119616` | `1.685470` |
| `10` | `projected_volume` | `0.25` | `0.057087` | `0.040789` | `0.069615` | `0.119217` | `1.687471` |
| `10` | `projected_volume` | `1.0` | `0.160526` | `0.040859` | `0.223254` | `0.116503` | `1.670311` |
| `23` | `original` | `0.0` | `0.084183` | `0.079789` | `0.088327` | `0.126297` | `2.337306` |
| `23` | `projected_volume` | `0.1` | `0.090963` | `0.079791` | `0.100863` | `0.126092` | `2.342238` |
| `23` | `projected_volume` | `1.0` | `0.385997` | `0.079778` | `0.539867` | `0.119305` | `2.263103` |

Interpretation:

- A small projected-volume feedback (`alpha=0.1`) keeps the rollout prediction
  close to the original HydroGraphNet state rollout:
  - length `10`: `0.043131 -> 0.045065`
  - length `23`: `0.084183 -> 0.090963`
- Full feedback (`alpha=1.0`) is too aggressive and mainly damages the volume
  channel:
  - length `23` volume RMSE: `0.088327 -> 0.539867`
- The projected closure metric stays near `0.12` across alpha values because
  alpha controls state feedback, not the edge-flux projection itself.
- This gives a more defensible research direction: edge-informed local
  conservation can be made inference-available and coupled back into rollout,
  but the coupling must be damped or learned.
- Zone 3 closure is still poor relative to the global projected closure, so the
  Automatic Fidelity Zoning contribution still needs a high-zone-aware
  projection or training objective.

Updated next step:

```text
1. Use alpha=0.1 as the current stable projected-volume feedback candidate.
2. Run a finer alpha sweep around 0.05--0.2 on H25-H30.
3. Add high-zone-weighted projection and compare global closure vs Zone 3
   closure.
4. Consider learning alpha or predicting a correction gate from node/edge
   features instead of manually choosing a fixed alpha.
```

## 2026-06-26 Update: High-Zone Feedback Mask

An important rollout-evaluator issue was found while testing
`projection_mode=high`: the projected volume update was initially fed back to
all nodes, even when the projection only constrained high-fidelity nodes.  This
made non-projected nodes use raw edge-head divergence as their volume update and
caused catastrophic volume errors.

The evaluator was updated so projected-volume feedback is applied only to nodes
selected by the projection mode:

```text
all           -> feedback all nodes
high          -> feedback only Zone 3 nodes
high_interior -> feedback only interior Zone 3 nodes
```

After this fix, `projection_mode=high` no longer explodes.

H25-H30, rollout length `23`, alpha `0.1`, stronger H1-H24 epoch-19 node model:

| Projection mode | Rollout RMSE | WD RMSE | Volume RMSE | Global projected closure rel RMSE | Zone 3 projected closure rel RMSE | Zone 3 rollout RMSE | Zone 3 volume RMSE | Correction RMS ft^3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `all` | `0.090963` | `0.079791` | `0.100863` | `0.126092` | `2.342238` | `0.108210` | `0.066276` | `17570.835` |
| `high` | `0.084146` | `0.079790` | `0.088255` | `1.633346` | `1.268259` | `0.109005` | `0.068881` | `1781.192` |
| `high_interior` | `0.084148` | `0.079790` | `0.088262` | `1.637266` | `29.696144` | `0.109026` | `0.068964` | `979.863` |

Interpretation:

- `all` projection gives much better global closure but does not solve Zone 3
  closure.
- `high` projection improves Zone 3 closure substantially
  (`2.342238 -> 1.268259`) while keeping rollout RMSE essentially at the
  original-state level.
- `high` projection worsens global closure because non-high nodes are not being
  constrained.  This is expected and should be reported as a locality trade-off,
  not as a failure.
- `high_interior` is not a good Zone 3 metric candidate in the current form:
  rollout remains stable, but the all-Zone-3 closure statistic becomes very
  poor because Zone 3 boundary/transition nodes are excluded from the projection
  while still included in the Zone 3 metric.
- The correction RMS is much smaller for high-zone projection, which supports
  the idea that targeted local conservation can be less intrusive than
  all-node projection.

Updated next step:

```text
1. Treat `all` and `high` projection as two different ablations:
   - all: best global closure
   - high: better fidelity-zone-local closure with low rollout disruption
2. Do not use `high_interior` as the main result unless the metric is changed to
   evaluate only high-interior nodes.
3. Add a weighted projection mode to reduce Zone 3 closure without fully
   ignoring surrounding transition nodes.
4. Report global closure, Zone 3 closure, rollout RMSE, and correction RMS
   together; no single metric is sufficient.
```

## 2026-06-26 Update: Zone-Weighted Projection

A new projection mode was added:

```text
projection_mode=zone_weighted
```

This keeps all nodes in the projection system but gives Zone 3 a larger residual
weight:

```text
weighted residual = sqrt(w_i) * (divergence_i - target_i)
```

The goal is to avoid the two extremes observed earlier:

- `all`: excellent global closure, but poor Zone 3 closure.
- `high`: better Zone 3 closure and low rollout disruption, but poor global
  closure because non-high nodes are not constrained.

Implementation details:

- `hecras_projection.py` now supports residual weights via:
  - `projection_high_weight`
  - `projection_low_weight`
- `evaluate_rollout_projected_edge_flux.py` records both weights in the output
  CSV.
- Existing projection modes default to weights of `1.0`, so previous all/high
  behavior is unchanged.

H25-H30, rollout length `23`, alpha `0.1`, stronger H1-H24 epoch-19 node model:

| Projection mode | High weight | Low weight | Rollout RMSE | Volume RMSE | Global projected closure rel RMSE | Zone 3 projected closure rel RMSE | Zone 3 rollout RMSE | Zone 3 volume RMSE | Correction RMS ft^3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `all` | `-` | `-` | `0.090963` | `0.100863` | `0.126092` | `2.342238` | `0.108210` | `0.066276` | `17570.835` |
| `zone_weighted` | `10` | `1` | `0.090949` | `0.100844` | `0.125933` | `0.393517` | `0.108895` | `0.068534` | `17583.547` |
| `zone_weighted` | `50` | `1` | `0.090956` | `0.100858` | `0.126002` | `0.099300` | `0.109178` | `0.069439` | `17586.541` |
| `high` | `-` | `-` | `0.084146` | `0.088255` | `1.633346` | `1.268259` | `0.109005` | `0.068881` | `1781.192` |

Interpretation:

- `zone_weighted` keeps global closure essentially equal to `all`:
  - all: `0.126092`
  - weighted high=50: `0.126002`
- `zone_weighted` dramatically improves Zone 3 closure:
  - all: `2.342238`
  - weighted high=10: `0.393517`
  - weighted high=50: `0.099300`
- Rollout RMSE remains nearly unchanged compared with all projection:
  - all: `0.090963`
  - weighted high=50: `0.090956`
- This is currently the strongest candidate for the paper claim because it
  preserves graph-level/global conservation while explicitly prioritizing the
  high-fidelity zone.
- The correction RMS remains close to all projection.  This means the weighted
  method is not necessarily less intrusive globally, but it redistributes the
  correction toward satisfying Zone 3.
- Solver cost increases for weighted all-node projection, especially with large
  high/low weight contrast.  A production-quality method should cache the
  projection matrix or use a reusable factorization/preconditioner.

Current best candidate:

```text
projection_mode=zone_weighted
projection_high_weight=50
projection_low_weight=1
projection_target=hecras_boundary_source
state_update=projected_volume
projected_volume_alpha=0.1
```

Updated next step:

```text
1. Treat zone_weighted high=50, low=1 as the current best edge-local
   conservation candidate.
2. Add projection-matrix caching or factorization reuse; the current weighted
   all-node projection is too slow for broad sweeps.
3. Run the same weighted projection on additional events when more HEC-RAS/HGN
   event data are generated.
4. Add a train-time loss or learned gating mechanism so alpha/weights are not
   purely manual hyperparameters.
```

## 2026-06-27 Update: Factorized Projection Solver and Local Sweep

The projection helper now supports a reusable projection solver:

```text
--projection-solver cg
--projection-solver factorized
```

For a fixed graph, projection mode, ridge, and zone-weight setting, the
factorized solver builds the weighted projection system once and reuses the
sparse factorization across rollout steps.  This makes the zone-weighted
projection practical for sweeps.

Validation check:

```text
zone_weighted high=50 low=1 alpha=0.1
cg/legacy result:        rollout RMSE 0.090956, Zone 3 closure 0.099300
factorized result:       rollout RMSE 0.090956, Zone 3 closure 0.099300
factorized elapsed time: ~37.39 seconds for H25-H30 length-23 rollout
```

After the first factorized run, repeated configurations completed in about
`15` seconds each.  This is a large improvement over the earlier weighted
projection runs, which took several minutes.

### Local Sweep

H25-H30, rollout length `23`, stronger H1-H24 epoch-19 node model:

| Projection mode | High weight | Alpha | Solver | Rollout RMSE | Volume RMSE | Global projected closure rel RMSE | Zone 3 projected closure rel RMSE | Zone 3 rollout RMSE | Correction RMS ft^3 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `all` | `-` | `0.0` | `cg/legacy` | `0.084183` | `0.088327` | `0.126297` | `2.337306` | `0.109291` | `18023.394` |
| `zone_weighted` | `10` | `0.1` | `cg/legacy` | `0.090949` | `0.100844` | `0.125933` | `0.393517` | `0.108895` | `17583.547` |
| `zone_weighted` | `25` | `0.1` | `factorized` | `0.090953` | `0.100853` | `0.125978` | `0.183840` | `0.109090` | `17585.591` |
| `zone_weighted` | `50` | `0.05` | `factorized` | `0.085161` | `0.090192` | `0.124965` | `0.099185` | `0.109234` | `17562.077` |
| `zone_weighted` | `50` | `0.1` | `factorized` | `0.090956` | `0.100858` | `0.126002` | `0.099300` | `0.109178` | `17586.541` |
| `zone_weighted` | `50` | `0.2` | `factorized` | `0.113306` | `0.138763` | `0.126102` | `0.099445` | `0.109068` | `17568.375` |
| `zone_weighted` | `100` | `0.05` | `factorized` | `0.085161` | `0.090192` | `0.124980` | `0.051953` | `0.109260` | `17562.672` |
| `zone_weighted` | `100` | `0.1` | `factorized` | `0.090957` | `0.100862` | `0.126017` | `0.052013` | `0.109229` | `17587.116` |
| `high` | `-` | `0.1` | `cg/legacy` | `0.084146` | `0.088255` | `1.633346` | `1.268259` | `0.109005` | `1781.192` |

Interpretation:

- Increasing the high-fidelity-zone residual weight improves Zone 3 closure:
  - high weight `10`: `0.393517`
  - high weight `25`: `0.183840`
  - high weight `50`: `0.099300`
  - high weight `100`: `0.052013`
- Reducing the feedback alpha from `0.1` to `0.05` preserves almost all of the
  original rollout accuracy:
  - original all, no projected state feedback: rollout RMSE `0.084183`
  - zone_weighted high=100, alpha=0.05: rollout RMSE `0.085161`
- The global projected closure remains about `0.125`, so the method preserves
  global graph-level conservation while strongly improving Zone 3 local closure.
- The strongest current balance is:

```text
projection_mode=zone_weighted
projection_high_weight=100
projection_low_weight=1
projection_solver=factorized
projection_target=hecras_boundary_source
state_update=projected_volume
projected_volume_alpha=0.05
```

This setting reduces Zone 3 projected closure relative RMSE from `2.337306` to
`0.051953`, while increasing rollout RMSE only from `0.084183` to `0.085161`.

Updated next step:

```text
1. Use zone_weighted high=100, low=1, alpha=0.05 as the current best inference
   projection candidate.
2. Generate report figures comparing all/high/zone_weighted:
   - rollout RMSE
   - global closure
   - Zone 3 closure
   - correction RMS
3. Move from inference-time projection toward train-time or learned-gated
   edge-local conservation.
4. Re-run the candidate on more events once additional HEC-RAS/HGN cases are
   available.
```

## 2026-06-27 Update: Spatial Residual Evidence

A spatial residual plotting script was added:

```text
plot_zone_weighted_spatial_residual.py
```

It computes per-node rollout RMS closure residuals for a selected event and
plots:

```text
1. original edge-head closure residual
2. zone-weighted projected closure residual
3. log10(original / projected) improvement
```

The current H25 map uses:

```text
projection_mode=zone_weighted
projection_high_weight=100
projection_low_weight=1
projection_solver=factorized
projected_volume_alpha=0.05
rollout_length=23
```

Generated report assets:

```text
/mnt/8tb_hdd2/joyce/Report/assets/minxiong_H25_zone_weighted_spatial_residual.png
/mnt/8tb_hdd2/joyce/Report/assets/minxiong_H25_zone_weighted_spatial_residual_summary.csv
```

H25 node-level summary from the residual CSV:

| Zone | Nodes | Median improvement ratio | Mean original residual ft^3 | Mean projected residual ft^3 |
|---:|---:|---:|---:|---:|
| `0` | `6193` | `1.659` | `10776.047` | `1443.278` |
| `1` | `3182` | `1.128` | `9450.037` | `976.764` |
| `2` | `2066` | `0.997` | `5706.214` | `942.975` |
| `3` | `1274` | `32.089` | `2849.818` | `17.777` |

Interpretation:

- The spatial map gives visual evidence that the zone-weighted projection is not
  only improving an aggregate scalar metric.
- Zone 3, the high-fidelity zone, shows the strongest local-conservation
  improvement: median node improvement is about `32x`.
- Mean Zone 3 closure residual drops from about `2849.8 ft^3` to `17.8 ft^3`
  on H25.
- Zone 2 is roughly neutral by median ratio, which is acceptable for the
  current method because the projection is intentionally prioritizing Zone 3.

Updated next step:

```text
1. Generate the same spatial residual maps for H26-H30 or a representative
   subset.
2. Add a report-ready section that combines:
   - metric table
   - trade-off plot
   - H25 spatial residual map
3. Start designing the train-time/learned-gating version so the current
   inference projection becomes a model-integrated method.
```

## 2026-06-28 Update: H25-H30 Spatial Residual Maps

Spatial residual maps were generated for all H25-H30 test events using the
current best setting:

```text
projection_mode=zone_weighted
projection_high_weight=100
projection_low_weight=1
projection_solver=factorized
projection_target=hecras_boundary_source
state_update=projected_volume
projected_volume_alpha=0.05
rollout_length=23
```

Generated assets:

```text
/mnt/8tb_hdd2/joyce/Report/assets/minxiong_H25_zone_weighted_spatial_residual.png
/mnt/8tb_hdd2/joyce/Report/assets/minxiong_H26_zone_weighted_spatial_residual.png
/mnt/8tb_hdd2/joyce/Report/assets/minxiong_H27_zone_weighted_spatial_residual.png
/mnt/8tb_hdd2/joyce/Report/assets/minxiong_H28_zone_weighted_spatial_residual.png
/mnt/8tb_hdd2/joyce/Report/assets/minxiong_H29_zone_weighted_spatial_residual.png
/mnt/8tb_hdd2/joyce/Report/assets/minxiong_H30_zone_weighted_spatial_residual.png
/mnt/8tb_hdd2/joyce/Report/assets/minxiong_H25_H30_zone_weighted_spatial_residual_zone_summary.csv
```

Zone 3 spatial residual summary:

| Event | Median improvement ratio | Mean original residual ft^3 | Mean projected residual ft^3 | Mean residual reduction |
|---|---:|---:|---:|---:|
| `H25` | `32.089` | `2849.818` | `17.777` | `99.38%` |
| `H26` | `32.535` | `3848.799` | `17.036` | `99.56%` |
| `H27` | `34.059` | `3266.981` | `19.117` | `99.41%` |
| `H28` | `33.631` | `3206.466` | `17.754` | `99.45%` |
| `H29` | `32.653` | `2665.717` | `18.597` | `99.30%` |
| `H30` | `31.545` | `3170.759` | `15.546` | `99.51%` |

Aggregate spatial interpretation:

- Zone 3 median improvement ratio across H25-H30 is about `32.594x`.
- Mean Zone 3 residual reduction across H25-H30 is about `99.43%`.
- Zone 3 is consistently improved across every held-out test event, not only in
  H25.
- Other zones also often improve in mean residual, but Zone 3 is the intended
  high-fidelity target and receives the strongest improvement.

All-zone aggregate reduction summary:

| Zone | Median of event median-improvement ratios | Mean residual reduction |
|---:|---:|---:|
| `0` | `1.768` | `87.25%` |
| `1` | `1.144` | `89.73%` |
| `2` | `1.000` | `84.07%` |
| `3` | `32.594` | `99.43%` |

Updated next step:

```text
1. Add these spatial maps and the H25-H30 zone summary to the progress report.
2. Generate a compact report page/table that explains why zone_weighted h100
   alpha=0.05 is the current best candidate.
3. Begin train-time integration planning:
   - differentiable weighted local-conservation loss
   - learned alpha/gating
   - validation-based automatic high-zone weighting
```

## 2026-06-28 Update: Train-Time Zone-Weighted Edge Loss Smoke

The train-time edge-flux-head loss now supports the same high-zone weighted
logic used by the inference projection:

```text
hecras_edge_flux_head_zone_mode=zone_weighted
hecras_edge_flux_head_zone_high_weight=100
hecras_edge_flux_head_zone_low_weight=1
```

Code changes:

- `compute_hecras_edge_flux_head_loss(...)` now accepts:
  - `zone_high_weight`
  - `zone_low_weight`
- Node closure/divergence residuals use Zone 3 high weights and non-Zone-3 low
  weights when `zone_mode=zone_weighted`.
- Face target residuals also use zone-weighted face weights, computed from the
  average of source/destination node weights.
- `train.py`, `conf/config.yaml`, and `evaluate_edge_flux_head.py` now expose
  and pass these weights.

Smoke evaluation:

```text
evaluate_edge_flux_head.py
zone_mode=zone_weighted
zone_high_weight=100
zone_low_weight=1
output=results_minxiong_h25h30_edge_flux_head_zone_weighted_h100_smoke_eval.csv
```

This verified that the selected loss path runs and records the zone-weighted
loss metrics.  The evaluation checkpoint lacked a saved `MeshGraphKAN` file in
that edge-head-only checkpoint directory, so this is a loss-path smoke check,
not a formal model-quality result.

Train smoke:

```text
train.py
data=train_h1h24.txt
epochs=1
max_train_batches=2
use_hecras_edge_flux_head=true
hecras_edge_flux_head_zone_mode=zone_weighted
hecras_edge_flux_head_zone_high_weight=100
hecras_edge_flux_head_zone_low_weight=1
hecras_edge_flux_head_loss_weight=1e-8
ckpt_path=./checkpoints_minxiong_h1h24_zoneweighted_edgeflux_train_smoke_seed0
```

Smoke result:

| Metric | Value |
|---|---:|
| `hecras_edge_flux_closure_loss` | `2.7214e+09` |
| `hecras_edge_flux_divergence_loss` | `1.5266e+09` |
| `hecras_edge_flux_face_loss` | `6.7819e-01` |
| `hecras_edge_flux_head_loss` | `4.2480e+09` |
| `mse_loss` | `1.4543e-01` |
| `physics_loss` | `1.9630e+01` |
| `total_loss` | `6.2256e+01` |

The train loop successfully:

- loaded the H1-H24 HEC-RAS edge-flow data,
- instantiated `MeshGraphKAN` and `HecRasEdgeFluxHead`,
- computed the zone-weighted edge local conservation loss,
- backpropagated through the combined loss,
- saved both `MeshGraphKAN` and `HecRasEdgeFluxHead` checkpoints.

Important limitation:

This is a train-time integration smoke test only.  It proves the path works, but
it is not yet evidence that training with this loss improves held-out rollout
quality.  A longer controlled training/evaluation experiment is still required.

Updated next step:

```text
1. Run a controlled short training experiment with zone_weighted train-time edge
   loss and compare against the current inference-projection baseline.
2. Use a conservative loss weight sweep because the raw closure/divergence
   losses are large:
   - 1e-10
   - 3e-10
   - 1e-9
   - 3e-9
3. Evaluate each checkpoint with:
   - rollout RMSE
   - global closure
   - Zone 3 closure
   - spatial residual maps
```

## 2026-06-28 Update: Controlled Train-Time / Warm-Start Results

Three controlled training/evaluation variants were tested after the train-time
zone-weighted edge-loss path was verified.

### 1. From-scratch short train is not sufficient

```text
ckpt_path=./checkpoints_minxiong_h1h24_zoneweighted_edgeflux_w1e10_e5_b40_seed0
epochs=5
max_train_batches=40
hecras_edge_flux_head_loss_weight=1e-10
```

Training loss decreased, but the node model was still under-trained.  Held-out
H25-H30 rollout RMSE increased to `0.395651`, so this run should be treated as
a negative control rather than a useful model-quality result.

### 2. Warm-start joint fine-tune works but hurts rollout

```text
warm-start node weights:
checkpoints_minxiong_h1h24_baseline_node_e20_seed0/MeshGraphKAN.0.19.mdlus

ckpt_path=./checkpoints_minxiong_h1h24_baseline_e20_warmstart_zoneweighted_edgeflux_w1e11_e5_b40_seed0
epochs=5
max_train_batches=40
hecras_edge_flux_head_loss_weight=1e-11
```

The run initialized `MeshGraphKAN` from the baseline e19 weights and trained a
new `HecRasEdgeFluxHead`.  The old optimizer state was intentionally not loaded
because the edge head adds new parameters and the baseline optimizer parameter
group is incompatible.

Training metrics improved over the 5-epoch run:

| Epoch | Edge head loss | MSE loss | Physics loss | Total loss |
|---:|---:|---:|---:|---:|
| `0` | `2.1691e+09` | `1.1456e-04` | `4.0772e-03` | `2.5882e-02` |
| `4` | `8.6424e+08` | `6.4428e-05` | `1.7379e-03` | `1.0445e-02` |

However, evaluating the jointly fine-tuned node model and edge head on H25-H30
showed a rollout trade-off:

| Experiment | Rollout RMSE | Global closure rel | Zone 3 closure rel |
|---|---:|---:|---:|
| Best inference projection | `0.085161` | `0.124980` | `0.051953` |
| Warm-start joint fine-tune | `0.101622` | `0.146612` | `0.054550` |

This suggests the edge loss can be optimized, but short joint fine-tuning moves
the node predictor away from the best rollout solution.

### 3. Best new controlled result: baseline node plus warm-start edge head

The most useful test kept the original baseline node model fixed at evaluation
time and used only the newly trained warm-start edge head:

```text
node_checkpoint=checkpoints_minxiong_h1h24_baseline_node_e20_seed0 epoch 19
edge_checkpoint=checkpoints_minxiong_h1h24_baseline_e20_warmstart_zoneweighted_edgeflux_w1e11_e5_b40_seed0 epoch 4
projection_mode=zone_weighted
projection_high_weight=100
projection_low_weight=1
state_update=projected_volume
projected_volume_alpha=0.05
```

Mean H25-H30 result:

| Experiment | Rollout RMSE | Original-state RMSE | Global closure rel | Zone 3 closure rel | Zone 3 original closure rel |
|---|---:|---:|---:|---:|---:|
| Best inference projection | `0.085161` | `0.084888` | `0.124980` | `0.051953` | `38.237` |
| Baseline node + warm-start edge head | `0.086176` | `0.085738` | `0.147394` | `0.043269` | `8.002` |

Interpretation:

- Rollout accuracy is nearly preserved relative to the current best baseline.
- Zone 3 closure improves from `0.051953` to `0.043269`.
- Global closure is worse than the best inference projection, so this is not yet
  the final model.
- The result strongly suggests the edge head should be trained separately or
  with the node model frozen / lower learning rate, instead of short joint
  fine-tuning.

Updated next step:

```text
1. Add a train-time option to freeze MeshGraphKAN and train only
   HecRasEdgeFluxHead from a baseline node checkpoint.
2. Re-run edge-head-only warm-start training with a small sweep:
   - 1e-11
   - 3e-11
   - 1e-10
3. Evaluate each with the fixed baseline node:
   - rollout RMSE
   - global closure
   - Zone 3 closure
   - spatial residual maps
```

## 2026-06-28 Update: Freeze-Node Edge-Head-Only Sweep

Implemented a formal training option:

```yaml
freeze_mesh_model_for_edge_flux_head: false
```

When enabled together with `use_hecras_edge_flux_head=true`, the trainer:

- freezes all `MeshGraphKAN` parameters,
- keeps `MeshGraphKAN` in eval mode,
- trains only `HecRasEdgeFluxHead`,
- avoids optimizer parameter-group mismatch when warm-starting from a baseline
  node checkpoint.

This converts the earlier manual separated evaluation into a repeatable
training procedure.

Three 5-epoch freeze-node edge-head-only runs were completed:

| Run | Checkpoint | Edge loss weight |
|---|---|---:|
| `freeze_w1e-11` | `checkpoints_minxiong_h1h24_baseline_e20_freeze_node_edgeflux_w1e11_e5_b40_seed0` | `1e-11` |
| `freeze_w3e-11` | `checkpoints_minxiong_h1h24_baseline_e20_freeze_node_edgeflux_w3e11_e5_b40_seed0` | `3e-11` |
| `freeze_w1e-10` | `checkpoints_minxiong_h1h24_baseline_e20_freeze_node_edgeflux_w1e10_e5_b40_seed0` | `1e-10` |

In these runs, `mse_loss` and `physics_loss` remained constant across epochs,
confirming that the node predictor was frozen.  The edge-head objective still
decreased from about `2.17e+09` to about `8.61e+08`.

Mean H25-H30 evaluation:

| Experiment | Rollout RMSE | Original-state RMSE | Global closure rel | Zone 3 closure rel | Zone 3 original closure rel | Correction RMS ft3 |
|---|---:|---:|---:|---:|---:|---:|
| Best previous inference projection | `0.085161` | `0.084888` | `0.124980` | `0.051953` | `38.237` | `17562.7` |
| Freeze-node edge head, `1e-11` | `0.086177` | `0.085739` | `0.147389` | `0.043272` | `7.411` | `11695.7` |
| Freeze-node edge head, `3e-11` | `0.086173` | `0.085735` | `0.147343` | `0.043290` | `6.824` | `11685.3` |
| Freeze-node edge head, `1e-10` | `0.086170` | `0.085733` | `0.147296` | `0.043283` | `6.766` | `11686.3` |

Interpretation:

- Freezing the node model prevents the rollout degradation seen in short joint
  fine-tuning.
- The edge-head-only runs preserve rollout accuracy within about `0.001` RMSE
  of the best previous result.
- Zone 3 local closure improves from `0.051953` to about `0.04328`.
- Global closure is still worse than the best inference projection, so the next
  step is to improve global/local balance rather than only optimize Zone 3.
- The `1e-10` run is the current best among the freeze-node sweep by global
  closure, with essentially the same rollout and Zone 3 closure as the other
  two weights.

Updated next step:

```text
1. Generate spatial residual maps for freeze_w1e-10.
2. Compare freeze_w1e-10 against best previous inference projection by zone.
3. Investigate mixed objective/projection weighting that preserves the better
   global closure of the inference projection while keeping the improved
   freeze-node Zone 3 closure.
```

## 2026-06-28 Update: Freeze-Node Spatial Residual Maps

Spatial local-conservation residual maps were generated for the current best
freeze-node setting:

```text
checkpoint=checkpoints_minxiong_h1h24_baseline_e20_freeze_node_edgeflux_w1e10_e5_b40_seed0
epoch=4
projection_mode=zone_weighted
projection_high_weight=100
projection_low_weight=1
projected_volume_alpha=0.05
```

Assets:

```text
/mnt/8tb_hdd2/joyce/Report/assets/freeze_node_w1e10/minxiong_H25_zone_weighted_spatial_residual.png
/mnt/8tb_hdd2/joyce/Report/assets/freeze_node_w1e10/minxiong_H26_zone_weighted_spatial_residual.png
/mnt/8tb_hdd2/joyce/Report/assets/freeze_node_w1e10/minxiong_H27_zone_weighted_spatial_residual.png
/mnt/8tb_hdd2/joyce/Report/assets/freeze_node_w1e10/minxiong_H28_zone_weighted_spatial_residual.png
/mnt/8tb_hdd2/joyce/Report/assets/freeze_node_w1e10/minxiong_H29_zone_weighted_spatial_residual.png
/mnt/8tb_hdd2/joyce/Report/assets/freeze_node_w1e10/minxiong_H30_zone_weighted_spatial_residual.png
/mnt/8tb_hdd2/joyce/Report/assets/freeze_node_w1e10/minxiong_H25_H30_freeze_node_w1e10_spatial_residual_zone_summary.csv
```

Cross-event per-zone summary:

| Zone | Median of event median-improvement ratios | Mean residual reduction | Mean projected RMS ft3 |
|---:|---:|---:|---:|
| `0` | `1.171x` | `44.71%` | `1294.239` |
| `1` | `0.969x` | `65.70%` | `809.328` |
| `2` | `0.943x` | `45.30%` | `721.452` |
| `3` | `30.511x` | `98.05%` | `15.556` |

Interpretation:

- The freeze-node edge head produces consistent Zone 3 spatial improvement
  across H25-H30.
- The improvement is intentionally concentrated in the high-fidelity zone,
  matching the automatic fidelity-zoning motivation.
- Lower-fidelity zones do not receive the same strong median improvement, which
  is acceptable for the current targeted-local-conservation hypothesis but
  explains why global closure remains worse than the best previous inference
  projection.

Updated next step:

```text
1. Design a mixed projection objective that keeps Zone 3 high weight while
   improving low-zone/global closure.
2. Consider separate reporting:
   - targeted high-zone local conservation result
   - global closure trade-off
3. For paper-level evidence, repeat with more HEC-RAS events and longer
   training once the larger event set is available.
```

## 2026-06-28 Update: Mixed Projection Low-Weight Sweep

The global/local trade-off was tested by keeping the high-fidelity Zone 3
projection weight fixed and increasing the non-Zone-3 weight:

```text
checkpoint=checkpoints_minxiong_h1h24_baseline_e20_freeze_node_edgeflux_w1e10_e5_b40_seed0
projection_high_weight=100
projection_low_weight in {1, 1.25, 1.5, 2, 5, 10, 25}
projected_volume_alpha=0.05
rollout_length=23
```

Mean H25-H30 sweep:

| Experiment | Rollout RMSE | Global closure rel | Zone 3 closure rel | Correction RMS ft3 |
|---|---:|---:|---:|---:|
| Previous best pretrained edge | `0.085161` | `0.124980` | `0.051953` | `17562.7` |
| Freeze edge, low=`1` | `0.086170` | `0.147296` | `0.043282` | `11686.3` |
| Freeze edge, low=`1.25` | `0.085648` | `0.136359` | `0.049122` | `12258.9` |
| Freeze edge, low=`1.5` | `0.085278` | `0.128046` | `0.054611` | `12743.4` |
| Freeze edge, low=`2` | `0.084791` | `0.116048` | `0.064881` | `13533.4` |
| Freeze edge, low=`5` | `0.083826` | `0.085658` | `0.117329` | `16249.0` |
| Freeze edge, low=`10` | `0.083472` | `0.067483` | `0.189478` | `18683.9` |
| Freeze edge, low=`25` | `0.083305` | `0.045912` | `0.359160` | `22740.0` |

Assets:

```text
/mnt/8tb_hdd2/joyce/Report/assets/minxiong_freeze_node_mixed_projection_sweep_summary.csv
/mnt/8tb_hdd2/joyce/Report/assets/minxiong_freeze_node_mixed_projection_tradeoff.png
```

Interpretation:

- Increasing `projection_low_weight` improves global closure because the
  low-fidelity zones are no longer under-weighted in the projection system.
- The improvement comes with a clear cost: Zone 3 local closure gradually
  worsens as non-Zone-3 nodes become more important.
- `low=1.25` is the current balanced candidate:
  - Zone 3 closure remains better than the previous best (`0.049122` vs
    `0.051953`).
  - Rollout remains close to the previous best (`0.085648` vs `0.085161`).
  - Global closure improves over freeze `low=1` (`0.136359` vs `0.147296`), but
    still does not beat the previous best pretrained projection (`0.124980`).
- `low=2` and above are useful global-closure candidates, but they begin to
  sacrifice the targeted high-fidelity-zone conservation objective.

Spatial maps for `low=1.25` were generated:

```text
/mnt/8tb_hdd2/joyce/Report/assets/freeze_node_w1e10_h100_l1p25/minxiong_H25_zone_weighted_spatial_residual.png
/mnt/8tb_hdd2/joyce/Report/assets/freeze_node_w1e10_h100_l1p25/minxiong_H26_zone_weighted_spatial_residual.png
/mnt/8tb_hdd2/joyce/Report/assets/freeze_node_w1e10_h100_l1p25/minxiong_H27_zone_weighted_spatial_residual.png
/mnt/8tb_hdd2/joyce/Report/assets/freeze_node_w1e10_h100_l1p25/minxiong_H28_zone_weighted_spatial_residual.png
/mnt/8tb_hdd2/joyce/Report/assets/freeze_node_w1e10_h100_l1p25/minxiong_H29_zone_weighted_spatial_residual.png
/mnt/8tb_hdd2/joyce/Report/assets/freeze_node_w1e10_h100_l1p25/minxiong_H30_zone_weighted_spatial_residual.png
/mnt/8tb_hdd2/joyce/Report/assets/freeze_node_w1e10_h100_l1p25/minxiong_H25_H30_freeze_node_w1e10_h100_l1p25_spatial_residual_zone_summary.csv
```

Cross-event spatial summary for `low=1.25`:

| Zone | Median of event median-improvement ratios | Mean residual reduction | Mean projected RMS ft3 |
|---:|---:|---:|---:|
| `0` | `1.178x` | `45.43%` | `1279.323` |
| `1` | `0.967x` | `66.02%` | `795.846` |
| `2` | `0.936x` | `45.57%` | `717.954` |
| `3` | `26.010x` | `97.74%` | `18.002` |

Updated next step:

```text
1. Treat low=1.25 as the current balanced reporting candidate.
2. Treat low=25 as the global-closure extreme and low=1 as the Zone 3 local
   closure extreme.
3. Next experiment should attempt adaptive or learned low-zone weighting, so
   the method does not depend on a manually selected low_weight.
```

## 2026-06-28 Update: Preliminary Validation-Selected Low Weight

To make `projection_low_weight` less hand-tuned, a preliminary validation split
was added:

```text
train_h1h20.txt = H1-H20
val_h21h24.txt = H21-H24
test_h25h30.txt = H25-H30
```

A new freeze-node edge head was trained on H1-H20:

```text
ckpt_path=./checkpoints_minxiong_h1h20_baseline_e20_freeze_node_edgeflux_w1e10_e5_b40_seed0
hydrograph_ids_file=train_h1h20.txt
freeze_mesh_model_for_edge_flux_head=true
hecras_edge_flux_head_loss_weight=1e-10
epochs=5
max_train_batches=40
```

Training result:

| Epoch | Edge head loss | MSE loss | Physics loss |
|---:|---:|---:|---:|
| `0` | `2.3015e+09` | `5.8084e-05` | `1.4258e-03` |
| `4` | `9.2785e+08` | `5.8084e-05` | `1.4258e-03` |

Validation sweep on H21-H24:

| Low weight | Rollout RMSE | Global closure rel | Zone 3 closure rel | Selected |
|---:|---:|---:|---:|---|
| `1` | `0.087141` | `0.147072` | `0.043249` | candidate |
| `1.25` | `0.086690` | `0.136135` | `0.049096` | selected |
| `1.5` | `0.086372` | `0.127827` | `0.054595` | |
| `2` | `0.085953` | `0.115842` | `0.064893` | |
| `3` | `0.085510` | `0.101124` | `0.083750` | |
| `5` | `0.085136` | `0.085505` | `0.117547` | |

Selection rule used for this preliminary test:

```text
1. Zone 3 closure must be no worse than the previous best pretrained
   projection baseline: <= 0.051953.
2. Rollout RMSE must remain <= 0.09.
3. Among eligible candidates, choose the lowest global closure relative RMSE.
```

This selected:

```text
projection_low_weight=1.25
```

Test evaluation on H25-H30 with the H1-H20-trained edge head and selected
`low=1.25`:

| Experiment | Rollout RMSE | Global closure rel | Zone 3 closure rel | Correction RMS ft3 |
|---|---:|---:|---:|---:|
| Previous best pretrained edge | `0.085161` | `0.124980` | `0.051953` | `17562.7` |
| H1-H24 freeze, manual/balanced `low=1.25` | `0.085648` | `0.136359` | `0.049122` | `12258.9` |
| H1-H20 freeze, validation-selected `low=1.25` | `0.085373` | `0.136571` | `0.048782` | `12237.4` |

Interpretation:

- The validation rule independently selected `low=1.25`, matching the previous
  manually identified balanced candidate.
- Test behavior remained stable after training the edge head only on H1-H20.
- Zone 3 closure is better than the previous best pretrained projection.
- Global closure remains worse than the previous best pretrained projection,
  so this is a targeted-local-conservation improvement rather than a universal
  metric win.

Important limitation:

The node-model checkpoint used here is still the existing H1-H24 baseline
checkpoint.  Therefore this is a preliminary validation-selection workflow, not
a completely leakage-free paper experiment.  For final paper evidence, the node
baseline should also be retrained on H1-H20, then validation and test should be
re-run with the same selection rule.

Updated next step:

```text
1. Add/report the validation-selection rule as the current method.
2. For paper-level rigor, train a new baseline node model on H1-H20 only.
3. Re-run:
   - freeze-node edge head on H1-H20
   - validation low-weight selection on H21-H24
   - final test on H25-H30
```
