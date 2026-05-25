# Matched H1-H6 Fixed-Scale Face Residual Ablation Summary

Date: 2026-05-25

## Purpose

Earlier multi-step face residual evaluation calibrated the HEC-RAS face proxy
against a volume delta derived from the autoregressive model state. After the
first rollout step, that state differs by checkpoint, so the physical proxy
scale and ground-truth residual were not a shared comparison baseline.

`evaluate_hecras_face_residual.py` now defaults to
`--proxy-scale-reference ground_truth`. At each step, it calibrates the HDF
face-velocity proxy against the observed transition from the true previous
volume to the true next volume. The legacy behavior remains available through
`--proxy-scale-reference autoregressive`.

## Setup

- Matched HEC-RAS events: H1-H6.
- Compared checkpoints: `noface`, `face-only`, `zone-only`, and `zone + face`,
  all trained for 5 epochs on the H1-H6 setup.
- Rollout lengths: `10`, `30`, and `45`.
- High-fidelity region in code metrics: `zone_3`, corresponding to Zone 4 in
  the study definition.
- Lower residual RMSE is better.

## Fixed-Scale Results

| Length | Run | Pred Face Residual RMSE | High-Fidelity Pred Face Residual RMSE | Pred vs GT Delta RMSE |
|---:|---|---:|---:|---:|
| 10 | zone + face | 128.5193 | 95.8901 | 189.7797 |
| 10 | zone-only | 148.3614 | 147.1825 | 210.8795 |
| 10 | face-only | 142.5445 | 111.4525 | 208.8201 |
| 10 | noface | 168.9392 | 169.3441 | 215.0911 |
| 30 | zone + face | 131.0226 | 95.4539 | 168.4606 |
| 30 | zone-only | 150.2893 | 152.5687 | 186.9529 |
| 30 | face-only | 139.6109 | 107.4618 | 183.8772 |
| 30 | noface | 169.3140 | 173.5179 | 197.8167 |
| 45 | zone + face | 134.5683 | 98.5216 | 160.9817 |
| 45 | zone-only | 152.1081 | 151.0363 | 177.6997 |
| 45 | face-only | 138.2291 | 105.9101 | 170.2438 |
| 45 | noface | 170.1384 | 174.5489 | 190.1185 |

## Baseline Consistency Check

For each rollout length, all checkpoints now share the same HDF-calibrated
ground-truth baseline, up to floating point noise:

| Length | Shared GT Face Residual RMSE | Shared Face Proxy Scale |
|---:|---:|---:|
| 10 | 166.4812 | 0.000172227 |
| 30 | 125.5277 | 0.000090936 |
| 45 | 99.6910 | 0.000061842 |

## Interpretation

- `zone + face` has the lowest global and high-fidelity face residual RMSE at
  all three rollout lengths under a checkpoint-independent HDF reference.
- `face-only` improves conservation residuals over `noface`, but combining the
  face objective with the fidelity-zone objective performs best.
- This result verifies the training-event HDF conservation signal. Generalizing
  the same claim to held-out H11-H15 still requires their event-specific HDF
  outputs and the corresponding fixed-scale evaluation.

## Output Files

- `results_hgn_matched_h1h6_face_residual_all_e5_full_ablation_fixedscale_len10.csv`
- `results_hgn_matched_h1h6_face_residual_all_e5_full_ablation_fixedscale_len30.csv`
- `results_hgn_matched_h1h6_face_residual_all_e5_full_ablation_fixedscale_len45.csv`
