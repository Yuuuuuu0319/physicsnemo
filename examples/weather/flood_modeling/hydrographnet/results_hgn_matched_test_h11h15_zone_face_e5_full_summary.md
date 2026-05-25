# Matched H11-H15 Test Zone + Face e5 Full Summary

Date: 2026-05-25

## Setup

- Checkpoints were trained on matched H1-H6 events from `/mnt/8tb_hdd2/joyce/train` for 5 epochs.
- Held-out evaluation events: H11-H15 from `/mnt/8tb_hdd2/joyce/test/test.txt`.
- Normalization statistics: `/mnt/8tb_hdd2/joyce/train`.
- Evaluation lengths: `10`, `30`, and `45`.
- GPU device: GPU 1 A6000, `CUDA_VISIBLE_DEVICES=1`.
- Metric field `zone_3` is the zero-based implementation label for the Zone 4 high-fidelity study region.

## Matched HDF Provenance

HEC-RAS H11-H15 event-specific outputs were generated from the HydroGraphNet test inflow and precipitation series. The initial batch completed H11, H13, H14, and H15; H12 reached the 120-minute runner timeout and its partial output is excluded. H12 was rerun in the isolated `Minxiong_hgn_test_h12_retry` workspace with a 240-minute timeout and completed successfully.

For evaluation, a symlink-only collection was created at:

`/mnt/8tb_hdd2/joyce/hecras-dataset/origin/Linux_RAS_v66/Minxiong_hgn_test_h11h15_matched_hdf/planH*/Minxiong.p01.hdf`

It points to the four successful original outputs plus the successful H12 retry output. The event matching report verifies that all five HDFs match their corresponding upstream inflow and precipitation series with `exact_or_close=1`.

## Fixed-Scale Conservation Method

`evaluate_hecras_face_residual.py` now defaults to `--proxy-scale-reference ground_truth`. The HEC-RAS face proxy is calibrated using the observed ground-truth volume transition rather than each checkpoint autoregressive rollout state. Consequently, each checkpoint is compared against the same HDF reference baseline at a given rollout length.

## Held-Out Rollout Prediction Results

Lower is better. These values do not use HDF face velocity.

| Length | Run | Rollout RMSE | Volume RMSE | WD RMSE | Zone 4 Volume RMSE (`zone_3`) | Zone 4 WD RMSE (`zone_3`) |
|---:|---|---:|---:|---:|---:|---:|
| 10 | zone + face | 0.042382 | 0.040727 | 0.043954 | 0.028884 | 0.054145 |
| 10 | zone-only | 0.042889 | 0.045049 | 0.040606 | 0.033823 | 0.050149 |
| 10 | face-only | 0.047653 | 0.045961 | 0.049277 | 0.038641 | 0.070281 |
| 10 | noface | 0.046598 | 0.045240 | 0.047898 | 0.040544 | 0.066504 |
| 30 | zone + face | 0.108276 | 0.096427 | 0.118693 | 0.078425 | 0.146132 |
| 30 | zone-only | 0.109646 | 0.110714 | 0.108481 | 0.094747 | 0.137301 |
| 30 | face-only | 0.124879 | 0.111158 | 0.137011 | 0.101929 | 0.191556 |
| 30 | noface | 0.122560 | 0.112161 | 0.131752 | 0.109835 | 0.184132 |
| 45 | zone + face | 0.149327 | 0.129893 | 0.166241 | 0.106892 | 0.198575 |
| 45 | zone-only | 0.151411 | 0.149450 | 0.153196 | 0.132165 | 0.188847 |
| 45 | face-only | 0.170683 | 0.146052 | 0.191883 | 0.137462 | 0.264144 |
| 45 | noface | 0.168192 | 0.153179 | 0.181638 | 0.152625 | 0.251760 |

## Held-Out Fixed-Scale Face Residual Results

Lower is better. These values use event-matched HDF face velocity.

| Length | Run | Pred Face Residual RMSE | Zone 4 Pred Face Residual RMSE (`zone_3`) | Pred vs GT Delta RMSE |
|---:|---|---:|---:|---:|
| 10 | zone + face | 132.9491 | 98.2495 | 190.6075 |
| 10 | zone-only | 150.6344 | 151.6393 | 210.4340 |
| 10 | face-only | 147.9581 | 117.3624 | 215.7516 |
| 10 | noface | 167.8533 | 167.6840 | 215.1341 |
| 30 | zone + face | 134.2425 | 97.6502 | 171.3945 |
| 30 | zone-only | 156.4819 | 155.6582 | 192.9248 |
| 30 | face-only | 146.8514 | 114.1979 | 191.2447 |
| 30 | noface | 173.5405 | 171.9413 | 202.5615 |
| 45 | zone + face | 138.3515 | 100.5817 | 164.1266 |
| 45 | zone-only | 159.4317 | 155.0628 | 184.4825 |
| 45 | face-only | 144.0998 | 111.8144 | 175.8309 |
| 45 | noface | 175.9270 | 174.7581 | 195.8921 |

The shared ground-truth HDF baseline residuals for lengths 10, 30, and 45 are `166.8913`, `125.4575`, and `98.6380`, respectively, across every checkpoint up to floating point noise.

## Interpretation

- `zone + face` has the lowest total rollout RMSE and volume RMSE on held-out H11-H15 at all three lengths.
- `zone-only` retains the lowest water-depth RMSE, so it remains the accuracy-focused ablation for depth prediction.
- Under matched HDF and checkpoint-independent scaling, `zone + face` has the lowest global and high-fidelity Zone 4 local conservation residual at every rollout length.
- The combined objective is therefore the strongest current candidate when the research goal includes both multi-fidelity volume prediction and HEC-RAS face-based local conservation.

## Output Files

- `results_hgn_matched_test_h11h15_event_check.csv`
- `results_hgn_matched_test_h11h15_event_check.md`
- `results_hgn_test_h11h15_zone_rollout_e5_full_ablation_len10.csv`
- `results_hgn_test_h11h15_zone_rollout_e5_full_ablation_len30.csv`
- `results_hgn_test_h11h15_zone_rollout_e5_full_ablation_len45.csv`
- `results_hgn_matched_test_h11h15_face_residual_e5_full_ablation_fixedscale_len10.csv`
- `results_hgn_matched_test_h11h15_face_residual_e5_full_ablation_fixedscale_len30.csv`
- `results_hgn_matched_test_h11h15_face_residual_e5_full_ablation_fixedscale_len45.csv`
