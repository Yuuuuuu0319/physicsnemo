# Test H11-H15 Rollout Evaluation For H1-H6 e5 Full Ablation

Date: 2026-05-25

## Setup

- Training checkpoints were trained on matched H1-H6 events from `/mnt/8tb_hdd2/joyce/train`.
- Held-out evaluation data: `/mnt/8tb_hdd2/joyce/test`, H11-H15 from `test.txt`.
- Normalization statistics: `/mnt/8tb_hdd2/joyce/train`.
- Evaluation lengths: `10`, `30`, `45`.
- Device: GPU 1 A6000, `CUDA_VISIBLE_DEVICES=1`.
- This rollout table does not use HDF face velocity. Matched H11-H15 HDF outputs have since been completed and are evaluated separately in the fixed-scale face residual report.
- Metric field `zone_3` is the zero-based implementation label for the study's Zone 4 high-fidelity region.

## Compared Checkpoints

| Checkpoint label | Zone loss | Matched HEC-RAS face loss during H1-H6 training |
|---|---:|---:|
| `h1h6_noface_e5_full` | `0.0` | `0.0` |
| `h1h6_w1e9_e5_full` | `0.0` | `1e-9` |
| `h1h6_zone1_e5_full` | `1.0` | `0.0` |
| `h1h6_zone1_w1e9_e5_full` | `1.0` | `1e-9` |

## Held-Out Rollout Results

Lower is better.

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

## Interpretation

- The H1-H6 train-side trade-off generalizes to held-out H11-H15 rollout:
  - `zone-only` remains best for water-depth RMSE.
  - `zone + face` remains best for volume RMSE, including high-fidelity Zone 4 (`zone_3`).
- `zone + face` also has the lowest total rollout RMSE at all three lengths on H11-H15, even though zone-only has lower WD RMSE.
- Face-only does not provide competitive rollout accuracy by itself; its value remains the conservation objective measured with matched HDF face residuals.
- The subsequent matched-HDF fixed-scale report confirms conservation generalization on H11-H15; see `results_hgn_matched_test_h11h15_zone_face_e5_full_summary.md`.

## Output Files

- `results_hgn_test_h11h15_zone_rollout_e5_full_ablation_len10.csv`
- `results_hgn_test_h11h15_zone_rollout_e5_full_ablation_len30.csv`
- `results_hgn_test_h11h15_zone_rollout_e5_full_ablation_len45.csv`
