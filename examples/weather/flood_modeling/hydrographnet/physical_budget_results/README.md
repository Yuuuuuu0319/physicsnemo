# Physical Budget Validation Results

These results evaluate whether HEC-RAS output can support a formal,
uncalibrated HydroGraphNet local-conservation target. They are intentionally
separate from model training and from the earlier calibrated
`velocity * face_length` proxy branch.

## Current Decision

Formal HEC-RAS local-conservation training should remain disabled for the
legacy capped `M80_V` target, but the corrected area-extrapolated target now
passes the core storage/output gates and substantially reduces the uncalibrated
local residual. No fitted scale factor is used in these checks.

| Evaluation | Stored face output interval | Native `Face Flow` | Storage aligned to HGN | Best HGN relative RMSE | Status |
|---|---:|---|---|---:|---|
| Train `H1-H6`, legacy HDF | 1800 s | No | No | 12336.956598 | Blocked |
| Held-out `H11-H15`, legacy HDF | 1800 s | No | No | 12789.891145 | Blocked |
| H1 high-frequency preflight, first 24 h | 30 s | No | No | 437.253423 | Blocked |
| User-provided `Minxiong.b03` plan preflight | 1800 s | Unknown until HDF exists | N/A | N/A | Blocked |
| `Minxiong_new` native-budget HDF, first 24 h | 30 s | Yes | No | 1.563688 | Blocked |
| `Minxiong_new` synchronized 30-min HGN export, first 24 h | 30 s | Yes | Yes | 0.837387 | Residual review |
| `Minxiong_new` area-extrapolated 30-min HGN export, first 24 h | 30 s | Yes | Yes | 0.061336 | Candidate |
| Matched `hecras-dataset` H1-H15 exports, H1 first 24 h | 30 s | Yes | Yes | 0.044605 | Candidate |
| Matched `hecras-dataset` H1-H15 exports, H15 first 24 h | 30 s | Yes | Yes | 0.031953 | Candidate |

The H1 preflight proves that the HDF output-frequency issue can be fixed: its
face-output interval is 30 seconds, matching the solver computation interval.
It does not solve the remaining physical problems: the HDF lacks native
internal `Face Flow`, and reconstructed HDF storage does not match the legacy
HGN `M80_V` target.

The 2026-05-28 user-provided `/mnt/8tb_hdd2/joyce/Minxiong/Minxiong.b03`
plan still uses `Computation Level Output = 30MIN`. That plan is useful as a
full-event HEC-RAS configuration reference, but it is not yet suitable for
formal local-budget validation because HGN targets are spaced at 30 minutes
while the HEC-RAS solver computation interval is 30 seconds. A budget-ready
candidate should keep the desired event window, set computation-level output
to `30SEC`, and write native `Face Flow` plus `Cell Flow Balance` into the
result HDF.

The 2026-06-03 `/mnt/8tb_hdd2/joyce/Minxiong/Minxiong_new` result passes the
native-output preflight: its plan/HDF store output every `30SEC`, and the HDF
contains both native `Face Flow` and `Cell Flow Balance`. This removes the
previous native-output blocker. However, event matching against current
`/train` and `/test` HGN events is not exact, and the 24-hour physical-budget
gate remains blocked:

- HDF reconstructed storage vs HGN `M80_V` alignment: RMSE
  `3863.649841 ft^3`, max abs `177406.298159 ft^3`.
- Best HGN-target uncalibrated relative RMSE: `1.563688`.
- Best HDF-reconstructed-storage relative RMSE: `0.837387`.
- `native_face_flow` and `native_cell_flow_balance` give nearly identical
  residuals, indicating the native outputs are internally consistent but not
  yet closed enough to become a formal HydroGraphNet training target.

After pulling the updated `/mnt/8tb_hdd2/joyce/hecras-dataset` repository, a
single-event synchronized HGN export was generated from
`/mnt/8tb_hdd2/joyce/Minxiong/Minxiong_new/Minxiong.p03.hdf` in an isolated
workspace:
`/mnt/8tb_hdd2/joyce/Minxiong/Minxiong_new/hgn_extract_workspace/outputs_native/HGN_dataset`.
The first extractor run used the upstream script's default long-run subsampling
and produced 500 non-uniform time steps (`420/450 s` spacing), which is not
valid for a formal conservation interval. The isolated extractor copy was then
configured to export fixed `1800 s` HGN targets from the 30-second HDF. This
produced `121` time steps over `60 h` and `12715` cells, with `M80_XY`
identical to the current `/train` mesh.

The synchronized export removes the event/storage mismatch:

- HDF reconstructed storage vs exported HGN `M80_V`: RMSE
  `0.000000 ft^3`, max abs `0.000000 ft^3`.
- HDF output interval: `30.000000 s`, matching the computation interval.
- Best capped-HGN-target uncalibrated relative RMSE remains `0.837387`.
- After extrapolating above finite HEC-RAS volume-elevation table tops with
  cell surface area, the best local budget improves to relative RMSE
  `0.061336`, cell RMSE `119.169194 ft^3`, domain RMSE `2173.014667 ft^3`,
  correlation `0.998037`, using `Cell Flow Balance + precipitation`.
- `Cell Flow Balance` has HDF units `ft^3/s` and description
  `Sum of inward flows at every cell`. Direct integration does not close the
  capped `M80_V` target, but it does closely match the corrected
  area-extrapolated target when HDF precipitation volume is included.
- Numerically, `Cell Flow Balance` equals the signed net sum of native
  `Face Flow`: RMSE `2.789812e-06 cfs`, correlation `1.000000`. It is not an
  inward-only quantity for this HDF (`inward-only` RMSE `101.170189 cfs`).
- With the corrected area-extrapolated target, zone-level residuals are much
  lower: Zone 0 relative RMSE `0.018559`, Zone 1 `0.079460`, Zone 2 `0.157654`,
  and Zone 3 `0.086696`. Wet/dry transition fraction remains about `0.0208`,
  so the remaining residual is not primarily a wet/dry switching artifact.
- Domain-level checks also do not fully close, and HEC-RAS
  `Computations/Volume` is not the same quantity as the active-cell
  volume-elevation reconstruction used for HGN `M80_V`.
- HEC-RAS global `Computations/Volume` does close against native external
  boundary flow plus precipitation: relative RMSE `0.000247`, RMSE
  `1377.068508 ft^3`, correlation `1.000000`. This means the HEC-RAS run is
  globally water-balanced; the remaining problem is the mismatch between that
  conserved global volume and the active-cell `M80_V` target used by HGN.
- Volume-table capping is the likely cause of that mismatch: `4079` active
  cells exceed their volume-table top at least once, final over-table fraction
  is `0.320409`, and max exceedance is `35.306687 ft`. Area-extrapolated
  active-cell volume matches HEC-RAS global volume delta with RMSE
  `39.235807 ft^3`.

Conclusion: the current blocker is no longer event synchronization or face-flow
orientation. The capped `M80_V` target produced from finite HEC-RAS
volume-elevation tables is the problem. A formal local-conservation loss should
use an above-table extrapolated storage target, then compare it against native
`Cell Flow Balance + precipitation`.

On 2026-06-04, the maintained `/mnt/8tb_hdd2/joyce/hecras-dataset` workflow was
updated with the budget-ready `Minxiong_new` HDF/plan resources, and H1-H15
were regenerated as event-specific HEC-RAS results. The batch simulation
completed successfully for all 15 events. The new HGN export is stored at
`/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong/outputs/HGN_dataset`
with 121 HGN time steps per event and 12715 active cells. A folder ordering bug
was fixed so `planH10` no longer maps to H2 during export.

Targeted H1 and H15 diagnostics on the matched H1-H15 dataset show that native
`Cell Flow Balance + precipitation` closes the local storage target without a
scale factor: H1 relative RMSE `0.044605`, H15 relative RMSE `0.031953`, with
correlations above `0.9989`. Native `Cell Flow Balance` also matches the signed
net sum of native `Face Flow` at numerical precision for both events.

Initial formal training was run on H1-H12 and evaluated on held-out H13-H15.
Compared with the no-cell-balance baseline, the `cell-balance` checkpoint
reduced held-out one-step MSE from `1.861854e-04` to `1.092053e-04`, Zone 4
volume RMSE from `9.307462e-03` to `6.657264e-03`, global cell-balance RMSE
from `965.380239 ft^3` to `578.911615 ft^3`, and relative cell-balance RMSE
from `1.433235` to `0.859471`. Held-out Zone 4 water-depth RMSE was slightly
worse (`1.590883e-02` to `1.698810e-02`), so the next tuning step is balancing
water-depth accuracy against formal conservation strength.

## Canonical Result Files

- `train_h1h6_start72_targets.md` and `.csv`: legacy training events evaluated
  after warm-up.
- `heldout_h11h15_start72_targets.md` and `.csv`: complete held-out events,
  including the independently regenerated `H12` HDF.
- `highfreq_h1_24h_native_boundary.md` and `.csv`: isolated 30-second H1
  preflight evaluated over its first 24 hours.
- `minxiong_b03_budget_input_preflight.md` and `.csv`: lightweight inspection
  of the user-provided `Minxiong.b03`, the prior 30-second H1 preflight seed,
  and the current `Minxiong.p03.hdf`.
- `minxiong_new_native_budget_preflight.md` and `.csv`: confirms the new
  `Minxiong_new` plan/HDF has 30-second native `Face Flow` and
  `Cell Flow Balance`.
- `minxiong_new_event_match.md` and `.csv`: checks whether the new HDF forcing
  matches current HGN H1-H15 events; no exact match was found.
- `minxiong_new_native_budget_h1_first24h.md` and `.csv`: first 24-hour
  uncalibrated physical-budget gate using the new native-output HDF.
- `minxiong_new_synced_30min_budget_h1_first24h.md` and `.csv`: first 24-hour
  budget gate for the isolated synchronized 30-minute HGN export generated
  directly from `Minxiong_new`.
- `minxiong_new_synced_30min_budget_terms_h1_first24h.md` and `.csv`: targeted
  diagnosis showing that synchronized storage is valid, `Cell Flow Balance`
  matches signed native `Face Flow`, and the remaining residual is not a
  face-orientation implementation error.
- `minxiong_new_area_extrap_budget_h1_first24h.md` and `.csv`: formal
  validator result for the corrected area-extrapolated HGN target; storage
  aligns and the best uncalibrated local residual is relative RMSE `0.061336`.
- `minxiong_hecras_dataset_h1_budget_terms.md` and `.csv`: H1 targeted
  diagnosis on the regenerated matched H1-H15 dataset.
- `minxiong_hecras_dataset_h15_budget_terms.md` and `.csv`: H15 targeted
  diagnosis on the regenerated matched H1-H15 dataset.
- `../results_minxiong_hecras_h1h12_cellbalance_e20_summary.md`: first matched
  H1-H12 training and held-out H13-H15 evaluation summary.

## Training Integration

The area-above-table extrapolation has been moved into the maintained
HEC-RAS-to-HGN export path in `/mnt/8tb_hdd2/joyce/hecras-dataset`, and
PhysicsNeMo training now has a separate formal objective named
`hecras_cell_balance_loss`. This branch is intentionally independent from the
earlier calibrated `hecras_face_loss` prototype.

When enabled with `+use_hecras_cell_balance_loss=true`, the dataset loads
event-specific HDFs from `+hecras_cell_balance_glob=...`, aligns the HGN event
time axis after the existing 72-step warm-up skip and peak-flow trim, integrates
native `Cell Flow Balance` over each HGN interval, adds native precipitation
volume, and attaches that denormalized ft^3 target to the graph as
`hecras_cell_balance_delta`. The training loss compares
`pred[:, 1] * volume_std` directly to that target. No fitted scale factor is
used.

Next work should expand beyond the current H1 candidate by generating/exporting
all needed train/test event HDFs with the same 30-second native outputs and
area-extrapolated `M80_V`, then run longer ablations:
`noface`, `face-only`, `zone-only`, `zone+face`, and the new formal
`cell-balance` variants.
