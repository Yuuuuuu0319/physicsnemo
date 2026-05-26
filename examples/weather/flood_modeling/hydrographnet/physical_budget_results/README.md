# Physical Budget Validation Results

These results evaluate whether HEC-RAS output can support a formal,
uncalibrated HydroGraphNet local-conservation target. They are intentionally
separate from model training and from the earlier calibrated
`velocity * face_length` proxy branch.

## Current Decision

Formal HEC-RAS local-conservation training remains **blocked**. No fitted scale
factor is used in these checks, but the current evidence fails the required
physical gates:

| Evaluation | Stored face output interval | Native `Face Flow` | Storage aligned to HGN | Best HGN relative RMSE | Status |
|---|---:|---|---|---:|---|
| Train `H1-H6`, legacy HDF | 1800 s | No | No | 12336.956598 | Blocked |
| Held-out `H11-H15`, legacy HDF | 1800 s | No | No | 12789.891145 | Blocked |
| H1 high-frequency preflight, first 24 h | 30 s | No | No | 437.253423 | Blocked |

The H1 preflight proves that the HDF output-frequency issue can be fixed: its
face-output interval is 30 seconds, matching the solver computation interval.
It does not solve the remaining physical problems: the HDF lacks native
internal `Face Flow`, and reconstructed HDF storage does not match the legacy
HGN `M80_V` target.

## Canonical Result Files

- `train_h1h6_start72_targets.md` and `.csv`: legacy training events evaluated
  after warm-up.
- `heldout_h11h15_start72_targets.md` and `.csv`: complete held-out events,
  including the independently regenerated `H12` HDF.
- `highfreq_h1_24h_native_boundary.md` and `.csv`: isolated 30-second H1
  preflight evaluated over its first 24 hours.

## Required Next Input

Regenerate an event-specific HDF with HEC-RAS optional HDF variables
`Face Flow` and `Cell Flow Balance` enabled, while retaining the 30-second
preflight output interval. The validator will automatically evaluate native
face discharge when `Face Flow` becomes available. Only after that budget
passes should a formal local-conservation loss be wired into GPU training.
