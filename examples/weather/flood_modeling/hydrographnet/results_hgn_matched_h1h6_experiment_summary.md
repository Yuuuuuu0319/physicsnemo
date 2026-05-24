# H1-H6 Matched HDF Experiment Summary

This table compares the current H1-H6 HydroGraphNet runs that use event-specific
HEC-RAS HDF files from `Minxiong_hgn_h1h6/outputs_seed/planH*/Minxiong.p01.hdf`.
All residual evaluations use matched HDF events and the same HEC-RAS face graph:
`/mnt/8tb_hdd2/joyce/Minxiong/hecras_hgn_face_graph_hgn_h1h6_check.npz`.

| Experiment | Face loss weight | Batches | Train total loss | 10-step pred face RMSE | 10-step zone 3 pred face RMSE | 30-step pred face RMSE | 30-step zone 3 pred face RMSE | Checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| h1h6_b150 | 1e-10 | 150 | 0.044122 | 580.424 | 495.295 |  |  | `checkpoints_hgn_h1h6_hecrasface_b150` |
| h1h6_b600 | 1e-10 | 600 | 0.016981 | 424.934 | 390.195 | 426.933 | 403.493 | `checkpoints_hgn_h1h6_hecrasface_b600` |
| h1h6_noface_b600 | 0 | 600 | 0.016859 | 437.066 | 424.935 | 445.740 | 444.969 | `checkpoints_hgn_h1h6_noface_b600` |
| h1h6_w1e9_b600 | 1e-9 | 600 | 0.018126 | 388.723 | 312.513 | 378.258 | 310.216 | `checkpoints_hgn_h1h6_hecrasface_w1e9_b600` |

Current interpretation:

- Increasing training from 150 to 600 batches substantially improved the matched-HDF face residual.
- The no-face baseline has similar supervised/physics training loss, but worse matched-HDF face residual than the HEC-RAS face-loss runs.
- `hecras_face_loss_weight=1e-9` is the best current setting for H1-H6 matched-HDF residuals, especially in high-fidelity zone 3.
- The next useful step is to test the same comparison on held-out H/T events once event-specific HDF outputs are available for those cases.
