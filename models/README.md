# Model Checkpoints

Saved PyTorch checkpoints from the original notebook runs. Checkpoints are
tracked directly in this repository under `models/checkpoints/`; the
evaluation CLI reads them from that directory after a clone.

## Filename Schema

```
model<arch>_hiddensize_<H>_drop_<DD>.pt
```

| Field | Values | Meaning |
| --- | --- | --- |
| `<arch>` | `04`, `22`, `1111`, `1212`, `skip`, `40`, `test` | GNN layer stack (see table below) |
| `<H>` | `32`, `64`, `128`, `256` | Hidden channel width |
| `<DD>` | `00`, `02`, `04`, `06` | Dropout rate (`DD / 10`, e.g. `04` -> 0.4) |

`scripts/evaluate_checkpoint.py` infers `<H>` and `<DD>` from the filename and
reads `<arch>` from the saved state dict. Keep checkpoint filenames aligned
with the metrics CSVs.

## Architecture Codes

The `<arch>` token maps to layer stacks defined in
`scripts/evaluate_checkpoint.py` and `pipe_network_completion/model.py`:

| `<arch>` | Layers | Skip connections | Reported in |
| --- | --- | --- | --- |
| `04` | sage, sage, sage, sage | no | `model_metrics04.csv` |
| `22` | gat, gat, sage, sage | no | `model_metrics.csv` |
| `1111` | gat, sage, gat, sage | no | `model_metrics.csv` |
| `1212` | gat, sage, sage, gat, sage, sage | no | `model_metrics1212.csv` |
| `skip` | gat, sage, sage, gat, sage, sage | yes (lin1, lin2) | `model_metrics1212_skip.csv` |
| `40` | exploratory, see legacy notebook | n/a | not published |
| `test` | exploratory placeholder | n/a | not published |

## Default Replication Target

`scripts/evaluate_checkpoint.py` defaults to:

- Checkpoint: `model1212_hiddensize_128_drop_00.pt`
- Metrics CSV: `results/metrics/model_metrics1212.csv`
- Split: `test`

This combination corresponds to the reported ISARC 2024 test metrics.

## Notes On Duplicates

Several files (e.g. `model22_hiddensize_32_drop_00.pt` through
`drop_06.pt`) have identical sizes. The dropout flag affects training-time
behavior but not the saved weight tensors for some sweeps. Keep those files so
the metrics CSV rows continue to resolve.

If a checkpoint is ever retired, append a deprecation note here and remove the
file from `models/checkpoints/`.
