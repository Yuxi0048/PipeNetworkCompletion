# Traceability Map

This repo keeps the original research code under `legacy/` and adds refactored
entry points around it. The notebook-era files remain available for audit, and
the reproducible commands now live in plain Python modules and scripts.

## Original Artifacts

| Original artifact | Role in original repo | Refactored or reproducible entry point |
| --- | --- | --- |
| `legacy/Inductive.ipynb`, setup cells | Colab package installation and local path setup | `environment.yml`, `scripts/create_env.ps1`, `scripts/verify_environment.py` |
| `legacy/Inductive.ipynb`, data load/build cells | Load preprocessed pickles, build `train_data`, `val_data`, `test_data` | `pipe_network_completion/dataset.py` keeps the transformation logic; `scripts/build_graphs.py` is the driver that mirrors the `set.union(*split[...])` notebook cell; prepared graph pickles live in `data/processed/graphs/` |
| `legacy/Inductive.ipynb`, loader cell | Build `LinkNeighborLoader` instances | `scripts/evaluate_checkpoint.py` |
| `legacy/Inductive.ipynb`, model cell | Defines `GNN`, dot-product classifier, and `Model` with global notebook state | `pipe_network_completion/model.py` |
| `legacy/Inductive.ipynb`, training/evaluation cells | Train model variants, save checkpoints, compute metrics | `pipe_network_completion/evaluation.py`, `scripts/evaluate_checkpoint.py` |
| `legacy/Inductive.ipynb`, explanation/plotting cells | Captum explanations and paper figures | Kept in the legacy notebook; not promoted to the replication CLI yet |
| `legacy/Module/` | Previous `Module` namespace retained for auditability | Replaced by `pipe_network_completion/` for runnable code |
| `legacy/linemap.py` | Notebook helper for Folium line visualization | Not part of the reproducibility CLI |
| `process.py` | Raw GIS preprocessing from shapefiles to pickle artifacts | Current standalone preprocessing CLI |
| `results/metrics/model_metrics*.csv` | Recorded model metrics from prior runs | Used by `scripts/evaluate_checkpoint.py --metrics ...` for comparison |
| `models/checkpoints/*.pt` | Saved PyTorch checkpoints | Loaded by `scripts/evaluate_checkpoint.py --checkpoint ...` |

## Notes On Reproducibility

The original evaluation uses `LinkNeighborLoader` with finite neighborhood
sampling. That means reported metrics can move slightly when PyTorch Geometric,
sampling backends, hardware, or random state differ. The checkpoint evaluation script sets
Python, NumPy, and PyTorch seeds so current runs are stable, but exact historical
matching still depends on the original Colab stack and sampler state.

The default replication target is:

```powershell
.\.conda\pipe-network-completion\python.exe scripts\evaluate_checkpoint.py `
  --checkpoint models\checkpoints\model1212_hiddensize_128_drop_00.pt `
  --metrics results\metrics\model_metrics1212.csv `
  --split test
```
