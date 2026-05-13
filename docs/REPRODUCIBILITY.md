# Reproducibility

## Create The Environment

### Windows (PowerShell)

```powershell
.\scripts\create_env.ps1
.\.conda\pipe-network-completion\python.exe -m pip install -e .
```

`create_env.ps1` calls Miniforge from `%USERPROFILE%\miniforge3\Scripts\conda.exe`
by default; override with `-CondaExe <path>` if Miniforge lives elsewhere.

### macOS / Linux (bash)

```bash
conda env create -f environment.yml
conda activate pipe-network-completion
pip install -e .
```

Equivalent explicit Windows command (no helper script):

```powershell
& "$env:USERPROFILE\miniforge3\Scripts\conda.exe" env create `
  -p .\.conda\pipe-network-completion `
  -f environment.yml
```

## Verify Artifacts

Download the release asset first on a fresh clone:

```bash
python scripts/download_assets.py --version v1.0.0
```

```bash
python scripts/verify_environment.py --load-data
```

This checks the package stack, prepared graph pickles, model checkpoint, and
metrics CSV used by the default replication command. Without downloaded
artifacts, run `python scripts/verify_environment.py` to check only the Python
environment and package imports.

## Rebuild From Raw (Optional)

If `data/processed/graphs/` is empty or the interim pickles are missing, and
you have access to the raw GIS inputs:

```bash
python process.py
python scripts/build_graphs.py
```

`process.py` reads the raw GIS shapefiles under `data/raw/` and writes the
`*_proc.pkl` + `split_mask.pkl` artifacts to `data/interim/`.
`build_graphs.py` then assembles per-split `HeteroData` pickles into
`data/processed/graphs/`.

## Smoke Test Replication

```bash
python scripts/evaluate_checkpoint.py --max-batches 2
```

This confirms that the saved model, PyTorch Geometric sampler, and prepared
data can run end to end without waiting for a full split evaluation. Partial
smoke samples may report `nan` for AUC when the sampled labels contain only one
class; the full split command below performs the actual metrics comparison.

## Full Test-Split Replication

```bash
python scripts/evaluate_checkpoint.py \
  --checkpoint models/checkpoints/model1212_hiddensize_128_drop_00.pt \
  --metrics results/metrics/model_metrics1212.csv \
  --split test
```

By default the script compares the observed metrics with the matching `Testing`
row in `results/metrics/model_metrics1212.csv`. See
[../models/README.md](../models/README.md) for the checkpoint naming scheme
and architecture decoding.

## Tests

The `pytest` suite checks the package layout and checkpoint/metrics inventory:

```bash
pip install -e ".[test]"
pytest
```
