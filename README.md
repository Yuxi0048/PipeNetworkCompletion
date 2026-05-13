# PipeNetworkCompletion

[![CI](https://github.com/Yuxi0048/PipeNetworkCompletion/actions/workflows/ci.yml/badge.svg)](https://github.com/Yuxi0048/PipeNetworkCompletion/actions/workflows/ci.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.22260%2FISARC2024%2F0121-blue)](https://doi.org/10.22260/ISARC2024/0121)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

<!-- After enabling Zenodo for this repo (see RELEASE.md), uncomment: -->
<!-- [![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

Code, data, and saved model artifacts for **Underground Utility Network
Completion based on Spatial Contextual Information of Ground Facilities and
Utility Anchor Points using Graph Neural Networks**.

The original research notebook and notebook-era helper code are kept under
[legacy/](legacy/). Refactored modules and scripts provide the installable
package and CLI entry points for environment checks and replication.

Refactored by Codex and Claude Code on 2026-05-13.

## Quick Start

### Windows (PowerShell)

```powershell
.\scripts\create_env.ps1
.\.conda\pipe-network-completion\python.exe -m pip install -e .
.\.conda\pipe-network-completion\python.exe scripts\download_assets.py --version v1.0.0
.\.conda\pipe-network-completion\python.exe scripts\verify_environment.py --load-data
.\.conda\pipe-network-completion\python.exe scripts\replicate_results.py --max-batches 2
```

### macOS / Linux (bash)

```bash
conda env create -f environment.yml
conda activate pipe-network-completion
pip install -e .
python scripts/download_assets.py --version v1.0.0
python scripts/verify_environment.py --load-data
python scripts/replicate_results.py --max-batches 2
```

`pip install -e .` is optional but recommended; it makes the package importable
from any working directory.

### Full Test-Split Replication

```bash
python scripts/replicate_results.py \
  --checkpoint models/checkpoints/model1212_hiddensize_128_drop_00.pt \
  --metrics results/metrics/model_metrics1212.csv \
  --split test
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for details,
[docs/TRACEABILITY.md](docs/TRACEABILITY.md) for how the refactored files map
back to the original notebook, [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md) for
the artifact layout, and [models/README.md](models/README.md) for the
checkpoint naming scheme.

## Pipeline

The reproducible CLI is three steps:

```text
data/raw/  ──process.py──▶  data/interim/  ──scripts/build_graphs.py──▶  data/processed/graphs/  ──scripts/replicate_results.py──▶  metrics
```

1. **Preprocess GIS shapefiles** into `*_proc.pkl` + `split_mask.pkl`:

   ```bash
   python process.py
   ```

2. **Assemble per-split HeteroData graphs** from those pickles:

   ```bash
   python scripts/build_graphs.py
   ```

3. **Evaluate a saved checkpoint** against the published metrics:

   ```bash
   python scripts/replicate_results.py
   ```

Each script supports `--help`.

## Main Artifacts

- [legacy/Inductive.ipynb](legacy/Inductive.ipynb): original Colab notebook and
  source of the training, evaluation, explanation, and plotting workflow.
- [legacy/Module/](legacy/Module/): original notebook support namespace retained
  for traceability.
- [process.py](process.py): raw GIS preprocessing script, now parameterized for
  local repo paths.
- [scripts/build_graphs.py](scripts/build_graphs.py): assembles
  `train_data.pkl`, `val_data.pkl`, `test_data.pkl` from the interim pickles.
- [pipe_network_completion/dataset.py](pipe_network_completion/dataset.py):
  graph dataset construction utilities.
- [pipe_network_completion/model.py](pipe_network_completion/model.py):
  importable GNN model definition refactored from the notebook.
- [pipe_network_completion/evaluation.py](pipe_network_completion/evaluation.py):
  shared binary classification metrics.
- [data/](data/): placeholder layout for raw, interim, processed, and
  experiment data artifacts. Large artifacts are distributed through releases.
- [models/checkpoints/](models/checkpoints/): saved PyTorch model checkpoints
  (see [models/README.md](models/README.md) for the naming scheme).
- [results/metrics/](results/metrics/): recorded metrics from previous model
  runs.

## Data

Prepared artifacts:

- `data/interim/MH_proc.pkl`
- `data/interim/Road_proc.pkl`
- `data/interim/MH_R_RL_proc.pkl`
- `data/interim/Line_proc.pkl`
- `data/interim/R_R_proc.pkl`
- `data/interim/split_mask.pkl`
- `data/processed/graphs/train_data.pkl`
- `data/processed/graphs/val_data.pkl`
- `data/processed/graphs/test_data.pkl`

Raw GIS artifacts:

- `data/raw/gis/sewer/SewerManholes_ExportFeatures.shp`
- `data/raw/gis/sewer/SewerGravityMa_ExportFeature2.shp`
- `data/raw/gis/sewer/SewerGravityMa_ExportFeature1.shp`
- `data/raw/gis/sewer/SewersqlSewerP_ExportFeature.shp`
- `data/raw/gis/roads/Roads_ExportFeatures.shp`
- `data/raw/mh_road/MH_Road.pkl`

Large data and checkpoint artifacts are not tracked in Git. Download the
release asset before running replication:

```bash
python scripts/download_assets.py --version v1.0.0
```

The default release bundle contains prepared graphs, interim pickles,
split-shapefile exports, experiment variants, model checkpoints, and metrics.
Raw GIS inputs may be distributed separately if data licensing requires it.

## Tests

The pytest suite checks imports, path constants, and the checkpoint/metrics
inventory:

```bash
pip install -e ".[test]"
pytest
```

These tests do not load checkpoints or large pickles.

## Releases

Tagged releases are published at
[github.com/Yuxi0048/PipeNetworkCompletion/releases](https://github.com/Yuxi0048/PipeNetworkCompletion/releases).
Each release bundles prepared artifacts into a
`pipe-network-artifacts-<version>.zip` asset with a SHA-256 sidecar. Fetch them
with:

```bash
python scripts/download_assets.py --version v1.0.0
```

The helper requires the [GitHub CLI](https://cli.github.com/); if `gh` is
unavailable it prints the equivalent `curl` command. The release procedure is
documented in [RELEASE.md](RELEASE.md), and the version history lives in
[CHANGELOG.md](CHANGELOG.md).

## Reproducibility Notes

The original notebook used `LinkNeighborLoader` with finite neighborhood
sampling. The replication script sets Python, NumPy, and PyTorch seeds, but
exact historical metric matching can still vary slightly with PyTorch
Geometric, sampler backend, hardware, and random state.

For traceability, the legacy notebook remains the record for architecture
variants that were only kept as commented cells. The refactored CLI targets the
saved model/data evaluation path.

## Citation

```bibtex
@inproceedings{10.22260/ISARC2024/0121,
  doi = {10.22260/ISARC2024/0121},
  year = {2024},
  month = {June},
  author = {Zhang, Yuxi and Cai, Hubo},
  title = {Underground Utility Network Completion based on Spatial Contextual Information of Ground Facilities and Utility Anchor Points using Graph Neural Networks},
  booktitle = {Proceedings of the 41st International Symposium on Automation and Robotics in Construction},
  isbn = {978-0-6458322-1-1},
  issn = {2413-5844},
  publisher = {International Association for Automation and Robotics in Construction (IAARC)},
  pages = {936-943},
  address = {Lille, France}
}
```

Contact: Yuxi Zhang, zhan2889@purdue.edu

## Location Encoder Citation

The location encoder implementation is adapted from the space2vec/grid-cell
spatial representation work:

```bibtex
@inproceedings{space2vec_iclr2020,
  title = {Multi-Scale Representation Learning for Spatial Feature Distributions using Grid Cells},
  author = {Mai, Gengchen and Janowicz, Krzysztof and Yan, Bo and Zhu, Rui and Cai, Ling and Lao, Ni},
  booktitle = {The Eighth International Conference on Learning Representations},
  year = {2020},
  organization = {OpenReview}
}
```
