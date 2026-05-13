# Topology Prediction of Underground Utility Networks with Graph Neural Networks

[![CI](https://github.com/Yuxi0048/PipeNetworkCompletion/actions/workflows/ci.yml/badge.svg)](https://github.com/Yuxi0048/PipeNetworkCompletion/actions/workflows/ci.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.22260%2FISARC2024%2F0121-blue)](https://doi.org/10.22260/ISARC2024/0121)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

<!-- After enabling Zenodo for this repo (see RELEASE.md), uncomment: -->
<!-- [![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

Research code package for **Underground Utility Network Completion based on
Spatial Contextual Information of Ground Facilities and Utility Anchor Points
using Graph Neural Networks**.

The repository includes an installable Python package, command-line evaluation
scripts, environment checks, tests, documentation, release-based data artifacts,
and branch-based source traceability to the notebook-era repository state.

Refactored by Codex and Claude Code on 2026-05-13.

## Quick Start

### Windows (PowerShell)

```powershell
.\scripts\create_env.ps1
.\.conda\pipe-network-completion\python.exe -m pip install -e .
.\.conda\pipe-network-completion\python.exe scripts\download_assets.py --version v1.0.0
.\.conda\pipe-network-completion\python.exe scripts\verify_environment.py --load-data
.\.conda\pipe-network-completion\python.exe scripts\evaluate_checkpoint.py --max-batches 2
```

### macOS / Linux (bash)

```bash
conda env create -f environment.yml
conda activate pipe-network-completion
pip install -e .
python scripts/download_assets.py --version v1.0.0
python scripts/verify_environment.py --load-data
python scripts/evaluate_checkpoint.py --max-batches 2
```

Editable installation keeps `pipe_network_completion` importable from scripts,
notebooks, and tests.

### Full Test-Split Evaluation

```bash
python scripts/evaluate_checkpoint.py \
  --checkpoint models/checkpoints/model1212_hiddensize_128_drop_00.pt \
  --metrics results/metrics/model_metrics1212.csv \
  --split test
```

Supporting documentation: [docs/TRACEABILITY.md](docs/TRACEABILITY.md) maps
refactored modules to the notebook workflow and archived branches;
[docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md) documents artifact locations; and
[models/README.md](models/README.md) explains the checkpoint naming scheme.

## Pipeline

The command-line workflow follows the artifact lifecycle:

```text
data/raw/ + data/raw_gis/
  -> pipe_network_completion.process
  -> data/interim/
  -> scripts/build_graphs.py
  -> data/processed/graphs/
  -> scripts/evaluate_checkpoint.py
  -> checkpoint metrics
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
   python scripts/evaluate_checkpoint.py
   ```

Each script supports `--help`.

## Main Artifacts

- [process.py](process.py): raw GIS preprocessing script parameterized for local
  repo paths.
- [scripts/build_graphs.py](scripts/build_graphs.py): assembles
  `train_data.pkl`, `val_data.pkl`, `test_data.pkl` from the interim pickles.
- [pipe_network_completion/dataset.py](pipe_network_completion/dataset.py):
  graph dataset construction utilities.
- [pipe_network_completion/model.py](pipe_network_completion/model.py):
  importable GNN model definition refactored from the notebook.
- [pipe_network_completion/evaluation.py](pipe_network_completion/evaluation.py):
  shared binary classification metrics.
- [data/](data/): local layout for raw, interim, processed, and experiment data
  artifacts populated from release assets.
- [models/checkpoints/](models/checkpoints/): saved PyTorch model checkpoints
  (see [models/README.md](models/README.md) for the naming scheme).
- [results/metrics/](results/metrics/): recorded metrics from previous model
  runs.

## Source Traceability

The `main` branch is the maintained research code package. Notebook-era files
and earlier repository layouts are preserved in archived remote branches, so
readers can inspect the historical code without adding those files to the
current runnable tree.

```bash
git fetch --all --tags
git branch -r
git switch --detach origin/Legacy-Final
```

Use `origin/Legacy-Final` for the previous final research state and
`origin/Legacy-main` for the earlier main-branch snapshot. Return to the current
codebase with:

```bash
git switch main
```

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

Data and checkpoint artifacts are provided through the release bundle. The
download helper fetches the archive, verifies the SHA-256 sidecar, and extracts
the prepared `data/`, `models/`, and `results/` files into the working tree:

```bash
python scripts/download_assets.py --version v1.0.0
```

The standard release bundle contains prepared graphs, interim pickles,
split-shapefile exports, experiment variants, model checkpoints, and metrics.
Raw GIS inputs are managed as separate project data when distribution terms
require it.

## Tests

The pytest suite checks imports, path constants, and the checkpoint/metrics
inventory:

```bash
pip install -e ".[test]"
pytest
```

Artifact-backed validation is handled by
`scripts/verify_environment.py --load-data`.

## Releases

Tagged releases are published at
[github.com/Yuxi0048/PipeNetworkCompletion/releases](https://github.com/Yuxi0048/PipeNetworkCompletion/releases).
Each release bundles prepared artifacts into a
`pipe-network-artifacts-<version>.zip` asset with a SHA-256 sidecar. Fetch them
with:

```bash
python scripts/download_assets.py --version v1.0.0
```

The helper uses the [GitHub CLI](https://cli.github.com/) and prints the
equivalent `curl` command for manual downloads. The release procedure is
documented in [RELEASE.md](RELEASE.md), and the version history lives in
[CHANGELOG.md](CHANGELOG.md).

## Run Notes

Evaluation uses `LinkNeighborLoader` with finite neighborhood sampling, matching
the notebook workflow. The checkpoint evaluation script sets Python, NumPy, and
PyTorch seeds and reports observed metrics beside the published metrics row.

The archived notebook on `origin/Legacy-Final` records exploratory architecture
variants. The maintained CLI focuses on checkpoint evaluation with the prepared
graph artifacts.

## Acknowledgments

This project appreciates [Urban Utilities](https://www.arcgis.com/home/item.html?id=36fdac21178a4364a04f9516aa0703e5%2F1000)
for public access to high-quality utility network data, and
[Brisbane City Council Open Data](https://www.brisbane.qld.gov.au/business/tools-and-resources/open-data)
for public geospatial context used alongside those utility assets.

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
