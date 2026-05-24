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
scripts, environment checks, tests, documentation, data-layout guidance, and
branch-based source traceability to the notebook-era repository state.

The initial version of this repository was published on December 26, 2023, and
the codebase was refactored by Codex and Claude Code on May 13, 2026.

> [!NOTE]
> The raw GIS inputs, interim pickles, and processed graph artifacts used in
> the referenced ISARC 2024 study are included under `data/` to support
> reproduction. Files exceeding GitHub's 50 MB warning threshold
> (`data/experiments/data_MH_Road_attr.pkl`,
> `data/processed/split_shapefiles/train.dbf`) are attached to GitHub Releases
> instead of tracked in Git history; download them from the matching release
> and place them at the documented paths. Model checkpoints are still not
> distributed. Subsequent data work added after 2026-05-13 (the context and
> buildings extension under `data/processed/aois/`,
> `data/processed/context/`, `data/raw/context/`) is excluded from this
> repository. Users remain responsible for following the original providers'
> terms of use when working with or further redistributing this data.

## Quick Start

### Windows (PowerShell)

```powershell
.\scripts\create_env.ps1
.\.conda\pipe-network-completion\python.exe -m pip install -e .
.\.conda\pipe-network-completion\python.exe scripts\verify_environment.py
```

### macOS / Linux (bash)

```bash
conda env create -f environment.yml
conda activate pipe-network-completion
pip install -e .
python scripts/verify_environment.py
```

Editable installation keeps `pipe_network_completion` importable from scripts,
notebooks, and tests.

### Full Test-Split Evaluation

Run this only after the local graph data, checkpoint, and metrics files are
available under the documented paths:

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
data/raw/
  gis/sewer/              Urban Utilities sewer shapefile bundles
  gis/roads/              Brisbane City Council road shapefile bundle
  mh_road/MH_Road.pkl     local manhole-road nearest-feature table
  -> process.py
  -> data/interim/*.pkl
  -> scripts/build_graphs.py
  -> data/processed/graphs/{train,val,test}_data.pkl
  + models/checkpoints/*.pt
  + results/metrics/*.csv
  -> scripts/evaluate_checkpoint.py
  -> checkpoint metrics
```

1. **Preprocess GIS shapefiles and the manhole-road near table** into
   `*_proc.pkl` + `split_mask.pkl`:

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
  artifacts generated from provider-supplied inputs.
- [models/checkpoints/](models/checkpoints/): saved PyTorch model checkpoints
  for local evaluation; checkpoint files are not distributed in this repository.
- [results/metrics/](results/metrics/): recorded metrics from previous model
  runs.

## Source Traceability

The `main` branch is the maintained research code package. Notebook-era source
code and earlier repository layouts are preserved in sanitized archived remote
branches, so readers can inspect the historical code without adding data files
to the current runnable tree.

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

The source code is public. The raw GIS inputs, interim pickles, and processed
graph artifacts used in the ISARC 2024 study are included under `data/` for
reproduction; the two files over 50 MB
(`data/experiments/data_MH_Road_attr.pkl` and
`data/processed/split_shapefiles/train.dbf`) are attached to GitHub Releases
rather than tracked in Git. Model checkpoints remain undistributed. Data work
added to the project after 2026-05-13 (context, buildings, and DEM extensions)
is not included here. Users remain responsible for following the original
providers' terms of use when working with or further redistributing any of
this data.

### Input Files For `process.py`

Place each shapefile bundle in the folder below. Keep every `.shp` file with
its matching sidecars, including `.dbf`, `.shx`, `.prj`, and any other files
exported with the same base name.

| File placement | Dataset / source | Type | Role in the workflow |
| --- | --- | --- | --- |
| `data/raw/gis/sewer/`<br>`SewerManholes_`<br>`ExportFeatures.shp` | Urban Utilities<br>sewer manholes | Point | Main manhole/anchor-point layer. Combined with `MH_Road.pkl` to build `MH_proc.pkl`. |
| `data/raw/gis/sewer/`<br>`SewerGravityMa_`<br>`ExportFeature1.shp` | Urban Utilities<br>sewer gravity main - trunk | Line | Trunk gravity-main segments. Combined with `SewerGravityMa_ExportFeature2.shp` to build `Line_proc.pkl`. |
| `data/raw/gis/sewer/`<br>`SewerGravityMa_`<br>`ExportFeature2.shp` | Urban Utilities<br>sewer gravity main | Line | Main gravity sewer segments. Combined with `SewerGravityMa_ExportFeature1.shp` to build `Line_proc.pkl`. |
| `data/raw/gis/sewer/`<br>`SewersqlSewerP_`<br>`ExportFeature.shp` | Urban Utilities<br>sewer pump assets | Point | Loaded by `process.py` as pump point assets. In the current preprocessing path, pump rows without the manhole-road near-table fields are filtered before `MH_proc.pkl` is written. |
| `data/raw/gis/roads/`<br>`Roads_`<br>`ExportFeatures.shp` | Brisbane City Council Open Data<br>road hierarchy | Line | Road context layer used to build road nodes and road-road relationships. |
| `data/raw/mh_road/`<br>`MH_Road.pkl` | Locally generated<br>manhole-road near table | Table | Links manholes to nearby roads. Expected fields are `OBJECTID`, `NEAR_FID`, `NEAR_POS`, `NEAR_DIST`, and `SIDE`. |

If pump assets should be represented as graph nodes, revise `process.py` and
regenerate the derived artifacts; otherwise the table above documents the
current notebook-compatible preprocessing path.

Generated local artifacts:

- `data/interim/MH_proc.pkl`
- `data/interim/Road_proc.pkl`
- `data/interim/MH_R_RL_proc.pkl`
- `data/interim/Line_proc.pkl`
- `data/interim/R_R_proc.pkl`
- `data/interim/split_mask.pkl`
- `data/processed/graphs/train_data.pkl`
- `data/processed/graphs/val_data.pkl`
- `data/processed/graphs/test_data.pkl`

Files over GitHub's 50 MB warning threshold are attached to releases rather
than committed to Git history (see the note above).

## Running With Restricted Data

Readers can use the repository at two levels:

1. **Code and environment check**: clone the repository, create the environment,
   install the package, and run `scripts/verify_environment.py`. This path does
   not require raw GIS files or model artifacts.
2. **Provider-data workflow**: obtain GIS data directly from the relevant public
   data providers, follow their terms of use, place the files in the documented
   `data/raw/` layout, then run `process.py` and `scripts/build_graphs.py`.
   Model evaluation requires locally generated graph artifacts and a checkpoint
   the reader is permitted to use.

## Tests

The pytest suite checks imports, path constants, and the checkpoint/metrics
inventory:

```bash
pip install -e ".[test]"
pytest
```

Artifact-backed validation for local provider data is handled by
`scripts/verify_environment.py --load-data`.

## Releases

Tagged source releases are published at
[github.com/Yuxi0048/PipeNetworkCompletion/releases](https://github.com/Yuxi0048/PipeNetworkCompletion/releases).
Public releases contain source code and documentation. Data archives are not
provided through this repository. The release procedure is documented in
[RELEASE.md](RELEASE.md), and the version history lives in
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
for public geospatial context used alongside those utility assets. Users are
responsible for following provider terms when accessing or redistributing raw or
derived data.

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
