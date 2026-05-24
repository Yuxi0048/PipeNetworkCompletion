# Data Artifacts

This directory holds the data artifacts used by the ISARC 2024 study. The
layout below mirrors the artifact lifecycle documented in the top-level
[README](../README.md) and [docs/DATA_LAYOUT.md](../docs/DATA_LAYOUT.md).

## What is included

The following are tracked in this repository:

- `raw/gis/sewer/`, `raw/gis/roads/` — provider shapefile bundles used as
  inputs to `process.py`.
- `raw/mh_road/MH_Road.pkl` — the locally generated manhole-road near table.
- `interim/*.pkl` — outputs of `process.py`.
- `processed/graphs/{train,val,test}_data.pkl` — outputs of
  `scripts/build_graphs.py`.
- `processed/split_shapefiles/` — train/val/test shapefile splits used by the
  study.
- `experiments/data_MH_Only.pkl`, `experiments/data_MH_Road_Connectivity.pkl`
  — experiment-specific intermediate datasets.

## Attached to GitHub Releases instead of Git history

Files exceeding GitHub's 50 MB warning threshold are not committed; download
them from the matching GitHub Release and place them at these paths:

- `data/experiments/data_MH_Road_attr.pkl`
- `data/processed/split_shapefiles/train.dbf`

## Not included

- Model checkpoints (`models/checkpoints/*.pt`).
- Data work added to the project after 2026-05-13: the context, buildings,
  and DEM extension layers under `processed/aois/`, `processed/context/`,
  and `raw/context/`.

Users are responsible for following the original providers' terms of use when
working with or further redistributing any of this data.
