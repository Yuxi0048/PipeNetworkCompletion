# Data Artifacts

This directory is the expected local layout for replication data. Git tracks the
folder structure only; large pickles, shapefiles, CSV data exports, and model
inputs are distributed through release assets or a separate data archive.

Populate the artifacts after cloning:

```bash
python scripts/download_assets.py --version v1.0.0
```

Raw GIS files may require separate permission before redistribution. Prepared
graphs under `data/processed/graphs/` are enough for the default replication
command.
