# Data Layout

The repository organizes artifacts by lifecycle. Git tracks the folder layout,
documentation, and small metadata; large data/checkpoint artifacts are
distributed through release assets.

```text
data/
  raw/
    gis/
      roads/
      sewer/
    mh_road/
  interim/
  processed/
    graphs/
    split_shapefiles/
  experiments/

models/
  checkpoints/

results/
  metrics/
```

## Folders

| Folder | Contents |
| --- | --- |
| `data/raw/gis/sewer/` | Original sewer shapefile bundles and sidecar GIS files |
| `data/raw/gis/roads/` | Original road shapefile bundle and sidecar GIS files |
| `data/raw/mh_road/` | Original manhole-road near-table files, including `MH_Road.pkl` and `MH_Road.csv` |
| `data/interim/` | Preprocessed intermediate pickles such as `MH_proc.pkl`, `Road_proc.pkl`, `Line_proc.pkl`, and `split_mask.pkl` |
| `data/processed/graphs/` | Model-ready PyTorch Geometric graph pickles: `train_data.pkl`, `val_data.pkl`, `test_data.pkl` |
| `data/processed/split_shapefiles/` | Train/validation/test line shapefiles exported from the notebook |
| `data/experiments/` | Alternative prepared graph/data variants used during experimentation |
| `models/checkpoints/` | Saved PyTorch checkpoints |
| `results/metrics/` | Recorded metric CSV files |

Maintained code should read from these paths through
`pipe_network_completion.paths` instead of hard-coding file locations.

## Artifact Delivery

On a fresh clone, populate the large files with:

```bash
python scripts/download_assets.py --version v1.0.0
```

The standard release bundle includes prepared graph pickles, interim pickles,
split-shapefile exports, experiment variants, model checkpoints, and metrics.
Raw GIS files may be omitted from public release assets if licensing or data
ownership requires separate distribution.

