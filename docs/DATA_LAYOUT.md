# Data Layout

The repository organizes artifacts by lifecycle. Git tracks the folder layout,
documentation, and small metadata; raw GIS files, processed graph pickles, and
model checkpoints are not redistributed in the public repository.

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

## Artifact Access

Users should obtain source GIS data directly from the relevant public data
providers and follow their terms of use. The authors do not redistribute raw or
derived data artifacts unless separate written permission is obtained from the
relevant data providers.

Public GitHub releases should not attach raw or derived utility-network data
unless redistribution permission is documented.

