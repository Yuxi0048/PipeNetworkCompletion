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

## Required Local Inputs

`process.py` expects the raw inputs in this layout. Keep each `.shp` file with
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

## Artifact Access

Users should obtain source GIS data directly from the relevant public data
providers and follow their terms of use. The authors do not redistribute raw or
derived data artifacts unless separate written permission is obtained from the
relevant data providers.

Public GitHub releases should not attach raw or derived utility-network data
unless redistribution permission is documented.

