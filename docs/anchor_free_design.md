# Anchor-Free Utility Network Prediction (Design Note)

Workstream: Codex + Claude merge

This document describes an experimental, additive variant of the ISARC 2024
pipeline that removes ground anchor points (manholes, pumps, valves, poles, and
similar surveyed utility-node coordinates) from the model inputs. It is an
**ablation companion** to the published anchor-based GNN, not a replacement.

> Disclaimer
>
> The anchor-free model is a planning/research-grade probabilistic prediction
> experiment. Outputs must not be used for excavation clearance, legal utility
> locating, or any safety-critical decision. Ground-truth utility geometry is
> only used to build labels and to evaluate predictions, never as a feature at
> inference.

## Research question

Can we predict utility (sewer/water) network topology without using ground
anchor points as model inputs?

Existing approach:

    anchor-based GNN link prediction between MH (manhole) nodes
    where MH-MH gravity-main edges are both the prediction target and a
    message-passing graph

New approach:

    anchor-free, road-constrained probabilistic utility-network generation,
    where road segments are candidate utility corridors and the model predicts
    p_e = P(road edge e carries a utility line | road/building/elevation/context)

## Existing ISARC pipeline (anchor-based)

```
data/raw/gis/sewer/SewerManholes_ExportFeatures.shp   <- anchors
data/raw/gis/sewer/SewerGravityMa_ExportFeature{1,2}.shp <- utility truth
data/raw/gis/roads/Roads_ExportFeatures.shp           <- road context
data/raw/mh_road/MH_Road.pkl                          <- anchor near-table
                          |
                          v
                      process.py
                          |
                          v
data/interim/{MH,Road,Line,MH_R_RL,R_R}_proc.pkl + split_mask.pkl
                          |
                          v
                scripts/build_graphs.py
                          |
                          v
data/processed/graphs/{train,val,test}_data.pkl
                          |
                          v
              scripts/evaluate_checkpoint.py
```

`pipe_network_completion.dataset.dataset()` assembles a `HeteroData` with:

| node type | source                              | leakage risk |
| --------- | ----------------------------------- | ------------ |
| `MH`      | sewer manholes + pumps              | **direct anchor leakage** |
| `Road`    | road shapefile **filtered** to roads near at least one MH | indirect anchor leakage via filtering |

and edge types:

| edge type                                | purpose                       | leakage risk |
| ---------------------------------------- | ----------------------------- | ------------ |
| `("MH","link","MH")`                     | prediction target             | label; OK to use as labels only |
| `("MH","near","Road")`                   | message-passing context       | **direct anchor leakage** |
| `("Road","link","Road")`                 | road-road adjacency context   | safe in principle, but the `Road` table itself is anchor-filtered |

`pipe_network_completion.model.Model` then runs a heterogeneous GAT/SAGE stack
(`to_hetero`) over those node/edge dictionaries and uses
`Classifier` (dot product on MH embeddings) to score candidate MH-MH edges.

The anchor-based path is preserved unchanged; existing commands
(`process.py`, `scripts/build_graphs.py`, `scripts/evaluate_checkpoint.py`)
continue to work.

## Anchor leakage in the existing path

The anchor-free design has to avoid these inputs:

1. **MH node features and coordinates** - feed the model the manhole
   distribution.
2. **MH-Road near-table edges** - encode "there is an anchor here" through edge
   incidence.
3. **MH-MH adjacency at message passing** - only the *label* is allowed to come
   from the utility-truth lines; using gravity-main MH-MH edges as a
   message-passing graph leaks utility topology.
4. **Anchor-filtered road table** - `process.py` drops roads with no nearby
   manhole. The anchor-free path re-reads the raw road shapefile (or a
   GeoJSON/GPKG equivalent) and does *not* filter it by MH proximity.
5. **Engineered features that proxy anchors** - distance-to-manhole,
   manhole-density, manhole-degree, etc. The feature guard rejects any column
   whose name matches forbidden tokens.

## Anchor-free pipeline (new)

```
data/raw/gis/roads/Roads_ExportFeatures.shp              <- candidate corridors
data/raw/gis/sewer/SewerGravityMa_ExportFeature{1,2}.shp <- labels only
[optional] buildings, parcels, land use, DEM, source/sink facilities
                          |
                          v
        pipe_network_completion.anchor_free.road_graph
                          |
                          v
                  road candidate graph G_R
                          |
                          v
        pipe_network_completion.anchor_free.features       <-- guarded
                          |
                          v
        pipe_network_completion.anchor_free.labels         <-- truth lines, labels only
                          |
                          v
   baseline (LogReg / RandomForest)  or  road-only GNN
                          |
                          v
        pipe_network_completion.anchor_free.decoder
        (threshold or cost-weighted Steiner-like decoder)
                          |
                          v
        pipe_network_completion.anchor_free.evaluation
                          |
                          v
   outputs/<experiment_name>/anchor_free/
      edge_predictions.geojson
      decoded_network.geojson
      metrics.json
      metrics.csv
      prediction_map.png
      config_resolved.yaml
```

For feature preparation without fitting a model, use:

```
scripts/prepare_anchor_free_features.py
```

It writes `road_edge_features.csv`, `road_edge_features.geojson`,
`road_graph_nodes.geojson`, `feature_metadata.json`, and `config_resolved.yaml`
under `outputs/<experiment_name>/features/`.

## Allowed and forbidden inputs

**Allowed**: road centerlines, road intersections, road segment geometry, road
class, building footprints, parcels, land use, zoning, DEM/elevation, slope,
population/demand proxies, service-area boundary, sparse source/sink facility
points (treated as a small fixed set, not dense anchors).

**Forbidden** (rejected by `anchor_free.features.assert_no_anchor_features`):
manhole, valve, pole, transformer, cabinet, anchor, utility_node, facility_node,
surveyed_node, mh, and substrings derived from them.

## Files added

```
pipe_network_completion/anchor_free/__init__.py
pipe_network_completion/anchor_free/road_graph.py        # candidate graph from roads
pipe_network_completion/anchor_free/features.py          # edge features + guard
pipe_network_completion/anchor_free/labels.py            # truth-line labeling
pipe_network_completion/anchor_free/baseline.py          # LogReg / RF baseline
pipe_network_completion/anchor_free/model.py             # tiny road-edge GNN
pipe_network_completion/anchor_free/decoder.py           # threshold + connected decoders
pipe_network_completion/anchor_free/evaluation.py        # edge + network metrics
pipe_network_completion/anchor_free/synthetic.py         # tiny synthetic generator
pipe_network_completion/anchor_free/pipeline.py          # end-to-end orchestration
pipe_network_completion/anchor_free/config.py            # YAML config loading
configs/anchor_free_isarc2024.yaml                       # default config
configs/anchor_free_real_context_features.yaml           # real road/building/DEM feature prep
scripts/prepare_anchor_free_features.py                  # feature export, no training
scripts/train_anchor_free.py                             # train + decode + evaluate
scripts/run_anchor_free_ablation.py                      # ablation runner
tests/test_anchor_free_pipeline.py                       # smoke tests for the path
docs/anchor_free_design.md                                # this file
```

The maintained anchor-based implementation files are not changed. The merged
anchor-free path keeps the Codex implementation modules and the Claude
module-level smoke tests. The pipeline reports full-network decoded metrics and
split-prefixed edge metrics (`train_*`, `val_*`, `test_*`) so model performance
can be read without mixing held-out edges with training edges.

## How to run

### Smoke test on the synthetic fixture

```bash
python scripts/train_anchor_free.py --config configs/anchor_free_isarc2024.yaml --synthetic
```

This runs the full pipeline on a tiny grid generated in memory. No raw GIS
data is required. Outputs land under
`outputs/anchor_free_isarc2024/anchor_free/`.

### Real data

Place road and utility-truth files in the paths referenced by the config and
run without `--synthetic`:

```bash
python scripts/train_anchor_free.py --config configs/anchor_free_isarc2024.yaml
```

### Feature export only

Prepare road, building-area, built-up-area, and DEM features without labels or
training. If `data.building_points_path` is configured, the export also adds
separate building-point count/proximity/category-count features:

```bash
python scripts/prepare_anchor_free_features.py --config configs/anchor_free_real_context_features.yaml
```

Add `--with-labels` only when you also want a separate `road_edge_labels.csv`.
The labels still come from utility truth lines and are not included as model
input features.

To prepare train-ready files without fitting a model:

```bash
python scripts/prepare_anchor_free_features.py --config configs/anchor_free_real_context_features.yaml --training-ready --label-buffer-meters 10 5 --no-geojson
```

This writes, for each label buffer:

- `road_edge_training_table_<buffer>.csv`: `edge_id`, `split`, `y`, and model
  feature columns only.
- `road_edge_features_standardized_<buffer>.csv`: z-scored feature matrix using
  train-split statistics.
- `feature_scaling_<buffer>.csv`: train-split mean and standard deviation.
- `train_val_test_split_<buffer>.csv`: deterministic split assignment.
- `road_edge_labels_<buffer>.csv`: label diagnostics including overlap length
  and overlap ratio; these columns are not model input features.

The real context config currently prepares both `10m` and `5m` labels. The
`5m` version is a stricter spatial-match target for evaluation sensitivity.
Future training runs using `scripts/train_anchor_free.py` also report
`label_5m_*` metrics when `evaluation.extra_label_buffers_m: [5]` is set.

### Ablation

Compare baseline / GNN / decoder variants and write `ablation_results.csv`:

```bash
python scripts/run_anchor_free_ablation.py --config configs/anchor_free_isarc2024.yaml --synthetic
```

### Tests

```bash
pytest tests/test_anchor_free_*.py
```

## Limitations / TODO

- The road-only GNN is intentionally small and is not a tuned production
  architecture; the goal is to compare against the anchor-based GNN and the
  classical baseline, not to set a new SOTA.
- Connected-decoder is a cost-weighted minimum spanning forest over high-score
  edges, not a full mixed-integer Steiner-tree solve.
- DEM/slope features are implemented through GDAL raster sampling when
  `data.dem_path` is supplied. Source/sink engineering terms are still a TODO.
- Spatial block train/val/test split uses an axis-aligned grid; for harder
  generalization studies a hand-drawn boundary or clustering split should be
  added.
