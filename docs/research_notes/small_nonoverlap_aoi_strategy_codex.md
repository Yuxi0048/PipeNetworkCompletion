# Small Non-Overlapping AOI Strategy

Workstream: Codex

Date: 2026-05-23

Status: implementation note

## Purpose

The next experiments should start from small, spatially separated AOIs instead
of the full sewer-truth extent. This avoids mixing areas where roads/context do
not cover the truth and reduces spatial autocorrelation leakage from random
edge-level splits.

## Why AOIs First

The full gravity-sewer truth extent is much larger than the current road and
DEM extent. Raw building/context layers cover the larger extent, but roads and
DEM currently do not. Training on the full truth layer would therefore blend two
different problems:

- valid road/context-supported inference inside the common coverage area;
- unsupported truth outside the candidate-support coverage.

Small AOIs make the unit of analysis explicit. Each AOI can be checked for road
length, truth length, and demand/context availability before training starts.

## AOI Design

The implemented AOI builder creates square tiles with an explicit gap:

```text
AOI size: tile_size_m x tile_size_m
AOI separation: gap_m
```

The gap is important. Adjacent urban infrastructure is spatially autocorrelated;
touching train/test AOIs would still leak local spatial structure. A gap does
not eliminate all regional dependence, but it is more credible than random
edge-level splitting.

Default starting values:

```text
tile_size_m = 4000
gap_m = 1000
extent_mode = roads_and_buildings
```

Each AOI is filtered by minimum content:

```text
road_length_m >= 5000
truth_length_m >= 1000
building_point_count >= 25
```

These thresholds are deliberately low for the first pass. They can be raised
after inspecting the selected AOIs in QGIS.

## Implemented Files

Added:

- `pipe_network_completion/anchor_free/aoi.py`
- `scripts/build_anchor_free_aois.py`
- `tests/test_anchor_free_aoi.py`

The script writes:

```text
data/processed/aois/anchor_free_small_nonoverlap/
  aoi_grid.geojson
  aoi_summary.geojson
  aoi_summary.csv
  selected_aois.geojson
  selected_aois.csv
  manifest.json
```

With `--clip-layers`, it also writes one folder per selected AOI:

```text
data/processed/aois/anchor_free_small_nonoverlap/<aoi_id>/
  aoi.geojson
  roads.geojson
  utility_truth_gravity_mains.geojson
  building_points.geojson
  building_areas.geojson
  built_up.geojson
```

and per-AOI configs:

```text
configs/aois/anchor_free_small_aoi_<aoi_id>.yaml
```

## Command

Generate and clip initial small AOIs:

```powershell
.\.conda\pipe-network-completion-cuda\python.exe scripts\build_anchor_free_aois.py --clip-layers
```

This does not train a model.

## Recommended Experiment Order

1. Inspect `selected_aois.geojson` in QGIS.
2. Remove AOIs that are mostly boundary artifacts, industrial edge cases, or
   unsupported by roads/context.
3. Run candidate representability per AOI before any classifier training.
4. Train on train AOIs, tune on validation AOIs, and report final metrics on
   held-out test AOIs.
5. Keep random edge splits only as a secondary diagnostic, not the headline
   research result.

## Research Claim

AOI-level splitting supports the stronger claim:

```text
The method generalizes across spatially separated urban subregions, not merely
across randomly withheld edges from the same local network.
```

The model output should still be described as buffered utility-corridor
reconstruction, not exact manhole topology recovery.
