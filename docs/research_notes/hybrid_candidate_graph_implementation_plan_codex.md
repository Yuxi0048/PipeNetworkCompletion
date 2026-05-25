# Hybrid Candidate Utility Graph Implementation Plan

Workstream: Codex

## Why This Replaces The Road-Only Candidate Graph

The previous anchor-free plan used the road network as the full candidate graph:

```text
G_R = (V_R, E_R)
road edge e -> candidate utility corridor
```

That is useful for a road-corridor occupancy experiment, but it is too restrictive for horizontal utility-network reconstruction. A quick length-sampling check on the Brisbane sewer gravity-main truth layers estimated that only about 29% of truth length is within 10 m of roads and about 66% is within 50 m. Therefore, a strict road-edge candidate graph cannot represent much of the target network, regardless of classifier quality.

This plan updates the earlier anchor-free design in `docs/anchor_free_design.md`, the leakage audit in `docs/anchor_free_leakage_audit_codex.md`, and the decoder note in `docs/anchor_free_engineering_decoder_note_codex.md`.

## Research Question

Can we reconstruct/predict a planning-grade horizontal utility network topology without using ground utility anchor points?

The revised anchor-free formulation is:

```text
Build a candidate utility corridor graph from roads, building demand proxies,
and non-anchor context layers. Then classify candidate utility corridors.
```

Ground-truth utility geometry is still used only for labels and evaluation.

## Candidate Graph Design

The candidate graph should contain multiple edge families:

1. `road_backbone`
   - Original road centerline segments.
   - Preserves road class columns where available.

2. `building_access`
   - Short connector from building demand point or demand cluster to the nearest road-centerline point.
   - This represents plausible service/lateral access, not a known utility anchor.

3. `demand_knn`
   - k-nearest-neighbor candidate corridors between building demand clusters.
   - Allows off-road corridors through built-up areas where utility truth is not close to a road.

4. `demand_mst`
   - Minimum-spanning-tree edges over demand clusters, optionally constrained by maximum edge length.
   - Provides sparse network continuity without forcing every possible kNN edge.

Optional later edge families:

5. `cost_surface_path`
   - Least-cost paths using road proximity, built-up area, parcels/easements, slope, and land-use cost.

6. `source_sink_connector`
   - Only for sparse system facilities, if allowed by config and clearly disclosed.

## Anchor-Free Safety Rules

Allowed inputs:

- road centerlines
- building footprints or building points
- building/demand clusters derived from buildings
- DEM/slope
- built-up/land-use/parcels/zoning
- service area boundary

Forbidden inputs:

- manholes
- valves
- utility poles
- transformers/cabinets as dense anchors
- surveyed utility junctions
- pipe junctions
- endpoints or nodes derived from true utility lines
- any feature computed from utility truth geometry

The candidate graph builder must not read the utility truth path. The labeler may read truth only after candidate edges already exist.

## Implementation Scope

Add a new module:

```text
pipe_network_completion/anchor_free/hybrid_candidate_graph.py
```

Core functions:

```python
build_demand_points(buildings_gdf=None, building_points_gdf=None, target_crs=None)
cluster_demand_points(points_gdf, grid_size_m=150)
build_hybrid_candidate_lines(
    roads_gdf,
    buildings_gdf=None,
    building_points_gdf=None,
    target_crs=None,
    demand_cluster_grid_m=150,
    nearest_road_max_distance_m=300,
    knn_k=3,
    knn_max_distance_m=500,
    include_road_backbone=True,
    include_building_access=True,
    include_demand_knn=True,
    include_demand_mst=True,
)
```

The output is a GeoDataFrame of candidate LineStrings that can be passed into the existing heterogeneous graph builder. It should include:

- `candidate_source`
- `candidate_weight`
- `demand_u`
- `demand_v`
- `nearest_road_distance_m`
- original road class columns when available

Then update:

```text
pipe_network_completion/anchor_free/pipeline.py
```

New config:

```yaml
graph:
  candidate_graph_type: hybrid   # road, hybrid
  demand_cluster_grid_m: 150
  nearest_road_max_distance_m: 300
  demand_knn_k: 3
  demand_knn_max_distance_m: 500
  include_road_backbone: true
  include_building_access: true
  include_demand_knn: true
  include_demand_mst: true
```

For backward compatibility, the default remains:

```yaml
graph:
  candidate_graph_type: road
```

## Features

Reuse `build_road_segment_features` where possible. The hybrid candidate line table must be compatible with `build_hetero_road_graph`.

Add/retain non-anchor features:

- length
- bearing sin/cos
- endpoint degrees
- local candidate-corridor density
- candidate source one-hot encoding
- nearest building distance
- building count within buffer
- building point count/density within buffer
- built-up overlap within buffer
- DEM/slope where configured

Do not add utility-truth-derived features.

## Labels

Reuse `label_road_segments_from_utility_lines`, but interpret labels as candidate-corridor labels, not road-edge labels.

Recommended reporting:

- primary label buffer: 10 m
- secondary label buffer: 5 m
- representability at 5/10/20/30/50 m:
  - fraction of truth length within any candidate edge buffer
  - fraction of candidate length labeled positive

## Evaluation

Keep existing metrics, but rename interpretation:

- edge-level metrics become candidate-corridor metrics
- length metrics are conditional on the candidate graph

Add candidate graph diagnostics:

- candidate edge count by `candidate_source`
- candidate total length by `candidate_source`
- truth representability by buffer
- road-only representability vs hybrid representability

## Ablations

The research-rewarding ablation should be candidate-graph first, architecture second:

1. road-only candidate graph + RF
2. road + building access + RF
3. road + building access + demand kNN/MST + RF
4. best hybrid candidate graph + GraphSAGE
5. best hybrid candidate graph + connected decoder

Only after those are stable:

6. GraphSAGE vs GAT vs GraphConv on the same hybrid candidate graph

## Tests

Minimum tests:

1. Hybrid candidate builder does not require utility truth.
2. It produces road, building-access, and demand-connection candidate edges.
3. Off-road candidate edges can be labeled positive when truth is away from roads.
4. Candidate feature matrix has one row per candidate segment.
5. Anchor feature guard still catches forbidden names.
6. Synthetic pipeline runs with `candidate_graph_type: hybrid`.
7. Representability metric improves on a synthetic off-road utility line compared with road-only.

## Immediate Implementation Order

1. Add `hybrid_candidate_graph.py`.
2. Add pipeline switch `graph.candidate_graph_type`.
3. Add `candidate_source` one-hot feature support through existing road-class encoding.
4. Add candidate graph diagnostics/representability helpers.
5. Add per-`candidate_source` metrics so gains are not hidden by extra candidate density.
6. Add tests.
7. Run CUDA-env smoke tests.
8. Run a small real-data feature/label generation pass before training.

## Review Addendum Before Implementation

Subagent review confirmed the hybrid candidate graph is a stronger research direction than GNN-layer ablation, but the first implementation must guard against three credibility traps:

1. Dense hybrid candidates can inflate positive labels because the same truth line may be near multiple candidate edges. The implementation should report candidate length and positive prevalence by `candidate_source`, and later should add duplicate-truth-match diagnostics.
2. Hyperparameters such as demand grid size, kNN degree, and maximum candidate distance must be predefined or tuned only on validation data. They should not be silently changed after looking at test metrics.
3. Representability is a headline metric, not a side diagnostic. Every road-only vs hybrid comparison must show truth-length coverage and total candidate length at the same buffer widths.

For the first patch, source/sink connectors are excluded. Straight-line demand kNN/MST edges are treated as candidate demand-connectivity corridors, not inferred pipes.
