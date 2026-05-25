# Anchor-Based Refactor With Building Context Anchors

Workstream: Codex

This note proposes a maintainable refactor for augmenting the original
ISARC-style anchor-based graph with OpenStreetMap building-footprint centroid
anchors and richer sewer asset anchors. It is a code-organization plan, not a
claim that building centroids are true utility nodes.

## Current Limitation

The original pipeline is compact but brittle:

- `process.py` builds a single anchor table named `MH_proc.pkl`.
- `pipe_network_completion.dataset.dataset()` assumes one utility anchor node
  type, `MH`, and slices feature columns by hard-coded offsets.
- Ground-truth line labels are kept only when a sewer gravity line can be
  represented as an edge between two retained anchors.
- Road context is attached through `MH_Road.pkl`, a precomputed nearest-road
  table for manholes.

That structure is acceptable for reproducing the paper, but it is not a good
long-term structure for comparing anchor policies, adding building evidence, or
auditing which line systems are excluded.

## New Artifacts Already Prepared

OSM building footprints have been downloaded for the 115 selected Brisbane AOIs:

- `data/raw/context/buildings/osm_buildings_selected_aois.geojson`

Building centroid/context-anchor artifacts have been derived:

- `data/processed/context/study_area/osm_building_anchor_points.geojson`
- `data/processed/context/study_area/osm_building_anchor_features.csv`

The OSM building centroid table contains 73,944 point anchors. Type groups:

| building type group | count |
| --- | ---: |
| residential | 41,669 |
| unknown | 21,096 |
| ancillary | 4,735 |
| commercial | 2,813 |
| institutional | 1,990 |
| industrial | 1,073 |
| other | 561 |
| utility | 7 |

Each point has:

- `anchor_family = building`
- `anchor_source = osm_building_footprint`
- `visibility = surface_inferred_from_osm`
- `is_utility_asset = false`
- `is_surface_visible = true`
- raw OSM building type
- normalized building type group
- one-hot group columns such as `bt_group_residential`
- one-hot columns for common raw OSM building tags
- footprint area and perimeter
- centroid coordinates

Most point geometries use the footprint centroid. For 400 polygons where the
centroid fell outside the footprint, the point falls back to a point-on-surface.

## Anchor Taxonomy

Do not put every point into one `MH` table. Use a canonical `anchors` table.

Required columns:

| column | meaning |
| --- | --- |
| `anchor_id` | stable id scoped by source |
| `anchor_family` | `utility`, `building`, `road_context`, etc. |
| `anchor_subtype` | manhole, pump_station, building_residential, fitting, etc. |
| `source_layer` | original file/layer |
| `visibility` | `surface_visible`, `surface_inferred`, `hidden_utility_record` |
| `leakage_risk` | `none`, `low`, `medium`, `high` |
| `is_utility_asset` | true for sewer asset records |
| `is_surface_visible` | true only if observable or defensibly surface-visible |
| `geometry` | point geometry in the project CRS |

Recommended anchor families:

1. `utility_surface`
   - manholes
   - sewer ends
   - pump stations
   - vents
   - valve covers / control valves
   - system valves
   - treatment plants

2. `utility_hidden`
   - fittings
   - tees
   - reducers
   - wyes
   - end caps
   - other underground utility point records

3. `building_context`
   - OSM building centroid points
   - government building points
   - footprint-derived centroids

The building anchors are not true utility nodes. They should be graph context
nodes or demand anchors, not endpoints of gravity-main labels.

## Refactor Structure

Add a new package rather than modifying the original paper reproduction path:

```text
pipe_network_completion/
  anchor_augmented/
    schemas.py              # canonical columns, allowed anchor policies
    sewer_assets.py         # load sewer point/line layers
    building_anchors.py     # OSM/government building point derivation
    line_matching.py        # match line endpoints to selected anchors
    orphan_audit.py         # two-anchor exclusion accounting
    graph_builder.py        # build HeteroData from canonical tables
    features.py             # typed node/edge feature encoders
    config.py               # YAML/JSON config loading
scripts/
  derive_osm_building_anchor_points.py
  audit_anchor_line_orphans.py
  build_anchor_augmented_graphs.py
configs/
  anchor_augmented/
    gravity_manhole_only.yaml
    gravity_visible_surface_plus_buildings.yaml
    gravity_all_utility_points_plus_buildings.yaml
```

Keep `process.py` and `pipe_network_completion.dataset` as the legacy
reproduction path. New experiments should use the additive package.

## Graph Design

Use a heterogeneous graph:

```text
Node types:
  UtilityAnchor
  BuildingAnchor
  Road

Edges:
  UtilityAnchor --gravity_link_label--> UtilityAnchor   # prediction target
  UtilityAnchor --near--> Road
  BuildingAnchor --near--> Road
  BuildingAnchor --near--> UtilityAnchor                # optional context only
  Road --link--> Road
  BuildingAnchor --near--> BuildingAnchor               # optional local demand graph
```

The original paper target remains utility-anchor link prediction:

```text
P(utility anchor u connected to utility anchor v)
```

Building centroids add demand/context information, for example:

- local building density
- building type mix
- distance from utility anchors to residential/commercial/institutional demand
- building-to-road relations
- building-to-utility-anchor relations in training only if framed as context,
  not as target labels

For service laterals, use a separate task:

```text
P(building or parcel connects to nearest predicted/known main)
```

Do not mix lateral prediction into the gravity-main topology task unless the
paper explicitly defines a multi-task model.

## Configurable Anchor Policies

Every result table should say which policy was used:

```yaml
anchor_policy:
  utility_points: visible_surface   # manholes_only, visible_surface, all_utility_points
  include_hidden_fittings: false
  include_osm_buildings: true
  building_anchor_role: context     # context, demand_terminal, target_endpoint
  endpoint_match_tolerance_m: 5
```

Recommended headline policies:

1. `manholes_only`
   - original paper reproduction baseline.

2. `visible_surface_plus_buildings`
   - manholes, ends, pump stations, vents, valves, devices, treatment plants;
   - OSM building centroids as context anchors.

3. `all_utility_points_plus_buildings`
   - includes hidden fittings;
   - useful diagnostic upper bound;
   - high leakage risk, not a fair surface-observable experiment.

## Orphan Lines Excluded By The Two-Anchor Assumption

A line is representable only if both endpoints match at least one selected
anchor within tolerance. Lines with zero or one matched endpoint are orphaned by
the two-anchor assumption.

Current endpoint audit, using 5 m tolerance:

| line group | manholes only orphans | visible surface orphans | visible + hidden fittings orphans |
| --- | ---: | ---: | ---: |
| gravity mains | 37,530 / 243,773 | 36,886 / 243,773 | 6,979 / 243,773 |
| pressure/rising mains | 5,143 / 6,575 | 1,376 / 6,575 | 1,196 / 6,575 |
| service laterals | 250,464 / 346,000 | 250,355 / 346,000 | 223,971 / 346,000 |
| vent pipes | 324 / 634 | 61 / 634 | 49 / 634 |

For the original paper-style gravity-main task, the important number is:

```text
visible surface policy, 5 m: 36,886 gravity-main lines orphaned
```

The original `process.py` spatial-intersection workflow produces 203,203
processed gravity-main records. The independent endpoint audit gives a very
close but not identical manhole-only count because it uses endpoint nearest
anchor matching instead of the original line/point intersection operation.

## Interpretation

Adding hidden fittings dramatically reduces gravity-main orphans:

```text
visible surface anchors:          36,886 gravity orphans
visible + hidden utility fittings: 6,979 gravity orphans
```

That is scientifically useful, but it also shows why hidden fittings are a
leakage-prone input: they are underground utility inventory points and reveal
latent topology. Treat them as an upper-bound diagnostic, not a surface-only
model.

Adding OSM building centroid anchors does not directly reduce gravity-main
orphans, because buildings are not utility line endpoints. Their role is to
improve node/edge context and demand-side reasoning.

## Implementation Sequence

1. Keep the original reproduction path frozen.
2. Build canonical anchor tables:
   - sewer surface anchors;
   - sewer hidden anchors;
   - OSM building context anchors.
3. Match sewer line endpoints to selected utility anchor policies.
4. Write `line_anchor_matches` with:
   - `line_id`;
   - `line_group`;
   - `subtype`;
   - `start_anchor_id`;
   - `end_anchor_id`;
   - `orphan_reason`;
   - `included_in_graph`.
5. Build `HeteroData` using typed nodes and typed edges.
6. Run three policies as separate experiments, not as one blended result.
7. Report orphan rates beside every performance table.

## Research Framing

Best long-term story:

> The original ISARC task is anchor-based sewer gravity-main topology
> completion between known utility anchors. We extend it into a typed,
> policy-controlled anchor graph that separates utility anchors from contextual
> building demand anchors and explicitly audits which raw sewer lines are
> excluded by the two-anchor representation assumption.

This is cleaner than saying "use buildings as anchor points" without
qualification. Building centroids are anchors in the graph-theoretic sense, but
not utility anchors in the asset-inventory sense.
