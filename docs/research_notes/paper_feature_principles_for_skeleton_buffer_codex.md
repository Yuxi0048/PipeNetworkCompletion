# Paper Feature Principles for Skeleton-Buffer Candidate Features

Workstream: Codex

Date: 2026-05-24

Status: design note, no model training in this file

## Purpose

This note checks the feature design used by the ISARC-style anchor-based paper
implementation and translates the useful principles into the newer
anchor-free skeleton-buffer setting.

The goal is not to copy manhole features into the anchor-free model. The goal is
to preserve the same feature-design logic:

- spatial position;
- categorical infrastructure type;
- binned geometry;
- relative spatial relation between feature supports;
- graph connectivity context.

Ground-truth sewer lines remain labels/evaluation only.

## What the Paper-Style Pipeline Actually Uses

The maintained paper-style preprocessing path is:

- `process.py`
- `scripts/build_graphs.py`
- `pipe_network_completion/dataset.py`
- `pipe_network_completion/model.py`

It builds a heterogeneous graph with node types:

- `MH`: manhole / pump-style anchor nodes;
- `Road`: road context nodes.

It builds edge types:

- `("MH", "link", "MH")`: sewer-link labels and supervised prediction target;
- `("MH", "near", "Road")`: manhole-to-nearest-road context edges;
- `("Road", "link", "Road")`: road-road adjacency;
- reverse edges added by `ToUndirected()`.

### Manhole Node Features

After splitting, the feature block used by `dataset.py` is:

- `x_coordinate`;
- `y_coordinate`;
- one-hot `SUBTYPECD_*`.

In `process.py`, `SUBTYPECD` is built as:

```text
SUBTYPECD + MANHOLEUSE
```

So the manhole node features encode:

- anchor location;
- anchor asset subtype/use.

This is useful for the original anchor-based model, but it is forbidden for the
anchor-free model.

### Road Node Features

The road feature block used by `dataset.py` is:

- `x_coordinate`;
- `y_coordinate`;
- one-hot `OVL2_CAT_*`;
- one-hot `angle_bin_*`;
- one-hot `len_bins_*`.

These are produced in `process.py` from:

- road centroid coordinates;
- road class/category `OVL2_CAT`;
- road segment bearing discretized into 6 angle bins;
- road length discretized into 5 quantile bins.

Although raw road columns also include `OVL_CAT`, `ROUTE_TYPE`, `Shape_Leng`,
and `angle`, the maintained graph feature slice only passes `OVL2_CAT`,
`angle_bin`, `len_bins`, and coordinates into the graph.

### Manhole-Road Edge Features

The `("MH", "near", "Road")` edge attributes are:

- `NEAR_POS`;
- `SIDE`;
- one-hot `dist_bins_*`.

`dist_bins` is a 5-bin quantile discretization of `NEAR_DIST`; the raw distance
is then dropped.

This is an important design principle: the paper-style model did not only use
node attributes. It also encoded the relative spatial relationship between an
anchor and a road.

### Features Not Used as Model Inputs

The sewer line dataframe has attributes such as:

- `SEGMENTTYP`;
- `MATERIAL`;
- `DIAMETER`;
- `USIL`;
- `DSIL`;
- `Shape_Leng`.

However, the maintained `dataset.py` does not pass these line attributes as
input features. The sewer lines are used to construct `MH1_index`/`MH2_index`
supervised edges.

For anchor-free work, this is the correct principle: truth geometry and truth
attributes should stay in labels/evaluation, not input features.

## Feature-Design Principles to Preserve

### 1. Keep Geometry, But Encode It Robustly

The paper uses coordinates, angle bins, and length bins. For skeleton-buffer
candidate segments, use:

- continuous `length_m` and `log1p_length_m`;
- 5-bin `length_bin_*` from train-split quantiles;
- continuous `bearing_sin` and `bearing_cos`;
- 6-bin `bearing_bin_*` matching the paper's road-angle logic;
- endpoint degree and degree-bin features;
- dead-end flags for candidate graph endpoints.

The bin thresholds must be fit on the training split only and then reused for
validation/test.

### 2. Preserve Categorical Surface Context

The paper uses road category one-hot features. For skeleton-buffer prediction,
keep and expand only non-utility surface categories:

- `OVL2_CAT_*`;
- `OVL_CAT_*`;
- `ROUTE_TYPE_*`;
- `candidate_source_*`, such as road, drainage, or region-skeleton;
- optional OSM/highway class if present.

These are safe because they come from road/surface context, not utility truth.

### 3. Replace Manhole-Road Relations With Candidate-Surface Relations

The anchor-based model has `NEAR_POS`, `SIDE`, and distance bins between
manholes and roads. The anchor-free analogue should be between each candidate
skeleton segment and allowed surface supports:

- nearest road distance;
- nearest road distance bin;
- projected position along nearest road, normalized from 0 to 1;
- side of candidate relative to nearest road direction;
- angle difference between candidate and nearest road;
- nearest drainage/watercourse distance and distance bin;
- whether the candidate lies inside road, drainage, built-up, or open-space
  support buffers.

This is the closest anchor-free equivalent of the paper's relation features.
It keeps the relational principle but removes the forbidden manhole endpoint.

### 4. Use Buildings as Demand Context, Not Utility Anchors

The current OSM building point layer contains useful non-utility columns:

- `building_type_group`;
- one-hot `bt_group_*`;
- `building_type_raw`;
- `footprint_area_m2`;
- `footprint_perimeter_m`;
- surface-visible metadata.

For skeleton-buffer features, aggregate these inside candidate buffers:

- building point count and density;
- nearest building point distance and distance bin;
- building footprint count;
- building footprint area fraction;
- residential/commercial/industrial/institutional building counts;
- weighted demand proxy, for example residential count plus larger weight for
  commercial/institutional buildings;
- left/right building balance relative to the candidate segment.

Do not treat buildings as sewer anchors in the anchor-free model. They are
demand/context support.

### 5. Add Local Graph Context

The paper relies on message passing over the manhole-road-road heterogeneous
graph. The skeleton-buffer version should expose local graph structure directly
and let the GNN smooth over it:

- endpoint degrees;
- local skeleton density in 50 m / 100 m buffers;
- candidate graph component size;
- distance in graph hops to nearest high-demand segment;
- short dead-end spur flag;
- local branch count around intersections.

These are candidate-graph features only; they do not require utility anchors.

### 6. Treat Coordinates Carefully

The original model uses a learned location encoder on the first two node
feature columns. That worked for the paper-style within-dataset setting, but it
can memorize Brisbane-specific spatial position under random or weak spatial
splits.

For the anchor-free skeleton-buffer model, use coordinates in controlled
ablations only:

- main result: no absolute coordinate features;
- optional ablation: AOI-normalized segment midpoint coordinates;
- optional ablation: learned location encoder using relative coordinates within
  each AOI;
- report spatial-block results separately if coordinates are included.

This keeps the paper principle available while avoiding an overclaim.

## Recommended Skeleton-Buffer Feature Set

### Core Paper-Analog Features

Use these first because they are closest to the paper:

- `length_m`;
- `log1p_length_m`;
- `length_bin_0..4`;
- `bearing_sin`;
- `bearing_cos`;
- `bearing_bin_0..5`;
- `candidate_source_*`;
- `OVL2_CAT_*`;
- `OVL_CAT_*`;
- `ROUTE_TYPE_*`;
- `nearest_road_distance_m`;
- `nearest_road_distance_bin_0..4`;
- `nearest_road_pos_0_1`;
- `nearest_road_side_left`;
- `nearest_road_side_right`;
- `nearest_road_side_on`;
- `road_candidate_angle_diff_sin`;
- `road_candidate_angle_diff_cos`;
- endpoint degree features;
- local skeleton density and density bins.

### Building/Demand Context Features

Add these as the next block:

- `building_point_count_25m`, `50m`, `100m`;
- `building_point_density_25m`, `50m`, `100m`;
- `nearest_building_point_m`;
- `nearest_building_point_bin_0..4`;
- `building_polygon_count_50m`;
- `building_polygon_area_fraction_50m`;
- `built_up_area_fraction_50m`;
- `bt_group_residential_count_50m`;
- `bt_group_commercial_count_50m`;
- `bt_group_industrial_count_50m`;
- `bt_group_institutional_count_50m`;
- `bt_group_unknown_count_50m`;
- `building_left_right_balance_50m`.

### Optional Hydrology / Terrain Features

Use only when coverage is complete for all AOIs being compared:

- elevation at segment midpoint;
- endpoint elevation difference;
- along-segment slope;
- nearest watercourse/drainage distance;
- drainage distance bin;
- flow-aligned/downhill flag.

For Brisbane sewer, elevation should be a weak feature because pressure mains
and pumped systems can violate a simple downhill assumption.

## Current Skeleton-Buffer Run Compared With This Design

The completed run in:

```text
outputs/skeleton_buffer_gnn_osm_bpoints_all_mains_sage/
```

already includes:

- continuous length and bearing;
- local skeleton density;
- road category one-hots;
- building point count/density/proximity;
- building footprint count/proximity/area fraction;
- built-up area fraction.

It does not yet include the most paper-like parts:

- length quantile bins;
- bearing angle bins;
- nearest-road relation features analogous to `NEAR_POS`, `SIDE`,
  and `dist_bins`;
- building type group counts;
- endpoint degree/dead-end features;
- controlled optional coordinate/location-encoder ablation.

Those are the next feature upgrades I would implement before retraining.

## Heterograph Upgrade Implemented

After the feature-principle review, the anchor-free graph was extended from a
single skeleton-edge feature table to a three-node-type heterograph:

- `SkeletonSegment`: prediction support;
- `Building`: OSM building-derived demand/context nodes;
- `RoadSegment`: road context nodes.

The implemented relation families are:

- `SkeletonSegment -> SkeletonSegment`: candidate-skeleton continuity;
- `Building -> Building`: local building-cluster context;
- `Building -> SkeletonSegment`: demand/context support to candidate support;
- `Building -> RoadSegment`: demand/context support to road support;
- `SkeletonSegment -> RoadSegment`: candidate support to road support;
- `RoadSegment -> RoadSegment`: road-network continuity.

This is the closest anchor-free analogue of the paper's `MH`/`Road`
heterogeneous graph. It preserves the idea of cross-support relational message
passing, but replaces forbidden manhole-road edges with allowed
building-road-skeleton edges.

Implemented files:

- `pipe_network_completion/anchor_free/skeleton_context_graph.py`;
- `scripts/train_skeleton_context_hetero_gnn.py`;
- `tests/test_skeleton_context_graph.py`.

Full first run:

```text
outputs/skeleton_context_hetero_gnn_osm_bpoints_all_mains_sage/
```

Configuration:

- 115 non-overlapping AOIs;
- complete sewer mains: gravity mains plus pressure mains;
- OSM building points;
- no absolute coordinate features;
- GraphSAGE;
- 100 epochs;
- CUDA device: NVIDIA GeForce GTX 1080 Ti.

Graph scale:

```text
SkeletonSegment nodes: 26,791
Building nodes:        73,941
RoadSegment nodes:     26,825

Skeleton-Skeleton edges:  89,544
Building-Building edges: 330,526
Building-Skeleton edges: 178,230
Building-Road edges:     178,232
Skeleton-Road edges:      78,983
Road-Road edges:          89,654
```

First-run test metrics:

```text
ROC AUC:      0.6328
PR AUC:       0.9073
F1:           0.9185
Precision:    0.8992
Recall:       0.9388
Length F1:    0.9301
Prevalence:   0.8818
All-pos F1:   0.9372
```

Interpretation:

The heterograph is structurally correct and more faithful to the original
paper's heterogeneous message-passing principle, but the first full result does
not improve over the simpler skeleton-buffer run. Train AUC is high (`0.9586`)
while validation/test AUC are much lower (`0.5882` / `0.6328`), so the richer
building context likely adds overfitting or split-specific signals. It should
be treated as a design-valid but not-yet-performance-superior variant.

Next credible refinements:

- remove or group high-cardinality raw building-type one-hots;
- use building-type group counts/embeddings rather than hundreds of sparse raw
  building columns;
- add explicit relation-distance bins as edge attributes or relation-specific
  node aggregates;
- compare against the simpler skeleton-buffer model using the same support,
  threshold grid, and AOI split;
- tune relation radii (`building_skeleton_radius_m`, `building_road_radius_m`,
  `building_building_radius_m`) on validation rather than fixing them upfront.

## Paper-Analog Single-Support Upgrade Implemented

The lighter follow-up was implemented in the existing single-support
`train_skeleton_buffer_gnn.py` path behind the optional flag:

```text
--paper-analog-features
```

This keeps the older skeleton-buffer behavior unchanged unless the flag is
passed.

Implemented paper-analog feature additions:

- road-angle style `bearing_bin_0..5`;
- train-split quantile `length_bin_0..4`;
- same-source candidate graph degree/dead-end/isolation features;
- nearest-road distance, fixed distance bins, normalized projected position,
  side-of-road indicators, and candidate-road angle difference;
- nearest-building point fixed distance bins;
- grouped OSM building context counts/densities, such as residential,
  commercial, industrial, institutional, utility, ancillary, other, and unknown.

The run in:

```text
outputs/skeleton_buffer_paper_analog_osm_bpoints_all_mains_sage/
```

used:

- 115 non-overlapping AOIs;
- complete sewer mains: gravity mains plus pressure mains;
- OSM building points and footprints;
- road/drainage/built-up context;
- no absolute coordinate features;
- GraphSAGE;
- 100 epochs;
- CUDA device: NVIDIA GeForce GTX 1080 Ti.

Run scale:

```text
Skeleton candidate edges: 26,791
Message-passing edges:    89,544
Feature columns:          81
Best validation threshold: 0.2
Runtime:                  301.9 s
```

Test metrics:

```text
ROC AUC:                 0.6591
PR AUC:                  0.9189
Precision:               0.9034
Recall:                  0.9240
F1:                      0.9136
Balanced accuracy:       0.5932
Brier score:             0.1832
Positive prevalence:     0.8818
All-positive F1:         0.9372
Selected truth recall:   0.5836
```

Compared with the earlier simpler skeleton-buffer GraphSAGE run
(`outputs/skeleton_buffer_gnn_osm_bpoints_all_mains_sage/`), the paper-analog
feature set slightly improves precision and balanced accuracy, but reduces ROC
AUC, PR AUC, F1, recall, and selected-buffer truth recall. It also does not
beat the all-positive F1 baseline because the support labels are highly
positive.

Compared with the heterograph run
(`outputs/skeleton_context_hetero_gnn_osm_bpoints_all_mains_sage/`), the
paper-analog single-support model is better on ROC AUC, PR AUC, precision, and
balanced accuracy, but worse on raw F1/recall.

Interpretation:

The paper-analog features are methodologically cleaner than the first rich
heterograph, but they do not solve the main bottleneck. The limiting issue is
still candidate support recall and label prevalence, not lack of classifier
features. The next research-rewarding step is to improve the region/skeleton
candidate support so it covers more true sewer-main length while occupying less
area, then rerun the same classifier family on that support.

## Research-Framing Recommendation

For a credible paper story, present this as:

```text
The original ISARC model used anchor coordinates, anchor type, road type,
geometric bins, and anchor-road relation features. The anchor-free variant
removes anchor nodes and replaces anchor-road relations with candidate-surface
and candidate-building relations over a skeleton-buffer support.
```

This is stronger than simply saying "we add more road/building features." It
shows continuity with the original method while making the change-of-support
problem explicit.
