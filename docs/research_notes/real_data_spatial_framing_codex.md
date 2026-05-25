# Real-Data Spatial Framing for Anchor-Free Utility Prediction

Workstream: Codex

Date: 2026-05-23

Status: research framing note, based on current Brisbane real-data layers

## Purpose

This note summarizes what the current real-world data layers provide and how
their spatial relationships should shape the research claim. The goal is to
avoid overstating the task as exact utility-node topology recovery when dense
ground anchors are deliberately excluded.

## Available Real-World Layers

Current usable non-anchor context layers:

- roads: 41,753 road geometries, about 6,783 km total length;
- building points: 54,107 points;
- building areas: 5,469 polygons, about 16.44 km2 total footprint area;
- built-up areas: 2,245 polygons, about 862.2 km2 total area;
- homesteads: 18 points;
- DEM: Brisbane 1 arc-second elevation raster, reprojected to EPSG:28356.

Current utility truth and asset layers:

- gravity sewer mains: 243,773 line features, about 9,660 km total length;
- service laterals: 346,000 line features, about 897 km total length;
- sewer manholes, fittings, valves, devices, vents, pumps, and treatment
  facilities are present in the raw data.

For the pure anchor-free experiment, dense sewer asset point layers must remain
forbidden as model inputs. Gravity-main lines may be used for labels and
evaluation only.

## Actual Context Features Prepared So Far

The current hybrid candidate feature table contains 42 edge-level features:

- geometric features: length, bearing sin/cos, endpoint degrees;
- hybrid candidate features: candidate source, candidate weight, nearest-road
  distance, demand-endpoint indicators;
- road-context features: local road density and road class one-hot fields from
  `OVL2_CAT` / `OVL_CAT`;
- building-point features: nearest building point distance, building point
  count/density within 50 m, and counts by building function;
- building functions currently include house, education, shed, health/medical,
  sports facility, aged care, defence, commercial premises, place of worship,
  post office, electrical substation, and youth groups.

The prepared hybrid candidate graph has:

- road backbone: 41,461 candidates, about 6,736.6 km;
- demand kNN candidates: 12,765 candidates, about 2,903.8 km;
- building access candidates: 5,650 candidates, about 422.4 km;
- total: 59,876 prediction units.

## Sampled Spatial Relationships

The quick inspection used a length-weighted sample of 25,000 gravity-main
features and exact nearest-neighbor spatial joins. Treat these as approximate
distribution estimates, not survey-grade measurements.

Gravity-main distance to nearest road:

- within 5 m: 11.66%;
- within 10 m: 29.33%;
- within 20 m: 37.44%;
- within 30 m: 44.89%;
- within 50 m: 66.70%;
- median distance: 34.44 m;
- 75th percentile: 177.70 m.

Gravity-main distance to nearest building point:

- within 20 m: 1.26%;
- within 50 m: 5.73%;
- within 100 m: 16.23%;
- within 200 m: 40.85%;
- median distance: 238.61 m.

Building-point distance to nearest gravity main, sampled over 25,000 building
points:

- within 20 m: 9.11%;
- within 50 m: 20.27%;
- within 100 m: 29.98%;
- within 200 m: 37.34%;
- median distance: 774.81 m.

These numbers support the earlier candidate-recall finding: a pure road-edge
candidate graph cannot represent a large share of the gravity-main network at
strict 5 m or 10 m tolerances.

## Interpretation

Roads are a strong but incomplete proxy.

The road network clearly contains signal: roughly two-thirds of sampled
gravity-main length falls within 50 m of a road. However, only about 29% falls
within 10 m. That means a strict road-edge classification task is too narrow if
the intended research question is utility-network reconstruction.

Buildings are demand proxies, not alignment proxies.

The building point layer is useful for demand intensity and land-use context,
but it does not by itself locate sewer-main centerlines. The nearest-building
distances are often much larger than a service-connection distance because the
gravity-main layer includes trunk/collector infrastructure and parts of the
network away from individual buildings. Building features should therefore be
used as contextual intensity features, not as direct pipe-coordinate anchors.

Service laterals are utility truth, not allowed context.

Service laterals would be very informative for demand-to-main connectivity, but
they are utility geometry. They should not be used as model inputs for the
anchor-free claim. They can be reserved for separate evaluation or a clearly
marked non-anchor-free upper-bound experiment.

## Recommended Research Framing

The most credible primary framing is:

```text
Anchor-free probabilistic utility corridor reconstruction with buffered
centerline evaluation.
```

The method predicts candidate utility corridors from roads, buildings, DEM, and
other non-anchor context. The decoded output is a planning-grade network
skeleton with spatial uncertainty, not exact manhole-to-manhole topology.

Use topology as a secondary product:

```text
Candidate corridor probabilities are post-processed into a connected network
skeleton, and topology diagnostics are reported after spatial representability
has been established.
```

Avoid making exact-topology claims:

```text
The model reconstructs exact sewer topology.
```

Better wording:

```text
The model estimates probable sewer corridor centerlines and evaluates them
against observed utility geometry under multiple buffer tolerances.
```

## Geostatistical View

This is best treated as a spatial point/line-process inference problem with
support mismatch:

- observed covariates live on roads, buildings, DEM cells, and land-use
  polygons;
- the target is a hidden utility-line process;
- generated graph nodes are computational support points, not observed utility
  nodes;
- labels come from buffered overlap between candidate corridors and truth lines.

Useful geostatistical concepts:

- spatial support: road edges, generated corridors, and true pipes are different
  spatial supports;
- change of support: road/building features must be aggregated onto candidate
  corridor segments;
- spatial autocorrelation: adjacent corridor probabilities should be correlated,
  which motivates GNN/message passing or graph smoothing;
- anisotropy: linear infrastructure follows directional corridors, so bearing,
  road hierarchy, and corridor continuity matter;
- nonstationarity: urban core, suburban, industrial, and boundary areas likely
  have different pipe-road relationships;
- spatial uncertainty: predictions should be evaluated with multiple tolerance
  buffers, not a single exact overlay threshold.

## Recommended Metrics

Report representability before classifier performance:

- fraction of truth length within 5, 10, 20, 30, and 50 m of the candidate graph;
- candidate total length by source family;
- positive prevalence by buffer threshold;
- length precision, length recall, and length F1 after decoding.

Then report classification:

- ROC AUC;
- PR AUC;
- Brier score;
- F1 compared with all-positive and prevalence baselines;
- calibration or reliability curve if time allows.

Then report topology diagnostics:

- connected component count;
- largest-component length fraction;
- dead-end density;
- cyclomatic number / loop density;
- predicted total length.

## Recommended Experimental Claim Order

1. Candidate graph representability: can non-anchor context create a high-recall
   corridor support?
2. Edge/corridor scoring: can road, building, DEM, and topology features rank
   plausible corridors better than prevalence and RF baselines?
3. Decoding: can a simple connected-network decoder improve length recall and
   reduce fragmented outputs without overfitting?
4. Topology interpretation: are the decoded networks structurally plausible for
   sewer systems, while acknowledging that exact node topology is not
   identifiable without anchors?

## Bottom Line

For this dataset, the strongest paper framing is not exact topology prediction.
It is anchor-free utility-corridor reconstruction with buffered centerline
evaluation and topology-aware post-processing.

The core scientific question should be:

```text
How much of a real sewer network can be represented and ranked using only
non-anchor spatial context?
```

That framing is credible, defensible, and still research-rewarding.
