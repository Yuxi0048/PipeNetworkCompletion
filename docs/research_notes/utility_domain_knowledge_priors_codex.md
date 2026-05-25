# Domain Knowledge Priors for Anchor-Free Utility Line Prediction

Workstream: Codex

Date: 2026-05-23

Status: design note, no implementation in this file

## Purpose

This note lists domain knowledge that can be used to improve anchor-free utility
line prediction without using dense ground utility anchor points.

The intended use is:

- feature design;
- model ablations;
- decoder design;
- topology-quality metrics;
- paper discussion.

The key rule remains:

```text
Use roads, buildings, land use, DEM, parcels, and demand proxies.
Do not use manholes, valves, poles, cabinets, transformers, surveyed utility
nodes, known pipe junctions, or true utility geometry as model inputs.
```

Ground-truth utility lines may be used only for labels and evaluation.

## Utility-Type Differences

The most important design choice is the utility type. The same topology prior
is not valid for every network.

### Sewer

Typical priors:

- gravity networks tend to be dendritic or tree-like;
- flow usually trends downhill toward outlets, trunk sewers, or pumping points;
- dead ends are plausible at upstream service branches;
- large loops are uncommon in gravity sewer;
- long uphill paths are suspicious unless pumps, siphons, or pressure mains are
  present;
- dense demand areas need nearby collection paths.

Caveat:

Urban systems can include lift stations, pressure sewers, inverted siphons, and
flat coastal terrain. For Brisbane, DEM direction should be treated as a weak
prior, not a hard rule.

### Water Distribution

Typical priors:

- loops and circulation are common for redundancy and pressure balance;
- dead ends exist but too many dead ends can indicate poor service reliability;
- higher-order roads may carry trunk or distribution mains;
- local streets may carry smaller service distribution;
- networks should connect demand clusters to source/storage zones if those are
  available as sparse system facilities.

Caveat:

Without pipe diameter, pressure zones, reservoirs, and pumps, loop quality can
only be approximated.

### Stormwater

Typical priors:

- flow follows topography and road drainage structure;
- networks often collect from local streets into larger downstream channels;
- outlets to creeks/rivers/coast are plausible;
- dead ends are plausible at upstream inlets;
- loops are generally not expected.

Caveat:

This differs from sanitary sewer and should not share the same decoder without
renaming the task.

### Gas / Electric / Telecom

Typical priors:

- route selection often follows roads and easements;
- redundancy differs by system and voltage/pressure level;
- topology can be radial, looped, or meshed depending on asset class;
- above-ground facilities would be useful but are forbidden if they are dense
  utility anchors for this experiment.

Caveat:

Do not transfer sewer gravity assumptions to these networks.

## Topology Priors

### Connectivity

Most utility networks are not arbitrary independent edge selections. Predicted
edges should form meaningful connected components.

Possible uses:

- decoder penalty for isolated single-edge components;
- metric: connected component count;
- metric: fraction of predicted length in the largest component;
- metric: average component size by length.

Research value:

High. Connectivity is broadly defensible, but the target number of components
depends on utility type and study-area boundary.

### Dead Ends

Dead ends are degree-1 nodes in the decoded network.

Interpretation:

- sewer: dead ends are expected at upstream terminal branches;
- water: some dead ends are expected, but excessive dead ends may be
  undesirable;
- stormwater: upstream dead ends are normal;
- electric/telecom: dead-end interpretation depends on network level.

Possible uses:

- metric: dead-end count;
- metric: dead-end density per km;
- decoder pruning of short low-probability dead-end branches;
- decoder penalty for dead ends in water mode;
- decoder allowance for dead ends in sewer/stormwater mode.

Do not use:

- known utility terminal nodes;
- manhole/end-cap locations;
- true network leaf nodes as inputs.

Research value:

Medium-high, if reported by utility type. It is more credible as an evaluation
metric or decoder diagnostic than as a hard constraint.

### Loops And Circulation

Loops can be measured by the cyclomatic number:

```text
cyclomatic_number = n_edges - n_nodes + n_components
```

Interpretation:

- water: loops/circulation are often useful and expected;
- sewer: loops are usually suspicious unless there are special hydraulic
  structures;
- stormwater: loops are usually suspicious;
- road graph loops do not imply utility loops.

Possible uses:

- metric: loop density = cyclomatic_number / n_edges;
- water decoder: allow limited loop addition from high-probability edges;
- sewer decoder: penalize loops or prune low-value loop-closing edges;
- ablation: threshold decoder vs loop-aware decoder.

Research value:

High for water, medium for sewer. For the current sewer-oriented experiment,
loop metrics are useful but should not dominate the method.

### Branching Degree

Utility junctions tend to have constrained degree patterns.

Typical patterns:

- long chains of degree-2 nodes are normal along corridors;
- degree-3 junctions are plausible at branch connections;
- very high-degree junctions are rare and may indicate road-intersection
  artifacts rather than utility topology.

Possible uses:

- metric: degree distribution of decoded network;
- penalty for high-degree nodes above a utility-specific threshold;
- pruning of tiny branches connected to high-degree road nodes if probability
  is low.

Research value:

Medium. Useful for topology diagnostics, but risky as a hard decoder rule
because road intersections can legitimately host multiple utility branches.

### Continuity Along Corridors

Utilities are expensive to install and often follow continuous corridors rather
than alternating on/off road segments.

Possible uses:

- edge smoothing over adjacent road segments;
- graph regularization encouraging neighboring road edges to have similar
  probabilities;
- decoder preference for paths with continuous high probability;
- metric: mean run length of selected road edges.

Research value:

High. This is a strong reason to compare GNN/message-passing models against
edge-only classical models.

### Hierarchy

Many utility systems are hierarchical.

Examples:

- sewer: local branches connect to collectors and trunks;
- water: local distribution connects to larger mains;
- stormwater: local drains connect to channels/outfalls.

Allowed proxies:

- road class;
- road width if available;
- road centrality;
- demand density;
- land use;
- elevation/catchment position.

Possible uses:

- feature: road class;
- feature: betweenness/centrality on road graph;
- decoder cost reduction for likely trunk corridors;
- metric: selected length by road class.

Research value:

High, especially for explaining why roads alone may contain predictive signal.

## Spatial And Civil Engineering Priors

### Road-Constrained Corridors

The core anchor-free assumption is that candidate utility corridors are road
edges.

Strength:

- defensible in urban infrastructure planning;
- directly supports QGIS-inspectable outputs;
- avoids using utility anchors.

Limitation:

- utilities can cross parcels, easements, parks, rail, waterways, or private
  corridors;
- road-only candidate graphs can never predict off-road utility geometry.

Recommended paper wording:

```text
The task is road-corridor utility presence, not exact utility alignment.
```

### Demand Proximity

Utilities usually serve demand.

Allowed demand proxies:

- building points;
- building footprints;
- building floor area if available;
- land use;
- zoning;
- population density;
- parcel density.

Possible uses:

- feature: building count within buffer;
- feature: nearest building distance;
- feature: demand density along edge;
- decoder terminals from demand clusters;
- metric: building service coverage.

Research value:

Very high. This is one of the most credible non-anchor signals.

### Terrain And Slope

Terrain matters most for gravity systems.

Allowed inputs:

- DEM;
- road-node elevation;
- slope along road edges;
- catchment or flow accumulation if derived from DEM.

Possible uses:

- feature: elevation and slope;
- sewer decoder: weak uphill penalty;
- stormwater decoder: stronger downhill prior;
- pseudo-outlet candidates from low-elevation boundary road nodes.

Risk:

Medium. DEM can be misleading in pumped or flat urban systems.

### Boundary Effects

Study-area boundaries cut networks artificially.

Implications:

- components may appear disconnected only because the network leaves the study
  area;
- dead ends near the boundary may be false dead ends;
- pseudo-outlets near boundary are plausible but uncertain.

Possible handling:

- tag boundary-near road nodes and edges;
- report metrics separately for boundary and interior zones;
- avoid penalizing dead ends near the boundary too strongly.

Research value:

Medium. Important for fair evaluation, less important as a model feature.

### Crossings And Barriers

Utilities may be affected by:

- rivers;
- rail corridors;
- highways;
- bridges;
- tunnels;
- steep slopes;
- protected land.

Allowed inputs if available:

- waterway polygons/lines;
- rail/road hierarchy;
- land-use constraints;
- terrain.

Possible uses:

- decoder penalties for crossing barriers;
- features for bridge/tunnel road classes;
- analysis of false positives near barriers.

Research value:

Medium. Useful if reliable barrier layers exist; otherwise can become a
distraction.

## How To Encode Domain Knowledge

### As Input Features

Good feature candidates:

- edge length;
- bearing sin/cos;
- road-node degree;
- local road density;
- road class;
- road centrality;
- building count/density;
- nearest building distance;
- building function counts;
- built-up area coverage;
- elevation and slope;
- distance to study boundary;
- boundary-near indicator.

Avoid as features:

- distance to manholes;
- distance to valves;
- utility-node degree;
- known pipe endpoint proximity;
- overlap with truth lines;
- label overlap ratio;
- any dense utility facility inventory.

### As Model Regularization

Possible GNN/ML regularizers:

- smooth probabilities over adjacent road edges;
- penalize isolated high-probability edges;
- constrain extreme class imbalance using calibrated thresholds;
- compare with and without absolute coordinates.

Research value:

Medium. Useful, but secondary to clean ablations.

### As Decoder Costs

Possible decoder terms:

```text
cost(e) =
    -log(p_e + eps)
    + lambda_length * length_m
    - lambda_demand * demand_reward(e)
    + lambda_uphill * uphill_penalty(e)
    + lambda_barrier * barrier_penalty(e)
```

Utility-specific terms:

- sewer/stormwater: uphill penalty, low loop budget, dead-end tolerance;
- water: loop allowance, demand coverage, redundancy;
- generic utility: length penalty and connectivity preference.

Research caution:

Every decoder parameter must be tuned on validation only. Do not tune decoder
choices on test results.

### As Evaluation Metrics

Recommended topology metrics:

- connected component count;
- largest-component length fraction;
- dead-end count;
- dead-end density per km;
- cyclomatic number;
- loop density;
- degree distribution;
- predicted total length;
- length precision/recall/F1;
- building service coverage;
- uphill length fraction for sewer/stormwater;
- boundary dead-end fraction.

These metrics should supplement edge ROC AUC, PR AUC, F1, and prevalence
baselines.

## Priority Ranking For This Project

### High Reward

Use these first:

1. demand proximity from building points/footprints;
2. road class and road hierarchy;
3. continuity along road corridors;
4. connected component diagnostics;
5. dead-end and loop metrics by utility type;
6. GNN with and without node coordinates;
7. one spatial-block sensitivity run.

### Medium Reward

Use after the core ablations are stable:

1. DEM slope and elevation;
2. distance to study boundary;
3. boundary-aware dead-end interpretation;
4. road centrality;
5. simple length-budget decoder.

### Lower Reward Or Higher Risk

Defer unless needed:

1. pseudo-outlet sewer decoder;
2. complex water loop-aware decoder;
3. barrier-crossing penalties without high-quality barrier layers;
4. per-region thresholds;
5. heavy topology optimization.

## Recommended Near-Term Experiments

Use the same split and label buffer within each table.

### Feature Ablation

```text
road only
road + building points
road + building polygons
road + building points + polygons
road + building + DEM
```

### Model Ablation

```text
RandomForest
GNN without node coordinates
GNN with node coordinates
```

### Topology Evaluation

For each model output, compute:

```text
edge PR AUC
edge ROC AUC
F1 and all-positive F1
length precision/recall/F1
component count
dead-end density
loop density
predicted total length
building service coverage
```

### Simple Decoder Ablation

Only after the base probabilities are stable:

```text
threshold decoder
length-budget decoder
```

Do not start with pseudo-outlet or sewer-directed decoding.

## Paper-Safe Claims

Safe:

```text
Road geometry and non-anchor demand/context layers contain measurable signal for
road-corridor utility presence.
```

Safe:

```text
Topology diagnostics such as dead-end density and loop density help distinguish
locally accurate edge predictions from coherent network predictions.
```

Safe with caveat:

```text
DEM-derived slope provides a weak engineering prior for gravity systems, but it
is unreliable in pumped or flat urban sewer systems.
```

Avoid:

```text
The model locates underground utilities exactly.
```

Avoid:

```text
The sewer decoder discovers true outlets.
```

Avoid:

```text
Loop or dead-end counts prove hydraulic correctness.
```

## Bottom Line

The best domain knowledge for this anchor-free project is not a complicated
engineering decoder. It is a clean set of defensible priors:

- demand creates service need;
- roads provide plausible corridors;
- utility networks have connectivity structure;
- sewer and stormwater are usually more tree-like;
- water networks can support loops and circulation;
- dead ends mean different things by utility type;
- DEM is useful but weak in pumped urban systems.

These priors should first be used for ablation design and topology metrics.
Only after the base predictive signal is credible should they be used as hard
decoder constraints.
