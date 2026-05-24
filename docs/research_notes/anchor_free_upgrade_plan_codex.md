# Anchor-Free Split and Decoder Upgrade Plan

Workstream: Codex

Related workstreams: Claude inductive split note, Codex leakage audit, Codex
engineering decoder note

Date: 2026-05-23

Status: proposal, with follow-up hygiene fixes started in code

## Purpose

This note consolidates the current Codex and Claude notes into an upgrade plan
for the anchor-free road-candidate utility prediction experiment.

The goal is not to rewrite the original ISARC-style anchor-based code. The goal
is to make the anchor-free experiment more defensible by improving:

1. data splitting and reporting;
2. post-model network decoding;
3. ablation naming and leakage controls.

The research question remains:

```text
Can utility corridor topology be predicted from roads and non-anchor context
without using dense ground utility anchor points?
```

## Current shared understanding

### What Codex found

The current anchor-free training tables do not appear to contain direct
forbidden utility-anchor features. In the prepared feature tables, the model
inputs are road, building, built-up, and DEM-derived columns. Utility truth is
used for labels and evaluation.

However, the current evaluation can still be optimistic:

- random edge splits place nearby road edges in different folds;
- the GNN computes node embeddings over the full road graph;
- some console summaries can blur full-graph metrics and held-out metrics;
- some ablation names can be wrong if context layers remain enabled;
- the GNN currently uses standardized road-node `x, y` coordinates and degree
  as node features.

These are not the same as using manholes or valves as inputs, but they affect
what claims can be made.

### What Claude found

Claude reconstructed the original ISARC-style split:

- the anchor-based implementation splits connected components of the true
  manhole-to-manhole utility graph;
- each split becomes a separate graph;
- train, validation, and test utility components are disjoint;
- this is close to an inductive component-level evaluation for the original
  anchor-based task.

Claude's final decision note also says the intended scope here is
within-network completion, not "train on one city and predict another city."
Therefore, a per-edge split can remain as a baseline, but stricter splits should
be available for stronger claims or reviewer requests.

## Recommended split policy

Use a three-level split policy instead of a single default split. Each level
answers a different research question.

### Split A: fixed random edge split

Purpose:

- baseline continuity with the current implementation;
- quick model iteration;
- comparable to the current RandomForest and GNN results.

Rules:

- split road edges into train, validation, and test by edge ID;
- stratify by label when labels are available;
- save the split file once and reuse it;
- use the same split file for 10 m and 5 m label thresholds when possible;
- fit scalers only on the training split;
- tune thresholds and decoder hyperparameters only on validation.

Required naming:

```text
split.strategy = fixed_random_edge
evaluation_scope = within_study_area_random_edge
```

Reporting limitation:

This split is useful and can be the default for the within-network-completion
framing. It is still spatially optimistic, so it must be named clearly and
should be accompanied by prevalence baselines and, when possible, a spatial
holdout sensitivity result.

### Split B: spatial block split

Purpose:

- stronger spatial generalization inside the same study area;
- avoids putting adjacent road edges in train and test;
- avoids using utility truth components to define the split.

Rules:

- create grid tiles, suburbs, catchments, or another spatial blocking unit;
- assign whole blocks to train, validation, and test;
- assign each road edge by midpoint or dominant geometry overlap;
- keep validation and test blocks spatially separated where practical;
- reuse the same block assignment for all label buffers and model variants.

Recommended use:

```text
split.strategy = spatial_block
evaluation_scope = within_study_area_spatial_holdout
```

Why this should be added:

It is easier to defend than random edge splitting, and it does not use utility
truth to decide the split. It tests whether the model works in held-out
neighborhoods rather than on adjacent road segments. It should become the main
reported split only if the paper claims spatial generalization to unseen
neighborhoods.

### Split C: utility component split

Purpose:

- optional rigorous ablation inspired by the original ISARC split;
- tests whether the model can generalize across disconnected utility
  subsystems or truth components.

Rules:

- build a graph from utility truth lines only for split assignment;
- snap truth-line endpoints with a small tolerance;
- compute connected components;
- assign whole truth components to train, validation, and test;
- assign road edges to the split of the nearby truth component region;
- assign background/no-truth road edges by spatial block or nearest component
  region;
- never use truth component IDs, truth geometry, or truth endpoints as model
  features.

Required naming:

```text
split.strategy = utility_component
evaluation_scope = truth_component_holdout
```

Reporting limitation:

This split uses utility truth to design the evaluation folds. That is acceptable
for supervised evaluation if disclosed, but it is less clean than spatial block
splitting for an anchor-free story. It should be optional unless the paper needs
an ISARC-like component-level comparison.

## GNN message-passing policy

The current GNN is transductive over the full road graph: the loss is masked to
training edges, but message passing uses all road nodes and road edges.

Add an explicit configuration switch:

```yaml
model:
  type: gnn
  message_passing_scope: full_graph   # full_graph, train_subgraph, split_subgraph
  use_node_xy: true                   # true, false
```

Recommended modes:

- `full_graph`: current behavior; valid as a transductive road-graph baseline.
- `train_subgraph`: during training, message passing uses only train-split road
  edges; validation/test are evaluated separately.
- `split_subgraph`: build separate train, validation, and test subgraphs for a
  stricter inductive-style evaluation.

Coordinate handling:

- keep `use_node_xy: true` as one variant, because coordinates can be useful for
  within-study-area completion;
- add `use_node_xy: false` as a leakage-sensitive ablation;
- report coordinate-aware results as road-coordinate-aware, not as evidence of
  new-city generalization.

## Metric and reporting cleanup

Before adding new model complexity, fix the reporting surface:

1. Console output should print `test_*` metrics by default.
2. Full-graph metrics should be named `all_edge_*` if retained.
3. F1 should always be reported beside positive prevalence and all-positive
   baselines.
4. `edge_predictions.geojson` should be split into:
   - `edge_predictions_inference_only.geojson`
   - `edge_predictions_for_evaluation.geojson`
5. Prediction figures that include utility truth should be named
   `prediction_vs_truth_map.png`.
6. Every ablation row should store:
   - resolved config;
   - split strategy;
   - label buffer;
   - model type;
   - decoder type;
   - context layer flags;
   - feature column list;
   - whether node coordinates were used.

## Post-network decoder design

The model predicts local edge probabilities:

```text
p_e = P(road edge e is a utility corridor | allowed non-anchor features)
```

The decoder should turn those edge probabilities into a coherent network. It
must remain anchor-free unless sparse public system facilities are explicitly
allowed by config.

### Decoder 0: validation-tuned threshold

Purpose:

- mandatory baseline;
- simple and easy to audit.

Algorithm:

1. Train the model.
2. Sweep thresholds on validation edges.
3. Select the threshold maximizing validation F1 or length F1.
4. Freeze the threshold.
5. Evaluate once on test.

Config:

```yaml
decoder:
  type: threshold
  threshold_selection: validation_grid
  threshold_grid: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
  selection_metric: val_f1
```

This should replace a hard-coded 0.5 threshold for reported results.

### Decoder 1: length-budget decoder

Purpose:

- prevent threshold predictions from selecting unrealistic total network
  length;
- provide a stronger non-graph baseline before adding engineering priors.

Algorithm:

1. Sort edges by probability or calibrated score.
2. Select edges until a validation-tuned length budget is reached.
3. Optionally require selected edges to exceed a minimum probability.

Allowed budget sources:

- validation positive length ratio;
- known planning-level network length target, if such a target is not derived
  from test truth;
- task-specific budget chosen before test evaluation.

Config:

```yaml
decoder:
  type: length_budget
  budget_mode: validation_positive_length_ratio
  min_probability: 0.2
```

### Decoder 2: demand-connected decoder

Purpose:

- turn isolated high-probability segments into a connected utility layout;
- use building points or building footprints as demand proxies without using
  utility anchors.

Inputs:

- road graph;
- predicted edge probabilities;
- building point or footprint demand weights;
- optional built-up context;
- no manholes, valves, poles, cabinets, transformers, or known pipe nodes.

Algorithm:

1. Cluster demand from building points or footprint centroids using a configured
   method such as fixed grid, DBSCAN, or top-k demand cells.
2. Map each demand cluster to the nearest road node.
3. Choose terminal clusters from validation-tuned demand thresholds.
4. Compute edge costs:

```text
cost(e) =
    -log(p_e + eps)
    + lambda_length * length_m
    - lambda_demand * demand_reward(e)
```

5. Connect terminals using repeated shortest paths, a minimum spanning tree on
   terminal-to-terminal path costs, or a Steiner-like approximation.
6. Optionally add high-probability nearby edges after the connected backbone is
   built.

Config:

```yaml
decoder:
  type: demand_connected
  terminal_source: building_points
  demand_cluster_method: grid   # grid, dbscan, top_k_cells
  demand_grid_m: 100
  terminal_min_demand: 1
  max_terminals: 500
  lambda_length: 0.001
  lambda_demand: 0.1
  add_local_probability_edges: true
```

### Decoder 3: sewer-directed pseudo-outlet decoder

Purpose:

- add sewer-specific engineering priors without using real surveyed outlets.

Anchor-free outlet rule:

If no sparse public outlet/system facility layer is available, do not invent a
known outlet. Derive pseudo-outlet candidates from allowed data:

```text
pseudo_outlet_candidates =
    road nodes with low DEM elevation
    near the study-area boundary
```

Example, with parameters tuned on validation:

```text
elevation <= 5th percentile of road-node elevation
and distance_to_study_boundary <= 200 m
```

Algorithm:

1. Sample DEM at road nodes.
2. Select low-elevation boundary pseudo-outlets.
3. Cluster building demand and map clusters to road nodes.
4. Connect demand terminals toward pseudo-outlets using directed or
   direction-penalized shortest paths.
5. Penalize uphill movement:

```text
cost(e, direction) =
    -log(p_e + eps)
    + lambda_length * length_m
    + lambda_uphill * max(elev_to - elev_from, 0)
    - lambda_demand * demand_reward(e)
```

6. Prune low-demand leaves.
7. Prefer tree-like topology unless validation supports loops.

Config:

```yaml
decoder:
  type: sewer_pseudo_outlet
  pseudo_outlet_percentile: 5
  pseudo_outlet_boundary_distance_m: 200
  pseudo_outlet_max_candidates: 100
  lambda_length: 0.001
  lambda_uphill: 1.0
  lambda_demand: 0.1
  prune_leaf_demand_threshold: 1
  loop_budget_fraction: 0.0
```

Limitations:

Brisbane is likely to include pumped or flat-terrain sewer behavior, so
"downhill toward a pseudo-outlet" is only a weak engineering prior for this
dataset. This can fail in pumped sewer systems, inverted siphons, pressure
mains, and areas where the true system crosses topographic ridges. It is a
planning-grade research prior, not a utility locating method.

### Decoder 4: water-style loop-aware decoder

Purpose:

- optional later variant for water distribution, where loops are plausible.

Algorithm:

1. Build a demand-connected backbone.
2. Add high-probability loop-closing edges up to a validation-tuned loop budget.
3. Penalize excessive length and disconnected components.

Config:

```yaml
decoder:
  type: water_loop_aware
  loop_budget_fraction: 0.05
  lambda_length: 0.001
```

This should not be used for sewer unless the task explicitly supports looped or
pumped network behavior.

## Hyperparameter tuning protocol

All decoder choices are model-selection choices. They must be tuned on
validation only.

Rules:

1. Split first.
2. Fit model on train.
3. Tune threshold and decoder hyperparameters on validation.
4. Freeze all choices.
5. Evaluate test once.
6. Do not pick the decoder based on test F1.

Store tuning results in:

```text
outputs/<experiment_name>/anchor_free/decoder_validation_sweep.csv
outputs/<experiment_name>/anchor_free/decoder_resolved_config.yaml
```

## Decoder acceptance gates

Do not pursue a more complex decoder indefinitely. A decoder should be kept only
if it beats the validation-tuned threshold baseline on at least one stated
network-level objective without unacceptable degradation in edge-level quality.

Suggested gates:

- length F1 improves by at least 0.03 on validation and does not fall on test;
- connected component count drops materially without a large false-positive
  length increase;
- for sewer mode, uphill length fraction decreases or outlet connectivity
  improves;
- the decoder remains stable across the 10 m and 5 m label evaluations.

Decoder-specific diagnostic metrics:

- tree excess: `n_edges - n_nodes + n_components`;
- degree distribution, especially fraction of degree-1, degree-2, and degree-3+
  nodes;
- average path length to nearest pseudo-outlet for sewer mode;
- branch count and branch length distribution;
- pseudo-outlet sanity check using truth geometry only for evaluation, such as
  distance from pseudo-outlet candidates to truth-network leaf endpoints.

## Recommended implementation order

### Phase 1: reporting and split hygiene

Implement first because it reduces research risk before adding algorithms.

- Fix CLI summaries to print held-out `test_*` metrics.
- Write inference-only and evaluation GeoJSON outputs separately.
- Add fixed split IDs independent of 5 m and 10 m labels.
- Make ablation rows record resolved context flags and feature names.
- Add `use_node_xy` to the GNN data builder.

### Phase 2: spatial block split

Implement next as the stronger spatial-generalization sensitivity split.

- Add `split.strategy: spatial_block`.
- Save `road_edge_split_spatial_block.csv`.
- Ensure 10 m and 5 m labels use the same split.
- Add tests that no road edge appears in more than one split.
- Add tests that scaler statistics are fit on train only.

### Phase 3: threshold and length-budget decoders

Implement before connected decoders.

- Add validation threshold sweep.
- Add length-budget selection.
- Compare both against hard 0.5 threshold.

### Phase 4: demand-connected decoder

Implement after the simpler decoders are stable.

- Build building-demand terminals from building points or footprints.
- Add shortest-path/MST connection over road graph costs.
- Export decoded network and terminal diagnostics.

### Phase 5: sewer pseudo-outlet decoder

Implement after demand-connected decoding.

- Derive pseudo-outlets from DEM and boundary.
- Add uphill penalties.
- Add pruning and optional tree preference.
- Report uphill length fraction and outlet connectivity.

### Phase 6: optional utility component split

Implement only if needed for ISARC-style comparison or reviewer concerns.

- Build truth components for split assignment only.
- Do not feed truth component information to the model.
- Clearly disclose that utility truth was used to create evaluation folds.

## Metrics to report

Keep edge-level and network-level metrics separate.

Edge-level:

- ROC AUC;
- PR AUC / average precision;
- positive prevalence;
- all-positive ROC AUC and F1;
- majority-class accuracy and F1;
- precision;
- recall;
- F1;
- balanced accuracy;
- IoU / Jaccard;
- Brier score.

Length/network-level:

- predicted total length;
- true total length;
- true-positive predicted length;
- false-positive length;
- false-negative length;
- length precision;
- length recall;
- length F1;
- connected component count;
- cyclomatic number or loop density;
- tree excess;
- node degree distribution;
- served building count or demand coverage;
- uphill length fraction for sewer decoder;
- pseudo-outlet connectivity for sewer decoder.

## Recommended ablation table

Use the same split and label buffer within each table.

Minimum rows:

1. Road-only RandomForest.
2. Road plus building RandomForest.
3. Road plus building plus DEM RandomForest.
4. Road-only GNN without node coordinates.
5. Road plus building GNN without node coordinates.
6. Road plus building GNN with node coordinates.
7. Best GNN plus threshold decoder.
8. Best GNN plus length-budget decoder.
9. Best GNN plus demand-connected decoder.
10. Best GNN plus sewer pseudo-outlet decoder.

Optional reference row:

- Original anchor-based ISARC-style GNN.

The original anchor-based row should be labeled as a reference, not a fair
one-to-one ablation, unless it uses the same study area, split, and metric
definitions.

## Research wording

Safe wording:

```text
The anchor-free model predicts road-edge utility corridor likelihoods using
road geometry and non-anchor context layers. Ground-truth utility lines are used
only to create supervised labels and evaluate predictions.
```

For the sewer decoder:

```text
In the absence of surveyed outlets, the sewer-directed decoder uses
terrain-derived pseudo-outlet candidates selected from low-elevation boundary
road nodes. These are engineering priors, not observed utility assets.
```

Avoid:

```text
The model locates underground utilities.
The decoder finds true sewer outlets.
The random edge split proves generalization to unseen neighborhoods.
```

## Bottom-line recommendation

Use this as the upgraded experimental structure:

1. Keep the fixed random edge split as the within-network-completion default.
2. Add spatial block split as the stronger spatial-generalization sensitivity
   result.
3. Add `use_node_xy` and `message_passing_scope` ablations for the GNN.
4. Report prevalence and all-positive baselines beside F1.
5. Implement decoders in this order:
   threshold tuning, length budget, demand-connected, sewer pseudo-outlet.
6. Treat the utility component split as an optional ISARC-style ablation, not
   the default anchor-free claim.

This combines Claude's useful ISARC component-split reasoning with the Codex
leakage audit and decoder design while keeping the anchor-free story clean.
