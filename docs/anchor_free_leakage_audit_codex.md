# Anchor-Free Leakage and Research-Integrity Audit

Workstream: Codex

Date: 2026-05-23

This note reviews the current anchor-free implementation for leakage, invalid
evaluation shortcuts, and research-reporting risks. It is not a legal review;
it is a research-methodology audit for the ISARC-style anchor-free topology
prediction experiment.

## Framing context

Claude's `docs/research_notes/inductive_split_plan.md` records the current
project framing as within-network completion, not cross-city or new-network
generalization. Under that framing, random edge splits and full-road-graph GNN
message passing are transductive evaluation choices that can be acceptable if
they are explicitly disclosed.

The stricter spatial-block and component-holdout recommendations below are
therefore not mandatory blockers for within-network completion claims. They are
required only for stronger claims about spatial generalization to unseen
neighborhoods, disconnected subsystems, or new project areas.

## Short conclusion

I did not find direct forbidden utility-anchor features in the current
train-ready anchor-free feature tables. The current prepared training tables do
not include `overlap_length`, `overlap_ratio`, utility truth geometry, manhole,
valve, pole, or known utility-node fields as model inputs.

However, several issues can still make results misleading if reported as
generalization performance:

1. random edge splits create spatial autocorrelation leakage;
2. the GNN is transductive over the full road graph;
3. some scripts print full-graph metrics where a reader may assume test
   metrics;
4. ablation names can be wrong when using a full-context config;
5. positive-label prevalence can make F1 look stronger than it is;
6. absolute road-node coordinates can act as a spatial shortcut;
7. anchor-based and anchor-free metrics are not directly comparable unless the
   same data, split, and target definition are enforced.

These are fixable, but they should be fixed or clearly disclosed before using
the results in a paper.

## What appears safe

### No direct anchor feature in current train-ready tables

Checked generated files:

- `outputs/anchor_free_real_context_features/features/feature_columns.json`
- `outputs/anchor_free_real_context_features/features/road_edge_training_table_10m.csv`
- `outputs/anchor_free_real_context_features/features/road_edge_training_table_5m.csv`

The feature guard passes, and the training-table columns are:

```text
edge_id, split, y, <74 model feature columns>
```

No `overlap_length`, `overlap_ratio`, truth, utility-node, manhole, valve, or
anchor columns are present in the model feature set.

The relevant guard is in:

- `pipe_network_completion/anchor_free/features.py`
  - `assert_no_anchor_features`
  - `build_road_edge_features`
  - `standardize_features`

### Utility truth is used for labels/evaluation, not features

The label code creates:

```text
edge_id, y, overlap_length, overlap_ratio
```

from road-edge buffers and utility truth lines. This is appropriate for
supervised labels as long as those diagnostic columns are not fed to the model.

The train-ready export currently merges only `edge_id` and `y` into the model
table. The overlap diagnostics remain in separate label CSVs.

### Scaling uses train split only

`standardize_features(features, train_index=...)` computes mean/std from the
training split. That avoids classic scaling leakage from validation/test rows.

## High-risk findings

### H1. Random edge splits are optimistic for spatial utility networks

Current split function:

- `pipe_network_completion/anchor_free/baseline.py`
  - `make_stratified_edge_splits`

It stratifies individual road edges randomly. In utility-network mapping, nearby
road edges share road class, DEM, buildings, and the same true pipe corridor.
Randomly placing neighboring edges into train and test makes test performance
look better than performance on a new neighborhood or unseen project area.

This is not illegal if disclosed as a random-edge transductive experiment, but
it is not a strong spatial generalization result.

Required fix before spatial-generalization claims:

- add spatial block split by grid, suburb, catchment, or connected road region;
- report random-edge results as a secondary, easier setting;
- use spatial-block test metrics for the main paper claim.

### H2. GNN message passing uses the full road graph, including test edges

Current PyG conversion:

- `pipe_network_completion/anchor_free/model.py`
  - `build_pyg_road_edge_data`

The GNN message-passing graph contains all road edges:

```text
edge_index = all road candidate edges, both directions
edge_label_index = all candidate edges for scoring
```

Training loss is restricted to `train_index`, but node embeddings are computed
on the full road network topology, including validation/test roads. This is a
transductive setup.

This does not leak utility labels directly. The road graph is an allowed input.
But it must not be reported as inductive performance on unseen areas.

Required fix before inductive or unseen-area GNN claims:

- either disclose "transductive road-graph split";
- or create spatial subgraphs where test-area roads are not used during
  training message passing.

### H3. Some scripts print full-graph metrics as if they are final performance

`scripts/train_anchor_free.py` prints:

```text
roc_auc, pr_auc, f1, length_precision, length_recall, length_f1
```

These are full-graph metrics from `metrics.values`, not `test_*` metrics.

`scripts/train_anchor_free_brisbane.py` prints "Test-set metrics:" but then
prints non-prefixed keys such as `roc_auc`, `pr_auc`, and `f1`. Those are also
full-graph metrics, not held-out test metrics.

This is a serious reporting risk. The saved `metrics.json` does contain
`test_*` metrics, but the console output can mislead.

Required fix:

- change CLI summary to print `test_roc_auc`, `test_pr_auc`, `test_f1`,
  `test_precision`, and `test_recall`;
- keep full-graph metrics only under names like `all_edge_*`;
- never call non-prefixed metrics "test" in logs or tables.

Status:

- fixed in `scripts/train_anchor_free.py` and
  `scripts/train_anchor_free_brisbane.py` by printing held-out `test_*` edge
  metrics separately from all-edge decoded-network metrics.

### H4. RandomForest train metrics show memorization; do not report all-edge RF
metrics

For the updated-feature RandomForest, saved metrics show:

```text
train_roc_auc = 1.0
train_pr_auc  = 1.0
train_f1      = 1.0
```

The full-graph RF metrics are much higher than held-out metrics. This is normal
for a flexible RF, but reporting full-graph metrics would be invalid.

Use only held-out test metrics, preferably spatial-block test metrics.

### H5. F1 must be compared with prevalence baselines

At the 10 m label buffer, the positive rate is high:

```text
10 m positives: 27,052 / 41,463 = 65.24%
```

An all-positive classifier would therefore get:

```text
all_positive_f1 = 2 * prevalence / (1 + prevalence) ~= 0.789
```

That means a reported F1 near 0.79 is not necessarily useful by itself. It may
only match a trivial prevalence baseline.

Required fix:

- report positive prevalence for every evaluated split;
- report all-positive F1 and all-positive ROC AUC;
- prefer PR AUC, length precision/recall, and spatial-block metrics when
  interpreting model value.

Status:

- fixed in `pipe_network_completion/anchor_free/evaluation.py` for all edge
  metric calls, including train/validation/test split metrics.

## Medium-risk findings

### M1. Ablation runner can produce mislabeled variants

`scripts/run_anchor_free_ablation.py` changes only:

```text
config["graph"]["use_buildings"] = use_buildings
```

It does not also toggle:

```text
use_building_points
use_built_up
use_dem
```

If this runner is used with `configs/anchor_free_real_context_features.yaml`,
a variant named "road-only" may still include building points, built-up areas,
and DEM. That would make the ablation table invalid.

Required fix:

- make ablation variants explicitly control every context layer;
- write resolved feature names per variant;
- include `n_features` and context toggles in `ablation_results.csv`.

Status:

- fixed in `scripts/run_anchor_free_ablation.py` by using explicit context
  flags for `use_buildings`, `use_building_points`, `use_built_up`, and
  `use_dem`.

### M2. Anchor-based reference metrics are not directly comparable

`scripts/run_anchor_free_ablation.py` reads an original anchor-based metrics CSV
and places it in the same table as anchor-free variants.

This is only acceptable as a rough reference if clearly labeled. It is not a
fair ablation unless both pipelines use:

- same study area;
- same train/val/test spatial split;
- same target definition;
- same metric definitions;
- same held-out test reporting.

Otherwise, the comparison mixes different tasks: anchor-based manhole link
prediction versus road-edge corridor classification.

### M3. Edge predictions export includes label diagnostics

`pipeline._save_outputs` writes `edge_predictions.geojson` with:

```text
y, overlap_length, overlap_ratio, probability
```

This is useful for evaluation and QGIS inspection, but it is not an
inference-only prediction artifact. If shown as a "prediction map" without
disclosure, it includes ground-truth-derived columns.

Recommended fix:

- write two files:
  - `edge_predictions_for_evaluation.geojson` with labels and overlaps;
  - `edge_predictions_inference_only.geojson` with only road attributes and
    model probability.

### M4. Prediction map overlays ground truth

`pipeline._save_prediction_map` overlays `utility_truth` in black on the output
map. This is acceptable for evaluation figures, but not for a prediction-only
visual.

Recommended fix:

- name this `prediction_vs_truth_map.png`;
- optionally also write `prediction_only_map.png`.

### M5. 10 m labels are forgiving and should be reported as such

The current primary real-data label target uses:

```text
label_buffer_m = 10
label_overlap_threshold = 0.25
```

This produces a high positive rate:

```text
10 m positives: 27,052 / 41,463 = 65.24%
5 m positives:  11,379 / 41,463 = 27.44%
```

The 10 m target is a valid planning-grade corridor-label target, but it is not
the same as accurately locating a utility line. Report it as "road-edge
corridor presence within 10 m buffer," not precise utility alignment.

### M6. 10 m and 5 m train-ready splits differ

The train-ready feature export creates separate stratified splits for 10 m and
5 m labels. That is reasonable for label balance, but it weakens direct
comparison between 10 m-trained and 5 m-trained models because the test edge
sets are not necessarily identical.

Recommended fix:

- create one fixed split assignment independent of label buffer;
- reuse it for 10 m, 5 m, and future targets;
- preferably make that split spatial.

### M7. Absolute coordinates can encourage location memorization

The GNN node features include normalized road-node `x`, `y`, and degree. Road
coordinates are allowed anchor-free inputs, but with random edge splits they
can become a location memorization shortcut.

Recommended ablation:

- GNN without absolute `x/y`;
- GNN with local geometric features only;
- spatial-block split as the main safeguard.

### M8. Threshold 0.5 is an untuned operating point

The pipeline has `evaluation.threshold_grid`, but the current run path decodes
with `decoder.threshold`, usually 0.5. That does not leak labels, but it makes
F1, precision, and recall depend on an arbitrary operating point.

Recommended fix:

- tune threshold on validation only;
- freeze the threshold before test evaluation;
- save the validation threshold sweep.

## Lower-risk notes

### L1. Building context is not utility-anchor leakage

Building polygons, building points, built-up areas, and DEM are allowed inputs
under the current anchor-free definition. They are not utility anchors.

Potential temporal caveat:

- if claiming historical prediction, ensure building/context data predates the
  utility target date;
- for static 2024 planning-grade reconstruction, this is acceptable if
  disclosed.

### L2. `sources_sinks_path` is configured but not implemented

The configs include `sources_sinks_path`, but the current training path does
not use it. That is safer than accidental source/sink leakage, but the paper
should not claim a source/sink-aware decoder is implemented yet.

## Not found

I did not find evidence that the current anchor-free feature builder uses:

- sewer manhole coordinates;
- valve coordinates;
- utility-pole coordinates;
- transformer/cabinet coordinates;
- surveyed utility nodes;
- known pipe junctions;
- utility truth geometry as a feature;
- overlap length/ratio as a model input.

The original ISARC-style anchor-based pipeline still intentionally uses MH
nodes and MH-road near edges. That is expected for the original baseline, but
those files must not be described as anchor-free.

## Required fixes before paper-quality claims

1. Rename or separate full-graph metrics from held-out metrics.
2. Fix CLI logging to print `test_*` metrics by default.
3. Report prevalence and all-positive baselines next to F1.
4. Fix ablation toggles so "road-only", "road+building", "road+building+DEM",
   etc. are true to their names.
5. Add inference-only prediction exports with no ground-truth-derived columns.
6. Add a spatial block split before making spatial-generalization claims.
7. Add GNN no-`x/y` ablation before making claims that do not rely on absolute
   position.
8. If using 10 m labels, phrase the task as "road-corridor utility presence
   within a 10 m matching buffer."
9. Do not compare anchor-based GNN and anchor-free road-edge models as a fair
   ablation unless the same area, split, and target definition are enforced.

## Suggested paper-safe wording

Use:

> We evaluate an anchor-free, road-corridor prediction task where road edges are
> labeled positive if utility truth lines overlap a buffered road corridor. No
> surveyed utility-node coordinates are used as model inputs. Results are
> reported on held-out road edges, and spatial-block performance is reported
> separately to assess geographic generalization.

Avoid:

> The model predicts exact underground utility locations.

Avoid unless fixed:

> Test-set metrics

when the values are non-prefixed full-graph metrics.

Avoid:

> Road-only ablation

if the config still includes building points, built-up areas, or DEM.
