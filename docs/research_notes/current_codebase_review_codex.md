# Current Codebase Review

Workstream: Codex

Date: 2026-05-23

Scope:

- original anchor-based code was not deeply modified in this pass;
- review focused on the current anchor-free implementation, scripts, configs,
  and tests;
- emphasis is on correctness, research validity, reproducibility, and avoiding
  misleading claims.

Test command run:

```text
.conda\pipe-network-completion\python.exe -m pytest tests\test_anchor_free_ablation.py tests\test_anchor_free_baseline.py tests\test_anchor_free_buffer_invariant_split.py tests\test_anchor_free_decoder.py tests\test_anchor_free_features.py tests\test_anchor_free_labels.py tests\test_anchor_free_metrics.py tests\test_anchor_free_model.py tests\test_anchor_free_pipeline.py tests\test_anchor_free_road_graph.py -q
```

Result:

```text
49 passed, 62 warnings
```

The first attempted wildcard command did not run because PowerShell did not
expand `tests\test_anchor_free_*.py`.

## Summary

The anchor-free modules are in a usable smoke-test state and I did not find
direct forbidden utility-anchor features being passed into the feature matrix.
The strongest parts are:

- explicit anchor-feature guard in `features.py`;
- separate road graph, feature, label, baseline, GNN, decoder, and evaluation
  modules;
- real-data configs for road-only and context-feature experiments;
- added prevalence baselines, GPU preflight reporting, and buffer-invariant
  feature-preparation split support;
- broad synthetic smoke tests.

The highest-risk issues are not syntax or crashes. They are research-validity
and reporting risks:

1. training still uses a label-dependent split while feature preparation uses a
   buffer-invariant split;
2. decoder names such as `sewer`, `water`, and `steiner` currently run the same
   lightweight connected decoder;
3. saved prediction artifacts still include ground-truth-derived columns;
4. the road graph may not actually be noded at road intersections if source
   centerlines are not already split;
5. the Brisbane convenience training script does not include building points,
   even though building points are one of the most useful new context layers.

## Findings

### P1. Training split is inconsistent with the feature-preparation split

Risk label: High research-validity and reproducibility risk

Files:

- `pipe_network_completion/anchor_free/pipeline.py:338`
- `scripts/prepare_anchor_free_features.py:275`
- `scripts/prepare_anchor_free_features.py:280`

`prepare_anchor_free_features.py` now derives train/validation/test splits once
from `edge_id` using `make_buffer_invariant_splits`, so 10 m and 5 m training
tables are comparable.

But `run_anchor_free_experiment()` still does this:

```text
make_stratified_edge_splits(labels.y, seed=seed)
```

That split depends on the label vector. If the primary config is changed from
10 m to 5 m, the train/test split changes too. This means model metrics from a
10 m primary run and a 5 m primary run are not directly apples-to-apples, even
though the prepared feature tables now are.

Recommendation:

- add a config field such as `split.strategy: stratified | buffer_invariant`;
- default training to `buffer_invariant` for current research tables;
- keep old stratified behavior only as an explicitly named legacy option;
- write the split CSV used by training into the experiment output directory.

### P1. `sewer`, `water`, and `steiner` decoder types are aliases, not real decoders

Risk label: High claim-validity risk

File:

- `pipe_network_completion/anchor_free/decoder.py:168`

The dispatcher treats these decoder types identically:

```text
connected, steiner, sewer, water
```

They all call `decode_connected()` and then relabel the result with the
requested decoder type.

This is risky because a run configured as `decoder.type: sewer` will produce
outputs labeled as sewer-decoded even though there is no sewer-specific DEM
slope, pseudo-outlet, tree constraint, or uphill penalty logic.

Recommendation:

- either raise `NotImplementedError` for `steiner`, `sewer`, and `water`;
- or return `decoder_type: connected` until those algorithms are actually
  implemented;
- only expose `sewer` and `water` in CLI choices after they have distinct
  behavior and tests.

### P1. Evaluation artifacts include ground-truth-derived columns

Risk label: High leakage-interpretation and reporting risk

File:

- `pipe_network_completion/anchor_free/pipeline.py:177`
- `pipe_network_completion/anchor_free/pipeline.py:181`
- `pipe_network_completion/anchor_free/pipeline.py:191`

`edge_predictions.geojson` merges:

```text
y, overlap_length, overlap_ratio, probability
```

The map renderer also overlays truth geometry. This is useful for evaluation,
but it is not an inference-only artifact. A QGIS user can easily mistake a
truth-assisted evaluation file for a pure prediction output.

Recommendation:

- write `edge_predictions_for_evaluation.geojson` with labels and overlap
  diagnostics;
- write `edge_predictions_inference_only.geojson` with road geometry and model
  probability only;
- rename `prediction_map.png` to `prediction_vs_truth_map.png`;
- optionally add `prediction_only_map.png`.

### P2. Road graph construction only uses source feature endpoints

Risk label: Medium-high topology-correctness risk

File:

- `pipe_network_completion/anchor_free/road_graph.py:196`
- `pipe_network_completion/anchor_free/road_graph.py:200`
- `pipe_network_completion/anchor_free/road_graph.py:201`

The road graph treats each input LineString/MultiLineString part as one edge
from its first coordinate to its last coordinate. It does not split lines at
interior vertices or geometric intersections.

If the road source is already segmented at intersections, this is fine. If not,
the graph topology is wrong:

- intersection degrees are undercounted;
- GNN message passing misses real road connectivity;
- connected decoding paths can be impossible or distorted;
- local road density and endpoint degree features become less meaningful.

Recommendation:

- add a preprocessing check that counts geometric crossings not represented as
  shared endpoints;
- if crossings are nontrivial, add optional noding/splitting at intersections;
- record `n_unmodeled_intersections` in graph metadata;
- add a synthetic test with crossing roads that must become connected after
  noding, if noding is enabled.

### P2. Multi-file utility truth loading in the training pipeline does not reproject per file

Risk label: Medium-high label-correctness risk

File:

- `pipe_network_completion/anchor_free/pipeline.py:103`
- `pipe_network_completion/anchor_free/pipeline.py:116`

`_read_vector_many()` concatenates multiple truth files and assigns the CRS of
the first frame to the combined GeoDataFrame. Unlike `_read_optional_vector_many`,
it does not reproject later frames whose CRS differs from the first.

Current Brisbane truth files may share a CRS, but the helper is unsafe for
general use. If one truth layer differs, labels are silently wrong because the
coordinates are treated as if they were in the first CRS.

Recommendation:

- match `_read_optional_vector_many()` behavior and reproject later truth files
  to the first CRS before concatenation;
- add a test with two tiny truth files in different CRSs.

### P2. Brisbane convenience script omits building points

Risk label: Medium ablation-validity and research-completeness risk

File:

- `scripts/train_anchor_free_brisbane.py:151`
- `scripts/train_anchor_free_brisbane.py:185`
- `scripts/train_anchor_free_brisbane.py:263`
- `scripts/train_anchor_free_brisbane.py:277`

The general config path can use `building_points_path`, but
`train_anchor_free_brisbane.py` has no default building-point path, no CLI flag
for building points, and never sets `cfg["data"]["building_points_path"]`.

This matters because building points were found to be meaningfully different
from building-area polygons and may be one of the most research-useful context
layers.

Recommendation:

- either remove this script from the main recommended workflow;
- or add `DEFAULT_BUILDING_POINTS`, `--building-points`, and
  `--no-building-points`;
- print building points in the enabled-feature list;
- ensure "road+building" and "road+context" variants are explicit about whether
  they mean polygons, points, or both.

### P2. Generic metrics keys still contain all-edge metrics

Risk label: Medium-high reporting risk

File:

- `pipe_network_completion/anchor_free/pipeline.py:376`
- `pipe_network_completion/anchor_free/pipeline.py:393`

The CLI now prints held-out `test_*` metrics, which is good. But
`metrics.json` and `metrics.csv` still contain generic keys like:

```text
roc_auc, pr_auc, f1
```

Those are computed on all candidate edges, not only held-out test edges. They
sit beside `test_roc_auc`, `test_pr_auc`, and `test_f1`, so it is still easy to
copy the wrong values into a paper table.

Recommendation:

- rename all-edge metrics to `all_edge_roc_auc`, `all_edge_pr_auc`,
  `all_edge_f1`, etc.;
- optionally retain old names for one release under a clearly deprecated block;
- update ablation CSV exports to prefer `test_*`.

### P2. GNN validation split is not used for training control

Risk label: Medium model-selection risk

File:

- `pipe_network_completion/anchor_free/model.py:223`
- `pipe_network_completion/anchor_free/model.py:239`
- `pipe_network_completion/anchor_free/model.py:250`

`train_road_edge_gnn()` accepts `val_index` and `test_index`, but training runs
for a fixed epoch count and never evaluates validation loss, early stopping, or
best-checkpoint selection. The function then returns probabilities from the
final epoch.

This is acceptable for smoke tests, but weak for paper runs.

Recommendation:

- add optional validation monitoring and best-epoch restore;
- log train/val loss curves;
- keep fixed-epoch behavior only as `early_stopping: false`.

### P2. The GNN is still full-graph transductive by default

Risk label: Medium claim-scope risk

File:

- `pipe_network_completion/anchor_free/model.py:172`
- `pipe_network_completion/anchor_free/model.py:179`
- `pipe_network_completion/anchor_free/model.py:181`

`build_pyg_road_edge_data()` builds message-passing edges from every road edge.
Loss is masked to the train split, but node embeddings are computed using the
full road graph.

This is acceptable only under the current within-network-completion framing.
It should not be reported as unseen-neighborhood or new-area generalization.

Recommendation:

- store `message_passing_scope: full_graph` in every GNN metrics row;
- add the spatial-block sensitivity experiment before making geographic
  robustness claims;
- consider a train-subgraph option only if reviewers ask.

### P2. Node-coordinate ablation exists in the model path but is not covered by tests

Risk label: Medium regression risk

File:

- `pipe_network_completion/anchor_free/model.py:77`
- `pipe_network_completion/anchor_free/model.py:149`
- `pipe_network_completion/anchor_free/pipeline.py:361`

The `include_node_coords` toggle is now wired into the pipeline, which is good.
But current tests only exercise the default coordinate-enabled path. A
regression could break the no-coordinate ablation without test coverage.

Recommendation:

- add a test that `build_pyg_road_edge_data(..., include_node_coords=False)`
  returns one node feature column;
- add a synthetic `run_anchor_free_experiment()` test with
  `model.include_node_coords: false`.

### P3. Context clipping can bias distance features near the study boundary

Risk label: Low-medium feature-bias risk

File:

- `scripts/filter_context_to_study_area.py:37`
- `scripts/filter_context_to_study_area.py:39`
- `scripts/filter_context_to_study_area.py:114`

The context filter clips data to the road total-bounds rectangle plus a fixed
buffer. Distance-to-nearest-building features near the boundary can be biased
upward if relevant buildings just outside the clip were removed.

Recommendation:

- keep the buffer at least as large as every downstream feature search radius;
- record the clip buffer in feature metadata;
- add a fallback-distance audit reporting the fraction of edges at fallback
  distance.

### P3. Local road density uses full candidate lengths, not clipped lengths

Risk label: Low-medium feature-quality risk

File:

- `pipe_network_completion/anchor_free/features.py:125`
- `pipe_network_completion/anchor_free/features.py:138`
- `pipe_network_completion/anchor_free/features.py:140`

For each edge buffer, the density calculation sums the full length of every
road segment intersecting the buffer. Long roads barely touching the buffer
contribute their entire length.

This is a feature approximation, not leakage, but it can distort density values
if source road segments are long.

Recommendation:

- compute clipped length inside the search buffer;
- or document the current value as "intersecting-road length density";
- add a test with one long segment that barely crosses the buffer.

### P3. Threshold grid exists in config but is not used by the pipeline

Risk label: Low-medium config/reporting ambiguity risk

File:

- `pipe_network_completion/anchor_free/config.py:66`
- `pipe_network_completion/anchor_free/pipeline.py:373`
- `pipe_network_completion/anchor_free/pipeline.py:374`

Configs define `evaluation.threshold_grid`, but the pipeline decodes with
`decoder.threshold` only. This is not a correctness bug if fixed-threshold
evaluation is intended, but it is easy to assume the threshold grid is active.

Recommendation:

- either remove the threshold grid from normal configs until the diagnostic
  script exists;
- or write a read-only threshold diagnostic script and clearly label it
  supplementary.

### P3. Baseline documentation has a small metric wording error

Risk label: Low documentation-accuracy risk

File:

- `pipe_network_completion/anchor_free/evaluation.py:55`
- `pipe_network_completion/anchor_free/evaluation.py:104`

The docstring says all-positive `pr_auc = recall = precision = p`. The code
correctly sets all-positive recall to 1.0 when positives exist. The wording
should say:

```text
pr_auc = precision = p, recall = 1.0
```

This is minor but worth fixing because the prevalence-baseline interpretation
is now central to the research story.

## Missing Tests To Add

I would add these before the next full real-data training pass:

1. training pipeline can use `split.strategy: buffer_invariant`;
2. 10 m and 5 m primary configs use identical split assignment when requested;
3. `decode_network(..., {"type": "sewer"})` raises until sewer logic exists;
4. `edge_predictions_inference_only.geojson` excludes `y`, `overlap_length`,
   and `overlap_ratio`;
5. multi-CRS utility truth files are reprojected before label creation;
6. road graph intersection noding check on crossing synthetic LineStrings;
7. GNN no-coordinate path works through `run_anchor_free_experiment()`;
8. Brisbane script includes or explicitly excludes building points.

## Positive Notes

- The feature guard is useful and catches obvious anchor leakage.
- Utility truth is kept in the label/evaluation path, not feature generation.
- The new prevalence baselines are the right direction for credible reporting.
- The buffer-invariant split helper is a good fix for prepared 10 m / 5 m
  feature tables.
- The anchor-free test suite is now broad enough to catch many basic
  regressions quickly.

## Recommended Next PR

I would not start with decoder improvements. The next PR should be a
research-validity cleanup PR:

1. wire `buffer_invariant` split into `run_anchor_free_experiment()`;
2. make unsupported decoder names fail loudly;
3. split prediction outputs into inference-only and evaluation files;
4. fix multi-CRS truth loading;
5. add tests for the above.

After that, rerun the core RF/GNN ablations and report model metrics beside the
all-positive baselines.
