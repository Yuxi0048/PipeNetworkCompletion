# Architectural Cleanup Plan — Anchor-Free, Long-Term Credibility

<!-- Workstream: Claude -->

**Status (2026-05-23, third update — research direction shifted):**
> Superseded as the canonical experiment plan by
> [`project_framing_v2.md`](project_framing_v2.md). The headline shift:
> the project is now a **change-of-support GeoAI** problem (visible
> urban supports → hidden sewer support), not road-edge classification.
> Phases 2.A and 2.B in this plan have **landed** and the artefacts
> (heterogeneous graph, clipped density, prevalence baselines, ISARC
> split) remain canonical. Phases 2.C / 2.D / 2.E / 2.F are
> **superseded** by the experiment plan in `project_framing_v2.md` §6
> (candidate-support ablation first, watercourse feature ablation
> second, AOI-block split as the only headline split).

**Status:** PROPOSED 2026-05-23 — awaiting one explicit "go" before
implementation starts.

**Owner direction (2026-05-23):**
> *"for long term goal and credible, change the improper ones as early as
> possible, don't defer or hide"*

**Owner direction (2026-05-23, second turn):**
> *"Heterogeneous"*
>
> Architecture choice — go with Option C (heterogeneous graph with both
> `RoadSegment` and `Intersection` node types and multiple relation types),
> not Option B (homogeneous roads-as-nodes). This matches the
> `HeteroData` / `to_hetero` pattern already used by ISARC's anchor-based
> model ([`pipe_network_completion/model.py:98`](../../pipe_network_completion/model.py)
> and [`pipe_network_completion/dataset.py`](../../pipe_network_completion/dataset.py)).

Therefore: every deferred item from the previous checkpoint is converted
into a Stage 2 sub-phase here. No item is hidden or punted.

**Sister docs:**
- [`audit_followup_implementation_plan.md`](audit_followup_implementation_plan.md)
  — original staged plan; this note supersedes its §16 Stage 2 entry.
- [`current_codebase_review_codex.md`](current_codebase_review_codex.md) —
  Codex's review; sources the P1/P2/P3 issues addressed below.
- [`inductive_split_plan.md`](inductive_split_plan.md) — within-network
  completion framing; unchanged.

---

## 0. What "improper" means here

Six items are currently in the codebase that would hurt the paper's
credibility if left to land:

| # | Item | Why improper |
|---|------|--------------|
| 1 | **Roads-as-edges graph formulation** without intersection noding | The GNN's `edge_index` literally misses real road-road connectivity at every unmodeled crossing. Message passing is incomplete in a way reviewers will catch. |
| 2 | **`local_road_density` uses full-segment lengths** instead of clipped lengths | A 2 km arterial that grazes a 100 m buffer contributes its full 2000 m to the density numerator. Feature value can be wrong by 100×. |
| 3 | **Generic `roc_auc`/`f1` keys in `metrics.json` are all-edge** | They sit beside `test_roc_auc`/`test_f1` and look identical to readers; nothing on the key signals "warning, this is the optimistic in-sample number". |
| 4 | **GNN trains for fixed 100 epochs, reports last-epoch state** | If the model overfits at epoch 50, we still report the worse epoch-100 numbers. We have no model selection. |
| 5 | **Threshold hardcoded at 0.5; `evaluation.threshold_grid` exists but is unused** | All reported precision/recall numbers are at an un-tuned operating point. Reviewers ask why. |
| 6 | **Single label per source-shapefile LineString** | A 2 km arterial gets one positive/negative label for the whole length. Within-segment heterogeneity is invisible to the model. |

The owner's directive is to fix all six **before** the next Stage 2
ablation run, so the next set of numbers is the canonical paper-grade
set.

---

## 1. Architecture decision: heterogeneous graph (`RoadSegment` + `Intersection`)

This is the biggest single change and it dissolves problems #1 and #6 while
matching ISARC's existing heterogeneous pattern.

### Data model

Two node types and three edge types:

```text
Node types
──────────────
RoadSegment   one node per LineString from the road shapefile.
              Carries: length_m, bearing_rad, road_class one-hot,
                       building/built-up/DEM context features.
              Prediction target: per-RoadSegment binary label
                                 "carries a utility line within label_buffer_m".

Intersection  one node per snapped geometric meeting point of road
              segments (endpoint snap + interior crossings if noded).
              Carries: x, y (normalised), degree (how many segments
                       touch it).
              No prediction target; provides structural / positional context.

Edge types (PyG relations)
──────────────────────────
("RoadSegment", "crosses", "RoadSegment")
    sjoin-derived adjacency (mirrors ISARC's
    spatial_join_intersects(roads, roads) at process.py:239-243).
    Catches every road-road meeting regardless of noding state.

("RoadSegment", "touches", "Intersection")
    For every (segment, intersection_point) where the segment's
    geometry passes through that point. Built from the endpoints
    of each LineString (and from interior crossings if noded).

("Intersection", "rev_touches", "RoadSegment")
    PyG-style reverse relation, added automatically by
    transforms.ToUndirected() so message passing flows both ways.
```

This **strictly generalises both Option A (roads-as-edges) and Option B
(roads-as-nodes)**:
- Like Option B, `RoadSegment` is the prediction unit, sjoin captures
  road-road adjacency, and ISARC's mental model transfers directly.
- Like Option A, we keep an explicit `Intersection` node type, so when
  the road shapefile is well-noded the GNN can also reason over
  intersection-level structure (endpoint degree, multi-road junctions).
- Unlike either pure option, the heterogeneous formulation lets each
  node type carry its **own** feature vector — `RoadSegment` carries the
  rich road/building/DEM features, `Intersection` carries position +
  degree only.

### Why heterogeneous (vs simpler pure roads-as-nodes)

- ISARC's anchor-based pipeline is already heterogeneous (`MH` +
  `Road`); it uses `torch_geometric.nn.to_hetero` to lift a homogeneous
  GNN over all relations
  ([`model.py:98-101`](../../pipe_network_completion/model.py)). We reuse
  exactly that pattern with `MH` swapped for `Intersection`.
- Reviewers reading both papers see the same architectural shape: one
  prediction-target node type, one context node type, sjoin-derived
  adjacency. Easy comparison.
- Adding an `Intersection` node type costs little (just snapped endpoint
  table + one edge type) but means we can later layer intersection-level
  features (signal-controlled? roundabout?) without another refactor.

### What changes structurally

```
Before (roads-as-edges, homogeneous)          After (heterogeneous)
─────────────────────────────────────         ──────────────────────────────────────
nodes = intersection endpoints                node types: RoadSegment, Intersection
edges = road segments                         relations:
prediction target = per-edge label              (RoadSegment, crosses, RoadSegment)
features computed per-edge                      (RoadSegment, touches, Intersection)
decoder selects edges                         prediction target = per-RoadSegment label
                                              features per-RoadSegment (rich) +
                                                       per-Intersection (sparse)
                                              decoder selects RoadSegment nodes →
                                                returns the LineStrings
```

### What is preserved

- All non-anchor feature math (length, bearing, building stats, DEM,
  built-up) is unchanged in *content*; only the per-row dimension changes
  from "edge" to "RoadSegment node".
- The anchor-feature guard (`assert_no_anchor_features`) is unchanged
  and runs on every relation's feature dict.
- The buffer-invariant split (Stage 1) is unchanged — re-keyed from
  `edge_id` to `road_segment_id` (the prediction-unit id).
- The synthetic fixture stays — `make_synthetic_anchor_free_data` is
  agnostic to the graph formulation.
- The within-network-completion framing
  ([`inductive_split_plan.md`](inductive_split_plan.md)) is unchanged.
- ISARC's `to_hetero` GNN pattern is reused — we are not inventing a new
  model architecture, just instantiating the existing pattern over the
  anchor-free relation set.

### What is invalidated (and how we handle it)

- Every `output/anchor_free_*` directory is already deleted (clean slate
  per the previous checkpoint).
- Every test referencing `RoadCandidateGraph.edges["edge_id"]` as a
  prediction target updates to `HeteroRoadGraph.road_segments["segment_id"]`.
- The old `road_graph.py` module stays for historical comparison but is
  banner-marked deprecated for the anchor-free pipeline.

---

## 2. Phases and AR-IDs

### Phase 2.A — Heterogeneous-graph refactor (the big one)

| AR-ID | Rule | Acceptance signal |
|-------|------|-------------------|
| **AR-AF-2A.1** | New module `pipe_network_completion/anchor_free/hetero_road_graph.py` exposes `build_hetero_road_graph(roads_gdf, *, target_crs, snap_tolerance_m, keep_columns)` returning a `HeteroRoadGraph` dataclass with: `.road_segments` (one row per LineString: `segment_id`, `length_m`, `bearing_rad`, kept attrs, geometry); `.intersections` (one row per snapped meeting point: `intersection_id`, `x`, `y`, `degree`, geometry); `.segment_crosses_segment` (sjoin-derived (2, n) array of `(segment_a, segment_b)` pairs, no self-loops); `.segment_touches_intersection` (2, n) array of `(segment, intersection)` pairs derived from each LineString's endpoints (and, when noding is enabled in a later phase, interior crossings); `.crs`, `.metadata`. | Module imports; new test `test_hetero_road_graph_built_from_synthetic` passes. |
| **AR-AF-2A.2** | `build_hetero_road_graph` uses `gpd.sjoin(roads, roads, predicate='intersects')` for the `crosses` relation, mirroring ISARC's `process.py:239-243`. The `touches` relation is built from snapped LineString endpoints with the existing `snap_tolerance_m` rounding helper from `road_graph.py`. | Synthetic test: canonical "+" shape (4 LineStrings meeting at one point) yields 4 `RoadSegment` nodes, 1 `Intersection` node, 6 `crosses` edges (every road pair touches at the centre), 8 `touches` edges (each road has 2 endpoints, one of which is the central intersection). |
| **AR-AF-2A.3** | `to_pyg_hetero_data(graph, features, labels) -> torch_geometric.data.HeteroData` lifts the dataclass into PyG: `data["RoadSegment"].x = ...`, `data["RoadSegment"].y = ...`, `data["Intersection"].x = ...`, `data["RoadSegment","crosses","RoadSegment"].edge_index = ...`, `data["RoadSegment","touches","Intersection"].edge_index = ...`. Apply `T.ToUndirected()` so the reverse relations are added automatically (this mirrors `dataset.py` line `T.ToUndirected()(data)` in ISARC's pipeline). | New test asserts the returned `HeteroData` has the expected node-type set `{"RoadSegment","Intersection"}` and the four expected directed relations after `ToUndirected`. |
| **AR-AF-2A.4** | The old `road_graph.py` and its `build_road_candidate_graph` are **kept** for historical comparison and tests; a clear `# Deprecated for the anchor-free pipeline — use hetero_road_graph` banner is added. | Banner present; existing tests still pass. |
| **AR-AF-2A.5** | `features.build_road_edge_features` is replaced by `features.build_road_segment_features(graph: HeteroRoadGraph, ...)` returning a `RoadSegmentFeatureTable`. Feature math is identical (length, bearing, building stats, DEM, built-up); only the per-row dimension changes from "edge" to "RoadSegment node". A small new `features.build_intersection_features(graph)` returns the sparse per-`Intersection` table (`x`, `y`, `degree`, optionally `include_coords=False` to honour the Stage-2 location-memorisation ablation). | Existing feature tests adapted; same `assert_no_anchor_features` guard fires on the same forbidden tokens against both feature tables. |
| **AR-AF-2A.6** | `labels.label_road_segments_from_utility_lines` replaces `label_road_edges_from_utility_lines`. Buffer/overlap-ratio math is identical; the per-row dimension changes to per-RoadSegment. | New test asserts both positive and negative `RoadSegment` labels on the synthetic fixture. |
| **AR-AF-2A.7** | `model.HeteroRoadGNN` follows ISARC's `to_hetero(GNN(...), metadata=data.metadata())` pattern. The base homogeneous GNN takes `(node_features, edge_index)` and emits per-node hidden states; `to_hetero` lifts it over the two node types and two relations. A small node-level head produces per-`RoadSegment` logits. The `include_node_coords: bool` Stage-2 flag now toggles the `(x, y)` columns inside `build_intersection_features`. | New test: one training epoch on the synthetic fixture; output shape `(n_road_segments,)`; `include_node_coords=False` removes (x, y) from the `Intersection` feature tensor and the run still completes. |
| **AR-AF-2A.8** | `decoder.decode_threshold` / `decode_connected` operate on `RoadSegment` nodes. `decode_threshold` selects RoadSegment nodes with `p >= threshold` and writes their LineStrings. `decode_connected` runs MST over the `crosses` relation weighted by `-log(p_avg(seg_a, seg_b)) + lambda_length * |seg_a.length + seg_b.length|`, returning the RoadSegment nodes in the chosen connected backbone. | `decoded.road_segments` (a GeoDataFrame of selected LineStrings) replaces `decoded.edges`; integration test confirms QGIS-loadable output. |
| **AR-AF-2A.9** | `pipeline.run_anchor_free_experiment` is updated end-to-end. All call sites and the buffer-invariant split now key on `segment_id`. Saved GeoJSONs use the new names: `road_segment_predictions_inference_only.geojson` (probability only), `road_segment_predictions_for_evaluation.geojson` (with `y`, `overlap_ratio`). The old `edge_predictions*.geojson` names are kept as deprecated copies for one release. | Smoke run on synthetic + every existing pipeline test pass. |
| **AR-AF-2A.10** | The Stage-1 buffer-invariant split helper is renamed `make_buffer_invariant_splits(segment_ids, ...)` with the old `edge_ids` parameter kept as a deprecation alias for one release. ISARC's `random.Random(seed)` mechanism is preserved. | Existing buffer-invariant split tests pass; new test asserts both parameter names produce identical output. |
| **AR-AF-T.2A** | New / renamed tests: `test_hetero_road_graph.py`, `test_road_segment_features.py`, `test_road_segment_labels.py`, `test_hetero_road_gnn.py`, `test_road_segment_decoder.py`. Old `test_anchor_free_*` files are kept where they still apply to the synthetic generator / guard / split, and renamed where they reference `RoadCandidateGraph` directly. | `pytest -q` ≥ 83 passing (current count); coverage on `anchor_free/` ≥ 90%. |

**Blast radius:** ~750 LOC across 7 modules (hetero_road_graph is new, plus
edits to features, labels, model, decoder, pipeline; new tests). About
~200 LOC across the test surface. Estimated effort: ~half to one focused
day. The to_hetero / HeteroData parts are mostly mechanical because the
ISARC pattern is already in the repo to copy from.

### Phase 2.B — Clipped-length local density

Fixes issue #2.

| AR-ID | Rule | Acceptance signal |
|-------|------|-------------------|
| **AR-AF-2B.1** | `features._edge_local_road_density` (which becomes `_node_local_road_density` in the refactor) sums `road.intersection(search_buffer).length` instead of `road.length` for each candidate within the buffer. | Recurrence test: a synthetic long-arterial-just-grazing-buffer scenario produces density value within 20% of the true clipped length, not 100× off. |
| **AR-AF-2B.2** | The feature column is renamed `local_road_clipped_length_density_{buffer}m` to signal the corrected math. The old column name is **not** emitted (clean break — there are no prior runs to preserve). | Grep for `local_road_density` outside the renamed function returns no production hits. |
| **AR-AF-T.2B** | Existing density test updated to test the clipped behaviour. | Test passes. |

### Phase 2.C — All-edge / all-node key rename

Fixes issue #3.

| AR-ID | Rule | Acceptance signal |
|-------|------|-------------------|
| **AR-AF-2C.1** | `compute_edge_metrics` writes its primary keys with the `all_node_` prefix (post-refactor). The unprefixed keys (`roc_auc`, `f1`, etc.) are **dropped**, not aliased — there are no downstream notebooks reading them yet (verified by grep of `*.ipynb` and `scripts/*.py`). | `cat metrics.json \| jq 'keys'` shows `all_node_*`, `train_*`, `val_*`, `test_*`; no bare `roc_auc`. |
| **AR-AF-T.2C** | New test asserts the metric dict shape; existing tests updated to the new key names. | All tests pass. |

### Phase 2.D — Early stopping with best-epoch restore on val PR-AUC

Fixes issue #4.

| AR-ID | Rule | Acceptance signal |
|-------|------|-------------------|
| **AR-AF-2D.1** | `train_road_node_gnn` accepts `early_stopping_metric: "val_pr_auc"` (default), `early_stopping_patience: int \| None` (default `None` = no patience, just best-epoch restore), and `restore_best_epoch: bool` (default `True`). | Function signature exposes the three new kwargs with documented defaults. |
| **AR-AF-2D.2** | After each epoch, the GNN computes the `val_pr_auc` if `val_index` has both classes; otherwise falls back to `val_loss`. State dict is saved when the metric improves. At end of training, the best state dict is restored. | Recurrence test: a training run where val_pr_auc plateaus early restores an earlier epoch's weights; resulting `test_pr_auc` matches the best-epoch's val number to within ±0.02. |
| **AR-AF-2D.3** | `metrics.json` records `best_epoch`, `best_val_pr_auc`, `final_epoch_val_pr_auc`. The CLI prints them so the user can see when the model peaked. | Synthetic smoke run shows all three keys + the CLI print. |
| **AR-AF-T.2D** | New test `test_road_node_gnn_early_stopping.py`. | Two tests pass (best-epoch restore + patience-based stopping). |

### Phase 2.E — Val-tuned threshold

Fixes issue #5.

| AR-ID | Rule | Acceptance signal |
|-------|------|-------------------|
| **AR-AF-2E.1** | `pipeline.run_anchor_free_experiment` reads `evaluation.threshold_grid` and `evaluation.tune_threshold_on_val` (default `true`). When tuning is on, the pipeline sweeps the grid on `val_index`, picks the F1-optimal threshold, and uses it for both the decoder and the reported test metrics. | `metrics.json` contains `tuned_threshold`, `configured_threshold`; CLI prints which one was used. |
| **AR-AF-2E.2** | A separate diagnostic remains available: `scripts/diagnose_threshold_sensitivity.py` produces the val/test sweep CSV+PNG for any saved geojson (Stage 4 / B.1 of the original audit-followup plan). | Diagnostic script writes the four expected files. |
| **AR-AF-T.2E** | New tests cover both: (a) pipeline tuning on val with synthetic fixture, (b) diagnostic script output shape. | All tests pass. |

### Phase 2.F — Stage 2 ablation re-run (post-refactor)

After 2.A–2.E land:

| AR-ID | Rule | Acceptance signal |
|-------|------|-------------------|
| **AR-AF-2F.1** | `scripts/run_anchor_free_stage2_ablation.py` is updated to drive the road-as-node pipeline. Its 7 variants are preserved verbatim. | Smoke run on synthetic completes. |
| **AR-AF-2F.2** | Full Brisbane run of all 7 variants on the post-cleanup pipeline writes one canonical `outputs/anchor_free_stage2/ablation_stage2.csv`. | File present; all variants reported a `test_*` row beside `all_node_all_positive_*` baselines. |

---

## 3. Order of operations

```
2.A (refactor) ─→ 2.B (clipped density) ─→ 2.C (key rename)
        ↓                  ↓                       ↓
        └─→ all of 2.A's tests must pass before 2.B starts ←─┘
                              ↓
                            2.D (early stopping) ─→ 2.E (val threshold)
                              ↓
                            2.F (re-run Brisbane)
```

- **2.A is monolithic** — it touches every anchor_free module. One PR.
- **2.B, 2.C** land in the same PR as 2.A (they live in the same files
  and would conflict if delayed).
- **2.D, 2.E** land as one separate PR after 2.A merges.
- **2.F** is a clean re-run; not a code change. Background job + report.

---

## 4. What does NOT change

- ISARC anchor-based pipeline (`process.py`, `scripts/build_graphs.py`,
  `scripts/evaluate_checkpoint.py`, `pipe_network_completion/dataset.py`)
  — untouched.
- `inductive_split_plan.md` framing decision — within-network completion.
- `anchor_free_design.md` overall design narrative (updated only to
  reflect roads-as-nodes vocabulary).
- `make_buffer_invariant_splits` — kept, only renamed argument from
  `edge_ids` to `node_ids` with deprecation alias.
- The prevalence baselines — kept.
- The CLI Δ-column print — kept.

---

## 5. Risks and how this plan handles them

| Risk | Mitigation |
|------|------------|
| **Refactor breaks something subtle** | All 83 existing tests must pass before any phase ships; new tests added in each phase. |
| **Sjoin on 41k roads is slow** | Brisbane test before final run confirms timing; rtree spatial index makes it O((n + k) log n) which is fine. |
| **Decoder semantics change** | `decoded.nodes` replaces `decoded.edges`; old `decoded_network.geojson` filename kept for one release as deprecated alias. |
| **Hand-rolled early stopping has bugs** | Recurrence test with known overfit curve verifies best-epoch restoration. |
| **Per-node features lose edge structure (bearing!)** | Bearing remains a per-node attribute of the road LineString itself — still informative. Local density and building stats also stay per-node-segment. |
| **Val-tuned threshold over-tunes if val is small** | Use the larger of val and train+val for thresholding when val has <100 positive examples. |

---

## 6. Definition of done (whole stage)

1. All AR-AF-2A.x through AR-AF-2F.x rules satisfied.
2. `pytest -q` green; coverage ≥ 90% on `pipe_network_completion/anchor_free/`.
3. `outputs/anchor_free_stage2/ablation_stage2.csv` exists with 7 rows,
   each carrying `test_*` metrics beside `all_node_all_positive_*` baselines
   and a `tuned_threshold` column.
4. `docs/anchor_free_design.md` updated to call roads-as-nodes the
   canonical architecture; old roads-as-edges section moved to an
   appendix labelled "previous formulation".
5. A short retrospective paragraph appended to this note recording any
   surprises encountered during implementation (so the next refactor
   benefits).

---

## 7. Status banner template

When a phase lands, append a status line to its header:

```
**Status (YYYY-MM-DD):** LANDED in <commit> | IN PROGRESS | REJECTED <reason>
```

When the whole stage lands, the top-of-doc status flips to:

```
**Status:** LANDED <date>; canonical anchor-free architecture is roads-as-nodes.
```

---

## 8. Ready to execute

Implementation will:

1. Start with Phase 2.A as one focused session (likely ~half a day).
2. Run the full test suite at the **end of 2.A+2.B+2.C** (single PR).
3. Pause for the explicit second checkpoint before 2.D+2.E.
4. Pause again before 2.F (the Brisbane re-run) so the owner sees the
   cleaned pipeline first.

No training run will be started without an explicit "go" at each
checkpoint. This note is the only thing landing this turn unless the
owner says otherwise.
