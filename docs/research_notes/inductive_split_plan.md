# Inductive Component-Based Splitting for the Anchor-Free Pipeline

<!-- Workstream: Claude -->

**Status (2026-05-23, fourth update — research direction shifted):**
> Superseded for the headline split decision by
> [`project_framing_v2.md`](project_framing_v2.md). The within-network
> completion framing remains a valid sub-experiment, but the headline
> split is now **AOI-block** (2 km tiles, 500 m gap; 115-AOI canonical,
> 112-AOI watercourse-complete subset). Random within-AOI splits are
> demoted to secondary diagnostics. The component-based design analysis
> below is preserved as the historical reasoning trail.

**Status:** Design note, no code changes yet.
**Author:** Claude
**Related:** [`docs/anchor_free_design.md`](../anchor_free_design.md), ISARC 2024
paper (Zhang & Cai, doi:10.22260/ISARC2024/0121).

**Linked from / linking to:**
[`anchor_free_leakage_audit_codex.md`](../anchor_free_leakage_audit_codex.md)
(this note's framing reconciliation block) ·
[`audit_followup_implementation_plan.md`](audit_followup_implementation_plan.md)
(Stage 1 = "Research hygiene"; spatial-block work deferred to Stage 3) ·
[`research_reward_upgrade_recommendation_codex.md`](research_reward_upgrade_recommendation_codex.md).

## Decision (post-review)

This note was reviewed by the project lead (2026-05-23). Key decisions:

* **Scope is within-network completion, not cross-network generalisation.**
  The ISARC 2024 paper does not claim "give the model a new city and it
  predicts its sewer". It claims "given a partially-observed utility network,
  fill in the missing connections". The anchor-free pipeline keeps the same
  scope.
* **Therefore the per-edge stratified split stays the default**
  (`baseline.make_stratified_edge_splits`). The full inductive
  component-based split described below is downgraded from "recommended
  canonical evaluation" to "optional ablation, only implement if a reviewer
  asks for cross-subsystem generalisation numbers".
* **Non-utility features (roads, buildings, built-up, DEM) are computed once
  globally and the split only controls which labels are visible during
  training.** This matches what ISARC does with its road table and avoids
  boundary-effect bias on density / distance features.
* **Boundary effects on edges that straddle the split are an accepted
  residual risk.** Not worth a special handling path; the headline
  metric framing absorbs it.
* **The smaller transductive-leak fix on the GNN forward pass remains an
  open question** — see "Scoped follow-up" at the bottom of this note.

The rest of this document is the original technical analysis that produced
those decisions; it is left intact so the reasoning trail is auditable.

## Why this note exists

While reviewing the just-completed anchor-free training runs we flagged two
data-leakage issues with the current splitting strategy
(`baseline.make_stratified_edge_splits` + the full-graph GNN forward pass):

1. **Spatial leakage**: a stratified random split places adjacent road
   segments in different folds; on a road network this lets the model solve
   the task by spatial interpolation rather than by learning a useful
   inductive bias.
2. **Transductive message passing**: `RoadEdgeGNN.forward` runs
   `SAGEConv` over `data.edge_index` containing *all* 41 463 undirected
   edges, so train-time node embeddings absorb val/test edge context even
   though no labels leak.

The ISARC 2024 paper (the anchor-based pipeline that lives in this repo
under [`process.py`](../../process.py) and
[`pipe_network_completion/dataset.py`](../../pipe_network_completion/dataset.py))
already solved the analogous problem in a clean way for the anchor-based
setting. This note (a) reconstructs what that paper actually does, (b)
checks whether it qualifies as inductive training, and (c) proposes how to
port the same idea to the anchor-free pipeline.

No code is touched here. Implementation will happen in a follow-up PR once
this design is approved.

---

## What ISARC 2024 actually does (with code references)

### Step 1 — Build the MH-MH prediction graph

[`process.py:130-186`](../../process.py)

```python
# Spatial-join each gravity-main line with manholes; an MH-MH edge exists
# for every pair of MHs incident on the same line.
edge_search = spatial_join_intersects(gdf_ml, gdf_ap)
line_mh = edge_search[["index", "index_right"]].values
...
connected_mh_array = np.array(connected_mh)  # (n_lines, 3) -> (mh_i, mh_j, line_id)

raw_graph = Data()
raw_graph.node_id = torch.arange(len(gdf_ap))
raw_graph.edge_index = torch.from_numpy(connected_mh_array[:, :2].astype(int)).T
raw_g = to_networkx(raw_graph).to_undirected()
components = list(nx.connected_components(raw_g))
```

The graph used for splitting has:

* nodes = manholes / utility anchor points
* edges = MH–MH connections derived from gravity-main lines that physically
  touch two or more manholes

`components` is therefore a list of **connected sewer subsystems**: a
single component is everything that hydraulically flows together.

### Step 2 — Random split of components

[`process.py:188`](../../process.py)

```python
d_train, d_val, d_test = split_list_by_ratio(components, (6, 2, 2), seed=seed)
split_mask = {"train": d_train, "val": d_val, "test": d_test}
```

`split_list_by_ratio` shuffles `components` (deterministically by `seed`)
and slices the list into 60 / 20 / 20 by **component count, not edge
count**. Each component is wholly assigned to exactly one split.

### Step 3 — Per-split independent subgraph

[`pipe_network_completion/dataset.py:27-53`](../../pipe_network_completion/dataset.py)

```python
def splitdata(line, mh, road, mh_r_rl, r_r_rl, split_set):
    split_set = set(split_set)
    new_line  = line[lines_touching_split_set(line, split_set)]
    new_mh    = mh.iloc[sorted(split_set)]
    new_mh_r_rl = mh_r_rl[mh_r_rl["index_MH"].isin(split_set)]
    new_road  = road[road.index.isin(new_mh_r_rl["index_Road"])]
    new_r_r_rl = pairs where both roads are in new_road
    # remap all IDs to fresh 0..n local indices
```

So **each split is a self-contained `HeteroData` with disjoint MH nodes**:

* `train_data` has only training-component MHs
* `val_data`   has only validation-component MHs
* `test_data`  has only test-component MHs
* their MH-MH edges are disjoint by construction
* the model never sees test MH coordinates, test MH features, or the
  shape of the test sewer subsystem during training

### Step 4 — Train / eval is local to the split

[`scripts/evaluate_checkpoint.py`](../../scripts/evaluate_checkpoint.py)
uses `LinkNeighborLoader` against `data[("MH","link","MH")].edge_label_index`
of *one* split at a time, and the model is built from `data.metadata()`.
Negative MH-MH edges are sampled from within the split's MH set
(`data_transform` in `dataset.py`).

### One caveat in the ISARC implementation

A single *road segment* can appear in two splits if it happens to be near
manholes from different components, because `new_road` is recomputed
independently per split (via `MH_R_RL`). The road's *node ID* is
remapped per split, so the GNN sees independent road nodes, but the
underlying geometry — and therefore any positional / class features
derived from that geometry — leaks. ISARC tolerated this because the
prediction target was MH-MH edges, not road edges; the road context was
just message-passing scaffolding.

---

## Is ISARC training inductive?

**Yes, with one nuance.**

The classic definition (Hamilton, *GraphSAGE*, 2017): inductive learning
generalises to nodes / subgraphs unseen at training time. The opposite,
transductive learning, fixes the node set up-front and learns embeddings
specific to those nodes.

Under ISARC's split:

* The held-out test components are **subgraphs the model has never seen**:
  no shared MH nodes, no shared MH-MH edges, no shared MH-Road incidences,
  and the MH/Road embeddings produced for test are computed *from scratch*
  from the features of test nodes.
* Therefore the *evaluation* setting is **inductive at the
  component / subsystem level**: it asks "given a brand-new sewer
  subsystem's manholes and surrounding roads, can the model predict
  which MH-MH pairs are connected?"
* Within a single split, message passing is unavoidably transductive
  (the model sees all the nodes in the split when computing embeddings
  for any of them) — but that is the standard, uncontroversial form of
  transductivity in graph learning. It does not leak across splits.

The road-geometry leak noted above is a *partial* transductive leak across
splits but it's restricted to road features; the MH-MH prediction target
is untouched.

**Conclusion for the design discussion**: ISARC achieves a clean inductive
evaluation by **splitting the prediction graph into connected components
and giving each split its own self-contained subgraph**. The same recipe is
what the anchor-free pipeline should adopt.

---

## What the current anchor-free split does, and why it is not inductive

[`baseline.make_stratified_edge_splits`](../../pipe_network_completion/anchor_free/baseline.py)
takes a per-edge label vector `y` of length `n_edges = 41 463` and shuffles
edges within each class, then slices 60 / 20 / 20.

[`model.RoadEdgeGNN.forward`](../../pipe_network_completion/anchor_free/model.py)
runs `SAGEConv` over `data.edge_index` containing all 41 463 undirected
edges regardless of split. Loss is masked to `train_index` only.

Consequences:

1. Train-time node embeddings absorb context from *every* val/test edge
   incident on a shared road intersection node.
2. Spatial autocorrelation lets the model interpolate from train neighbours
   to test edges in the same neighbourhood. We see this in the
   train ↔ test ROC AUC gaps of `< 0.01` across all three Brisbane runs,
   which is far smaller than typical gaps under spatial-block evaluation.

So under the GraphSAGE definition the current pipeline is **transductive
*and* leaky**: not just transductive in the harmless within-subgraph sense,
but actively leaking spatial context across the split boundary.

---

## Proposed anchor-free inductive split

### Principle

Reproduce ISARC's design: **split the connected components of the
prediction-target graph; build a self-contained per-split subgraph; never
let train see val/test road geometry, road IDs, or edge_index during
training.**

For the anchor-free model the prediction target is per-road-edge
"is this segment a utility corridor?". The natural splitting unit is
**connected components of the ground-truth utility-line graph**.

### Step-by-step

1. **Build the truth utility graph** `G_T = (V_T, E_T)`:
   * Nodes `V_T` = endpoints of every truth utility LineString.
   * Edges `E_T` = the truth segments themselves.
   * Two truth segments share a node when their endpoints coincide within
     a small snap tolerance (1 m is enough on Brisbane data).

2. **Compute connected components** `C_1, ..., C_K` of `G_T` (NetworkX
   `connected_components`).

3. **Assign each component to a split**. Three options, in order of
   preference:
   * **(A) Edge-count-balanced shuffle (matches ISARC closest)**: shuffle
     components, slice 60 / 20 / 20 by component count. Trivial to
     implement; will produce imbalanced edge counts when one component
     dominates.
   * **(B) Greedy edge-balanced assignment**: assign components in
     decreasing size to the currently-smallest split. Yields tight
     train/val/test edge-fraction balance.
   * **(C) Spatial-block-aware assignment**: tile the bounding box of all
     components into a `k × k` grid; assign each component to a split
     based on the tile that contains its centroid (with deterministic
     shuffle of tiles). Gives geographic separation that is more
     defensible for spatial-CV literature reviewers.

   Recommended default: **(B)** for the headline metric, **(C)** as an
   ablation row to show generalisation across regions.

4. **Define a per-component spatial region** `R_k` for each component
   `C_k`:
   * `R_k = union(buffer(line, region_buffer_m) for line in C_k)`
   * `region_buffer_m` defaults to `2 * label_buffer_m + 50 m` so that
     the region comfortably contains every road segment a labelling step
     would mark positive for this component.

5. **Assign road edges to splits**:
   * For each candidate road edge, compute its midpoint and test which
     `R_k` regions contain it.
   * If a midpoint falls inside exactly one region, the edge goes to that
     component's split.
   * If it falls inside multiple regions, the edge goes to the component
     whose buffered geometry overlaps it most (tie-break on smaller
     component id for determinism).
   * If it falls inside zero regions ("background" edges, far from any
     utility line), tile-shuffle them deterministically across the three
     splits in the same fractions as the components. These background
     edges are trivial true negatives; spreading them stops one split from
     becoming a degenerate all-negative pool.

6. **Build per-split self-contained data**:
   * Subset the road candidate graph to the edges assigned to the split
     (plus the nodes incident on those edges). Re-derive node degrees.
   * Compute features only on this subset.
   * Compute labels only against the truth lines of the split's components
     (so a train road can never be labelled positive by a test
     component's truth line and vice versa).
   * Result: three independent `RoadCandidateGraph` / feature matrix /
     label tensor triples.

7. **Train inductively**:
   * **Strict variant**: train one GNN on the train subgraph only
     (`edge_index` = only train edges). Evaluate by building a *fresh*
     PyG `Data` for val and another for test, each with its own
     `edge_index`. The GNN must process unseen nodes at evaluation time
     — this is the inductive setting the ISARC paper achieves.
   * **Practical variant**: keep the full road graph but mask `edge_index`
     during the train forward pass to only include train-split edges.
     Evaluate with masked val/test edge sets. Functionally identical to
     the strict variant but cheaper to implement because we keep the same
     `Data` object.

   Either variant fixes the transductive-leak issue.

8. **Negative sampling & class imbalance**: each split has its own
   positive rate. `_positive_weight` already computes pos_weight per
   split for the GNN; this stays unchanged.

### What this looks like in the existing module layout

```text
pipe_network_completion/anchor_free/
├── splits/                       # NEW package
│   ├── __init__.py
│   ├── component_split.py        # build_inductive_component_split(...)
│   └── spatial_block_split.py    # build_spatial_block_split(...)
├── baseline.py                   # gains: kw `split_strategy` -> dispatch to splits.*
├── pipeline.py                   # gains: cfg["split"]["strategy"] dispatch
├── model.py                      # gains: train_inductive(train_data, val_data, test_data)
└── ...
```

### Proposed public API (no implementation yet, just signatures)

```python
# pipe_network_completion/anchor_free/splits/component_split.py
@dataclass(frozen=True)
class InductiveSplit:
    train_edge_ids: np.ndarray
    val_edge_ids:   np.ndarray
    test_edge_ids:  np.ndarray
    train_truth:    gpd.GeoDataFrame   # truth lines for this split only
    val_truth:      gpd.GeoDataFrame
    test_truth:     gpd.GeoDataFrame
    component_ids:  np.ndarray         # per-edge component id (-1 for background)
    background_mask: np.ndarray        # per-edge bool

def build_inductive_component_split(
    graph: RoadCandidateGraph,
    utility_truth_gdf: gpd.GeoDataFrame,
    *,
    label_buffer_m: float = 10.0,
    region_buffer_m: float | None = None,   # defaults to 2*label_buffer_m + 50
    balance: Literal["count", "edges", "tiles"] = "edges",
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 42,
) -> InductiveSplit: ...
```

### Pipeline integration

[`pipeline.run_anchor_free_experiment`](../../pipe_network_completion/anchor_free/pipeline.py)
gains a small dispatch:

```python
split_cfg = config.get("split", {"strategy": "stratified"})
if split_cfg["strategy"] == "inductive_component":
    split = build_inductive_component_split(graph, utility_truth, **split_cfg.get("kwargs", {}))
    train_index, val_index, test_index = (
        split.train_edge_ids, split.val_edge_ids, split.test_edge_ids
    )
    # labels are now per-split — re-run label_road_edges_from_utility_lines on
    # each split's truth subset, not the whole truth set
else:
    train_index, val_index, test_index = make_stratified_edge_splits(labels.y, seed=seed)
```

The existing `make_stratified_edge_splits` is kept as the default so the
already-published Brisbane numbers stay reproducible.

### Model integration

[`model.train_road_edge_gnn`](../../pipe_network_completion/anchor_free/model.py)
gains an `inductive: bool` flag (default `False` to preserve current
behaviour). When `True`, the function builds an edge-index mask:

```python
if inductive:
    train_edge_mask = np.zeros(data.edge_label_index.shape[1], dtype=bool)
    train_edge_mask[train_index] = True
    message_passing_edge_index = data.edge_index[:, undirected_mask_from(train_edge_mask)]
else:
    message_passing_edge_index = data.edge_index
```

…and the forward pass uses `message_passing_edge_index` during training,
swapping to the full edge index only at val/test forward time (and even
then masking out cross-split edges if we want strict inductiveness).

### Config additions

```yaml
split:
  strategy: inductive_component         # or "stratified" (current default)
  kwargs:
    label_buffer_m: 10.0                # mirrors graph.label_buffer_m
    region_buffer_m: 70.0
    balance: edges                      # "count" / "edges" / "tiles"
    fractions: [0.6, 0.2, 0.2]
    seed: 42
model:
  inductive_message_passing: true       # new flag wired by train_road_edge_gnn
```

---

## Backward compatibility

* Default `split.strategy` is `"stratified"` so existing
  `configs/anchor_free_isarc2024.yaml` runs reproduce today's numbers
  bit-for-bit.
* `make_stratified_edge_splits` is kept as-is.
* `RoadEdgeGNN` model architecture does not change; only the forward pass
  receives a (possibly masked) `edge_index`.
* All existing tests continue to pass; new tests cover:
  1. `build_inductive_component_split` returns disjoint splits.
  2. Each split's labels are only ever positive on edges whose truth
     parent is in that split.
  3. Background edges are deterministically tiled.
  4. With `inductive_message_passing=True`, the GNN's train forward pass
     never touches a node that appears only in val/test edges.

---

## Expected metric impact

Based on published spatial-block-CV studies (Roberts 2017, Ploton 2020) and
on the very small train ↔ test gap we currently observe:

| Metric                | Current stratified (full Brisbane, all features) | Inductive component (expected) |
| --------------------- | ----------------------------------------------- | ------------------------------- |
| `test_roc_auc`        | 0.77                                            | **0.60 – 0.68**                 |
| `test_pr_auc`         | 0.82                                            | **0.55 – 0.65**                 |
| `test_f1`             | 0.79                                            | 0.55 – 0.65                     |
| train ↔ test gap (auc)| `< 0.01`                                        | `0.05 – 0.15` (healthy)         |

The headline number will go down — and that is exactly the point. The
inductive number is what reviewers will accept as a realistic estimate of
how the model behaves on a new utility-network region it has never seen.

---

## Open questions for follow-up

1. **What's "a new region"?** Is the unit of generalisation a sewer
   subsystem (a single connected component), a council area, or a square
   tile of a fixed size? ISARC's choice (connected component) is the most
   defensible for sewer because subsystems are operationally meaningful;
   for water networks with looped topology a tile-based split may be
   more natural.

2. **Background edges**: do we evaluate metrics on them or hold them out?
   Reporting metrics over only "edges within `region_buffer_m` of some
   utility line" sidesteps the trivial-true-negative inflation but
   changes the prevalence assumption. Decision likely depends on whether
   the downstream decoder is meant to predict for *all* roads or only
   *roads-near-something*.

3. **Per-component features vs global features**: features like
   `local_road_density` depend on the surrounding road set. If we strictly
   subset roads per split, density estimates become biased near the split
   boundary. Options:
   (a) accept the bias,
   (b) compute density features on the global road set and only subset at
       the model-input stage,
   (c) compute density per split but extend the spatial query radius beyond
       the split boundary.

   Option (b) is the most defensible — features are derived once globally
   from non-utility inputs (roads, buildings, DEM) and the split only
   controls *which labels are visible* and *which edges are message-passed
   during training*. This matches ISARC: their road table is built once
   globally and then projected into each split.

4. **Reproducing today's numbers**: keep `make_stratified_edge_splits` as
   the default so the existing
   `outputs/anchor_free_brisbane_*` runs stay reproducible. Add the
   inductive-component split as an opt-in for the next ablation table.

---

## Recommended next actions (revised after review)

The full inductive-component-split implementation listed above is
**deferred** per the decision at the top of this note. The implementation
plan is preserved verbatim so a future contributor can resurrect it in one
PR if a reviewer ever asks for cross-subsystem numbers.

### Global-feature contract (accepted, documented for the record)

The current pipeline already computes features globally:

* `build_road_edge_features` reads `roads`, `buildings`, `built_up`,
  `dem_path` once and produces a `(n_edges, n_features)` table for the
  whole graph. None of these inputs are split-dependent.
* `standardize_features(features.features, train_index=train_index)` fits
  `mean` / `std` on train rows only and applies the transform to all
  rows. Already correct.
* Per-split labels are derived from the global truth table by spatial
  matching; nothing about feature computation knows about the split.

This satisfies the "non-utility features computed globally" decision with
no code change required. The contract is now documented here so future
edits don't regress it accidentally.

### Boundary effect — accepted risk, mitigations available if needed

If boundary effects ever look like they matter (e.g. metrics jump when
`region_buffer_m` is changed in some future ablation), the cheap
mitigations in order of effort are:

1. Increase the spatial extent over which density / distance features
   are computed so it always exceeds the buffer used for splitting.
2. Drop a small ring of edges near the split boundary from evaluation
   (not training) so reported metrics ignore the most ambiguous edges.
3. Switch to the spatial-block split (option C earlier) which gives
   cleaner geographic separation.

None of these are needed right now.

## Scoped follow-up: GNN transductive-leak fix (optional)

Independent of the split-strategy decision, there is a smaller technical
question worth surfacing.

Today's `RoadEdgeGNN.forward` runs `SAGEConv` over the full
`data.edge_index` (all 41 463 undirected edges) during *both* the train
and the eval forward pass. That means:

* During training, a train edge's two endpoint embeddings absorb context
  from val/test edges that share those nodes.
* No labels leak — but val/test *edge features* (length, bearing, building
  context, DEM) feed into the message passing.

Under the within-network-completion framing this is **arguably fine** —
the test edges' features are part of "the observed network" too. ISARC's
per-split disjoint subgraphs accidentally avoid this because they
re-index each split's roads from scratch, but that is a side-effect of
their splitting choice, not a deliberate inductive-message-passing
design.

Two options, neither blocking:

* **(A) Leave it alone.** Document that anchor-free message passing is
  transductive (sees all edges during training) and that this is
  consistent with the within-network-completion framing.
* **(B) Add an `inductive_message_passing: bool` flag to
  `train_road_edge_gnn`** that masks `edge_index` to train-only edges
  during the train forward pass. Defaults to `False` to keep current
  numbers. Cheap to implement (~30 lines), useful if a reviewer ever
  flags the leak.

Recommendation: **(A) for now, leave a TODO that points back to this
note**. Re-open if a reviewer asks.

## Status of the original "recommended next actions" list

Kept here for traceability; all six items below are now deferred per the
decision at the top.

1. ~~Implement `pipe_network_completion/anchor_free/splits/component_split.py`.~~ (deferred)
2. ~~Wire the `split.strategy` config dispatch in `pipeline.py`.~~ (deferred)
3. ~~Add `inductive: bool` to `train_road_edge_gnn`.~~ (open — see
   "Scoped follow-up" above)
4. ~~Add three test files mirroring the behavioural assertions.~~ (deferred)
5. ~~Re-run the Brisbane experiments under
   `split.strategy = inductive_component`.~~ (deferred)
6. ~~Update `docs/anchor_free_design.md` to call the inductive component
   split the canonical evaluation protocol.~~ (deferred — the canonical
   protocol stays per-edge stratified, with this note linked as the
   discussion of why)
