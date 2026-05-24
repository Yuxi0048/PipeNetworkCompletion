# Project Framing v2 — Anchor-Free Sewer Prediction as Change-of-Support GeoAI

<!-- Workstream: Claude (transcribing owner-authored framing 2026-05-23) -->

**Status:** CANONICAL 2026-05-23. Supersedes the narrower framing in
[`../anchor_free_design.md`](../anchor_free_design.md),
[`audit_followup_implementation_plan.md`](audit_followup_implementation_plan.md),
[`architectural_cleanup_plan.md`](architectural_cleanup_plan.md), and
[`inductive_split_plan.md`](inductive_split_plan.md). Those documents are
preserved for historical reasoning; their next-step recommendations are
overridden by the experiment plan in §6 of this note.

**Source:** Owner-authored framing message, 2026-05-23 (after Codex's
hybrid-candidate-graph + AOI work landed). This note transcribes the
owner's framing verbatim into the project's traceability format so future
contributors find a single source of truth.

---

## 1. From narrow to credible framing

### First-prompt framing (now too narrow)

```text
G_R = road graph
road edges are candidate utility corridors
predict P(road edge belongs to utility network)
```

This is still a useful **baseline**, but it presumes the road network is
the true utility support. Empirically that does not hold: on the
2 km / 500 m AOI setup, the hybrid road/building candidate graph has
truth-length recall of only **~16.8% at 5 m, ~40.6% at 10 m, ~89.8% at
50 m**. A road-centred classifier cannot recover ~60% of sewer mains at
the 10 m tolerance regardless of model quality.

### v2 framing (canonical going forward)

**Anchor-free sewer corridor prediction as a change-of-support GeoAI
problem.**

* **Visible urban supports:** roads, buildings, DEM, watercourses,
  land-use, parcels.
* **Hidden target support:** sewer utility-network geometry.
* **Research problem:** how can evidence from visible supports be
  transferred to a latent sewer-corridor support?

The model estimates:

```text
P(hidden sewer exists within r meters of candidate support edge e | visible context)
```

where `r` is a support tolerance:

| `r` | interpretation |
| --- | -------------- |
| 5 m  | strict alignment |
| 10 m | near-centerline |
| 20–30 m | local corridor |
| 50 m | planning corridor |

---

## 2. What changes vs. v1

| Aspect | v1 (narrow) | v2 (canonical) |
| --- | --- | --- |
| Candidate graph | road centerlines only | hybrid + road offsets + watercourses + multi-support |
| Headline target | per-road-edge binary label | per-candidate-corridor binary, evaluated at multiple tolerances |
| Split | random within-network, then within-AOI buffer-invariant | **AOI-block only** (2 km tiles, 500 m gap) for headline; within-AOI random as secondary diagnostic |
| Headline metric | edge-level F1 / ROC AUC at one tolerance | **representability** (truth-length recall at 5/10/20/30/50 m) first, then classifier metrics |
| Paper claim | "predict utility corridors without anchors" | "change-of-support GeoAI; recover broad corridor support; strict centerline recovery remains limited" |
| Sparse-anchor work | folded into anchor-free | **separate setting** — see §7 |

The Phase 2.A heterogeneous (`RoadSegment` + `Intersection`) refactor
remains useful as the GNN substrate. The `RoadSegment` node type is
generalised to `CandidateCorridor` so it can hold road, road-offset,
building-access, demand-kNN, demand-MST, and watercourse edges in one
graph (`candidate_source` one-hot becomes a first-class feature column).

---

## 3. Current AOI benchmark

The canonical experiment unit is now an AOI, not the whole Brisbane area.

### Default AOI set

* tile size: 2 km × 2 km
* gap between tiles: 500 m
* total AOIs: **115**
* split: **69 train / 23 val / 23 test**

### Watercourse-enabled subset

Where source coverage of watercourse layers is complete:

* usable AOIs: **112**
* split: **67 train / 23 val / 22 test**
* explicitly excluded (incomplete source coverage):
  `small_aoi_00_11`, `small_aoi_00_12`, `small_aoi_00_13`

The excluded AOIs **must not receive zero-valued watercourse features**;
they fall outside the common watercourse source extent and treating zeros
as evidence-of-absence would be a silent bias.

---

## 4. Current best GNN result (for honest reporting)

The strongest GNN result so far is **candidate-corridor classification**,
not exact topology reconstruction.

**Setup:**
* 2 km / 500 m AOIs
* hybrid candidate graph (road + building-access + demand-kNN)
* GraphSAGE, no absolute node coordinates
* all available non-anchor features
* 10 m label tolerance
* CUDA GTX 1080 Ti

**Test metrics:**

| Metric | Value |
| --- | --- |
| ROC AUC | **0.7799** |
| PR AUC | **0.7612** |
| F1 | **0.7797** |
| Precision | **0.7067** |
| Recall | **0.8696** |
| Length F1 | **0.7756** |

**Honest interpretation:** these numbers describe **classifier accuracy
on the candidate graph**. They do not measure recovery of sewer geometry
that falls outside the candidate support — that ceiling is the
representability metric in §5. Any future "improvement" claim must
disclose both the classifier metric and the candidate-graph
representability it is conditioned on.

---

## 5. Representability is the new headline metric

Before training any classifier, report the candidate graph's representability:

* **truth-length recall** at 5 / 10 / 20 / 30 / 50 m buffers
* **candidate count** (total prediction units)
* **candidate total length** (km)
* **recall per km of candidate support** (efficiency of the candidate generator)

This tells reviewers whether a candidate-graph variant honestly improves
support recall vs. just adding excessive candidate geometry. Higher
recall at the cost of 10× more candidates is not free.

### Where representability lives in code

* `pipe_network_completion/anchor_free/candidate_recall.py` — diagnostic
* `pipe_network_completion/anchor_free/candidate_variants.py` — A/B variants
* `scripts/evaluate_candidate_graph_recall.py` — CLI driver
* `scripts/evaluate_candidate_graph_variants.py` — CLI driver for variant sweeps

These were built by Codex and are the **canonical first-step diagnostic**
for any new experiment, regardless of model architecture.

---

## 6. Experiment plan (priority order)

The next experiments are not about tuning the GNN — they are about
finding the best candidate support and reporting representability before
classification.

### A. Candidate-support ablation (highest priority)

Compare these candidate supports head-to-head, reporting representability
at every tolerance:

1. **Road centerlines** (baseline)
2. **Hybrid road + building-demand** (current default)
3. **Road offset corridors** (±15 m, ±30 m bands)
4. **Watercourse / drainage corridor graph** (watercourse-complete AOIs only)
5. **Hybrid + watercourses**
6. **Multi-support** = hybrid + road offsets + watercourses

Output: a single table indexed by candidate support × buffer width, with
recall, candidate count, candidate length, recall/km columns. This is
the headline figure of the paper before any model number.

### B. Watercourse feature ablation

With the chosen candidate support fixed, ablate watercourses as
**feature** (distance to drainage, drainage length within buffer,
waterway corridor overlap):

* off: pure non-watercourse features
* on: with watercourse features
* candidate-on (i.e., watercourses as candidate edges) + feature-on

Run on the 112-AOI watercourse-complete subset only.

### C. AOI-block split as the only headline split

Random within-AOI splits are demoted to a **secondary diagnostic**, not a
headline number. Every reported metric pairs (model, AOI-block split,
candidate support, tolerance buffer).

### D. No-coordinate GNN as default

`include_node_coords=False` on the `Intersection` (or generalised
`CandidateCorridor` endpoint) node features is the default for headline
GNN runs. A coordinate-on run can be reported as a transductive
location-aware sensitivity row, clearly labelled.

### E. Prevalence baselines always visible

`all_positive_f1`, `all_positive_pr_auc`, `majority_class_f1` printed
beside every model row. Already wired (Stage 1 of the audit-followup
plan).

### What is deferred (and why)

| Item | Why deferred |
| --- | --- |
| Sewer pseudo-outlet / hydraulic-aware decoder | Cannot attribute lift until candidate-support choice is stable. |
| Water loop-budget decoder | Out of scope; project is sewer-focused. |
| Barrier penalties (rivers, rail, bridges) | Only if reliable barrier layers are available; poor layers add noise. |
| Stratified / random-edge split as headline | Demoted to secondary diagnostic per §6.C. |

---

## 7. Sparse anchor extension — explicitly separate setting

Sparse anchors are **not part of the anchor-free headline**. They are a
separate reportable setting with its own ablation matrix:

1. **anchor-free baseline** (this project's main claim)
2. **sparse-field-observation conditioning** (a few anchors at known sites)
3. **oracle sparse-anchor ablation** (upper-bound feasibility)
4. **anchor-based reference** (the ISARC 2024 setting, for comparison)

Sparse anchors, if added, can act as:
* distance/count features
* seed nodes in the candidate graph
* decoder constraints
* an active-learning / site-visit budget proxy

But they must be clearly labelled and must not pollute the anchor-free
rows of any comparison table.

---

## 8. Paper claim — wording to use and to avoid

### Use

> We formulate anchor-free sewer prediction as a **change-of-support
> GeoAI problem** and show that visible urban supports can recover broad
> sewer corridor support, while strict centerline recovery remains
> limited without anchors or richer support layers.

### Avoid

* "We reconstruct exact sewer topology without anchors."
* "Legal utility locating."
* "Excavation clearance."
* "Validated hydraulic design."
* "Complete pipe-network reconstruction."
* "A sewer/water decoder with real engineering constraints" (unless one
  is actually implemented, tested, and validated against truth — none is
  yet).

---

## 9. Research safety rules (unchanged, restated)

Forbidden inputs to any anchor-free model:

* manholes
* valves
* utility poles
* transformers / cabinets as dense anchors
* surveyed utility junctions
* pipe junctions
* endpoints or nodes derived from true utility lines
* any feature computed from utility truth geometry

Allowed inputs:

* road centerlines, road offsets
* building footprints, building points
* DEM / slope
* watercourses / drainage centrelines (where source coverage is complete)
* built-up / land-use / parcels / zoning
* service-area boundary
* (separate setting) sparse field observations — see §7

Utility truth lines are **labels and evaluation only**. The candidate
graph builder must not read the utility truth path. The labeller may read
truth only after candidate edges already exist.

Stormwater and easement layers are useful but should be **separate
disclosed ablations**, not silent default inputs.

---

## 10. What this means for code I just landed

The Phase 1 / 1.5 / 2.A / 2.B work survives mostly intact, with three
naming/scoping adjustments before the next code change lands:

| Piece | Status after framing change |
| --- | --- |
| `make_buffer_invariant_splits` (Stage 1) | Still used **within** an AOI. AOI-block split is the new outer layer. |
| Prevalence baselines + Δ-column CLI | Unchanged; still required. |
| `assert_no_anchor_features` guard | Unchanged; still required. |
| Decoder reserved-name `NotImplementedError` | Unchanged; still required. |
| Multi-CRS truth reprojection | Unchanged. |
| Inference-only vs evaluation GeoJSON | Unchanged. |
| Heterogeneous (`RoadSegment` + `Intersection`) graph | **Generalise** the node-type name to `CandidateCorridor` so road, road-offset, building-access, demand-kNN, demand-MST, and watercourse edges all live under one node type with `candidate_source` as a one-hot. The `Intersection` node type stays. The PyG `to_hetero` lift continues to work. |
| Clipped-length local-road density | **Generalise** to clipped-length candidate-corridor density (the legacy "road" naming is misleading once the candidate graph holds non-road edges). |
| Stage 2 ablation matrix | **Replaced** by the candidate-support ablation in §6.A. The 7-variant matrix becomes a sub-matrix at one chosen candidate support. |
| Buffer-invariant split as headline | **Demoted** to within-AOI secondary diagnostic. |
| `inductive_split_plan.md`'s "within-network completion" framing | **Superseded** for the headline claim; AOI-block split is the headline. The within-network framing remains valid for a sub-experiment. |
| `audit_followup_implementation_plan.md` Stages 3–5 | Reordered: Stage 3 (spatial sensitivity) becomes Stage 0 (the headline). Stage 4 (threshold tuning) and Stage 5 (decoders) follow. |
| `architectural_cleanup_plan.md` Phases 2.D / 2.E / 2.F | Still apply, but only after §6.A representability + classifier ablations land. |
| `train_anchor_free_brisbane.py` (whole-Brisbane CLI) | **Demoted** to "convenience for one-off whole-area inspection". The canonical training entry point is `scripts/train_anchor_free_aoi_blocks.py`. |

---

## 11. Definition of done for the next research milestone

The next milestone lands when:

1. `outputs/candidate_support_ablation/representability_table.csv` exists
   with one row per (candidate support, buffer width) pair and complete
   recall / count / length / recall-per-km columns, on the 115-AOI set.
2. The same table on the 112-AOI watercourse-complete subset, plus
   watercourse-as-candidate rows.
3. A `paper_table_1_representability.md` summary in `docs/results/`
   transcribing the headline rows with one-paragraph interpretation.
4. The classifier-side ablation (RF / GraphSAGE × candidate supports ×
   with/without watercourse features) is run on the best candidate
   support from #1, with the AOI-block split, no-coord GNN as the
   default, and prevalence baselines beside every row.
5. A status banner on this document moves from CANONICAL → LANDED with
   the commit reference.

No training run is started without an explicit owner "go" per the
existing checkpoint discipline.
