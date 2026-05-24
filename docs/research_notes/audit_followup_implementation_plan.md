# Implementation Plan: Audit-Followup Hardening for Anchor-Free Pipeline

<!-- Workstream: Claude -->

**Status (2026-05-23, fourth update — research direction shifted):**
> Superseded as the canonical experiment plan by
> [`project_framing_v2.md`](project_framing_v2.md). The Stage 1 hygiene
> items (ISARC-seeded shared split, prevalence baselines, CLI Δ-column,
> Brier baselines, dtype polish, audit reconciliation) have **landed**
> and remain canonical. Stage 2 (Codex's 7-variant ablation) is
> **replaced** by the candidate-support ablation in
> `project_framing_v2.md` §6.A; Stage 3 (spatial-block sensitivity) is
> **promoted** to the headline split via AOI-block evaluation; Stages 4
> (threshold diagnostic) and 5 (simple decoders) still apply but only
> after the candidate-support choice is fixed.

**Status:** PROPOSED 2026-05-23 — awaiting owner approval before any phase
lands or `CHANGELOG.md` is updated.

**Status (2026-05-23 update):** Phase A and Phase B revised after owner
review. Phase A now explicitly mirrors the ISARC paper's seeding
mechanism (`random.Random(seed)` with `seed=42`) rather than numpy's
`default_rng`. Phase B is split into a diagnostic step (B.1 — landable
now) and an optimisation-design step (B.2 — TO BE DESIGNED after B.1
results are reviewed). No threshold-tuning code lands until B.1 has
characterised the threshold sensitivity surface.

**Status (2026-05-23 second update, after Codex review):** Codex's
`research_reward_upgrade_recommendation_codex.md` reframes priorities:
the strongest research contribution is the **ablation matrix** (which
non-anchor sources matter), not algorithmic tricks. Threshold tuning is
hygiene, not contribution. The original A–G phase list is now grouped
into Codex's 5-stage execution order (§16, new). Each stage ends with
an explicit owner-review checkpoint before the next stage starts; this
plan does not commit to running stages 2–5 without those checkpoints.
Stage 2 adds **one new ablation row** (GNN with vs without absolute x,y
coords) that Codex flagged as a memorisation-shortcut test.

**Author:** Claude
**Reviewer:** TBD (project lead)

**Sister docs:**
- [`inductive_split_plan.md`](inductive_split_plan.md) — framing decision
  (within-network completion, not generalisation). Phase G in this plan keeps
  that decision visible in the leakage audit.
- [`../anchor_free_leakage_audit_codex.md`](../anchor_free_leakage_audit_codex.md) —
  Codex's audit; this plan implements the unresolved recommendations.
- [`../anchor_free_engineering_decoder_note_codex.md`](../anchor_free_engineering_decoder_note_codex.md) —
  decoder design; explicitly **not** in scope for this plan (defer until the
  baseline numbers below are clean).
- [`../anchor_free_design.md`](../anchor_free_design.md) — top-level design.

**Aligns with:**
- Claude's review of Codex's audit (2026-05-23) — the six prioritised
  suggestions captured at the end of that review.
- The "What landed" status table in that same review (M1, prevalence
  baseline, CLI `test_*` already done).

**NOT in scope here:**
- Spatial-block / inductive component split (deferred per
  `inductive_split_plan.md`).
- Inductive message passing (`inductive_message_passing` flag on
  `train_road_edge_gnn`) — also deferred per the same note.
- Engineering decoder (pseudo-outlets, sewer-directed cost terms) — defer
  until phases A–C below produce stable baseline numbers.

---

## 1. Why phase these together

Six audit-followup items remain after Codex's recent round. They share two
operational invariants:

1. **Reportable metrics must be apples-to-apples across label buffers and
   thresholds.** Without this, ablation tables are not trustable.
2. **The trivial baselines must be visible everywhere a model number is
   visible**, so a small lift over the prevalence ceiling cannot be mistaken
   for a strong result.

Phase A and B fix the apples-to-apples problem (split sharing + threshold
tuning). Phase C and E make the trivial baseline visible in CLI and JSON.
Phase D guards a feature-engineering boundary effect introduced by the new
bbox clip. Phase F is polish. Phase G is a documentation reconciliation.

Effort calibration: this whole plan is ~1.5 days of focused work. Phases
A–C are the headline-moving ones and fit in a single PR (~3–4 hours);
D–G are independent follow-ups.

---

## 2. Phase summary

| Phase | Goal | Headline effect | Effort | Depends on |
|-------|------|-----------------|--------|------------|
| **A** — Shared, ISARC-seeded split across label buffers | Make 10m / 5m results directly comparable; mirror ISARC seeding | makes ablation table valid; aligns with paper | S | none |
| **B.1** — Threshold-sensitivity diagnostic (read-only script) | *Characterise* the threshold surface before designing a tuner | reporting only, no metric change | S | A landed (so val/test are stable) |
| **B.2** — Threshold-tuning optimisation design + impl | Pick a tuner that fits what B.1 reveals | TBD after B.1 | TBD | **GATED on B.1 results + owner approval** |
| **C** — Surface prevalence in CLI | Stop misreading marginal lifts as wins | reporting only, no metric change | XS | none |
| **D** — Bbox-clip bias guard | Prevent silent distance-feature corruption | risk-mitigation, no expected metric change | S | none |
| **E** — Brier-score baselines | Complete the trivial-baseline coverage | reporting only | XS | C (same file/touch) |
| **F** — Metric-dtype polish | `majority_class_label` int, docstrings | none | XS | C or E |
| **G** — Reconcile audit with framing decision | Stop the audit reading as "spatial-block required" | none, removes reviewer confusion | XS | none |

Effort key: XS = <1h, S = 1–4h, M = 4–8h, L = full day, XL = >1 day.

---

## 3. Acceptance-rule scheme

AR-IDs follow `AR-AF-{phase}.{n}` where `AF` = anchor-free.

- `P` rules are **code/contract changes** (something a `pytest -q` or `grep`
  can verify).
- `T` rules are **test additions** (new tests must exist and pass).
- `D` rules are **documentation/reporting** changes.

Every AR has an **acceptance signal**: a concrete way to verify it on the
post-merge commit.

---

## 4. Phase A — Shared split across label buffers

Owns the M6 fix from
[`anchor_free_leakage_audit_codex.md`](../anchor_free_leakage_audit_codex.md)
M6. The fix is split logically into a *diagnosis* step (A.1) and a *contract*
step (A.2): we first verify whether the current
`prepare_anchor_free_features.py` re-stratifies per buffer (if not, phase A
collapses to just A.2 documentation), then commit the contract that future
edits can't regress it.

### A.1 ISARC-seeded reference mechanism

The ISARC paper's split discipline ([`process.py:33-46`](../../process.py),
[`process.py:188`](../../process.py)) is:

```python
def split_list_by_ratio(lst, ratio, seed: int):
    items = list(lst)
    random.Random(seed).shuffle(items)        # Python stdlib RNG
    ...

d_train, d_val, d_test = split_list_by_ratio(
    components, (6, 2, 2), seed=seed,         # default seed=42
)
```

Two paper-published commitments to preserve:

1. **Seed value `42`** — the default in `process.py:272` and the value used
   to produce every metrics row in `results/metrics/model_metrics1212.csv`.
2. **RNG mechanism `random.Random(seed).shuffle(...)`** — Python's stdlib
   Mersenne-Twister-derived RNG, not numpy's PCG64. The two RNGs produce
   different orderings even with the same integer seed.

The current anchor-free split
([`baseline.py:43`](../../pipe_network_completion/anchor_free/baseline.py))
uses `np.random.default_rng(seed)`, so it is **not** bit-for-bit
ISARC-comparable even when `seed=42`. The buffer-invariant split designed
below explicitly mirrors ISARC's mechanism so a future "matched-seed"
comparison row is meaningful.

### A.2 Goal — diagnose the current behaviour (no code change)

**Verify** whether the current implementation already shares the split
across buffers in `training_ready_label_buffers_m: [10, 5]`. The diagnostic
is a 30-second grep + a small repro that loads both training tables and
diffs their `edge_id → split` mapping.

### A.3 Goal — commit the shared, ISARC-seeded split contract

Add a `make_buffer_invariant_splits(...)` helper that:

- Derives the split from a deterministic hash of `(seed, edge_id)` only,
  not from `y`, so the assignment is identical across label buffers.
- Uses **`random.Random(seed)` (Python stdlib), not `np.random.default_rng`**
  — matching ISARC's `split_list_by_ratio` mechanism. The intent is
  identical seeding discipline to the published paper, even though the
  splitting *unit* differs (anchor-free splits edges; ISARC splits
  components).
- Defaults to `seed=42` to match the ISARC publication seed.

`prepare_anchor_free_features.py` then calls it once per dataset and
reuses the result for every buffer in `training_ready_label_buffers_m`.

### A.4 Acceptance rules

| AR-ID | Rule | Acceptance signal |
|-------|------|-------------------|
| **AR-AF-A.1** | A helper `make_buffer_invariant_splits(edge_ids, *, seed=42, train_fraction=0.6, val_fraction=0.2)` exists in `pipe_network_completion.anchor_free.baseline`. It computes the split from a deterministic hash of `(seed, edge_id)` so the split assignment is identical for any two label vectors over the same `edge_ids`. | `grep -n make_buffer_invariant_splits` returns the definition + at least one test reference. |
| **AR-AF-A.2** | The helper uses **`random.Random(seed)` (Python stdlib)**, mirroring ISARC's `process.py::split_list_by_ratio`. Implementation is documented in a docstring that cites `process.py:33` and explains the bit-for-bit-seeding intent. | Source file contains the docstring + the `random.Random(seed)` call; CI grep matches both. |
| **AR-AF-A.3** | `prepare_anchor_free_features.py` calls `make_buffer_invariant_splits` once and reuses the result for every buffer in `training_ready_label_buffers_m`. | `grep -c make_stratified_edge_splits scripts/prepare_anchor_free_features.py` returns `0` after the change. |
| **AR-AF-A.4** | Two output training tables for `[10, 5]` have **identical `edge_id` → `split` mapping** for every shared edge. | Recurrence test: load both tables, assert `assert_frame_equal(t10[["edge_id","split"]], t5[["edge_id","split"]])`. |
| **AR-AF-A.5** | Backward compat: `make_stratified_edge_splits` is kept as-is so the existing `outputs/anchor_free_brisbane_*` runs are bit-for-bit reproducible. | Existing tests for `make_stratified_edge_splits` continue to pass; no signature change. |
| **AR-AF-A.6** | A frozen reference test fixture records the ISARC-seeded split of a 1000-edge synthetic graph at `seed=42`. The fixture is regenerated only if the seeding mechanism is intentionally changed (which would constitute a paper-relevant change and requires owner approval). | `tests/fixtures/buffer_invariant_split_seed42.json` exists and is loaded by `tests/test_anchor_free_buffer_invariant_split.py::test_seed42_split_is_frozen`. |
| **AR-AF-T.A** | New test file `tests/test_anchor_free_buffer_invariant_split.py` covers AR-AF-A.1 (existence + hash determinism), AR-AF-A.4 (cross-buffer equivalence), and AR-AF-A.6 (frozen reference). | `pytest tests/test_anchor_free_buffer_invariant_split.py -q` reports 3+ passing tests. |

### A.5 Blast radius

- `pipe_network_completion/anchor_free/baseline.py` (+1 function, ~30 LOC
  including the ISARC-citing docstring)
- `scripts/prepare_anchor_free_features.py` (~3-line call swap)
- `tests/test_anchor_free_buffer_invariant_split.py` (new, ~60 LOC)
- `tests/fixtures/buffer_invariant_split_seed42.json` (new, ~10 KB)

### A.6 Decision branches

- **If A.2 grep shows the split is already shared** (e.g. Codex already
  refactored), reduce phase A to: rename helper for clarity, add the
  ISARC-seeded RNG swap, add the recurrence + frozen-fixture tests, update
  the contract docstring. No metric impact unless the RNG swap reshuffles
  the existing splits (it will, deterministically).
- **If the split is per-buffer**: full implementation; **expect small
  metric shifts on the 10m runs** when the canonical split is adopted,
  because (a) the exact edges in each fold will change and (b) the RNG
  swap from numpy → stdlib reshuffles them.

---

## 5. Phase B — Threshold tuning (split: diagnose first, then design)

The pipeline currently hardcodes `decoder.threshold: 0.5`. With test F1
already at the all-positive prevalence ceiling
(`all_positive_f1 ≈ 0.789` at 65% positive rate vs test F1 ≈ 0.79), the
threshold is the dominant variable a model can change.

**Revised approach (owner direction, 2026-05-23):** Do **not** jump
straight to "val-F1 argmax over `threshold_grid`". That would lock in an
optimisation method before we know what the threshold-sensitivity surface
actually looks like on this data. Split Phase B into:

- **B.1 — Diagnostic test (landable now).** A one-shot study script that
  scans a dense threshold grid on the already-trained models, plots
  precision / recall / F1 / IoU / Brier on both val and test, dumps a
  CSV + PNG per experiment. **No pipeline change.** Status: PROPOSED.
- **B.2 — Optimisation method design (TO BE DESIGNED after B.1).** Based
  on what B.1 reveals (single peak, plateau, multi-modal, val-test
  decorrelation, etc.), propose a specific optimisation method and a
  separate implementation plan. Status: GATED on B.1 results.

### B.1 — Diagnostic test

#### B.1.a Goal

Produce a per-experiment **threshold sensitivity report** that lets us
*see* whether tuning will help, and if so by how much, before we commit
to any tuning mechanism.

#### B.1.b What the report contains

For each completed Brisbane experiment
(`outputs/anchor_free_brisbane_*`), the script reads
`edge_predictions.geojson` (already on disk; no retraining) and produces:

1. **Threshold grid scan** at 51 points `[0.00, 0.02, ..., 1.00]` for both
   the val and test splits. Recorded metrics per threshold:
   `precision`, `recall`, `f1`, `iou_jaccard`, `balanced_accuracy`,
   plus length-weighted versions (`length_precision`, `length_recall`,
   `length_f1`).
2. **Two curves per plot**: val and test side-by-side, so any
   val-test decorrelation is immediately visible. If val-best
   threshold ≠ test-best threshold by more than 0.05, the val-set is
   a misleading proxy and naïve val-argmax will under-perform.
3. **Comparison to baselines**: `all_positive_f1` line and
   `majority_class_f1` line drawn as horizontal references.
4. **Summary CSV row** with: `val_best_threshold`, `val_best_f1`,
   `test_best_threshold`, `test_best_f1`, `f1_at_0.5`,
   `f1_gain_at_val_best = test_f1_at_val_best - f1_at_0.5`,
   `f1_gain_at_test_best = test_best_f1 - f1_at_0.5`.

The headline question B.1 must answer:

> Is `f1_gain_at_val_best` consistently positive across experiments, and
> how close to `f1_gain_at_test_best` is it?

- If yes and close → simple val-argmax (the original Phase B proposal) is
  enough; B.2 can land as the original AR-AF-B.x rules.
- If yes but `f1_gain_at_val_best ≪ f1_gain_at_test_best` → there is
  val-test decorrelation; B.2 needs cross-validated tuning or a different
  proxy.
- If `f1_gain_at_val_best` is near zero or negative → the model is
  saturated at the prevalence ceiling and the right research move is
  features or labels, not thresholds. B.2 collapses to a one-line "we
  recommend not tuning thresholds; report at fixed 0.5".

#### B.1.c Acceptance rules

| AR-ID | Rule | Acceptance signal |
|-------|------|-------------------|
| **AR-AF-B.1.1** | New script `scripts/diagnose_threshold_sensitivity.py` reads `edge_predictions.geojson` from a list of experiment dirs and writes one report per experiment under `<experiment>/threshold_diagnostic/`. | Script file exists; `python scripts/diagnose_threshold_sensitivity.py --help` returns non-zero usage text. |
| **AR-AF-B.1.2** | Per-experiment outputs: `threshold_curve.csv` (51 rows × ≥ 9 columns), `threshold_curve.png` (val + test + baseline lines), `summary.json` containing the 6 summary fields listed in §B.1.b. | All four files present after running the script on any existing experiment dir. |
| **AR-AF-B.1.3** | A top-level `outputs/threshold_diagnostic_summary.csv` aggregates the per-experiment `summary.json` rows so the headline question can be answered with a single table. | File exists; one row per scanned experiment; finite values in `f1_gain_at_val_best`. |
| **AR-AF-B.1.4** | No change to `pipeline.py`, `decoder.py`, or `evaluation.py`. The script is read-only with respect to the training-time codepath. | `git diff --stat HEAD~1 -- pipe_network_completion/anchor_free/` shows zero changes in those files for the B.1 commit. |
| **AR-AF-T.B.1** | New test `tests/test_diagnose_threshold_sensitivity.py` runs the script against the synthetic fixture's outputs and asserts the four output files exist and `summary.json` has all six expected fields. | 1+ test passes; runs in < 5 s. |

#### B.1.d Blast radius

- `scripts/diagnose_threshold_sensitivity.py` (new, ~150 LOC)
- `tests/test_diagnose_threshold_sensitivity.py` (new, ~40 LOC)
- Zero changes to `pipeline.py` / `decoder.py` / `evaluation.py` / `config.py`.

#### B.1.e Expected output (illustrative)

```
outputs/threshold_diagnostic_summary.csv:
experiment                                  val_best_thr  val_best_f1  test_best_thr  test_best_f1  f1_at_0.5  f1_gain_at_val_best  f1_gain_at_test_best
anchor_free_brisbane_road_only              0.46          0.815        0.48           0.812         0.791       0.021                0.021
anchor_free_brisbane_road_buildings_*       0.44          0.832        0.46           0.829         0.787       0.042                0.042
anchor_free_brisbane_full_5m                0.55          0.612        0.52           0.609         0.550       0.059                0.059
```

This is a hypothetical; the actual numbers come from running B.1. The
shape of these rows determines what B.2 looks like.

### B.2 — Optimisation method design (TO BE DESIGNED)

Status: **GATED** on B.1 results.

Once B.1 lands and the summary CSV is reviewed with the owner, a follow-up
note `docs/research_notes/threshold_tuning_design.md` will be written
proposing the specific optimisation method. The candidate methods to
consider in that note, ranked by complexity:

1. **Val-F1 argmax over `evaluation.threshold_grid`** — the original Phase
   B proposal; appropriate if val-test threshold agreement is tight.
2. **Val-F1 argmax on a dense grid (resolution 0.01) + smoothing** — same
   thing but at higher resolution and with a 5-point moving-average to
   avoid val-noise lock-in.
3. **F-β with β tuned to operational cost ratio** — appropriate if
   precision and recall have asymmetric downstream cost (e.g. a wrongly
   excavated road costs more than a missed sewer).
4. **Probability calibration + decision-theory threshold** — fit Platt or
   isotonic on val probabilities, then choose the threshold that
   minimises expected loss under a stated cost matrix. Appropriate if
   probabilities are mis-calibrated and the cost matrix is meaningful.
5. **Cross-validated tuning** — k-fold CV over the train+val pool,
   appropriate if B.1 shows val-test decorrelation.
6. **Per-region thresholds** — appropriate if B.1 plots show thresholds
   that vary by geographic region.

The design note will pick one (or two, for an ablation) and turn into a
proper Phase B.2 implementation plan with its own AR-IDs.

**Hard rule:** no threshold-tuning code lands in `pipeline.py` until B.2
is approved.

---

## 6. Phase C — Surface prevalence baselines in CLI summary

### C.1 Goal

When the user runs `python scripts/train_anchor_free.py` or
`python scripts/train_anchor_free_brisbane.py`, the headline metric block
printed at the end must include the prevalence baselines that
`compute_prevalence_baseline_metrics` already computes and stores in
`metrics.json`.

### C.2 Acceptance rules

| AR-ID | Rule | Acceptance signal |
|-------|------|-------------------|
| **AR-AF-C.1** | The CLI summary in both top-level training scripts prints, in this order: `test_positive_prevalence`, `test_all_positive_f1`, `test_all_positive_pr_auc`, then the model's `test_f1`, `test_pr_auc`, `test_roc_auc`. | `grep -E "(positive_prevalence\|all_positive_f1)" scripts/train_anchor_free*.py` returns matches. |
| **AR-AF-C.2** | The printed block visually pairs model vs baseline (e.g. two-column "model / all-positive" or a `Δ` column). | Manual inspection of the printed block on a synthetic run. |
| **AR-AF-T.C** | A smoke test asserts that the script's printed stdout contains both `positive_prevalence` and `all_positive_f1` substrings on the synthetic fixture. | New `tests/test_cli_prints_prevalence.py` runs `subprocess.run` on the script, captures stdout, asserts substrings. |

### C.3 Blast radius

- `scripts/train_anchor_free.py` (small edit to the summary printer)
- `scripts/train_anchor_free_brisbane.py` (same)
- `tests/test_cli_prints_prevalence.py` (new, ~25 LOC)

### C.4 Expected output (illustrative)

```
Test-set metrics:
                       metric         model     all-positive    Δ
              positive_prevalence       —          0.6524       —
                         roc_auc     0.7706         0.5000   +0.2706
                          pr_auc     0.8171         0.6524   +0.1647
                              f1     0.7870         0.7895   -0.0025
```

The `Δ` column for F1 in the example above is what makes Phase B
(threshold tuning) worth doing — at the default threshold the model is
*below* the trivial all-positive predictor on F1.

---

## 7. Phase D — Bbox-clip distance-feature bias guard

`scripts/filter_context_to_study_area.py` clips buildings/built-up/building
points to the road bbox. Distance features computed for road edges near
the bbox boundary now systematically over-estimate distance-to-nearest-building
because real buildings just outside the bbox aren't loaded.

### D.1 Goal

Two guardrails:
1. The clip buffer used by `filter_context_to_study_area.py` must be at
   least `max(feature_search_radius)` for every distance/density feature
   that the pipeline computes.
2. A small audit script reports the fraction of road edges whose
   `nearest_building_distance_m` equals the bbox-clip fallback value
   (currently `building_buffer_m * 10`), so we can see when the bias
   bites.

### D.2 Acceptance rules

| AR-ID | Rule | Acceptance signal |
|-------|------|-------------------|
| **AR-AF-D.1** | `filter_context_to_study_area.py` validates that its `--buffer-m` argument is `>= max(building_buffer_m, building_point_buffer_m, built_up_buffer_m, road_density_buffer_m, dem_sample_spacing_m * dem_max_samples_per_edge)` derived from the resolved config. Below that minimum the script raises a clear error. | Recurrence test: call the script with a too-small buffer and assert `SystemExit` + an explanatory message. |
| **AR-AF-D.2** | New CLI flag `--audit-fallback-fractions` on `prepare_anchor_free_features.py` writes `outputs/<exp>/feature_fallback_audit.json` reporting per-feature: fraction of rows at the fallback value, mean / median distance, mean distance for edges within `0.2 * bbox_width` of the bbox edge. | File exists post-run; values are finite. |
| **AR-AF-T.D** | New test `tests/test_filter_context_buffer_validation.py` covers AR-AF-D.1. | 1+ test passes. |

### D.3 Blast radius

- `scripts/filter_context_to_study_area.py` (validation + clearer error,
  ~15 LOC)
- `scripts/prepare_anchor_free_features.py` (audit flag, ~30 LOC)
- `tests/test_filter_context_buffer_validation.py` (new, ~30 LOC)

### D.4 Risk

If the recent Brisbane runs used too small a buffer in the filter step,
their distance features are already biased. The audit script (D.2) will
flag it; if the fallback fraction is small (<5%), Phase D is just an
insurance policy. If it's large (>20%), we re-run the affected
experiments.

---

## 8. Phase E — Brier-score baseline + small metric additions

### E.1 Goal

Round out the trivial-baseline coverage so every reported model metric has
a published baseline value:

- `all_positive_brier_score = (1 - p)` for an all-positive predictor.
- `random_brier_score = p(1 - p)` for a calibrated random predictor.
- `all_positive_balanced_accuracy = 0.5`.

### E.2 Acceptance rules

| AR-ID | Rule | Acceptance signal |
|-------|------|-------------------|
| **AR-AF-E.1** | `compute_prevalence_baseline_metrics` returns the three new keys above in addition to its existing keys. | `grep -n all_positive_brier_score pipe_network_completion/anchor_free/evaluation.py` returns the dict literal. |
| **AR-AF-T.E** | The existing `tests/test_anchor_free_metrics.py::test_compute_edge_metrics_reports_prevalence_baseline` is extended to cover the three new keys with closed-form expected values. | Existing test still passes; new assertions added. |

### E.3 Blast radius

- `pipe_network_completion/anchor_free/evaluation.py` (~10 LOC)
- `tests/test_anchor_free_metrics.py` (~10 LOC)

---

## 9. Phase F — Metric-dtype polish

Cosmetic, ride along with E.

### F.1 Acceptance rules

| AR-ID | Rule | Acceptance signal |
|-------|------|-------------------|
| **AR-AF-F.1** | `majority_class_label` in `metrics.json` is stored as `int` (0 or 1), not `float`. | `python -c "import json; print(type(json.load(open('outputs/anchor_free_brisbane_full_5m/anchor_free/metrics.json'))['test_majority_class_label']))"` returns `<class 'int'>`. |
| **AR-AF-F.2** | `compute_prevalence_baseline_metrics` docstring explains that `all_positive_f1` and `majority_class_f1` reach `1.0` when prevalence reaches `1.0`, which is mathematically correct but counterintuitive. | Docstring contains the sentence "when prevalence reaches 1.0". |

### F.2 Blast radius

- `pipe_network_completion/anchor_free/evaluation.py` (~3 LOC)

---

## 10. Phase G — Reconcile leakage audit with framing decision

The audit document
[`anchor_free_leakage_audit_codex.md`](../anchor_free_leakage_audit_codex.md)
opens by recommending "Add a spatial block split and make it the headline
result" (H1) and "create spatial subgraphs where test-area roads are not
used during training message passing" (H2). The framing decision recorded
in [`inductive_split_plan.md`](inductive_split_plan.md) explicitly downgrades
both to optional-only. Without a reconciliation paragraph, a future reader
of the audit will think the project is non-compliant.

### G.1 Acceptance rules

| AR-ID | Rule | Acceptance signal |
|-------|------|-------------------|
| **AR-AF-G.1** | A "Framing context (2026-05-23 update)" block is added to the **top** of `anchor_free_leakage_audit_codex.md` that links to `inductive_split_plan.md` and explicitly downgrades H1 and H2 from "Required fix" to "Deferred — see linked note". | Block exists at top of file; markdown links resolve. |
| **AR-AF-G.2** | The "Required fixes before paper-quality claims" list in the audit removes items #1 and #2 (spatial-block split, inductive subgraphs) and replaces them with a single line "see `inductive_split_plan.md` for the framing decision". | Numbered list has the changed line. |
| **AR-AF-G.3** | `inductive_split_plan.md` adds a back-link to the audit so the relationship is bidirectional. | Both files contain a relative link to each other. |

### G.2 Blast radius

- `docs/anchor_free_leakage_audit_codex.md` (~20 LOC of additions)
- `docs/research_notes/inductive_split_plan.md` (~5 LOC for the back-link)

---

## 11. Sequencing recommendation (revised)

```
                                  ┌─ C (CLI prevalence)
A ──→ B.1 (diagnostic) ──→ owner review ──→ B.2 (TO BE DESIGNED)
                                  └─ E (Brier baselines) ──→ F (polish)

D (bbox audit) — independent
G (audit reconciliation) — independent, no code
```

- **First PR**: A + C + E + F. Tight metric/split cluster, no tuning.
  Estimated ~2–3 hours including tests.
- **Second PR**: B.1 alone. Diagnostic-only script; produces the
  threshold-sensitivity summary that B.2 will be designed against.
  Estimated ~2 hours.
- **Owner review milestone**: after B.1 results land in
  `outputs/threshold_diagnostic_summary.csv`, review with the owner and
  pick a B.2 candidate method from the list in §5 (B.2). Write a separate
  design note `docs/research_notes/threshold_tuning_design.md` with its own
  AR-IDs.
- **Third PR**: B.2 implementation, only after the design note is approved.
- **Fourth PR**: D. Independent, can land any time.
- **Fifth PR** (doc-only): G. Can be done in 15 minutes.

After A + B.1 + C + E + F + G have landed:
- Re-run the three Brisbane experiments and update the comparison table in
  `docs/anchor_free_design.md`. **Report at fixed threshold 0.5 until B.2
  lands.** Use the B.1 diagnostic summary as a supplementary table.
- **Defer** the engineering decoder note (Codex's
  `anchor_free_engineering_decoder_note_codex.md`) until B.2 lands.
  Without a tuned threshold baseline, the decoder's "improvement over
  threshold-only" claim is not attributable.

---

## 12. Out of scope (and why)

| Item | Why deferred | Re-open trigger |
|------|--------------|-----------------|
| Spatial-block / inductive component split | Framing decision in `inductive_split_plan.md` — "within-network completion" doesn't require it. | Reviewer asks for cross-region numbers. |
| `inductive_message_passing: bool` on `train_road_edge_gnn` | Same framing decision; transductive message passing is acceptable under within-network-completion. | Reviewer flags H2 explicitly. |
| Engineering decoder (pseudo-outlets, sewer-directed cost) | Cannot attribute decoder lift until baseline threshold is tuned (Phase B). | Phase B lands and baseline numbers are stable. |
| Schema rewrite for `metrics.json` | Out of scope; current schema is additive-friendly. | Number of metric keys exceeds ~80, then split into nested dicts. |
| Renaming `edge_predictions.geojson` to inference-only vs eval-only | Codex audit M3 — useful but cosmetic; defer until paper figure work begins. | Paper figure preparation. |

---

## 13. Definition of done (whole plan)

This plan is done when:

1. All AR-AF-{A..G}.x rules above have their acceptance signals satisfied.
2. `pytest -q` is green; coverage on `pipe_network_completion/anchor_free/`
   is `>= 90%` (a soft target; current coverage is ~88% per the last full
   suite run — Phase A and B add tests that should bump it).
3. A regenerated `outputs/anchor_free_brisbane_full_5m/anchor_free/metrics.json`
   contains `tuned_threshold`, `all_positive_brier_score`, and prints in
   the CLI summary the prevalence comparison block.
4. The leakage audit (`anchor_free_leakage_audit_codex.md`) opens with the
   reconciliation block; readers can follow the link to
   `inductive_split_plan.md` for the framing context.

---

## 14. Status banners (to be added per phase as work lands)

To match the GPR-Simulationv3 reporting style, each phase header gets a
status line in the form:

```
**Status (YYYY-MM-DD):** PROPOSED | IN PROGRESS | LANDED in <commit> | REJECTED <reason>
```

When a phase LANDS, the AR table for that phase moves to a strikethrough
form (or is mirrored into a `CHANGELOG.md` entry), and the headline
"Status" banner at the very top of this document gets updated.

---

## 16. Stage execution order (Codex-aligned, 2026-05-23)

Codex's review (see
[`research_reward_upgrade_recommendation_codex.md`](research_reward_upgrade_recommendation_codex.md))
recommends grouping the AR-AF-* phases into five stages, each followed
by an **explicit owner-review checkpoint** before the next stage starts.
The AR-IDs are unchanged; only the grouping and the checkpoint gates
are new.

### Stage 1 — Research hygiene  ⟶  **CHECKPOINT 1**
Maps to: Phase A + Phase C + Phase E + Phase F + Phase G of this plan.
- AR-AF-A.1 through A.6, T.A
- AR-AF-C.1 through C.2, T.C
- AR-AF-E.1, T.E
- AR-AF-F.1, F.2
- AR-AF-G.1 through G.3

Deliverables: shared ISARC-seeded split, CLI prevalence display, Brier
baselines, dtype polish, audit reconciliation. Existing test suite must
stay green; new tests added per AR-T.x rules.

Checkpoint 1 review questions:
1. Do `train_anchor_free.py` and `train_anchor_free_brisbane.py` print a
   `model / all-positive / Δ` block with the new prevalence baselines?
2. Does the buffer-invariant split test pass deterministically?
3. Does the leakage audit open with the framing reconciliation block?
4. Have any Brisbane metric numbers shifted? If yes, explain why.

### Stage 2 — Core ablation matrix  ⟶  **CHECKPOINT 2**
New phase, replaces my earlier "rerun three Brisbane experiments". Adds
Codex's 7-variant matrix (one new ablation: GNN with/without x,y coords).

Variants:
1. Road-only RandomForest
2. Road + building-points RandomForest
3. Road + building-points + DEM RandomForest
4. Road-only GNN
5. Road + building-points GNN
6. GNN **without** node x,y (location-memorisation control)
7. GNN **with** node x,y (current baseline)

Run all seven on the same shared-seed split (from Stage 1), report:
ROC AUC, PR AUC, F1 beside `all_positive_f1`, length precision/recall/F1,
predicted total length, component count, train/test gap.

Stage 2 acceptance criteria are written as a separate AR-AF-S2 sub-plan
once Checkpoint 1 is approved.

Checkpoint 2 review questions:
1. Does the GNN beat RF? Or do classical features dominate?
2. Does dropping x,y from the GNN move metrics by more than the
   prevalence-baseline noise floor? (Answers Codex's L1 / memorisation
   concern.)
3. Are building-point and DEM additions net positive on test, not just
   train?

### Stage 3 — One spatial-block sensitivity run  ⟶  **CHECKPOINT 3**
Per Codex's "one spatial sensitivity experiment, not a pivot".
Reuses the deferred `spatial_block_split` design from
[`inductive_split_plan.md`](inductive_split_plan.md) for **one** model
(the best variant from Stage 2). Reported as a supplementary row in
the comparison table, not as a replacement for the headline.

Stage 3 acceptance criteria written once Checkpoint 2 is approved.

### Stage 4 — Threshold diagnostic (B.1 from §5)  ⟶  **CHECKPOINT 4**
AR-AF-B.1.1 through B.1.4 + T.B.1 above. Read-only diagnostic script.
At Checkpoint 4 the owner decides whether to commission B.2 (a separate
design note) or accept Codex's policy *"if val/test threshold curves
agree, use val-selected threshold; if they disagree, report fixed 0.5
and discuss instability"*.

### Stage 5 — Simple decoders only  ⟶  **FINAL CHECKPOINT**
If Stage 2 metrics are clearly above the prevalence ceiling and Stage 4
shows the threshold is stable, add:
- length-budget decoder
- (optional) connected decoder

**Defer** pseudo-outlet sewer decoding indefinitely per Codex's
recommendation. Re-open trigger: a reviewer asks for hydraulic
plausibility.

### Phase D (bbox-clip guard) — orthogonal to stages
Can land in any stage; not on the critical path.

---

## 17. Rejection criteria

This plan should be rejected (and rewritten) if any of:

- The shared-split contract in Phase A turns out to require regenerating
  the training tables from scratch in a way that invalidates the Brisbane
  metric history already reported in
  `outputs/anchor_free_brisbane_*/anchor_free/metrics.json`. (If so, Phase
  A needs a migration path, not just a code change.) Note: the RNG swap
  from `np.random.default_rng` to `random.Random` will deterministically
  reshuffle the splits even at the same `seed=42`; the existing metric
  history becomes one snapshot at the old RNG and Phase A produces a new
  snapshot at the ISARC-mirrored RNG. Both should be retained in the
  comparison table with explicit labels.
- Phase B.1's diagnostic shows that the val and test splits are correlated
  enough that val-tuned thresholds *do* generalise to test perfectly
  (`f1_gain_at_val_best ≈ f1_gain_at_test_best`) — in which case B.2
  collapses to the simple val-argmax originally proposed and the plan
  hasn't been wasted; we just gained a measurement before committing.
- Phase B.1's diagnostic shows that the model is saturated at the
  prevalence ceiling (`f1_gain_at_test_best < 0.01`). Then B.2 should not
  ship — the right research move is features or labels, not thresholds,
  and a new plan opens for that.
- The owner decides that the "within-network completion" framing should be
  revisited (e.g. for a journal version of the paper). Then this plan is
  superseded by the original `inductive_split_plan.md` implementation
  proposal.
