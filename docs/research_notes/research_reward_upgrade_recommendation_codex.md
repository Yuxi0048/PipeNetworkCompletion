# Research-Reward Upgrade Recommendation

Workstream: Codex

Date: 2026-05-23

Related notes:

- `docs/research_notes/audit_followup_implementation_plan.md`
- `docs/research_notes/inductive_split_plan.md`
- `docs/research_notes/anchor_free_upgrade_plan_codex.md`
- `docs/anchor_free_leakage_audit_codex.md`

## Short Recommendation

Claude's updated implementation plan is useful for hardening the anchor-free
pipeline. I recommend accepting the cleanup parts, but not letting the project
turn into a threshold-tuning or decoder-engineering exercise.

The most credible and research-rewarding story is:

```text
How much utility-corridor topology can be predicted without dense utility
anchor points, and which non-anchor information sources actually matter?
```

That means the main paper value is in clean evaluation and ablation design, not
in subtle post-processing tricks.

## What To Prioritize

### 1. Credible Evaluation Before More Algorithms

The first priority is to make reported numbers hard to misread.

Keep:

- shared split assignment for 10 m and 5 m labels;
- held-out `test_*` metrics in CLI and tables;
- positive prevalence and all-positive baselines beside F1;
- clear distinction between full-graph, validation, and test metrics.

Why this matters:

At the 10 m buffer, the positive rate is high. A trivial all-positive classifier
can get a strong-looking F1. If the model only matches that, the result is not
research-strong even if the number looks acceptable.

### 2. Core Ablations Are More Valuable Than Decoders

The most publishable table should focus on what information improves
anchor-free prediction.

Recommended main ablation:

1. Road-only RandomForest.
2. Road plus building points RandomForest.
3. Road plus building points plus DEM RandomForest.
4. Road-only GNN.
5. Road plus building points GNN.
6. GNN without road-node `x/y`.
7. GNN with road-node `x/y`.

This answers the actual research question:

- Are roads alone informative?
- Do building points add signal beyond road geometry?
- Does DEM help, or is it weak/noisy in this sewer setting?
- Does GNN message passing add value over classical edge features?
- Are coordinate features acting as useful context or location memorization?

If RandomForest beats GNN, that is still a valid and interesting result. It
would suggest that local road/context features dominate graph message passing
for this dataset.

### 3. Add One Spatial-Block Sensitivity Experiment

The current within-network random edge split can remain the default because the
project framing follows ISARC-style completion, not new-city prediction.

However, one spatial-block sensitivity run is worth doing.

Purpose:

- check whether the model is only interpolating nearby road segments;
- provide reviewer-friendly evidence about geographic robustness;
- avoid overclaiming from random edge splits.

How to report it:

```text
Primary setting: within-study-area random edge completion.
Sensitivity setting: spatial-block holdout across neighborhoods.
```

Do not make the spatial split a blocker for the whole project, but include it
before making strong claims about transfer to unseen neighborhoods.

### 4. Threshold Diagnostics Are Useful, But Not A Contribution

Claude's threshold-sensitivity diagnostic is worth doing. It helps show whether
the fixed 0.5 threshold is arbitrary or harmful.

But threshold tuning should be treated as reporting hygiene, not a research
contribution.

Recommended policy:

1. Run the diagnostic.
2. If validation and test curves agree, use a validation-selected threshold.
3. If they do not agree, report fixed 0.5 and discuss threshold instability.
4. Do not spend major project time optimizing threshold selection.

### 5. Defer Engineering Decoders

The sewer pseudo-outlet and downhill-flow decoder is conceptually clean as an
anchor-free engineering prior, but it is not the next best research move.

Reasons to defer:

- Brisbane likely includes pumped and flat-terrain sewer behavior;
- DEM-directed flow may become a fragile assumption;
- a decoder can improve component count while hurting edge truth;
- any decoder gain is hard to interpret until the threshold baseline is stable.

The only decoder worth adding soon is a simple, transparent baseline:

- validation-tuned threshold decoder;
- optional length-budget decoder.

Demand-connected, sewer-directed, and pseudo-outlet decoders should wait until
the base ablation results are stable.

## Recommended Execution Order

### Phase 1: Research Hygiene

Do first.

- Shared split across 10 m and 5 m labels.
- Prevalence baselines in all metrics.
- CLI prints held-out test metrics and trivial baselines.
- Ablation rows record exactly which context layers are enabled.

This makes the existing results interpretable.

### Phase 2: Core Ablation

Run the main scientific comparison:

- road-only;
- road plus building points;
- road plus building points plus DEM;
- RF vs GNN;
- GNN with and without node coordinates.

Report:

- ROC AUC;
- PR AUC;
- F1 beside all-positive F1;
- length precision/recall/F1;
- predicted total length;
- component count.

### Phase 3: Spatial Sensitivity

Run one spatial-block split using the best few variants from Phase 2.

Do not run every possible ablation spatially unless needed. The goal is to show
whether conclusions survive geographic separation.

### Phase 4: Threshold Diagnostic

Run threshold curves on completed experiments.

Use this to decide whether threshold tuning is justified. Keep it
supplementary.

### Phase 5: Simple Decoder Only

Add only if the base model results are stable:

- threshold decoder;
- length-budget decoder.

Do not implement pseudo-outlet sewer decoding until there is clear evidence
that local prediction quality is strong enough to support network-level
post-processing.

## What To Defer

Defer these unless reviewers ask or the core ablations are already stable:

- full inductive utility-component split;
- train-subgraph or split-subgraph GNN message passing;
- pseudo-outlet sewer decoder;
- water loop-aware decoder;
- complex probability calibration;
- per-region thresholds;
- large metrics schema rewrite.

These are not wrong. They are just lower reward right now.

## Suggested Paper Framing

Use wording like:

```text
We evaluate an anchor-free road-corridor prediction task. Candidate utility
corridors are road edges, and labels are derived from buffered overlap with
ground-truth utility lines. Dense utility anchors such as manholes, valves,
utility poles, cabinets, transformers, and known pipe junctions are not used as
model inputs.
```

For the main contribution:

```text
The study quantifies how much topology signal is available from road geometry
alone and how much non-anchor context such as building points and elevation
adds.
```

For the split:

```text
The primary evaluation follows a within-study-area completion setting. A
spatial-block sensitivity experiment is reported separately to assess geographic
robustness.
```

Avoid:

```text
The model locates underground utilities.
The model generalizes to unseen cities.
The sewer decoder reconstructs true hydraulic flow.
```

## Bottom Line

The strongest research path is:

1. make the metrics honest;
2. compare road-only, building, and DEM information sources;
3. compare classical ML against GNN;
4. test whether node coordinates are a shortcut;
5. add one spatial sensitivity experiment;
6. keep decoders simple until the base prediction result is clearly stronger
   than prevalence baselines.

This gives a credible anchor-free contribution without relying on subtle
engineering tricks.
