# Domain Design And Code Alignment Review

Workstream: Codex

Date: 2026-05-23

## Purpose

This note checks whether the current anchor-free codebase aligns with the
domain-prior design in `utility_domain_knowledge_priors_codex.md` and the
broader anchor-free experiment design. The emphasis is research credibility and
research reward, not clever engineering additions.

## Overall Verdict

The codebase is aligned with the core anchor-free research question:

> Can utility corridor presence be predicted from roads and non-anchor context
> layers, without using ground utility anchor points as model inputs?

The implementation is strongest as a supervised road-edge classification
pipeline. It builds road candidates, computes non-anchor features, creates
labels from utility truth geometry, trains RandomForest/logistic/GNN models,
writes QGIS-inspectable outputs, and reports edge/length metrics with trivial
baseline comparisons.

The implementation is only partially aligned with the richer domain-topology
design. It has a threshold decoder, a simple connected decoder, component and
loop metrics, and experimental heterogeneous road-graph code. It does not yet
implement utility-type-specific topology algorithms such as sewer-directed
trees, water loop budgets, dead-end-aware decoding, boundary-aware topology
evaluation, or DEM-guided pseudo-outlet routing.

For a credible paper, the near-term priority should be:

1. Make the supervised anchor-free comparison clean and defensible.
2. Prove whether building points / built-up / DEM add value over roads alone.
3. Add topology diagnostics as evaluation first, before adding complicated
   post-processing algorithms.
4. Treat advanced decoders as secondary experiments unless they deliver clear
   validation/test gains.

## Alignment Matrix

| Design item | Current code alignment | Pros | Cons / gaps | Recommendation |
| --- | --- | --- | --- | --- |
| Anchor-free input boundary | Strong | Feature builders use roads, buildings, building points, built-up areas, and DEM. `assert_no_anchor_features` blocks obvious anchor feature names. Inference-only GeoJSON output excludes labels and overlap columns. | The guard only catches names, not semantic leakage hidden behind innocent column names. Human review of raw context layers is still needed. | Keep the guard, but document all input layers in each experiment's resolved config and metadata. |
| Road network as candidate graph | Strong for endpoint graph; partial for true intersection graph | `road_graph.py` creates road-edge candidates with IDs, geometry, length, bearing, endpoints, and CRS warnings. | The default graph uses source feature endpoints. If roads are not noded at crossings, message passing misses real intersections. | Use the heterogeneous road-segment graph for a sensitivity run, or add a preprocessing/noding step before claiming intersection-level topology. |
| Utility truth for labels only | Strong | `labels.py` buffers road edges and intersects truth lines to create `y`, `overlap_length`, and `overlap_ratio`. Output splitting separates prediction-only and evaluation files. | The task definition is sensitive to `label_buffer_m` and overlap threshold. A 5 m vs 10 m label can materially change prevalence and metrics. | Report both 5 m and 10 m results using the same edge split, and frame them as label-definition sensitivity. |
| Road-only baseline | Strong | Configs and ablation scripts can disable buildings, building points, built-up, and DEM. RandomForest/logistic baselines exist. | A road-only model can still learn strong spatial/geometry priors; results should not be interpreted as utility discovery from roads alone without prevalence and all-positive baselines. | Keep road-only as the first baseline and always report lift over all-positive and majority baselines. |
| Building / demand proximity | Strong feature support; partial research validation | Building polygons, building points, built-up areas, counts, distances, densities, and category-derived features are present. | The code does not yet turn demand into terminals/clusters for topology decoding. It also has not established whether building points add independent value over building areas. | Run road-only, road+building-points, road+built-up, road+DEM, and combined ablations. This is high research reward. |
| DEM / slope priors | Partial | DEM-derived elevation, range, slope, and valid-fraction features are implemented. | DEM is not used in decoding. No uphill-length metric or outlet-directed evaluation exists. Brisbane may include pumped/flat sewer conditions where DEM direction is weak. | Treat DEM as a feature ablation first. Do not claim hydraulic flow correctness from DEM. |
| Connectivity prior | Partial | The connected decoder can connect high-probability terminals using probability/length cost. Evaluation reports connected component count. | The connected decoder is a generic graph heuristic, not a utility-specific reconstruction algorithm. It does not enforce tree-like sewer topology, water loops, or demand coverage. | Use it as a diagnostic post-processor only after threshold results are credible. Tune on validation, not test. |
| Dead ends | Weak | Dead ends can be derived from decoded graph geometry, but they are not first-class metrics or decoder constraints. | No dead-end count, dead-end density, boundary dead-end fraction, or utility-type-specific dead-end interpretation. | Add dead-end metrics before writing dead-end-aware decoders. This is cheap and research-useful. |
| Loops and circulation | Partial | Cyclomatic number and loop density are reported. Reserved `sewer`, `water`, and `steiner` decoder names now fail loudly instead of silently aliasing to generic connected decoding. | No water loop-budget decoder and no sewer loop penalty. Loop density is not compared to truth topology. | Keep loop density as an evaluation metric. Defer loop-aware decoding unless baseline topology metrics show a clear failure mode. |
| Continuity along corridors | Partial | GNN message passing and connected decoding encourage local continuity indirectly. | There is no explicit corridor-continuity metric, smoothing loss, run-length metric, or penalty for isolated high-probability edges. | Add simple diagnostics: isolated selected edge fraction, mean selected run length, and largest-component length fraction. |
| Road hierarchy | Moderate | Road class one-hot features, local road density, length, bearing, and endpoint degree exist. | No road centrality, distance to arterial, road width, access class hierarchy, or network betweenness. | Add road centrality only if it improves ablations; it is plausible but not as high reward as building/demand validation. |
| Boundary effects | Weak | Study-area clipping is documented as a risk. | No distance-to-boundary feature, boundary-near tag, boundary/interior metric split, or special treatment of boundary dead ends. | Add boundary distance and boundary-aware reporting before interpreting dead-end or component metrics. |
| Barriers / crossings | Not implemented | No evidence of illegal barrier-derived leakage. | Rivers, rail, highways, bridges/tunnels, and crossing penalties are absent. | Defer unless reliable barrier layers are available. Poor barrier layers can create more noise than signal. |
| Coordinate/location encoder | Configurable but risky | `include_node_coords` can be disabled. Stage 2 ablation variants include xy/no-xy comparisons. | Default coordinate features can inflate random held-out edge metrics by letting the GNN memorize location-specific prevalence. | For headline claims, report a no-coordinate GNN or explicitly call the xy model transductive/location-aware. |
| Data split credibility | Improved but not fully inductive | Default split is buffer-invariant, so 5 m and 10 m labels can share the same edge partition. Stratified split remains available. | Full-graph GNN message passing is still transductive. Random edge splits do not measure generalization to a new neighborhood. | Use current split for within-network completion. Add one spatial-block or component-held-out sensitivity experiment for credibility. |
| Evaluation | Stronger than initial version | ROC AUC, PR AUC, F1, balanced accuracy, Brier score, length metrics, building service coverage, connected components, loop density, and trivial baselines are present. | Threshold grid exists but is not yet a full validation-tuned operating point. Missing topology diagnostics listed above. | Report threshold-independent metrics first, then val-tuned threshold metrics. Add topology diagnostics. |
| Heterogeneous road graph | Promising but not mainline | Claude's `hetero_road_graph.py` models road segments and intersections, including segment-crosses-segment adjacency through spatial join. Tests cover smoke behavior. | It is not the default training path. It adds complexity before it has demonstrated better validation/test results. | Use it as a controlled ablation against the simpler endpoint graph. Do not make it the default until it shows benefit. |

## Current Strengths

The best parts of the current implementation are research-useful:

- The anchor-free boundary is explicit and mostly enforceable.
- Real-data configs can run road-only and context-feature variants.
- Building points, built-up areas, and DEM are wired as optional context, not
  mandatory hidden inputs.
- Labels are generated from truth lines in a supervised way, and prediction-only
  outputs are separated from evaluation outputs.
- The code now includes trivial baselines, which are essential because high F1
  can be caused by high positive prevalence.
- The 5 m / 10 m label comparison is becoming more credible because the split
  can be held fixed across label buffers.
- Unsupported decoder names fail loudly, reducing the risk of claiming a sewer
  or water decoder that does not actually exist.

These strengths are enough to support a credible anchor-free supervised
prediction study if the experiments are reported conservatively.

## Main Weaknesses

The main weaknesses are not small engineering details; they affect the research
claim.

### 1. The current model is mostly an edge classifier, not a topology generator

The original design asks for road-constrained probabilistic utility network
generation. The current code predicts edge probabilities and can threshold or
connect selected edges, but the post-processing is not yet utility-aware. This
is acceptable if the paper frames the method as:

> anchor-free road-edge utility-corridor probability prediction with topology
> diagnostics.

It is not yet enough to claim:

> a full sewer/water topology reconstruction algorithm with engineering
> constraints.

### 2. Random held-out edge metrics can overstate generalization

Within-network edge splits answer a useful question: can the model complete
missing parts of a known service area using nearby road/context structure?

They do not answer: can the model transfer to a new neighborhood or unseen
spatial region?

The coordinate encoder increases this risk. The code has an ablation switch,
which is good. The headline results should either disable coordinates or clearly
state that coordinates are part of a transductive within-area setting.

### 3. Road topology may be wrong if centerlines are not noded

The default `road_graph.py` graph uses endpoints from source road features. If
the road dataset stores long unsplit centerlines crossing at intersections, the
GNN will not see those intersections. Claude's heterogeneous graph addresses
this more directly, but it is not the main pipeline yet.

This affects topology claims more than feature-only RandomForest claims.

### 4. Domain priors are mostly features and metrics, not algorithms

Demand, DEM, road class, degree, density, and loop metrics exist. But dead-end,
boundary, flow-direction, and loop-budget logic are not implemented as tested
network algorithms.

That is not a failure if the immediate contribution is the data/feature
ablation. It would be a problem if the write-up claims hydraulic or operational
network correctness.

## Best Research Framing

The most credible framing is:

> We test whether anchor-free, road-constrained utility corridor prediction is
> possible using road geometry and public/non-anchor context layers. We compare
> road-only, demand-context, and DEM-context variants, benchmark against trivial
> prevalence baselines and classical ML, and evaluate both edge-level accuracy
> and basic topology diagnostics.

This framing is honest, publishable, and aligned with the current code.

Avoid framing the current system as:

- legal utility locating;
- excavation clearance;
- validated hydraulic design;
- complete pipe-network reconstruction;
- a sewer/water decoder with real engineering constraints.

## Highest-Reward Next Changes

### High Priority

1. Run a clean ablation table with fixed splits:
   - road-only RF;
   - road + building points RF;
   - road + building points + DEM RF;
   - road-only GNN without coordinates;
   - road + building points GNN without coordinates;
   - optional GNN with coordinates clearly labeled as location-aware.

2. Add topology diagnostics before advanced decoders:
   - dead-end count;
   - dead-end density per km;
   - largest connected component length fraction;
   - isolated selected edge fraction;
   - degree histogram;
   - boundary dead-end fraction if a study boundary is available.

3. Add boundary distance features and boundary/interior metric splits.

4. Use validation-tuned thresholds for F1/length-F1 and keep ROC AUC / PR AUC
   as threshold-independent headline metrics.

5. Run one spatial-block or component-held-out sensitivity experiment. It does
   not need to replace the within-network split, but it should bound the
   generalization claim.

### Medium Priority

1. Compare the endpoint road graph with the heterogeneous road-segment graph.
2. Add road centrality / betweenness if it is cheap on the study graph.
3. Add DEM uphill-length diagnostics for sewer/stormwater only.
4. Add demand cluster summaries, but only as evaluation/diagnostics first.

### Defer

1. Sewer pseudo-outlet routing unless DEM and boundary candidates show clear
   validation value.
2. Water loop-budget decoding unless the task is explicitly water distribution.
3. Barrier penalties unless reliable river/rail/highway/bridge layers are
   available.
4. Complex Steiner-like algorithms until threshold and simple connected
   decoders have been benchmarked fairly.

## Bottom Line

The codebase aligns well with the anchor-free supervised prediction design and
is close to supporting a credible empirical study. Its biggest research value is
not a subtle decoder trick; it is a clean answer to:

> How much utility-corridor signal is contained in roads alone, and how much do
> non-anchor demand/context layers add?

The current code should be presented as planning-grade probabilistic research,
not as excavation-grade utility locating or legal asset confirmation.
