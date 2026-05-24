# Anchor-Free Engineering Decoder Note

Workstream: Codex

This note captures the design decision discussed on 2026-05-23 about the
post-GNN engineering decoder for the anchor-free utility-network prediction
experiment. It is intended for the Claude/Codex parallel implementation work.

## Anchor-free definition

The decoder can remain anchor-free if it does not use dense known utility-node
locations or surveyed network endpoints as inputs. Forbidden inputs remain:

- manholes
- valves
- pipe junctions
- cabinets
- utility poles
- transformers
- surveyed utility-node coordinates
- known utility-line vertices as graph nodes
- ground-truth utility geometry except for label creation and evaluation

Using an outlet is only anchor-free if it is not a surveyed utility asset. If no
site visit or outlet inventory is available, do not invent a real outlet. Use
terrain-derived pseudo-outlet candidates instead.

## Proposed pseudo-outlet rule

For sewer-mode decoding without known outlets:

1. Sample DEM elevation at road-graph nodes.
2. Identify road nodes close to the study-area boundary.
3. Select low-elevation boundary candidates.

Example rule:

```text
pseudo_outlet_candidates =
    road nodes where
        elevation <= 5th percentile of road-node elevation
        and distance_to_study_boundary <= 200 m
```

These pseudo-outlets are derived only from the road graph, DEM, and study-area
boundary. They are not utility anchors and should be documented as engineering
priors, not observed infrastructure.

## Decoder concept

The GNN or RandomForest predicts local road-edge probability:

```text
p_e = P(road edge e carries utility | road/building/DEM/context features)
```

The engineering decoder then reconstructs a network using non-anchor priors:

- edge probability from the model
- road-edge length
- building-point demand clusters
- building-polygon demand/land-use context
- DEM slope/elevation
- terrain-derived pseudo-outlets when true sparse outlet facilities are absent

Base cost:

```text
cost(e) =
    -log(p_e + eps)
    + lambda_length * length_m
    + lambda_uphill * uphill_penalty(e)
    - lambda_demand * demand_reward(e)
```

For sewer, orient paths toward pseudo-outlets and penalize uphill movement. For
water, allow limited loops and do not require downhill orientation.

## Recommended implementation order

1. Keep validation-tuned threshold decoding as the baseline.
2. Add top-k or length-budget decoding, tuned on validation.
3. Add demand-connected decoding:
   - cluster building points into demand terminals
   - connect terminals over road graph using probability/length cost
4. Add sewer-directed decoding:
   - derive pseudo-outlets from DEM and boundary
   - penalize uphill paths
   - prune low-demand leaves
   - prefer tree-like topology

Do not claim the engineering decoder helps until it is compared against the
threshold-only decoder on held-out test metrics.

## Paper wording

Use wording like:

> In the absence of surveyed outlets, the sewer-directed decoder uses
> terrain-derived pseudo-outlet candidates selected from low-elevation road
> graph boundary nodes. These candidates are not observed sewer assets and are
> used only as anchor-free engineering priors.

Avoid wording like:

> known outlets

unless the outlet file is a sparse public system-facility layer and not a dense
utility-node/anchor inventory.

## Main limitation

Terrain-derived pseudo-outlets can be wrong in pumped urban sewer systems,
inverted siphons, pressure mains, or networks crossing topographic ridges. The
decoder remains a planning/research-grade probabilistic reconstruction method
and must not be used for excavation clearance or legal utility locating.
