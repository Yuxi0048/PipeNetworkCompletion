"""Anchor-free, road-constrained utility-network prediction experiment.

This is an additive experimental variant of the ISARC 2024 anchor-based GNN
pipeline. It treats road segments as candidate utility corridors and predicts
per-edge probabilities without using ground anchor points (manholes, valves,
poles, transformers, cabinets, surveyed utility nodes) as model inputs.

Ground-truth utility geometry is only consumed by ``labels`` and
``evaluation``; it must not be used as a feature at inference time. See
``docs/anchor_free_design.md`` for the full design note.
"""

# Workstream: Codex

from __future__ import annotations

__all__ = [
    "road_graph",
    "features",
    "labels",
    "baseline",
    "model",
    "decoder",
    "evaluation",
    "synthetic",
    "pipeline",
    "config",
]
