"""Tests for the include_node_coords ablation flag (Stage 2).

# Workstream: Claude

Covers the new ``include_node_coords`` parameter on ``_node_feature_tensor``
and ``build_pyg_road_edge_data`` from
``docs/research_notes/audit_followup_implementation_plan.md`` §16 Stage 2.
"""

from __future__ import annotations

import importlib.util

import pytest

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
_PYG_AVAILABLE = importlib.util.find_spec("torch_geometric") is not None

pytestmark = pytest.mark.skipif(
    not (_TORCH_AVAILABLE and _PYG_AVAILABLE),
    reason="torch and torch_geometric required for the GNN coord-ablation test.",
)


def test_node_feature_tensor_drops_xy_when_include_coords_false():
    from pipe_network_completion.anchor_free.features import (
        build_road_edge_features,
    )
    from pipe_network_completion.anchor_free.model import _node_feature_tensor
    from pipe_network_completion.anchor_free.road_graph import (
        build_road_candidate_graph,
    )
    from pipe_network_completion.anchor_free.synthetic import (
        make_synthetic_anchor_free_data,
    )

    data = make_synthetic_anchor_free_data()
    graph = build_road_candidate_graph(data.roads, target_crs="EPSG:3857")
    build_road_edge_features(graph)  # ensure features pipeline still works

    x_with = _node_feature_tensor(graph, include_coords=True)
    x_without = _node_feature_tensor(graph, include_coords=False)
    assert x_with.shape[1] == 3   # (x, y, degree)
    assert x_without.shape[1] == 1  # (degree)
    assert x_with.shape[0] == x_without.shape[0] == len(graph.nodes)


def test_build_pyg_road_edge_data_respects_include_node_coords():
    from pipe_network_completion.anchor_free.features import (
        build_road_edge_features,
    )
    from pipe_network_completion.anchor_free.labels import (
        label_road_edges_from_utility_lines,
    )
    from pipe_network_completion.anchor_free.model import (
        build_pyg_road_edge_data,
    )
    from pipe_network_completion.anchor_free.road_graph import (
        build_road_candidate_graph,
    )
    from pipe_network_completion.anchor_free.synthetic import (
        make_synthetic_anchor_free_data,
    )

    data = make_synthetic_anchor_free_data()
    graph = build_road_candidate_graph(data.roads, target_crs="EPSG:3857")
    features = build_road_edge_features(graph)
    labels = label_road_edges_from_utility_lines(graph, data.utility_truth)

    with_coords = build_pyg_road_edge_data(
        graph, features, labels=labels.y, include_node_coords=True
    )
    no_coords = build_pyg_road_edge_data(
        graph, features, labels=labels.y, include_node_coords=False
    )
    assert with_coords.x.shape[1] == 3
    assert no_coords.x.shape[1] == 1
    # edge_index / edge_label_index / edge_label_attr must be unchanged.
    assert with_coords.edge_index.shape == no_coords.edge_index.shape
    assert with_coords.edge_label_index.shape == no_coords.edge_label_index.shape
    assert with_coords.edge_label_attr.shape == no_coords.edge_label_attr.shape
