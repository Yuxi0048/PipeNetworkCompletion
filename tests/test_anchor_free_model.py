"""One-epoch smoke test for the road-edge GNN."""

# Workstream: Claude + Codex merge

from __future__ import annotations

import importlib

import numpy as np
import pytest


_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
_PYG_AVAILABLE = importlib.util.find_spec("torch_geometric") is not None

pytestmark = pytest.mark.skipif(
    not (_TORCH_AVAILABLE and _PYG_AVAILABLE),
    reason="torch and torch_geometric are required for the GNN smoke test.",
)


def test_gnn_forward_and_one_epoch():
    from pipe_network_completion.anchor_free.baseline import (
        make_stratified_edge_splits,
    )
    from pipe_network_completion.anchor_free.features import build_road_edge_features
    from pipe_network_completion.anchor_free.labels import (
        label_road_edges_from_utility_lines,
    )
    from pipe_network_completion.anchor_free.model import (
        build_pyg_road_edge_data,
        train_road_edge_gnn,
    )
    from pipe_network_completion.anchor_free.road_graph import (
        build_road_candidate_graph,
    )
    from pipe_network_completion.anchor_free.synthetic import (
        make_synthetic_anchor_free_data,
    )

    data = make_synthetic_anchor_free_data()
    graph = build_road_candidate_graph(data.roads, target_crs="EPSG:3857")
    features = build_road_edge_features(
        graph, buildings_gdf=data.buildings, road_class_columns="road_class"
    )
    labels = label_road_edges_from_utility_lines(graph, data.utility_truth)
    train, val, test = make_stratified_edge_splits(labels.y, seed=0)

    pyg_data = build_pyg_road_edge_data(graph, features, labels=labels.y)
    result = train_road_edge_gnn(
        pyg_data,
        train_index=train,
        val_index=val,
        test_index=test,
        seed=0,
        epochs=2,
        hidden_dim=8,
        num_layers=2,
    )
    assert result.probabilities.shape == (len(labels.y),)
    assert result.probabilities.min() >= 0.0
    assert result.probabilities.max() <= 1.0
    assert len(result.losses) == 2
    assert np.isfinite(result.losses[0])
    assert result.device in {"cpu", "cuda"}


def test_torch_device_report_has_expected_keys():
    from pipe_network_completion.anchor_free.model import (
        resolve_torch_device,
        torch_device_report,
    )

    report = torch_device_report()
    assert {"torch_version", "cuda_available", "cuda_device_count", "cuda_devices"}.issubset(
        report
    )
    assert resolve_torch_device("auto").type in {"cpu", "cuda"}
