"""Smoke tests for the classical baseline."""

# Workstream: Claude

from __future__ import annotations

import numpy as np
import pytest

from pipe_network_completion.anchor_free.baseline import (
    make_stratified_edge_splits,
    train_baseline,
)
from pipe_network_completion.anchor_free.features import build_road_edge_features
from pipe_network_completion.anchor_free.labels import (
    label_road_edges_from_utility_lines,
)
from pipe_network_completion.anchor_free.road_graph import build_road_candidate_graph
from pipe_network_completion.anchor_free.synthetic import (
    make_synthetic_anchor_free_data,
)


@pytest.fixture(scope="module")
def synthetic_artifacts():
    data = make_synthetic_anchor_free_data()
    graph = build_road_candidate_graph(data.roads, target_crs="EPSG:3857")
    features = build_road_edge_features(
        graph, buildings_gdf=data.buildings, road_class_columns="road_class"
    )
    labels = label_road_edges_from_utility_lines(graph, data.utility_truth)
    return data, graph, features, labels


def test_stratified_split_preserves_each_class(synthetic_artifacts):
    _, _, _, labels = synthetic_artifacts
    train, val, test = make_stratified_edge_splits(labels.y, seed=0)
    union = np.concatenate([train, val, test])
    assert len(np.unique(union)) == len(union), "splits must not overlap"
    assert set(union.tolist()) == set(range(len(labels.y)))


@pytest.mark.parametrize("kind", ["logistic_regression", "random_forest"])
def test_baseline_produces_probabilities_for_every_edge(synthetic_artifacts, kind):
    _, _, features, labels = synthetic_artifacts
    train, val, test = make_stratified_edge_splits(labels.y, seed=0)
    result = train_baseline(
        features.features,
        labels.y,
        kind=kind,
        train_index=train,
        val_index=val,
        test_index=test,
        seed=0,
    )
    assert result.probabilities.shape == (len(labels.y),)
    assert result.probabilities.min() >= 0.0
    assert result.probabilities.max() <= 1.0
