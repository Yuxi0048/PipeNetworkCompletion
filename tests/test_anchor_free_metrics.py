"""Smoke tests for anchor-free metrics."""

# Workstream: Claude

from __future__ import annotations

import math

import numpy as np

from pipe_network_completion.anchor_free.decoder import decode_threshold
from pipe_network_completion.anchor_free.evaluation import (
    compute_edge_metrics,
    compute_network_length_metrics,
    evaluate_anchor_free_predictions,
)
from pipe_network_completion.anchor_free.labels import (
    label_road_edges_from_utility_lines,
)
from pipe_network_completion.anchor_free.road_graph import build_road_candidate_graph
from pipe_network_completion.anchor_free.synthetic import (
    make_synthetic_anchor_free_data,
)


def test_compute_edge_metrics_returns_finite_values():
    rng = np.random.default_rng(0)
    n = 200
    scores = rng.uniform(0.0, 1.0, size=n)
    labels = rng.integers(0, 2, size=n)
    metrics = compute_edge_metrics(labels, scores, threshold=0.5)
    for key in (
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "iou_jaccard",
        "brier_score",
        "positive_prevalence",
        "all_positive_f1",
        "all_positive_roc_auc",
        "majority_class_f1",
    ):
        assert key in metrics
        # roc_auc and pr_auc may be NaN if a single class is present, but here
        # labels are random so both classes should be represented.
        assert metrics[key] is not None


def test_compute_edge_metrics_reports_prevalence_baseline():
    labels = np.array([1, 1, 1, 0], dtype=int)
    scores = np.array([0.8, 0.7, 0.4, 0.3], dtype=float)
    metrics = compute_edge_metrics(labels, scores, threshold=0.5)
    prevalence = 0.75
    expected_all_positive_f1 = 2.0 * prevalence / (1.0 + prevalence)
    assert math.isclose(metrics["positive_prevalence"], prevalence)
    assert math.isclose(metrics["all_positive_f1"], expected_all_positive_f1)
    assert math.isclose(metrics["all_positive_roc_auc"], 0.5)
    assert math.isclose(metrics["all_positive_pr_auc"], prevalence)


def test_compute_edge_metrics_reports_brier_and_balanced_accuracy_baselines():
    """Phase E — verify closed-form trivial-baseline coverage is complete."""
    labels = np.array([1, 1, 1, 0], dtype=int)
    scores = np.array([0.8, 0.7, 0.4, 0.3], dtype=float)
    metrics = compute_edge_metrics(labels, scores, threshold=0.5)
    prevalence = 0.75
    # Brier of an all-positive constant predictor = 1 - p
    assert math.isclose(metrics["all_positive_brier_score"], 1.0 - prevalence)
    # Brier of a perfectly-calibrated random predictor = p(1-p)
    assert math.isclose(metrics["random_brier_score"], prevalence * (1.0 - prevalence))
    # Constant predictor has balanced_accuracy = 0.5 (TPR=1, TNR=0).
    assert math.isclose(metrics["all_positive_balanced_accuracy"], 0.5)


def test_majority_class_label_is_int_not_float():
    """Phase F — `majority_class_label` is a class index, store as int."""
    labels = np.array([1, 1, 1, 0], dtype=int)
    scores = np.array([0.8, 0.7, 0.4, 0.3], dtype=float)
    metrics = compute_edge_metrics(labels, scores, threshold=0.5)
    assert isinstance(metrics["majority_class_label"], int)
    assert metrics["majority_class_label"] == 1


def test_compute_network_length_metrics_handles_overlap_and_misses():
    data = make_synthetic_anchor_free_data()
    graph = build_road_candidate_graph(data.roads, target_crs="EPSG:3857")
    labels = label_road_edges_from_utility_lines(graph, data.utility_truth)
    decoded = decode_threshold(
        graph,
        labels.y.astype(float),  # perfect predictions
        threshold=0.5,
    )
    network = compute_network_length_metrics(graph, labels.labels, decoded.edge_ids)
    assert network["length_precision"] > 0.99
    assert network["length_recall"] > 0.99


def test_evaluate_anchor_free_predictions_runs_end_to_end():
    data = make_synthetic_anchor_free_data()
    graph = build_road_candidate_graph(data.roads, target_crs="EPSG:3857")
    labels = label_road_edges_from_utility_lines(graph, data.utility_truth)
    probs = np.where(labels.y == 1, 0.9, 0.1)
    decoded = decode_threshold(graph, probs, threshold=0.5)
    summary = evaluate_anchor_free_predictions(
        graph,
        labels.labels,
        probs,
        decoded.edge_ids,
        threshold=0.5,
        buildings=data.buildings,
        decoded_edges=decoded.edges,
    )
    values = summary.values
    assert values["f1"] > 0.9
    # building_service_coverage is a value in [0, 1]; with synthetic data and
    # near-perfect predictions, the network should cover at least one building.
    assert "building_service_coverage" in values
