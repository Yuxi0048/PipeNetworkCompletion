"""Unit tests for pure-Python helpers in pipe_network_completion."""

from __future__ import annotations

import numpy as np

from pipe_network_completion.evaluation import compute_binary_metrics
from pipe_network_completion.model import infer_architecture_from_state_dict


def test_infer_architecture_pure_sage() -> None:
    state_dict = {
        "gnn.conv1.lin.weight": None,
        "gnn.conv2.lin.weight": None,
        "gnn.conv3.lin.weight": None,
        "gnn.conv4.lin.weight": None,
    }
    layers, skip = infer_architecture_from_state_dict(state_dict)
    assert layers == ("sage", "sage", "sage", "sage")
    assert skip is False


def test_infer_architecture_mixed_with_skip() -> None:
    state_dict = {
        "gnn.conv1.att_src": None,
        "gnn.conv2.lin.weight": None,
        "gnn.conv3.lin.weight": None,
        "gnn.conv4.att_src": None,
        "gnn.conv5.lin.weight": None,
        "gnn.conv6.lin.weight": None,
        "gnn.lin1.weight": None,
        "gnn.lin2.weight": None,
    }
    layers, skip = infer_architecture_from_state_dict(state_dict)
    assert layers == ("gat", "sage", "sage", "gat", "sage", "sage")
    assert skip is True


def test_compute_binary_metrics_perfect_separation() -> None:
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    labels = np.array([1, 1, 0, 0])
    metrics = compute_binary_metrics(scores, labels, threshold=0.5)
    assert metrics.auc == 1.0
    assert metrics.f1 == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.accuracy == 1.0
    assert metrics.mcc == 1.0


def test_compute_binary_metrics_single_class_auc_is_nan() -> None:
    scores = np.array([0.4, 0.6, 0.7])
    labels = np.array([1, 1, 1])
    metrics = compute_binary_metrics(scores, labels, threshold=0.5)
    assert np.isnan(metrics.auc)
