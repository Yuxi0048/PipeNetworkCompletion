"""Evaluation metrics for anchor-free road-edge predictions."""

# Workstream: Codex

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from pipe_network_completion.anchor_free.road_graph import RoadCandidateGraph


@dataclass(frozen=True)
class AnchorFreeMetrics:
    values: dict[str, float]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.values])


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _safe_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def compute_prevalence_baseline_metrics(y_true: np.ndarray) -> dict:
    """Return trivial baselines for judging imbalanced edge labels.

    Closed-form baselines emitted (let p = positive prevalence):

      * all_positive_*: a constant "1" predictor
          - roc_auc = 0.5 (random ranking for any constant scorer)
          - pr_auc = precision = p, recall = 1.0
          - f1 = 2p / (1 + p)
          - brier_score = 1 - p
          - balanced_accuracy = 0.5
      * majority_class_*: predict 0 if p<0.5 else 1
      * random_brier_score = p(1-p) (a perfectly calibrated random predictor)

    These are reported alongside model metrics so a small lift over the
    trivial baselines cannot be misread as a strong result. Note that when
    prevalence approaches 1.0 the all-positive baseline approaches F1 = 1.0,
    which is mathematically correct but counterintuitive — read it as "the
    task is degenerate at this label setting", not "the model is great".

    Phase E + Phase F of
    ``docs/research_notes/audit_followup_implementation_plan.md``.
    """

    y_true = np.asarray(y_true, dtype=int)
    if y_true.size == 0:
        return {
            "positive_prevalence": float("nan"),
            "all_positive_roc_auc": float("nan"),
            "all_positive_pr_auc": float("nan"),
            "all_positive_precision": float("nan"),
            "all_positive_recall": float("nan"),
            "all_positive_f1": float("nan"),
            "all_positive_brier_score": float("nan"),
            "all_positive_balanced_accuracy": float("nan"),
            "random_brier_score": float("nan"),
            "majority_class_label": -1,
            "majority_class_accuracy": float("nan"),
            "majority_class_f1": float("nan"),
        }

    prevalence = float(y_true.mean())
    has_both_classes = np.unique(y_true).size == 2
    all_positive_f1 = (
        2.0 * prevalence / (1.0 + prevalence)
        if prevalence > 0.0
        else 0.0
    )
    majority_label = 1 if prevalence >= 0.5 else 0
    majority_accuracy = prevalence if majority_label == 1 else 1.0 - prevalence
    majority_f1 = all_positive_f1 if majority_label == 1 else 0.0
    return {
        "positive_prevalence": prevalence,
        "all_positive_roc_auc": 0.5 if has_both_classes else float("nan"),
        "all_positive_pr_auc": prevalence,
        "all_positive_precision": prevalence,
        "all_positive_recall": 1.0 if prevalence > 0.0 else 0.0,
        "all_positive_f1": all_positive_f1,
        # Brier score of an all-positive constant predictor: each y_i in {0,1},
        # prediction = 1, error = (1 - y_i)**2 -> mean = (1 - prevalence).
        "all_positive_brier_score": 1.0 - prevalence,
        # Constant predictor has TPR = 1, TNR = 0 -> balanced_accuracy = 0.5.
        "all_positive_balanced_accuracy": 0.5,
        # Perfectly calibrated random predictor (p_i == prevalence for all i):
        # Brier = mean of (p - y_i)**2 = p*(1-p)**2 + (1-p)*p**2 = p(1-p).
        "random_brier_score": prevalence * (1.0 - prevalence),
        # Phase F: store as int — logically a class index, was float before.
        "majority_class_label": int(majority_label),
        "majority_class_accuracy": float(majority_accuracy),
        "majority_class_f1": float(majority_f1),
    }


def compute_edge_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute binary edge-classification metrics."""

    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    y_pred = (probabilities >= float(threshold)).astype(int)
    metrics = {
        "roc_auc": _safe_auc(y_true, probabilities),
        "pr_auc": _safe_average_precision(y_true, probabilities),
        "average_precision": _safe_average_precision(y_true, probabilities),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "iou_jaccard": float(jaccard_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }
    metrics.update(compute_prevalence_baseline_metrics(y_true))
    return metrics


def _selected_component_metrics(selected_edges: pd.DataFrame) -> dict[str, float]:
    if selected_edges.empty:
        return {
            "connected_component_count": 0.0,
            "cyclomatic_number": 0.0,
            "loop_density": 0.0,
        }
    graph = nx.Graph()
    for edge in selected_edges.itertuples(index=False):
        graph.add_edge(int(edge.u), int(edge.v), edge_id=int(edge.edge_id))
    component_count = nx.number_connected_components(graph)
    cyclomatic = graph.number_of_edges() - graph.number_of_nodes() + component_count
    return {
        "connected_component_count": float(component_count),
        "cyclomatic_number": float(max(cyclomatic, 0)),
        "loop_density": float(max(cyclomatic, 0) / max(graph.number_of_edges(), 1)),
    }


def compute_network_length_metrics(
    graph: RoadCandidateGraph,
    labels: pd.DataFrame,
    selected_edge_ids: np.ndarray | list[int],
) -> dict[str, float]:
    """Compute length-weighted and graph-topology metrics."""

    selected_set = set(int(edge_id) for edge_id in selected_edge_ids)
    edge_table = graph.edges[["edge_id", "u", "v", "length_m"]].copy()
    label_table = labels[["edge_id", "y"]].copy()
    merged = edge_table.merge(label_table, on="edge_id", how="left").fillna({"y": 0})
    merged["selected"] = merged["edge_id"].astype(int).isin(selected_set)
    merged["y"] = merged["y"].astype(int)

    true_positive = merged[(merged["selected"]) & (merged["y"] == 1)]["length_m"].sum()
    false_positive = merged[(merged["selected"]) & (merged["y"] == 0)]["length_m"].sum()
    false_negative = merged[(~merged["selected"]) & (merged["y"] == 1)]["length_m"].sum()
    predicted_total = merged[merged["selected"]]["length_m"].sum()
    true_total = merged[merged["y"] == 1]["length_m"].sum()

    length_precision = true_positive / predicted_total if predicted_total > 0 else 0.0
    length_recall = true_positive / true_total if true_total > 0 else 0.0
    denom = length_precision + length_recall
    length_f1 = 2.0 * length_precision * length_recall / denom if denom > 0 else 0.0

    metrics = {
        "predicted_total_length": float(predicted_total),
        "true_total_length": float(true_total),
        "true_positive_predicted_length": float(true_positive),
        "false_positive_length": float(false_positive),
        "false_negative_length": float(false_negative),
        "length_precision": float(length_precision),
        "length_recall": float(length_recall),
        "length_f1": float(length_f1),
    }
    metrics.update(_selected_component_metrics(merged[merged["selected"]]))
    return metrics


def compute_building_service_coverage(
    decoded_edges: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame | None,
    *,
    buffer_m: float = 50.0,
) -> dict[str, float]:
    if buildings is None or buildings.empty:
        return {}
    if decoded_edges.empty:
        return {"building_service_coverage": 0.0, "served_building_count": 0.0}
    if decoded_edges.crs and buildings.crs and str(decoded_edges.crs) != str(buildings.crs):
        buildings = buildings.to_crs(decoded_edges.crs)
    service_area = decoded_edges.geometry.buffer(float(buffer_m)).unary_union
    served = sum(1 for geom in buildings.geometry if geom is not None and service_area.intersects(geom))
    return {
        "building_service_coverage": float(served / max(len(buildings), 1)),
        "served_building_count": float(served),
    }


def _selected_component_metrics_segments(
    selected_segments: pd.DataFrame,
    segment_crosses_segment: np.ndarray,
) -> dict[str, float]:
    """Component / cyclomatic / loop-density metrics for the hetero pipeline.

    Phase 2.A of architectural_cleanup_plan.md. Builds the predicted-network
    adjacency from the heterogeneous graph's ``segment_crosses_segment``
    relation restricted to the selected RoadSegment ids.
    """
    if selected_segments.empty:
        return {
            "connected_component_count": 0.0,
            "cyclomatic_number": 0.0,
            "loop_density": 0.0,
        }
    sel_ids = set(selected_segments["segment_id"].astype(int).tolist())
    g = nx.Graph()
    for sid in sel_ids:
        g.add_node(int(sid))
    if segment_crosses_segment is not None and segment_crosses_segment.size:
        for a, b in zip(segment_crosses_segment[0], segment_crosses_segment[1]):
            a, b = int(a), int(b)
            if a in sel_ids and b in sel_ids and a != b:
                g.add_edge(a, b)
    component_count = nx.number_connected_components(g)
    cyclomatic = g.number_of_edges() - g.number_of_nodes() + component_count
    return {
        "connected_component_count": float(component_count),
        "cyclomatic_number": float(max(cyclomatic, 0)),
        "loop_density": float(
            max(cyclomatic, 0) / max(g.number_of_edges(), 1)
        ),
    }


def compute_network_length_metrics_segments(
    graph,
    labels: pd.DataFrame,
    selected_segment_ids: np.ndarray | list[int],
) -> dict[str, float]:
    """Length-weighted + topology metrics for the heterogeneous pipeline.

    Hetero counterpart of ``compute_network_length_metrics``. Keys on
    ``segment_id`` instead of ``edge_id`` and uses
    ``graph.segment_crosses_segment`` for the topology view.
    """
    selected_set = set(int(sid) for sid in selected_segment_ids)
    seg_table = graph.road_segments[["segment_id", "length_m"]].copy()
    label_table = labels[["segment_id", "y"]].copy()
    merged = seg_table.merge(label_table, on="segment_id", how="left").fillna({"y": 0})
    merged["selected"] = merged["segment_id"].astype(int).isin(selected_set)
    merged["y"] = merged["y"].astype(int)

    true_positive = merged[(merged["selected"]) & (merged["y"] == 1)]["length_m"].sum()
    false_positive = merged[(merged["selected"]) & (merged["y"] == 0)]["length_m"].sum()
    false_negative = merged[(~merged["selected"]) & (merged["y"] == 1)]["length_m"].sum()
    predicted_total = merged[merged["selected"]]["length_m"].sum()
    true_total = merged[merged["y"] == 1]["length_m"].sum()

    length_precision = true_positive / predicted_total if predicted_total > 0 else 0.0
    length_recall = true_positive / true_total if true_total > 0 else 0.0
    denom = length_precision + length_recall
    length_f1 = 2.0 * length_precision * length_recall / denom if denom > 0 else 0.0

    out = {
        "predicted_total_length": float(predicted_total),
        "true_total_length": float(true_total),
        "true_positive_predicted_length": float(true_positive),
        "false_positive_length": float(false_positive),
        "false_negative_length": float(false_negative),
        "length_precision": float(length_precision),
        "length_recall": float(length_recall),
        "length_f1": float(length_f1),
    }
    out.update(
        _selected_component_metrics_segments(
            merged[merged["selected"]],
            graph.segment_crosses_segment,
        )
    )
    return out


def evaluate_hetero_predictions(
    graph,
    labels: pd.DataFrame,
    probabilities: np.ndarray,
    selected_segment_ids: np.ndarray | list[int],
    *,
    threshold: float = 0.5,
    buildings: gpd.GeoDataFrame | None = None,
    building_service_buffer_m: float = 50.0,
    decoded_segments: gpd.GeoDataFrame | None = None,
    extra: Mapping[str, float | str] | None = None,
) -> AnchorFreeMetrics:
    """Heterogeneous-pipeline counterpart of evaluate_anchor_free_predictions."""
    edge_metrics = compute_edge_metrics(
        labels["y"].to_numpy(dtype=int), probabilities, threshold=threshold
    )
    network_metrics = compute_network_length_metrics_segments(
        graph, labels, selected_segment_ids
    )
    coverage = compute_building_service_coverage(
        decoded_segments if decoded_segments is not None else gpd.GeoDataFrame(),
        buildings,
        buffer_m=building_service_buffer_m,
    )
    values: dict[str, float | str] = {}
    values.update(edge_metrics)
    values.update(network_metrics)
    values.update(coverage)
    if extra:
        values.update(dict(extra))
    return AnchorFreeMetrics(values=dict(values))


def evaluate_anchor_free_predictions(
    graph: RoadCandidateGraph,
    labels: pd.DataFrame,
    probabilities: np.ndarray,
    selected_edge_ids: np.ndarray | list[int],
    *,
    threshold: float = 0.5,
    buildings: gpd.GeoDataFrame | None = None,
    building_service_buffer_m: float = 50.0,
    decoded_edges: gpd.GeoDataFrame | None = None,
    extra: Mapping[str, float | str] | None = None,
) -> AnchorFreeMetrics:
    """Compute edge-level, length-weighted, and optional coverage metrics."""

    edge_metrics = compute_edge_metrics(labels["y"].to_numpy(dtype=int), probabilities, threshold=threshold)
    network_metrics = compute_network_length_metrics(graph, labels, selected_edge_ids)
    coverage = compute_building_service_coverage(
        decoded_edges if decoded_edges is not None else gpd.GeoDataFrame(),
        buildings,
        buffer_m=building_service_buffer_m,
    )
    values: dict[str, float | str] = {}
    values.update(edge_metrics)
    values.update(network_metrics)
    values.update(coverage)
    if extra:
        values.update(dict(extra))
    return AnchorFreeMetrics(values=dict(values))
