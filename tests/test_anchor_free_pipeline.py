"""Smoke tests for the Codex anchor-free experiment modules."""

# Workstream: Codex

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import shutil
from pathlib import Path

from pipe_network_completion.anchor_free.baseline import train_baseline
from pipe_network_completion.anchor_free.config import load_anchor_free_config
from pipe_network_completion.anchor_free.decoder import decode_connected, decode_threshold
from pipe_network_completion.anchor_free.evaluation import evaluate_anchor_free_predictions
from pipe_network_completion.anchor_free.features import (
    assert_no_anchor_features,
    build_road_edge_features,
    standardize_features,
)
from pipe_network_completion.anchor_free.labels import label_road_edges_from_utility_lines
from pipe_network_completion.anchor_free.model import (
    build_pyg_road_edge_data,
    train_road_edge_gnn,
)
from pipe_network_completion.anchor_free.pipeline import run_anchor_free_experiment
from pipe_network_completion.anchor_free.road_graph import build_road_candidate_graph
from pipe_network_completion.anchor_free.synthetic import make_synthetic_anchor_free_data
from scripts.prepare_anchor_free_features import prepare_features


@pytest.fixture()
def synthetic_graph_bundle():
    synthetic = make_synthetic_anchor_free_data()
    graph = build_road_candidate_graph(
        synthetic.roads,
        target_crs="EPSG:3857",
        keep_columns=["road_class"],
    )
    return synthetic, graph


def test_road_graph_can_be_built_without_anchor_points(synthetic_graph_bundle) -> None:
    _, graph = synthetic_graph_bundle
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0
    assert "edge_id" in graph.edges.columns
    assert "road_class" in graph.edges.columns
    assert not any("manhole" in column.lower() for column in graph.edges.columns)


def test_feature_matrix_has_one_row_per_road_edge(synthetic_graph_bundle) -> None:
    synthetic, graph = synthetic_graph_bundle
    features = build_road_edge_features(
        graph,
        buildings_gdf=synthetic.buildings,
        road_class_columns=["road_class"],
    )
    assert features.features.shape[0] == len(graph.edges)
    assert "nearest_building_distance_m" in features.features.columns


def test_anchor_feature_guard_catches_forbidden_names() -> None:
    with pytest.raises(ValueError):
        assert_no_anchor_features(["length_m", "distance_to_manhole"])


def test_labeling_creates_positive_and_negative_edges(synthetic_graph_bundle) -> None:
    synthetic, graph = synthetic_graph_bundle
    labels = label_road_edges_from_utility_lines(
        graph,
        synthetic.utility_truth,
        label_buffer_m=5.0,
        label_overlap_threshold=0.25,
    )
    assert labels.y.sum() > 0
    assert labels.y.sum() < len(labels.y)


def test_random_forest_baseline_trains_on_synthetic_data(synthetic_graph_bundle) -> None:
    synthetic, graph = synthetic_graph_bundle
    features = build_road_edge_features(graph, buildings_gdf=synthetic.buildings)
    labels = label_road_edges_from_utility_lines(graph, synthetic.utility_truth)
    scaled, _, _ = standardize_features(features.features)
    result = train_baseline(scaled, labels.y, kind="random_forest", seed=7)
    assert result.probabilities.shape == labels.y.shape
    assert np.all((result.probabilities >= 0.0) & (result.probabilities <= 1.0))


def test_gnn_runs_one_tiny_training_epoch(synthetic_graph_bundle) -> None:
    synthetic, graph = synthetic_graph_bundle
    features = build_road_edge_features(graph, buildings_gdf=synthetic.buildings)
    labels = label_road_edges_from_utility_lines(graph, synthetic.utility_truth)
    scaled, _, _ = standardize_features(features.features)
    data = build_pyg_road_edge_data(graph, scaled, labels=labels.y)
    result = train_road_edge_gnn(
        data,
        train_index=np.arange(len(labels.y)),
        epochs=1,
        hidden_dim=8,
        num_layers=1,
        seed=3,
    )
    assert result.probabilities.shape == labels.y.shape


def test_decoders_return_valid_geometries(synthetic_graph_bundle) -> None:
    _, graph = synthetic_graph_bundle
    probabilities = np.linspace(0.1, 0.9, len(graph.edges))
    threshold_decoded = decode_threshold(graph, probabilities, threshold=0.5)
    connected_decoded = decode_connected(graph, probabilities, threshold=0.8)
    assert not threshold_decoded.edges.empty
    assert threshold_decoded.edges.geometry.notna().all()
    assert connected_decoded.edges.geometry.notna().all()


def test_metrics_run_without_crashing(synthetic_graph_bundle) -> None:
    synthetic, graph = synthetic_graph_bundle
    labels = label_road_edges_from_utility_lines(graph, synthetic.utility_truth)
    probabilities = np.linspace(0.05, 0.95, len(graph.edges))
    decoded = decode_threshold(graph, probabilities, threshold=0.5)
    metrics = evaluate_anchor_free_predictions(
        graph,
        labels.labels,
        probabilities,
        decoded.edge_ids,
        threshold=0.5,
        decoded_edges=decoded.edges,
    )
    assert "length_f1" in metrics.values


def test_full_synthetic_pipeline_writes_qgis_outputs() -> None:
    config = load_anchor_free_config()
    config["model"]["type"] = "gnn"
    config["model"]["epochs"] = 1
    output_root = Path("anchor_free_test_outputs")
    shutil.rmtree(output_root, ignore_errors=True)
    try:
        result = run_anchor_free_experiment(config, synthetic=True, output_root=output_root)
        assert (result.output_dir / "edge_predictions.geojson").exists()
        assert (result.output_dir / "decoded_network.geojson").exists()
        assert (result.output_dir / "metrics.json").exists()
        assert (result.output_dir / "metrics.csv").exists()
        assert (result.output_dir / "config_resolved.yaml").exists()
        assert "test_f1" in result.metrics.values
        assert "label_5m_f1" in result.metrics.values
    finally:
        shutil.rmtree(output_root, ignore_errors=True)


def test_prepare_features_writes_training_ready_label_buffers() -> None:
    synthetic = make_synthetic_anchor_free_data()
    output_root = Path("anchor_free_test_outputs") / "prepare_features"
    shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)
    roads_path = output_root / "roads.geojson"
    buildings_path = output_root / "buildings.geojson"
    truth_path = output_root / "truth.geojson"
    roads_path.write_text(synthetic.roads.to_json(drop_id=True), encoding="utf-8")
    buildings_path.write_text(synthetic.buildings.to_json(drop_id=True), encoding="utf-8")
    truth_path.write_text(synthetic.utility_truth.to_json(drop_id=True), encoding="utf-8")

    try:
        config = load_anchor_free_config()
        config["data"]["roads_path"] = str(roads_path)
        config["data"]["buildings_path"] = str(buildings_path)
        config["data"]["building_points_path"] = str(buildings_path)
        config["data"]["utility_truth_path"] = str(truth_path)
        config["data"]["dem_path"] = ""
        config["graph"]["target_crs"] = "EPSG:3857"
        config["graph"]["road_class_columns"] = ["road_class"]
        config["graph"]["training_ready_label_buffers_m"] = [10, 5]

        metadata = prepare_features(
            config,
            output_dir=output_root / "features",
            training_ready=True,
            with_labels=True,
            write_geojson=False,
        )
        assert metadata["training_ready"] is True
        table_10m = output_root / "features" / "road_segment_training_table_10m.csv"
        table_5m = output_root / "features" / "road_segment_training_table_5m.csv"
        assert table_10m.exists()
        assert table_5m.exists()
        assert (output_root / "features" / "road_segment_features.csv").exists()
        assert (output_root / "features" / "intersection_features.csv").exists()
        assert (output_root / "features" / "label_buffer_comparison.csv").exists()
        assert (output_root / "features" / "feature_columns.json").exists()
        training_table = pd.read_csv(table_10m)
        assert "segment_id" in training_table.columns
        assert "edge_id" not in training_table.columns
        assert metadata["n_road_segments"] == len(training_table)
    finally:
        shutil.rmtree(output_root.parents[0], ignore_errors=True)
