from __future__ import annotations

import inspect
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point


def _offroad_fixture():
    roads = gpd.GeoDataFrame(
        [{"road_class": "local", "geometry": LineString([(0, 0), (100, 0)])}],
        geometry="geometry",
        crs="EPSG:3857",
    )
    buildings = gpd.GeoDataFrame(
        [
            {"building_id": 1, "geometry": Point(0, 100)},
            {"building_id": 2, "geometry": Point(100, 100)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    truth = gpd.GeoDataFrame(
        [{"truth_id": 1, "geometry": LineString([(0, 100), (100, 100)])}],
        geometry="geometry",
        crs="EPSG:3857",
    )
    return roads, buildings, truth


def test_hybrid_builder_has_no_truth_input_and_builds_sources():
    from pipe_network_completion.anchor_free.hybrid_candidate_graph import (
        BUILDING_ACCESS,
        DEMAND_KNN,
        DEMAND_MST,
        ROAD_BACKBONE,
        build_hybrid_candidate_lines,
    )

    assert "utility_truth" not in inspect.signature(build_hybrid_candidate_lines).parameters
    roads, buildings, _ = _offroad_fixture()
    result = build_hybrid_candidate_lines(
        roads,
        buildings_gdf=buildings,
        target_crs="EPSG:3857",
        demand_cluster_grid_m=50,
        nearest_road_max_distance_m=150,
        knn_k=1,
        knn_max_distance_m=150,
    )

    sources = set(result.candidates["candidate_source"].tolist())
    assert ROAD_BACKBONE in sources
    assert BUILDING_ACCESS in sources
    assert DEMAND_KNN in sources
    assert result.metadata["n_demand_clusters"] == 2
    mst_only = build_hybrid_candidate_lines(
        roads,
        buildings_gdf=buildings,
        target_crs="EPSG:3857",
        demand_cluster_grid_m=50,
        knn_k=1,
        knn_max_distance_m=150,
        include_road_backbone=False,
        include_building_access=False,
        include_demand_knn=False,
        include_demand_mst=True,
    )
    assert DEMAND_MST in set(mst_only.candidates["candidate_source"].tolist())


def test_hybrid_candidate_can_represent_offroad_truth_when_road_only_cannot():
    from pipe_network_completion.anchor_free.hetero_road_graph import (
        build_hetero_road_graph,
    )
    from pipe_network_completion.anchor_free.hybrid_candidate_graph import (
        build_hybrid_candidate_lines,
    )
    from pipe_network_completion.anchor_free.labels import (
        label_road_segments_from_utility_lines,
    )

    roads, buildings, truth = _offroad_fixture()
    road_graph = build_hetero_road_graph(roads, target_crs="EPSG:3857")
    road_labels = label_road_segments_from_utility_lines(
        road_graph, truth, label_buffer_m=10, label_overlap_threshold=0.25
    )
    assert int(road_labels.y.sum()) == 0

    hybrid = build_hybrid_candidate_lines(
        roads,
        buildings_gdf=buildings,
        target_crs="EPSG:3857",
        include_road_backbone=False,
        include_building_access=False,
        include_demand_knn=True,
        include_demand_mst=False,
        demand_cluster_grid_m=50,
        knn_k=1,
        knn_max_distance_m=150,
    )
    graph = build_hetero_road_graph(
        hybrid.candidates,
        target_crs="EPSG:3857",
        keep_columns=["candidate_source"],
    )
    labels = label_road_segments_from_utility_lines(
        graph, truth, label_buffer_m=10, label_overlap_threshold=0.25
    )
    assert int(labels.y.sum()) >= 1


def test_hybrid_candidate_features_include_source_family():
    from pipe_network_completion.anchor_free.features import build_road_segment_features
    from pipe_network_completion.anchor_free.hetero_road_graph import (
        build_hetero_road_graph,
    )
    from pipe_network_completion.anchor_free.hybrid_candidate_graph import (
        build_hybrid_candidate_lines,
    )

    roads, buildings, _ = _offroad_fixture()
    hybrid = build_hybrid_candidate_lines(
        roads,
        buildings_gdf=buildings,
        target_crs="EPSG:3857",
        demand_cluster_grid_m=50,
        nearest_road_max_distance_m=150,
        knn_k=1,
        knn_max_distance_m=150,
    )
    graph = build_hetero_road_graph(
        hybrid.candidates,
        target_crs="EPSG:3857",
        keep_columns=[
            "candidate_source",
            "candidate_weight",
            "demand_u",
            "demand_v",
            "nearest_road_distance_m",
        ],
    )
    features = build_road_segment_features(
        graph,
        buildings_gdf=buildings,
        road_class_columns=["candidate_source"],
    )

    assert features.features.shape[0] == len(graph.road_segments)
    assert any(c.startswith("candidate_source_") for c in features.feature_names)
    assert "candidate_weight" in features.feature_names
    assert "nearest_road_distance_m" in features.feature_names


def test_synthetic_pipeline_runs_with_hybrid_candidate_graph():
    from pipe_network_completion.anchor_free.config import load_anchor_free_config
    from pipe_network_completion.anchor_free.pipeline import run_anchor_free_experiment

    output_root = Path("anchor_free_test_outputs") / "hybrid_pipeline"
    shutil.rmtree(output_root, ignore_errors=True)
    try:
        config = load_anchor_free_config()
        config["experiment_name"] = "pytest_hybrid_candidate_graph"
        config["graph"]["candidate_graph_type"] = "hybrid"
        config["graph"]["target_crs"] = "EPSG:3857"
        config["graph"]["road_class_columns"] = ["road_class"]
        config["graph"]["demand_cluster_grid_m"] = 100
        config["graph"]["demand_knn_k"] = 2
        config["graph"]["demand_knn_max_distance_m"] = 175
        config["model"]["type"] = "random_forest"
        config["evaluation"]["representability_sample_spacing_m"] = 50

        result = run_anchor_free_experiment(
            config, synthetic=True, output_root=output_root
        )

        assert result.graph.metadata["candidate_graph_type"] == "hybrid"
        assert "candidate_source_road_backbone_count" in result.metrics.values
        assert "candidate_representability_10m_truth_fraction" in result.metrics.values
        assert np.isfinite(result.metrics.values["candidate_total_length"])
    finally:
        shutil.rmtree(output_root, ignore_errors=True)
