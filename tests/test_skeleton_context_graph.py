from __future__ import annotations

import importlib.util

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point


_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
_PYG_AVAILABLE = importlib.util.find_spec("torch_geometric") is not None


def _toy_roads() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "OVL2_CAT": ["INF_NR", "INF_NR", "INF_DR"],
            "ROUTE_TYPE": ["Neighbourhood / local", "Neighbourhood / local", "District"],
            "geometry": [
                LineString([(0, 0), (100, 0)]),
                LineString([(100, 0), (200, 0)]),
                LineString([(100, -50), (100, 50)]),
            ],
        },
        crs="EPSG:3857",
    )


def _toy_buildings() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "building_type_group": ["residential", "residential", "commercial"],
            "bt_group_residential": [1, 1, 0],
            "bt_group_commercial": [0, 0, 1],
            "footprint_area_m2": [120.0, 140.0, 500.0],
            "footprint_perimeter_m": [44.0, 48.0, 96.0],
            "geometry": [Point(20, 10), Point(35, 12), Point(160, 20)],
        },
        crs="EPSG:3857",
    )


def test_skeleton_context_graph_has_required_relation_families() -> None:
    from pipe_network_completion.anchor_free.skeleton_context_graph import (
        build_skeleton_context_features,
        build_skeleton_context_graph,
    )

    graph = build_skeleton_context_graph(
        roads=_toy_roads(),
        building_points=_toy_buildings(),
        target_crs="EPSG:3857",
        building_building_radius_m=50,
        building_skeleton_radius_m=60,
        building_road_radius_m=60,
        skeleton_road_radius_m=5,
    )

    assert len(graph.skeleton_segments) == 3
    assert len(graph.road_segments) == 3
    assert len(graph.buildings) == 3
    assert graph.skeleton_adjacent_skeleton.shape[1] > 0
    assert graph.building_near_building.shape[1] > 0
    assert graph.building_near_skeleton.shape[1] > 0
    assert graph.building_near_road.shape[1] > 0
    assert graph.skeleton_near_road.shape[1] > 0
    assert graph.road_adjacent_road.shape[1] > 0

    features = build_skeleton_context_features(graph)
    assert "bearing_bin_0" in features["SkeletonSegment"].columns
    assert "candidate_source_road" in features["SkeletonSegment"].columns
    assert "building_type_group_residential" in features["Building"].columns
    assert "OVL2_CAT_INF_NR" in features["RoadSegment"].columns


@pytest.mark.skipif(
    not (_TORCH_AVAILABLE and _PYG_AVAILABLE),
    reason="torch and torch_geometric required for hetero GNN smoke test.",
)
def test_skeleton_context_hetero_gnn_one_epoch_runs() -> None:
    from pipe_network_completion.anchor_free.skeleton_context_graph import (
        build_skeleton_context_features,
        build_skeleton_context_graph,
    )
    from scripts.train_skeleton_context_hetero_gnn import (
        _build_pyg_data,
        _label_skeleton_segments,
        _train_model,
    )

    graph = build_skeleton_context_graph(
        roads=_toy_roads(),
        building_points=_toy_buildings(),
        target_crs="EPSG:3857",
        building_building_radius_m=50,
        building_skeleton_radius_m=60,
        building_road_radius_m=60,
        skeleton_road_radius_m=5,
    )
    features = build_skeleton_context_features(graph)
    truth = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (100, 0)])],
        crs="EPSG:3857",
    )
    labels = _label_skeleton_segments(
        graph.skeleton_segments,
        truth,
        label_buffer_m=5.0,
        min_truth_length_m=10.0,
        min_overlap_ratio=0.25,
    )
    data = _build_pyg_data(graph, features, labels)
    model, probabilities, losses, device = _train_model(
        data,
        train_index=np.arange(len(labels)),
        seed=0,
        hidden_dim=8,
        num_layers=2,
        dropout=0.0,
        lr=0.01,
        weight_decay=0.0,
        epochs=2,
        device="cpu",
        layer_type="sage",
        gat_heads=1,
    )

    assert probabilities.shape == labels["y"].shape
    assert np.isfinite(losses[0])
    assert device == "cpu"
    assert model is not None
