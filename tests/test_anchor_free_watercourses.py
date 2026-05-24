"""Watercourse context features for anchor-free corridor prediction."""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import LineString, Polygon


def test_watercourse_features_are_optional_and_anchor_safe():
    from pipe_network_completion.anchor_free.features import (
        assert_no_anchor_features,
        build_road_edge_features,
    )
    from pipe_network_completion.anchor_free.road_graph import build_road_candidate_graph

    roads = gpd.GeoDataFrame(
        [{"road_class": "local", "geometry": LineString([(0, 0), (100, 0)])}],
        geometry="geometry",
        crs="EPSG:3857",
    )
    graph = build_road_candidate_graph(roads, target_crs="EPSG:3857")

    baseline = build_road_edge_features(graph)
    assert not any(name.startswith("watercourse_") for name in baseline.feature_names)

    drainage = gpd.GeoDataFrame(
        [{"geometry": LineString([(50, -20), (50, 20)])}],
        geometry="geometry",
        crs="EPSG:3857",
    )
    corridors = gpd.GeoDataFrame(
        [{"OVL2_CAT": "ENV_WLC", "geometry": Polygon([(40, -30), (60, -30), (60, 30), (40, 30)])}],
        geometry="geometry",
        crs="EPSG:3857",
    )
    table = build_road_edge_features(
        graph,
        watercourse_drainage_lines_gdf=drainage,
        watercourse_corridors_gdf=corridors,
        watercourse_buffer_m=25,
    )

    assert "watercourse_drainage_nearest_distance_m" in table.feature_names
    assert "watercourse_drainage_length_sum_25m" in table.feature_names
    assert "watercourse_corridor_area_coverage_25m" in table.feature_names
    assert_no_anchor_features(table.feature_names)
    assert float(table.features["watercourse_drainage_length_sum_25m"].iloc[0]) > 0.0


def test_watercourse_config_defaults_are_disabled_until_completeness_checked():
    from pipe_network_completion.anchor_free.config import load_anchor_free_config

    config = load_anchor_free_config()
    assert config["graph"]["use_watercourses"] is False
    assert config["graph"]["watercourse_context_complete"] is False
