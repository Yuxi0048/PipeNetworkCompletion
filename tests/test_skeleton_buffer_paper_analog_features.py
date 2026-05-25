from __future__ import annotations

import geopandas as gpd
from shapely.geometry import LineString, Point


def test_skeleton_buffer_paper_analog_features_are_non_anchor() -> None:
    from pipe_network_completion.anchor_free.features import assert_no_anchor_features
    from scripts.train_skeleton_buffer_gnn import (
        _build_adjacency,
        _build_features,
        _prepare_skeleton_candidates,
    )

    roads = gpd.GeoDataFrame(
        {
            "OVL2_CAT": ["INF_NR", "INF_DR"],
            "ROUTE_TYPE": ["Neighbourhood / local", "District"],
            "geometry": [
                LineString([(0, 0), (100, 0)]),
                LineString([(100, 0), (100, 100)]),
            ],
        },
        crs="EPSG:3857",
    )
    buildings = gpd.GeoDataFrame(
        {
            "geometry": [
                Point(20, 10).buffer(5),
                Point(80, 10).buffer(5),
            ],
        },
        crs="EPSG:3857",
    )
    building_points = gpd.GeoDataFrame(
        {
            "bt_group_residential": [1, 1, 0],
            "bt_group_commercial": [0, 0, 1],
            "geometry": [Point(20, 10), Point(80, 10), Point(105, 80)],
        },
        crs="EPSG:3857",
    )
    built_up = gpd.GeoDataFrame(geometry=[Point(50, 5).buffer(30)], crs="EPSG:3857")
    candidates = _prepare_skeleton_candidates(
        roads,
        gpd.GeoDataFrame(geometry=[], crs="EPSG:3857"),
        include_drainage=False,
    )
    edge_pairs = _build_adjacency(candidates, snap_tolerance_m=1.0)
    features = _build_features(
        candidates,
        roads,
        buildings,
        building_points,
        built_up,
        edge_pairs,
        feature_buffer_m=50.0,
        density_buffer_m=100.0,
        paper_analog_features=True,
    )

    expected = {
        "bearing_bin_0",
        "length_m",
        "same_type_degree",
        "same_type_dead_end",
        "nearest_road_distance_m",
        "nearest_road_distance_bin_0",
        "nearest_road_pos_0_1",
        "nearest_road_side_on",
        "road_candidate_angle_diff_cos",
        "nearest_building_point_bin_1",
        "building_residential_count_buffer",
        "building_commercial_count_buffer",
    }
    assert expected.issubset(set(features.columns))
    assert len(features) == len(candidates)
    assert_no_anchor_features(features.columns)
