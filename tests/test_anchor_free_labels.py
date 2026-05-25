"""Tests for the anchor-free road-edge label generator."""

# Workstream: Claude

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import LineString

from pipe_network_completion.anchor_free.labels import (
    FIXED_ROAD_OFFSET_LANES,
    label_road_offset_lanes_from_utility_lines,
    label_road_edges_from_utility_lines,
)
from pipe_network_completion.anchor_free.candidate_variants import build_candidate_variant_lines
from pipe_network_completion.anchor_free.hetero_road_graph import build_hetero_road_graph
from pipe_network_completion.anchor_free.road_graph import build_road_candidate_graph
from pipe_network_completion.anchor_free.synthetic import (
    make_synthetic_anchor_free_data,
)


def test_labels_have_both_positive_and_negative_edges():
    data = make_synthetic_anchor_free_data()
    graph = build_road_candidate_graph(data.roads, target_crs="EPSG:3857")
    labels = label_road_edges_from_utility_lines(
        graph,
        data.utility_truth,
        label_buffer_m=10.0,
        label_overlap_threshold=0.25,
    )
    y = labels.y
    # Need at least one positive and one negative; otherwise downstream
    # metrics and the stratified split degenerate.
    assert int(y.sum()) > 0
    assert int((y == 0).sum()) > 0
    # One label per candidate edge.
    assert len(labels.labels) == len(graph.edges)


def test_labels_overlap_ratio_is_between_zero_and_one():
    data = make_synthetic_anchor_free_data()
    graph = build_road_candidate_graph(data.roads, target_crs="EPSG:3857")
    labels = label_road_edges_from_utility_lines(graph, data.utility_truth)
    ratios = labels.labels["overlap_ratio"].values
    assert ratios.min() >= 0.0
    assert ratios.max() <= 1.0


def test_labels_with_empty_truth_yield_all_negative():
    data = make_synthetic_anchor_free_data()
    graph = build_road_candidate_graph(data.roads, target_crs="EPSG:3857")
    empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")
    labels = label_road_edges_from_utility_lines(graph, empty)
    assert int(labels.y.sum()) == 0
    assert len(labels.labels) == len(graph.edges)


def _single_road_offset_graph():
    roads = gpd.GeoDataFrame(
        [{"road_class": "local", "geometry": LineString([(0, 0), (100, 0)])}],
        geometry="geometry",
        crs="EPSG:3857",
    )
    candidates = build_candidate_variant_lines(
        roads,
        variant="road_offsets",
        target_crs="EPSG:3857",
        keep_columns=["road_class"],
        offset_distances_m=[15, 30],
    )
    return build_hetero_road_graph(
        candidates.candidates,
        target_crs="EPSG:3857",
        keep_columns=[
            "candidate_source",
            "road_offset_side",
            "road_offset_distance_m",
            "source_index",
        ],
    )


def test_road_offset_lane_labels_pick_nearest_fixed_lane():
    graph = _single_road_offset_graph()
    truth = gpd.GeoDataFrame(
        [{"geometry": LineString([(0, 15), (100, 15)])}],
        geometry="geometry",
        crs="EPSG:3857",
    )

    labels = label_road_offset_lanes_from_utility_lines(
        graph,
        truth,
        sample_spacing_m=10,
        max_assignment_distance_m=20,
        min_assigned_truth_length_m=1,
    )

    assert labels.lane_names == FIXED_ROAD_OFFSET_LANES
    assert len(labels.labels) == 1
    row = labels.labels.iloc[0]
    assert row["y"] == 1
    assert row["lane_name"] == "left_15m"
    assert row["lane_class"] == FIXED_ROAD_OFFSET_LANES.index("left_15m")
    assert row["truth_length_left_15m_m"] > 0


def test_road_offset_lane_labels_empty_truth_is_negative():
    graph = _single_road_offset_graph()
    truth = gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")

    labels = label_road_offset_lanes_from_utility_lines(
        graph,
        truth,
        sample_spacing_m=10,
        max_assignment_distance_m=20,
    )

    assert int(labels.y.sum()) == 0
    assert int(labels.lane_class[0]) == -1
    assert labels.labels.iloc[0]["lane_name"] == "none"
