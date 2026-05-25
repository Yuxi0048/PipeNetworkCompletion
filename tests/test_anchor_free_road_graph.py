"""Smoke tests for the anchor-free road candidate graph."""

# Workstream: Claude

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import LineString

from pipe_network_completion.anchor_free.road_graph import (
    build_road_candidate_graph,
    edge_midpoints,
)
from pipe_network_completion.anchor_free.synthetic import (
    make_synthetic_anchor_free_data,
)


def test_road_candidate_graph_from_synthetic_grid_has_no_anchor_inputs():
    data = make_synthetic_anchor_free_data()
    graph = build_road_candidate_graph(data.roads, target_crs="EPSG:3857")

    # Anchor-free requirement: nothing in the graph schema mentions anchor-like
    # entities. The presence of these columns would itself be an information
    # leak from the original anchor-based pipeline.
    forbidden = {"MH", "mh", "manhole", "valve", "pole", "anchor", "facility"}
    assert forbidden.isdisjoint(set(graph.nodes.columns))
    assert forbidden.isdisjoint(set(graph.edges.columns))
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0


def test_road_candidate_graph_dedupes_shared_endpoints():
    a = LineString([(0.0, 0.0), (10.0, 0.0)])
    b = LineString([(10.0, 0.0), (10.0, 10.0)])
    c = LineString([(10.0, 10.0), (0.0, 10.0)])
    gdf = gpd.GeoDataFrame({"geometry": [a, b, c]}, crs="EPSG:3857")
    graph = build_road_candidate_graph(gdf)
    # Three segments share two intersection nodes, so the candidate graph
    # should have exactly 4 nodes and 3 edges.
    assert len(graph.edges) == 3
    assert len(graph.nodes) == 4
    assert int(graph.nodes["degree"].max()) >= 2


def test_road_candidate_graph_edge_lengths_sum_inputs():
    a = LineString([(0.0, 0.0), (3.0, 0.0)])
    b = LineString([(3.0, 0.0), (3.0, 4.0)])
    gdf = gpd.GeoDataFrame({"geometry": [a, b]}, crs="EPSG:3857")
    graph = build_road_candidate_graph(gdf)
    assert pytest.approx(graph.edges["length_m"].sum(), rel=1e-6) == 7.0


def test_edge_midpoints_shape_matches_edge_count():
    data = make_synthetic_anchor_free_data()
    graph = build_road_candidate_graph(data.roads, target_crs="EPSG:3857")
    midpoints = edge_midpoints(graph)
    assert midpoints.shape == (len(graph.edges), 2)
