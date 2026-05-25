"""Smoke tests for the Phase 2.A heterogeneous road graph.

# Workstream: Claude

Covers the new RoadSegment + Intersection graph, segment + intersection
feature builders, segment labels, decoders, and the HeteroRoadGNN one-epoch
training loop. Existing roads-as-edges tests are unchanged; the new code
lives alongside the old.
"""

from __future__ import annotations

import importlib.util

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
_PYG_AVAILABLE = importlib.util.find_spec("torch_geometric") is not None


# ---------------------------------------------------------------------------
# AR-AF-2A.2 — the canonical "+" shape (4 LineStrings through one centre)
# ---------------------------------------------------------------------------
def _plus_roads(crs: str = "EPSG:3857") -> gpd.GeoDataFrame:
    """Four roads meeting at the origin."""
    return gpd.GeoDataFrame(
        {
            "name": ["north", "south", "east", "west"],
            "geometry": [
                LineString([(0, 0), (0, 100)]),
                LineString([(0, 0), (0, -100)]),
                LineString([(0, 0), (100, 0)]),
                LineString([(0, 0), (-100, 0)]),
            ],
        },
        crs=crs,
    )


def test_hetero_road_graph_plus_shape_has_one_intersection():
    """AR-AF-2A.2 — "+" shape: 4 RoadSegment nodes, 1 Intersection node,
    every road pair crosses at the centre."""
    from pipe_network_completion.anchor_free.hetero_road_graph import (
        build_hetero_road_graph,
    )

    roads = _plus_roads()
    graph = build_hetero_road_graph(roads, target_crs="EPSG:3857")
    assert len(graph.road_segments) == 4
    # Each road has 2 endpoints: the centre (shared) and an outer tip.
    # Outer tips are 4 distinct points; centre is 1. So 5 intersections.
    assert len(graph.intersections) == 5
    # crosses: 4 LineStrings touch each other only at the centre →
    # each pair (4 choose 2 = 6) appears as an intersection, stored in
    # both directions = 12 entries.
    assert graph.segment_crosses_segment.shape == (2, 12)
    # touches: 4 segments × 2 endpoints each = 8 segment↔intersection edges.
    assert graph.segment_touches_intersection.shape == (2, 8)
    # The centre intersection has degree 4 (all four segments touch it).
    max_deg = int(graph.intersections["degree"].max())
    assert max_deg == 4


def test_hetero_road_graph_records_metadata():
    from pipe_network_completion.anchor_free.hetero_road_graph import (
        build_hetero_road_graph,
    )

    graph = build_hetero_road_graph(_plus_roads(), target_crs="EPSG:3857")
    md = graph.metadata
    assert md["n_road_segments"] == 4
    assert md["n_intersections"] == 5
    assert md["n_crosses_edges"] == 12
    assert md["n_touches_edges"] == 8


# ---------------------------------------------------------------------------
# AR-AF-2A.5 — RoadSegment + Intersection feature tables
# ---------------------------------------------------------------------------
def test_road_segment_features_have_one_row_per_segment_and_pass_guard():
    from pipe_network_completion.anchor_free.features import (
        assert_no_anchor_features,
        build_road_segment_features,
    )
    from pipe_network_completion.anchor_free.hetero_road_graph import (
        build_hetero_road_graph,
    )

    graph = build_hetero_road_graph(_plus_roads(), target_crs="EPSG:3857")
    table = build_road_segment_features(graph)
    assert table.features.shape[0] == len(graph.road_segments)
    assert "length_m" in table.feature_names
    assert "bearing_sin" in table.feature_names
    assert "endpoint_degree_max" in table.feature_names
    # Phase 2.B — clipped-length density column is what the new code emits.
    assert any(
        name.startswith("local_road_clipped_length_density")
        for name in table.feature_names
    )
    assert_no_anchor_features(table.feature_names)


def test_intersection_features_drop_xy_when_include_coords_false():
    from pipe_network_completion.anchor_free.features import (
        build_intersection_features,
    )
    from pipe_network_completion.anchor_free.hetero_road_graph import (
        build_hetero_road_graph,
    )

    graph = build_hetero_road_graph(_plus_roads(), target_crs="EPSG:3857")
    with_coords = build_intersection_features(graph, include_coords=True)
    without = build_intersection_features(graph, include_coords=False)
    assert "intersection_x_norm" in with_coords.feature_names
    assert "intersection_y_norm" in with_coords.feature_names
    assert "intersection_x_norm" not in without.feature_names
    assert "intersection_y_norm" not in without.feature_names
    # Degree features stay either way.
    assert "intersection_degree" in without.feature_names


# ---------------------------------------------------------------------------
# Phase 2.B — clipped-length density: long road grazing buffer
# ---------------------------------------------------------------------------
def test_clipped_density_does_not_count_long_arterial_fully():
    """A 1km arterial grazing a tiny 5m road's 10m buffer must not
    contribute its full 1km to the density numerator."""
    from pipe_network_completion.anchor_free.features import (
        build_road_segment_features,
    )
    from pipe_network_completion.anchor_free.hetero_road_graph import (
        build_hetero_road_graph,
    )

    # Tiny road at the origin; long arterial passing 4m to the north.
    roads = gpd.GeoDataFrame(
        {
            "geometry": [
                LineString([(0, 0), (5, 0)]),         # tiny target road
                LineString([(-500, 4), (500, 4)]),    # long arterial
            ],
        },
        crs="EPSG:3857",
    )
    graph = build_hetero_road_graph(roads, target_crs="EPSG:3857")
    table = build_road_segment_features(graph, road_density_buffer_m=10)
    col = next(
        n
        for n in table.feature_names
        if n.startswith("local_road_clipped_length_density")
    )
    # Tiny road's buffer (10m) intersects only a short slice of the
    # arterial — definitely NOT the full 1000m. We expect the density
    # (length / area) to be modest (< 0.1) instead of huge (>> 1).
    tiny_density = float(table.features.iloc[0][col])
    assert tiny_density < 0.5, (
        f"density for tiny road grazed by 1km arterial = {tiny_density}; "
        "clipped-length math must not count the whole arterial"
    )


# ---------------------------------------------------------------------------
# AR-AF-2A.6 — RoadSegment labels
# ---------------------------------------------------------------------------
def test_road_segment_labels_match_truth_lines():
    from pipe_network_completion.anchor_free.hetero_road_graph import (
        build_hetero_road_graph,
    )
    from pipe_network_completion.anchor_free.labels import (
        label_road_segments_from_utility_lines,
    )

    roads = _plus_roads()
    graph = build_hetero_road_graph(roads, target_crs="EPSG:3857")
    # Truth = a sewer line that exactly follows the north and south roads.
    truth = gpd.GeoDataFrame(
        geometry=[LineString([(0, -100), (0, 100)])], crs="EPSG:3857"
    )
    labels = label_road_segments_from_utility_lines(
        graph, truth, label_buffer_m=5.0, label_overlap_threshold=0.5
    )
    assert len(labels.labels) == 4
    # North + South segments overlap nearly 100% of their length; East +
    # West only touch the truth at the centre point (overlap ~ 0).
    pos_count = int(labels.y.sum())
    assert 2 <= pos_count <= 3, (
        f"expected 2 positive segments (north+south) got {pos_count}"
    )


# ---------------------------------------------------------------------------
# AR-AF-2A.8 — segment decoders
# ---------------------------------------------------------------------------
def test_decode_threshold_segments_selects_above_threshold():
    from pipe_network_completion.anchor_free.decoder import decode_threshold_segments
    from pipe_network_completion.anchor_free.hetero_road_graph import (
        build_hetero_road_graph,
    )

    graph = build_hetero_road_graph(_plus_roads(), target_crs="EPSG:3857")
    probs = np.array([0.9, 0.1, 0.8, 0.2])  # 4 segments
    out = decode_threshold_segments(graph, probs, threshold=0.5)
    assert out.decoder_type == "threshold"
    assert set(out.segment_ids.tolist()) == {0, 2}


def test_decode_connected_segments_returns_subset():
    from pipe_network_completion.anchor_free.decoder import decode_connected_segments
    from pipe_network_completion.anchor_free.hetero_road_graph import (
        build_hetero_road_graph,
    )

    graph = build_hetero_road_graph(_plus_roads(), target_crs="EPSG:3857")
    probs = np.array([0.9, 0.8, 0.85, 0.7])  # all above threshold
    out = decode_connected_segments(graph, probs, threshold=0.5)
    selected = set(out.segment_ids.tolist())
    assert selected.issubset({0, 1, 2, 3})
    assert len(selected) >= 2  # MST over the centre connects the kept set


def test_decode_connected_segments_adds_low_probability_connector():
    from pipe_network_completion.anchor_free.decoder import decode_connected_segments
    from pipe_network_completion.anchor_free.hetero_road_graph import (
        build_hetero_road_graph,
    )

    roads = gpd.GeoDataFrame(
        {
            "geometry": [
                LineString([(0, 0), (100, 0)]),
                LineString([(100, 0), (200, 0)]),
                LineString([(200, 0), (300, 0)]),
            ],
        },
        crs="EPSG:3857",
    )
    graph = build_hetero_road_graph(roads, target_crs="EPSG:3857")
    probs = np.array([0.95, 0.05, 0.95])
    out = decode_connected_segments(graph, probs, threshold=0.9)
    assert set(out.segment_ids.tolist()) == {0, 1, 2}


# ---------------------------------------------------------------------------
# AR-AF-2A.7 — HeteroRoadGNN one-epoch forward + backward
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (_TORCH_AVAILABLE and _PYG_AVAILABLE),
    reason="torch and torch_geometric required for HeteroRoadGNN.",
)
def test_hetero_road_gnn_one_epoch_runs():
    from pipe_network_completion.anchor_free.features import (
        build_intersection_features,
        build_road_segment_features,
    )
    from pipe_network_completion.anchor_free.hetero_road_graph import (
        build_hetero_road_graph,
    )
    from pipe_network_completion.anchor_free.labels import (
        label_road_segments_from_utility_lines,
    )
    from pipe_network_completion.anchor_free.model import (
        build_hetero_pyg_data,
        train_hetero_road_gnn,
    )

    roads = _plus_roads()
    graph = build_hetero_road_graph(roads, target_crs="EPSG:3857")
    truth = gpd.GeoDataFrame(
        geometry=[LineString([(0, -100), (0, 100)])], crs="EPSG:3857"
    )
    seg_feat = build_road_segment_features(graph)
    inter_feat = build_intersection_features(graph)
    labels = label_road_segments_from_utility_lines(
        graph, truth, label_buffer_m=5.0, label_overlap_threshold=0.5
    )
    pyg = build_hetero_pyg_data(graph, seg_feat, inter_feat, labels=labels.y)
    train_index = np.arange(len(labels.y))
    result = train_hetero_road_gnn(
        pyg,
        train_index=train_index,
        seed=0,
        epochs=2,
        hidden_dim=8,
        num_layers=2,
    )
    assert result.probabilities.shape == labels.y.shape
    assert np.isfinite(result.losses[0])
    assert result.device in {"cpu", "cuda", "cuda:0"}


@pytest.mark.skipif(
    not (_TORCH_AVAILABLE and _PYG_AVAILABLE),
    reason="torch and torch_geometric required for HeteroRoadGNN.",
)
def test_hetero_road_gnn_layer_type_variants_run_one_epoch():
    from pipe_network_completion.anchor_free.features import (
        build_intersection_features,
        build_road_segment_features,
    )
    from pipe_network_completion.anchor_free.hetero_road_graph import (
        build_hetero_road_graph,
    )
    from pipe_network_completion.anchor_free.labels import (
        label_road_segments_from_utility_lines,
    )
    from pipe_network_completion.anchor_free.model import (
        build_hetero_pyg_data,
        train_hetero_road_gnn,
    )

    graph = build_hetero_road_graph(_plus_roads(), target_crs="EPSG:3857")
    truth = gpd.GeoDataFrame(
        geometry=[LineString([(0, -100), (0, 100)])], crs="EPSG:3857"
    )
    seg_feat = build_road_segment_features(graph)
    inter_feat = build_intersection_features(graph)
    labels = label_road_segments_from_utility_lines(
        graph, truth, label_buffer_m=5.0, label_overlap_threshold=0.5
    )
    pyg = build_hetero_pyg_data(graph, seg_feat, inter_feat, labels=labels.y)
    train_index = np.arange(len(labels.y))

    for layer_type in ("sage", "gat", "graphconv"):
        result = train_hetero_road_gnn(
            pyg,
            train_index=train_index,
            seed=0,
            epochs=1,
            hidden_dim=8,
            num_layers=1,
            device="cpu",
            layer_type=layer_type,
        )
        assert result.probabilities.shape == labels.y.shape
        assert np.isfinite(result.losses[0])
