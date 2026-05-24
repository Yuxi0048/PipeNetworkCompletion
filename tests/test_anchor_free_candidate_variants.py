"""Candidate support variant utilities."""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import LineString


def test_representability_collapses_duplicate_nearest_candidates():
    from pipe_network_completion.anchor_free.candidate_recall import (
        candidate_representability_metrics,
    )
    from pipe_network_completion.anchor_free.hetero_road_graph import build_hetero_road_graph

    candidates = gpd.GeoDataFrame(
        [
            {"candidate_source": "a", "geometry": LineString([(0, 0), (100, 0)])},
            {"candidate_source": "b", "geometry": LineString([(0, 0), (100, 0)])},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    truth = gpd.GeoDataFrame(
        [{"geometry": LineString([(0, 0), (100, 0)])}],
        geometry="geometry",
        crs="EPSG:3857",
    )
    graph = build_hetero_road_graph(
        candidates,
        target_crs="EPSG:3857",
        keep_columns=["candidate_source"],
    )
    metrics = candidate_representability_metrics(
        graph,
        truth,
        buffers_m=[5],
        sample_spacing_m=10,
    )

    assert metrics["recall_5m"] == 1.0


def test_road_offsets_variant_adds_offset_candidates_without_truth():
    from pipe_network_completion.anchor_free.candidate_variants import (
        ROAD_OFFSET,
        build_candidate_variant_lines,
    )

    roads = gpd.GeoDataFrame(
        [{"road_class": "local", "geometry": LineString([(0, 0), (100, 0)])}],
        geometry="geometry",
        crs="EPSG:3857",
    )
    result = build_candidate_variant_lines(
        roads,
        variant="road_offsets",
        target_crs="EPSG:3857",
        keep_columns=["road_class"],
        offset_distances_m=[10],
    )

    assert ROAD_OFFSET in set(result.candidates["candidate_source"])
    assert "road_offset_distance_m" in result.candidates.columns
    assert "road_offset_side" in result.candidates.columns


def test_pipeline_builds_road_offsets_with_offset_features():
    from pipe_network_completion.anchor_free.config import load_anchor_free_config
    from pipe_network_completion.anchor_free.features import build_road_segment_features
    from pipe_network_completion.anchor_free.pipeline import prepare_anchor_free_inputs

    config = load_anchor_free_config()
    config["graph"]["candidate_graph_type"] = "road_offsets"
    config["graph"]["target_crs"] = "EPSG:3857"
    config["graph"]["road_class_columns"] = ["road_class"]
    config["graph"]["offset_distances_m"] = [10.0]

    graph, buildings, *_ = prepare_anchor_free_inputs(config, synthetic=True)
    features = build_road_segment_features(
        graph,
        buildings_gdf=buildings,
        road_class_columns=["road_class", "candidate_source", "road_offset_side"],
    )

    assert graph.metadata["candidate_graph_type"] == "road_offsets"
    assert "source_index" in graph.road_segments.columns
    assert "road_offset_distance_m" in graph.road_segments.columns
    assert "road_offset_distance_m" in features.feature_names
    assert any(name.startswith("candidate_source_road_offset") for name in features.feature_names)
    assert any(name.startswith("road_offset_side_left") for name in features.feature_names)


def test_offset_lane_predictions_selects_max_probability_lane():
    from scripts.train_anchor_free_aoi_blocks import _offset_lane_predictions

    candidates = gpd.GeoDataFrame(
        [
            {
                "segment_id": 0,
                "aoi_id": "aoi",
                "aoi_split": "test",
                "source_index": 7,
                "candidate_source": "road_backbone",
                "probability": 0.25,
                "geometry": LineString([(0, 0), (100, 0)]),
            },
            {
                "segment_id": 1,
                "aoi_id": "aoi",
                "aoi_split": "test",
                "source_index": 7,
                "candidate_source": "road_offset",
                "road_offset_side": "left",
                "road_offset_distance_m": 10.0,
                "probability": 0.85,
                "geometry": LineString([(0, 10), (100, 10)]),
            },
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )

    lanes = _offset_lane_predictions(candidates, threshold=0.5)

    assert len(lanes) == 1
    assert lanes.iloc[0]["best_offset_lane"] == "left_10m"
    assert bool(lanes.iloc[0]["predicted_presence"]) is True
