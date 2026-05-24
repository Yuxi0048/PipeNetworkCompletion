from __future__ import annotations

import geopandas as gpd
from shapely.geometry import LineString

from pipe_network_completion.anchor_free.candidate_recall import (
    candidate_representability_metrics,
)
from pipe_network_completion.anchor_free.hetero_road_graph import build_hetero_road_graph


def test_candidate_representability_reports_truth_length_recall():
    candidates = gpd.GeoDataFrame(
        {"geometry": [LineString([(0, 0), (100, 0)])]},
        geometry="geometry",
        crs="EPSG:3857",
    )
    truth = gpd.GeoDataFrame(
        {"geometry": [LineString([(0, 3), (100, 3)])]},
        geometry="geometry",
        crs="EPSG:3857",
    )
    graph = build_hetero_road_graph(candidates, target_crs="EPSG:3857")

    metrics = candidate_representability_metrics(
        graph,
        truth,
        buffers_m=[2, 5],
        sample_spacing_m=10,
    )

    assert metrics["candidate_count"] == 1
    assert metrics["recall_2m"] == 0.0
    assert metrics["recall_5m"] == 1.0

