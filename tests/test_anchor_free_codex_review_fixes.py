"""Tests for the Stage 1.5 cleanup fixes from Codex's current_codebase_review.

# Workstream: Claude

Covers:
* P1 — sewer/water/steiner decoder types must fail loudly
* P1 — pipeline writes both inference-only and evaluation GeoJSONs
* P2 — _read_vector_many reprojects all input frames to the first frame's CRS
* P2 — Brisbane CLI script wires building_points by default
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


# ---------------------------------------------------------------------------
# P1 — unsupported decoder names
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("reserved_name", ["sewer", "water", "steiner"])
def test_decoder_reserved_names_raise(reserved_name):
    """Codex P1: reserved decoder names must not silently alias to connected."""
    from pipe_network_completion.anchor_free.decoder import decode_network
    from pipe_network_completion.anchor_free.road_graph import build_road_candidate_graph
    from pipe_network_completion.anchor_free.synthetic import (
        make_synthetic_anchor_free_data,
    )

    data = make_synthetic_anchor_free_data()
    graph = build_road_candidate_graph(data.roads, target_crs="EPSG:3857")
    probs = np.full(len(graph.edges), 0.6)
    with pytest.raises(NotImplementedError) as exc:
        decode_network(graph, probs, {"type": reserved_name, "threshold": 0.5})
    assert reserved_name in str(exc.value)


def test_decoder_unknown_name_still_raises_value_error():
    from pipe_network_completion.anchor_free.decoder import decode_network
    from pipe_network_completion.anchor_free.road_graph import build_road_candidate_graph
    from pipe_network_completion.anchor_free.synthetic import (
        make_synthetic_anchor_free_data,
    )

    data = make_synthetic_anchor_free_data()
    graph = build_road_candidate_graph(data.roads, target_crs="EPSG:3857")
    probs = np.full(len(graph.edges), 0.6)
    with pytest.raises(ValueError):
        decode_network(graph, probs, {"type": "not-a-real-decoder"})


# ---------------------------------------------------------------------------
# P1 — inference-only vs evaluation GeoJSON split
# ---------------------------------------------------------------------------
def test_pipeline_writes_inference_only_and_evaluation_geojsons(tmp_path):
    """Codex P1: inference-only geojson must exclude truth-derived columns."""
    import json

    from pipe_network_completion.anchor_free.config import load_anchor_free_config
    from pipe_network_completion.anchor_free.pipeline import run_anchor_free_experiment

    cfg = load_anchor_free_config(None)
    cfg["model"]["type"] = "logistic_regression"
    cfg["experiment_name"] = "split_geojson_test"
    result = run_anchor_free_experiment(cfg, synthetic=True, output_root=tmp_path)

    out_dir = Path(result.output_dir)
    inference_path = out_dir / "edge_predictions_inference_only.geojson"
    eval_path = out_dir / "edge_predictions_for_evaluation.geojson"
    assert inference_path.exists(), "inference-only geojson must be written"
    assert eval_path.exists(), "evaluation geojson must be written"

    inf_payload = json.loads(inference_path.read_text())
    if inf_payload["features"]:
        props = inf_payload["features"][0]["properties"]
        for forbidden in ("y", "overlap_length", "overlap_ratio"):
            assert forbidden not in props, (
                f"inference-only geojson must not contain label/overlap "
                f"diagnostic {forbidden!r}; got props={list(props)}"
            )
        # Probability MUST be present.
        assert "probability" in props

    eval_payload = json.loads(eval_path.read_text())
    if eval_payload["features"]:
        props = eval_payload["features"][0]["properties"]
        for required in ("probability", "y", "overlap_ratio"):
            assert required in props


# ---------------------------------------------------------------------------
# P2 — multi-CRS utility-truth reprojection
# ---------------------------------------------------------------------------
def test_read_vector_many_reprojects_to_first_crs(tmp_path):
    """Codex P2: truth files in different CRSs must be reprojected before
    concatenation; otherwise labels are silently wrong."""
    import geopandas as gpd
    from shapely.geometry import LineString

    from pipe_network_completion.anchor_free.pipeline import _read_vector_many

    # File 1 in EPSG:3857 (web mercator).
    gdf1 = gpd.GeoDataFrame(
        {"src": [1]},
        geometry=[LineString([(0, 0), (1000, 0)])],
        crs="EPSG:3857",
    )
    # File 2 in EPSG:4326 (lat/lon). Same content, different CRS.
    gdf2 = gpd.GeoDataFrame(
        {"src": [2]},
        geometry=[LineString([(151.0, -27.5), (151.001, -27.5)])],
        crs="EPSG:4326",
    )
    p1 = tmp_path / "truth1.gpkg"
    p2 = tmp_path / "truth2.gpkg"
    gdf1.to_file(p1, driver="GPKG")
    gdf2.to_file(p2, driver="GPKG")

    combined = _read_vector_many([str(p1), str(p2)])
    assert combined.crs is not None
    # Resulting CRS == first file's CRS.
    assert str(combined.crs) == str(gdf1.crs)
    # And the second frame's coordinates should be huge web-mercator numbers
    # (~16800000), not the small lat/lon they came in as. If reprojection
    # silently failed, the coords would still be ~151.
    second_xs = combined.iloc[1].geometry.coords[0][0]
    assert abs(second_xs) > 1_000_000, (
        f"second frame coordinates {second_xs} look unprojected; "
        "_read_vector_many silently kept lat/lon values"
    )


# ---------------------------------------------------------------------------
# P2 — Brisbane CLI script wires building points
# ---------------------------------------------------------------------------
def test_brisbane_script_exports_building_points_default():
    """Codex P2: train_anchor_free_brisbane.py must include building points
    in its default real-data feature set."""
    from scripts import train_anchor_free_brisbane as bri

    # Default path constant exists.
    assert hasattr(bri, "DEFAULT_BUILDING_POINTS")
    assert "building_points" in str(bri.DEFAULT_BUILDING_POINTS)

    # CLI arg accepts both opt-in and opt-out.
    parser_args = bri.parse_args.__wrapped__ if hasattr(
        bri.parse_args, "__wrapped__"
    ) else None
    # Use the runtime parser instead — argparse builds it inside parse_args.
    import argparse
    import sys

    saved = sys.argv
    try:
        sys.argv = ["prog", "--no-building-points"]
        args = bri.parse_args()
        assert args.no_building_points is True
        assert args.building_points == bri.DEFAULT_BUILDING_POINTS
    finally:
        sys.argv = saved


def test_brisbane_fast_patch_covers_active_segment_labeler():
    """The main pipeline now labels RoadSegment nodes, so the Brisbane speed
    patch must target the segment label function, not only the legacy edge
    label function."""
    from pipe_network_completion.anchor_free import labels as af_labels
    from pipe_network_completion.anchor_free import pipeline as af_pipeline
    from scripts import train_anchor_free_brisbane as bri

    old_labels_segment = af_labels.label_road_segments_from_utility_lines
    old_pipeline_segment = af_pipeline.label_road_segments_from_utility_lines
    try:
        bri.apply_fast_patches()
        assert (
            af_labels.label_road_segments_from_utility_lines
            is bri._fast_label_road_segments_from_utility_lines
        )
        assert (
            af_pipeline.label_road_segments_from_utility_lines
            is bri._fast_label_road_segments_from_utility_lines
        )
    finally:
        af_labels.label_road_segments_from_utility_lines = old_labels_segment
        af_pipeline.label_road_segments_from_utility_lines = old_pipeline_segment


# ---------------------------------------------------------------------------
# P2 — Coordinate-ablation tests already exist in test_anchor_free_no_coords.py;
# this is the integration-level guard Codex called out.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not _TORCH_AVAILABLE, reason="GNN integration test needs torch."
)
def test_run_anchor_free_experiment_respects_include_node_coords_false(tmp_path):
    """Codex P2: full pipeline path must honor model.include_node_coords=False."""
    from pipe_network_completion.anchor_free.config import load_anchor_free_config
    from pipe_network_completion.anchor_free.pipeline import run_anchor_free_experiment

    cfg = load_anchor_free_config(None)
    cfg["model"]["type"] = "gnn"
    cfg["model"]["epochs"] = 2
    cfg["model"]["hidden_dim"] = 8
    cfg["model"]["num_layers"] = 2
    cfg["model"]["include_node_coords"] = False
    cfg["experiment_name"] = "include_node_coords_false_smoke"
    result = run_anchor_free_experiment(cfg, synthetic=True, output_root=tmp_path)
    assert result.probabilities.shape == result.labels.y.shape


@pytest.mark.skipif(
    not _TORCH_AVAILABLE, reason="GNN integration test needs torch."
)
def test_run_anchor_free_experiment_passes_scaled_segment_features_to_gnn(
    tmp_path,
    monkeypatch,
):
    """Regression guard: hetero GNN must receive train-split standardized
    RoadSegment features, not raw length/distance feature scales."""
    from pipe_network_completion.anchor_free.config import load_anchor_free_config
    from pipe_network_completion.anchor_free import pipeline as af_pipeline

    original_builder = af_pipeline.build_hetero_pyg_data
    captured = {}

    def wrapped_builder(graph, segment_features, intersection_features, labels=None):
        captured["segment_features"] = segment_features.features.copy()
        return original_builder(graph, segment_features, intersection_features, labels=labels)

    monkeypatch.setattr(af_pipeline, "build_hetero_pyg_data", wrapped_builder)

    cfg = load_anchor_free_config(None)
    cfg["model"]["type"] = "gnn"
    cfg["model"]["epochs"] = 1
    cfg["model"]["hidden_dim"] = 8
    cfg["model"]["num_layers"] = 1
    cfg["experiment_name"] = "scaled_segment_features_smoke"
    result = af_pipeline.run_anchor_free_experiment(cfg, synthetic=True, output_root=tmp_path)

    train_features = captured["segment_features"].iloc[result.train_index]
    assert float(train_features.mean(axis=0).abs().max()) < 1e-6
