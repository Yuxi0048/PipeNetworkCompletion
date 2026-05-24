"""Tests for the threshold and connected decoders."""

# Workstream: Claude

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString

from pipe_network_completion.anchor_free.decoder import (
    decode_connected,
    decode_network,
    decode_threshold,
)
from pipe_network_completion.anchor_free.road_graph import build_road_candidate_graph
from pipe_network_completion.anchor_free.synthetic import (
    make_synthetic_anchor_free_data,
)


@pytest.fixture(scope="module")
def synthetic_graph():
    data = make_synthetic_anchor_free_data()
    return build_road_candidate_graph(data.roads, target_crs="EPSG:3857")


def test_threshold_decoder_returns_valid_linestring_geometries(synthetic_graph):
    rng = np.random.default_rng(0)
    probs = rng.uniform(0.0, 1.0, size=len(synthetic_graph.edges))
    decoded = decode_threshold(synthetic_graph, probs, threshold=0.5)
    assert decoded.decoder_type == "threshold"
    for geom in decoded.edges.geometry:
        assert isinstance(geom, LineString)
        assert not geom.is_empty


def test_threshold_decoder_obeys_threshold(synthetic_graph):
    probs = np.full(len(synthetic_graph.edges), 0.6)
    high = decode_threshold(synthetic_graph, probs, threshold=0.5)
    low = decode_threshold(synthetic_graph, probs, threshold=0.9)
    assert len(high.edges) == len(synthetic_graph.edges)
    assert len(low.edges) == 0


def test_connected_decoder_outputs_subset_of_candidates(synthetic_graph):
    rng = np.random.default_rng(0)
    probs = rng.uniform(0.0, 1.0, size=len(synthetic_graph.edges))
    decoded = decode_connected(synthetic_graph, probs, threshold=0.4)
    candidate_ids = set(synthetic_graph.edges["edge_id"].astype(int).tolist())
    decoded_ids = set(decoded.edges["edge_id"].astype(int).tolist())
    assert decoded_ids.issubset(candidate_ids)


def test_decode_network_dispatch(synthetic_graph):
    probs = np.full(len(synthetic_graph.edges), 0.6)
    decoded_t = decode_network(synthetic_graph, probs, {"type": "threshold", "threshold": 0.5})
    decoded_c = decode_network(
        synthetic_graph,
        probs,
        {"type": "connected", "threshold": 0.5, "lambda_length": 0.001},
    )
    assert decoded_t.decoder_type == "threshold"
    assert decoded_c.decoder_type == "connected"
    with pytest.raises(ValueError):
        decode_network(synthetic_graph, probs, {"type": "not-a-decoder"})
