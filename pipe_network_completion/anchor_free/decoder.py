"""Decode edge probabilities into inspectable utility-network geometries."""

# Workstream: Codex

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import geopandas as gpd
import networkx as nx
import numpy as np

from pipe_network_completion.anchor_free.road_graph import RoadCandidateGraph


@dataclass(frozen=True)
class DecodedNetwork:
    edges: gpd.GeoDataFrame
    decoder_type: str
    threshold: float

    @property
    def edge_ids(self) -> np.ndarray:
        if self.edges.empty:
            return np.array([], dtype=int)
        return self.edges["edge_id"].to_numpy(dtype=int)


def _edges_with_probabilities(
    graph: RoadCandidateGraph,
    probabilities: np.ndarray,
) -> gpd.GeoDataFrame:
    edges = graph.edges.sort_values("edge_id").copy()
    probabilities = np.asarray(probabilities, dtype=float)
    if len(edges) != len(probabilities):
        raise ValueError(
            f"Expected {len(edges)} probabilities, received {len(probabilities)}."
        )
    edges["probability"] = probabilities
    return edges


def decode_threshold(
    graph: RoadCandidateGraph,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> DecodedNetwork:
    """Select all candidate edges whose probability exceeds the threshold."""

    edges = _edges_with_probabilities(graph, probabilities)
    selected = edges[edges["probability"] >= float(threshold)].copy()
    return DecodedNetwork(selected, decoder_type="threshold", threshold=float(threshold))


def _costed_simple_graph(
    edges: gpd.GeoDataFrame,
    *,
    lambda_length: float,
    eps: float = 1e-6,
) -> nx.Graph:
    graph = nx.Graph()
    for edge in edges.itertuples(index=False):
        probability = float(edge.probability)
        cost = -np.log(max(probability, eps)) + float(lambda_length) * float(edge.length_m)
        u = int(edge.u)
        v = int(edge.v)
        existing = graph.get_edge_data(u, v)
        if existing is None or cost < existing["cost"]:
            graph.add_edge(
                u,
                v,
                cost=float(cost),
                edge_id=int(edge.edge_id),
                probability=probability,
            )
    return graph


def _terminal_nodes(edges: gpd.GeoDataFrame, threshold: float) -> set[int]:
    selected = edges[edges["probability"] >= float(threshold)]
    if selected.empty and not edges.empty:
        selected = edges.nlargest(1, "probability")
    terminals: set[int] = set()
    for edge in selected.itertuples(index=False):
        terminals.add(int(edge.u))
        terminals.add(int(edge.v))
    return terminals


def _steiner_like_edge_ids(
    cost_graph: nx.Graph,
    terminals: set[int],
) -> set[int]:
    selected_edge_ids: set[int] = set()
    for component_nodes in nx.connected_components(cost_graph):
        component_terminals = sorted(terminals.intersection(component_nodes))
        if len(component_terminals) < 2:
            continue
        metric_graph = nx.Graph()
        paths: dict[tuple[int, int], list[int]] = {}
        for i, source in enumerate(component_terminals):
            lengths, shortest_paths = nx.single_source_dijkstra(
                cost_graph.subgraph(component_nodes),
                source,
                weight="cost",
            )
            for target in component_terminals[i + 1 :]:
                if target not in lengths:
                    continue
                metric_graph.add_edge(source, target, cost=float(lengths[target]))
                paths[(source, target)] = shortest_paths[target]
                paths[(target, source)] = list(reversed(shortest_paths[target]))
        if metric_graph.number_of_edges() == 0:
            continue
        terminal_tree = nx.minimum_spanning_tree(metric_graph, weight="cost")
        for source, target in terminal_tree.edges:
            path = paths[(source, target)]
            for u, v in zip(path[:-1], path[1:]):
                edge_data = cost_graph.get_edge_data(u, v)
                if edge_data is not None:
                    selected_edge_ids.add(int(edge_data["edge_id"]))
    return selected_edge_ids


def decode_connected(
    graph: RoadCandidateGraph,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
    lambda_length: float = 0.001,
) -> DecodedNetwork:
    """Connect high-probability road regions with shortest paths over roads.

    This is a lightweight Steiner-like approximation: high-probability edge
    endpoints become terminals, terminal shortest paths are computed using
    ``-log(p_e) + lambda_length * length``, and a minimum spanning tree over the
    terminal metric closure is expanded back to road edges.
    """

    edges = _edges_with_probabilities(graph, probabilities)
    if edges.empty:
        return DecodedNetwork(edges.copy(), decoder_type="connected", threshold=float(threshold))

    cost_graph = _costed_simple_graph(edges, lambda_length=lambda_length)
    terminals = _terminal_nodes(edges, threshold)
    selected_edge_ids = _steiner_like_edge_ids(cost_graph, terminals)
    if not selected_edge_ids:
        return decode_threshold(graph, probabilities, threshold=threshold)

    selected = edges[edges["edge_id"].isin(selected_edge_ids)].copy()
    return DecodedNetwork(selected, decoder_type="connected", threshold=float(threshold))


def decode_network(
    graph: RoadCandidateGraph,
    probabilities: np.ndarray,
    decoder_config: Mapping | None = None,
) -> DecodedNetwork:
    """Dispatch a configured decoder.

    Fix for P1 of ``docs/research_notes/current_codebase_review_codex.md``:
    ``sewer``, ``water``, and ``steiner`` previously aliased to
    ``decode_connected`` and silently relabelled the output. That made a
    run configured as ``decoder.type: sewer`` produce outputs labelled
    sewer-decoded even though no sewer-specific (uphill penalty,
    pseudo-outlet, tree constraint, etc.) logic exists yet. Until those
    algorithms are implemented (see
    ``docs/anchor_free_engineering_decoder_note_codex.md``), these names
    raise ``NotImplementedError`` to prevent misleading paper claims.
    """

    decoder_config = dict(decoder_config or {})
    decoder_type = str(decoder_config.get("type", "threshold")).lower()
    threshold = float(decoder_config.get("threshold", 0.5))
    if decoder_type == "threshold":
        return decode_threshold(graph, probabilities, threshold=threshold)
    if decoder_type == "connected":
        return decode_connected(
            graph,
            probabilities,
            threshold=threshold,
            lambda_length=float(decoder_config.get("lambda_length", 0.001)),
        )
    if decoder_type in {"sewer", "water", "steiner"}:
        raise NotImplementedError(
            f"decoder.type={decoder_type!r} is reserved for future implementations "
            f"(see docs/anchor_free_engineering_decoder_note_codex.md). The current "
            f"codebase only implements 'threshold' and 'connected'. Use one of "
            f"those, or drop this decoder type until the {decoder_type}-specific "
            f"algorithm (uphill penalty / pseudo-outlet / tree constraint / "
            f"loop budget) is implemented and tested."
        )
    raise ValueError(
        f"Unsupported decoder type: {decoder_type!r}. "
        f"Supported: 'threshold', 'connected'."
    )


# ---------------------------------------------------------------------------
# Phase 2.A — decoders for the heterogeneous (RoadSegment + Intersection) graph.
# Workstream: Claude
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DecodedRoadSegmentNetwork:
    """Decoded network in the heterogeneous formulation.

    Selected ``RoadSegment`` nodes (with their LineStrings) become the
    predicted utility corridors.
    """

    road_segments: gpd.GeoDataFrame
    decoder_type: str
    threshold: float

    @property
    def segment_ids(self) -> np.ndarray:
        if self.road_segments.empty:
            return np.array([], dtype=int)
        return self.road_segments["segment_id"].to_numpy(dtype=int)


def _segments_with_probabilities(graph, probabilities: np.ndarray) -> gpd.GeoDataFrame:
    segments = graph.road_segments.sort_values("segment_id").copy()
    probabilities = np.asarray(probabilities, dtype=float)
    if len(segments) != len(probabilities):
        raise ValueError(
            f"Expected {len(segments)} probabilities, received {len(probabilities)}."
        )
    segments["probability"] = probabilities
    return segments


def decode_threshold_segments(
    graph, probabilities: np.ndarray, *, threshold: float = 0.5
) -> DecodedRoadSegmentNetwork:
    """Select all RoadSegment nodes with probability >= threshold."""
    segments = _segments_with_probabilities(graph, probabilities)
    selected = segments[segments["probability"] >= float(threshold)].copy()
    return DecodedRoadSegmentNetwork(
        road_segments=selected,
        decoder_type="threshold",
        threshold=float(threshold),
    )


def _connected_segment_ids(
    graph,
    segments: gpd.GeoDataFrame,
    *,
    threshold: float,
    lambda_length: float,
) -> set[int]:
    terminals = set(
        segments.loc[
            segments["probability"] >= float(threshold),
            "segment_id",
        ]
        .astype(int)
        .tolist()
    )
    if not terminals and not segments.empty:
        terminals = set(segments.nlargest(1, "probability")["segment_id"].astype(int).tolist())

    crosses = np.asarray(graph.segment_crosses_segment, dtype=np.int64)
    cost_graph = nx.Graph()
    seg_prob = dict(zip(segments["segment_id"].astype(int), segments["probability"]))
    seg_len = dict(zip(segments["segment_id"].astype(int), segments["length_m"].astype(float)))
    for sid in segments["segment_id"].astype(int):
        cost_graph.add_node(int(sid))
    for a, b in zip(crosses[0], crosses[1]):
        a, b = int(a), int(b)
        if a == b:
            continue
        p_avg = 0.5 * (float(seg_prob[a]) + float(seg_prob[b]))
        len_avg = 0.5 * (float(seg_len[a]) + float(seg_len[b]))
        cost = -float(np.log(max(p_avg, 1e-6))) + float(lambda_length) * len_avg
        prior = cost_graph.get_edge_data(a, b)
        if prior is None or cost < prior["cost"]:
            cost_graph.add_edge(a, b, cost=float(cost))

    selected_ids = set(terminals)
    for component_nodes in nx.connected_components(cost_graph):
        component_terminals = sorted(terminals.intersection(component_nodes))
        if len(component_terminals) < 2:
            continue
        subgraph = cost_graph.subgraph(component_nodes)
        metric_graph = nx.Graph()
        paths: dict[tuple[int, int], list[int]] = {}
        for i, source in enumerate(component_terminals):
            lengths, shortest_paths = nx.single_source_dijkstra(
                subgraph,
                source,
                weight="cost",
            )
            for target in component_terminals[i + 1 :]:
                if target not in lengths:
                    continue
                metric_graph.add_edge(source, target, cost=float(lengths[target]))
                paths[(source, target)] = shortest_paths[target]
                paths[(target, source)] = list(reversed(shortest_paths[target]))
        if metric_graph.number_of_edges() == 0:
            continue
        terminal_tree = nx.minimum_spanning_tree(metric_graph, weight="cost")
        for source, target in terminal_tree.edges:
            selected_ids.update(paths[(int(source), int(target))])
    return selected_ids


def decode_connected_segments(
    graph,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
    lambda_length: float = 0.001,
) -> DecodedRoadSegmentNetwork:
    """Connect high-probability RoadSegments over the full segment graph.

    High-probability segments are terminals. Shortest paths are computed over
    the full RoadSegment x RoadSegment adjacency, so lower-probability connector
    segments can be added when needed to join two terminals.
    """
    segments = _segments_with_probabilities(graph, probabilities)
    selected_node_ids = _connected_segment_ids(
        graph,
        segments,
        threshold=float(threshold),
        lambda_length=float(lambda_length),
    )
    selected = segments[segments["segment_id"].isin(selected_node_ids)].copy()
    return DecodedRoadSegmentNetwork(
        road_segments=selected,
        decoder_type="connected",
        threshold=float(threshold),
    )


def decode_segment_network(
    graph,
    probabilities: np.ndarray,
    decoder_config: Mapping | None = None,
) -> DecodedRoadSegmentNetwork:
    """Dispatch a configured decoder for the heterogeneous pipeline."""
    decoder_config = dict(decoder_config or {})
    decoder_type = str(decoder_config.get("type", "threshold")).lower()
    threshold = float(decoder_config.get("threshold", 0.5))
    if decoder_type == "threshold":
        return decode_threshold_segments(graph, probabilities, threshold=threshold)
    if decoder_type == "connected":
        return decode_connected_segments(
            graph,
            probabilities,
            threshold=threshold,
            lambda_length=float(decoder_config.get("lambda_length", 0.001)),
        )
    if decoder_type in {"sewer", "water", "steiner"}:
        raise NotImplementedError(
            f"decoder.type={decoder_type!r} is reserved; not yet implemented "
            "for the heterogeneous pipeline."
        )
    raise ValueError(
        f"Unsupported decoder type: {decoder_type!r}. Supported: 'threshold', 'connected'."
    )
