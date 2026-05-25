"""Candidate-graph representability metrics.

Workstream: Codex
"""

from __future__ import annotations

from typing import Iterable

import geopandas as gpd

from pipe_network_completion.anchor_free.hetero_road_graph import HeteroRoadGraph


def iter_line_parts(geom):
    """Yield LineString-like parts from single or multi geometries."""

    if geom is None or geom.is_empty:
        return
    if geom.geom_type in {"LineString", "LinearRing"}:
        yield geom
    elif hasattr(geom, "geoms"):
        for part in geom.geoms:
            yield from iter_line_parts(part)


def truth_sample_points(
    utility_truth: gpd.GeoDataFrame,
    *,
    sample_spacing_m: float,
) -> gpd.GeoDataFrame:
    """Sample truth lines into weighted points for approximate length recall."""

    records: list[dict] = []
    spacing = max(float(sample_spacing_m), 1.0)
    for geom in utility_truth.geometry:
        for line in iter_line_parts(geom):
            length = float(line.length)
            if length <= 0.0:
                continue
            n_samples = max(1, int(round(length / spacing)))
            weight = length / n_samples
            for i in range(n_samples):
                distance = min((i + 0.5) * weight, length)
                records.append(
                    {
                        "weight_m": float(weight),
                        "geometry": line.interpolate(distance),
                    }
                )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=utility_truth.crs)


def metric_label_suffix(label_buffer_m: float) -> str:
    return f"{float(label_buffer_m):g}".replace(".", "p") + "m"


def candidate_source_summary(graph: HeteroRoadGraph) -> dict[str, dict[str, float]]:
    """Summarize candidate counts and length by source family."""

    if graph.road_segments.empty or "candidate_source" not in graph.road_segments.columns:
        return {}
    summary: dict[str, dict[str, float]] = {}
    for source, group in graph.road_segments.groupby("candidate_source"):
        summary[str(source)] = {
            "count": float(len(group)),
            "length_m": float(group["length_m"].sum())
            if "length_m" in group.columns
            else float(group.geometry.length.sum()),
        }
    return summary


def candidate_representability_metrics(
    graph: HeteroRoadGraph,
    utility_truth: gpd.GeoDataFrame,
    *,
    buffers_m: Iterable[float],
    sample_spacing_m: float = 50.0,
) -> dict[str, float]:
    """Estimate truth-length recall for a candidate graph at multiple buffers.

    This is not model F1. It is the candidate support's upper-bound recall:
    how much observed truth length lies within each tolerance of any candidate
    edge.
    """

    buffers = sorted(float(buffer) for buffer in buffers_m if float(buffer) >= 0.0)
    total_candidate_length = (
        float(graph.road_segments["length_m"].sum())
        if "length_m" in graph.road_segments.columns
        else float(graph.road_segments.geometry.length.sum())
    )
    out: dict[str, float] = {
        "candidate_count": float(len(graph.road_segments)),
        "candidate_total_length_m": total_candidate_length,
    }
    if not buffers or graph.road_segments.empty or utility_truth.empty:
        return out

    truth = utility_truth
    if (
        graph.road_segments.crs is not None
        and truth.crs is not None
        and str(truth.crs) != str(graph.road_segments.crs)
    ):
        truth = truth.to_crs(graph.road_segments.crs)

    samples = truth_sample_points(truth, sample_spacing_m=float(sample_spacing_m))
    total_truth_length = float(samples["weight_m"].sum()) if not samples.empty else 0.0
    out["truth_total_length_sampled_m"] = total_truth_length
    out["truth_sample_count"] = float(len(samples))
    out["truth_sample_spacing_m"] = float(sample_spacing_m)
    if samples.empty or total_truth_length <= 0.0:
        return out

    candidates = graph.road_segments[["segment_id", "geometry"]].copy()
    max_buffer = max(buffers)
    joined = gpd.sjoin_nearest(
        samples,
        candidates,
        how="left",
        max_distance=max_buffer,
        distance_col="nearest_candidate_distance_m",
    )
    # GeoPandas can return multiple rows for one truth sample when several
    # candidate geometries tie for nearest distance. Collapse to one distance
    # per sampled truth point before summing weights, otherwise recall can
    # exceed 1.0 for overlapping candidate supports.
    nearest_distance = joined.groupby(level=0)["nearest_candidate_distance_m"].min()
    for buffer_m in buffers:
        suffix = metric_label_suffix(buffer_m)
        covered_sample_index = nearest_distance.index[
            nearest_distance.notna() & (nearest_distance <= buffer_m)
        ]
        covered_length = float(samples.loc[covered_sample_index, "weight_m"].sum())
        out[f"recall_{suffix}"] = covered_length / total_truth_length
        out[f"covered_truth_length_{suffix}_m"] = covered_length
    return out
