"""Label road-edge candidates from ground-truth utility lines.

Utility truth geometry is allowed here for supervised label creation and
evaluation only. It must not be passed into feature generation or inference.
"""

# Workstream: Codex + Claude merge

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union

from pipe_network_completion.anchor_free.road_graph import RoadCandidateGraph


FIXED_ROAD_OFFSET_LANES = (
    "center",
    "left_15m",
    "right_15m",
    "left_30m",
    "right_30m",
)
ROAD_OFFSET_LANE_TO_CLASS = {
    lane_name: lane_index
    for lane_index, lane_name in enumerate(FIXED_ROAD_OFFSET_LANES)
}


@dataclass(frozen=True)
class RoadEdgeLabels:
    """Binary labels aligned with road candidate edges."""

    labels: gpd.GeoDataFrame

    @property
    def y(self) -> np.ndarray:
        return self.labels["y"].to_numpy(dtype=int)

    @property
    def edge_ids(self) -> np.ndarray:
        return self.labels["edge_id"].to_numpy(dtype=int)


def _compatible_truth_gdf(
    utility_truth_gdf: gpd.GeoDataFrame,
    graph: RoadCandidateGraph,
) -> gpd.GeoDataFrame:
    if graph.crs and utility_truth_gdf.crs and str(utility_truth_gdf.crs) != str(graph.crs):
        return utility_truth_gdf.to_crs(graph.crs)
    return utility_truth_gdf


def _line_like_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    mask = gdf.geometry.apply(
        lambda geom: geom is not None
        and not geom.is_empty
        and isinstance(geom, (LineString, MultiLineString))
    )
    return gdf.loc[mask].copy()


def label_road_edges_from_utility_lines(
    graph: RoadCandidateGraph,
    utility_truth_gdf: gpd.GeoDataFrame,
    *,
    label_buffer_m: float = 10.0,
    label_overlap_threshold: float = 0.25,
) -> RoadEdgeLabels:
    """Create labels by spatially matching utility truth lines to road edges.

    Each candidate road edge is buffered by ``label_buffer_m`` and intersected
    with nearby ground-truth utility lines found through a spatial index. The
    overlap ratio is the intersected truth-line length divided by the road-edge
    length, clipped to ``[0, 1]``. The positive label is assigned when that
    ratio reaches ``label_overlap_threshold``.
    """

    edges = graph.edges.copy()
    if edges.empty:
        empty = gpd.GeoDataFrame(
            {"edge_id": [], "y": [], "overlap_length": [], "overlap_ratio": []},
            geometry=[],
            crs=graph.crs,
        )
        return RoadEdgeLabels(empty)

    truth = _line_like_gdf(_compatible_truth_gdf(utility_truth_gdf, graph))
    truth_geometries = list(truth.geometry)
    truth_sindex = truth.sindex if truth_geometries else None

    records = []
    for edge in edges.itertuples(index=False):
        length = max(float(edge.length_m), 1e-12)
        if truth_sindex is None:
            overlap_length = 0.0
        else:
            edge_buffer = edge.geometry.buffer(float(label_buffer_m))
            try:
                candidate_indexes = truth_sindex.query(edge_buffer, predicate="intersects")
            except TypeError:
                candidate_indexes = [
                    idx
                    for idx in truth_sindex.intersection(edge_buffer.bounds)
                    if edge_buffer.intersects(truth_geometries[idx])
                ]
            overlap_length = 0.0
            for candidate_index in candidate_indexes:
                overlap = edge_buffer.intersection(truth_geometries[int(candidate_index)])
                if not overlap.is_empty:
                    overlap_length += float(overlap.length)
        overlap_ratio = min(overlap_length / length, 1.0)
        records.append(
            {
                "edge_id": int(edge.edge_id),
                "y": int(overlap_ratio >= float(label_overlap_threshold)),
                "overlap_length": overlap_length,
                "overlap_ratio": overlap_ratio,
                "geometry": edge.geometry,
            }
        )

    label_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=edges.crs)
    return RoadEdgeLabels(label_gdf)


def labels_to_frame(labels: RoadEdgeLabels) -> pd.DataFrame:
    """Return a non-geometry label table for model training."""

    return pd.DataFrame(labels.labels.drop(columns="geometry"))


# ---------------------------------------------------------------------------
# Phase 2.A — labels for the heterogeneous road graph.
# Workstream: Claude
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RoadSegmentLabels:
    """Per-RoadSegment-node labels for the heterogeneous pipeline."""

    labels: gpd.GeoDataFrame

    @property
    def y(self) -> np.ndarray:
        return self.labels["y"].to_numpy(dtype=int)

    @property
    def segment_ids(self) -> np.ndarray:
        return self.labels["segment_id"].to_numpy(dtype=int)


@dataclass(frozen=True)
class RoadOffsetLaneLabels:
    """One label row per source road for five fixed offset-lane candidates.

    ``y`` is the source-road presence label. ``lane_class`` is one of
    ``0..4`` for positive rows and ``-1`` for negative rows. The fixed class
    order is ``FIXED_ROAD_OFFSET_LANES``.
    """

    labels: gpd.GeoDataFrame
    lane_names: tuple[str, ...] = FIXED_ROAD_OFFSET_LANES

    @property
    def y(self) -> np.ndarray:
        return self.labels["y"].to_numpy(dtype=int)

    @property
    def lane_class(self) -> np.ndarray:
        return self.labels["lane_class"].to_numpy(dtype=int)


def label_road_segments_from_utility_lines(
    graph,
    utility_truth_gdf: gpd.GeoDataFrame,
    *,
    label_buffer_m: float = 10.0,
    label_overlap_threshold: float = 0.25,
) -> RoadSegmentLabels:
    """Heterogeneous-graph version of ``label_road_edges_from_utility_lines``.

    Same buffer/overlap-ratio math; per-row key is ``segment_id`` instead
    of ``edge_id``.
    """
    segments = graph.road_segments.copy()
    if segments.empty:
        empty = gpd.GeoDataFrame(
            {"segment_id": [], "y": [], "overlap_length": [], "overlap_ratio": []},
            geometry=[],
            crs=getattr(graph, "crs", None),
        )
        return RoadSegmentLabels(empty)

    truth = utility_truth_gdf
    if graph.crs and truth.crs and str(truth.crs) != str(graph.crs):
        truth = truth.to_crs(graph.crs)

    truth_line_gdf = _line_like_gdf(truth)
    truth_geometries = list(truth_line_gdf.geometry)
    truth_sindex = truth_line_gdf.sindex if truth_geometries else None

    records = []
    for seg in segments.itertuples(index=False):
        length = max(float(seg.length_m), 1e-12)
        if truth_sindex is None:
            overlap_length = 0.0
        else:
            segment_buffer = seg.geometry.buffer(float(label_buffer_m))
            try:
                candidate_indexes = truth_sindex.query(segment_buffer, predicate="intersects")
            except TypeError:
                candidate_indexes = [
                    idx
                    for idx in truth_sindex.intersection(segment_buffer.bounds)
                    if segment_buffer.intersects(truth_geometries[idx])
                ]
            overlap_length = 0.0
            for candidate_index in candidate_indexes:
                overlap = segment_buffer.intersection(truth_geometries[int(candidate_index)])
                if not overlap.is_empty:
                    overlap_length += float(overlap.length)
        overlap_ratio = min(overlap_length / length, 1.0)
        records.append(
            {
                "segment_id": int(seg.segment_id),
                "y": int(overlap_ratio >= float(label_overlap_threshold)),
                "overlap_length": overlap_length,
                "overlap_ratio": overlap_ratio,
                "geometry": seg.geometry,
            }
        )
    return RoadSegmentLabels(
        gpd.GeoDataFrame(records, geometry="geometry", crs=segments.crs)
    )


def _format_offset_distance(value) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not np.isfinite(numeric):
        return "unknown"
    rounded = round(numeric)
    if abs(numeric - rounded) < 1e-6:
        return str(int(rounded))
    return f"{numeric:g}"


def road_offset_lane_name(row) -> str:
    """Return the fixed lane token for a road-offset candidate row."""

    source = str(row.get("candidate_source", "road_backbone"))
    if source == "road_offset":
        side = str(row.get("road_offset_side", "unknown"))
        distance = _format_offset_distance(row.get("road_offset_distance_m", 0.0))
        return f"{side}_{distance}m"
    if source == "road_backbone":
        return "center"
    return source


def _sample_truth_lines(
    utility_truth_gdf: gpd.GeoDataFrame,
    *,
    sample_spacing_m: float,
) -> gpd.GeoDataFrame:
    records: list[dict] = []
    spacing = max(float(sample_spacing_m), 1.0)
    for truth_id, geom in enumerate(utility_truth_gdf.geometry):
        for line in _iter_line_parts(geom):
            length = float(line.length)
            if length <= 0.0:
                continue
            n_samples = max(1, int(round(length / spacing)))
            weight = length / n_samples
            for i in range(n_samples):
                distance = min((i + 0.5) * weight, length)
                records.append(
                    {
                        "truth_id": int(truth_id),
                        "weight_m": float(weight),
                        "geometry": line.interpolate(distance),
                    }
                )
    if not records:
        return gpd.GeoDataFrame(
            {"truth_id": [], "weight_m": []},
            geometry=[],
            crs=utility_truth_gdf.crs,
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=utility_truth_gdf.crs)


def _iter_line_parts(geom):
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, LineString):
        yield geom
    elif isinstance(geom, MultiLineString) or hasattr(geom, "geoms"):
        for part in geom.geoms:
            yield from _iter_line_parts(part)


def _lane_candidate_frame(graph) -> gpd.GeoDataFrame:
    segments = graph.road_segments.copy()
    required = {"source_index", "candidate_source"}
    missing = required.difference(segments.columns)
    if missing:
        raise ValueError(
            "Road-offset lane labels require a road_offsets candidate graph with "
            f"columns: {sorted(required)}; missing {sorted(missing)}"
        )
    segments["offset_lane"] = segments.apply(road_offset_lane_name, axis=1)
    unknown = sorted(set(segments["offset_lane"]) - set(FIXED_ROAD_OFFSET_LANES))
    if unknown:
        raise ValueError(
            "Road-offset lane labels expect exactly the fixed five-lane support "
            f"{FIXED_ROAD_OFFSET_LANES}; found unsupported lanes {unknown}"
        )
    segments["lane_class"] = segments["offset_lane"].map(ROAD_OFFSET_LANE_TO_CLASS).astype(int)
    return segments


def label_road_offset_lanes_from_utility_lines(
    graph,
    utility_truth_gdf: gpd.GeoDataFrame,
    *,
    sample_spacing_m: float = 10.0,
    max_assignment_distance_m: float | None = 50.0,
    min_assigned_truth_length_m: float = 10.0,
    min_assigned_fraction: float = 0.0,
    ambiguous_weight_margin: float = 0.1,
) -> RoadOffsetLaneLabels:
    """Assign one nearest-lane label to each source road in a five-lane graph.

    This is the labeler for the road-as-node formulation. It samples utility
    truth lines into length-weighted points, assigns each point to its nearest
    candidate lane, and aggregates those assignments by ``source_index``.
    Utility truth is used only here for supervised label preparation.
    """

    candidates = _lane_candidate_frame(graph)
    group_columns = ["source_index"]
    if "aoi_id" in candidates.columns:
        group_columns.insert(0, "aoi_id")

    truth = utility_truth_gdf
    if candidates.crs and truth.crs and str(truth.crs) != str(candidates.crs):
        truth = truth.to_crs(candidates.crs)
    truth = _line_like_gdf(truth)
    samples = _sample_truth_lines(truth, sample_spacing_m=sample_spacing_m)

    matched = pd.DataFrame()
    if not samples.empty and not candidates.empty:
        max_distance = (
            None
            if max_assignment_distance_m is None or float(max_assignment_distance_m) <= 0.0
            else float(max_assignment_distance_m)
        )
        joined = gpd.sjoin_nearest(
            samples,
            candidates[
                [
                    "segment_id",
                    *group_columns,
                    "offset_lane",
                    "lane_class",
                    "geometry",
                ]
            ],
            how="left",
            max_distance=max_distance,
            distance_col="nearest_lane_distance_m",
        )
        joined = joined[joined["segment_id"].notna()].copy()
        if not joined.empty:
            joined["sample_index"] = joined.index
            joined = joined.sort_values(
                ["sample_index", "nearest_lane_distance_m", "segment_id"],
                ascending=[True, True, True],
            )
            matched = joined.groupby("sample_index", dropna=False).head(1)

    if not matched.empty:
        weight_by_lane = (
            matched.groupby([*group_columns, "offset_lane"], dropna=False)["weight_m"]
            .sum()
            .unstack(fill_value=0.0)
        )
        count_by_group = matched.groupby(group_columns, dropna=False).size()
        mean_distance_by_group = matched.groupby(group_columns, dropna=False)[
            "nearest_lane_distance_m"
        ].mean()
    else:
        weight_by_lane = pd.DataFrame(columns=FIXED_ROAD_OFFSET_LANES, dtype=float)
        count_by_group = pd.Series(dtype=float)
        mean_distance_by_group = pd.Series(dtype=float)

    for lane_name in FIXED_ROAD_OFFSET_LANES:
        if lane_name not in weight_by_lane.columns:
            weight_by_lane[lane_name] = 0.0
    weight_by_lane = weight_by_lane[list(FIXED_ROAD_OFFSET_LANES)]

    records: list[dict] = []
    groupby_key = group_columns[0] if len(group_columns) == 1 else group_columns
    for key, group in candidates.groupby(groupby_key, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_values = dict(zip(group_columns, key_tuple))
        center = group[group["offset_lane"] == "center"]
        source_length = (
            float(center.geometry.length.sum())
            if not center.empty
            else float(group.geometry.length.max())
        )
        geometry_source = center if not center.empty else group
        geometry = unary_union(list(geometry_source.geometry))

        try:
            lane_weights = weight_by_lane.loc[key]
        except KeyError:
            lane_weights = pd.Series(0.0, index=FIXED_ROAD_OFFSET_LANES)
        lane_weights = lane_weights.reindex(FIXED_ROAD_OFFSET_LANES).fillna(0.0)
        ordered = lane_weights.sort_values(ascending=False, kind="mergesort")
        best_lane = str(ordered.index[0])
        best_weight = float(ordered.iloc[0])
        second_lane = str(ordered.index[1]) if len(ordered) > 1 else ""
        second_weight = float(ordered.iloc[1]) if len(ordered) > 1 else 0.0
        total_weight = float(lane_weights.sum())
        best_fraction = best_weight / max(source_length, 1e-12)
        y = int(
            best_weight >= float(min_assigned_truth_length_m)
            and best_fraction >= float(min_assigned_fraction)
        )
        ambiguous = bool(
            y
            and best_weight > 0.0
            and second_weight >= best_weight * (1.0 - float(ambiguous_weight_margin))
        )
        try:
            n_samples = int(count_by_group.loc[key])
        except KeyError:
            n_samples = 0
        try:
            mean_distance = float(mean_distance_by_group.loc[key])
        except KeyError:
            mean_distance = np.nan

        record = {
            **key_values,
            "y": y,
            "lane_class": int(ROAD_OFFSET_LANE_TO_CLASS[best_lane]) if y else -1,
            "lane_name": best_lane if y else "none",
            "is_ambiguous": ambiguous,
            "assigned_truth_length_m": best_weight if y else 0.0,
            "total_assigned_truth_length_m": total_weight,
            "second_lane_name": second_lane,
            "second_lane_truth_length_m": second_weight,
            "lane_confidence": best_weight / total_weight if total_weight > 0.0 else 0.0,
            "lane_margin_m": best_weight - second_weight,
            "source_length_m": source_length,
            "winning_lane_truth_fraction": best_fraction,
            "matched_truth_sample_count": n_samples,
            "mean_assignment_distance_m": mean_distance,
            "geometry": geometry,
        }
        for lane_name in FIXED_ROAD_OFFSET_LANES:
            record[f"truth_length_{lane_name}_m"] = float(lane_weights[lane_name])
        records.append(record)

    label_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=candidates.crs)
    return RoadOffsetLaneLabels(label_gdf)
