"""Skeleton-building-road heterograph for anchor-free sewer-main prediction.

This graph is the anchor-free analogue of the ISARC manhole-road graph:
surface skeleton segments are prediction nodes, while road and building nodes
provide contextual message passing. Utility truth is not used here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point

from pipe_network_completion.anchor_free.features import assert_no_anchor_features


DEFAULT_CONTEXT_COLUMNS = (
    "candidate_source",
    "ROUTE_TYPE",
    "OVL_CAT",
    "OVL2_CAT",
    "CLASS",
    "highway",
)


@dataclass(frozen=True)
class SkeletonContextGraph:
    """Heterogeneous surface-support graph.

    Node types:
    - SkeletonSegment: candidate sewer-main support and prediction target.
    - Building: demand/context nodes derived from non-utility building data.
    - RoadSegment: road-context nodes.

    Edge arrays are shaped ``(2, n_edges)`` and use local integer ids for the
    source and target node tables listed in the edge name.
    """

    skeleton_segments: gpd.GeoDataFrame
    buildings: gpd.GeoDataFrame
    road_segments: gpd.GeoDataFrame
    skeleton_adjacent_skeleton: np.ndarray
    building_near_building: np.ndarray
    building_near_skeleton: np.ndarray
    building_near_road: np.ndarray
    skeleton_near_road: np.ndarray
    road_adjacent_road: np.ndarray
    crs: str | None = None
    metadata: dict = field(default_factory=dict)


def _iter_lines(geom):
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, LineString):
        if len(geom.coords) >= 2:
            yield geom
    elif isinstance(geom, MultiLineString) or hasattr(geom, "geoms"):
        for part in geom.geoms:
            yield from _iter_lines(part)


def _bearing_rad(line: LineString) -> float:
    coords = list(line.coords)
    x0, y0 = coords[0][:2]
    x1, y1 = coords[-1][:2]
    return math.atan2(float(y1 - y0), float(x1 - x0))


def _as_target_crs(gdf: gpd.GeoDataFrame | None, target_crs) -> gpd.GeoDataFrame:
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)
    frame = gdf.copy()
    if target_crs is not None and frame.crs is not None and str(frame.crs) != str(target_crs):
        frame = frame.to_crs(target_crs)
    return frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()


def _explode_line_nodes(
    gdf: gpd.GeoDataFrame,
    *,
    id_column: str,
    source_column: str | None = None,
    source_value: str | None = None,
    keep_columns: Iterable[str] = DEFAULT_CONTEXT_COLUMNS,
) -> gpd.GeoDataFrame:
    records: list[dict] = []
    keep = [column for column in keep_columns if column in gdf.columns]
    for source_index, row in gdf.iterrows():
        for line in _iter_lines(row.geometry):
            record = {
                id_column: len(records),
                "source_index": int(source_index) if isinstance(source_index, (int, np.integer)) else str(source_index),
                "length_m": float(line.length),
                "bearing_rad": _bearing_rad(line),
                "geometry": line,
            }
            if source_column is not None:
                record[source_column] = source_value
            for column in keep:
                record[column] = row[column]
            records.append(record)
    if not records:
        columns = [id_column, "source_index", "length_m", "bearing_rad", *keep]
        if source_column is not None:
            columns.append(source_column)
        return gpd.GeoDataFrame({column: [] for column in columns}, geometry=[], crs=gdf.crs)
    return gpd.GeoDataFrame(records, geometry="geometry", crs=gdf.crs)


def _prepare_skeleton_segments(
    roads: gpd.GeoDataFrame,
    drainage_lines: gpd.GeoDataFrame,
    *,
    include_drainage: bool,
) -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = []
    if not roads.empty:
        road_skeleton = _explode_line_nodes(
            roads,
            id_column="skeleton_id",
            source_column="candidate_source",
            source_value="road",
        )
        frames.append(road_skeleton)
    if include_drainage and not drainage_lines.empty:
        drainage_skeleton = _explode_line_nodes(
            drainage_lines,
            id_column="skeleton_id",
            source_column="candidate_source",
            source_value="drainage",
        )
        if "ROUTE_TYPE" not in drainage_skeleton.columns:
            drainage_skeleton["ROUTE_TYPE"] = "drainage"
        frames.append(drainage_skeleton)
    if not frames:
        return gpd.GeoDataFrame(
            {"skeleton_id": [], "source_index": [], "length_m": [], "bearing_rad": []},
            geometry=[],
            crs=roads.crs or drainage_lines.crs,
        )
    out = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs)
    out = out[out.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    out = out[out["length_m"].astype(float) > 1.0].reset_index(drop=True)
    out["skeleton_id"] = np.arange(len(out), dtype=int)
    return out


def _prepare_building_nodes(
    building_points: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    source = building_points if not building_points.empty else buildings
    if source.empty:
        return gpd.GeoDataFrame(
            {"building_id": [], "source_index": []},
            geometry=[],
            crs=building_points.crs or buildings.crs,
        )

    records: list[dict] = []
    passthrough = [
        column
        for column in source.columns
        if (
            column.startswith("bt_group_")
            or column.startswith("bt_raw_")
            or column
            in {
                "building_type_group",
                "building_type_raw",
                "footprint_area_m2",
                "footprint_perimeter_m",
                "amenity",
            }
        )
    ]
    for source_index, row in source.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        point = geom if geom.geom_type == "Point" else geom.representative_point()
        record = {
            "building_id": len(records),
            "source_index": int(source_index) if isinstance(source_index, (int, np.integer)) else str(source_index),
            "geometry": point,
        }
        if "footprint_area_m2" not in passthrough and geom.geom_type != "Point":
            record["footprint_area_m2"] = float(geom.area)
        if "footprint_perimeter_m" not in passthrough and geom.geom_type != "Point":
            record["footprint_perimeter_m"] = float(geom.length)
        for column in passthrough:
            record[column] = row[column]
        records.append(record)

    return gpd.GeoDataFrame(records, geometry="geometry", crs=source.crs)


def _pairs_to_array(pairs: Iterable[tuple[int, int]], *, symmetric: bool = False) -> np.ndarray:
    pair_set: set[tuple[int, int]] = set()
    for a, b in pairs:
        a_i = int(a)
        b_i = int(b)
        if a_i == b_i and symmetric:
            continue
        pair_set.add((a_i, b_i))
        if symmetric:
            pair_set.add((b_i, a_i))
    if not pair_set:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray(sorted(pair_set), dtype=np.int64).T


def _same_type_near_pairs(
    gdf: gpd.GeoDataFrame,
    *,
    id_column: str,
    radius_m: float,
    max_neighbors: int | None,
) -> np.ndarray:
    if gdf.empty or len(gdf) < 2:
        return np.zeros((2, 0), dtype=np.int64)
    geoms = list(gdf.geometry)
    ids = gdf[id_column].astype(int).to_numpy()
    spatial_index = gdf.sindex
    pairs: list[tuple[int, int]] = []
    for pos, geom in enumerate(geoms):
        search = geom.buffer(float(radius_m))
        try:
            idxs = spatial_index.query(search, predicate="intersects")
        except TypeError:
            idxs = list(spatial_index.intersection(search.bounds))
        candidates = []
        for idx_raw in idxs:
            idx = int(idx_raw)
            if idx == pos:
                continue
            distance = float(geom.distance(geoms[idx]))
            if distance <= float(radius_m):
                candidates.append((distance, int(ids[idx])))
        candidates.sort(key=lambda item: (item[0], item[1]))
        if max_neighbors is not None:
            candidates = candidates[: max(int(max_neighbors), 0)]
        pairs.extend((int(ids[pos]), neighbor_id) for _, neighbor_id in candidates)
    return _pairs_to_array(pairs, symmetric=True)


def _cross_type_near_pairs(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    *,
    left_id_column: str,
    right_id_column: str,
    radius_m: float,
    max_neighbors: int | None,
) -> np.ndarray:
    if left.empty or right.empty:
        return np.zeros((2, 0), dtype=np.int64)
    right_geoms = list(right.geometry)
    right_ids = right[right_id_column].astype(int).to_numpy()
    spatial_index = right.sindex
    pairs: list[tuple[int, int]] = []
    for row in left.itertuples(index=False):
        geom = row.geometry
        search = geom.buffer(float(radius_m))
        try:
            idxs = spatial_index.query(search, predicate="intersects")
        except TypeError:
            idxs = list(spatial_index.intersection(search.bounds))
        candidates = []
        for idx_raw in idxs:
            idx = int(idx_raw)
            distance = float(geom.distance(right_geoms[idx]))
            if distance <= float(radius_m):
                candidates.append((distance, int(right_ids[idx])))
        candidates.sort(key=lambda item: (item[0], item[1]))
        if max_neighbors is not None:
            candidates = candidates[: max(int(max_neighbors), 0)]
        left_id = int(getattr(row, left_id_column))
        pairs.extend((left_id, right_id) for _, right_id in candidates)
    return _pairs_to_array(pairs, symmetric=False)


def _bearing_bin(angle_rad: float) -> int:
    degrees = math.degrees(float(angle_rad) % math.pi)
    if degrees < 15.0 or degrees >= 165.0:
        return 0
    return int((degrees - 15.0) / 30.0) + 1


def _node_degrees(edge_index: np.ndarray, n_nodes: int) -> np.ndarray:
    degree = np.zeros(int(n_nodes), dtype=float)
    if edge_index.size:
        values, counts = np.unique(edge_index[0], return_counts=True)
        degree[values.astype(int)] = counts.astype(float)
    return degree


def _incoming_counts(edge_index: np.ndarray, n_targets: int) -> np.ndarray:
    counts = np.zeros(int(n_targets), dtype=float)
    if edge_index.size:
        values, group_counts = np.unique(edge_index[1], return_counts=True)
        counts[values.astype(int)] = group_counts.astype(float)
    return counts


def _outgoing_counts(edge_index: np.ndarray, n_sources: int) -> np.ndarray:
    counts = np.zeros(int(n_sources), dtype=float)
    if edge_index.size:
        values, group_counts = np.unique(edge_index[0], return_counts=True)
        counts[values.astype(int)] = group_counts.astype(float)
    return counts


def _line_node_features(
    frame: gpd.GeoDataFrame,
    *,
    id_column: str,
    adjacent_edges: np.ndarray,
    linked_building_edges: np.ndarray | None = None,
    linked_other_edges: np.ndarray | None = None,
    include_coords: bool = False,
) -> pd.DataFrame:
    ids = frame[id_column].astype(int).to_numpy()
    features = pd.DataFrame(index=ids)
    features.index.name = id_column
    length = frame["length_m"].astype(float).to_numpy() if "length_m" in frame else np.zeros(len(frame))
    bearing = frame["bearing_rad"].astype(float).to_numpy() if "bearing_rad" in frame else np.zeros(len(frame))
    features["length_m"] = length
    features["log1p_length_m"] = np.log1p(np.maximum(length, 0.0))
    features["bearing_sin"] = np.sin(bearing)
    features["bearing_cos"] = np.cos(bearing)
    for bin_id in range(6):
        features[f"bearing_bin_{bin_id}"] = [1.0 if _bearing_bin(value) == bin_id else 0.0 for value in bearing]
    degree = _node_degrees(adjacent_edges, len(frame))
    features["same_type_degree"] = degree
    features["same_type_dead_end"] = (degree <= 1.0).astype(float)
    if linked_building_edges is not None:
        features["linked_building_count"] = _incoming_counts(linked_building_edges, len(frame))
    if linked_other_edges is not None:
        features["linked_context_count"] = _outgoing_counts(linked_other_edges, len(frame))
    for column in DEFAULT_CONTEXT_COLUMNS:
        if column in frame.columns:
            encoded = pd.get_dummies(frame[column].fillna("missing").astype(str), prefix=column, dtype=float)
            encoded.index = features.index
            features = pd.concat([features, encoded], axis=1)
    if include_coords and not frame.empty:
        centroids = frame.geometry.centroid
        x = centroids.x.to_numpy(dtype=float)
        y = centroids.y.to_numpy(dtype=float)
        features["x_norm"] = (x - x.mean()) / (x.std() if x.std() > 1e-12 else 1.0)
        features["y_norm"] = (y - y.mean()) / (y.std() if y.std() > 1e-12 else 1.0)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    assert_no_anchor_features(features.columns)
    return features


def build_skeleton_context_features(
    graph: SkeletonContextGraph,
    *,
    include_coords: bool = False,
) -> dict[str, pd.DataFrame]:
    """Build per-node-type feature tables for the heterograph."""

    skeleton_features = _line_node_features(
        graph.skeleton_segments,
        id_column="skeleton_id",
        adjacent_edges=graph.skeleton_adjacent_skeleton,
        linked_building_edges=graph.building_near_skeleton,
        linked_other_edges=graph.skeleton_near_road,
        include_coords=include_coords,
    )
    road_features = _line_node_features(
        graph.road_segments,
        id_column="road_id",
        adjacent_edges=graph.road_adjacent_road,
        linked_building_edges=graph.building_near_road,
        include_coords=include_coords,
    )

    building_ids = graph.buildings["building_id"].astype(int).to_numpy() if not graph.buildings.empty else np.zeros(0, dtype=int)
    building_features = pd.DataFrame(index=building_ids)
    building_features.index.name = "building_id"
    if not graph.buildings.empty:
        buildings = graph.buildings
        area_source = (
            buildings["footprint_area_m2"]
            if "footprint_area_m2" in buildings.columns
            else pd.Series(np.zeros(len(buildings)), index=buildings.index)
        )
        perimeter_source = (
            buildings["footprint_perimeter_m"]
            if "footprint_perimeter_m" in buildings.columns
            else pd.Series(np.zeros(len(buildings)), index=buildings.index)
        )
        area = pd.to_numeric(area_source, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        perimeter = pd.to_numeric(perimeter_source, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        building_features["log1p_footprint_area_m2"] = np.log1p(np.maximum(area, 0.0))
        building_features["log1p_footprint_perimeter_m"] = np.log1p(np.maximum(perimeter, 0.0))
        building_features["building_neighbor_count"] = _node_degrees(graph.building_near_building, len(buildings))
        building_features["linked_skeleton_count"] = _outgoing_counts(graph.building_near_skeleton, len(buildings))
        building_features["linked_road_count"] = _outgoing_counts(graph.building_near_road, len(buildings))
        for column in buildings.columns:
            if column.startswith("bt_group_") or column.startswith("bt_raw_"):
                building_features[column] = pd.to_numeric(buildings[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        for column in ["building_type_group", "building_type_raw", "amenity"]:
            if column in buildings.columns:
                encoded = pd.get_dummies(buildings[column].fillna("missing").astype(str), prefix=column, dtype=float)
                encoded.index = building_features.index
                building_features = pd.concat([building_features, encoded], axis=1)
        if include_coords:
            x = buildings.geometry.x.to_numpy(dtype=float)
            y = buildings.geometry.y.to_numpy(dtype=float)
            building_features["x_norm"] = (x - x.mean()) / (x.std() if x.std() > 1e-12 else 1.0)
            building_features["y_norm"] = (y - y.mean()) / (y.std() if y.std() > 1e-12 else 1.0)
    building_features = building_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    assert_no_anchor_features(building_features.columns)
    return {
        "SkeletonSegment": skeleton_features,
        "Building": building_features,
        "RoadSegment": road_features,
    }


def build_skeleton_context_graph(
    *,
    roads: gpd.GeoDataFrame,
    building_points: gpd.GeoDataFrame | None = None,
    buildings: gpd.GeoDataFrame | None = None,
    drainage_lines: gpd.GeoDataFrame | None = None,
    target_crs: str | None = None,
    include_drainage: bool = True,
    skeleton_snap_tolerance_m: float = 1.0,
    road_snap_tolerance_m: float = 1.0,
    building_building_radius_m: float = 80.0,
    building_skeleton_radius_m: float = 80.0,
    building_road_radius_m: float = 80.0,
    skeleton_road_radius_m: float = 30.0,
    building_knn: int = 4,
    context_knn: int = 3,
) -> SkeletonContextGraph:
    """Build a skeleton-building-road heterograph from allowed surface data."""

    roads = _as_target_crs(roads, target_crs)
    building_points = _as_target_crs(building_points, roads.crs if target_crs is None else target_crs)
    buildings = _as_target_crs(buildings, roads.crs if target_crs is None else target_crs)
    drainage_lines = _as_target_crs(drainage_lines, roads.crs if target_crs is None else target_crs)

    skeleton = _prepare_skeleton_segments(roads, drainage_lines, include_drainage=include_drainage)
    road_segments = _explode_line_nodes(roads, id_column="road_id")
    buildings_nodes = _prepare_building_nodes(building_points, buildings)

    skeleton_adjacent = _same_type_near_pairs(
        skeleton,
        id_column="skeleton_id",
        radius_m=float(skeleton_snap_tolerance_m),
        max_neighbors=None,
    )
    road_adjacent = _same_type_near_pairs(
        road_segments,
        id_column="road_id",
        radius_m=float(road_snap_tolerance_m),
        max_neighbors=None,
    )
    building_near_building = _same_type_near_pairs(
        buildings_nodes,
        id_column="building_id",
        radius_m=float(building_building_radius_m),
        max_neighbors=int(building_knn),
    )
    building_near_skeleton = _cross_type_near_pairs(
        buildings_nodes,
        skeleton,
        left_id_column="building_id",
        right_id_column="skeleton_id",
        radius_m=float(building_skeleton_radius_m),
        max_neighbors=int(context_knn),
    )
    building_near_road = _cross_type_near_pairs(
        buildings_nodes,
        road_segments,
        left_id_column="building_id",
        right_id_column="road_id",
        radius_m=float(building_road_radius_m),
        max_neighbors=int(context_knn),
    )
    skeleton_near_road = _cross_type_near_pairs(
        skeleton,
        road_segments,
        left_id_column="skeleton_id",
        right_id_column="road_id",
        radius_m=float(skeleton_road_radius_m),
        max_neighbors=int(context_knn),
    )

    crs = str(skeleton.crs or road_segments.crs or buildings_nodes.crs) if (skeleton.crs or road_segments.crs or buildings_nodes.crs) else None
    return SkeletonContextGraph(
        skeleton_segments=skeleton,
        buildings=buildings_nodes,
        road_segments=road_segments,
        skeleton_adjacent_skeleton=skeleton_adjacent,
        building_near_building=building_near_building,
        building_near_skeleton=building_near_skeleton,
        building_near_road=building_near_road,
        skeleton_near_road=skeleton_near_road,
        road_adjacent_road=road_adjacent,
        crs=crs,
        metadata={
            "include_drainage": bool(include_drainage),
            "n_skeleton_segments": int(len(skeleton)),
            "n_buildings": int(len(buildings_nodes)),
            "n_road_segments": int(len(road_segments)),
            "n_skeleton_adjacent_edges": int(skeleton_adjacent.shape[1]),
            "n_building_building_edges": int(building_near_building.shape[1]),
            "n_building_skeleton_edges": int(building_near_skeleton.shape[1]),
            "n_building_road_edges": int(building_near_road.shape[1]),
            "n_skeleton_road_edges": int(skeleton_near_road.shape[1]),
            "n_road_road_edges": int(road_adjacent.shape[1]),
        },
    )
