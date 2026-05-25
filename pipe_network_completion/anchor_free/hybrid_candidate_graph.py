"""Hybrid anchor-free candidate corridor generation.

# Workstream: Codex

This module builds candidate utility corridors from roads and building demand
proxies only. It intentionally has no utility-truth input; truth geometry is
used later by the labeler/evaluator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import nearest_points


ROAD_BACKBONE = "road_backbone"
BUILDING_ACCESS = "building_access"
DEMAND_KNN = "demand_knn"
DEMAND_MST = "demand_mst"


@dataclass(frozen=True)
class HybridCandidateGraphResult:
    """Candidate corridor GeoDataFrame plus demand-cluster diagnostics."""

    candidates: gpd.GeoDataFrame
    demand_points: gpd.GeoDataFrame
    demand_clusters: gpd.GeoDataFrame
    metadata: dict


def _iter_linestrings(geom):
    if isinstance(geom, LineString):
        yield geom
    elif isinstance(geom, MultiLineString):
        yield from geom.geoms


def _coerce_crs(gdf: gpd.GeoDataFrame | None, target_crs: str | None):
    if gdf is None:
        return None
    out = gdf.copy()
    if target_crs and out.crs is not None and str(out.crs) != str(target_crs):
        out = out.to_crs(target_crs)
    return out


def _point_from_geometry(geom) -> Point | None:
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Point):
        return geom
    try:
        return geom.representative_point()
    except Exception:
        return geom.centroid


def build_demand_points(
    *,
    buildings_gdf: gpd.GeoDataFrame | None = None,
    building_points_gdf: gpd.GeoDataFrame | None = None,
    target_crs: str | None = None,
) -> gpd.GeoDataFrame:
    """Build demand points from non-anchor building data.

    Building points are preferred when supplied. Otherwise building footprint
    representative points are used. No utility truth or utility anchor layers
    are accepted by this function.
    """

    source = _coerce_crs(building_points_gdf, target_crs)
    source_name = "building_point"
    if source is None or source.empty:
        source = _coerce_crs(buildings_gdf, target_crs)
        source_name = "building_footprint"
    if source is None or source.empty:
        return gpd.GeoDataFrame(
            {"demand_id": [], "demand_source": [], "demand_proxy": []},
            geometry=[],
            crs=target_crs,
        )

    records: list[dict] = []
    for i, row in source.reset_index(drop=True).iterrows():
        point = _point_from_geometry(row.geometry)
        if point is None:
            continue
        demand_proxy = 1.0
        for column in ("demand_proxy", "population", "units", "building_count"):
            if column in source.columns:
                try:
                    value = float(row[column])
                    if math.isfinite(value):
                        demand_proxy = value
                        break
                except Exception:
                    pass
        records.append(
            {
                "demand_id": len(records),
                "source_index": int(i),
                "demand_source": source_name,
                "demand_proxy": float(demand_proxy),
                "geometry": point,
            }
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=source.crs)


def cluster_demand_points(
    demand_points: gpd.GeoDataFrame,
    *,
    grid_size_m: float = 150.0,
) -> gpd.GeoDataFrame:
    """Aggregate demand points into regular-grid demand clusters."""

    if demand_points is None or demand_points.empty:
        return gpd.GeoDataFrame(
            {
                "demand_cluster_id": [],
                "demand_count": [],
                "demand_proxy_sum": [],
                "grid_x": [],
                "grid_y": [],
            },
            geometry=[],
            crs=getattr(demand_points, "crs", None),
        )
    grid_size = max(float(grid_size_m), 1.0)
    frame = demand_points.copy()
    frame["grid_x"] = np.floor(frame.geometry.x.astype(float) / grid_size).astype(int)
    frame["grid_y"] = np.floor(frame.geometry.y.astype(float) / grid_size).astype(int)
    if "demand_proxy" not in frame.columns:
        frame["demand_proxy"] = 1.0

    records: list[dict] = []
    for (grid_x, grid_y), group in frame.groupby(["grid_x", "grid_y"], sort=True):
        weights = group["demand_proxy"].astype(float).clip(lower=0.0).to_numpy()
        if float(weights.sum()) <= 0.0:
            weights = np.ones(len(group), dtype=float)
        xs = group.geometry.x.to_numpy(dtype=float)
        ys = group.geometry.y.to_numpy(dtype=float)
        cx = float(np.average(xs, weights=weights))
        cy = float(np.average(ys, weights=weights))
        records.append(
            {
                "demand_cluster_id": len(records),
                "demand_count": int(len(group)),
                "demand_proxy_sum": float(group["demand_proxy"].astype(float).sum()),
                "grid_x": int(grid_x),
                "grid_y": int(grid_y),
                "geometry": Point(cx, cy),
            }
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=demand_points.crs)


def _road_lines(
    roads_gdf: gpd.GeoDataFrame,
    keep_columns: Iterable[str] | None,
) -> list[dict]:
    keep_columns = [c for c in list(keep_columns or []) if c in roads_gdf.columns]
    records: list[dict] = []
    for source_index, row in roads_gdf.reset_index(drop=True).iterrows():
        for geom in _iter_linestrings(row.geometry):
            if geom is None or geom.is_empty or float(geom.length) <= 0.0:
                continue
            record = {
                "candidate_source": ROAD_BACKBONE,
                "candidate_weight": 1.0,
                "demand_u": -1,
                "demand_v": -1,
                "nearest_road_distance_m": 0.0,
                "source_index": int(source_index),
                "geometry": geom,
            }
            for column in keep_columns:
                record[column] = row[column]
            records.append(record)
    return records


def _empty_candidates(crs) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "candidate_id": [],
            "candidate_source": [],
            "candidate_weight": [],
            "demand_u": [],
            "demand_v": [],
            "nearest_road_distance_m": [],
        },
        geometry=[],
        crs=crs,
    )


def _dedupe_records(records: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for record in records:
        geom = record.get("geometry")
        if geom is None or geom.is_empty:
            continue
        coords = list(geom.coords)
        if len(coords) < 2 or float(geom.length) <= 0.0:
            continue
        endpoints = tuple(
            sorted(
                [
                    (round(coords[0][0], 3), round(coords[0][1], 3)),
                    (round(coords[-1][0], 3), round(coords[-1][1], 3)),
                ]
            )
        )
        key = endpoints
        if key in seen:
            continue
        seen.add(key)
        record = dict(record)
        record["candidate_id"] = len(deduped)
        deduped.append(record)
    return deduped


def _add_building_access_records(
    records: list[dict],
    *,
    roads_gdf: gpd.GeoDataFrame,
    demand_clusters: gpd.GeoDataFrame,
    nearest_road_max_distance_m: float,
) -> None:
    if roads_gdf.empty or demand_clusters.empty:
        return
    road_lines = list(roads_gdf.geometry)
    road_index = roads_gdf.sindex
    max_distance = float(nearest_road_max_distance_m)
    for row in demand_clusters.itertuples(index=False):
        point = row.geometry
        search_area = point.buffer(max_distance)
        try:
            candidate_indexes = list(road_index.query(search_area, predicate="intersects"))
        except TypeError:
            candidate_indexes = list(road_index.query(search_area))
        best_distance = float("inf")
        best_point = None
        for idx in candidate_indexes:
            road = road_lines[int(idx)]
            distance = float(point.distance(road))
            if distance < best_distance:
                best_distance = distance
                _, road_point = nearest_points(point, road)
                best_point = road_point
        if best_point is None or best_distance > max_distance:
            continue
        line = LineString([point, best_point])
        if float(line.length) <= 0.0:
            continue
        records.append(
            {
                "candidate_source": BUILDING_ACCESS,
                "candidate_weight": float(getattr(row, "demand_proxy_sum", 1.0)),
                "demand_u": int(row.demand_cluster_id),
                "demand_v": -1,
                "nearest_road_distance_m": best_distance,
                "source_index": -1,
                "geometry": line,
            }
        )


def _demand_neighbor_pairs(
    demand_clusters: gpd.GeoDataFrame,
    *,
    k: int,
    max_distance_m: float,
) -> list[tuple[int, int, float]]:
    if demand_clusters.empty or len(demand_clusters) < 2 or int(k) <= 0:
        return []
    coords = np.column_stack(
        [
            demand_clusters.geometry.x.to_numpy(dtype=float),
            demand_clusters.geometry.y.to_numpy(dtype=float),
        ]
    )
    try:
        from sklearn.neighbors import NearestNeighbors

        n_neighbors = min(len(coords), int(k) + 1)
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(coords)
        distances, indexes = nn.kneighbors(coords)
    except Exception:
        diff = coords[:, None, :] - coords[None, :, :]
        distance_matrix = np.sqrt((diff * diff).sum(axis=2))
        indexes = np.argsort(distance_matrix, axis=1)[:, : int(k) + 1]
        distances = np.take_along_axis(distance_matrix, indexes, axis=1)

    ids = demand_clusters["demand_cluster_id"].astype(int).to_numpy()
    pairs: dict[tuple[int, int], float] = {}
    for row_index in range(len(coords)):
        for distance, neighbor_row in zip(distances[row_index], indexes[row_index]):
            if int(neighbor_row) == row_index:
                continue
            if float(distance) > float(max_distance_m):
                continue
            a, b = sorted((int(ids[row_index]), int(ids[int(neighbor_row)])))
            if a == b:
                continue
            current = pairs.get((a, b))
            if current is None or float(distance) < current:
                pairs[(a, b)] = float(distance)
    return [(a, b, d) for (a, b), d in pairs.items()]


def _add_demand_connection_records(
    records: list[dict],
    *,
    demand_clusters: gpd.GeoDataFrame,
    pairs: list[tuple[int, int, float]],
    source: str,
) -> None:
    if not pairs:
        return
    points = {
        int(row.demand_cluster_id): row.geometry
        for row in demand_clusters.itertuples(index=False)
    }
    demand_proxy = {
        int(row.demand_cluster_id): float(getattr(row, "demand_proxy_sum", 1.0))
        for row in demand_clusters.itertuples(index=False)
    }
    for a, b, distance in pairs:
        pa = points.get(int(a))
        pb = points.get(int(b))
        if pa is None or pb is None:
            continue
        line = LineString([pa, pb])
        if float(line.length) <= 0.0:
            continue
        records.append(
            {
                "candidate_source": source,
                "candidate_weight": 0.5 * (demand_proxy[int(a)] + demand_proxy[int(b)]),
                "demand_u": int(a),
                "demand_v": int(b),
                "nearest_road_distance_m": float(distance),
                "source_index": -1,
                "geometry": line,
            }
        )


def _mst_pairs_from_neighbor_pairs(
    pairs: list[tuple[int, int, float]],
) -> list[tuple[int, int, float]]:
    graph = nx.Graph()
    for a, b, distance in pairs:
        graph.add_edge(int(a), int(b), weight=float(distance))
    out: list[tuple[int, int, float]] = []
    for nodes in nx.connected_components(graph):
        tree = nx.minimum_spanning_tree(graph.subgraph(nodes), weight="weight")
        for a, b, data in tree.edges(data=True):
            out.append((int(a), int(b), float(data.get("weight", 0.0))))
    return out


def build_hybrid_candidate_lines(
    roads_gdf: gpd.GeoDataFrame,
    *,
    buildings_gdf: gpd.GeoDataFrame | None = None,
    building_points_gdf: gpd.GeoDataFrame | None = None,
    target_crs: str | None = None,
    keep_columns: Iterable[str] | None = None,
    demand_cluster_grid_m: float = 150.0,
    nearest_road_max_distance_m: float = 300.0,
    knn_k: int = 3,
    knn_max_distance_m: float = 500.0,
    include_road_backbone: bool = True,
    include_building_access: bool = True,
    include_demand_knn: bool = True,
    include_demand_mst: bool = True,
) -> HybridCandidateGraphResult:
    """Build road + demand candidate utility corridors without truth inputs."""

    roads = _coerce_crs(roads_gdf, target_crs)
    if roads is None:
        roads = gpd.GeoDataFrame(geometry=[], crs=target_crs)
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
    demand_points = build_demand_points(
        buildings_gdf=buildings_gdf,
        building_points_gdf=building_points_gdf,
        target_crs=target_crs or (str(roads.crs) if roads.crs is not None else None),
    )
    if roads.crs is not None and demand_points.crs is not None and str(demand_points.crs) != str(roads.crs):
        demand_points = demand_points.to_crs(roads.crs)
    demand_clusters = cluster_demand_points(
        demand_points,
        grid_size_m=float(demand_cluster_grid_m),
    )

    records: list[dict] = []
    if include_road_backbone:
        records.extend(_road_lines(roads, keep_columns))
    if include_building_access:
        _add_building_access_records(
            records,
            roads_gdf=roads,
            demand_clusters=demand_clusters,
            nearest_road_max_distance_m=float(nearest_road_max_distance_m),
        )

    pairs = _demand_neighbor_pairs(
        demand_clusters,
        k=int(knn_k),
        max_distance_m=float(knn_max_distance_m),
    )
    if include_demand_knn:
        _add_demand_connection_records(
            records,
            demand_clusters=demand_clusters,
            pairs=pairs,
            source=DEMAND_KNN,
        )
    if include_demand_mst:
        _add_demand_connection_records(
            records,
            demand_clusters=demand_clusters,
            pairs=_mst_pairs_from_neighbor_pairs(pairs),
            source=DEMAND_MST,
        )

    records = _dedupe_records(records)
    if not records:
        candidates = _empty_candidates(roads.crs)
    else:
        candidates = gpd.GeoDataFrame(records, geometry="geometry", crs=roads.crs)
    if not candidates.empty:
        candidates["candidate_id"] = candidates["candidate_id"].astype(int)
        candidates["candidate_source"] = candidates["candidate_source"].fillna("unknown")
        candidates["candidate_weight"] = candidates["candidate_weight"].fillna(1.0).astype(float)
        candidates["demand_u"] = candidates["demand_u"].fillna(-1).astype(int)
        candidates["demand_v"] = candidates["demand_v"].fillna(-1).astype(int)
        candidates["nearest_road_distance_m"] = (
            candidates["nearest_road_distance_m"].fillna(0.0).astype(float)
        )

    by_source = {}
    if not candidates.empty:
        by_source = {
            str(source): {
                "count": int(len(group)),
                "length_m": float(group.geometry.length.sum()),
            }
            for source, group in candidates.groupby("candidate_source")
        }
    metadata = {
        "candidate_graph_type": "hybrid",
        "n_candidates": int(len(candidates)),
        "n_demand_points": int(len(demand_points)),
        "n_demand_clusters": int(len(demand_clusters)),
        "candidate_by_source": by_source,
        "demand_cluster_grid_m": float(demand_cluster_grid_m),
        "nearest_road_max_distance_m": float(nearest_road_max_distance_m),
        "knn_k": int(knn_k),
        "knn_max_distance_m": float(knn_max_distance_m),
    }
    return HybridCandidateGraphResult(
        candidates=candidates,
        demand_points=demand_points,
        demand_clusters=demand_clusters,
        metadata=metadata,
    )
