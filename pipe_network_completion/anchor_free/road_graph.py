"""Build a candidate utility-corridor graph from road centerlines only.

The road network supplies both the nodes (intersections / segment endpoints)
and the candidate edges. Ground anchor points (manholes, valves, poles, etc.)
are **not** read by this module.
"""

# Workstream: Codex

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point


# Generous "snap" tolerance for matching road-segment endpoints into shared
# intersection nodes. The default is in CRS units (meters for projected CRS).
DEFAULT_SNAP_TOLERANCE_M = 1.0

# CRS guards: warn (do not fail) when the user supplies a geographic CRS.
GEOGRAPHIC_CRS_HINTS = ("4326", "WGS 84", "GCS", "NAD83 (lat", "EPSG:4326")


@dataclass(frozen=True)
class RoadCandidateGraph:
    """A road-derived graph that serves as the candidate utility corridor set.

    Attributes
    ----------
    nodes : GeoDataFrame
        Point geometries for road-network nodes. Columns: ``node_id`` (int),
        ``x``, ``y``, ``degree``, and ``geometry``.
    edges : GeoDataFrame
        LineString geometries for road segments. Columns include
        ``edge_id`` (int), ``u`` and ``v`` (node ids), ``length_m``,
        ``bearing_rad`` and ``geometry``, plus any pass-through attributes
        from the source road table.
    crs : str | None
        Coordinate reference system of the nodes/edges tables.
    """

    nodes: gpd.GeoDataFrame
    edges: gpd.GeoDataFrame
    crs: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_networkx(self) -> nx.MultiGraph:
        """Return a NetworkX MultiGraph keyed by ``node_id`` / ``edge_id``."""
        graph = nx.MultiGraph()
        for row in self.nodes.itertuples(index=False):
            graph.add_node(int(row.node_id), x=float(row.x), y=float(row.y))
        for row in self.edges.itertuples(index=False):
            graph.add_edge(
                int(row.u),
                int(row.v),
                key=int(row.edge_id),
                edge_id=int(row.edge_id),
                length_m=float(row.length_m),
            )
        return graph


def _crs_is_geographic(crs) -> bool:
    if crs is None:
        return False
    try:
        return bool(getattr(crs, "is_geographic", False))
    except Exception:
        crs_repr = str(crs).upper()
        return any(hint.upper() in crs_repr for hint in GEOGRAPHIC_CRS_HINTS)


def _iter_linestrings(geom) -> Iterable[LineString]:
    if isinstance(geom, LineString):
        yield geom
    elif isinstance(geom, MultiLineString):
        for sub in geom.geoms:
            yield sub
    else:
        # Skip points, polygons, and mixed geometries; only line-like inputs are
        # meaningful for road centerline graphs.
        return


def _round_point(point: Point, snap_tolerance: float) -> tuple[float, float]:
    if snap_tolerance <= 0:
        return (float(point.x), float(point.y))
    # snap to nearest multiple of snap_tolerance to merge near-identical
    # endpoints across adjacent segments
    factor = 1.0 / snap_tolerance
    return (round(point.x * factor) / factor, round(point.y * factor) / factor)


def _bearing_rad(start: Point, end: Point) -> float:
    return math.atan2(end.y - start.y, end.x - start.x)


def build_road_candidate_graph(
    roads_gdf: gpd.GeoDataFrame,
    *,
    target_crs: str | None = None,
    snap_tolerance_m: float = DEFAULT_SNAP_TOLERANCE_M,
    keep_columns: Iterable[str] | None = None,
) -> RoadCandidateGraph:
    """Build a candidate utility graph from a road-centerline GeoDataFrame.

    Parameters
    ----------
    roads_gdf : GeoDataFrame
        Road centerlines (LineString or MultiLineString). Must define a CRS,
        or be re-projected by passing ``target_crs``.
    target_crs : str, optional
        If given, reproject ``roads_gdf`` to this CRS before graph construction.
    snap_tolerance_m : float
        Endpoint coordinates are rounded to this resolution (in CRS units)
        before being keyed into intersection nodes. The default of ``1.0``
        merges endpoints that agree to one meter.
    keep_columns : iterable of str, optional
        Attribute columns from ``roads_gdf`` to pass through onto the edge
        table (e.g. road class). Missing columns are silently ignored.
    """
    if roads_gdf.empty:
        empty_nodes = gpd.GeoDataFrame(
            {"node_id": [], "x": [], "y": [], "degree": []},
            geometry=[],
            crs=target_crs,
        )
        empty_edges = gpd.GeoDataFrame(
            {
                "edge_id": [],
                "u": [],
                "v": [],
                "length_m": [],
                "bearing_rad": [],
            },
            geometry=[],
            crs=target_crs,
        )
        return RoadCandidateGraph(
            nodes=empty_nodes,
            edges=empty_edges,
            crs=target_crs,
            metadata={"empty": True},
        )

    if target_crs is not None and roads_gdf.crs is not None and str(
        roads_gdf.crs
    ) != str(target_crs):
        roads_gdf = roads_gdf.to_crs(target_crs)

    if _crs_is_geographic(roads_gdf.crs):
        warnings.warn(
            "Road CRS appears geographic (lat/lon). Edge lengths are computed in "
            "CRS units; set target_crs to a projected CRS (e.g. EPSG:3857 or a "
            "local UTM zone) for meter-valued lengths.",
            stacklevel=2,
        )

    keep_columns = list(keep_columns or [])
    keep_columns = [c for c in keep_columns if c in roads_gdf.columns]

    node_lookup: dict[tuple[float, float], int] = {}
    node_records: list[dict] = []
    edge_records: list[dict] = []

    def _get_node(point: Point) -> int:
        key = _round_point(point, snap_tolerance_m)
        node_id = node_lookup.get(key)
        if node_id is None:
            node_id = len(node_lookup)
            node_lookup[key] = node_id
            node_records.append(
                {
                    "node_id": node_id,
                    "x": float(key[0]),
                    "y": float(key[1]),
                    "geometry": Point(key[0], key[1]),
                }
            )
        return node_id

    next_edge_id = 0
    for source_index, row in roads_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        for sub in _iter_linestrings(geom):
            coords = list(sub.coords)
            if len(coords) < 2:
                continue
            start_pt = Point(coords[0])
            end_pt = Point(coords[-1])
            u = _get_node(start_pt)
            v = _get_node(end_pt)
            if u == v:
                # Self-loop after snapping; skip because it is not meaningful
                # as a candidate corridor.
                continue
            length = float(sub.length)
            bearing = _bearing_rad(start_pt, end_pt)
            record = {
                "edge_id": next_edge_id,
                "u": int(u),
                "v": int(v),
                "length_m": length,
                "bearing_rad": bearing,
                "source_index": source_index,
                "geometry": sub,
            }
            for col in keep_columns:
                record[col] = row[col]
            edge_records.append(record)
            next_edge_id += 1

    nodes_df = gpd.GeoDataFrame(node_records, crs=roads_gdf.crs)
    edges_df = gpd.GeoDataFrame(edge_records, crs=roads_gdf.crs)

    # Compute node degree from the edge table.
    if not edges_df.empty:
        degree = pd.concat(
            [edges_df["u"], edges_df["v"]], ignore_index=True
        ).value_counts()
        nodes_df["degree"] = nodes_df["node_id"].map(degree).fillna(0).astype(int)
    else:
        nodes_df["degree"] = 0

    return RoadCandidateGraph(
        nodes=nodes_df,
        edges=edges_df,
        crs=str(nodes_df.crs) if nodes_df.crs is not None else None,
        metadata={
            "snap_tolerance_m": snap_tolerance_m,
            "n_nodes": len(nodes_df),
            "n_edges": len(edges_df),
        },
    )


def load_road_candidate_graph(
    path: str | Path,
    *,
    target_crs: str | None = None,
    snap_tolerance_m: float = DEFAULT_SNAP_TOLERANCE_M,
    keep_columns: Iterable[str] | None = None,
) -> RoadCandidateGraph:
    """Load roads from a vector file and build the candidate graph."""
    roads_gdf = gpd.read_file(path)
    return build_road_candidate_graph(
        roads_gdf,
        target_crs=target_crs,
        snap_tolerance_m=snap_tolerance_m,
        keep_columns=keep_columns,
    )


def edge_midpoints(graph: RoadCandidateGraph) -> np.ndarray:
    """Return an ``(n_edges, 2)`` array of edge midpoint coordinates."""
    if graph.edges.empty:
        return np.zeros((0, 2), dtype=float)
    centroids = graph.edges.geometry.interpolate(0.5, normalized=True)
    return np.column_stack([centroids.x.values, centroids.y.values])
