"""Heterogeneous road graph for the anchor-free pipeline.

# Workstream: Claude

Phase 2.A of ``docs/research_notes/architectural_cleanup_plan.md``.

The graph has two node types and two relation types (plus the reverse
relations that PyG adds via ``T.ToUndirected``):

* ``RoadSegment`` nodes — one per LineString in the source roads table.
  Carries the rich features (length, bearing, building / built-up / DEM
  context). This is the **prediction target**: per-segment binary label
  "does this road segment carry a utility line within ``label_buffer_m``?".
* ``Intersection`` nodes — one per snapped geometric meeting point.
  Carries sparse features (``x``, ``y``, ``degree``). Provides
  positional / structural context.

* ``("RoadSegment", "crosses", "RoadSegment")`` — sjoin-derived
  adjacency. Mirrors ISARC's ``spatial_join_intersects(roads, roads)``
  at ``process.py:239-243``. Catches every road-road meeting regardless
  of whether the LineStrings are noded at intersections.
* ``("RoadSegment", "touches", "Intersection")`` — built from snapped
  LineString endpoints. When the source data is well-noded, this matches
  the geometric reality.

The data model intentionally mirrors ISARC's anchor-based heterogeneous
graph (``MH`` + ``Road`` node types with ``("MH", "link", "MH")``
prediction-target edges and ``("MH", "near", "Road")`` context edges):
we replace ``MH`` with ``Intersection`` and move the prediction target
from MH-MH edges to per-``RoadSegment`` node labels.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point


# Same snap tolerance / CRS guard conventions as the legacy road_graph module.
DEFAULT_SNAP_TOLERANCE_M = 1.0
GEOGRAPHIC_CRS_HINTS = ("4326", "WGS 84", "GCS", "NAD83 (lat", "EPSG:4326")


@dataclass(frozen=True)
class HeteroRoadGraph:
    """Anchor-free heterogeneous road graph.

    Attributes
    ----------
    road_segments : GeoDataFrame
        One row per road LineString. Columns:
        ``segment_id`` (int), ``length_m`` (float), ``bearing_rad`` (float),
        any passthrough columns from the source ``keep_columns``, and
        ``geometry``.
    intersections : GeoDataFrame
        One row per snapped meeting point. Columns:
        ``intersection_id`` (int), ``x``, ``y``, ``degree``, ``geometry``.
    segment_crosses_segment : np.ndarray
        Shape ``(2, n_cross_edges)``. Each column is a pair of segment
        ids that physically intersect (sjoin predicate='intersects').
        Self-pairs filtered. Both directions are stored so the relation
        is symmetric.
    segment_touches_intersection : np.ndarray
        Shape ``(2, n_touch_edges)``. Each column is a pair
        ``(segment_id, intersection_id)``.
    crs : str | None
        CRS of the geometries.
    metadata : dict
        Build-time diagnostics (snap tolerance, source counts, etc.).
    """

    road_segments: gpd.GeoDataFrame
    intersections: gpd.GeoDataFrame
    segment_crosses_segment: np.ndarray
    segment_touches_intersection: np.ndarray
    crs: str | None = None
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Lightweight accessors used by features / labels / pipeline.
    # ------------------------------------------------------------------
    @property
    def segment_ids(self) -> np.ndarray:
        return self.road_segments["segment_id"].to_numpy(dtype=int)

    @property
    def intersection_ids(self) -> np.ndarray:
        return self.intersections["intersection_id"].to_numpy(dtype=int)


def _crs_is_geographic(crs) -> bool:
    if crs is None:
        return False
    try:
        return bool(getattr(crs, "is_geographic", False))
    except Exception:
        crs_repr = str(crs).upper()
        return any(hint.upper() in crs_repr for hint in GEOGRAPHIC_CRS_HINTS)


def _iter_linestrings(geom):
    if isinstance(geom, LineString):
        yield geom
    elif isinstance(geom, MultiLineString):
        for sub in geom.geoms:
            yield sub


def _snap_point(point: Point, snap_tolerance: float) -> tuple[float, float]:
    if snap_tolerance <= 0:
        return (float(point.x), float(point.y))
    factor = 1.0 / snap_tolerance
    return (round(point.x * factor) / factor, round(point.y * factor) / factor)


def _bearing_rad(start: Point, end: Point) -> float:
    return math.atan2(end.y - start.y, end.x - start.x)


def build_hetero_road_graph(
    roads_gdf: gpd.GeoDataFrame,
    *,
    target_crs: str | None = None,
    snap_tolerance_m: float = DEFAULT_SNAP_TOLERANCE_M,
    keep_columns: Iterable[str] | None = None,
) -> HeteroRoadGraph:
    """Build the heterogeneous road graph from a road-centerline GeoDataFrame.

    Mirrors ``road_graph.build_road_candidate_graph`` for the segment-side
    extraction (one LineString → one ``RoadSegment``) and uses
    ``gpd.sjoin(predicate='intersects')`` for road-road adjacency,
    matching ISARC's ``process.py:239-243``.
    """
    keep_columns = list(keep_columns or [])

    if roads_gdf.empty:
        empty_segments = gpd.GeoDataFrame(
            {
                "segment_id": [],
                "length_m": [],
                "bearing_rad": [],
                **{col: [] for col in keep_columns if col in roads_gdf.columns},
            },
            geometry=[],
            crs=target_crs,
        )
        empty_inter = gpd.GeoDataFrame(
            {"intersection_id": [], "x": [], "y": [], "degree": []},
            geometry=[],
            crs=target_crs,
        )
        return HeteroRoadGraph(
            road_segments=empty_segments,
            intersections=empty_inter,
            segment_crosses_segment=np.zeros((2, 0), dtype=np.int64),
            segment_touches_intersection=np.zeros((2, 0), dtype=np.int64),
            crs=target_crs,
            metadata={"empty": True},
        )

    if (
        target_crs is not None
        and roads_gdf.crs is not None
        and str(roads_gdf.crs) != str(target_crs)
    ):
        roads_gdf = roads_gdf.to_crs(target_crs)

    if _crs_is_geographic(roads_gdf.crs):
        warnings.warn(
            "Road CRS appears geographic (lat/lon). Edge lengths are computed "
            "in CRS units; set target_crs to a projected CRS (e.g. EPSG:3857 "
            "or a local UTM zone) for meter-valued lengths.",
            stacklevel=2,
        )

    keep_columns = [c for c in keep_columns if c in roads_gdf.columns]

    # ------------------------------------------------------------------
    # Build RoadSegment + Intersection node tables.
    # ------------------------------------------------------------------
    intersection_lookup: dict[tuple[float, float], int] = {}
    intersection_records: list[dict] = []

    def _get_intersection(point: Point) -> int:
        key = _snap_point(point, snap_tolerance_m)
        i = intersection_lookup.get(key)
        if i is None:
            i = len(intersection_lookup)
            intersection_lookup[key] = i
            intersection_records.append(
                {
                    "intersection_id": i,
                    "x": float(key[0]),
                    "y": float(key[1]),
                    "geometry": Point(key[0], key[1]),
                }
            )
        return i

    segment_records: list[dict] = []
    touches_pairs: list[tuple[int, int]] = []
    next_segment_id = 0

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
            i_u = _get_intersection(start_pt)
            i_v = _get_intersection(end_pt)
            length = float(sub.length)
            bearing = _bearing_rad(start_pt, end_pt)
            record = {
                "segment_id": next_segment_id,
                "length_m": length,
                "bearing_rad": bearing,
                "source_index": source_index,
                "geometry": sub,
            }
            for col in keep_columns:
                record[col] = row[col]
            segment_records.append(record)
            # Each segment touches its start and (if different) end
            # intersection. Self-loop touches (start==end) are kept
            # because the intersection still has a degree contribution.
            touches_pairs.append((next_segment_id, i_u))
            if i_v != i_u:
                touches_pairs.append((next_segment_id, i_v))
            next_segment_id += 1

    segments_df = gpd.GeoDataFrame(segment_records, crs=roads_gdf.crs)
    inter_df = gpd.GeoDataFrame(intersection_records, crs=roads_gdf.crs)

    # Intersection degree = number of distinct segments that touch it.
    if touches_pairs:
        touches_arr = np.asarray(touches_pairs, dtype=np.int64).T  # (2, n)
        # degree counts unique segments per intersection
        deg_series = (
            pd.DataFrame(
                {
                    "segment_id": touches_arr[0],
                    "intersection_id": touches_arr[1],
                }
            )
            .drop_duplicates()
            .groupby("intersection_id")
            .size()
        )
        inter_df["degree"] = (
            inter_df["intersection_id"].map(deg_series).fillna(0).astype(int)
        )
    else:
        touches_arr = np.zeros((2, 0), dtype=np.int64)
        inter_df["degree"] = 0

    # ------------------------------------------------------------------
    # ("RoadSegment", "crosses", "RoadSegment") via sjoin.
    # Same recipe as process.py:239-243 in the ISARC pipeline.
    # ------------------------------------------------------------------
    if not segments_df.empty:
        try:
            sjoin = gpd.sjoin(
                segments_df[["segment_id", "geometry"]],
                segments_df[["segment_id", "geometry"]],
                predicate="intersects",
                how="inner",
            )
        except TypeError:
            sjoin = gpd.sjoin(
                segments_df[["segment_id", "geometry"]],
                segments_df[["segment_id", "geometry"]],
                op="intersects",
                how="inner",
            )
        a = sjoin["segment_id_left"].to_numpy(dtype=np.int64)
        b = sjoin["segment_id_right"].to_numpy(dtype=np.int64)
        mask = a != b
        a = a[mask]
        b = b[mask]
        # Store both directions so the relation is symmetric in the GNN.
        crosses_arr = np.vstack(
            [
                np.concatenate([a, b]),
                np.concatenate([b, a]),
            ]
        ).astype(np.int64)
        # Deduplicate ordered pairs (sjoin already does, but the double
        # stack can introduce dupes if one direction was present and the
        # other wasn't).
        pair_df = pd.DataFrame(
            {"a": crosses_arr[0], "b": crosses_arr[1]}
        ).drop_duplicates()
        crosses_arr = np.vstack([pair_df["a"].to_numpy(), pair_df["b"].to_numpy()])
    else:
        crosses_arr = np.zeros((2, 0), dtype=np.int64)

    return HeteroRoadGraph(
        road_segments=segments_df,
        intersections=inter_df,
        segment_crosses_segment=crosses_arr,
        segment_touches_intersection=touches_arr,
        crs=str(segments_df.crs) if segments_df.crs is not None else None,
        metadata={
            "snap_tolerance_m": float(snap_tolerance_m),
            "n_road_segments": int(len(segments_df)),
            "n_intersections": int(len(inter_df)),
            "n_crosses_edges": int(crosses_arr.shape[1]),
            "n_touches_edges": int(touches_arr.shape[1]),
        },
    )


def load_hetero_road_graph(
    path: str | Path,
    *,
    target_crs: str | None = None,
    snap_tolerance_m: float = DEFAULT_SNAP_TOLERANCE_M,
    keep_columns: Iterable[str] | None = None,
) -> HeteroRoadGraph:
    """Convenience reader: load a roads vector file and build the hetero graph."""
    roads = gpd.read_file(path)
    return build_hetero_road_graph(
        roads,
        target_crs=target_crs,
        snap_tolerance_m=snap_tolerance_m,
        keep_columns=keep_columns,
    )
