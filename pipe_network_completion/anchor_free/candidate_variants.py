"""Candidate-support variants for anchor-free utility corridor experiments.

Workstream: Codex

These builders use only visible/non-utility context. Utility truth is not an
input; it belongs in label generation and representability evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString

from pipe_network_completion.anchor_free.hybrid_candidate_graph import (
    build_hybrid_candidate_lines,
)


ROAD_BACKBONE = "road_backbone"
ROAD_OFFSET = "road_offset"
WATERCOURSE_DRAINAGE = "watercourse_drainage"
WATERCOURSE_CENTRELINE = "watercourse_corridor_centreline"

ROAD = "road"
HYBRID = "hybrid"
ROAD_OFFSETS = "road_offsets"
WATERCOURSES = "watercourses"
HYBRID_WATERCOURSES = "hybrid_watercourses"
MULTI_SUPPORT = "multi_support"

SUPPORTED_VARIANTS = (
    ROAD,
    HYBRID,
    ROAD_OFFSETS,
    WATERCOURSES,
    HYBRID_WATERCOURSES,
    MULTI_SUPPORT,
)


@dataclass(frozen=True)
class CandidateVariantResult:
    candidates: gpd.GeoDataFrame
    metadata: dict


def _coerce_crs(gdf: gpd.GeoDataFrame | None, target_crs: str | None):
    if gdf is None:
        return None
    out = gdf.copy()
    if target_crs and out.crs is not None and str(out.crs) != str(target_crs):
        out = out.to_crs(target_crs)
    return out


def _iter_linestrings(geom):
    if isinstance(geom, LineString):
        yield geom
    elif isinstance(geom, MultiLineString):
        yield from geom.geoms
    elif hasattr(geom, "geoms"):
        for part in geom.geoms:
            yield from _iter_linestrings(part)


def _line_record(
    *,
    source: str,
    source_index: int,
    geometry: LineString,
    candidate_weight: float = 1.0,
    nearest_road_distance_m: float = 0.0,
    extra: dict | None = None,
) -> dict:
    record = {
        "candidate_source": source,
        "candidate_weight": float(candidate_weight),
        "demand_u": -1,
        "demand_v": -1,
        "nearest_road_distance_m": float(nearest_road_distance_m),
        "source_index": int(source_index),
        "geometry": geometry,
    }
    if extra:
        record.update(extra)
    return record


def _records_from_lines(
    gdf: gpd.GeoDataFrame | None,
    *,
    source: str,
    keep_columns: Iterable[str] | None = None,
    candidate_weight: float = 1.0,
) -> list[dict]:
    if gdf is None or gdf.empty:
        return []
    keep_columns = [column for column in list(keep_columns or []) if column in gdf.columns]
    records: list[dict] = []
    for source_index, row in gdf.reset_index(drop=True).iterrows():
        extra = {column: row[column] for column in keep_columns}
        for line in _iter_linestrings(row.geometry):
            if line is None or line.is_empty or float(line.length) <= 0.0:
                continue
            records.append(
                _line_record(
                    source=source,
                    source_index=int(source_index),
                    geometry=line,
                    candidate_weight=candidate_weight,
                    extra=extra,
                )
            )
    return records


def _offset_curve(line: LineString, *, distance_m: float, side: str):
    try:
        return line.parallel_offset(
            float(distance_m),
            side,
            resolution=8,
            join_style=2,
            mitre_limit=5.0,
        )
    except TypeError:
        try:
            return line.parallel_offset(float(distance_m), side)
        except Exception:
            return None
    except Exception:
        return None


def _road_offset_records(
    roads: gpd.GeoDataFrame | None,
    *,
    offset_distances_m: Iterable[float],
    keep_columns: Iterable[str] | None = None,
) -> list[dict]:
    if roads is None or roads.empty:
        return []
    keep_columns = [column for column in list(keep_columns or []) if column in roads.columns]
    records: list[dict] = []
    for source_index, row in roads.reset_index(drop=True).iterrows():
        extra_base = {column: row[column] for column in keep_columns}
        for line in _iter_linestrings(row.geometry):
            if line is None or line.is_empty or float(line.length) <= 0.0:
                continue
            for distance_m in offset_distances_m:
                for side in ("left", "right"):
                    offset = _offset_curve(line, distance_m=float(distance_m), side=side)
                    for offset_line in _iter_linestrings(offset):
                        if (
                            offset_line is None
                            or offset_line.is_empty
                            or float(offset_line.length) <= 0.0
                        ):
                            continue
                        extra = dict(extra_base)
                        extra["road_offset_distance_m"] = float(distance_m)
                        extra["road_offset_side"] = side
                        records.append(
                            _line_record(
                                source=ROAD_OFFSET,
                                source_index=int(source_index),
                                geometry=offset_line,
                                candidate_weight=0.75,
                                nearest_road_distance_m=float(distance_m),
                                extra=extra,
                            )
                        )
    return records


def _geom_key(geom) -> tuple:
    coords = []
    for line in _iter_linestrings(geom):
        coords.extend((round(float(x), 3), round(float(y), 3)) for x, y in line.coords)
    if not coords:
        return ()
    forward = tuple(coords)
    reverse = tuple(reversed(coords))
    return min(forward, reverse)


def _dedupe_records(records: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out: list[dict] = []
    for record in records:
        geom = record.get("geometry")
        if geom is None or geom.is_empty or float(getattr(geom, "length", 0.0)) <= 0.0:
            continue
        key = (record.get("candidate_source", "unknown"), _geom_key(geom))
        if key in seen:
            continue
        seen.add(key)
        new_record = dict(record)
        new_record["candidate_id"] = len(out)
        out.append(new_record)
    return out


def _candidate_frame(records: list[dict], crs) -> gpd.GeoDataFrame:
    records = _dedupe_records(records)
    if not records:
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
    frame = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    frame["candidate_id"] = frame["candidate_id"].astype(int)
    frame["candidate_source"] = frame["candidate_source"].fillna("unknown")
    frame["candidate_weight"] = frame["candidate_weight"].fillna(1.0).astype(float)
    frame["demand_u"] = frame["demand_u"].fillna(-1).astype(int)
    frame["demand_v"] = frame["demand_v"].fillna(-1).astype(int)
    frame["nearest_road_distance_m"] = (
        frame["nearest_road_distance_m"].fillna(0.0).astype(float)
    )
    return frame


def _source_summary(candidates: gpd.GeoDataFrame) -> dict[str, dict[str, float]]:
    if candidates.empty:
        return {}
    return {
        str(source): {
            "count": int(len(group)),
            "length_m": float(group.geometry.length.sum()),
        }
        for source, group in candidates.groupby("candidate_source")
    }


def build_candidate_variant_lines(
    roads_gdf: gpd.GeoDataFrame,
    *,
    variant: str,
    buildings_gdf: gpd.GeoDataFrame | None = None,
    building_points_gdf: gpd.GeoDataFrame | None = None,
    watercourse_drainage_lines_gdf: gpd.GeoDataFrame | None = None,
    watercourse_corridor_centrelines_gdf: gpd.GeoDataFrame | None = None,
    target_crs: str | None = None,
    keep_columns: Iterable[str] | None = None,
    offset_distances_m: Iterable[float] = (15.0, 30.0),
    demand_cluster_grid_m: float = 150.0,
    nearest_road_max_distance_m: float = 300.0,
    knn_k: int = 3,
    knn_max_distance_m: float = 500.0,
) -> CandidateVariantResult:
    """Build candidate LineStrings for a named support variant."""

    variant = str(variant).lower()
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported candidate variant: {variant!r}")

    roads = _coerce_crs(roads_gdf, target_crs)
    if roads is None:
        roads = gpd.GeoDataFrame(geometry=[], crs=target_crs)
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
    crs = roads.crs
    keep_columns = list(keep_columns or [])
    records: list[dict] = []

    if variant == ROAD:
        records.extend(_records_from_lines(roads, source=ROAD_BACKBONE, keep_columns=keep_columns))

    elif variant == HYBRID:
        hybrid = build_hybrid_candidate_lines(
            roads,
            buildings_gdf=buildings_gdf,
            building_points_gdf=building_points_gdf,
            target_crs=target_crs,
            keep_columns=keep_columns,
            demand_cluster_grid_m=float(demand_cluster_grid_m),
            nearest_road_max_distance_m=float(nearest_road_max_distance_m),
            knn_k=int(knn_k),
            knn_max_distance_m=float(knn_max_distance_m),
        )
        candidates = hybrid.candidates.copy()
        candidates["candidate_id"] = range(len(candidates))
        return CandidateVariantResult(
            candidates=candidates,
            metadata={
                **hybrid.metadata,
                "candidate_graph_type": HYBRID,
                "candidate_by_source": _source_summary(candidates),
            },
        )

    elif variant == ROAD_OFFSETS:
        records.extend(_records_from_lines(roads, source=ROAD_BACKBONE, keep_columns=keep_columns))
        records.extend(
            _road_offset_records(
                roads,
                offset_distances_m=offset_distances_m,
                keep_columns=keep_columns,
            )
        )

    elif variant == WATERCOURSES:
        drainage = _coerce_crs(watercourse_drainage_lines_gdf, target_crs)
        centrelines = _coerce_crs(watercourse_corridor_centrelines_gdf, target_crs)
        records.extend(_records_from_lines(drainage, source=WATERCOURSE_DRAINAGE))
        records.extend(_records_from_lines(centrelines, source=WATERCOURSE_CENTRELINE))

    elif variant == HYBRID_WATERCOURSES:
        hybrid = build_hybrid_candidate_lines(
            roads,
            buildings_gdf=buildings_gdf,
            building_points_gdf=building_points_gdf,
            target_crs=target_crs,
            keep_columns=keep_columns,
            demand_cluster_grid_m=float(demand_cluster_grid_m),
            nearest_road_max_distance_m=float(nearest_road_max_distance_m),
            knn_k=int(knn_k),
            knn_max_distance_m=float(knn_max_distance_m),
        )
        records.extend(hybrid.candidates.drop(columns=["candidate_id"], errors="ignore").to_dict("records"))
        drainage = _coerce_crs(watercourse_drainage_lines_gdf, target_crs)
        centrelines = _coerce_crs(watercourse_corridor_centrelines_gdf, target_crs)
        records.extend(_records_from_lines(drainage, source=WATERCOURSE_DRAINAGE))
        records.extend(_records_from_lines(centrelines, source=WATERCOURSE_CENTRELINE))

    elif variant == MULTI_SUPPORT:
        hybrid = build_hybrid_candidate_lines(
            roads,
            buildings_gdf=buildings_gdf,
            building_points_gdf=building_points_gdf,
            target_crs=target_crs,
            keep_columns=keep_columns,
            demand_cluster_grid_m=float(demand_cluster_grid_m),
            nearest_road_max_distance_m=float(nearest_road_max_distance_m),
            knn_k=int(knn_k),
            knn_max_distance_m=float(knn_max_distance_m),
        )
        records.extend(hybrid.candidates.drop(columns=["candidate_id"], errors="ignore").to_dict("records"))
        records.extend(
            _road_offset_records(
                roads,
                offset_distances_m=offset_distances_m,
                keep_columns=keep_columns,
            )
        )
        drainage = _coerce_crs(watercourse_drainage_lines_gdf, target_crs)
        centrelines = _coerce_crs(watercourse_corridor_centrelines_gdf, target_crs)
        records.extend(_records_from_lines(drainage, source=WATERCOURSE_DRAINAGE))
        records.extend(_records_from_lines(centrelines, source=WATERCOURSE_CENTRELINE))

    candidates = _candidate_frame(records, crs)
    metadata = {
        "candidate_graph_type": variant,
        "n_candidates": int(len(candidates)),
        "candidate_by_source": _source_summary(candidates),
        "offset_distances_m": [float(value) for value in offset_distances_m],
    }
    return CandidateVariantResult(candidates=candidates, metadata=metadata)
