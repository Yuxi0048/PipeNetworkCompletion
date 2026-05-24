"""Small non-overlapping AOI helpers for anchor-free experiments.

Workstream: Codex
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box


@dataclass(frozen=True)
class AOIThresholds:
    """Minimum content required before an AOI is useful for training/evaluation."""

    min_road_length_m: float = 5_000.0
    min_truth_length_m: float = 1_000.0
    min_building_points: int = 25


def make_non_overlapping_grid_aois(
    bounds: Iterable[float],
    *,
    tile_size_m: float,
    gap_m: float = 0.0,
    crs: str | None = None,
    aoi_prefix: str = "aoi",
) -> gpd.GeoDataFrame:
    """Create square AOIs separated by a configurable gap.

    The AOIs are fully contained by `bounds`. `gap_m` is space between adjacent
    AOIs, which keeps train/validation/test blocks from touching each other.
    """

    minx, miny, maxx, maxy = [float(value) for value in bounds]
    tile_size = float(tile_size_m)
    gap = max(float(gap_m), 0.0)
    if tile_size <= 0:
        raise ValueError("tile_size_m must be positive.")
    if maxx <= minx or maxy <= miny:
        raise ValueError("bounds must have positive width and height.")

    step = tile_size + gap
    records: list[dict] = []
    row = 0
    y = miny
    while y + tile_size <= maxy + 1e-9:
        col = 0
        x = minx
        while x + tile_size <= maxx + 1e-9:
            records.append(
                {
                    "aoi_id": f"{aoi_prefix}_{row:02d}_{col:02d}",
                    "grid_row": row,
                    "grid_col": col,
                    "tile_size_m": tile_size,
                    "gap_m": gap,
                    "geometry": box(x, y, x + tile_size, y + tile_size),
                }
            )
            col += 1
            x += step
        row += 1
        y += step

    return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)


def _to_crs(gdf: gpd.GeoDataFrame | None, crs) -> gpd.GeoDataFrame | None:
    if gdf is None:
        return None
    if crs is not None and gdf.crs is not None and str(gdf.crs) != str(crs):
        return gdf.to_crs(crs)
    return gdf


def _intersecting_subset(gdf: gpd.GeoDataFrame, geom) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()
    try:
        idx = gdf.sindex.query(geom, predicate="intersects")
        return gdf.iloc[np.asarray(idx, dtype=int)].copy()
    except Exception:
        return gdf[gdf.geometry.intersects(geom)].copy()


def _length_within(gdf: gpd.GeoDataFrame, geom) -> float:
    subset = _intersecting_subset(gdf, geom)
    if subset.empty:
        return 0.0
    clipped = gpd.clip(subset, geom)
    if clipped.empty:
        return 0.0
    return float(clipped.geometry.length.sum())


def _point_count_within(gdf: gpd.GeoDataFrame | None, geom) -> int:
    if gdf is None or gdf.empty:
        return 0
    subset = _intersecting_subset(gdf, geom)
    if subset.empty:
        return 0
    return int(subset.geometry.within(geom).sum())


def summarize_aoi_content(
    aois: gpd.GeoDataFrame,
    *,
    roads: gpd.GeoDataFrame,
    utility_truth: gpd.GeoDataFrame,
    building_points: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Summarize non-anchor context and truth-label availability per AOI."""

    if aois.crs is None:
        raise ValueError("AOIs must have a CRS.")
    roads = _to_crs(roads, aois.crs)
    utility_truth = _to_crs(utility_truth, aois.crs)
    building_points = _to_crs(building_points, aois.crs)
    if roads is None or utility_truth is None:
        raise ValueError("roads and utility_truth are required.")

    records: list[dict] = []
    for row in aois.itertuples():
        geom = row.geometry
        road_length_m = _length_within(roads, geom)
        truth_length_m = _length_within(utility_truth, geom)
        building_point_count = _point_count_within(building_points, geom)
        records.append(
            {
                "aoi_id": row.aoi_id,
                "grid_row": int(row.grid_row),
                "grid_col": int(row.grid_col),
                "road_length_m": road_length_m,
                "truth_length_m": truth_length_m,
                "building_point_count": int(building_point_count),
                "score": float(np.log1p(road_length_m) + np.log1p(truth_length_m)),
                "geometry": geom,
            }
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=aois.crs)


def select_viable_aois(
    aoi_summary: gpd.GeoDataFrame,
    *,
    thresholds: AOIThresholds | None = None,
    max_aois: int | None = None,
    min_gap_m: float = 0.0,
) -> gpd.GeoDataFrame:
    """Select high-content AOIs while enforcing a minimum separation distance."""

    thresholds = thresholds or AOIThresholds()
    table = aoi_summary[
        (aoi_summary["road_length_m"] >= float(thresholds.min_road_length_m))
        & (aoi_summary["truth_length_m"] >= float(thresholds.min_truth_length_m))
        & (aoi_summary["building_point_count"] >= int(thresholds.min_building_points))
    ].copy()
    table = table.sort_values(
        ["score", "truth_length_m", "road_length_m"],
        ascending=[False, False, False],
    )

    selected_rows = []
    selected_geoms = []
    gap = max(float(min_gap_m), 0.0)
    for _, row in table.iterrows():
        geom = row.geometry
        if all(float(geom.distance(other)) >= gap for other in selected_geoms):
            selected_rows.append(row)
            selected_geoms.append(geom)
            if max_aois is not None and len(selected_rows) >= int(max_aois):
                break

    if not selected_rows:
        return gpd.GeoDataFrame(columns=list(aoi_summary.columns), geometry="geometry", crs=aoi_summary.crs)
    return gpd.GeoDataFrame(selected_rows, geometry="geometry", crs=aoi_summary.crs).reset_index(
        drop=True
    )


def assign_aoi_splits(
    selected_aois: gpd.GeoDataFrame,
    *,
    seed: int = 42,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
) -> gpd.GeoDataFrame:
    """Assign train/val/test labels at AOI level, not edge level."""

    out = selected_aois.copy()
    n = len(out)
    if n == 0:
        out["split"] = []
        return out
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n)
    n_train = int(np.floor(n * float(train_fraction)))
    n_val = int(np.floor(n * float(val_fraction)))
    if n >= 3:
        n_train = max(1, min(n_train, n - 2))
        n_val = max(1, min(n_val, n - n_train - 1))
    elif n == 2:
        n_train = 1
        n_val = 0
    else:
        n_train = 1
        n_val = 0

    splits = np.full(n, "test", dtype=object)
    splits[order[:n_train]] = "train"
    splits[order[n_train : n_train + n_val]] = "val"
    out["split"] = splits
    return out.sort_values(["split", "aoi_id"]).reset_index(drop=True)


def clip_vector_to_aoi(
    gdf: gpd.GeoDataFrame,
    aoi: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Clip a vector layer to a single AOI polygon."""

    if len(aoi) != 1:
        raise ValueError("clip_vector_to_aoi expects exactly one AOI row.")
    layer = _to_crs(gdf, aoi.crs)
    if layer is None or layer.empty:
        return gpd.GeoDataFrame(columns=list(gdf.columns), geometry="geometry", crs=aoi.crs)
    geom = aoi.geometry.iloc[0]
    subset = _intersecting_subset(layer, geom)
    if subset.empty:
        return gpd.GeoDataFrame(columns=list(layer.columns), geometry=layer.geometry.name, crs=aoi.crs)
    return gpd.clip(subset, geom)

