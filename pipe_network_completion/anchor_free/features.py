"""Non-anchor feature generation for road-edge utility candidates."""

# Workstream: Codex + Claude merge

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from pipe_network_completion.anchor_free.road_graph import RoadCandidateGraph


FORBIDDEN_ANCHOR_TOKENS = (
    "manhole",
    "valve",
    "pole",
    "transformer",
    "cabinet",
    "anchor",
    "utility_node",
    "facility_node",
    "surveyed_node",
    "known_pipe",
    "utility_junction",
    "junction_point",
)

FORBIDDEN_ANCHOR_ABBREVIATIONS = {"mh"}

DEFAULT_ROAD_CLASS_COLUMNS = (
    "road_class",
    "ROAD_CLASS",
    "OVL2_CAT",
    "CLASS",
    "class",
    "type",
    "highway",
)

DEFAULT_BUILDING_NUMERIC_COLUMNS = (
    "dimension_m2",
    "ground_elevation",
    "eave_height",
    "maximum_roof_height",
)

DEFAULT_BUILT_UP_CATEGORY_COLUMNS = ("function", "feature_type")
DEFAULT_BUILDING_POINT_CATEGORY_COLUMNS = ("function", "feature_type")


@dataclass(frozen=True)
class RoadEdgeFeatureTable:
    """Feature matrix aligned one-to-one with ``RoadCandidateGraph.edges``."""

    edge_ids: np.ndarray
    features: pd.DataFrame

    @property
    def feature_names(self) -> list[str]:
        return list(self.features.columns)

    def to_numpy(self) -> np.ndarray:
        return self.features.to_numpy(dtype=float)


def _normalize_feature_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def assert_no_anchor_features(feature_names: Iterable[str]) -> None:
    """Fail if feature names suggest direct or proxy utility-anchor leakage."""

    offending: list[str] = []
    for name in feature_names:
        normalized = _normalize_feature_name(name)
        tokens = set(normalized.split("_"))
        if any(token in normalized for token in FORBIDDEN_ANCHOR_TOKENS):
            offending.append(str(name))
            continue
        if tokens.intersection(FORBIDDEN_ANCHOR_ABBREVIATIONS):
            offending.append(str(name))

    if offending:
        raise ValueError(
            "Anchor-free feature guard rejected suspicious feature names: "
            f"{sorted(set(offending))}"
        )


def _compatible_gdf(
    gdf: gpd.GeoDataFrame | None,
    graph: RoadCandidateGraph,
) -> gpd.GeoDataFrame | None:
    if gdf is None or gdf.empty:
        return None
    if graph.crs and gdf.crs and str(gdf.crs) != str(graph.crs):
        return gdf.to_crs(graph.crs)
    return gdf


def _sanitize_feature_token(value: object) -> str:
    normalized = _normalize_feature_name(str(value))
    return normalized if normalized else "unknown"


def _edge_local_road_density(
    edges: gpd.GeoDataFrame,
    buffer_m: float,
) -> np.ndarray:
    if edges.empty:
        return np.zeros(0, dtype=float)
    densities = []
    geometries = list(edges.geometry)
    lengths = np.asarray(edges["length_m"], dtype=float)
    spatial_index = edges.sindex
    for index, geom in enumerate(geometries):
        search_area = geom.buffer(buffer_m)
        area = max(float(search_area.area), 1e-12)
        total_length = 0.0
        try:
            candidate_indexes = spatial_index.query(search_area, predicate="intersects")
        except TypeError:
            candidate_indexes = [
                idx
                for idx in spatial_index.intersection(search_area.bounds)
                if search_area.intersects(geometries[idx])
            ]
        if len(candidate_indexes) == 0:
            candidate_indexes = [index]
        for candidate_index in candidate_indexes:
            total_length += float(lengths[int(candidate_index)])
        densities.append(total_length / area)
    return np.asarray(densities, dtype=float)


def _nearest_building_distance(
    edges: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    fallback_distance: float,
) -> np.ndarray:
    if edges.empty:
        return np.zeros(0, dtype=float)
    if buildings.empty:
        return np.full(len(edges), fallback_distance, dtype=float)
    building_geometries = [geom for geom in buildings.geometry if geom is not None]
    if not building_geometries:
        return np.full(len(edges), fallback_distance, dtype=float)
    building_series = gpd.GeoSeries(building_geometries, crs=buildings.crs)
    try:
        edge_points = gpd.GeoDataFrame(geometry=edges.geometry, crs=edges.crs)
        building_frame = gpd.GeoDataFrame(geometry=building_series, crs=buildings.crs)
        joined = edge_points.sjoin_nearest(
            building_frame,
            how="left",
            distance_col="distance_m",
        )
        distances = (
            joined.groupby(level=0)["distance_m"].min().reindex(edges.index).fillna(fallback_distance)
        )
        return distances.to_numpy(dtype=float)
    except Exception:
        spatial_index = building_series.sindex
        distances = []
        for edge_geom in edges.geometry:
            nearest = list(spatial_index.nearest(edge_geom.bounds, 1))
            if nearest:
                distances.append(float(edge_geom.distance(building_series.iloc[int(nearest[0])])))
            else:
                distances.append(float(fallback_distance))
        return np.asarray(distances, dtype=float)


def _building_count_within_buffer(
    edges: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    buffer_m: float,
) -> np.ndarray:
    if edges.empty:
        return np.zeros(0, dtype=float)
    if buildings.empty:
        return np.zeros(len(edges), dtype=float)
    building_geometries = [geom for geom in buildings.geometry if geom is not None]
    if not building_geometries:
        return np.zeros(len(edges), dtype=float)
    building_series = gpd.GeoSeries(building_geometries, crs=buildings.crs)
    spatial_index = building_series.sindex
    counts = []
    for edge_geom in edges.geometry:
        search_area = edge_geom.buffer(buffer_m)
        try:
            candidate_indexes = spatial_index.query(search_area, predicate="intersects")
        except TypeError:
            candidate_indexes = [
                idx
                for idx in spatial_index.intersection(search_area.bounds)
                if search_area.intersects(building_series.iloc[int(idx)])
            ]
        counts.append(len(candidate_indexes))
    return np.asarray(counts, dtype=float)


def _candidate_indexes(spatial_index, geometry) -> list[int]:
    try:
        indexes = spatial_index.query(geometry, predicate="intersects")
    except TypeError:
        indexes = list(spatial_index.intersection(geometry.bounds))
    return [int(idx) for idx in indexes]


def _context_nearest_distance(
    edges: gpd.GeoDataFrame,
    context: gpd.GeoDataFrame,
    fallback_distance: float,
) -> np.ndarray:
    if edges.empty:
        return np.zeros(0, dtype=float)
    if context.empty:
        return np.full(len(edges), fallback_distance, dtype=float)
    geometries = [geom for geom in context.geometry if geom is not None and not geom.is_empty]
    if not geometries:
        return np.full(len(edges), fallback_distance, dtype=float)
    context_series = gpd.GeoSeries(geometries, crs=context.crs)
    try:
        edge_frame = gpd.GeoDataFrame(geometry=edges.geometry, crs=edges.crs)
        context_frame = gpd.GeoDataFrame(geometry=context_series, crs=context.crs)
        joined = edge_frame.sjoin_nearest(
            context_frame,
            how="left",
            distance_col="distance_m",
        )
        distances = (
            joined.groupby(level=0)["distance_m"].min().reindex(edges.index).fillna(fallback_distance)
        )
        return distances.to_numpy(dtype=float)
    except Exception:
        pass
    spatial_index = context_series.sindex
    distances: list[float] = []
    for edge_geom in edges.geometry:
        nearest = list(spatial_index.nearest(edge_geom.bounds, 1))
        if nearest:
            distances.append(float(edge_geom.distance(context_series.iloc[int(nearest[0])])))
        else:
            distances.append(float(fallback_distance))
    return np.asarray(distances, dtype=float)


def _building_context_stats(
    edges: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    buffer_m: float,
    numeric_columns: Iterable[str] = DEFAULT_BUILDING_NUMERIC_COLUMNS,
) -> pd.DataFrame:
    suffix = f"{int(buffer_m)}m"
    index = edges["edge_id"].astype(int).to_numpy()
    columns = {
        f"building_count_{suffix}": np.zeros(len(edges), dtype=float),
        f"building_footprint_area_sum_{suffix}": np.zeros(len(edges), dtype=float),
        f"building_footprint_area_density_{suffix}": np.zeros(len(edges), dtype=float),
    }
    available_numeric = [
        column
        for column in numeric_columns
        if column in buildings.columns and pd.api.types.is_numeric_dtype(buildings[column])
    ]
    for column in available_numeric:
        token = _sanitize_feature_token(column)
        columns[f"building_{token}_mean_{suffix}"] = np.zeros(len(edges), dtype=float)
        columns[f"building_{token}_max_{suffix}"] = np.zeros(len(edges), dtype=float)

    stats = pd.DataFrame(columns, index=index)
    stats.index.name = "edge_id"
    if edges.empty or buildings.empty:
        return stats

    buildings = buildings[buildings.geometry.notna() & ~buildings.geometry.is_empty].copy()
    if buildings.empty:
        return stats

    geometries = list(buildings.geometry)
    spatial_index = buildings.sindex
    numeric_values = {
        column: pd.to_numeric(buildings[column], errors="coerce").to_numpy(dtype=float)
        for column in available_numeric
    }

    for edge_offset, edge_geom in enumerate(edges.geometry):
        search_area = edge_geom.buffer(buffer_m)
        area = max(float(search_area.area), 1e-12)
        candidate_indexes = _candidate_indexes(spatial_index, search_area)
        candidate_indexes = [
            idx for idx in candidate_indexes if search_area.intersects(geometries[idx])
        ]
        if not candidate_indexes:
            continue

        row_index = int(index[edge_offset])
        stats.at[row_index, f"building_count_{suffix}"] = float(len(candidate_indexes))
        footprint_area = 0.0
        for idx in candidate_indexes:
            geom = geometries[idx]
            if geom.geom_type in {"Polygon", "MultiPolygon"}:
                footprint_area += float(geom.intersection(search_area).area)
        stats.at[row_index, f"building_footprint_area_sum_{suffix}"] = footprint_area
        stats.at[row_index, f"building_footprint_area_density_{suffix}"] = footprint_area / area

        for column, values in numeric_values.items():
            selected = values[candidate_indexes]
            selected = selected[np.isfinite(selected)]
            if selected.size:
                token = _sanitize_feature_token(column)
                stats.at[row_index, f"building_{token}_mean_{suffix}"] = float(
                    np.mean(selected)
                )
                stats.at[row_index, f"building_{token}_max_{suffix}"] = float(
                    np.max(selected)
                )

    return stats


def _polygon_context_stats(
    edges: gpd.GeoDataFrame,
    context: gpd.GeoDataFrame,
    *,
    prefix: str,
    buffer_m: float,
    category_columns: Iterable[str] = DEFAULT_BUILT_UP_CATEGORY_COLUMNS,
    max_categories: int = 12,
) -> pd.DataFrame:
    suffix = f"{int(buffer_m)}m"
    index = edges["edge_id"].astype(int).to_numpy()
    stats = pd.DataFrame(
        {
            f"{prefix}_count_{suffix}": np.zeros(len(edges), dtype=float),
            f"{prefix}_area_sum_{suffix}": np.zeros(len(edges), dtype=float),
            f"{prefix}_area_coverage_{suffix}": np.zeros(len(edges), dtype=float),
            f"{prefix}_nearest_distance_m": np.zeros(len(edges), dtype=float),
        },
        index=index,
    )
    stats.index.name = "edge_id"
    if edges.empty or context.empty:
        return stats

    context = context[context.geometry.notna() & ~context.geometry.is_empty].copy()
    if context.empty:
        return stats

    fallback_distance = max(float(buffer_m) * 10.0, 1.0)
    stats[f"{prefix}_nearest_distance_m"] = _context_nearest_distance(
        edges,
        context,
        fallback_distance,
    )

    category_column = next((column for column in category_columns if column in context.columns), None)
    category_values: pd.Series | None = None
    category_tokens: list[str] = []
    if category_column is not None:
        category_values = context[category_column].fillna("unknown").astype(str)
        category_tokens = [
            _sanitize_feature_token(value)
            for value in category_values.value_counts().head(max_categories).index
        ]
        for token in category_tokens:
            stats[f"{prefix}_{category_column}_{token}_area_ratio_{suffix}"] = 0.0

    geometries = list(context.geometry)
    spatial_index = context.sindex
    for edge_offset, edge_geom in enumerate(edges.geometry):
        search_area = edge_geom.buffer(buffer_m)
        area = max(float(search_area.area), 1e-12)
        candidate_indexes = _candidate_indexes(spatial_index, search_area)
        candidate_indexes = [
            idx for idx in candidate_indexes if search_area.intersects(geometries[idx])
        ]
        if not candidate_indexes:
            continue

        row_index = int(index[edge_offset])
        total_area = 0.0
        category_area = {token: 0.0 for token in category_tokens}
        for idx in candidate_indexes:
            overlap_area = float(geometries[idx].intersection(search_area).area)
            total_area += overlap_area
            if category_values is not None:
                token = _sanitize_feature_token(category_values.iloc[idx])
                if token in category_area:
                    category_area[token] += overlap_area

        stats.at[row_index, f"{prefix}_count_{suffix}"] = float(len(candidate_indexes))
        stats.at[row_index, f"{prefix}_area_sum_{suffix}"] = total_area
        stats.at[row_index, f"{prefix}_area_coverage_{suffix}"] = min(total_area / area, 1.0)
        for token, value in category_area.items():
            stats.at[
                row_index,
                f"{prefix}_{category_column}_{token}_area_ratio_{suffix}",
            ] = min(value / area, 1.0)

    return stats


def _line_context_stats(
    edges: gpd.GeoDataFrame,
    context: gpd.GeoDataFrame,
    *,
    prefix: str,
    buffer_m: float,
) -> pd.DataFrame:
    suffix = f"{int(buffer_m)}m"
    index = edges["edge_id"].astype(int).to_numpy()
    stats = pd.DataFrame(
        {
            f"{prefix}_nearest_distance_m": np.zeros(len(edges), dtype=float),
            f"{prefix}_count_{suffix}": np.zeros(len(edges), dtype=float),
            f"{prefix}_length_sum_{suffix}": np.zeros(len(edges), dtype=float),
            f"{prefix}_length_density_{suffix}": np.zeros(len(edges), dtype=float),
        },
        index=index,
    )
    stats.index.name = "edge_id"
    if edges.empty or context.empty:
        return stats

    context = context[context.geometry.notna() & ~context.geometry.is_empty].copy()
    if context.empty:
        return stats

    fallback_distance = max(float(buffer_m) * 10.0, 1.0)
    stats[f"{prefix}_nearest_distance_m"] = _context_nearest_distance(
        edges,
        context,
        fallback_distance,
    )

    geometries = list(context.geometry)
    spatial_index = context.sindex
    for edge_offset, edge_geom in enumerate(edges.geometry):
        search_area = edge_geom.buffer(buffer_m)
        area = max(float(search_area.area), 1e-12)
        candidate_indexes = _candidate_indexes(spatial_index, search_area)
        candidate_indexes = [
            idx for idx in candidate_indexes if search_area.intersects(geometries[idx])
        ]
        if not candidate_indexes:
            continue

        row_index = int(index[edge_offset])
        total_length = 0.0
        for idx in candidate_indexes:
            clipped = geometries[idx].intersection(search_area)
            total_length += float(getattr(clipped, "length", 0.0))

        stats.at[row_index, f"{prefix}_count_{suffix}"] = float(len(candidate_indexes))
        stats.at[row_index, f"{prefix}_length_sum_{suffix}"] = total_length
        stats.at[row_index, f"{prefix}_length_density_{suffix}"] = total_length / area

    return stats


def _point_context_stats(
    edges: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
    *,
    buffer_m: float,
    category_columns: Iterable[str] = DEFAULT_BUILDING_POINT_CATEGORY_COLUMNS,
    max_categories: int = 12,
) -> pd.DataFrame:
    suffix = f"{int(buffer_m)}m"
    index = edges["edge_id"].astype(int).to_numpy()
    stats = pd.DataFrame(
        {
            "nearest_building_point_distance_m": np.zeros(len(edges), dtype=float),
            f"building_point_count_{suffix}": np.zeros(len(edges), dtype=float),
            f"building_point_density_{suffix}": np.zeros(len(edges), dtype=float),
        },
        index=index,
    )
    stats.index.name = "edge_id"
    if edges.empty or points.empty:
        return stats

    points = points[points.geometry.notna() & ~points.geometry.is_empty].copy()
    if points.empty:
        return stats

    fallback_distance = max(float(buffer_m) * 10.0, 1.0)
    stats["nearest_building_point_distance_m"] = _context_nearest_distance(
        edges,
        points,
        fallback_distance,
    )

    category_column = next((column for column in category_columns if column in points.columns), None)
    category_values: pd.Series | None = None
    category_tokens: list[str] = []
    if category_column is not None:
        category_values = points[category_column].fillna("unknown").astype(str)
        category_tokens = [
            _sanitize_feature_token(value)
            for value in category_values.value_counts().head(max_categories).index
        ]
        for token in category_tokens:
            stats[f"building_point_{category_column}_{token}_count_{suffix}"] = 0.0

    geometries = list(points.geometry)
    spatial_index = points.sindex
    for edge_offset, edge_geom in enumerate(edges.geometry):
        search_area = edge_geom.buffer(buffer_m)
        area = max(float(search_area.area), 1e-12)
        candidate_indexes = _candidate_indexes(spatial_index, search_area)
        candidate_indexes = [
            idx for idx in candidate_indexes if search_area.intersects(geometries[idx])
        ]
        if not candidate_indexes:
            continue

        row_index = int(index[edge_offset])
        stats.at[row_index, f"building_point_count_{suffix}"] = float(len(candidate_indexes))
        stats.at[row_index, f"building_point_density_{suffix}"] = float(
            len(candidate_indexes) / area
        )
        if category_values is not None:
            selected = category_values.iloc[candidate_indexes]
            counts = selected.map(_sanitize_feature_token).value_counts()
            for token in category_tokens:
                stats.at[
                    row_index,
                    f"building_point_{category_column}_{token}_count_{suffix}",
                ] = float(counts.get(token, 0.0))

    return stats


def _dem_crs_string(dataset) -> str | None:
    projection = dataset.GetProjectionRef()
    if not projection:
        return None
    try:
        from pyproj import CRS

        return CRS.from_wkt(projection).to_string()
    except Exception:
        return projection


def _points_for_dem_crs(
    points: list[Point],
    *,
    source_crs: str | None,
    dem_crs: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not points:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    series = gpd.GeoSeries(points, crs=source_crs)
    if source_crs and dem_crs and str(source_crs) != str(dem_crs):
        try:
            series = series.to_crs(dem_crs)
        except Exception:
            # Fall back to the original coordinates. The caller still gets
            # valid missing-value handling if they are outside raster bounds.
            pass
    return series.x.to_numpy(dtype=float), series.y.to_numpy(dtype=float)


def _sample_dem_points(
    dem_path: str | Path,
    points: list[Point],
    *,
    source_crs: str | None,
) -> np.ndarray:
    if not points:
        return np.zeros(0, dtype=float)
    try:
        from osgeo import gdal
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "DEM feature extraction requires the GDAL Python bindings (osgeo.gdal)."
        ) from exc

    dataset = gdal.Open(str(dem_path))
    if dataset is None:
        raise FileNotFoundError(f"Could not open DEM raster: {dem_path}")
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray().astype(float)
    nodata = band.GetNoDataValue()
    transform = dataset.GetGeoTransform()
    inverse = gdal.InvGeoTransform(transform)
    if inverse is None:
        raise ValueError(f"DEM raster has a non-invertible geotransform: {dem_path}")
    if isinstance(inverse, tuple) and len(inverse) == 2 and isinstance(inverse[1], tuple):
        inverse = inverse[1]

    xs, ys = _points_for_dem_crs(
        points,
        source_crs=source_crs,
        dem_crs=_dem_crs_string(dataset),
    )
    cols = np.floor(inverse[0] + inverse[1] * xs + inverse[2] * ys).astype(int)
    rows = np.floor(inverse[3] + inverse[4] * xs + inverse[5] * ys).astype(int)
    values = np.full(len(points), np.nan, dtype=float)
    inside = (
        (rows >= 0)
        & (rows < dataset.RasterYSize)
        & (cols >= 0)
        & (cols < dataset.RasterXSize)
    )
    if np.any(inside):
        sampled = array[rows[inside], cols[inside]].astype(float)
        if nodata is not None:
            sampled = np.where(sampled == nodata, np.nan, sampled)
        sampled = np.where(np.isfinite(sampled), sampled, np.nan)
        values[inside] = sampled
    return values


def _edge_dem_features(
    edges: gpd.GeoDataFrame,
    dem_path: str | Path,
    *,
    graph_crs: str | None,
    sample_spacing_m: float = 30.0,
    max_samples_per_edge: int = 64,
) -> pd.DataFrame:
    index = edges["edge_id"].astype(int).to_numpy()
    result = pd.DataFrame(index=index)
    result.index.name = "edge_id"
    if edges.empty:
        return result

    start_points = [Point(list(geom.coords)[0]) for geom in edges.geometry]
    end_points = [Point(list(geom.coords)[-1]) for geom in edges.geometry]
    mid_points = [geom.interpolate(0.5, normalized=True) for geom in edges.geometry]

    elevation_u = _sample_dem_points(dem_path, start_points, source_crs=graph_crs)
    elevation_v = _sample_dem_points(dem_path, end_points, source_crs=graph_crs)
    elevation_mid = _sample_dem_points(dem_path, mid_points, source_crs=graph_crs)

    profile_points: list[Point] = []
    profile_slices: list[tuple[int, int]] = []
    spacing = max(float(sample_spacing_m), 1.0)
    max_samples = max(int(max_samples_per_edge), 2)
    cursor = 0
    for geom in edges.geometry:
        length = max(float(geom.length), 0.0)
        sample_count = min(max(int(math.ceil(length / spacing)) + 1, 2), max_samples)
        distances = np.linspace(0.0, length, sample_count)
        for distance in distances:
            profile_points.append(geom.interpolate(float(distance)))
        profile_slices.append((cursor, cursor + sample_count))
        cursor += sample_count

    profile_values = _sample_dem_points(dem_path, profile_points, source_crs=graph_crs)
    mean_values = np.full(len(edges), np.nan, dtype=float)
    min_values = np.full(len(edges), np.nan, dtype=float)
    max_values = np.full(len(edges), np.nan, dtype=float)
    valid_fraction = np.zeros(len(edges), dtype=float)
    for edge_offset, (start, stop) in enumerate(profile_slices):
        values = profile_values[start:stop]
        valid = values[np.isfinite(values)]
        valid_fraction[edge_offset] = float(valid.size / max(len(values), 1))
        if valid.size:
            mean_values[edge_offset] = float(np.mean(valid))
            min_values[edge_offset] = float(np.min(valid))
            max_values[edge_offset] = float(np.max(valid))

    length = edges["length_m"].astype(float).to_numpy()
    safe_length = np.where(length > 0.0, length, np.nan)
    elevation_delta = elevation_v - elevation_u
    slope = elevation_delta / safe_length

    result["elevation_u_m"] = elevation_u
    result["elevation_v_m"] = elevation_v
    result["elevation_mid_m"] = elevation_mid
    result["elevation_mean_m"] = mean_values
    result["elevation_min_m"] = min_values
    result["elevation_max_m"] = max_values
    result["elevation_range_m"] = max_values - min_values
    result["elevation_delta_uv_m"] = elevation_delta
    result["slope_uv"] = slope
    result["abs_slope_uv"] = np.abs(slope)
    result["dem_valid_fraction"] = valid_fraction
    return result


def _road_class_columns(
    edges: gpd.GeoDataFrame,
    configured: str | Iterable[str] | None,
) -> list[str]:
    if configured is None:
        candidates = list(DEFAULT_ROAD_CLASS_COLUMNS)
    elif isinstance(configured, str):
        candidates = [configured]
    else:
        candidates = list(configured)
    return [column for column in candidates if column in edges.columns]


def build_road_edge_features(
    graph: RoadCandidateGraph,
    *,
    buildings_gdf: gpd.GeoDataFrame | None = None,
    building_points_gdf: gpd.GeoDataFrame | None = None,
    built_up_gdf: gpd.GeoDataFrame | None = None,
    watercourse_drainage_lines_gdf: gpd.GeoDataFrame | None = None,
    watercourse_corridor_centrelines_gdf: gpd.GeoDataFrame | None = None,
    watercourse_corridors_gdf: gpd.GeoDataFrame | None = None,
    dem_path: str | Path | None = None,
    road_class_columns: str | Iterable[str] | None = None,
    building_buffer_m: float = 50.0,
    building_point_buffer_m: float | None = None,
    built_up_buffer_m: float | None = None,
    watercourse_buffer_m: float = 100.0,
    road_density_buffer_m: float = 100.0,
    dem_sample_spacing_m: float = 30.0,
    dem_max_samples_per_edge: int = 64,
) -> RoadEdgeFeatureTable:
    """Compute road-edge features from non-anchor GIS context only.

    Ground utility anchors and truth-network-derived nodes are deliberately not
    accepted by this function. Utility truth lines belong in
    ``anchor_free.labels`` only.
    """

    edges = graph.edges.copy()
    nodes = graph.nodes.set_index("node_id") if not graph.nodes.empty else graph.nodes

    feature_df = pd.DataFrame(index=edges["edge_id"].astype(int).to_numpy())
    feature_df.index.name = "edge_id"

    length = edges["length_m"].astype(float).to_numpy()
    bearing = edges["bearing_rad"].astype(float).to_numpy()
    feature_df["length_m"] = length
    feature_df["bearing_sin"] = np.sin(bearing)
    feature_df["bearing_cos"] = np.cos(bearing)

    if len(nodes):
        degree_u = edges["u"].map(nodes["degree"]).fillna(0).astype(float).to_numpy()
        degree_v = edges["v"].map(nodes["degree"]).fillna(0).astype(float).to_numpy()
    else:
        degree_u = np.zeros(len(edges), dtype=float)
        degree_v = np.zeros(len(edges), dtype=float)
    feature_df["degree_u"] = degree_u
    feature_df["degree_v"] = degree_v
    feature_df["degree_sum"] = degree_u + degree_v

    for column in (
        "candidate_weight",
        "nearest_road_distance_m",
        "road_offset_distance_m",
    ):
        if column in edges.columns:
            feature_df[column] = pd.to_numeric(
                edges[column], errors="coerce"
            ).fillna(0.0).astype(float).to_numpy()

    density_buffer = float(road_density_buffer_m)
    feature_df[f"local_road_density_{int(density_buffer)}m"] = _edge_local_road_density(
        edges,
        density_buffer,
    )

    class_columns = _road_class_columns(edges, road_class_columns)
    for column in class_columns:
        encoded = pd.get_dummies(edges[column].fillna("unknown"), prefix=column)
        encoded.index = feature_df.index
        feature_df = pd.concat([feature_df, encoded.astype(float)], axis=1)

    buildings = _compatible_gdf(buildings_gdf, graph)
    if buildings is not None:
        fallback_distance = max(float(building_buffer_m) * 10.0, 1.0)
        feature_df["nearest_building_distance_m"] = _nearest_building_distance(
            edges,
            buildings,
            fallback_distance,
        )
        building_stats = _building_context_stats(edges, buildings, float(building_buffer_m))
        feature_df = pd.concat([feature_df, building_stats], axis=1)

    building_points = _compatible_gdf(building_points_gdf, graph)
    if building_points is not None:
        point_buffer = (
            float(building_point_buffer_m)
            if building_point_buffer_m is not None
            else float(building_buffer_m)
        )
        point_stats = _point_context_stats(
            edges,
            building_points,
            buffer_m=point_buffer,
        )
        feature_df = pd.concat([feature_df, point_stats], axis=1)

    built_up = _compatible_gdf(built_up_gdf, graph)
    if built_up is not None:
        context_buffer = (
            float(built_up_buffer_m)
            if built_up_buffer_m is not None
            else float(building_buffer_m)
        )
        built_up_stats = _polygon_context_stats(
            edges,
            built_up,
            prefix="built_up",
            buffer_m=context_buffer,
        )
        feature_df = pd.concat([feature_df, built_up_stats], axis=1)

    watercourse_buffer = float(watercourse_buffer_m)
    drainage_lines = _compatible_gdf(watercourse_drainage_lines_gdf, graph)
    if drainage_lines is not None:
        drainage_stats = _line_context_stats(
            edges,
            drainage_lines,
            prefix="watercourse_drainage",
            buffer_m=watercourse_buffer,
        )
        feature_df = pd.concat([feature_df, drainage_stats], axis=1)

    corridor_centrelines = _compatible_gdf(watercourse_corridor_centrelines_gdf, graph)
    if corridor_centrelines is not None:
        centreline_stats = _line_context_stats(
            edges,
            corridor_centrelines,
            prefix="watercourse_corridor_centreline",
            buffer_m=watercourse_buffer,
        )
        feature_df = pd.concat([feature_df, centreline_stats], axis=1)

    watercourse_corridors = _compatible_gdf(watercourse_corridors_gdf, graph)
    if watercourse_corridors is not None:
        corridor_stats = _polygon_context_stats(
            edges,
            watercourse_corridors,
            prefix="watercourse_corridor",
            buffer_m=watercourse_buffer,
            category_columns=("OVL2_CAT", "OVL2_DESC", "DESCRIPTION"),
        )
        feature_df = pd.concat([feature_df, corridor_stats], axis=1)

    if dem_path not in (None, ""):
        dem_features = _edge_dem_features(
            edges,
            dem_path,
            graph_crs=graph.crs,
            sample_spacing_m=float(dem_sample_spacing_m),
            max_samples_per_edge=int(dem_max_samples_per_edge),
        )
        feature_df = pd.concat([feature_df, dem_features], axis=1)

    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    assert_no_anchor_features(feature_df.columns)

    return RoadEdgeFeatureTable(
        edge_ids=edges["edge_id"].astype(int).to_numpy(),
        features=feature_df,
    )


def standardize_features(
    features: pd.DataFrame,
    train_index: np.ndarray | list[int] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return z-scored features plus the mean/std used for scaling."""

    if train_index is None:
        train_frame = features
    else:
        train_frame = features.iloc[list(train_index)]
    mean = train_frame.mean(axis=0)
    std = train_frame.std(axis=0)
    std = std.mask(std.abs() < 1e-12, 1.0).fillna(1.0)
    scaled = (features - mean) / std
    assert_no_anchor_features(scaled.columns)
    return scaled.fillna(0.0), mean, std


# ---------------------------------------------------------------------------
# Phase 2.A — feature builders for the heterogeneous road graph.
# Phase 2.B — clipped-length local-road density (fixes the long-arterial bug).
# Workstream: Claude
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RoadSegmentFeatureTable:
    """Per-RoadSegment-node feature matrix for the heterogeneous pipeline.

    Mirrors ``RoadEdgeFeatureTable`` but keys on ``segment_id`` rather than
    ``edge_id`` — the math is identical.
    """

    segment_ids: np.ndarray
    features: pd.DataFrame

    @property
    def feature_names(self) -> list[str]:
        return list(self.features.columns)

    def to_numpy(self) -> np.ndarray:
        return self.features.to_numpy(dtype=float)


@dataclass(frozen=True)
class IntersectionFeatureTable:
    """Per-Intersection-node feature matrix for the heterogeneous pipeline.

    Sparse by design: (x, y, degree) only. ``include_coords=False`` drops
    the (x, y) columns to test the location-memorisation shortcut (Stage 2
    of ``audit_followup_implementation_plan.md``).
    """

    intersection_ids: np.ndarray
    features: pd.DataFrame

    @property
    def feature_names(self) -> list[str]:
        return list(self.features.columns)

    def to_numpy(self) -> np.ndarray:
        return self.features.to_numpy(dtype=float)


def _segment_clipped_road_density(
    segments: gpd.GeoDataFrame,
    buffer_m: float,
) -> np.ndarray:
    """Phase 2.B fix — use clipped intersection length, not full segment length.

    For each segment we compute the search buffer, then for every neighbour
    segment we add only the length of the neighbour's geometry that
    actually falls inside the buffer. The legacy
    ``_edge_local_road_density`` added the neighbour's *whole* length —
    which made a long arterial that grazed a buffer contribute its
    full 2 km to the density numerator. This fixes that.
    """
    if segments.empty or buffer_m <= 0:
        return np.zeros(len(segments), dtype=float)
    geometries = list(segments.geometry)
    spatial_index = segments.sindex
    densities = []
    for index, geom in enumerate(geometries):
        search_area = geom.buffer(buffer_m)
        area = max(float(search_area.area), 1e-12)
        try:
            candidate_indexes = spatial_index.query(search_area, predicate="intersects")
        except TypeError:
            candidate_indexes = [
                idx
                for idx in spatial_index.intersection(search_area.bounds)
                if search_area.intersects(geometries[idx])
            ]
        total_length = 0.0
        for ci in candidate_indexes:
            other = geometries[int(ci)]
            try:
                clipped = other.intersection(search_area)
            except Exception:
                # Topology errors on pathological inputs: fall back to 0.
                continue
            if clipped.is_empty:
                continue
            # Length is well-defined for LineString/MultiLineString. For a
            # collection just sum the line components.
            if hasattr(clipped, "length"):
                total_length += float(clipped.length)
        densities.append(total_length / area)
    return np.asarray(densities, dtype=float)


def build_road_segment_features(
    graph,
    *,
    buildings_gdf: gpd.GeoDataFrame | None = None,
    building_points_gdf: gpd.GeoDataFrame | None = None,
    built_up_gdf: gpd.GeoDataFrame | None = None,
    watercourse_drainage_lines_gdf: gpd.GeoDataFrame | None = None,
    watercourse_corridor_centrelines_gdf: gpd.GeoDataFrame | None = None,
    watercourse_corridors_gdf: gpd.GeoDataFrame | None = None,
    dem_path: str | Path | None = None,
    road_class_columns: str | Iterable[str] | None = None,
    building_buffer_m: float = 50.0,
    building_point_buffer_m: float | None = None,
    built_up_buffer_m: float | None = None,
    watercourse_buffer_m: float = 100.0,
    road_density_buffer_m: float = 100.0,
    dem_sample_spacing_m: float = 30.0,
    dem_max_samples_per_edge: int = 64,
) -> RoadSegmentFeatureTable:
    """Compute per-RoadSegment features for the heterogeneous pipeline.

    Heterogeneous-graph version of ``build_road_edge_features`` — same
    feature math, applied to ``graph.road_segments`` (one row per
    LineString). Endpoint-intersection degrees are derived from
    ``graph.segment_touches_intersection`` so that segment-side context
    summaries still include the multi-road-junction signal.
    """
    segments = graph.road_segments.copy()
    if segments.empty:
        empty = pd.DataFrame()
        return RoadSegmentFeatureTable(segment_ids=np.zeros(0, dtype=int), features=empty)

    feature_df = pd.DataFrame(index=segments["segment_id"].astype(int).to_numpy())
    feature_df.index.name = "segment_id"

    length = segments["length_m"].astype(float).to_numpy()
    bearing = segments["bearing_rad"].astype(float).to_numpy()
    feature_df["length_m"] = length
    feature_df["bearing_sin"] = np.sin(bearing)
    feature_df["bearing_cos"] = np.cos(bearing)

    # Per-segment endpoint-degree summary derived from touches.
    touches = getattr(graph, "segment_touches_intersection", None)
    if touches is not None and touches.size:
        inter_deg = (
            graph.intersections.set_index("intersection_id")["degree"]
            if not graph.intersections.empty
            else pd.Series(dtype=float)
        )
        degrees_per_seg: dict[int, list[float]] = {}
        for seg_id, inter_id in zip(touches[0], touches[1]):
            degrees_per_seg.setdefault(int(seg_id), []).append(
                float(inter_deg.get(int(inter_id), 0))
            )
        endpoint_min = np.array(
            [
                min(degrees_per_seg.get(int(sid), [0.0]))
                for sid in feature_df.index
            ]
        )
        endpoint_max = np.array(
            [
                max(degrees_per_seg.get(int(sid), [0.0]))
                for sid in feature_df.index
            ]
        )
        endpoint_sum = np.array(
            [
                sum(degrees_per_seg.get(int(sid), [0.0]))
                for sid in feature_df.index
            ]
        )
    else:
        endpoint_min = np.zeros(len(feature_df), dtype=float)
        endpoint_max = np.zeros(len(feature_df), dtype=float)
        endpoint_sum = np.zeros(len(feature_df), dtype=float)
    feature_df["endpoint_degree_min"] = endpoint_min
    feature_df["endpoint_degree_max"] = endpoint_max
    feature_df["endpoint_degree_sum"] = endpoint_sum

    for column in (
        "candidate_weight",
        "nearest_road_distance_m",
        "road_offset_distance_m",
    ):
        if column in segments.columns:
            feature_df[column] = pd.to_numeric(
                segments[column], errors="coerce"
            ).fillna(0.0).astype(float).to_numpy()
    if "demand_u" in segments.columns:
        feature_df["candidate_has_demand_u"] = (
            pd.to_numeric(segments["demand_u"], errors="coerce").fillna(-1) >= 0
        ).astype(float).to_numpy()
    if "demand_v" in segments.columns:
        feature_df["candidate_has_demand_v"] = (
            pd.to_numeric(segments["demand_v"], errors="coerce").fillna(-1) >= 0
        ).astype(float).to_numpy()

    # Phase 2.B — clipped-length local-road density.
    density_buffer = float(road_density_buffer_m)
    feature_df[f"local_road_clipped_length_density_{int(density_buffer)}m"] = (
        _segment_clipped_road_density(segments, density_buffer)
    )

    class_columns = _road_class_columns(segments, road_class_columns)
    for column in class_columns:
        encoded = pd.get_dummies(segments[column].fillna("unknown"), prefix=column)
        encoded.index = feature_df.index
        feature_df = pd.concat([feature_df, encoded.astype(float)], axis=1)

    buildings = _compatible_gdf(buildings_gdf, graph)
    if buildings is not None:
        fallback_distance = max(float(building_buffer_m) * 10.0, 1.0)
        feature_df["nearest_building_distance_m"] = _nearest_building_distance(
            segments, buildings, fallback_distance
        )
        building_stats = _building_context_stats(
            segments.rename(columns={"segment_id": "edge_id"}),
            buildings,
            float(building_buffer_m),
        )
        building_stats.index.name = "segment_id"
        feature_df = pd.concat([feature_df, building_stats.reset_index(drop=True).set_index(feature_df.index)], axis=1)

    building_points = _compatible_gdf(building_points_gdf, graph)
    if building_points is not None:
        point_buffer = (
            float(building_point_buffer_m)
            if building_point_buffer_m is not None
            else float(building_buffer_m)
        )
        point_stats = _point_context_stats(
            segments.rename(columns={"segment_id": "edge_id"}),
            building_points,
            buffer_m=point_buffer,
        )
        point_stats.index.name = "segment_id"
        feature_df = pd.concat([feature_df, point_stats.reset_index(drop=True).set_index(feature_df.index)], axis=1)

    built_up = _compatible_gdf(built_up_gdf, graph)
    if built_up is not None:
        context_buffer = (
            float(built_up_buffer_m)
            if built_up_buffer_m is not None
            else float(building_buffer_m)
        )
        built_up_stats = _polygon_context_stats(
            segments.rename(columns={"segment_id": "edge_id"}),
            built_up,
            prefix="built_up",
            buffer_m=context_buffer,
        )
        built_up_stats.index.name = "segment_id"
        feature_df = pd.concat([feature_df, built_up_stats.reset_index(drop=True).set_index(feature_df.index)], axis=1)

    watercourse_buffer = float(watercourse_buffer_m)
    drainage_lines = _compatible_gdf(watercourse_drainage_lines_gdf, graph)
    if drainage_lines is not None:
        drainage_stats = _line_context_stats(
            segments.rename(columns={"segment_id": "edge_id"}),
            drainage_lines,
            prefix="watercourse_drainage",
            buffer_m=watercourse_buffer,
        )
        drainage_stats.index.name = "segment_id"
        feature_df = pd.concat(
            [feature_df, drainage_stats.reset_index(drop=True).set_index(feature_df.index)],
            axis=1,
        )

    corridor_centrelines = _compatible_gdf(watercourse_corridor_centrelines_gdf, graph)
    if corridor_centrelines is not None:
        centreline_stats = _line_context_stats(
            segments.rename(columns={"segment_id": "edge_id"}),
            corridor_centrelines,
            prefix="watercourse_corridor_centreline",
            buffer_m=watercourse_buffer,
        )
        centreline_stats.index.name = "segment_id"
        feature_df = pd.concat(
            [feature_df, centreline_stats.reset_index(drop=True).set_index(feature_df.index)],
            axis=1,
        )

    watercourse_corridors = _compatible_gdf(watercourse_corridors_gdf, graph)
    if watercourse_corridors is not None:
        corridor_stats = _polygon_context_stats(
            segments.rename(columns={"segment_id": "edge_id"}),
            watercourse_corridors,
            prefix="watercourse_corridor",
            buffer_m=watercourse_buffer,
            category_columns=("OVL2_CAT", "OVL2_DESC", "DESCRIPTION"),
        )
        corridor_stats.index.name = "segment_id"
        feature_df = pd.concat(
            [feature_df, corridor_stats.reset_index(drop=True).set_index(feature_df.index)],
            axis=1,
        )

    if dem_path not in (None, ""):
        dem_features = _edge_dem_features(
            segments.rename(columns={"segment_id": "edge_id"}),
            dem_path,
            graph_crs=graph.crs,
            sample_spacing_m=float(dem_sample_spacing_m),
            max_samples_per_edge=int(dem_max_samples_per_edge),
        )
        dem_features.index.name = "segment_id"
        feature_df = pd.concat([feature_df, dem_features.reset_index(drop=True).set_index(feature_df.index)], axis=1)

    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    assert_no_anchor_features(feature_df.columns)

    return RoadSegmentFeatureTable(
        segment_ids=segments["segment_id"].astype(int).to_numpy(),
        features=feature_df,
    )


def build_intersection_features(
    graph,
    *,
    include_coords: bool = True,
    coord_scale_m: float = 1000.0,
) -> IntersectionFeatureTable:
    """Sparse per-Intersection features: (x, y, degree).

    ``include_coords=False`` drops the (x, y) columns — the
    location-memorisation shortcut control from Stage 2 of the
    audit-followup plan moves here for the heterogeneous design (so the
    GNN's per-Intersection node features are the place where coordinate
    leakage would happen, and the toggle is co-located with the columns).
    """
    inter = graph.intersections.copy()
    if inter.empty:
        return IntersectionFeatureTable(
            intersection_ids=np.zeros(0, dtype=int),
            features=pd.DataFrame(),
        )
    feature_df = pd.DataFrame(index=inter["intersection_id"].astype(int).to_numpy())
    feature_df.index.name = "intersection_id"

    if include_coords:
        x = inter["x"].astype(float).to_numpy()
        y = inter["y"].astype(float).to_numpy()
        feature_df["intersection_x_norm"] = (x - x.mean()) / max(x.std(), 1.0)
        feature_df["intersection_y_norm"] = (y - y.mean()) / max(y.std(), 1.0)

    degree = inter["degree"].astype(float).to_numpy()
    feature_df["intersection_degree"] = degree
    feature_df["intersection_log_degree"] = np.log1p(np.clip(degree, 0, None))

    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    assert_no_anchor_features(feature_df.columns)
    return IntersectionFeatureTable(
        intersection_ids=inter["intersection_id"].astype(int).to_numpy(),
        features=feature_df,
    )


def bearing_degrees_from_sin_cos(sin_value: float, cos_value: float) -> float:
    """Utility for diagnostics and tests."""

    return math.degrees(math.atan2(float(sin_value), float(cos_value))) % 360.0
