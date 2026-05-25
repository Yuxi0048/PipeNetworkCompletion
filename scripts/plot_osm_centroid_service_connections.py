"""Plot OSM-building-centroid service connections against sewer laterals.

Workstream: Codex

This is a diagnostic plot, not a training script. It tests the assumption that
service laterals can be approximated as connections from visible sewer assets to
OSM building centroids.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, Point, box, shape
from shapely.ops import transform as shapely_transform


VISIBLE_UTILITY_ANCHORS = [
    "SewerManholes_ExportFeatures.shp",
    "SewersqlSewerP_ExportFeature.shp",
    "SewerPumpStati_ExportFeature.shp",
    "SewerVent_ExportFeatures.shp",
    "SewerControlVa_ExportFeature.shp",
    "SewerSystemVal_ExportFeature.shp",
    "SewerDevice_ExportFeatures.shp",
    "UUSewertreatme_ExportFeature.shp",
]


def _source_crs(src: fiona.Collection) -> CRS:
    return CRS.from_user_input(src.crs_wkt or src.crs)


def _transform_geometry(geometry, source_crs: CRS, target_crs: CRS):
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

    def project(x, y, z=None):
        return transformer.transform(x, y)

    return shapely_transform(project, geometry)


def _source_bbox(target_bounds, target_crs: CRS, source_crs: CRS, buffer_m: float) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = target_bounds
    buffered = box(minx, miny, maxx, maxy).buffer(buffer_m)
    minx, miny, maxx, maxy = buffered.bounds
    transformer = Transformer.from_crs(target_crs, source_crs, always_xy=True)
    corners = [
        transformer.transform(minx, miny),
        transformer.transform(minx, maxy),
        transformer.transform(maxx, miny),
        transformer.transform(maxx, maxy),
    ]
    xs = [xy[0] for xy in corners]
    ys = [xy[1] for xy in corners]
    return min(xs), min(ys), max(xs), max(ys)


def _line_endpoints(geometry) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if isinstance(geometry, LineString):
        coords = list(geometry.coords)
        if len(coords) < 2:
            return None
        return tuple(coords[0][:2]), tuple(coords[-1][:2])
    if isinstance(geometry, MultiLineString):
        parts = [part for part in geometry.geoms if len(part.coords) >= 2]
        if not parts:
            return None
        return tuple(parts[0].coords[0][:2]), tuple(parts[-1].coords[-1][:2])
    return None


def _read_aoi(aoi_root: Path, aoi_id: str, target_crs: str) -> gpd.GeoDataFrame:
    path = aoi_root / aoi_id / "aoi.geojson"
    if not path.exists():
        raise FileNotFoundError(f"AOI not found: {path}")
    return gpd.read_file(path).to_crs(target_crs)


def _read_context_layer(path: Path, target_crs: str, clip_geometry, buffer_m: float = 0.0) -> gpd.GeoDataFrame:
    if not path.exists():
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"Layer has no CRS: {path}")
    gdf = gdf.to_crs(target_crs)
    clip = clip_geometry.buffer(buffer_m) if buffer_m else clip_geometry
    gdf = gdf[gdf.intersects(clip)].copy()
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    return gdf


def _read_visible_anchor_points(
    sewer_root: Path,
    target_crs: CRS,
    clip_geometry,
    buffer_m: float,
) -> gpd.GeoDataFrame:
    points: list[Point] = []
    sources: list[str] = []
    for filename in VISIBLE_UTILITY_ANCHORS:
        path = sewer_root / filename
        if not path.exists():
            continue
        with fiona.open(path) as src:
            source_crs = _source_crs(src)
            bbox_src = _source_bbox(clip_geometry.bounds, target_crs, source_crs, buffer_m)
            for feature in src.filter(bbox=bbox_src):
                geometry = feature.get("geometry")
                if not geometry or geometry.get("type") != "Point":
                    continue
                point = _transform_geometry(shape(geometry), source_crs, target_crs)
                if point.intersects(clip_geometry.buffer(buffer_m)):
                    points.append(point)
                    sources.append(filename)
    return gpd.GeoDataFrame({"source_layer": sources}, geometry=points, crs=target_crs)


def _read_service_laterals(
    sewer_root: Path,
    target_crs: CRS,
    clip_geometry,
    buffer_m: float,
) -> gpd.GeoDataFrame:
    path = sewer_root / "SewerService_ExportFeatures.shp"
    rows = []
    with fiona.open(path) as src:
        source_crs = _source_crs(src)
        bbox_src = _source_bbox(clip_geometry.bounds, target_crs, source_crs, buffer_m)
        for index, feature in enumerate(src.filter(bbox=bbox_src)):
            geometry = feature.get("geometry")
            if not geometry:
                continue
            line = _transform_geometry(shape(geometry), source_crs, target_crs)
            if line.is_empty or not line.intersects(clip_geometry):
                continue
            endpoints = _line_endpoints(line)
            if endpoints is None:
                continue
            start_xy, end_xy = endpoints
            rows.append(
                {
                    "service_index": index,
                    "start_x": start_xy[0],
                    "start_y": start_xy[1],
                    "end_x": end_xy[0],
                    "end_y": end_xy[1],
                    "truth_length_m": float(line.length),
                    "geometry": line,
                }
            )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=target_crs)


def _coords(gdf: gpd.GeoDataFrame) -> np.ndarray:
    return np.asarray([(geom.x, geom.y) for geom in gdf.geometry], dtype=float)


def _build_centroid_connections(
    laterals: gpd.GeoDataFrame,
    utility_points: gpd.GeoDataFrame,
    building_centroids: gpd.GeoDataFrame,
    utility_tolerance_m: float,
    building_tolerance_m: float,
) -> gpd.GeoDataFrame:
    if laterals.empty or utility_points.empty or building_centroids.empty:
        return gpd.GeoDataFrame(geometry=[], crs=laterals.crs)

    utility_coords = _coords(utility_points)
    building_coords = _coords(building_centroids)
    utility_tree = cKDTree(utility_coords)
    building_tree = cKDTree(building_coords)

    start_coords = laterals[["start_x", "start_y"]].to_numpy(dtype=float)
    end_coords = laterals[["end_x", "end_y"]].to_numpy(dtype=float)

    u_start_dist, u_start_idx = utility_tree.query(start_coords, k=1)
    u_end_dist, u_end_idx = utility_tree.query(end_coords, k=1)
    b_start_dist, b_start_idx = building_tree.query(start_coords, k=1)
    b_end_dist, b_end_idx = building_tree.query(end_coords, k=1)

    rows = []
    for row_position, lateral in enumerate(laterals.itertuples(index=False)):
        option_start_utility = u_start_dist[row_position] + b_end_dist[row_position]
        option_end_utility = u_end_dist[row_position] + b_start_dist[row_position]
        if option_start_utility <= option_end_utility:
            utility_distance = float(u_start_dist[row_position])
            building_distance = float(b_end_dist[row_position])
            utility_index = int(u_start_idx[row_position])
            building_index = int(b_end_idx[row_position])
        else:
            utility_distance = float(u_end_dist[row_position])
            building_distance = float(b_start_dist[row_position])
            utility_index = int(u_end_idx[row_position])
            building_index = int(b_start_idx[row_position])

        supported = (
            utility_distance <= utility_tolerance_m
            and building_distance <= building_tolerance_m
        )
        if not supported:
            continue

        utility_xy = tuple(utility_coords[utility_index])
        building_xy = tuple(building_coords[building_index])
        line = LineString([utility_xy, building_xy])
        truth_geometry = lateral.geometry
        rows.append(
            {
                "service_index": int(lateral.service_index),
                "utility_anchor_index": utility_index,
                "building_centroid_index": building_index,
                "utility_snap_distance_m": utility_distance,
                "building_snap_distance_m": building_distance,
                "total_snap_distance_m": utility_distance + building_distance,
                "truth_length_m": float(lateral.truth_length_m),
                "inferred_length_m": float(line.length),
                "length_error_m": float(line.length - lateral.truth_length_m),
                "truth_to_connection_distance_m": float(truth_geometry.distance(line)),
                "hausdorff_distance_m": float(truth_geometry.hausdorff_distance(line)),
                "geometry": line,
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=laterals.crs)


def _plot_panel_base(
    ax,
    aoi: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    footprints: gpd.GeoDataFrame,
    centroids: gpd.GeoDataFrame,
    utility_points: gpd.GeoDataFrame,
) -> None:
    if not footprints.empty:
        footprints.plot(
            ax=ax,
            facecolor="#d7f0e7",
            edgecolor="#68a99b",
            linewidth=0.18,
            alpha=0.45,
            zorder=1,
        )
    if not roads.empty:
        roads.plot(ax=ax, color="#cfcfcf", linewidth=0.55, alpha=0.75, zorder=2)
    if not centroids.empty:
        centroids.plot(
            ax=ax,
            color="#5e3c99",
            markersize=5,
            alpha=0.78,
            zorder=5,
        )
    if not utility_points.empty:
        utility_points.plot(
            ax=ax,
            color="#1f78b4",
            markersize=7,
            alpha=0.8,
            zorder=6,
        )
    aoi.boundary.plot(ax=ax, color="#252525", linewidth=1.0, zorder=10)


def _plot_outputs(
    aoi: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    footprints: gpd.GeoDataFrame,
    centroids: gpd.GeoDataFrame,
    utility_points: gpd.GeoDataFrame,
    laterals: gpd.GeoDataFrame,
    connections: gpd.GeoDataFrame,
    output_png: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    titles = [
        f"Ground-truth service laterals ({len(laterals):,})",
        f"OSM centroid surrogate links ({len(connections):,})",
        "Overlay comparison",
    ]
    minx, miny, maxx, maxy = aoi.total_bounds
    dx = (maxx - minx) * 0.03
    dy = (maxy - miny) * 0.03

    for ax, title in zip(axes, titles):
        _plot_panel_base(ax, aoi, roads, footprints, centroids, utility_points)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(minx - dx, maxx + dx)
        ax.set_ylim(miny - dy, maxy + dy)
        ax.set_axis_off()

    if not laterals.empty:
        laterals.plot(ax=axes[0], color="#d7301f", linewidth=0.9, alpha=0.82, zorder=8)
        laterals.plot(ax=axes[2], color="#d7301f", linewidth=1.0, alpha=0.58, zorder=8)
    if not connections.empty:
        connections.plot(ax=axes[1], color="#08519c", linewidth=0.9, alpha=0.76, zorder=8)
        connections.plot(ax=axes[2], color="#08519c", linewidth=0.8, alpha=0.68, zorder=9)

    legend_items = [
        plt.Line2D([0], [0], color="#d7301f", lw=2, label="truth service lateral"),
        plt.Line2D([0], [0], color="#08519c", lw=2, label="centroid surrogate link"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#5e3c99", markersize=6, label="OSM building centroid"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f78b4", markersize=6, label="visible sewer asset"),
        plt.Line2D([0], [0], color="#cfcfcf", lw=2, label="road"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=5, frameon=False)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi-id", default="small_aoi_10_12")
    parser.add_argument(
        "--aoi-root",
        type=Path,
        default=Path("data/processed/aois/anchor_free_2km_gap500_115"),
    )
    parser.add_argument("--sewer-root", type=Path, default=Path("data/raw/gis/sewer"))
    parser.add_argument(
        "--osm-centroids",
        type=Path,
        default=Path("data/processed/context/study_area/osm_building_anchor_points.geojson"),
    )
    parser.add_argument(
        "--osm-footprints",
        type=Path,
        default=Path("data/raw/context/buildings/osm_buildings_selected_aois.geojson"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/service_lateral_osm_centroid_connections"),
    )
    parser.add_argument("--target-crs", default="EPSG:28356")
    parser.add_argument("--utility-tolerance-m", type=float, default=20.0)
    parser.add_argument("--building-tolerance-m", type=float, default=20.0)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_crs = CRS.from_user_input(args.target_crs)
    output_dir = args.output_dir / args.aoi_id
    output_dir.mkdir(parents=True, exist_ok=True)

    aoi = _read_aoi(args.aoi_root, args.aoi_id, args.target_crs)
    aoi_union = aoi.geometry.unary_union
    search_buffer_m = max(args.utility_tolerance_m, args.building_tolerance_m) + 50.0

    roads = _read_context_layer(args.aoi_root / args.aoi_id / "roads.geojson", args.target_crs, aoi_union)
    footprints = _read_context_layer(args.osm_footprints, args.target_crs, aoi_union, buffer_m=0.0)
    centroids = _read_context_layer(
        args.osm_centroids,
        args.target_crs,
        aoi_union,
        buffer_m=args.building_tolerance_m,
    )
    utility_points = _read_visible_anchor_points(args.sewer_root, target_crs, aoi_union, search_buffer_m)
    laterals = _read_service_laterals(args.sewer_root, target_crs, aoi_union, search_buffer_m)
    connections = _build_centroid_connections(
        laterals,
        utility_points,
        centroids,
        args.utility_tolerance_m,
        args.building_tolerance_m,
    )

    truth_path = output_dir / "truth_service_laterals.geojson"
    connection_path = output_dir / "osm_centroid_service_connections.geojson"
    summary_path = output_dir / "summary.json"
    png_path = output_dir / "osm_centroid_connections_vs_truth.png"

    laterals.to_file(truth_path, driver="GeoJSON")
    connections.to_file(connection_path, driver="GeoJSON")

    supported_fraction = float(len(connections) / len(laterals)) if len(laterals) else 0.0
    summary = {
        "aoi_id": args.aoi_id,
        "target_crs": args.target_crs,
        "utility_tolerance_m": args.utility_tolerance_m,
        "building_tolerance_m": args.building_tolerance_m,
        "service_lateral_count": int(len(laterals)),
        "visible_utility_anchor_count": int(len(utility_points)),
        "osm_building_centroid_count": int(len(centroids)),
        "supported_connection_count": int(len(connections)),
        "supported_fraction": supported_fraction,
    }
    if not connections.empty:
        summary.update(
            {
                "median_total_snap_distance_m": float(connections["total_snap_distance_m"].median()),
                "median_hausdorff_distance_m": float(connections["hausdorff_distance_m"].median()),
                "median_length_error_m": float(connections["length_error_m"].median()),
            }
        )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _plot_outputs(
        aoi=aoi,
        roads=roads,
        footprints=footprints,
        centroids=centroids,
        utility_points=utility_points,
        laterals=laterals,
        connections=connections,
        output_png=png_path,
        dpi=args.dpi,
    )

    print(json.dumps(summary, indent=2))
    print(f"Wrote plot: {png_path}")
    print(f"Wrote connections: {connection_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
