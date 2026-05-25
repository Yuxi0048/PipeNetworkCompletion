"""Plot shortest OSM-building-footprint projections to known sewer mains.

Workstream: Codex

This diagnostic assumes gravity mains are known and draws a candidate service
lateral from the nearest point on each OSM building footprint to the nearest
known gravity main. This is a service-lateral completion setting, not an
anchor-free full-network prediction setting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, box
from shapely.ops import nearest_points, unary_union


def _read_layer(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        return gpd.GeoDataFrame(geometry=[])
    return gpd.read_file(path)


def _clip_layer(gdf: gpd.GeoDataFrame, window, target_crs=None) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    if target_crs is not None and gdf.crs is not None and str(gdf.crs) != str(target_crs):
        gdf = gdf.to_crs(target_crs)
    clipped = gdf[gdf.intersects(window)].copy()
    if clipped.empty:
        return clipped
    clipped["geometry"] = clipped.geometry.intersection(window)
    clipped = clipped[~clipped.geometry.is_empty & clipped.geometry.notna()].copy()
    return clipped


def _filter_to_window(gdf: gpd.GeoDataFrame, window, target_crs=None, buffer_m: float = 0.0) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    if target_crs is not None and gdf.crs is not None and str(gdf.crs) != str(target_crs):
        gdf = gdf.to_crs(target_crs)
    search_window = window.buffer(buffer_m) if buffer_m else window
    filtered = gdf[gdf.intersects(search_window)].copy()
    filtered = filtered[~filtered.geometry.is_empty & filtered.geometry.notna()].copy()
    return filtered


def _window_from_summary(summary_path: Path):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return box(*summary["window_bounds"]), summary


def _footprint_to_main_projections(
    footprints: gpd.GeoDataFrame,
    mains: gpd.GeoDataFrame,
    max_length_m: float,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if footprints.empty or mains.empty:
        empty = gpd.GeoDataFrame(geometry=[], crs=footprints.crs or mains.crs)
        return empty, empty, empty

    main_union = unary_union(list(mains.geometry))
    rows = []
    building_points = []
    main_points = []
    for footprint_index, footprint in enumerate(footprints.geometry):
        building_point, main_point = nearest_points(footprint, main_union)
        line = LineString([building_point, main_point])
        length_m = float(line.length)
        if length_m > max_length_m:
            continue
        rows.append(
            {
                "footprint_index": int(footprint_index),
                "projection_length_m": length_m,
                "geometry": line,
            }
        )
        building_points.append(building_point)
        main_points.append(main_point)

    crs = footprints.crs or mains.crs
    projections = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    building_gdf = gpd.GeoDataFrame(geometry=building_points, crs=crs)
    main_gdf = gpd.GeoDataFrame(geometry=main_points, crs=crs)
    return projections, building_gdf, main_gdf


def _plot(
    truth_laterals: gpd.GeoDataFrame,
    mains: gpd.GeoDataFrame,
    projections: gpd.GeoDataFrame,
    footprint_points: gpd.GeoDataFrame,
    main_points: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    footprints_display: gpd.GeoDataFrame,
    output_png: Path,
    title: str,
    dpi: int,
) -> None:
    bounds_source = footprints_display if not footprints_display.empty else mains
    minx, miny, maxx, maxy = bounds_source.total_bounds
    pad_x = (maxx - minx) * 0.04
    pad_y = (maxy - miny) * 0.04

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    for ax in axes:
        if not footprints_display.empty:
            footprints_display.plot(
                ax=ax,
                facecolor="#d7f0e7",
                edgecolor="#72a89e",
                linewidth=0.35,
                alpha=0.56,
                zorder=1,
            )
        if not roads.empty:
            roads.plot(ax=ax, color="#c9c9c9", linewidth=0.9, alpha=0.9, zorder=2)
        if not mains.empty:
            mains.plot(ax=ax, color="#222222", linewidth=1.5, alpha=0.9, zorder=6)
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
        ax.set_aspect("equal")
        ax.set_axis_off()

    axes[0].set_title("Truth laterals and known gravity mains")
    axes[1].set_title("Footprint-nearest-point to main projections")

    if not truth_laterals.empty:
        truth_laterals.plot(ax=axes[0], color="#d7301f", linewidth=1.8, alpha=0.9, zorder=8)
        truth_laterals.plot(ax=axes[1], color="#d7301f", linewidth=1.4, alpha=0.35, zorder=7)
    if not projections.empty:
        projections.plot(ax=axes[1], color="#08519c", linewidth=1.3, alpha=0.82, zorder=9)
    if not footprint_points.empty:
        footprint_points.plot(
            ax=axes[1],
            color="#5e3c99",
            edgecolor="white",
            linewidth=0.25,
            markersize=15,
            alpha=0.95,
            zorder=10,
        )
    if not main_points.empty:
        main_points.plot(
            ax=axes[1],
            color="#fdae61",
            edgecolor="#7f3b08",
            linewidth=0.25,
            markersize=11,
            alpha=0.95,
            zorder=10,
        )

    legend_items = [
        plt.Line2D([0], [0], color="#222222", lw=2, label="known gravity main"),
        plt.Line2D([0], [0], color="#d7301f", lw=2, label="truth service lateral"),
        plt.Line2D([0], [0], color="#08519c", lw=2, label="footprint-to-main projection"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#5e3c99", markersize=7, label="nearest footprint point"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#fdae61", markersize=7, label="nearest main point"),
        plt.Line2D([0], [0], color="#c9c9c9", lw=2, label="road"),
    ]
    fig.suptitle(title)
    fig.legend(handles=legend_items, loc="lower center", ncol=6, frameon=False)
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
    parser.add_argument(
        "--service-output-root",
        type=Path,
        default=Path("outputs/service_lateral_osm_centroid_connections"),
    )
    parser.add_argument(
        "--osm-footprints",
        type=Path,
        default=Path("data/raw/context/buildings/osm_buildings_selected_aois.geojson"),
    )
    parser.add_argument("--window-size-m", type=float, default=250.0)
    parser.add_argument("--max-projection-length-m", type=float, default=80.0)
    parser.add_argument("--dpi", type=int, default=260)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.service_output_root / args.aoi_id
    zoom_dir = source_dir / f"zoom_{int(args.window_size_m)}m"
    summary_path = zoom_dir / "summary_zoom.json"
    if summary_path.exists():
        window, window_summary = _window_from_summary(summary_path)
    else:
        aoi = _read_layer(args.aoi_root / args.aoi_id / "aoi.geojson")
        if aoi.empty:
            raise FileNotFoundError(f"AOI file not found for {args.aoi_id}")
        minx, miny, maxx, maxy = aoi.total_bounds
        cx = (minx + maxx) / 2.0
        cy = (miny + maxy) / 2.0
        half = args.window_size_m / 2.0
        window = box(cx - half, cy - half, cx + half, cy + half)
        window_summary = {"window_bounds": list(window.bounds)}

    mains = _read_layer(args.aoi_root / args.aoi_id / "utility_truth_gravity_mains.geojson")
    target_crs = mains.crs
    roads = _clip_layer(_read_layer(args.aoi_root / args.aoi_id / "roads.geojson"), window, target_crs)
    footprints_full = _filter_to_window(_read_layer(args.osm_footprints), window, target_crs)
    footprints_display = _clip_layer(footprints_full, window, target_crs)
    truth_laterals = _clip_layer(_read_layer(source_dir / "truth_service_laterals.geojson"), window, target_crs)
    mains_zoom = _clip_layer(mains, window.buffer(args.max_projection_length_m), target_crs)

    projections, footprint_points, main_points = _footprint_to_main_projections(
        footprints=footprints_full,
        mains=mains_zoom,
        max_length_m=args.max_projection_length_m,
    )
    projections = _clip_layer(projections, window, target_crs)
    footprint_points = _clip_layer(footprint_points, window, target_crs)
    main_points = _clip_layer(main_points, window, target_crs)
    mains_display = _clip_layer(mains, window, target_crs)

    output_dir = zoom_dir / "footprint_to_main"
    output_dir.mkdir(parents=True, exist_ok=True)
    projection_path = output_dir / "osm_footprint_to_main_projections.geojson"
    png_path = output_dir / "osm_footprint_to_main_projections_vs_truth.png"
    summary_path_out = output_dir / "summary_footprint_to_main.json"
    projections.to_file(projection_path, driver="GeoJSON")

    summary = {
        "aoi_id": args.aoi_id,
        "window_size_m": args.window_size_m,
        "window_bounds": window_summary["window_bounds"],
        "max_projection_length_m": args.max_projection_length_m,
        "truth_lateral_count_in_window": int(len(truth_laterals)),
        "gravity_main_count_in_window": int(len(mains_display)),
        "osm_footprint_count_in_window": int(len(footprints_display)),
        "projection_count": int(len(projections)),
    }
    if not projections.empty:
        summary.update(
            {
                "median_projection_length_m": float(projections["projection_length_m"].median()),
                "mean_projection_length_m": float(projections["projection_length_m"].mean()),
                "p90_projection_length_m": float(projections["projection_length_m"].quantile(0.9)),
            }
        )
    summary_path_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    title = (
        f"{args.aoi_id} {args.window_size_m:.0f} m zoom: "
        f"{len(projections):,} footprint-to-main projections"
    )
    _plot(
        truth_laterals=truth_laterals,
        mains=mains_display,
        projections=projections,
        footprint_points=footprint_points,
        main_points=main_points,
        roads=roads,
        footprints_display=footprints_display,
        output_png=png_path,
        title=title,
        dpi=args.dpi,
    )

    print(json.dumps(summary, indent=2))
    print(f"Wrote plot: {png_path}")
    print(f"Wrote projections: {projection_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
