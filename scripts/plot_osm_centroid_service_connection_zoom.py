"""Create a zoomed comparison of OSM-centroid service links and truth laterals.

Workstream: Codex

This script consumes the GeoJSON outputs from
plot_osm_centroid_service_connections.py and creates a smaller, readable AOI
window. It does not rebuild labels or train a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, box


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


def _connection_endpoints(connections: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    utility_points = []
    building_points = []
    for geometry in connections.geometry:
        if not isinstance(geometry, LineString) or len(geometry.coords) < 2:
            continue
        utility_points.append(Point(geometry.coords[0]))
        building_points.append(Point(geometry.coords[-1]))
    utility = gpd.GeoDataFrame(geometry=utility_points, crs=connections.crs)
    building = gpd.GeoDataFrame(geometry=building_points, crs=connections.crs)
    return utility, building


def _choose_dense_window(connections: gpd.GeoDataFrame, window_size_m: float):
    centroids = connections.geometry.centroid
    coords = [(point.x, point.y) for point in centroids]
    half = window_size_m / 2.0
    best_score = -1
    best_center = coords[0]
    for cx, cy in coords:
        candidate = box(cx - half, cy - half, cx + half, cy + half)
        score = int(connections.intersects(candidate).sum())
        if score > best_score:
            best_score = score
            best_center = (cx, cy)
    cx, cy = best_center
    return box(cx - half, cy - half, cx + half, cy + half), best_score


def _plot_zoom(
    truth: gpd.GeoDataFrame,
    connections: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    footprints: gpd.GeoDataFrame,
    output_png: Path,
    dpi: int,
    title: str,
) -> None:
    utility_points, building_points = _connection_endpoints(connections)
    bounds_source = truth if not truth.empty else connections
    minx, miny, maxx, maxy = bounds_source.total_bounds
    pad_x = (maxx - minx) * 0.04
    pad_y = (maxy - miny) * 0.04

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    for ax in axes:
        if not footprints.empty:
            footprints.plot(
                ax=ax,
                facecolor="#d7f0e7",
                edgecolor="#72a89e",
                linewidth=0.35,
                alpha=0.55,
                zorder=1,
            )
        if not roads.empty:
            roads.plot(ax=ax, color="#c9c9c9", linewidth=0.9, alpha=0.9, zorder=2)
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
        ax.set_aspect("equal")
        ax.set_axis_off()

    axes[0].set_title("Ground-truth service laterals")
    axes[1].set_title("Centroid surrogate links over truth")

    if not truth.empty:
        truth.plot(ax=axes[0], color="#d7301f", linewidth=1.7, alpha=0.88, zorder=5)
        truth.plot(ax=axes[1], color="#d7301f", linewidth=1.5, alpha=0.55, zorder=5)
    if not connections.empty:
        connections.plot(ax=axes[1], color="#08519c", linewidth=1.4, alpha=0.82, zorder=7)
    if not utility_points.empty:
        utility_points.plot(
            ax=axes[1],
            color="#1f78b4",
            edgecolor="white",
            linewidth=0.3,
            markersize=16,
            alpha=0.95,
            zorder=9,
        )
    if not building_points.empty:
        building_points.plot(
            ax=axes[1],
            color="#5e3c99",
            edgecolor="white",
            linewidth=0.3,
            markersize=16,
            alpha=0.95,
            zorder=10,
        )

    legend_items = [
        plt.Line2D([0], [0], color="#d7301f", lw=2, label="truth service lateral"),
        plt.Line2D([0], [0], color="#08519c", lw=2, label="centroid surrogate link"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#5e3c99", markersize=7, label="OSM centroid endpoint"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f78b4", markersize=7, label="visible sewer endpoint"),
        plt.Line2D([0], [0], color="#c9c9c9", lw=2, label="road"),
    ]
    fig.suptitle(title)
    fig.legend(handles=legend_items, loc="lower center", ncol=5, frameon=False)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi-id", default="small_aoi_10_12")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("outputs/service_lateral_osm_centroid_connections"),
    )
    parser.add_argument(
        "--aoi-root",
        type=Path,
        default=Path("data/processed/aois/anchor_free_2km_gap500_115"),
    )
    parser.add_argument(
        "--osm-footprints",
        type=Path,
        default=Path("data/raw/context/buildings/osm_buildings_selected_aois.geojson"),
    )
    parser.add_argument("--window-size-m", type=float, default=450.0)
    parser.add_argument("--center-x", type=float, default=None)
    parser.add_argument("--center-y", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=260)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source_root / args.aoi_id
    truth = _read_layer(source_dir / "truth_service_laterals.geojson")
    connections = _read_layer(source_dir / "osm_centroid_service_connections.geojson")
    if connections.empty:
        raise ValueError(f"No connection GeoJSON found in {source_dir}")

    if args.center_x is not None and args.center_y is not None:
        half = args.window_size_m / 2.0
        window = box(args.center_x - half, args.center_y - half, args.center_x + half, args.center_y + half)
        dense_count = int(connections.intersects(window).sum())
    else:
        window, dense_count = _choose_dense_window(connections, args.window_size_m)

    target_crs = connections.crs
    truth_zoom = _clip_layer(truth, window, target_crs)
    connections_zoom = _clip_layer(connections, window, target_crs)
    roads = _clip_layer(_read_layer(args.aoi_root / args.aoi_id / "roads.geojson"), window, target_crs)
    footprints = _clip_layer(_read_layer(args.osm_footprints), window, target_crs)

    output_dir = source_dir / f"zoom_{int(args.window_size_m)}m"
    output_png = output_dir / "osm_centroid_connections_vs_truth_zoom.png"
    truth_path = output_dir / "truth_service_laterals_zoom.geojson"
    connections_path = output_dir / "osm_centroid_service_connections_zoom.geojson"
    summary_path = output_dir / "summary_zoom.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    truth_zoom.to_file(truth_path, driver="GeoJSON")
    connections_zoom.to_file(connections_path, driver="GeoJSON")

    summary = {
        "aoi_id": args.aoi_id,
        "window_size_m": args.window_size_m,
        "window_bounds": list(window.bounds),
        "dense_connection_count_before_clipping": dense_count,
        "truth_lateral_count_in_window": int(len(truth_zoom)),
        "centroid_connection_count_in_window": int(len(connections_zoom)),
    }
    if not connections_zoom.empty:
        summary.update(
            {
                "median_hausdorff_distance_m": float(connections_zoom["hausdorff_distance_m"].median()),
                "median_length_error_m": float(connections_zoom["length_error_m"].median()),
                "median_total_snap_distance_m": float(connections_zoom["total_snap_distance_m"].median()),
            }
        )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    title = (
        f"{args.aoi_id} zoom, {args.window_size_m:.0f} m window: "
        f"{len(connections_zoom):,} centroid links vs {len(truth_zoom):,} truth laterals"
    )
    _plot_zoom(
        truth=truth_zoom,
        connections=connections_zoom,
        roads=roads,
        footprints=footprints,
        output_png=output_png,
        dpi=args.dpi,
        title=title,
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote zoom plot: {output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
