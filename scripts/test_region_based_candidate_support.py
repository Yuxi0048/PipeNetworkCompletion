"""Test anchor-free region-based sewer-main candidate support.

Workstream: Codex

This script tests the first stage of a region-to-network story:

1. Build feasible sewer-main corridor regions from allowed surface supports.
2. Treat building footprints as obstacles / low-suitability regions.
3. Use road and drainage centerlines as a skeleton proxy for the candidate graph.
4. Measure how much true gravity-main length the candidate support can cover.

No manholes, valves, surveyed utility nodes, or true utility endpoints are used as
inputs. The true gravity-main layer is used only for candidate-recall evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import GeometryCollection
from shapely.ops import unary_union


def _read_layer(path: Path, target_crs=None) -> gpd.GeoDataFrame:
    if not path.exists():
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)
    gdf = gpd.read_file(path)
    if target_crs is not None and gdf.crs is not None and str(gdf.crs) != str(target_crs):
        gdf = gdf.to_crs(target_crs)
    return gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()


def _clip_layer(gdf: gpd.GeoDataFrame, clip_geometry) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    clipped = gdf[gdf.intersects(clip_geometry)].copy()
    if clipped.empty:
        return clipped
    clipped["geometry"] = clipped.geometry.intersection(clip_geometry)
    clipped = clipped[~clipped.geometry.is_empty & clipped.geometry.notna()].copy()
    return clipped


def _safe_union(geometries: Iterable) -> GeometryCollection:
    valid = [geom for geom in geometries if geom is not None and not geom.is_empty]
    if not valid:
        return GeometryCollection()
    return unary_union(valid)


def _buffer_union(gdf: gpd.GeoDataFrame, distance_m: float):
    if gdf.empty:
        return GeometryCollection()
    return _safe_union(gdf.geometry.buffer(distance_m))


def _difference(base, obstacle):
    if base.is_empty:
        return base
    if obstacle.is_empty:
        return base
    return base.difference(obstacle)


def _intersection(base, mask):
    if base.is_empty or mask.is_empty:
        return GeometryCollection()
    return base.intersection(mask)


def _length_within(truth: gpd.GeoDataFrame, support_geometry, tolerance_m: float) -> float:
    if truth.empty or support_geometry.is_empty:
        return 0.0
    support = support_geometry.buffer(tolerance_m) if tolerance_m > 0 else support_geometry
    return float(truth.geometry.intersection(support).length.sum())


def _support_length(support_geometry) -> float:
    if support_geometry.is_empty:
        return 0.0
    return float(support_geometry.length)


def _support_area(support_geometry) -> float:
    if support_geometry.is_empty:
        return 0.0
    return float(support_geometry.area)


def _to_gdf(geometry, crs) -> gpd.GeoDataFrame:
    if geometry.is_empty:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    if hasattr(geometry, "geoms"):
        geoms = [geom for geom in geometry.geoms if not geom.is_empty]
    else:
        geoms = [geometry]
    return gpd.GeoDataFrame(geometry=geoms, crs=crs)


def _skeleton_proxy(lines: list[gpd.GeoDataFrame], region):
    geometries = []
    for gdf in lines:
        if gdf.empty:
            continue
        if region.is_empty:
            geometries.extend(list(gdf.geometry))
        else:
            clipped = gdf.geometry.intersection(region)
            geometries.extend([geom for geom in clipped if geom is not None and not geom.is_empty])
    return _safe_union(geometries)


def _plot_result(
    aoi: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    drainage_lines: gpd.GeoDataFrame,
    footprints: gpd.GeoDataFrame,
    truth: gpd.GeoDataFrame,
    candidate_region,
    skeleton,
    output_png: Path,
    title: str,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    candidate_gdf = _to_gdf(candidate_region, aoi.crs)
    skeleton_gdf = _to_gdf(skeleton, aoi.crs)
    minx, miny, maxx, maxy = aoi.total_bounds
    pad_x = (maxx - minx) * 0.02
    pad_y = (maxy - miny) * 0.02

    for ax in axes:
        if not candidate_gdf.empty:
            candidate_gdf.plot(ax=ax, facecolor="#c6dbef", edgecolor="none", alpha=0.35, zorder=1)
        if not footprints.empty:
            footprints.plot(
                ax=ax,
                facecolor="#f2f2f2",
                edgecolor="#8c8c8c",
                linewidth=0.2,
                alpha=0.7,
                zorder=2,
            )
        if not roads.empty:
            roads.plot(ax=ax, color="#bdbdbd", linewidth=0.6, alpha=0.8, zorder=3)
        if not drainage_lines.empty:
            drainage_lines.plot(ax=ax, color="#41b6c4", linewidth=1.0, alpha=0.9, zorder=4)
        aoi.boundary.plot(ax=ax, color="#252525", linewidth=1.0, zorder=10)
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
        ax.set_aspect("equal")
        ax.set_axis_off()

    axes[0].set_title("Candidate region vs truth")
    axes[1].set_title("Skeleton proxy vs truth")

    if not truth.empty:
        truth.plot(ax=axes[0], color="#d7301f", linewidth=1.15, alpha=0.82, zorder=7)
        truth.plot(ax=axes[1], color="#d7301f", linewidth=1.0, alpha=0.48, zorder=7)
    if not skeleton_gdf.empty:
        skeleton_gdf.plot(ax=axes[1], color="#08519c", linewidth=1.15, alpha=0.85, zorder=8)

    legend_items = [
        plt.Line2D([0], [0], color="#d7301f", lw=2, label="truth gravity main"),
        plt.Line2D([0], [0], color="#08519c", lw=2, label="candidate skeleton proxy"),
        plt.Line2D([0], [0], color="#bdbdbd", lw=2, label="road"),
        plt.Line2D([0], [0], color="#41b6c4", lw=2, label="drainage line"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#c6dbef", alpha=0.35, label="candidate region"),
    ]
    fig.suptitle(title)
    fig.legend(handles=legend_items, loc="lower center", ncol=5, frameon=False)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _evaluate_support(
    variant: str,
    support_kind: str,
    support_geometry,
    truth: gpd.GeoDataFrame,
    truth_total_length_m: float,
    tolerances_m: list[float],
) -> list[dict[str, object]]:
    rows = []
    for tolerance in tolerances_m:
        covered = _length_within(truth, support_geometry, tolerance)
        recall = covered / truth_total_length_m if truth_total_length_m else 0.0
        rows.append(
            {
                "variant": variant,
                "support_kind": support_kind,
                "tolerance_m": float(tolerance),
                "truth_total_length_m": truth_total_length_m,
                "covered_truth_length_m": covered,
                "support_recall_ceiling": recall,
                "irreducible_fn_fraction": 1.0 - recall,
                "support_area_m2": _support_area(support_geometry),
                "support_length_m": _support_length(support_geometry),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi-id", default="small_aoi_10_12")
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
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/region_based_candidate_support"),
    )
    parser.add_argument("--road-buffer-m", type=float, default=18.0)
    parser.add_argument("--drainage-buffer-m", type=float, default=25.0)
    parser.add_argument("--building-exclusion-buffer-m", type=float, default=1.0)
    parser.add_argument("--building-demand-buffer-m", type=float, default=120.0)
    parser.add_argument("--skeleton-buffer-m", type=float, default=12.0)
    parser.add_argument("--tolerances-m", type=float, nargs="+", default=[0, 5, 10, 20, 50])
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    aoi_dir = args.aoi_root / args.aoi_id
    aoi = _read_layer(aoi_dir / "aoi.geojson")
    if aoi.empty:
        raise FileNotFoundError(f"AOI not found: {aoi_dir / 'aoi.geojson'}")
    target_crs = aoi.crs
    aoi_union = aoi.geometry.unary_union

    roads = _clip_layer(_read_layer(aoi_dir / "roads.geojson", target_crs), aoi_union)
    truth = _clip_layer(_read_layer(aoi_dir / "utility_truth_gravity_mains.geojson", target_crs), aoi_union)
    watercourse_corridors = _clip_layer(_read_layer(aoi_dir / "watercourse_corridors.geojson", target_crs), aoi_union)
    watercourse_lines = _clip_layer(_read_layer(aoi_dir / "watercourse_drainage_lines.geojson", target_crs), aoi_union)
    if watercourse_lines.empty:
        watercourse_lines = _clip_layer(
            _read_layer(aoi_dir / "watercourse_corridor_centrelines.geojson", target_crs),
            aoi_union,
        )
    footprints = _clip_layer(_read_layer(args.osm_footprints, target_crs), aoi_union)

    road_region = _buffer_union(roads, args.road_buffer_m)
    drainage_region = _safe_union(list(watercourse_corridors.geometry))
    if drainage_region.is_empty:
        drainage_region = _buffer_union(watercourse_lines, args.drainage_buffer_m)
    support_region = _intersection(_safe_union([road_region, drainage_region]), aoi_union)
    building_obstacle = _buffer_union(footprints, args.building_exclusion_buffer_m)
    building_demand_region = _intersection(_buffer_union(footprints, args.building_demand_buffer_m), aoi_union)

    variants = {
        "road_region": _intersection(road_region, aoi_union),
        "road_region_minus_buildings": _difference(_intersection(road_region, aoi_union), building_obstacle),
        "road_drainage_region": support_region,
        "road_drainage_region_minus_buildings": _difference(support_region, building_obstacle),
        "demand_shaped_road_drainage_minus_buildings": _intersection(
            _difference(support_region, building_obstacle),
            building_demand_region,
        ),
    }

    skeletons = {
        "road_centerline_proxy": _skeleton_proxy([roads], GeometryCollection()),
        "road_drainage_centerline_proxy": _skeleton_proxy([roads, watercourse_lines], GeometryCollection()),
        "skeleton_proxy_inside_best_region": _skeleton_proxy(
            [roads, watercourse_lines],
            variants["road_drainage_region_minus_buildings"],
        ),
    }

    output_dir = args.output_root / args.aoi_id
    output_dir.mkdir(parents=True, exist_ok=True)

    truth_total_length_m = float(truth.length.sum())
    rows = []
    for variant_name, geometry in variants.items():
        rows.extend(
            _evaluate_support(
                variant_name,
                "region",
                geometry,
                truth,
                truth_total_length_m,
                args.tolerances_m,
            )
        )
    for skeleton_name, geometry in skeletons.items():
        rows.extend(
            _evaluate_support(
                skeleton_name,
                "centerline_skeleton_proxy",
                geometry,
                truth,
                truth_total_length_m,
                args.tolerances_m,
            )
        )
        buffer_name = f"{skeleton_name}_buffer_{args.skeleton_buffer_m:g}m"
        rows.extend(
            _evaluate_support(
                buffer_name,
                "rerendered_skeleton_buffer",
                geometry.buffer(args.skeleton_buffer_m),
                truth,
                truth_total_length_m,
                args.tolerances_m,
            )
        )

    metrics_path = output_dir / "candidate_support_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best_region = variants["road_drainage_region_minus_buildings"]
    best_skeleton = skeletons["skeleton_proxy_inside_best_region"]
    _to_gdf(best_region, target_crs).to_file(output_dir / "candidate_region.geojson", driver="GeoJSON")
    _to_gdf(best_skeleton, target_crs).to_file(output_dir / "candidate_skeleton_proxy.geojson", driver="GeoJSON")
    _to_gdf(best_skeleton.buffer(args.skeleton_buffer_m), target_crs).to_file(
        output_dir / f"candidate_skeleton_proxy_buffer_{args.skeleton_buffer_m:g}m.geojson",
        driver="GeoJSON",
    )

    summary = {
        "aoi_id": args.aoi_id,
        "road_buffer_m": args.road_buffer_m,
        "drainage_buffer_m": args.drainage_buffer_m,
        "building_exclusion_buffer_m": args.building_exclusion_buffer_m,
        "building_demand_buffer_m": args.building_demand_buffer_m,
        "skeleton_buffer_m": args.skeleton_buffer_m,
        "truth_total_length_m": truth_total_length_m,
        "road_count": int(len(roads)),
        "watercourse_line_count": int(len(watercourse_lines)),
        "osm_footprint_count": int(len(footprints)),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_path = output_dir / "region_candidate_support_vs_truth.png"
    _plot_result(
        aoi=aoi,
        roads=roads,
        drainage_lines=watercourse_lines,
        footprints=footprints,
        truth=truth,
        candidate_region=best_region,
        skeleton=best_skeleton,
        output_png=plot_path,
        title=f"{args.aoi_id}: region candidate and skeleton proxy",
        dpi=args.dpi,
    )

    print(json.dumps(summary, indent=2))
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
