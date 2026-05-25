"""Plot ground-truth utility lines against estimated road-offset utilities.

Workstream: Codex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipe_network_completion.anchor_free.config import load_anchor_free_config  # noqa: E402
from pipe_network_completion.anchor_free.labels import road_offset_lane_name  # noqa: E402
from pipe_network_completion.anchor_free.pipeline import prepare_anchor_free_inputs  # noqa: E402


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gdf.empty:
        path.write_text('{"type":"FeatureCollection","features":[]}\n', encoding="utf-8")
    else:
        path.write_text(gdf.to_json(drop_id=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save a PNG overlay of truth sewer lines and predicted road-offset utilities."
    )
    parser.add_argument("--aoi-id", default="small_aoi_10_12")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="AOI config path. Defaults to configs/aois_2km_gap500_115 for --aoi-id.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "road_offset_lane_gnn_5lane_aoi115_sage_no_xy"
        / "road_offset_lane_node_predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "road_offset_lane_gnn_5lane_aoi115_sage_no_xy"
        / "plots",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--show-building-points",
        action="store_true",
        help="Overlay building point features from the AOI config when available.",
    )
    parser.add_argument(
        "--show-building-footprints",
        action="store_true",
        help="Overlay building footprint polygons from the AOI config when available.",
    )
    parser.add_argument(
        "--osm-building-footprints",
        type=Path,
        default=None,
        help=(
            "Optional OpenStreetMap building-footprint GeoJSON/GPKG to overlay. "
            "The layer is clipped to the requested AOI before plotting."
        ),
    )
    parser.add_argument(
        "--building-footprint-overlap-m",
        type=float,
        default=20.0,
        help=(
            "Distance tolerance for plotting building points as footprint-associated "
            "when both points and footprints are shown."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    aoi_id = str(args.aoi_id)
    config_path = (
        _resolve(args.config)
        if args.config is not None
        else REPO_ROOT
        / "configs"
        / "aois_2km_gap500_115"
        / f"anchor_free_2km_gap500_aoi115_{aoi_id}.yaml"
    )
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_anchor_free_config(config_path)
    config.setdefault("graph", {})["candidate_graph_type"] = "road_offsets"
    graph, building_footprints, building_points, _, _, truth = prepare_anchor_free_inputs(config)
    osm_building_footprints = None
    if args.osm_building_footprints is not None:
        osm_building_footprints = gpd.read_file(_resolve(args.osm_building_footprints))

    segments = graph.road_segments.copy()
    segments["offset_lane"] = segments.apply(road_offset_lane_name, axis=1)
    roads = segments[segments["offset_lane"] == "center"].copy()

    predictions = pd.read_csv(_resolve(args.predictions))
    predictions = predictions[predictions["aoi_id"].astype(str) == aoi_id].copy()
    predictions["predicted_presence_bool"] = (
        predictions["predicted_presence"]
        .astype(str)
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    selected = predictions[predictions["predicted_presence_bool"]][
        ["source_index", "predicted_lane_name", "presence_probability"]
    ].copy()
    estimated = segments.merge(
        selected,
        left_on=["source_index", "offset_lane"],
        right_on=["source_index", "predicted_lane_name"],
        how="inner",
    )
    estimated = gpd.GeoDataFrame(estimated, geometry="geometry", crs=segments.crs)

    fig, ax = plt.subplots(figsize=(11, 11))
    if osm_building_footprints is not None and not osm_building_footprints.empty:
        if (
            osm_building_footprints.crs
            and segments.crs
            and str(osm_building_footprints.crs) != str(segments.crs)
        ):
            osm_building_footprints = osm_building_footprints.to_crs(segments.crs)
        aoi_source = config.get("aoi", {}).get("source")
        if aoi_source is not None:
            aoi_path = _resolve(Path(aoi_source) / aoi_id / "aoi.geojson")
            if aoi_path.exists():
                aoi_geom = gpd.read_file(aoi_path)
                if aoi_geom.crs and segments.crs and str(aoi_geom.crs) != str(segments.crs):
                    aoi_geom = aoi_geom.to_crs(segments.crs)
                aoi_union = aoi_geom.geometry.unary_union
                osm_building_footprints = osm_building_footprints[
                    osm_building_footprints.intersects(aoi_union)
                ].copy()
                if not osm_building_footprints.empty:
                    osm_building_footprints["geometry"] = (
                        osm_building_footprints.geometry.intersection(aoi_union)
                    )
                    osm_building_footprints = osm_building_footprints[
                        ~osm_building_footprints.geometry.is_empty
                    ].copy()
        if not osm_building_footprints.empty:
            osm_building_footprints.plot(
                ax=ax,
                facecolor="#8dd3c7",
                edgecolor="#1b6f67",
                linewidth=0.25,
                alpha=0.42,
                zorder=1,
            )
    if (
        args.show_building_footprints
        and building_footprints is not None
        and not building_footprints.empty
    ):
        if (
            building_footprints.crs
            and segments.crs
            and str(building_footprints.crs) != str(segments.crs)
        ):
            building_footprints = building_footprints.to_crs(segments.crs)
        building_footprints.plot(
            ax=ax,
            facecolor="#f2c94c",
            edgecolor="#7a5c00",
            linewidth=0.35,
            alpha=0.32,
            zorder=1,
        )
    roads.plot(ax=ax, color="#d0d0d0", linewidth=0.6, alpha=0.7, zorder=2)
    if not estimated.empty:
        estimated.plot(
            ax=ax,
            column="presence_probability",
            cmap="Blues",
            linewidth=1.6,
            alpha=0.9,
            zorder=4,
            legend=True,
            legend_kwds={"label": "Predicted presence probability", "shrink": 0.55},
        )
    if not truth.empty:
        truth.plot(ax=ax, color="#d62728", linewidth=1.4, alpha=0.85, zorder=5)
    if args.show_building_points and building_points is not None and not building_points.empty:
        if building_points.crs and segments.crs and str(building_points.crs) != str(segments.crs):
            building_points = building_points.to_crs(segments.crs)
        overlap_count = None
        point_only_count = None
        if (
            args.show_building_footprints
            and building_footprints is not None
            and not building_footprints.empty
        ):
            nearest = gpd.sjoin_nearest(
                building_points[["geometry"]],
                building_footprints[["geometry"]],
                how="left",
                distance_col="footprint_dist_m",
            )
            min_dist = nearest.groupby(level=0)["footprint_dist_m"].min()
            min_dist = min_dist.reindex(building_points.index)
            overlap_mask = min_dist <= float(args.building_footprint_overlap_m)
            footprint_points = building_points[overlap_mask.fillna(False)].copy()
            point_only = building_points[~overlap_mask.fillna(False)].copy()
            overlap_count = len(footprint_points)
            point_only_count = len(point_only)
            if not footprint_points.empty:
                footprint_points.plot(
                    ax=ax,
                    color="#006d2c",
                    edgecolor="white",
                    linewidth=0.35,
                    markersize=10,
                    alpha=0.95,
                    zorder=7,
                )
            if not point_only.empty:
                point_only.plot(
                    ax=ax,
                    color="#ff7f0e",
                    marker="x",
                    markersize=12,
                    alpha=0.95,
                    zorder=7,
                )
        else:
            building_points.plot(
                ax=ax,
                color="#2ca02c",
                markersize=5,
                alpha=0.85,
                zorder=7,
            )
    ax.set_title(
        f"{aoi_id}: Ground Truth Utility vs Estimated Road-Offset Utilities",
        fontsize=13,
    )
    ax.set_axis_off()
    handles = [
        Line2D([0], [0], color="#d0d0d0", lw=1.2, label="Road centerline"),
        Line2D([0], [0], color="#1f77b4", lw=2.0, label="Estimated utility"),
        Line2D([0], [0], color="#d62728", lw=2.0, label="Ground truth sewer"),
    ]
    if args.show_building_points and args.show_building_footprints:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#006d2c",
                markeredgecolor="white",
                markersize=6,
                label=f"Building points near footprints <= {args.building_footprint_overlap_m:g} m",
            )
        )
        handles.append(
            Line2D(
                [0],
                [0],
                marker="x",
                color="#ff7f0e",
                markersize=6,
                label="Point-only buildings",
            )
        )
    elif args.show_building_points:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#2ca02c",
                markeredgecolor="#2ca02c",
                markersize=5,
                label="Building points",
            )
        )
    if args.show_building_footprints:
        handles.append(
            Patch(
                facecolor="#f2c94c",
                edgecolor="#7a5c00",
                alpha=0.32,
                label="Building footprints",
            )
        )
    if osm_building_footprints is not None:
        handles.append(
            Patch(
                facecolor="#8dd3c7",
                edgecolor="#1b6f67",
                alpha=0.42,
                label="OSM building footprints",
            )
        )
    ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=0.9)
    fig.tight_layout()

    png_path = output_dir / f"{aoi_id}_truth_vs_estimated_utility.png"
    fig.savefig(png_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    estimated_path = output_dir / f"{aoi_id}_estimated_utility.geojson"
    _write_geojson(
        estimated[
            [
                "source_index",
                "offset_lane",
                "predicted_lane_name",
                "presence_probability",
                "geometry",
            ]
        ],
        estimated_path,
    )

    print(f"Saved PNG: {png_path}")
    print(f"Saved estimated GeoJSON: {estimated_path}")
    print(
        f"AOI={aoi_id} truth_lines={len(truth)} "
        f"estimated_segments={len(estimated)} predicted_source_roads={len(selected)}"
    )
    if osm_building_footprints is not None:
        print(f"OSM building footprints plotted: {len(osm_building_footprints)}")
    if args.show_building_points and "overlap_count" in locals() and overlap_count is not None:
        print(
            f"Building point-footprint plot classes: "
            f"near_footprint={overlap_count} point_only={point_only_count} "
            f"tolerance_m={args.building_footprint_overlap_m:g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
