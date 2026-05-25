"""Plot AOIs with missed truth mains for skeleton-buffer predictions.

Workstream: Codex
"""

from __future__ import annotations

import argparse
import glob
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


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _expand_configs(patterns: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for pattern in patterns:
        matches = sorted(Path(match) for match in glob.glob(str(_resolve(pattern))))
        if not matches:
            path = _resolve(pattern)
            if not path.exists():
                raise FileNotFoundError(pattern)
            matches = [path]
        for path in matches:
            cfg = load_anchor_free_config(path)
            aoi_id = str(cfg.get("aoi", {}).get("aoi_id") or path.stem)
            out[aoi_id] = path
    return out


def _prediction_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _aoi_col(gdf: gpd.GeoDataFrame) -> str:
    for column in ("aoi_id", "aoi_id_x", "aoi"):
        if column in gdf.columns:
            return column
    raise ValueError("Missing AOI id column")


def _read_optional(path: str | Path | None, target_crs=None) -> gpd.GeoDataFrame:
    if not path:
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)
    resolved = _resolve(path)
    if not resolved.exists():
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)
    gdf = gpd.read_file(resolved)
    if target_crs is not None:
        if gdf.crs is None:
            gdf = gdf.set_crs(target_crs)
        elif str(gdf.crs) != str(target_crs):
            gdf = gdf.to_crs(target_crs)
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()


def _clip_to_aoi(gdf: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    geom = aoi.geometry.unary_union
    clipped = gdf[gdf.intersects(geom)].copy()
    if clipped.empty:
        return clipped
    clipped["geometry"] = clipped.geometry.intersection(geom)
    return clipped[clipped.geometry.notna() & ~clipped.geometry.is_empty].copy()


def _top_aoi_ids(summary_csv: Path, candidate_set: str, tolerance_m: float, top_n: int) -> list[str]:
    summary = pd.read_csv(summary_csv)
    subset = summary[
        (summary["candidate_set"].astype(str) == candidate_set)
        & (summary["tolerance_m"].astype(float) == float(tolerance_m))
    ].copy()
    subset = subset.sort_values("uncovered_truth_length_m", ascending=False)
    return [str(value) for value in subset["aoi_id"].head(top_n).tolist()]


def _load_aoi_layers(config_path: Path):
    cfg = load_anchor_free_config(config_path)
    data = cfg.get("data", {})
    aoi = cfg.get("aoi", {})
    aoi_id = str(aoi.get("aoi_id") or config_path.stem)
    aoi_path = _resolve(Path(str(aoi.get("source"))) / aoi_id / "aoi.geojson") if aoi.get("source") else None

    roads = _read_optional(data.get("roads_path"))
    target_crs = roads.crs if not roads.empty else None
    aoi_gdf = _read_optional(aoi_path, target_crs)
    truth = _read_optional(data.get("utility_truth_path"), target_crs)
    building_points = _read_optional(data.get("building_points_path"), target_crs)
    building_areas = _read_optional(data.get("buildings_path"), target_crs)
    drainage = _read_optional(data.get("watercourse_drainage_lines_path"), target_crs)

    if not aoi_gdf.empty:
        roads = _clip_to_aoi(roads, aoi_gdf)
        truth = _clip_to_aoi(truth, aoi_gdf)
        building_points = _clip_to_aoi(building_points, aoi_gdf)
        building_areas = _clip_to_aoi(building_areas, aoi_gdf)
        drainage = _clip_to_aoi(drainage, aoi_gdf)
    return aoi_gdf, roads, truth, building_points, building_areas, drainage


def _plot_single(
    *,
    aoi_id: str,
    config_path: Path,
    predictions: gpd.GeoDataFrame,
    uncovered: gpd.GeoDataFrame,
    candidate_set: str,
    tolerance_m: float,
    output_path: Path,
    ax=None,
) -> None:
    aoi, roads, truth, building_points, building_areas, drainage = _load_aoi_layers(config_path)
    target_crs = roads.crs if not roads.empty else truth.crs
    pred = predictions[predictions[_aoi_col(predictions)].astype(str) == aoi_id].copy()
    if target_crs is not None and pred.crs is not None and str(pred.crs) != str(target_crs):
        pred = pred.to_crs(target_crs)
    if candidate_set == "selected_prediction" and "predicted_presence" in pred.columns:
        pred = pred[_prediction_bool(pred["predicted_presence"])].copy()

    miss = uncovered[uncovered["aoi_id"].astype(str) == aoi_id].copy()
    if target_crs is not None and miss.crs is not None and str(miss.crs) != str(target_crs):
        miss = miss.to_crs(target_crs)

    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(9.5, 9.5))
    else:
        fig = ax.figure

    if not aoi.empty:
        aoi.boundary.plot(ax=ax, color="#111111", linewidth=1.0, zorder=10)
    if not building_areas.empty:
        building_areas.plot(ax=ax, facecolor="#d9f0d3", edgecolor="none", alpha=0.35, zorder=1)
    if not building_points.empty:
        building_points.plot(ax=ax, color="#238b45", markersize=2.0, alpha=0.35, zorder=2)
    if not drainage.empty:
        drainage.plot(ax=ax, color="#2b8cbe", linewidth=0.7, alpha=0.55, zorder=3)
    if not roads.empty:
        roads.plot(ax=ax, color="#bdbdbd", linewidth=0.55, alpha=0.8, zorder=4)
    if not pred.empty:
        pred.plot(ax=ax, color="#225ea8", linewidth=0.95, alpha=0.72, zorder=5)
    if not truth.empty:
        truth.plot(ax=ax, color="#fb6a4a", linewidth=1.0, alpha=0.65, zorder=6)
    if not miss.empty:
        miss.plot(ax=ax, color="#e31a1c", linewidth=2.2, alpha=0.95, zorder=7)

    ax.set_title(f"{aoi_id}: missed sewer mains at {tolerance_m:g} m", fontsize=11)
    ax.set_axis_off()
    ax.set_aspect("equal")
    if own_figure:
        handles = [
            Patch(facecolor="#d9f0d3", edgecolor="none", alpha=0.35, label="Building areas"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#238b45", markersize=4, alpha=0.65, label="Building points"),
            Line2D([0], [0], color="#bdbdbd", lw=1.2, label="Roads"),
            Line2D([0], [0], color="#225ea8", lw=1.6, label="Predicted candidates"),
            Line2D([0], [0], color="#fb6a4a", lw=1.6, label="Truth sewer mains"),
            Line2D([0], [0], color="#e31a1c", lw=2.4, label="Uncovered truth"),
            Line2D([0], [0], color="#2b8cbe", lw=1.2, label="Drainage/watercourse"),
        ]
        ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=True)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--uncovered-dir", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", default=["configs/aois_2km_gap500_115_osm_bpoints_all_mains/*.yaml"])
    parser.add_argument("--candidate-set", choices=["all_candidates", "selected_prediction"], default="selected_prediction")
    parser.add_argument("--tolerance-m", type=float, default=30.0)
    parser.add_argument("--top-n", type=int, default=6)
    parser.add_argument("--aoi-ids", nargs="*", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_by_aoi = _expand_configs(args.configs)

    predictions = gpd.read_file(_resolve(args.predictions))
    suffix = f"{float(args.tolerance_m):g}m".replace(".", "p")
    uncovered_path = _resolve(args.uncovered_dir) / f"uncovered_truth_{args.candidate_set}_{suffix}.geojson"
    uncovered = gpd.read_file(uncovered_path)

    if args.aoi_ids:
        aoi_ids = [str(value) for value in args.aoi_ids]
    else:
        aoi_ids = _top_aoi_ids(
            _resolve(args.uncovered_dir) / "uncovered_truth_summary_by_aoi.csv",
            args.candidate_set,
            args.tolerance_m,
            args.top_n,
        )

    for aoi_id in aoi_ids:
        _plot_single(
            aoi_id=aoi_id,
            config_path=config_by_aoi[aoi_id],
            predictions=predictions,
            uncovered=uncovered,
            candidate_set=args.candidate_set,
            tolerance_m=float(args.tolerance_m),
            output_path=output_dir / f"{aoi_id}_{args.candidate_set}_{suffix}.png",
        )

    ncols = min(3, len(aoi_ids))
    nrows = (len(aoi_ids) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 6.2 * nrows))
    if not isinstance(axes, (list, tuple)):
        axes_flat = list(getattr(axes, "flat", [axes]))
    else:
        axes_flat = list(axes)
    for ax, aoi_id in zip(axes_flat, aoi_ids):
        _plot_single(
            aoi_id=aoi_id,
            config_path=config_by_aoi[aoi_id],
            predictions=predictions,
            uncovered=uncovered,
            candidate_set=args.candidate_set,
            tolerance_m=float(args.tolerance_m),
            output_path=output_dir / f"{aoi_id}_{args.candidate_set}_{suffix}.png",
            ax=ax,
        )
    for ax in axes_flat[len(aoi_ids) :]:
        ax.set_axis_off()
    handles = [
        Patch(facecolor="#d9f0d3", edgecolor="none", alpha=0.35, label="Building areas"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#238b45", markersize=4, alpha=0.65, label="Building points"),
        Line2D([0], [0], color="#bdbdbd", lw=1.2, label="Roads"),
        Line2D([0], [0], color="#225ea8", lw=1.6, label="Predicted candidates"),
        Line2D([0], [0], color="#fb6a4a", lw=1.6, label="Truth sewer mains"),
        Line2D([0], [0], color="#e31a1c", lw=2.4, label="Uncovered truth"),
        Line2D([0], [0], color="#2b8cbe", lw=1.2, label="Drainage/watercourse"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle(f"Top missed AOIs: {args.candidate_set}, {args.tolerance_m:g} m tolerance", fontsize=14)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    overview_path = output_dir / f"top_{len(aoi_ids)}_{args.candidate_set}_{suffix}_overview.png"
    fig.savefig(overview_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("Plotted AOIs:")
    for aoi_id in aoi_ids:
        print(f"  {aoi_id}")
    print(f"Overview: {overview_path}")


if __name__ == "__main__":
    main()
