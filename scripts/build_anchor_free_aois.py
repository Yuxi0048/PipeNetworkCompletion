"""Build small non-overlapping AOIs for anchor-free utility experiments.

Workstream: Codex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipe_network_completion.anchor_free.aoi import (  # noqa: E402
    AOIThresholds,
    assign_aoi_splits,
    clip_vector_to_aoi,
    make_non_overlapping_grid_aois,
    select_viable_aois,
    summarize_aoi_content,
)
from pipe_network_completion.anchor_free.config import (  # noqa: E402
    deep_update,
    load_anchor_free_config,
    write_resolved_config,
)


DEFAULT_TRUTH_PATHS = [
    REPO_ROOT / "data" / "raw" / "gis" / "sewer" / "SewerGravityMa_ExportFeature1.shp",
    REPO_ROOT / "data" / "raw" / "gis" / "sewer" / "SewerGravityMa_ExportFeature2.shp",
]


def _resolve(path: str | Path | None) -> Path | None:
    if path in (None, ""):
        return None
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(resolved)


def _read_vector(path: str | Path | None, *, target_crs: str | None = None) -> gpd.GeoDataFrame | None:
    resolved = _resolve(path)
    if resolved is None:
        return None
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    data = gpd.read_file(resolved)
    if target_crs and data.crs is not None and str(data.crs) != str(target_crs):
        data = data.to_crs(target_crs)
    return data[data.geometry.notna() & ~data.geometry.is_empty].copy()


def _read_vector_many(
    paths: Iterable[str | Path],
    *,
    target_crs: str | None = None,
) -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = []
    crs = None
    for path in paths:
        frame = _read_vector(path, target_crs=target_crs)
        if frame is None:
            continue
        if crs is None:
            crs = frame.crs
        elif frame.crs is not None and str(frame.crs) != str(crs):
            frame = frame.to_crs(crs)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No vector files could be loaded.")
    if len(frames) == 1:
        return frames[0]
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=crs)


def _write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = gdf.copy()
    for column in frame.columns:
        if column != frame.geometry.name and is_datetime64_any_dtype(frame[column]):
            frame[column] = frame[column].astype(str)
    if frame.empty:
        path.write_text('{"type":"FeatureCollection","features":[]}\n', encoding="utf-8")
    else:
        path.write_text(frame.to_json(drop_id=True), encoding="utf-8")


def _bounds_intersection(bounds_a, bounds_b) -> list[float]:
    minx = max(float(bounds_a[0]), float(bounds_b[0]))
    miny = max(float(bounds_a[1]), float(bounds_b[1]))
    maxx = min(float(bounds_a[2]), float(bounds_b[2]))
    maxy = min(float(bounds_a[3]), float(bounds_b[3]))
    if maxx <= minx or maxy <= miny:
        raise ValueError("Layer bounds do not overlap.")
    return [minx, miny, maxx, maxy]


def _analysis_bounds(
    *,
    roads: gpd.GeoDataFrame,
    building_points: gpd.GeoDataFrame | None,
    mode: str,
    shrink_m: float,
) -> list[float]:
    bounds = [float(v) for v in roads.total_bounds]
    if mode == "roads_and_buildings" and building_points is not None and not building_points.empty:
        bounds = _bounds_intersection(bounds, building_points.total_bounds)
    elif mode != "roads":
        raise ValueError("extent_mode must be 'roads' or 'roads_and_buildings'.")

    shrink = max(float(shrink_m), 0.0)
    if shrink:
        bounds = [
            bounds[0] + shrink,
            bounds[1] + shrink,
            bounds[2] - shrink,
            bounds[3] - shrink,
        ]
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError("AOI extent is empty after applying shrink_m.")
    return bounds


def _clip_and_write(
    *,
    gdf: gpd.GeoDataFrame | None,
    aoi: gpd.GeoDataFrame,
    path: Path,
) -> Path | None:
    if gdf is None:
        return None
    clipped = clip_vector_to_aoi(gdf, aoi)
    _write_geojson(clipped, path)
    return path


def _write_aoi_config(
    *,
    base_config_path: Path | None,
    experiment_name: str,
    aoi_id: str,
    split: str,
    output_dir: Path,
    config_dir: Path,
    roads_path: Path,
    truth_path: Path,
    building_points_path: Path | None,
    building_areas_path: Path | None,
    built_up_path: Path | None,
    dem_path: Path | None,
    target_crs: str,
    candidate_graph_type: str,
) -> Path:
    config = load_anchor_free_config(base_config_path)
    override = {
        "experiment_name": f"{experiment_name}_{aoi_id}",
        "mode": "anchor_free",
        "task_type": "sewer",
        "data": {
            "roads_path": _relative(roads_path),
            "utility_truth_path": _relative(truth_path),
            "building_points_path": _relative(building_points_path) if building_points_path else "",
            "buildings_path": _relative(building_areas_path) if building_areas_path else "",
            "built_up_path": _relative(built_up_path) if built_up_path else "",
            "dem_path": _relative(dem_path) if dem_path else "",
        },
        "graph": {
            "target_crs": target_crs,
            "candidate_graph_type": candidate_graph_type,
        },
        "aoi": {
            "aoi_id": aoi_id,
            "split": split,
            "source": _relative(output_dir),
        },
    }
    config = deep_update(config, override)
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{experiment_name}_{aoi_id}.yaml"
    write_resolved_config(config, config_path)
    return config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create small non-overlapping AOIs and optional clipped datasets for "
            "anchor-free utility corridor experiments."
        )
    )
    parser.add_argument(
        "--roads",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "gis" / "roads" / "Roads_ExportFeatures.shp",
    )
    parser.add_argument(
        "--utility-truth",
        nargs="+",
        type=Path,
        default=DEFAULT_TRUTH_PATHS,
    )
    parser.add_argument(
        "--building-points",
        type=Path,
        default=REPO_ROOT
        / "data"
        / "processed"
        / "context"
        / "study_area"
        / "building_points_study_area.geojson",
    )
    parser.add_argument(
        "--building-areas",
        type=Path,
        default=REPO_ROOT
        / "data"
        / "processed"
        / "context"
        / "study_area"
        / "building_areas_study_area.geojson",
    )
    parser.add_argument(
        "--built-up",
        type=Path,
        default=REPO_ROOT
        / "data"
        / "processed"
        / "context"
        / "study_area"
        / "build_up_areas_study_area.geojson",
    )
    parser.add_argument(
        "--dem",
        type=Path,
        default=REPO_ROOT
        / "data"
        / "processed"
        / "context"
        / "study_area"
        / "brisbane_dem_h_1sec_epsg28356.tif",
    )
    parser.add_argument("--target-crs", default="EPSG:28356")
    parser.add_argument("--tile-size-m", type=float, default=4_000.0)
    parser.add_argument("--gap-m", type=float, default=1_000.0)
    parser.add_argument("--extent-mode", choices=["roads", "roads_and_buildings"], default="roads_and_buildings")
    parser.add_argument("--extent-shrink-m", type=float, default=0.0)
    parser.add_argument("--min-road-length-m", type=float, default=5_000.0)
    parser.add_argument("--min-truth-length-m", type=float, default=1_000.0)
    parser.add_argument("--min-building-points", type=int, default=25)
    parser.add_argument("--max-aois", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--experiment-name", default="anchor_free_small_aoi")
    parser.add_argument("--candidate-graph-type", choices=["road", "hybrid"], default="hybrid")
    parser.add_argument("--base-config", type=Path, default=REPO_ROOT / "configs" / "anchor_free_isarc2024.yaml")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "aois" / "anchor_free_small_nonoverlap",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=REPO_ROOT / "configs" / "aois",
    )
    parser.add_argument(
        "--clip-layers",
        action="store_true",
        help="Write clipped road/context/truth GeoJSON files and per-AOI configs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    config_dir = _resolve(args.config_dir)
    if output_dir is None or config_dir is None:
        raise ValueError("output-dir and config-dir are required.")

    roads = _read_vector(args.roads, target_crs=args.target_crs)
    if roads is None:
        raise FileNotFoundError(args.roads)
    truth = _read_vector_many(args.utility_truth, target_crs=args.target_crs)
    building_points = _read_vector(args.building_points, target_crs=args.target_crs)
    building_areas = _read_vector(args.building_areas, target_crs=args.target_crs)
    built_up = _read_vector(args.built_up, target_crs=args.target_crs)

    bounds = _analysis_bounds(
        roads=roads,
        building_points=building_points,
        mode=args.extent_mode,
        shrink_m=args.extent_shrink_m,
    )
    aois = make_non_overlapping_grid_aois(
        bounds,
        tile_size_m=args.tile_size_m,
        gap_m=args.gap_m,
        crs=args.target_crs,
        aoi_prefix="small_aoi",
    )
    summary = summarize_aoi_content(
        aois,
        roads=roads,
        utility_truth=truth,
        building_points=building_points,
    )
    selected = select_viable_aois(
        summary,
        thresholds=AOIThresholds(
            min_road_length_m=args.min_road_length_m,
            min_truth_length_m=args.min_truth_length_m,
            min_building_points=args.min_building_points,
        ),
        max_aois=args.max_aois,
        min_gap_m=args.gap_m,
    )
    selected = assign_aoi_splits(
        selected,
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_geojson(aois, output_dir / "aoi_grid.geojson")
    _write_geojson(summary, output_dir / "aoi_summary.geojson")
    _write_geojson(selected, output_dir / "selected_aois.geojson")
    summary.drop(columns="geometry").to_csv(output_dir / "aoi_summary.csv", index=False)
    selected.drop(columns="geometry").to_csv(output_dir / "selected_aois.csv", index=False)

    config_paths = []
    if args.clip_layers:
        for row in selected.itertuples():
            aoi = gpd.GeoDataFrame(
                [{"aoi_id": row.aoi_id, "split": row.split, "geometry": row.geometry}],
                geometry="geometry",
                crs=selected.crs,
            )
            aoi_dir = output_dir / str(row.aoi_id)
            _write_geojson(aoi, aoi_dir / "aoi.geojson")
            roads_path = _clip_and_write(gdf=roads, aoi=aoi, path=aoi_dir / "roads.geojson")
            truth_path = _clip_and_write(
                gdf=truth,
                aoi=aoi,
                path=aoi_dir / "utility_truth_gravity_mains.geojson",
            )
            bpoints_path = _clip_and_write(
                gdf=building_points,
                aoi=aoi,
                path=aoi_dir / "building_points.geojson",
            )
            bareas_path = _clip_and_write(
                gdf=building_areas,
                aoi=aoi,
                path=aoi_dir / "building_areas.geojson",
            )
            built_path = _clip_and_write(
                gdf=built_up,
                aoi=aoi,
                path=aoi_dir / "built_up.geojson",
            )
            if roads_path is None or truth_path is None:
                continue
            config_path = _write_aoi_config(
                base_config_path=_resolve(args.base_config),
                experiment_name=args.experiment_name,
                aoi_id=str(row.aoi_id),
                split=str(row.split),
                output_dir=output_dir,
                config_dir=config_dir,
                roads_path=roads_path,
                truth_path=truth_path,
                building_points_path=bpoints_path,
                building_areas_path=bareas_path,
                built_up_path=built_path,
                dem_path=_resolve(args.dem) if args.dem and _resolve(args.dem).exists() else None,
                target_crs=args.target_crs,
                candidate_graph_type=args.candidate_graph_type,
            )
            config_paths.append(_relative(config_path))

    manifest = {
        "workstream": "Codex",
        "purpose": "small non-overlapping AOIs for spatially separated anchor-free experiments",
        "target_crs": args.target_crs,
        "tile_size_m": float(args.tile_size_m),
        "gap_m": float(args.gap_m),
        "extent_mode": args.extent_mode,
        "n_grid_aois": int(len(aois)),
        "n_selected_aois": int(len(selected)),
        "split_counts": selected["split"].value_counts().to_dict() if len(selected) else {},
        "clip_layers": bool(args.clip_layers),
        "config_paths": config_paths,
        "source_paths": {
            "roads": _relative(_resolve(args.roads)),
            "utility_truth": [_relative(_resolve(path)) for path in args.utility_truth],
            "building_points": _relative(_resolve(args.building_points)),
            "building_areas": _relative(_resolve(args.building_areas)),
            "built_up": _relative(_resolve(args.built_up)),
            "dem": _relative(_resolve(args.dem)) if args.dem and _resolve(args.dem).exists() else "",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"Wrote AOI grid: {_relative(output_dir / 'aoi_grid.geojson')}")
    print(f"Wrote selected AOIs: {_relative(output_dir / 'selected_aois.geojson')}")
    print(f"Selected {len(selected)} AOIs from {len(aois)} candidates.")
    if len(selected):
        print(selected[["aoi_id", "split", "road_length_m", "truth_length_m", "building_point_count"]].to_string(index=False))
    if config_paths:
        print(f"Wrote {len(config_paths)} per-AOI configs under {_relative(config_dir)}")
    print("No model training was started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

