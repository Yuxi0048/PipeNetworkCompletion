"""Filter non-utility context layers to the road study area.

# Workstream: Codex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fiona
import geopandas as gpd
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import shapely
from shapely.geometry import box

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


DEFAULT_CONTEXT_FILES = (
    REPO_ROOT / "data" / "raw" / "context" / "buildings" / "build_up_areas.gpkg",
    REPO_ROOT / "data" / "raw" / "context" / "buildings" / "building_areas.gpkg",
    REPO_ROOT / "data" / "raw" / "context" / "buildings" / "building_points.gpkg",
    REPO_ROOT / "data" / "raw" / "context" / "buildings" / "homesteads.gpkg",
)


def _road_study_polygon(
    roads_path: Path,
    *,
    target_crs: str,
    buffer_m: float,
) -> gpd.GeoDataFrame:
    roads = gpd.read_file(roads_path).to_crs(target_crs)
    xmin, ymin, xmax, ymax = roads.total_bounds
    polygon = box(xmin, ymin, xmax, ymax).buffer(float(buffer_m))
    return gpd.GeoDataFrame({"name": ["road_study_area"]}, geometry=[polygon], crs=target_crs)


def _bbox_for_layer_crs(study_area: gpd.GeoDataFrame, layer_crs) -> tuple[float, float, float, float]:
    if layer_crs and study_area.crs and str(layer_crs) != str(study_area.crs):
        return tuple(study_area.to_crs(layer_crs).total_bounds)
    return tuple(study_area.total_bounds)


def _read_all_layers(path: Path, study_area: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    frames = []
    for layer in fiona.listlayers(path):
        with fiona.open(path, layer=layer) as src:
            bbox = _bbox_for_layer_crs(study_area, src.crs)
        frame = gpd.read_file(path, layer=layer, bbox=bbox)
        frame["source_layer"] = layer
        frames.append(frame)
    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=None)
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)


def filter_context_file(
    path: Path,
    study_area: gpd.GeoDataFrame,
    output_dir: Path,
) -> Path:
    data = _read_all_layers(path, study_area)
    if data.empty:
        filtered = data
    else:
        data = data.to_crs(study_area.crs)
        data = data[data.geometry.notna() & ~data.geometry.is_empty].copy()
        invalid_mask = ~data.geometry.is_valid
        if invalid_mask.any():
            try:
                data.loc[invalid_mask, "geometry"] = shapely.make_valid(
                    data.loc[invalid_mask, "geometry"].values
                )
            except AttributeError:
                data.loc[invalid_mask, "geometry"] = data.loc[
                    invalid_mask, "geometry"
                ].buffer(0)
        filtered = gpd.clip(data, study_area)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{path.stem}_study_area.geojson"
    if output_path.exists():
        output_path.unlink()
    for column in filtered.columns:
        if column != filtered.geometry.name and is_datetime64_any_dtype(filtered[column]):
            filtered[column] = filtered[column].astype(str)
    output_path.write_text(filtered.to_json(drop_id=True), encoding="utf-8")
    print(
        f"{path.name}: input={len(data)} filtered={len(filtered)} -> "
        f"{output_path.relative_to(REPO_ROOT)}"
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clip downloaded context layers to the road study-area extent."
    )
    parser.add_argument(
        "--roads",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "gis" / "roads" / "Roads_ExportFeatures.shp",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "context" / "study_area",
    )
    parser.add_argument("--target-crs", default="EPSG:28356")
    parser.add_argument("--buffer-m", type=float, default=500.0)
    parser.add_argument(
        "context_files",
        nargs="*",
        type=Path,
        default=list(DEFAULT_CONTEXT_FILES),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    study_area = _road_study_polygon(
        args.roads,
        target_crs=args.target_crs,
        buffer_m=args.buffer_m,
    )
    for path in args.context_files:
        resolved = path if path.is_absolute() else REPO_ROOT / path
        if not resolved.exists():
            print(f"missing: {resolved}")
            continue
        filter_context_file(resolved, study_area, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
