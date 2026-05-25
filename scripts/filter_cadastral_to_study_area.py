"""Reproject and clip Queensland cadastral layers to the road study area.

# Workstream: Codex
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_GDAL_DATA = Path(sys.prefix) / "Library" / "share" / "gdal"
if "GDAL_DATA" not in os.environ and DEFAULT_GDAL_DATA.exists():
    os.environ["GDAL_DATA"] = str(DEFAULT_GDAL_DATA)

import fiona
import geopandas as gpd
import pandas as pd
from pyproj import CRS, Transformer
import shapely
from pandas.api.types import is_datetime64_any_dtype
from shapely.geometry import box, mapping, shape
from shapely.ops import transform as transform_geometry

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


LAYER_MAP = {
    "QLD_CADASTRE_DCDB": "dcdb_parcels",
    "QLD_CADASTRE_ROAD": "cadastral_roads",
    "QLD_LOCATION_ADDRESS": "address_points",
    "QLD_CADASTRE_NATBDY": "natural_boundaries",
}


def road_study_area(roads_path: Path, target_crs: str, buffer_m: float) -> gpd.GeoDataFrame:
    roads = gpd.read_file(roads_path).to_crs(target_crs)
    xmin, ymin, xmax, ymax = roads.total_bounds
    study_geom = box(xmin, ymin, xmax, ymax).buffer(float(buffer_m))
    return gpd.GeoDataFrame(
        {
            "name": ["road_study_area"],
            "source": [str(roads_path.relative_to(REPO_ROOT))],
            "buffer_m": [float(buffer_m)],
        },
        geometry=[study_geom],
        crs=target_crs,
    )


def sewer_split_study_area(
    split_dir: Path,
    target_crs: str,
    buffer_m: float,
) -> gpd.GeoDataFrame:
    split_paths = [split_dir / f"{name}.shp" for name in ("train", "val", "test")]
    missing = [path for path in split_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing sewer split shapefiles: "
            + ", ".join(str(path) for path in missing)
        )

    target = CRS.from_user_input(target_crs)
    xs: list[float] = []
    ys: list[float] = []
    feature_count = 0
    for path in split_paths:
        with fiona.open(path) as src:
            source = CRS.from_user_input(src.crs)
            transformer = Transformer.from_crs(source, target, always_xy=True)
            xmin_src, ymin_src, xmax_src, ymax_src = src.bounds
            corners = (
                (xmin_src, ymin_src),
                (xmin_src, ymax_src),
                (xmax_src, ymin_src),
                (xmax_src, ymax_src),
            )
            for x_src, y_src in corners:
                x, y = transformer.transform(x_src, y_src)
                xs.append(float(x))
                ys.append(float(y))
            feature_count += len(src)

    xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
    study_geom = box(xmin, ymin, xmax, ymax).buffer(float(buffer_m))
    return gpd.GeoDataFrame(
        {
            "name": ["sewer_split_study_area"],
            "source": [str(split_dir.relative_to(REPO_ROOT))],
            "buffer_m": [float(buffer_m)],
            "feature_count": [int(feature_count)],
        },
        geometry=[study_geom],
        crs=target_crs,
    )


def bbox_in_layer_crs(study_area: gpd.GeoDataFrame, layer_crs) -> tuple[float, float, float, float]:
    if layer_crs and study_area.crs and str(layer_crs) != str(study_area.crs):
        return tuple(study_area.to_crs(layer_crs).total_bounds)
    return tuple(study_area.total_bounds)


def sanitize_for_gpkg(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    data = gdf.copy()
    for column in data.columns:
        if column == data.geometry.name:
            continue
        if is_datetime64_any_dtype(data[column]):
            data[column] = data[column].astype(str)
        elif data[column].dtype == object:
            data[column] = data[column].map(
                lambda value: json.dumps(value) if isinstance(value, (dict, list)) else value
            )
    return data


def fix_geometry(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    data = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if data.empty:
        return data
    invalid = ~data.geometry.is_valid
    if invalid.any():
        try:
            data.loc[invalid, "geometry"] = shapely.make_valid(data.loc[invalid, "geometry"].values)
        except AttributeError:
            data.loc[invalid, "geometry"] = data.loc[invalid, "geometry"].buffer(0)
    return data[data.geometry.notna() & ~data.geometry.is_empty].copy()


def add_metric_columns(gdf: gpd.GeoDataFrame, layer_name: str) -> gpd.GeoDataFrame:
    data = gdf.copy()
    geom_types = set(data.geometry.geom_type.dropna().unique())
    if not geom_types:
        return data
    if any("Polygon" in geom_type for geom_type in geom_types):
        data["area_m2"] = data.geometry.area
    if any("LineString" in geom_type for geom_type in geom_types):
        data["length_m"] = data.geometry.length
    data["source_layer"] = layer_name
    return data


def metric_property_schema(geometry_type: str | None) -> dict[str, str]:
    if geometry_type and "Polygon" in geometry_type:
        return {"area_m2": "float"}
    if geometry_type and "LineString" in geometry_type:
        return {"length_m": "float"}
    return {}


def metric_properties(geometry, geometry_type: str | None) -> dict[str, float]:
    if geometry_type and "Polygon" in geometry_type:
        return {"area_m2": float(geometry.area)}
    if geometry_type and "LineString" in geometry_type:
        return {"length_m": float(geometry.length)}
    return {}


def stream_clip_layer_to_file(
    gdb_path: Path,
    source_layer: str,
    output_path: Path,
    study_area: gpd.GeoDataFrame,
    *,
    driver: str,
    overwrite: bool,
) -> tuple[int, set[str]]:
    if output_path.exists():
        if not overwrite:
            raise SystemExit(f"output exists; pass --overwrite to replace: {output_path}")
        output_path.unlink()

    study_geom = study_area.geometry.iloc[0]
    target_crs = CRS.from_user_input(study_area.crs)
    lotplans: set[str] = set()
    count = 0

    with fiona.open(gdb_path, layer=source_layer) as src:
        bbox = bbox_in_layer_crs(study_area, src.crs)
        source_crs = CRS.from_user_input(src.crs) if src.crs else target_crs
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        source_schema = src.schema.copy()
        source_properties = dict(source_schema.get("properties", {}))
        geometry_type = source_schema.get("geometry")
        output_schema = {
            "geometry": "Unknown",
            "properties": {
                **source_properties,
                "source_layer": "str:80",
                **metric_property_schema(geometry_type),
            },
        }

        with fiona.open(
            output_path,
            "w",
            driver=driver,
            schema=output_schema,
            crs_wkt=target_crs.to_wkt(),
        ) as dst:
            for feature in src.filter(bbox=bbox):
                if not feature.get("geometry"):
                    continue
                geom = shape(feature["geometry"])
                if geom.is_empty:
                    continue
                projected = transform_geometry(transformer.transform, geom)
                if not projected.is_valid:
                    try:
                        projected = shapely.make_valid(projected)
                    except AttributeError:
                        projected = projected.buffer(0)
                clipped = projected.intersection(study_geom)
                if clipped.is_empty:
                    continue
                if not clipped.is_valid:
                    try:
                        clipped = shapely.make_valid(clipped)
                    except AttributeError:
                        clipped = clipped.buffer(0)
                if clipped.is_empty:
                    continue

                props = dict(feature["properties"])
                props["source_layer"] = source_layer
                props.update(metric_properties(clipped, geometry_type))
                if source_layer == "QLD_CADASTRE_DCDB" and props.get("LOTPLAN") is not None:
                    lotplans.add(str(props["LOTPLAN"]))
                dst.write({"geometry": mapping(clipped), "properties": props})
                count += 1
    return count, lotplans


def clip_layer(
    gdb_path: Path,
    source_layer: str,
    output_layer: str,
    study_area: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    with fiona.open(gdb_path, layer=source_layer) as src:
        bbox = bbox_in_layer_crs(study_area, src.crs)

    raw = gpd.read_file(gdb_path, layer=source_layer, bbox=bbox)
    if raw.empty:
        return gpd.GeoDataFrame(geometry=[], crs=study_area.crs)

    projected = fix_geometry(raw.to_crs(study_area.crs))
    if projected.empty:
        return projected

    clipped = gpd.clip(projected, study_area)
    clipped = fix_geometry(clipped)
    clipped = add_metric_columns(clipped, source_layer)
    return sanitize_for_gpkg(clipped)


def filter_bup_lot_table(gdb_path: Path, lotplans: set[str], output_csv: Path) -> int:
    rows = []
    with fiona.open(gdb_path, layer="QLD_CADASTRE_BUP_LOT") as src:
        columns = list(src.schema.get("properties", {}).keys())
        for feature in src:
            props = dict(feature["properties"])
            if str(props.get("LOTPLAN")) in lotplans:
                rows.append(props)
    out = pd.DataFrame(rows, columns=columns)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return len(out)


def _write_vector_file(
    data: gpd.GeoDataFrame,
    path: Path,
    *,
    driver: str,
    overwrite: bool,
) -> None:
    if path.exists():
        if not overwrite:
            raise SystemExit(f"output exists; pass --overwrite to replace: {path}")
        path.unlink()
    data.to_file(path, driver=driver)


def write_gpkg_layers(
    gdb_path: Path,
    output_gpkg: Path,
    study_area: gpd.GeoDataFrame,
    *,
    overwrite: bool,
) -> dict[str, int]:
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    if output_gpkg.exists():
        if not overwrite:
            raise SystemExit(f"output exists; pass --overwrite to replace: {output_gpkg}")
        output_gpkg.unlink()

    counts: dict[str, int] = {}
    sanitize_for_gpkg(study_area).to_file(output_gpkg, layer="study_area_boundary", driver="GPKG")
    counts["study_area_boundary"] = len(study_area)

    clipped_lotplans: set[str] = set()
    for source_layer, output_layer in LAYER_MAP.items():
        clipped = clip_layer(gdb_path, source_layer, output_layer, study_area)
        clipped.to_file(output_gpkg, layer=output_layer, driver="GPKG")
        counts[output_layer] = len(clipped)
        if output_layer == "dcdb_parcels" and "LOTPLAN" in clipped.columns:
            clipped_lotplans = set(clipped["LOTPLAN"].dropna().astype(str))
        print(f"{source_layer}: clipped={len(clipped)} -> {output_gpkg.name}:{output_layer}")

    bup_csv = output_gpkg.with_name("bup_lot_table_study_area.csv")
    counts["bup_lot_table"] = filter_bup_lot_table(gdb_path, clipped_lotplans, bup_csv)
    print(f"QLD_CADASTRE_BUP_LOT: filtered={counts['bup_lot_table']} -> {bup_csv.name}")
    return counts


def write_file_layers(
    gdb_path: Path,
    output_dir: Path,
    study_area: gpd.GeoDataFrame,
    *,
    output_format: str,
    overwrite: bool,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    extension = ".fgb" if output_format == "fgb" else ".geojson"
    driver = "FlatGeobuf" if output_format == "fgb" else "GeoJSON"

    counts: dict[str, int] = {}
    boundary_path = output_dir / f"study_area_boundary_epsg28356{extension}"
    _write_vector_file(
        sanitize_for_gpkg(study_area),
        boundary_path,
        driver=driver,
        overwrite=overwrite,
    )
    counts["study_area_boundary"] = len(study_area)

    clipped_lotplans: set[str] = set()
    for source_layer, output_layer in LAYER_MAP.items():
        output_path = output_dir / f"{output_layer}_epsg28356{extension}"
        count, lotplans = stream_clip_layer_to_file(
            gdb_path,
            source_layer,
            output_path,
            study_area,
            driver=driver,
            overwrite=overwrite,
        )
        counts[output_layer] = count
        if output_layer == "dcdb_parcels":
            clipped_lotplans = lotplans
        print(f"{source_layer}: clipped={count} -> {output_path.name}")

    bup_csv = output_dir / "bup_lot_table_study_area.csv"
    if bup_csv.exists() and not overwrite:
        raise SystemExit(f"output exists; pass --overwrite to replace: {bup_csv}")
    counts["bup_lot_table"] = filter_bup_lot_table(gdb_path, clipped_lotplans, bup_csv)
    print(f"QLD_CADASTRE_BUP_LOT: filtered={counts['bup_lot_table']} -> {bup_csv.name}")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clip Queensland cadastral layers to the full road study area and reproject to EPSG:28356."
    )
    parser.add_argument(
        "--cadastral-gdb",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "context" / "Cadastral.gdb",
    )
    parser.add_argument(
        "--roads",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "gis" / "roads" / "Roads_ExportFeatures.shp",
    )
    parser.add_argument(
        "--split-shapefile-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "split_shapefiles",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "context" / "study_area" / "cadastral",
    )
    parser.add_argument("--target-crs", default="EPSG:28356")
    parser.add_argument("--buffer-m", type=float, default=500.0)
    parser.add_argument(
        "--extent-source",
        choices=("roads", "sewer_splits"),
        default="roads",
        help="Use roads bbox for context studies or sewer split bbox for the anchor-based paper extent.",
    )
    parser.add_argument(
        "--output-format",
        choices=("fgb", "geojson", "gpkg"),
        default="fgb",
        help="FlatGeobuf is the default because this Fiona/GDAL environment can read FileGDB but fails GPKG writes.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gdb_path = args.cadastral_gdb if args.cadastral_gdb.is_absolute() else REPO_ROOT / args.cadastral_gdb
    roads_path = args.roads if args.roads.is_absolute() else REPO_ROOT / args.roads
    split_dir = (
        args.split_shapefile_dir
        if args.split_shapefile_dir.is_absolute()
        else REPO_ROOT / args.split_shapefile_dir
    )
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_gpkg = output_dir / "cadastral_study_area_epsg28356.gpkg"
    summary_path = output_dir / "cadastral_study_area_summary.json"

    if args.extent_source == "sewer_splits":
        study_area = sewer_split_study_area(split_dir, args.target_crs, args.buffer_m)
        extent_source_path = split_dir
    else:
        study_area = road_study_area(roads_path, args.target_crs, args.buffer_m)
        extent_source_path = roads_path

    if args.output_format == "gpkg":
        counts = write_gpkg_layers(
            gdb_path,
            output_gpkg,
            study_area,
            overwrite=args.overwrite,
        )
        output_dataset = output_gpkg
    else:
        counts = write_file_layers(
            gdb_path,
            output_dir,
            study_area,
            output_format=args.output_format,
            overwrite=args.overwrite,
        )
        output_dataset = output_dir

    summary = {
        "source_gdb": str(gdb_path.relative_to(REPO_ROOT)),
        "extent_source": args.extent_source,
        "extent_source_path": str(extent_source_path.relative_to(REPO_ROOT)),
        "target_crs": args.target_crs,
        "buffer_m": args.buffer_m,
        "output_format": args.output_format,
        "output": str(output_dataset.relative_to(REPO_ROOT)),
        "counts": counts,
        "study_area_bounds": [float(v) for v in study_area.total_bounds],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary -> {summary_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
