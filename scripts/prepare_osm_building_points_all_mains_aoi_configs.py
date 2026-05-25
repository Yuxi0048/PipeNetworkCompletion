"""Prepare AOI configs with OSM building points and complete sewer-main truth.

Workstream: Codex

This creates an additive experiment input set. It does not modify the existing
AOI folders or configs. "Complete sewer mains" means gravity mains plus pressure
mains; service laterals and vent pipes are excluded from the main-line target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


MAIN_TRUTH_FILES = [
    "SewerGravityMa_ExportFeature1.shp",
    "SewerGravityMa_ExportFeature2.shp",
    "SewerPressureM_ExportFeature.shp",
]


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


def _write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gdf.empty:
        path.write_text('{"type":"FeatureCollection","features":[]}\n', encoding="utf-8")
    else:
        gdf.to_file(path, driver="GeoJSON")


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _load_main_truth(sewer_root: Path, target_crs) -> gpd.GeoDataFrame:
    frames = []
    for filename in MAIN_TRUTH_FILES:
        path = sewer_root / filename
        layer = _read_layer(path, target_crs)
        if layer.empty:
            continue
        layer = layer.copy()
        layer["truth_source_file"] = filename
        if filename.startswith("SewerGravity"):
            layer["main_system"] = "gravity_main"
        elif filename.startswith("SewerPressure"):
            layer["main_system"] = "pressure_main"
        else:
            layer["main_system"] = "sewer_main"
        frames.append(layer)
    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config-dir",
        type=Path,
        default=Path("configs/aois_2km_gap500_115"),
    )
    parser.add_argument(
        "--output-config-dir",
        type=Path,
        default=Path("configs/aois_2km_gap500_115_osm_bpoints_all_mains"),
    )
    parser.add_argument(
        "--aoi-root",
        type=Path,
        default=Path("data/processed/aois/anchor_free_2km_gap500_115"),
    )
    parser.add_argument(
        "--output-aoi-root",
        type=Path,
        default=Path("data/processed/aois/anchor_free_2km_gap500_115_osm_bpoints_all_mains"),
    )
    parser.add_argument(
        "--selected-aois",
        type=Path,
        default=Path("data/processed/aois/anchor_free_2km_gap500_115/selected_aois.geojson"),
    )
    parser.add_argument(
        "--osm-building-points",
        type=Path,
        default=Path("data/processed/context/study_area/osm_building_anchor_points.geojson"),
    )
    parser.add_argument("--sewer-root", type=Path, default=Path("data/raw/gis/sewer"))
    parser.add_argument("--target-crs", default="EPSG:28356")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = gpd.read_file(args.selected_aois).to_crs(args.target_crs)
    osm_points = _read_layer(args.osm_building_points, args.target_crs)
    main_truth = _load_main_truth(args.sewer_root, args.target_crs)
    if osm_points.empty:
        raise FileNotFoundError(f"No OSM building points loaded from {args.osm_building_points}")
    if main_truth.empty:
        raise FileNotFoundError(f"No sewer-main truth loaded from {args.sewer_root}")

    args.output_config_dir.mkdir(parents=True, exist_ok=True)
    args.output_aoi_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for aoi_row in selected.itertuples(index=False):
        aoi_id = str(aoi_row.aoi_id)
        aoi_geom = aoi_row.geometry
        base_config_path = args.base_config_dir / f"anchor_free_2km_gap500_aoi115_{aoi_id}.yaml"
        if not base_config_path.exists():
            raise FileNotFoundError(base_config_path)

        out_aoi_dir = args.output_aoi_root / aoi_id
        out_aoi_dir.mkdir(parents=True, exist_ok=True)
        osm_clip = _clip_layer(osm_points, aoi_geom)
        truth_clip = _clip_layer(main_truth, aoi_geom)
        osm_path = out_aoi_dir / "osm_building_points.geojson"
        truth_path = out_aoi_dir / "utility_truth_all_sewer_mains.geojson"
        _write_geojson(osm_clip, osm_path)
        _write_geojson(truth_clip, truth_path)

        config = json.loads(base_config_path.read_text(encoding="utf-8"))
        config.setdefault("data", {})["building_points_path"] = _relative(osm_path, Path.cwd())
        config["data"]["utility_truth_path"] = _relative(truth_path, Path.cwd())
        config.setdefault("graph", {})["use_building_points"] = True
        config.setdefault("aoi", {})["source"] = _relative(args.aoi_root, Path.cwd())
        config["experiment_name"] = f"anchor_free_2km_gap500_aoi115_osm_bpoints_all_mains_{aoi_id}"
        config["truth_definition"] = {
            "target": "complete_sewer_mains",
            "included_layers": MAIN_TRUTH_FILES,
            "excluded_layers": [
                "SewerService_ExportFeatures.shp",
                "SewerVentPipe_ExportFeatures.shp",
            ],
            "note": (
                "Ground-truth sewer geometry is used only for label generation and evaluation, "
                "not as an input feature."
            ),
        }
        out_config_path = args.output_config_dir / base_config_path.name
        out_config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        rows.append(
            {
                "aoi_id": aoi_id,
                "split": str(getattr(aoi_row, "split", "")),
                "osm_building_points": int(len(osm_clip)),
                "all_main_truth_features": int(len(truth_clip)),
                "all_main_truth_length_m": float(truth_clip.length.sum()) if not truth_clip.empty else 0.0,
                "osm_points_path": _relative(osm_path, Path.cwd()),
                "truth_path": _relative(truth_path, Path.cwd()),
                "config_path": _relative(out_config_path, Path.cwd()),
            }
        )
        print(
            f"{aoi_id}: osm_points={len(osm_clip)} all_main_truth_features={len(truth_clip)} "
            f"truth_km={float(truth_clip.length.sum()) / 1000.0:.2f}",
            flush=True,
        )

    summary = pd.DataFrame(rows)
    summary_path = args.output_aoi_root / "osm_bpoints_all_mains_summary.csv"
    summary.to_csv(summary_path, index=False)
    (args.output_aoi_root / "truth_definition.json").write_text(
        json.dumps(
            {
                "target": "complete_sewer_mains",
                "included_layers": MAIN_TRUTH_FILES,
                "excluded_layers": [
                    "SewerService_ExportFeatures.shp",
                    "SewerVentPipe_ExportFeatures.shp",
                ],
                "n_aois": int(len(summary)),
                "total_truth_length_m": float(summary["all_main_truth_length_m"].sum()),
                "total_osm_building_points": int(summary["osm_building_points"].sum()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote configs: {args.output_config_dir}")
    print(f"Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
