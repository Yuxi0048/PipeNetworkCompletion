"""Clip downloaded watercourse context and create completeness-gated AOI configs.

Workstream: Codex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from shapely.geometry import box

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipe_network_completion.anchor_free.aoi import clip_vector_to_aoi  # noqa: E402
from pipe_network_completion.anchor_free.config import (  # noqa: E402
    deep_update,
    load_anchor_free_config,
    write_resolved_config,
)


def _resolve(path: str | Path | None) -> Path | None:
    if path in (None, ""):
        return None
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _relative(path: Path | None) -> str:
    if path is None:
        return ""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(resolved)


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


def _read_context(path: Path, target_crs: str) -> gpd.GeoDataFrame:
    data = gpd.read_file(path)
    data = data[data.geometry.notna() & ~data.geometry.is_empty].copy()
    if data.crs is not None and str(data.crs) != str(target_crs):
        data = data.to_crs(target_crs)
    elif data.crs is None:
        data = data.set_crs(target_crs)
    return data


def _bounds_intersection(layers: list[gpd.GeoDataFrame]) -> list[float]:
    bounds = [float(v) for v in layers[0].total_bounds]
    for layer in layers[1:]:
        next_bounds = [float(v) for v in layer.total_bounds]
        bounds = [
            max(bounds[0], next_bounds[0]),
            max(bounds[1], next_bounds[1]),
            min(bounds[2], next_bounds[2]),
            min(bounds[3], next_bounds[3]),
        ]
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError("Watercourse source layer bounds do not overlap.")
    return bounds


def _find_source_config(config_dir: Path, aoi_id: str) -> Path:
    matches = sorted(config_dir.glob(f"*_{aoi_id}.yaml"))
    if not matches:
        matches = sorted(config_dir.glob(f"*{aoi_id}*.yaml"))
    if not matches:
        raise FileNotFoundError(f"No source config found for AOI {aoi_id!r} in {config_dir}")
    return matches[0]


def _clip_layer(
    layer: gpd.GeoDataFrame,
    aoi: gpd.GeoDataFrame,
    path: Path,
) -> tuple[Path, int, float]:
    clipped = clip_vector_to_aoi(layer, aoi)
    _write_geojson(clipped, path)
    length_or_area = 0.0
    if not clipped.empty:
        if clipped.geom_type.isin(["Polygon", "MultiPolygon"]).any():
            length_or_area = float(clipped.geometry.area.sum())
        else:
            length_or_area = float(clipped.geometry.length.sum())
    return path, int(len(clipped)), length_or_area


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process downloaded Brisbane watercourse context layers and write "
            "new AOI configs that use the layers only where source coverage is complete."
        )
    )
    parser.add_argument(
        "--selected-aois",
        type=Path,
        default=REPO_ROOT
        / "data"
        / "processed"
        / "aois"
        / "anchor_free_2km_gap500_115"
        / "selected_aois.geojson",
    )
    parser.add_argument(
        "--aoi-root",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "aois" / "anchor_free_2km_gap500_115",
    )
    parser.add_argument(
        "--source-config-dir",
        type=Path,
        default=REPO_ROOT / "configs" / "aois_2km_gap500_115",
    )
    parser.add_argument(
        "--output-config-dir",
        type=Path,
        default=REPO_ROOT / "configs" / "aois_2km_gap500_115_watercourses_complete",
    )
    parser.add_argument(
        "--processed-context-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "context" / "watercourses",
    )
    parser.add_argument(
        "--drainage-lines",
        type=Path,
        default=REPO_ROOT
        / "data"
        / "raw"
        / "context"
        / "watercourses"
        / "bcc_waterway_drainage_lines.gpkg",
    )
    parser.add_argument(
        "--corridor-centrelines",
        type=Path,
        default=REPO_ROOT
        / "data"
        / "raw"
        / "context"
        / "watercourses"
        / "bcc_cp14_waterway_corridor_centrelines.gpkg",
    )
    parser.add_argument(
        "--corridors",
        type=Path,
        default=REPO_ROOT
        / "data"
        / "raw"
        / "context"
        / "watercourses"
        / "bcc_cp14_waterway_corridors.gpkg",
    )
    parser.add_argument("--target-crs", default="EPSG:28356")
    parser.add_argument("--coverage-tolerance-m", type=float, default=0.0)
    parser.add_argument("--watercourse-buffer-m", type=float, default=100.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_aois = gpd.read_file(_resolve(args.selected_aois)).to_crs(args.target_crs)
    selected_aois = selected_aois[selected_aois.geometry.notna() & ~selected_aois.geometry.is_empty].copy()

    drainage_lines = _read_context(_resolve(args.drainage_lines), args.target_crs)
    corridor_centrelines = _read_context(_resolve(args.corridor_centrelines), args.target_crs)
    corridors = _read_context(_resolve(args.corridors), args.target_crs)
    layers = [drainage_lines, corridor_centrelines, corridors]
    coverage_bounds = _bounds_intersection(layers)
    coverage_area = box(*coverage_bounds).buffer(max(float(args.coverage_tolerance_m), 0.0))

    processed_context_dir = _resolve(args.processed_context_dir)
    aoi_root = _resolve(args.aoi_root)
    source_config_dir = _resolve(args.source_config_dir)
    output_config_dir = _resolve(args.output_config_dir)
    if processed_context_dir is None or aoi_root is None or source_config_dir is None or output_config_dir is None:
        raise ValueError("Output paths are required.")
    processed_context_dir.mkdir(parents=True, exist_ok=True)
    output_config_dir.mkdir(parents=True, exist_ok=True)

    union_geometry = (
        selected_aois.geometry.union_all()
        if hasattr(selected_aois.geometry, "union_all")
        else selected_aois.geometry.unary_union
    )
    study_area = gpd.GeoDataFrame(
        [{"geometry": union_geometry}],
        geometry="geometry",
        crs=selected_aois.crs,
    )
    study_paths = {
        "watercourse_drainage_lines": processed_context_dir / "bcc_waterway_drainage_lines_study_aois.geojson",
        "watercourse_corridor_centrelines": processed_context_dir
        / "bcc_cp14_waterway_corridor_centrelines_study_aois.geojson",
        "watercourse_corridors": processed_context_dir / "bcc_cp14_waterway_corridors_study_aois.geojson",
    }
    _write_geojson(clip_vector_to_aoi(drainage_lines, study_area), study_paths["watercourse_drainage_lines"])
    _write_geojson(
        clip_vector_to_aoi(corridor_centrelines, study_area),
        study_paths["watercourse_corridor_centrelines"],
    )
    _write_geojson(clip_vector_to_aoi(corridors, study_area), study_paths["watercourse_corridors"])

    records: list[dict] = []
    config_paths: list[str] = []
    for row in selected_aois.itertuples():
        aoi_id = str(row.aoi_id)
        split = str(row.split)
        aoi = gpd.GeoDataFrame(
            [{"aoi_id": aoi_id, "split": split, "geometry": row.geometry}],
            geometry="geometry",
            crs=selected_aois.crs,
        )
        complete = bool(coverage_area.covers(row.geometry))
        record = {
            "aoi_id": aoi_id,
            "split": split,
            "watercourse_context_complete": complete,
            "coverage_tolerance_m": float(args.coverage_tolerance_m),
        }
        if complete:
            aoi_dir = aoi_root / aoi_id
            drainage_path, drainage_count, drainage_length = _clip_layer(
                drainage_lines,
                aoi,
                aoi_dir / "watercourse_drainage_lines.geojson",
            )
            centreline_path, centreline_count, centreline_length = _clip_layer(
                corridor_centrelines,
                aoi,
                aoi_dir / "watercourse_corridor_centrelines.geojson",
            )
            corridor_path, corridor_count, corridor_area = _clip_layer(
                corridors,
                aoi,
                aoi_dir / "watercourse_corridors.geojson",
            )
            source_config = _find_source_config(source_config_dir, aoi_id)
            config = load_anchor_free_config(source_config)
            config = deep_update(
                config,
                {
                    "data": {
                        "watercourse_drainage_lines_path": _relative(drainage_path),
                        "watercourse_corridor_centrelines_path": _relative(centreline_path),
                        "watercourse_corridors_path": _relative(corridor_path),
                    },
                    "graph": {
                        "use_watercourses": True,
                        "watercourse_context_complete": True,
                        "watercourse_buffer_m": float(args.watercourse_buffer_m),
                    },
                    "aoi": {
                        "watercourse_context_complete": True,
                    },
                },
            )
            output_config = output_config_dir / source_config.name
            write_resolved_config(config, output_config)
            config_paths.append(_relative(output_config))
            record.update(
                {
                    "drainage_line_count": drainage_count,
                    "drainage_line_length_m": drainage_length,
                    "corridor_centreline_count": centreline_count,
                    "corridor_centreline_length_m": centreline_length,
                    "corridor_polygon_count": corridor_count,
                    "corridor_polygon_area_m2": corridor_area,
                    "config_path": _relative(output_config),
                }
            )
        else:
            record.update(
                {
                    "drainage_line_count": 0,
                    "drainage_line_length_m": 0.0,
                    "corridor_centreline_count": 0,
                    "corridor_centreline_length_m": 0.0,
                    "corridor_polygon_count": 0,
                    "corridor_polygon_area_m2": 0.0,
                    "config_path": "",
                }
            )
        records.append(record)

    report = pd.DataFrame(records)
    report_path = processed_context_dir / "watercourse_completeness_by_aoi.csv"
    report.to_csv(report_path, index=False)
    (output_config_dir / "complete_config_paths.txt").write_text(
        "\n".join(config_paths) + ("\n" if config_paths else ""),
        encoding="utf-8",
    )
    manifest = {
        "workstream": "Codex",
        "purpose": "watercourse/drainage context with AOI completeness gating",
        "target_crs": args.target_crs,
        "coverage_tolerance_m": float(args.coverage_tolerance_m),
        "watercourse_buffer_m": float(args.watercourse_buffer_m),
        "coverage_bounds": coverage_bounds,
        "n_aois": int(len(selected_aois)),
        "n_complete_aois": int(report["watercourse_context_complete"].sum()),
        "n_incomplete_aois": int((~report["watercourse_context_complete"]).sum()),
        "complete_split_counts": report.loc[
            report["watercourse_context_complete"], "split"
        ].value_counts().to_dict(),
        "excluded_incomplete_aois": report.loc[
            ~report["watercourse_context_complete"], "aoi_id"
        ].tolist(),
        "source_paths": {
            "drainage_lines": _relative(_resolve(args.drainage_lines)),
            "corridor_centrelines": _relative(_resolve(args.corridor_centrelines)),
            "corridors": _relative(_resolve(args.corridors)),
        },
        "study_area_paths": {key: _relative(path) for key, path in study_paths.items()},
        "config_dir": _relative(output_config_dir),
    }
    (processed_context_dir / "watercourse_context_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"Wrote completeness report: {_relative(report_path)}")
    print(f"Complete AOIs: {manifest['n_complete_aois']} / {manifest['n_aois']}")
    if manifest["excluded_incomplete_aois"]:
        print("Excluded incomplete AOIs:", ", ".join(manifest["excluded_incomplete_aois"]))
    print(f"Wrote gated configs: {_relative(output_config_dir)}")
    print("No model training was started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
