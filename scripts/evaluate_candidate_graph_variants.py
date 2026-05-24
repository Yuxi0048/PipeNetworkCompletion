"""Compare anchor-free candidate-support generation variants.

Workstream: Codex

This evaluates candidate graph representability only. It does not train a
classifier and does not use utility truth to generate candidates.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipe_network_completion.anchor_free.candidate_recall import (  # noqa: E402
    candidate_representability_metrics,
    candidate_source_summary,
)
from pipe_network_completion.anchor_free.candidate_variants import (  # noqa: E402
    SUPPORTED_VARIANTS,
    build_candidate_variant_lines,
)
from pipe_network_completion.anchor_free.config import load_anchor_free_config  # noqa: E402
from pipe_network_completion.anchor_free.hetero_road_graph import (  # noqa: E402
    build_hetero_road_graph,
)


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


def _expand_configs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(match) for match in glob.glob(str(_resolve(pattern))))
        if matches:
            paths.extend(matches)
        else:
            path = _resolve(pattern)
            if path is None or not path.exists():
                raise FileNotFoundError(pattern)
            paths.append(path)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            deduped.append(path)
            seen.add(resolved)
    return deduped


def _read_vector(path_value: str | Path | None) -> gpd.GeoDataFrame | None:
    path = _resolve(path_value)
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    data = gpd.read_file(path)
    return data[data.geometry.notna() & ~data.geometry.is_empty].copy()


def _read_optional_vector_many(path_values) -> gpd.GeoDataFrame | None:
    if path_values in (None, ""):
        return None
    values = path_values if isinstance(path_values, (list, tuple)) else [path_values]
    frames: list[gpd.GeoDataFrame] = []
    target_crs = None
    for value in values:
        frame = _read_vector(value)
        if frame is None:
            continue
        if target_crs is None:
            target_crs = frame.crs
        elif frame.crs and target_crs and str(frame.crs) != str(target_crs):
            frame = frame.to_crs(target_crs)
        frames.append(frame)
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=target_crs)


def _read_vector_many(path_values) -> gpd.GeoDataFrame:
    data = _read_optional_vector_many(path_values)
    if data is None:
        raise FileNotFoundError("No vector files configured.")
    return data


def _road_class_columns(config: dict) -> list[str]:
    graph_config = dict(config.get("graph", {}))
    columns = graph_config.get("road_class_columns")
    if isinstance(columns, str):
        return [columns]
    return list(columns or [])


def _needs_watercourses(variant: str) -> bool:
    return "watercourse" in variant or variant == "multi_support"


def _variant_keep_columns(config: dict) -> list[str]:
    columns = _road_class_columns(config)
    for column in (
        "candidate_source",
        "candidate_weight",
        "demand_u",
        "demand_v",
        "nearest_road_distance_m",
        "road_offset_distance_m",
        "road_offset_side",
    ):
        if column not in columns:
            columns.append(column)
    return columns


def _write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gdf.empty:
        path.write_text('{"type":"FeatureCollection","features":[]}\n', encoding="utf-8")
    else:
        path.write_text(gdf.to_json(drop_id=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate candidate support variants by truth-length recall. "
            "No model training is run."
        )
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["configs/aois_2km_gap500_115_watercourses_complete/*.yaml"],
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=SUPPORTED_VARIANTS,
        default=[
            "road",
            "hybrid",
            "road_offsets",
            "watercourses",
            "hybrid_watercourses",
            "multi_support",
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "candidate_recall" / "candidate_variant_comparison",
    )
    parser.add_argument("--buffers-m", nargs="+", type=float, default=[5, 10, 20, 30, 50])
    parser.add_argument("--sample-spacing-m", type=float, default=50.0)
    parser.add_argument("--offset-distances-m", nargs="+", type=float, default=[15.0, 30.0])
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--write-candidates", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    if output_dir is None:
        raise ValueError("output-dir is required.")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_paths = _expand_configs(args.configs)
    if int(args.max_configs) > 0:
        config_paths = config_paths[: int(args.max_configs)]

    rows: list[dict] = []
    source_rows: list[dict] = []
    skipped_rows: list[dict] = []
    for variant in args.variants:
        for config_path in config_paths:
            start = time.perf_counter()
            config = load_anchor_free_config(config_path)
            graph_config = dict(config.get("graph", {}))
            data_config = dict(config.get("data", {}))
            aoi_config = dict(config.get("aoi", {}))
            aoi_id = str(aoi_config.get("aoi_id", config_path.stem))
            split = str(aoi_config.get("split", "unknown"))

            if _needs_watercourses(variant) and not bool(
                graph_config.get("watercourse_context_complete", False)
            ):
                skipped_rows.append(
                    {
                        "variant": variant,
                        "aoi_id": aoi_id,
                        "split": split,
                        "reason": "watercourse_context_incomplete",
                    }
                )
                continue

            roads = _read_vector(data_config.get("roads_path"))
            if roads is None:
                raise FileNotFoundError("No roads_path configured.")
            buildings = (
                _read_optional_vector_many(data_config.get("buildings_path"))
                if graph_config.get("use_buildings", True) is not False
                else None
            )
            building_points = (
                _read_optional_vector_many(data_config.get("building_points_path"))
                if graph_config.get("use_building_points", True) is not False
                else None
            )
            drainage_lines = None
            corridor_centrelines = None
            if _needs_watercourses(variant):
                drainage_lines = _read_optional_vector_many(
                    data_config.get("watercourse_drainage_lines_path")
                )
                corridor_centrelines = _read_optional_vector_many(
                    data_config.get("watercourse_corridor_centrelines_path")
                )
            utility_truth = _read_vector_many(data_config.get("utility_truth_path"))

            candidate_result = build_candidate_variant_lines(
                roads,
                variant=variant,
                buildings_gdf=buildings,
                building_points_gdf=building_points,
                watercourse_drainage_lines_gdf=drainage_lines,
                watercourse_corridor_centrelines_gdf=corridor_centrelines,
                target_crs=graph_config.get("target_crs"),
                keep_columns=_road_class_columns(config),
                offset_distances_m=args.offset_distances_m,
                demand_cluster_grid_m=float(graph_config.get("demand_cluster_grid_m", 150.0)),
                nearest_road_max_distance_m=float(
                    graph_config.get("nearest_road_max_distance_m", 300.0)
                ),
                knn_k=int(graph_config.get("demand_knn_k", 3)),
                knn_max_distance_m=float(graph_config.get("demand_knn_max_distance_m", 500.0)),
            )
            graph = build_hetero_road_graph(
                candidate_result.candidates,
                target_crs=graph_config.get("target_crs"),
                snap_tolerance_m=float(graph_config.get("snap_tolerance_m", 1.0)),
                keep_columns=_variant_keep_columns(config),
            )
            graph.metadata.update(candidate_result.metadata)
            metrics = candidate_representability_metrics(
                graph,
                utility_truth,
                buffers_m=args.buffers_m,
                sample_spacing_m=float(args.sample_spacing_m),
            )
            row = {
                "variant": variant,
                "aoi_id": aoi_id,
                "split": split,
                "config_path": _relative(config_path),
                "n_intersections": int(len(graph.intersections)),
                "runtime_sec": round(time.perf_counter() - start, 3),
            }
            row.update(metrics)
            rows.append(row)
            for source, summary in candidate_source_summary(graph).items():
                source_rows.append(
                    {
                        "variant": variant,
                        "aoi_id": aoi_id,
                        "split": split,
                        "candidate_source": source,
                        "count": int(summary["count"]),
                        "length_m": float(summary["length_m"]),
                    }
                )
            if args.write_candidates:
                candidates = graph.road_segments.copy()
                candidates["aoi_id"] = aoi_id
                candidates["split"] = split
                candidates["variant"] = variant
                _write_geojson(
                    candidates,
                    output_dir / "candidate_graphs" / variant / f"{aoi_id}.geojson",
                )

            if not args.quiet:
                print(
                    f"{variant} {aoi_id} ({split}): "
                    f"segments={int(metrics.get('candidate_count', 0))} "
                    f"recall_10m={metrics.get('recall_10m', float('nan')):.3f} "
                    f"recall_50m={metrics.get('recall_50m', float('nan')):.3f}"
                )

    results = pd.DataFrame(rows)
    source_summary = pd.DataFrame(source_rows)
    skipped = pd.DataFrame(skipped_rows)
    results.to_csv(output_dir / "candidate_variant_recall_by_aoi.csv", index=False)
    source_summary.to_csv(output_dir / "candidate_variant_source_summary.csv", index=False)
    skipped.to_csv(output_dir / "candidate_variant_skipped.csv", index=False)

    variant_summary_rows: list[dict] = []
    for variant, group in results.groupby("variant"):
        weights = group["truth_total_length_sampled_m"].astype(float)
        total_truth = float(weights.sum())
        summary_row = {
            "variant": variant,
            "n_aois": int(len(group)),
            "candidate_count_mean": float(group["candidate_count"].mean()),
            "candidate_total_length_km": float(group["candidate_total_length_m"].sum() / 1000.0),
            "truth_total_length_km": float(total_truth / 1000.0),
            "runtime_sec_total": float(group["runtime_sec"].sum()),
        }
        for column in sorted(c for c in group.columns if c.startswith("recall_")):
            summary_row[column] = (
                float((group[column].astype(float) * weights).sum() / total_truth)
                if total_truth > 0
                else float("nan")
            )
        variant_summary_rows.append(summary_row)
    variant_summary = pd.DataFrame(variant_summary_rows).sort_values("recall_10m", ascending=False)
    variant_summary.to_csv(output_dir / "candidate_variant_summary.csv", index=False)

    manifest = {
        "workstream": "Codex",
        "description": "candidate-support variant representability; no model training",
        "n_configs": len(config_paths),
        "variants": list(args.variants),
        "buffers_m": [float(value) for value in args.buffers_m],
        "sample_spacing_m": float(args.sample_spacing_m),
        "offset_distances_m": [float(value) for value in args.offset_distances_m],
        "output_dir": _relative(output_dir),
    }
    (output_dir / "candidate_variant_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote {_relative(output_dir / 'candidate_variant_summary.csv')}")
    print("No model training was started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
