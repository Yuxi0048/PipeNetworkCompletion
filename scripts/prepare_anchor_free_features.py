"""Prepare anchor-free road-segment features without training a model.

Workstream: Codex
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipe_network_completion.anchor_free.config import (  # noqa: E402
    load_anchor_free_config,
    write_resolved_config,
)
from pipe_network_completion.anchor_free.baseline import (  # noqa: E402
    make_buffer_invariant_splits,
    make_stratified_edge_splits,  # noqa: F401 — kept for backward compatibility
)
from pipe_network_completion.anchor_free.features import (  # noqa: E402
    assert_no_anchor_features,
    build_intersection_features,
    build_road_segment_features,
    standardize_features,
)
from pipe_network_completion.anchor_free.labels import (  # noqa: E402
    label_road_segments_from_utility_lines,
)
from pipe_network_completion.anchor_free.hetero_road_graph import (  # noqa: E402
    build_hetero_road_graph,
)
from pipe_network_completion.anchor_free.hybrid_candidate_graph import (  # noqa: E402
    build_hybrid_candidate_lines,
)
from pipe_network_completion.anchor_free.candidate_variants import (  # noqa: E402
    build_candidate_variant_lines,
)


def _resolve_path(path_value: str | Path | None) -> Path | None:
    if path_value in (None, ""):
        return None
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def _read_vector(path_value: str | Path | None) -> gpd.GeoDataFrame | None:
    path = _resolve_path(path_value)
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    return gpd.read_file(path)


def _read_vector_many(
    path_values: str | Path | Iterable[str | Path] | None,
) -> gpd.GeoDataFrame | None:
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


def _write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    if gdf.empty:
        path.write_text('{"type":"FeatureCollection","features":[]}\n', encoding="utf-8")
    else:
        path.write_text(gdf.to_json(drop_id=True), encoding="utf-8")


def _default_output_dir(config: dict, output_root: Path | None) -> Path:
    root = output_root if output_root is not None else REPO_ROOT / "outputs"
    return root / str(config.get("experiment_name", "anchor_free_features")) / "features"


def _source_metadata(path_values) -> list[str]:
    if path_values in (None, ""):
        return []
    values = path_values if isinstance(path_values, (list, tuple)) else [path_values]
    return [str(_resolve_path(value)) for value in values if _resolve_path(value) is not None]


def _label_suffix(label_buffer_m: float) -> str:
    text = f"{float(label_buffer_m):g}".replace(".", "p")
    return f"{text}m"


def _split_frame(
    unit_ids: np.ndarray,
    *,
    train_index: np.ndarray,
    val_index: np.ndarray,
    test_index: np.ndarray,
    id_column: str = "segment_id",
) -> pd.DataFrame:
    split = np.full(len(unit_ids), "unused", dtype=object)
    split[np.asarray(train_index, dtype=int)] = "train"
    split[np.asarray(val_index, dtype=int)] = "val"
    split[np.asarray(test_index, dtype=int)] = "test"
    return pd.DataFrame({id_column: unit_ids.astype(int), "split": split})


def _label_summary(
    *,
    label_buffer_m: float,
    label_overlap_threshold: float,
    labels: pd.DataFrame,
    graph_segments: gpd.GeoDataFrame,
    train_index: np.ndarray,
    val_index: np.ndarray,
    test_index: np.ndarray,
) -> dict:
    y = labels["y"].to_numpy(dtype=int)
    segment_lengths = (
        graph_segments.sort_values("segment_id")["length_m"].to_numpy(dtype=float)
    )
    positive_length = float(segment_lengths[y == 1].sum())
    summary = {
        "label_buffer_m": float(label_buffer_m),
        "label_overlap_threshold": float(label_overlap_threshold),
        "n_segments": int(len(y)),
        "n_positive_segments": int(y.sum()),
        "n_negative_segments": int(len(y) - y.sum()),
        "positive_segment_fraction": float(y.mean()) if len(y) else 0.0,
        "positive_length_m": positive_length,
        "negative_length_m": float(segment_lengths[y == 0].sum()),
        "train_segments": int(len(train_index)),
        "val_segments": int(len(val_index)),
        "test_segments": int(len(test_index)),
        "train_positive_fraction": float(y[train_index].mean()) if len(train_index) else 0.0,
        "val_positive_fraction": float(y[val_index].mean()) if len(val_index) else 0.0,
        "test_positive_fraction": float(y[test_index].mean()) if len(test_index) else 0.0,
    }
    return summary


def _agreement_metrics(reference_y: np.ndarray, candidate_y: np.ndarray) -> dict[str, float]:
    reference_y = np.asarray(reference_y, dtype=int)
    candidate_y = np.asarray(candidate_y, dtype=int)
    tp = float(((reference_y == 1) & (candidate_y == 1)).sum())
    fp = float(((reference_y == 0) & (candidate_y == 1)).sum())
    fn = float(((reference_y == 1) & (candidate_y == 0)).sum())
    tn = float(((reference_y == 0) & (candidate_y == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {
        "true_positive_edges": tp,
        "false_positive_edges": fp,
        "false_negative_edges": fn,
        "true_negative_edges": tn,
        "precision_vs_reference": precision,
        "recall_vs_reference": recall,
        "f1_vs_reference": f1,
        "jaccard_vs_reference": jaccard,
        "agreement_fraction": float((reference_y == candidate_y).mean()) if len(reference_y) else 0.0,
    }


def _configured_label_buffers(
    graph_config: dict,
    explicit: Iterable[float] | None,
) -> list[float]:
    if explicit is not None:
        values = [float(value) for value in explicit]
    else:
        raw_values = graph_config.get("training_ready_label_buffers_m")
        if raw_values is None:
            raw_values = [graph_config.get("label_buffer_m", 10.0)]
        elif not isinstance(raw_values, (list, tuple)):
            raw_values = [raw_values]
        values = [float(value) for value in raw_values]
    deduped: list[float] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def prepare_features(
    config: dict,
    *,
    output_dir: Path,
    with_labels: bool = False,
    training_ready: bool = False,
    label_buffers_m: Iterable[float] | None = None,
    write_geojson: bool = True,
) -> dict:
    start = time.perf_counter()
    graph_config = dict(config.get("graph", {}))
    data_config = dict(config.get("data", {}))

    roads = _read_vector(data_config.get("roads_path"))
    if roads is None:
        raise FileNotFoundError("No roads_path configured.")

    buildings = None
    if graph_config.get("use_buildings", True) is not False:
        buildings = _read_vector_many(data_config.get("buildings_path"))

    building_points = None
    if graph_config.get("use_building_points", True) is not False:
        building_points = _read_vector_many(data_config.get("building_points_path"))

    built_up = None
    if graph_config.get("use_built_up", True) is not False:
        built_up = _read_vector_many(data_config.get("built_up_path"))

    dem_path = None
    if graph_config.get("use_dem", True) is not False:
        dem_path = _resolve_path(data_config.get("dem_path"))

    use_watercourses = bool(graph_config.get("use_watercourses", False))
    watercourse_complete = bool(graph_config.get("watercourse_context_complete", False))
    watercourse_drainage_lines = None
    watercourse_corridor_centrelines = None
    watercourse_corridors = None
    if use_watercourses and watercourse_complete:
        watercourse_drainage_lines = _read_vector_many(
            data_config.get("watercourse_drainage_lines_path")
        )
        watercourse_corridor_centrelines = _read_vector_many(
            data_config.get("watercourse_corridor_centrelines_path")
        )
        watercourse_corridors = _read_vector_many(
            data_config.get("watercourse_corridors_path")
        )

    road_class_columns = graph_config.get("road_class_columns")
    if isinstance(road_class_columns, str):
        keep_columns = [road_class_columns]
        feature_road_class_columns = [road_class_columns]
    else:
        keep_columns = list(road_class_columns or [])
        feature_road_class_columns = list(road_class_columns or [])
    candidate_graph_type = str(graph_config.get("candidate_graph_type", "road")).lower()
    if candidate_graph_type == "hybrid":
        for column in (
            "candidate_source",
            "candidate_weight",
            "demand_u",
            "demand_v",
            "nearest_road_distance_m",
        ):
            if column not in keep_columns:
                keep_columns.append(column)
        if "candidate_source" not in feature_road_class_columns:
            feature_road_class_columns.append("candidate_source")
        hybrid = build_hybrid_candidate_lines(
            roads,
            buildings_gdf=buildings,
            building_points_gdf=building_points,
            target_crs=graph_config.get("target_crs"),
            keep_columns=road_class_columns,
            demand_cluster_grid_m=float(graph_config.get("demand_cluster_grid_m", 150.0)),
            nearest_road_max_distance_m=float(
                graph_config.get("nearest_road_max_distance_m", 300.0)
            ),
            knn_k=int(graph_config.get("demand_knn_k", 3)),
            knn_max_distance_m=float(graph_config.get("demand_knn_max_distance_m", 500.0)),
            include_road_backbone=bool(graph_config.get("include_road_backbone", True)),
            include_building_access=bool(
                graph_config.get("include_building_access", True)
            ),
            include_demand_knn=bool(graph_config.get("include_demand_knn", True)),
            include_demand_mst=bool(graph_config.get("include_demand_mst", True)),
        )
        roads = hybrid.candidates
    elif candidate_graph_type == "road_offsets":
        for column in (
            "candidate_id",
            "candidate_source",
            "candidate_weight",
            "demand_u",
            "demand_v",
            "nearest_road_distance_m",
            "road_offset_distance_m",
            "road_offset_side",
            "source_index",
        ):
            if column not in keep_columns:
                keep_columns.append(column)
        for column in ("candidate_source", "road_offset_side"):
            if column not in feature_road_class_columns:
                feature_road_class_columns.append(column)
        road_offsets = build_candidate_variant_lines(
            roads,
            variant="road_offsets",
            target_crs=graph_config.get("target_crs"),
            keep_columns=road_class_columns,
            offset_distances_m=graph_config.get("offset_distances_m", [15.0, 30.0]),
        )
        roads = road_offsets.candidates
    elif candidate_graph_type != "road":
        raise ValueError(
            f"Unsupported graph.candidate_graph_type={candidate_graph_type!r}; "
            "expected 'road', 'hybrid', or 'road_offsets'."
        )

    graph = build_hetero_road_graph(
        roads,
        target_crs=graph_config.get("target_crs"),
        snap_tolerance_m=float(graph_config.get("snap_tolerance_m", 1.0)),
        keep_columns=keep_columns,
    )

    features = build_road_segment_features(
        graph,
        buildings_gdf=buildings,
        building_points_gdf=building_points,
        built_up_gdf=built_up,
        watercourse_drainage_lines_gdf=watercourse_drainage_lines,
        watercourse_corridor_centrelines_gdf=watercourse_corridor_centrelines,
        watercourse_corridors_gdf=watercourse_corridors,
        dem_path=dem_path,
        road_class_columns=feature_road_class_columns,
        building_buffer_m=float(graph_config.get("building_buffer_m", 50.0)),
        building_point_buffer_m=float(graph_config.get("building_point_buffer_m", 50.0)),
        built_up_buffer_m=float(graph_config.get("built_up_buffer_m", 50.0)),
        watercourse_buffer_m=float(graph_config.get("watercourse_buffer_m", 100.0)),
        road_density_buffer_m=float(graph_config.get("road_density_buffer_m", 100.0)),
        dem_sample_spacing_m=float(graph_config.get("dem_sample_spacing_m", 30.0)),
        dem_max_samples_per_edge=int(graph_config.get("dem_max_samples_per_edge", 64)),
    )
    assert_no_anchor_features(features.feature_names)
    intersection_features = build_intersection_features(
        graph,
        include_coords=bool(config.get("model", {}).get("include_node_coords", True)),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_frame = features.features.reset_index()
    feature_frame.to_csv(output_dir / "road_segment_features.csv", index=False)
    intersection_features.features.reset_index().to_csv(
        output_dir / "intersection_features.csv",
        index=False,
    )

    if write_geojson:
        segment_features = graph.road_segments.merge(
            feature_frame,
            on="segment_id",
            how="left",
        )
        _write_geojson(
            segment_features,
            output_dir / "road_segment_features.geojson",
        )
        _write_geojson(graph.intersections, output_dir / "road_graph_intersections.geojson")

    labels_written: list[str] = []
    train_ready_written: list[str] = []
    label_summary = {}
    if with_labels:
        utility_truth = _read_vector_many(data_config.get("utility_truth_path"))
        if utility_truth is None:
            raise FileNotFoundError("No utility_truth_path configured for label export.")
        overlap_threshold = float(graph_config.get("label_overlap_threshold", 0.25))
        buffer_values = _configured_label_buffers(graph_config, label_buffers_m)
        all_label_summaries: dict[str, dict] = {}
        label_y_by_suffix: dict[str, np.ndarray] = {}

        # AR-AF-A.3: derive the train/val/test split ONCE from segment_ids so it
        # is identical across every label buffer in `buffer_values`. This is
        # the M6 fix from anchor_free_leakage_audit_codex.md and is required
        # for 10 m vs 5 m comparisons to be apples-to-apples. See
        # docs/research_notes/audit_followup_implementation_plan.md §4.
        train_index, val_index, test_index = make_buffer_invariant_splits(
            features.segment_ids,
            seed=int(config.get("seed", 42)),
        )

        for label_buffer_m in buffer_values:
            suffix = _label_suffix(label_buffer_m)
            labels = label_road_segments_from_utility_lines(
                graph,
                utility_truth,
                label_buffer_m=float(label_buffer_m),
                label_overlap_threshold=overlap_threshold,
            )
            labels_path = output_dir / f"road_segment_labels_{suffix}.csv"
            label_columns = labels.labels.drop(columns="geometry", errors="ignore")
            label_columns.to_csv(labels_path, index=False)
            labels_written.append(labels_path.name)
            label_y_by_suffix[suffix] = labels.y
            split_table = _split_frame(
                features.segment_ids,
                train_index=train_index,
                val_index=val_index,
                test_index=test_index,
                id_column="segment_id",
            )
            split_path = output_dir / f"train_val_test_split_{suffix}.csv"
            split_table.to_csv(split_path, index=False)

            if training_ready:
                scaled_features, mean, std = standardize_features(
                    features.features,
                    train_index=train_index,
                )
                standardized_path = output_dir / f"road_segment_features_standardized_{suffix}.csv"
                scaled_features.reset_index().to_csv(standardized_path, index=False)
                scaling_path = output_dir / f"feature_scaling_{suffix}.csv"
                pd.DataFrame(
                    {
                        "feature": features.feature_names,
                        "mean": mean.reindex(features.feature_names).to_numpy(dtype=float),
                        "std": std.reindex(features.feature_names).to_numpy(dtype=float),
                    }
                ).to_csv(scaling_path, index=False)

                training_table = (
                    feature_frame.merge(label_columns[["segment_id", "y"]], on="segment_id", how="left")
                    .merge(split_table, on="segment_id", how="left")
                )
                ordered_columns = ["segment_id", "split", "y"] + features.feature_names
                training_table = training_table[ordered_columns]
                training_path = output_dir / f"road_segment_training_table_{suffix}.csv"
                training_table.to_csv(training_path, index=False)
                train_ready_written.extend(
                    [split_path.name, standardized_path.name, scaling_path.name, training_path.name]
                )
            else:
                train_ready_written.append(split_path.name)

            all_label_summaries[suffix] = _label_summary(
                label_buffer_m=float(label_buffer_m),
                label_overlap_threshold=overlap_threshold,
                labels=label_columns,
                graph_segments=graph.road_segments,
                train_index=train_index,
                val_index=val_index,
                test_index=test_index,
            )

        comparison_rows = []
        if len(buffer_values) > 1:
            reference_suffix = _label_suffix(buffer_values[0])
            reference_y = label_y_by_suffix[reference_suffix]
            for label_buffer_m in buffer_values[1:]:
                suffix = _label_suffix(label_buffer_m)
                row = {
                    "reference_label_buffer": reference_suffix,
                    "candidate_label_buffer": suffix,
                }
                row.update(_agreement_metrics(reference_y, label_y_by_suffix[suffix]))
                comparison_rows.append(row)
            comparison_path = output_dir / "label_buffer_comparison.csv"
            pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
            train_ready_written.append(comparison_path.name)

        label_summary = {
            "label_summaries": all_label_summaries,
            "label_buffer_comparisons": comparison_rows,
        }

    metadata = {
        "workstream": "Codex",
        "training_started": False,
        "runtime_sec": time.perf_counter() - start,
        "candidate_graph_type": candidate_graph_type,
        "candidate_by_source": {
            str(source): {
                "count": int(len(group)),
                "length_m": float(group.geometry.length.sum()),
            }
            for source, group in graph.road_segments.groupby("candidate_source")
        }
        if "candidate_source" in graph.road_segments.columns
        else {},
        "n_intersections": int(len(graph.intersections)),
        "n_road_segments": int(len(graph.road_segments)),
        "n_prediction_units": int(len(graph.road_segments)),
        "n_features": int(len(features.feature_names)),
        "n_intersection_features": int(len(intersection_features.feature_names)),
        "feature_names": features.feature_names,
        "intersection_feature_names": intersection_features.feature_names,
        "crs": str(graph.crs),
        "source_paths": {
            "roads": _source_metadata(data_config.get("roads_path")),
            "buildings": _source_metadata(data_config.get("buildings_path")),
            "building_points": _source_metadata(data_config.get("building_points_path")),
            "built_up": _source_metadata(data_config.get("built_up_path")),
            "dem": _source_metadata(data_config.get("dem_path")),
            "utility_truth_labels_only": _source_metadata(data_config.get("utility_truth_path"))
            if with_labels
            else [],
        },
        "labels_written": labels_written,
        "training_ready_written": train_ready_written,
        "training_ready": bool(training_ready),
        "model_feature_columns_file": "feature_columns.json",
        **label_summary,
    }
    (output_dir / "feature_columns.json").write_text(
        json.dumps(features.feature_names, indent=2),
        encoding="utf-8",
    )
    (output_dir / "feature_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_resolved_config(config, output_dir / "config_resolved.yaml")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare anchor-free road-segment features without model training."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "anchor_free_real_context_features.yaml",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--with-labels", action="store_true")
    parser.add_argument(
        "--training-ready",
        action="store_true",
        help="Write label-aligned training tables, splits, and train-split scaling stats.",
    )
    parser.add_argument(
        "--label-buffer-meters",
        nargs="+",
        type=float,
        default=None,
        help="One or more label/evaluation buffers in meters, e.g. 10 5.",
    )
    parser.add_argument("--no-geojson", action="store_true")
    args = parser.parse_args()

    config = load_anchor_free_config(args.config)
    output_dir = _default_output_dir(config, args.output_root)
    print("Preparing anchor-free features only; no model training will be started.")
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")

    metadata = prepare_features(
        config,
        output_dir=output_dir,
        with_labels=args.with_labels or args.training_ready,
        training_ready=args.training_ready,
        label_buffers_m=args.label_buffer_meters,
        write_geojson=not args.no_geojson,
    )
    print(
        "Prepared "
        f"{metadata['n_features']} features for {metadata['n_road_segments']} road segments "
        f"in {metadata['runtime_sec']:.1f}s."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
