"""Train anchor-free GNN with AOI-block train/val/test splits.

Workstream: Codex
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipe_network_completion.anchor_free.candidate_recall import (  # noqa: E402
    candidate_representability_metrics,
    candidate_source_summary,
)
from pipe_network_completion.anchor_free.config import load_anchor_free_config  # noqa: E402
from pipe_network_completion.anchor_free.decoder import decode_segment_network  # noqa: E402
from pipe_network_completion.anchor_free.evaluation import (  # noqa: E402
    compute_edge_metrics,
)
from pipe_network_completion.anchor_free.features import (  # noqa: E402
    RoadSegmentFeatureTable,
    assert_no_anchor_features,
    build_intersection_features,
    build_road_segment_features,
    standardize_features,
)
from pipe_network_completion.anchor_free.hetero_road_graph import HeteroRoadGraph  # noqa: E402
from pipe_network_completion.anchor_free.labels import label_road_segments_from_utility_lines  # noqa: E402
from pipe_network_completion.anchor_free.model import (  # noqa: E402
    build_hetero_pyg_data,
    torch_device_report,
    train_hetero_road_gnn,
)
from pipe_network_completion.anchor_free.pipeline import prepare_anchor_free_inputs  # noqa: E402


@dataclass(frozen=True)
class AOIPart:
    aoi_id: str
    split: str
    config_path: Path
    graph: HeteroRoadGraph
    features: pd.DataFrame
    labels: gpd.GeoDataFrame
    utility_truth: gpd.GeoDataFrame


def _resolve(path: str | Path) -> Path:
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
            if not path.exists():
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


def _read_optional_vector_many(path_values) -> gpd.GeoDataFrame | None:
    if path_values in (None, ""):
        return None
    values = path_values if isinstance(path_values, (list, tuple)) else [path_values]
    frames: list[gpd.GeoDataFrame] = []
    target_crs = None
    for value in values:
        path = _resolve(value)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = gpd.read_file(path)
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


def _feature_road_class_columns(config: dict, graph: HeteroRoadGraph):
    graph_config = dict(config.get("graph", {}))
    columns = graph_config.get("road_class_columns")
    if isinstance(columns, str):
        columns = [columns]
    else:
        columns = list(columns or [])
    candidate_graph_type = graph.metadata.get("candidate_graph_type")
    if candidate_graph_type in {"hybrid", "road_offsets"} and "candidate_source" not in columns:
        columns.append("candidate_source")
    if candidate_graph_type == "road_offsets" and "road_offset_side" not in columns:
        columns.append("road_offset_side")
    return columns


def _prepare_part(
    config_path: Path,
    *,
    candidate_graph_type: str | None,
    include_node_coords: bool | None,
) -> tuple[AOIPart, dict]:
    config = load_anchor_free_config(config_path)
    if candidate_graph_type is not None:
        config.setdefault("graph", {})["candidate_graph_type"] = candidate_graph_type
    if include_node_coords is not None:
        config.setdefault("model", {})["include_node_coords"] = bool(include_node_coords)

    graph_config = dict(config.get("graph", {}))
    graph, buildings, building_points, built_up, dem_path, utility_truth = prepare_anchor_free_inputs(
        config
    )
    if graph_config.get("use_buildings", True) is False:
        buildings = None
    if graph_config.get("use_building_points", True) is False:
        building_points = None
    if graph_config.get("use_built_up", True) is False:
        built_up = None
    if graph_config.get("use_dem", True) is False:
        dem_path = None
    data_config = dict(config.get("data", {}))
    use_watercourses = bool(graph_config.get("use_watercourses", False))
    watercourse_complete = bool(graph_config.get("watercourse_context_complete", False))
    watercourse_drainage_lines = None
    watercourse_corridor_centrelines = None
    watercourse_corridors = None
    if use_watercourses and watercourse_complete:
        watercourse_drainage_lines = _read_optional_vector_many(
            data_config.get("watercourse_drainage_lines_path")
        )
        watercourse_corridor_centrelines = _read_optional_vector_many(
            data_config.get("watercourse_corridor_centrelines_path")
        )
        watercourse_corridors = _read_optional_vector_many(
            data_config.get("watercourse_corridors_path")
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
        road_class_columns=_feature_road_class_columns(config, graph),
        building_buffer_m=float(graph_config.get("building_buffer_m", 50.0)),
        building_point_buffer_m=float(graph_config.get("building_point_buffer_m", 50.0)),
        built_up_buffer_m=float(graph_config.get("built_up_buffer_m", 50.0)),
        watercourse_buffer_m=float(graph_config.get("watercourse_buffer_m", 100.0)),
        road_density_buffer_m=float(graph_config.get("road_density_buffer_m", 100.0)),
        dem_sample_spacing_m=float(graph_config.get("dem_sample_spacing_m", 30.0)),
        dem_max_samples_per_edge=int(graph_config.get("dem_max_samples_per_edge", 64)),
    )
    labels = label_road_segments_from_utility_lines(
        graph,
        utility_truth,
        label_buffer_m=float(graph_config.get("label_buffer_m", 10.0)),
        label_overlap_threshold=float(graph_config.get("label_overlap_threshold", 0.25)),
    )

    aoi_config = dict(config.get("aoi", {}))
    aoi_id = str(aoi_config.get("aoi_id", config.get("experiment_name", config_path.stem)))
    split = str(aoi_config.get("split", "train")).lower()
    part = AOIPart(
        aoi_id=aoi_id,
        split=split,
        config_path=config_path,
        graph=graph,
        features=features.features,
        labels=labels.labels,
        utility_truth=utility_truth,
    )
    return part, config


def _shift_array(array: np.ndarray, offsets: tuple[int, int]) -> np.ndarray:
    if array.size == 0:
        return np.zeros_like(array, dtype=np.int64)
    shifted = np.asarray(array, dtype=np.int64).copy()
    shifted[0] += int(offsets[0])
    shifted[1] += int(offsets[1])
    return shifted


def _combine_parts(parts: list[AOIPart]) -> tuple[
    HeteroRoadGraph,
    pd.DataFrame,
    gpd.GeoDataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    segment_frames: list[gpd.GeoDataFrame] = []
    intersection_frames: list[gpd.GeoDataFrame] = []
    label_frames: list[gpd.GeoDataFrame] = []
    feature_frames: list[pd.DataFrame] = []
    crosses: list[np.ndarray] = []
    touches: list[np.ndarray] = []
    train_index: list[int] = []
    val_index: list[int] = []
    test_index: list[int] = []

    segment_offset = 0
    intersection_offset = 0
    for part in parts:
        graph = part.graph
        n_segments = len(graph.road_segments)
        n_intersections = len(graph.intersections)

        segments = graph.road_segments.copy()
        segments["segment_id"] = segments["segment_id"].astype(int) + segment_offset
        segments["aoi_id"] = part.aoi_id
        segments["aoi_split"] = part.split
        segment_frames.append(segments)

        intersections = graph.intersections.copy()
        intersections["intersection_id"] = intersections["intersection_id"].astype(int) + intersection_offset
        intersections["aoi_id"] = part.aoi_id
        intersections["aoi_split"] = part.split
        intersection_frames.append(intersections)

        labels = part.labels.copy()
        labels["segment_id"] = labels["segment_id"].astype(int) + segment_offset
        labels["aoi_id"] = part.aoi_id
        labels["aoi_split"] = part.split
        label_frames.append(labels)

        features = part.features.copy()
        features.index = features.index.astype(int) + segment_offset
        feature_frames.append(features)

        crosses.append(_shift_array(graph.segment_crosses_segment, (segment_offset, segment_offset)))
        touches.append(
            _shift_array(
                graph.segment_touches_intersection,
                (segment_offset, intersection_offset),
            )
        )

        indexes = range(segment_offset, segment_offset + n_segments)
        if part.split == "train":
            train_index.extend(indexes)
        elif part.split in {"val", "valid", "validation"}:
            val_index.extend(indexes)
        elif part.split == "test":
            test_index.extend(indexes)
        else:
            train_index.extend(indexes)

        segment_offset += n_segments
        intersection_offset += n_intersections

    road_segments = gpd.GeoDataFrame(pd.concat(segment_frames, ignore_index=True), crs=parts[0].graph.road_segments.crs)
    intersections = gpd.GeoDataFrame(
        pd.concat(intersection_frames, ignore_index=True),
        crs=parts[0].graph.intersections.crs,
    )
    labels = gpd.GeoDataFrame(pd.concat(label_frames, ignore_index=True), crs=parts[0].labels.crs)
    features = pd.concat(feature_frames, axis=0).sort_index()
    features = features.reindex(sorted(features.columns), axis=1).fillna(0.0)
    assert_no_anchor_features(features.columns)

    graph = HeteroRoadGraph(
        road_segments=road_segments.sort_values("segment_id").reset_index(drop=True),
        intersections=intersections.sort_values("intersection_id").reset_index(drop=True),
        segment_crosses_segment=np.concatenate(crosses, axis=1)
        if crosses
        else np.zeros((2, 0), dtype=np.int64),
        segment_touches_intersection=np.concatenate(touches, axis=1)
        if touches
        else np.zeros((2, 0), dtype=np.int64),
        crs=parts[0].graph.crs,
        metadata={
            "candidate_graph_type": parts[0].graph.metadata.get("candidate_graph_type", "unknown"),
            "offset_distances_m": parts[0].graph.metadata.get("offset_distances_m", []),
            "n_aois": len(parts),
            "n_road_segments": int(segment_offset),
            "n_intersections": int(intersection_offset),
        },
    )
    return (
        graph,
        features,
        labels.sort_values("segment_id").reset_index(drop=True),
        np.asarray(train_index, dtype=int),
        np.asarray(val_index, dtype=int),
        np.asarray(test_index, dtype=int),
    )


def _select_threshold(
    y: np.ndarray,
    probabilities: np.ndarray,
    val_index: np.ndarray,
    thresholds: Iterable[float],
) -> tuple[float, pd.DataFrame]:
    rows: list[dict] = []
    for threshold in thresholds:
        metrics = compute_edge_metrics(
            y[val_index],
            probabilities[val_index],
            threshold=float(threshold),
        )
        rows.append({"threshold": float(threshold), **metrics})
    table = pd.DataFrame(rows)
    table = table.sort_values(
        ["f1", "pr_auc", "balanced_accuracy"],
        ascending=[False, False, False],
    )
    best = float(table.iloc[0]["threshold"]) if len(table) else 0.5
    return best, pd.DataFrame(rows)


def _split_metrics_table(
    graph: HeteroRoadGraph,
    labels: gpd.GeoDataFrame,
    probabilities: np.ndarray,
    *,
    threshold: float,
    split_indices: dict[str, np.ndarray],
) -> pd.DataFrame:
    labels = labels.sort_values("segment_id").reset_index(drop=True)
    y = labels["y"].to_numpy(dtype=int)
    lengths = graph.road_segments.sort_values("segment_id")["length_m"].to_numpy(dtype=float)
    selected = probabilities >= float(threshold)
    rows: list[dict] = []
    for split, index in split_indices.items():
        index = np.asarray(index, dtype=int)
        if len(index) == 0:
            continue
        metrics = compute_edge_metrics(y[index], probabilities[index], threshold=threshold)
        split_selected = selected[index]
        split_y = y[index]
        split_lengths = lengths[index]
        true_positive = float(split_lengths[(split_selected) & (split_y == 1)].sum())
        false_positive = float(split_lengths[(split_selected) & (split_y == 0)].sum())
        false_negative = float(split_lengths[(~split_selected) & (split_y == 1)].sum())
        predicted_total = true_positive + false_positive
        true_total = true_positive + false_negative
        length_precision = true_positive / predicted_total if predicted_total > 0 else 0.0
        length_recall = true_positive / true_total if true_total > 0 else 0.0
        length_f1 = (
            2.0 * length_precision * length_recall / (length_precision + length_recall)
            if length_precision + length_recall > 0
            else 0.0
        )
        rows.append(
            {
                "split": split,
                "n_segments": int(len(index)),
                "positive_prevalence": float(split_y.mean()),
                **{f"edge_{key}": value for key, value in metrics.items()},
                "length_precision": length_precision,
                "length_recall": length_recall,
                "length_f1": length_f1,
                "predicted_total_length_m": predicted_total,
                "true_total_length_m": true_total,
                "true_positive_predicted_length_m": true_positive,
                "false_positive_length_m": false_positive,
                "false_negative_length_m": false_negative,
            }
        )
    return pd.DataFrame(rows)


def _per_aoi_metrics_table(
    graph: HeteroRoadGraph,
    labels: gpd.GeoDataFrame,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> pd.DataFrame:
    seg = graph.road_segments.sort_values("segment_id")[["segment_id", "aoi_id", "aoi_split", "length_m"]]
    table = seg.merge(labels[["segment_id", "y"]], on="segment_id", how="left")
    table["probability"] = probabilities
    rows = []
    for (aoi_id, split), group in table.groupby(["aoi_id", "aoi_split"]):
        y = group["y"].to_numpy(dtype=int)
        p = group["probability"].to_numpy(dtype=float)
        metrics = compute_edge_metrics(y, p, threshold=threshold)
        rows.append(
            {
                "aoi_id": aoi_id,
                "split": split,
                "n_segments": int(len(group)),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _format_offset_distance(value) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not np.isfinite(numeric):
        return "unknown"
    rounded = round(numeric)
    if abs(numeric - rounded) < 1e-6:
        return str(int(rounded))
    return f"{numeric:g}"


def _offset_lane_name(row: pd.Series) -> str:
    source = str(row.get("candidate_source", "road_backbone"))
    if source == "road_offset":
        side = str(row.get("road_offset_side", "unknown"))
        distance = _format_offset_distance(row.get("road_offset_distance_m", 0.0))
        return f"{side}_{distance}m"
    if source == "road_backbone":
        return "center"
    return source


def _offset_lane_predictions(
    predictions: gpd.GeoDataFrame,
    *,
    threshold: float,
) -> gpd.GeoDataFrame:
    required = {"aoi_id", "source_index", "candidate_source", "probability", "geometry"}
    if predictions.empty or not required.issubset(predictions.columns):
        return gpd.GeoDataFrame(geometry=[], crs=predictions.crs)

    table = predictions.copy()
    table["offset_lane"] = table.apply(_offset_lane_name, axis=1)
    table["predicted_presence"] = table["probability"].astype(float) >= float(threshold)
    table = table.sort_values(
        ["aoi_id", "source_index", "probability", "segment_id"],
        ascending=[True, True, False, True],
    )
    best = table.groupby(["aoi_id", "source_index"], dropna=False).head(1).copy()
    best["best_segment_id"] = best["segment_id"].astype(int)
    best["best_probability"] = best["probability"].astype(float)
    best["best_offset_lane"] = best["offset_lane"]
    best["best_candidate_source"] = best["candidate_source"].fillna("unknown")
    empty_text = pd.Series(index=best.index, dtype=object)
    best["best_offset_side"] = best.get("road_offset_side", empty_text).fillna("center")
    best["best_offset_distance_m"] = pd.to_numeric(
        best.get("road_offset_distance_m", pd.Series(index=best.index, dtype=float)),
        errors="coerce",
    ).fillna(0.0)

    preferred = [
        "aoi_id",
        "aoi_split",
        "source_index",
        "best_segment_id",
        "best_probability",
        "predicted_presence",
        "best_offset_lane",
        "best_candidate_source",
        "best_offset_side",
        "best_offset_distance_m",
        "y",
        "overlap_length",
        "overlap_ratio",
        "geometry",
    ]
    columns = [column for column in preferred if column in best.columns]
    return gpd.GeoDataFrame(best[columns], geometry="geometry", crs=predictions.crs)


def _offset_lane_diagnostics(
    predictions: gpd.GeoDataFrame,
    lanes: gpd.GeoDataFrame,
    *,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"aoi_id", "source_index", "y", "overlap_ratio", "overlap_length"}
    if lanes.empty or predictions.empty or not required.issubset(predictions.columns):
        return pd.DataFrame(), pd.DataFrame()

    candidates = predictions.copy()
    candidates["offset_lane"] = candidates.apply(_offset_lane_name, axis=1)
    truth = candidates.sort_values(
        ["aoi_id", "source_index", "y", "overlap_ratio", "overlap_length"],
        ascending=[True, True, False, False, False],
    ).groupby(["aoi_id", "source_index"], dropna=False).head(1)
    truth = truth[
        ["aoi_id", "source_index", "aoi_split", "y", "offset_lane", "overlap_ratio"]
    ].rename(
        columns={
            "y": "truth_presence",
            "offset_lane": "truth_best_lane",
            "overlap_ratio": "truth_best_overlap_ratio",
        }
    )

    lane_table = pd.DataFrame(lanes.drop(columns=["geometry"], errors="ignore")).merge(
        truth.drop(columns=["aoi_split"], errors="ignore"),
        on=["aoi_id", "source_index"],
        how="left",
    )
    rows: list[dict] = []
    for split, group in lane_table.groupby("aoi_split", dropna=False):
        y = group["truth_presence"].fillna(0).to_numpy(dtype=int)
        p = group["best_probability"].to_numpy(dtype=float)
        metrics = compute_edge_metrics(y, p, threshold=threshold)
        present = group[group["truth_presence"].fillna(0).astype(int) == 1]
        lane_accuracy = (
            float(present["best_offset_lane"].eq(present["truth_best_lane"]).mean())
            if len(present)
            else 0.0
        )
        rows.append(
            {
                "split": split,
                "n_lane_groups": int(len(group)),
                "truth_present_groups": int(y.sum()),
                "truth_presence_prevalence": float(y.mean()) if len(y) else 0.0,
                "truth_present_best_lane_accuracy": lane_accuracy,
                **{f"group_presence_{key}": value for key, value in metrics.items()},
            }
        )

    truth_present = truth[truth["truth_presence"].fillna(0).astype(int) == 1]
    if truth_present.empty:
        truth_summary = pd.DataFrame(
            columns=["aoi_split", "truth_best_lane", "truth_best_count"]
        )
    else:
        truth_summary = (
            truth_present.groupby(["aoi_split", "truth_best_lane"], dropna=False)
            .size()
            .reset_index(name="truth_best_count")
        )
    return pd.DataFrame(rows), truth_summary


def _write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gdf.empty:
        path.write_text('{"type":"FeatureCollection","features":[]}\n', encoding="utf-8")
    else:
        path.write_text(gdf.to_json(drop_id=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate AOI candidate graphs, compute all available non-anchor "
            "features, and train a hetero GNN with AOI-block splits."
        )
    )
    parser.add_argument("--configs", nargs="+", default=["configs/aois/anchor_free_small_aoi_*.yaml"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "aoi_block_gnn_all_features",
    )
    parser.add_argument(
        "--candidate-graph-type",
        choices=["config", "road", "hybrid", "road_offsets"],
        default="hybrid",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--gnn-layer-type", choices=["sage", "gat", "graphconv"], default=None)
    parser.add_argument("--include-node-coords", choices=["config", "true", "false"], default="config")
    parser.add_argument("--sample-spacing-m", type=float, default=25.0)
    parser.add_argument("--representability-buffers-m", nargs="+", type=float, default=[5, 10, 20, 30, 50])
    parser.add_argument("--write-candidates", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    device_report = torch_device_report()
    print(f"GPU preflight: {device_report}", flush=True)

    config_paths = _expand_configs(args.configs)
    parts: list[AOIPart] = []
    loaded_configs: list[dict] = []
    for config_path in config_paths:
        part, config = _prepare_part(
            config_path,
            candidate_graph_type=None
            if args.candidate_graph_type == "config"
            else args.candidate_graph_type,
            include_node_coords=None
            if args.include_node_coords == "config"
            else args.include_node_coords == "true",
        )
        parts.append(part)
        loaded_configs.append(config)
        print(
            f"prepared {part.aoi_id} ({part.split}): "
            f"segments={len(part.graph.road_segments)} "
            f"features={part.features.shape[1]} "
            f"positives={int(part.labels['y'].sum())}/{len(part.labels)}",
            flush=True,
        )

    graph, features, labels, train_index, val_index, test_index = _combine_parts(parts)
    base_config = loaded_configs[0] if loaded_configs else load_anchor_free_config()
    model_config = dict(base_config.get("model", {}))
    graph_config = dict(base_config.get("graph", {}))
    evaluation_config = dict(base_config.get("evaluation", {}))
    decoder_config = dict(base_config.get("decoder", {}))

    include_node_coords = (
        bool(model_config.get("include_node_coords", True))
        if args.include_node_coords == "config"
        else args.include_node_coords == "true"
    )
    intersection_features = build_intersection_features(
        graph,
        include_coords=include_node_coords,
    )
    scaled_features, mean, std = standardize_features(features, train_index=train_index)
    segment_features = RoadSegmentFeatureTable(
        segment_ids=graph.segment_ids,
        features=scaled_features,
    )
    y = labels.sort_values("segment_id")["y"].to_numpy(dtype=int)
    data = build_hetero_pyg_data(
        graph,
        segment_features,
        intersection_features,
        labels=y,
    )

    trained = train_hetero_road_gnn(
        data,
        train_index=train_index,
        val_index=val_index,
        test_index=test_index,
        seed=int(base_config.get("seed", 42)),
        hidden_dim=int(args.hidden_dim or model_config.get("hidden_dim", 64)),
        num_layers=int(args.num_layers or model_config.get("num_layers", 3)),
        dropout=float(model_config.get("dropout", 0.1)),
        lr=float(model_config.get("lr", 0.001)),
        epochs=int(args.epochs or model_config.get("epochs", 100)),
        device=args.device if args.device else model_config.get("device", "auto"),
        layer_type=str(args.gnn_layer_type or model_config.get("gnn_layer_type", "sage")),
        gat_heads=int(model_config.get("gat_heads", 1)),
    )
    probabilities = trained.probabilities

    threshold_values = evaluation_config.get("threshold_grid", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    best_threshold, threshold_table = _select_threshold(
        y,
        probabilities,
        val_index,
        threshold_values,
    )
    decoder_config = dict(decoder_config)
    decoder_config["threshold"] = best_threshold
    decoded = decode_segment_network(graph, probabilities, decoder_config)

    split_indices = {"train": train_index, "val": val_index, "test": test_index}
    split_metrics = _split_metrics_table(
        graph,
        labels,
        probabilities,
        threshold=best_threshold,
        split_indices=split_indices,
    )
    per_aoi_metrics = _per_aoi_metrics_table(
        graph,
        labels,
        probabilities,
        threshold=best_threshold,
    )

    predictions = graph.road_segments.sort_values("segment_id").copy()
    predictions["probability"] = probabilities
    predictions = predictions.merge(
        labels[["segment_id", "y", "overlap_length", "overlap_ratio"]],
        on="segment_id",
        how="left",
    )

    candidate_source_rows = []
    for source, summary in candidate_source_summary(graph).items():
        candidate_source_rows.append(
            {
                "candidate_source": source,
                "count": int(summary["count"]),
                "length_m": float(summary["length_m"]),
            }
        )
    candidate_source_table = pd.DataFrame(candidate_source_rows)

    recall_rows = []
    for part in parts:
        recall = candidate_representability_metrics(
            part.graph,
            part.utility_truth,
            buffers_m=args.representability_buffers_m,
            sample_spacing_m=args.sample_spacing_m,
        )
        recall_rows.append({"aoi_id": part.aoi_id, "split": part.split, **recall})
    candidate_recall = pd.DataFrame(recall_rows)

    if args.write_candidates:
        for part in parts:
            candidates = part.graph.road_segments.copy()
            candidates["aoi_id"] = part.aoi_id
            candidates["split"] = part.split
            _write_geojson(candidates, output_dir / "candidate_graphs" / f"{part.aoi_id}.geojson")

    offset_lane_predictions = gpd.GeoDataFrame(geometry=[], crs=predictions.crs)
    if graph.metadata.get("candidate_graph_type") == "road_offsets":
        offset_lane_predictions = _offset_lane_predictions(
            predictions,
            threshold=best_threshold,
        )
        _write_geojson(offset_lane_predictions, output_dir / "offset_lane_predictions.geojson")
        offset_lane_predictions.drop(columns=["geometry"], errors="ignore").to_csv(
            output_dir / "offset_lane_predictions.csv",
            index=False,
        )
        if not offset_lane_predictions.empty:
            (
                offset_lane_predictions.groupby(
                    ["aoi_split", "best_offset_lane", "predicted_presence"],
                    dropna=False,
                )
                .size()
                .reset_index(name="count")
                .to_csv(output_dir / "offset_lane_summary.csv", index=False)
            )
        offset_lane_metrics, offset_lane_truth_summary = _offset_lane_diagnostics(
            predictions,
            offset_lane_predictions,
            threshold=best_threshold,
        )
        if not offset_lane_metrics.empty:
            offset_lane_metrics.to_csv(output_dir / "offset_lane_group_metrics.csv", index=False)
        if not offset_lane_truth_summary.empty:
            offset_lane_truth_summary.to_csv(
                output_dir / "offset_lane_truth_summary.csv",
                index=False,
            )

    _write_geojson(predictions, output_dir / "road_segment_predictions_for_evaluation.geojson")
    _write_geojson(
        predictions.drop(columns=["y", "overlap_length", "overlap_ratio"], errors="ignore"),
        output_dir / "road_segment_predictions_inference_only.geojson",
    )
    _write_geojson(decoded.road_segments, output_dir / "decoded_network.geojson")
    threshold_table.to_csv(output_dir / "threshold_tuning_val.csv", index=False)
    split_metrics.to_csv(output_dir / "split_metrics.csv", index=False)
    per_aoi_metrics.to_csv(output_dir / "per_aoi_metrics.csv", index=False)
    candidate_recall.to_csv(output_dir / "candidate_recall_by_aoi.csv", index=False)
    candidate_source_table.to_csv(output_dir / "candidate_source_summary.csv", index=False)
    pd.DataFrame({"loss": trained.losses}).to_csv(output_dir / "training_losses.csv", index=False)
    pd.DataFrame(
        {
            "feature": list(features.columns),
            "mean": mean.reindex(features.columns).to_numpy(dtype=float),
            "std": std.reindex(features.columns).to_numpy(dtype=float),
        }
    ).to_csv(output_dir / "feature_scaling.csv", index=False)
    (output_dir / "feature_columns.json").write_text(
        json.dumps(list(features.columns), indent=2),
        encoding="utf-8",
    )

    test_row = split_metrics[split_metrics["split"] == "test"].iloc[0].to_dict()
    summary = {
        "workstream": "Codex",
        "description": "AOI-block hybrid candidate graph + all-feature hetero GNN classification",
        "runtime_sec": time.perf_counter() - start,
        "device_report": device_report,
        "training_device": trained.device,
        "n_aois": len(parts),
        "split_counts_segments": {
            "train": int(len(train_index)),
            "val": int(len(val_index)),
            "test": int(len(test_index)),
        },
        "aoi_splits": {part.aoi_id: part.split for part in parts},
        "candidate_graph_type": graph.metadata.get("candidate_graph_type"),
        "offset_distances_m": graph.metadata.get("offset_distances_m", []),
        "n_offset_lane_groups": int(len(offset_lane_predictions)),
        "n_segments": int(len(graph.road_segments)),
        "n_intersections": int(len(graph.intersections)),
        "n_features": int(features.shape[1]),
        "n_intersection_features": int(intersection_features.features.shape[1]),
        "include_node_coords": include_node_coords,
        "label_buffer_m": float(graph_config.get("label_buffer_m", 10.0)),
        "label_overlap_threshold": float(graph_config.get("label_overlap_threshold", 0.25)),
        "gnn_layer_type": str(args.gnn_layer_type or model_config.get("gnn_layer_type", "sage")),
        "epochs": int(args.epochs or model_config.get("epochs", 100)),
        "best_threshold_from_val": best_threshold,
        "test_metrics": test_row,
        "source_config_paths": [_relative(path) for path in config_paths],
    }
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    print(f"Best threshold from val: {best_threshold:.3f}", flush=True)
    print("Split metrics:", flush=True)
    print(
        split_metrics[
            [
                "split",
                "n_segments",
                "positive_prevalence",
                "edge_roc_auc",
                "edge_pr_auc",
                "edge_f1",
                "edge_precision",
                "edge_recall",
                "length_precision",
                "length_recall",
                "length_f1",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(f"Wrote outputs to {_relative(output_dir)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
