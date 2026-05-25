"""End-to-end anchor-free road-edge prediction pipeline."""

# Workstream: Codex + Claude merge

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd

from pipe_network_completion.anchor_free.baseline import (
    make_buffer_invariant_splits,
    make_stratified_edge_splits,
    train_baseline,
)
from pipe_network_completion.anchor_free.candidate_variants import (  # noqa: E402
    build_candidate_variant_lines,
)
# Phase 2.A — heterogeneous pipeline imports.
from pipe_network_completion.anchor_free.hetero_road_graph import (  # noqa: E402
    HeteroRoadGraph,
    build_hetero_road_graph,
)
from pipe_network_completion.anchor_free.hybrid_candidate_graph import (  # noqa: E402
    build_hybrid_candidate_lines,
)
from pipe_network_completion.anchor_free.features import (  # noqa: E402
    build_intersection_features,
    build_road_segment_features,
)
from pipe_network_completion.anchor_free.labels import (  # noqa: E402
    label_road_segments_from_utility_lines,
)
from pipe_network_completion.anchor_free.decoder import (  # noqa: E402
    decode_segment_network,
)
from pipe_network_completion.anchor_free.evaluation import (  # noqa: E402
    evaluate_hetero_predictions,
)
from pipe_network_completion.anchor_free.model import (  # noqa: E402
    build_hetero_pyg_data,
    train_hetero_road_gnn,
)
from pipe_network_completion.anchor_free.config import write_resolved_config
from pipe_network_completion.anchor_free.decoder import DecodedNetwork, decode_network
from pipe_network_completion.anchor_free.evaluation import (
    AnchorFreeMetrics,
    compute_edge_metrics,
    evaluate_anchor_free_predictions,
)
from pipe_network_completion.anchor_free.features import (
    RoadEdgeFeatureTable,
    RoadSegmentFeatureTable,
    build_road_edge_features,
    standardize_features,
)
from pipe_network_completion.anchor_free.labels import (
    RoadEdgeLabels,
    label_road_edges_from_utility_lines,
)
from pipe_network_completion.anchor_free.model import (
    build_pyg_road_edge_data,
    train_road_edge_gnn,
)
from pipe_network_completion.anchor_free.road_graph import (
    RoadCandidateGraph,
    build_road_candidate_graph,
)
from pipe_network_completion.anchor_free.synthetic import make_synthetic_anchor_free_data
from pipe_network_completion.paths import REPO_ROOT


@dataclass(frozen=True)
class AnchorFreeRunResult:
    output_dir: Path
    # Phase 2.A: graph/features/labels are now heterogeneous types.
    graph: HeteroRoadGraph
    features: object  # RoadSegmentFeatureTable; intentionally loose to avoid circular type import
    labels: object  # RoadSegmentLabels
    probabilities: np.ndarray
    decoded: object  # DecodedRoadSegmentNetwork
    metrics: AnchorFreeMetrics
    train_index: np.ndarray
    val_index: np.ndarray
    test_index: np.ndarray
    intersection_features: object | None = None


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


def _read_optional_vector_many(
    path_values: str | Path | Iterable[str | Path] | None,
) -> gpd.GeoDataFrame | None:
    if path_values in (None, ""):
        return None
    values = path_values if isinstance(path_values, (list, tuple)) else [path_values]
    frames = []
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


def _read_vector_many(path_values: str | Path | Iterable[str | Path] | None) -> gpd.GeoDataFrame:
    """Load one or more truth-line vector files; reproject to a single CRS.

    Fix for P2 of ``docs/research_notes/current_codebase_review_codex.md``:
    the prior version concatenated frames without reprojecting later files
    to the first file's CRS, so labels could be silently miscomputed if
    truth shapefiles disagreed on CRS. We now mirror the behaviour of
    ``_read_optional_vector_many`` (above).
    """
    if path_values in (None, ""):
        raise FileNotFoundError("No utility_truth_path configured.")
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
        raise FileNotFoundError("No utility truth files could be loaded.")
    if len(frames) == 1:
        return frames[0]
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=target_crs)


def _output_dir(config: Mapping[str, Any], output_root: str | Path | None) -> Path:
    root = _resolve_path(output_root) if output_root is not None else REPO_ROOT / "outputs"
    experiment_name = str(config.get("experiment_name", "anchor_free_isarc2024"))
    return root / experiment_name / "anchor_free"


def _save_prediction_map(
    output_path: Path,
    edge_predictions: gpd.GeoDataFrame,
    decoded_edges: gpd.GeoDataFrame,
    utility_truth: gpd.GeoDataFrame,
) -> None:
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))
        edge_predictions.plot(
            ax=ax,
            column="probability",
            cmap="viridis",
            linewidth=1.5,
            legend=True,
        )
        if not utility_truth.empty:
            utility_truth.plot(ax=ax, color="black", linewidth=1.0, alpha=0.6)
        if not decoded_edges.empty:
            decoded_edges.plot(ax=ax, color="red", linewidth=2.5)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
    except Exception as exc:
        output_path.with_suffix(".txt").write_text(
            f"Map rendering skipped: {exc}\n",
            encoding="utf-8",
        )


def _write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    if gdf.empty:
        path.write_text('{"type":"FeatureCollection","features":[]}\n', encoding="utf-8")
    else:
        path.write_text(gdf.to_json(drop_id=True), encoding="utf-8")


def _save_outputs(
    *,
    output_dir: Path,
    config: Mapping[str, Any],
    graph: HeteroRoadGraph,
    labels,
    probabilities: np.ndarray,
    decoded,
    metrics: AnchorFreeMetrics,
    utility_truth: gpd.GeoDataFrame,
) -> None:
    """Phase 2.A — segment-keyed outputs for the heterogeneous pipeline.

    Writes one inference-only and one for-evaluation GeoJSON, named by the
    new ``road_segment_predictions_*`` convention. The legacy
    ``edge_predictions_*`` names are kept as deprecated copies for one
    release so downstream notebooks keep working.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    seg_inference = graph.road_segments.sort_values("segment_id").copy()
    seg_inference["probability"] = np.asarray(probabilities, dtype=float)
    inference_cols = [
        c
        for c in seg_inference.columns
        if c not in {"y", "overlap_length", "overlap_ratio"}
    ]
    seg_inference = seg_inference[inference_cols]
    _write_geojson(
        seg_inference,
        output_dir / "road_segment_predictions_inference_only.geojson",
    )
    # Deprecated alias for old downstream scripts (1 release grace).
    _write_geojson(
        seg_inference,
        output_dir / "edge_predictions_inference_only.geojson",
    )

    label_columns = labels.labels[
        ["segment_id", "y", "overlap_length", "overlap_ratio"]
    ]
    seg_eval = seg_inference.merge(label_columns, on="segment_id", how="left")
    _write_geojson(
        seg_eval,
        output_dir / "road_segment_predictions_for_evaluation.geojson",
    )
    _write_geojson(seg_eval, output_dir / "edge_predictions_for_evaluation.geojson")
    _write_geojson(seg_eval, output_dir / "edge_predictions.geojson")

    _write_geojson(
        decoded.road_segments, output_dir / "decoded_network.geojson"
    )

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics.values, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    metrics.to_frame().to_csv(output_dir / "metrics.csv", index=False)
    write_resolved_config(config, output_dir / "config_resolved.yaml")
    _save_prediction_map(
        output_dir / "prediction_vs_truth_map.png",
        seg_eval,
        decoded.road_segments,
        utility_truth,
    )
    _save_prediction_map(
        output_dir / "prediction_map.png",
        seg_eval,
        decoded.road_segments,
        utility_truth,
    )


def _split_edge_metrics(
    y: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    train_index: np.ndarray,
    val_index: np.ndarray,
    test_index: np.ndarray,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for split_name, split_index in (
        ("train", train_index),
        ("val", val_index),
        ("test", test_index),
    ):
        if len(split_index) == 0:
            continue
        split_metrics = compute_edge_metrics(
            y[np.asarray(split_index, dtype=int)],
            probabilities[np.asarray(split_index, dtype=int)],
            threshold=threshold,
        )
        metrics.update({f"{split_name}_{key}": value for key, value in split_metrics.items()})
    return metrics


def _metric_label_suffix(label_buffer_m: float) -> str:
    return f"{float(label_buffer_m):g}".replace(".", "p") + "m"


def _configured_extra_label_buffers(
    evaluation_config: Mapping[str, Any],
    primary_label_buffer_m: float,
) -> list[float]:
    raw_values = evaluation_config.get("extra_label_buffers_m", [])
    if raw_values in (None, ""):
        return []
    if not isinstance(raw_values, (list, tuple)):
        raw_values = [raw_values]
    values: list[float] = []
    for raw_value in raw_values:
        value = float(raw_value)
        if abs(value - float(primary_label_buffer_m)) < 1e-9:
            continue
        if value not in values:
            values.append(value)
    return values


def _metric_safe_token(value: object) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return token or "unknown"


def _candidate_source_metrics(
    graph: HeteroRoadGraph,
    labels,
    probabilities: np.ndarray,
    selected_segment_ids: np.ndarray | list[int],
) -> dict[str, float]:
    if "candidate_source" not in graph.road_segments.columns:
        return {}
    selected = set(int(sid) for sid in selected_segment_ids)
    table = graph.road_segments[
        ["segment_id", "length_m", "candidate_source"]
    ].copy()
    table["candidate_source"] = table["candidate_source"].fillna("unknown")
    table = table.merge(labels.labels[["segment_id", "y"]], on="segment_id", how="left")
    table["y"] = table["y"].fillna(0).astype(int)
    table["probability"] = np.asarray(probabilities, dtype=float)
    table["selected"] = table["segment_id"].astype(int).isin(selected)

    out: dict[str, float] = {}
    for source, group in table.groupby("candidate_source"):
        prefix = f"candidate_source_{_metric_safe_token(source)}"
        total_length = float(group["length_m"].sum())
        positive_length = float(group.loc[group["y"] == 1, "length_m"].sum())
        selected_length = float(group.loc[group["selected"], "length_m"].sum())
        out[f"{prefix}_count"] = float(len(group))
        out[f"{prefix}_total_length"] = total_length
        out[f"{prefix}_positive_count"] = float(group["y"].sum())
        out[f"{prefix}_positive_length"] = positive_length
        out[f"{prefix}_positive_fraction"] = (
            float(group["y"].mean()) if len(group) else 0.0
        )
        out[f"{prefix}_positive_length_fraction"] = (
            positive_length / total_length if total_length > 0 else 0.0
        )
        out[f"{prefix}_selected_count"] = float(group["selected"].sum())
        out[f"{prefix}_selected_length"] = selected_length
        out[f"{prefix}_mean_probability"] = float(group["probability"].mean())
    return out


def _iter_line_parts(geom):
    if geom is None or geom.is_empty:
        return
    if geom.geom_type in {"LineString", "LinearRing"}:
        yield geom
    elif hasattr(geom, "geoms"):
        for part in geom.geoms:
            yield from _iter_line_parts(part)


def _truth_sample_points(
    utility_truth: gpd.GeoDataFrame,
    *,
    sample_spacing_m: float,
) -> gpd.GeoDataFrame:
    records: list[dict] = []
    spacing = max(float(sample_spacing_m), 1.0)
    for geom in utility_truth.geometry:
        for line in _iter_line_parts(geom):
            length = float(line.length)
            if length <= 0.0:
                continue
            n_samples = max(1, int(round(length / spacing)))
            weight = length / n_samples
            for i in range(n_samples):
                distance = min((i + 0.5) * weight, length)
                records.append(
                    {
                        "weight_m": float(weight),
                        "geometry": line.interpolate(distance),
                    }
                )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=utility_truth.crs)


def _configured_representability_buffers(
    evaluation_config: Mapping[str, Any],
) -> list[float]:
    raw_values = evaluation_config.get("representability_buffers_m", [5, 10, 20, 30, 50])
    if raw_values in (None, ""):
        return []
    if not isinstance(raw_values, (list, tuple)):
        raw_values = [raw_values]
    return [float(value) for value in raw_values]


def _candidate_representability_metrics(
    graph: HeteroRoadGraph,
    utility_truth: gpd.GeoDataFrame,
    *,
    buffers_m: Iterable[float],
    sample_spacing_m: float = 100.0,
) -> dict[str, float]:
    buffers = sorted(float(b) for b in buffers_m if float(b) >= 0.0)
    out: dict[str, float] = {
        "candidate_total_length": float(graph.road_segments["length_m"].sum())
        if not graph.road_segments.empty
        else 0.0,
    }
    if not buffers or graph.road_segments.empty or utility_truth.empty:
        return out
    truth = utility_truth
    if (
        graph.road_segments.crs is not None
        and truth.crs is not None
        and str(truth.crs) != str(graph.road_segments.crs)
    ):
        truth = truth.to_crs(graph.road_segments.crs)
    samples = _truth_sample_points(truth, sample_spacing_m=float(sample_spacing_m))
    total = float(samples["weight_m"].sum()) if not samples.empty else 0.0
    out["truth_total_length_sampled"] = total
    out["truth_representability_sample_spacing_m"] = float(sample_spacing_m)
    if samples.empty or total <= 0.0:
        return out

    max_buffer = max(buffers)
    candidates = graph.road_segments[["segment_id", "geometry"]].copy()
    joined = gpd.sjoin_nearest(
        samples,
        candidates,
        how="left",
        max_distance=max_buffer,
        distance_col="nearest_candidate_distance_m",
    )
    nearest_distance = joined.groupby(level=0)["nearest_candidate_distance_m"].min()
    for buffer_m in buffers:
        suffix = _metric_label_suffix(buffer_m)
        covered_sample_index = nearest_distance.index[
            nearest_distance.notna() & (nearest_distance <= float(buffer_m))
        ]
        covered = float(samples.loc[covered_sample_index, "weight_m"].sum())
        out[f"candidate_representability_{suffix}_truth_length"] = covered
        out[f"candidate_representability_{suffix}_truth_fraction"] = covered / total
    return out


def prepare_anchor_free_inputs(
    config: Mapping[str, Any],
    *,
    synthetic: bool = False,
) -> tuple[
    HeteroRoadGraph,
    gpd.GeoDataFrame | None,
    gpd.GeoDataFrame | None,
    gpd.GeoDataFrame | None,
    Path | None,
    gpd.GeoDataFrame,
]:
    """Phase 2.A — builds the heterogeneous (RoadSegment + Intersection) graph
    instead of the legacy road-as-edge candidate graph."""
    graph_config = dict(config.get("graph", {}))
    data_config = dict(config.get("data", {}))
    candidate_graph_type = str(graph_config.get("candidate_graph_type", "road")).lower()
    road_class_columns = graph_config.get("road_class_columns")
    if isinstance(road_class_columns, str):
        keep_columns = [road_class_columns]
    else:
        keep_columns = list(road_class_columns or [])

    if synthetic:
        synthetic_data = make_synthetic_anchor_free_data()
        roads = synthetic_data.roads
        buildings = synthetic_data.buildings
        building_points = synthetic_data.buildings
        built_up = None
        dem_path = None
        utility_truth = synthetic_data.utility_truth
    else:
        roads = _read_vector(data_config.get("roads_path"))
        if roads is None:
            raise FileNotFoundError("No roads_path configured.")
        buildings = _read_optional_vector_many(data_config.get("buildings_path"))
        building_points = _read_optional_vector_many(data_config.get("building_points_path"))
        built_up = _read_optional_vector_many(data_config.get("built_up_path"))
        dem_path = _resolve_path(data_config.get("dem_path"))
        utility_truth = _read_vector_many(data_config.get("utility_truth_path"))

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
        hybrid = build_hybrid_candidate_lines(
            roads,
            buildings_gdf=(
                None if graph_config.get("use_buildings", True) is False else buildings
            ),
            building_points_gdf=(
                None
                if graph_config.get("use_building_points", True) is False
                else building_points
            ),
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
    graph.metadata["candidate_graph_type"] = candidate_graph_type
    if candidate_graph_type == "hybrid":
        graph.metadata["hybrid_candidate_by_source"] = hybrid.metadata.get(
            "candidate_by_source", {}
        )
        graph.metadata["n_demand_points"] = hybrid.metadata.get("n_demand_points", 0)
        graph.metadata["n_demand_clusters"] = hybrid.metadata.get(
            "n_demand_clusters", 0
        )
    elif candidate_graph_type == "road_offsets":
        graph.metadata["road_offset_candidate_by_source"] = road_offsets.metadata.get(
            "candidate_by_source", {}
        )
        graph.metadata["offset_distances_m"] = road_offsets.metadata.get(
            "offset_distances_m", []
        )
    return graph, buildings, building_points, built_up, dem_path, utility_truth


def run_anchor_free_experiment(
    config: Mapping[str, Any],
    *,
    synthetic: bool = False,
    output_root: str | Path | None = None,
) -> AnchorFreeRunResult:
    """Run graph construction, training, decoding, evaluation, and export."""

    start = time.perf_counter()
    seed = int(config.get("seed", 42))
    graph_config = dict(config.get("graph", {}))
    model_config = dict(config.get("model", {}))
    decoder_config = dict(config.get("decoder", {}))
    evaluation_config = dict(config.get("evaluation", {}))

    graph, buildings, building_points, built_up, dem_path, utility_truth = prepare_anchor_free_inputs(
        config,
        synthetic=synthetic,
    )
    if graph_config.get("use_buildings", True) is False:
        buildings = None
    if graph_config.get("use_building_points", True) is False:
        building_points = None
    if graph_config.get("use_built_up", True) is False:
        built_up = None
    if graph_config.get("use_dem", True) is False:
        dem_path = None
    use_watercourses = bool(graph_config.get("use_watercourses", False))
    watercourse_complete = bool(graph_config.get("watercourse_context_complete", False))
    watercourse_drainage_lines = None
    watercourse_corridor_centrelines = None
    watercourse_corridors = None
    if use_watercourses and watercourse_complete:
        data_config = dict(config.get("data", {}))
        watercourse_drainage_lines = _read_optional_vector_many(
            data_config.get("watercourse_drainage_lines_path")
        )
        watercourse_corridor_centrelines = _read_optional_vector_many(
            data_config.get("watercourse_corridor_centrelines_path")
        )
        watercourse_corridors = _read_optional_vector_many(
            data_config.get("watercourse_corridors_path")
        )
    feature_road_class_columns = graph_config.get("road_class_columns")
    if graph.metadata.get("candidate_graph_type") in {"hybrid", "road_offsets"}:
        if isinstance(feature_road_class_columns, str):
            feature_road_class_columns = [feature_road_class_columns]
        else:
            feature_road_class_columns = list(feature_road_class_columns or [])
        if "candidate_source" not in feature_road_class_columns:
            feature_road_class_columns.append("candidate_source")
        if (
            graph.metadata.get("candidate_graph_type") == "road_offsets"
            and "road_offset_side" not in feature_road_class_columns
        ):
            feature_road_class_columns.append("road_offset_side")
    # Phase 2.A — heterogeneous feature + label builders.
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
    intersection_features = build_intersection_features(
        graph,
        include_coords=bool(model_config.get("include_node_coords", True)),
    )
    labels = label_road_segments_from_utility_lines(
        graph,
        utility_truth,
        label_buffer_m=float(graph_config.get("label_buffer_m", 10.0)),
        label_overlap_threshold=float(graph_config.get("label_overlap_threshold", 0.25)),
    )
    # Stage 1 of audit_followup_implementation_plan.md, completing AR-AF-A.3:
    # The training pipeline must use the ISARC-seeded buffer-invariant split
    # so per-edge metrics are comparable across label buffers AND so the
    # split is not stratified on label values (which leaks marginal info).
    # ``split.strategy: "stratified"`` is still accepted to reproduce the
    # pre-Stage-1 metric history archived under
    # ``outputs/_archive_leaky_split/``.
    split_config = dict(config.get("split", {}))
    split_strategy = str(split_config.get("strategy", "buffer_invariant")).lower()
    if split_strategy == "buffer_invariant":
        # Phase 2.A — split is keyed on segment_id now (was edge_id).
        train_index, val_index, test_index = make_buffer_invariant_splits(
            features.segment_ids,
            seed=seed,
            train_fraction=float(split_config.get("train_fraction", 0.6)),
            val_fraction=float(split_config.get("val_fraction", 0.2)),
        )
    elif split_strategy == "stratified":
        train_index, val_index, test_index = make_stratified_edge_splits(
            labels.y,
            seed=seed,
            train_fraction=float(split_config.get("train_fraction", 0.6)),
            val_fraction=float(split_config.get("val_fraction", 0.2)),
        )
    else:
        raise ValueError(
            f"Unsupported split.strategy {split_strategy!r}; expected "
            "'buffer_invariant' (default) or 'stratified'."
        )
    scaled_features, _, _ = standardize_features(features.features, train_index=train_index)
    scaled_feature_table = RoadSegmentFeatureTable(
        segment_ids=features.segment_ids,
        features=scaled_features,
    )

    model_type = str(model_config.get("type", "gnn")).lower()
    if model_type in {"logistic_regression", "random_forest"}:
        trained = train_baseline(
            scaled_features,
            labels.y,
            kind=model_type,  # type: ignore[arg-type]
            train_index=train_index,
            val_index=val_index,
            test_index=test_index,
            seed=seed,
            class_weight=model_config.get("class_weight", "balanced"),
        )
        probabilities = trained.probabilities
    elif model_type == "gnn":
        # Phase 2.A — heterogeneous GNN over (RoadSegment + Intersection).
        # ``model.include_node_coords`` now toggles the (x, y) columns on
        # the Intersection feature table (see build_intersection_features).
        data = build_hetero_pyg_data(
            graph,
            scaled_feature_table,
            intersection_features,
            labels=labels.y,
        )
        trained_gnn = train_hetero_road_gnn(
            data,
            train_index=train_index,
            val_index=val_index,
            test_index=test_index,
            seed=seed,
            hidden_dim=int(model_config.get("hidden_dim", 64)),
            num_layers=int(model_config.get("num_layers", 3)),
            dropout=float(model_config.get("dropout", 0.1)),
            lr=float(model_config.get("lr", 0.001)),
            epochs=int(model_config.get("epochs", 100)),
            device=model_config.get("device", "auto"),
            layer_type=str(model_config.get("gnn_layer_type", "sage")),
            gat_heads=int(model_config.get("gat_heads", 1)),
        )
        probabilities = trained_gnn.probabilities
    else:
        raise ValueError(f"Unsupported anchor-free model type: {model_type}")

    decoded = decode_segment_network(graph, probabilities, decoder_config)
    threshold = float(decoder_config.get("threshold", 0.5))
    runtime_sec = time.perf_counter() - start
    # Phase 2.A — hetero evaluation. Labels carry `segment_id` and
    # `decoded.segment_ids` keys the selected nodes.
    metrics = evaluate_hetero_predictions(
        graph,
        labels.labels,
        probabilities,
        decoded.segment_ids,
        threshold=threshold,
        buildings=buildings,
        building_service_buffer_m=float(evaluation_config.get("building_service_buffer_m", 50.0)),
        decoded_segments=decoded.road_segments,
        extra={
            "runtime_sec": float(runtime_sec),
            "model_type": model_type,
            "gnn_layer_type": str(model_config.get("gnn_layer_type", "sage")),
            "decoder_type": decoded.decoder_type,
            "candidate_graph_type": str(graph.metadata.get("candidate_graph_type", "road")),
            "n_road_segments": float(len(graph.road_segments)),
            "n_intersections": float(len(graph.intersections)),
            "n_demand_points": float(graph.metadata.get("n_demand_points", 0)),
            "n_demand_clusters": float(graph.metadata.get("n_demand_clusters", 0)),
        },
    )
    merged_metric_values = dict(metrics.values)
    merged_metric_values.update(
        _split_edge_metrics(
            labels.y,
            probabilities,
            threshold=threshold,
            train_index=train_index,
            val_index=val_index,
            test_index=test_index,
        )
    )
    merged_metric_values.update(
        _candidate_source_metrics(graph, labels, probabilities, decoded.segment_ids)
    )
    merged_metric_values.update(
        _candidate_representability_metrics(
            graph,
            utility_truth,
            buffers_m=_configured_representability_buffers(evaluation_config),
            sample_spacing_m=float(
                evaluation_config.get("representability_sample_spacing_m", 100.0)
            ),
        )
    )
    primary_label_buffer_m = float(graph_config.get("label_buffer_m", 10.0))
    for extra_label_buffer_m in _configured_extra_label_buffers(
        evaluation_config,
        primary_label_buffer_m,
    ):
        extra_labels = label_road_segments_from_utility_lines(
            graph,
            utility_truth,
            label_buffer_m=float(extra_label_buffer_m),
            label_overlap_threshold=float(graph_config.get("label_overlap_threshold", 0.25)),
        )
        extra_metrics = evaluate_hetero_predictions(
            graph,
            extra_labels.labels,
            probabilities,
            decoded.segment_ids,
            threshold=threshold,
            decoded_segments=decoded.road_segments,
        )
        prefix = f"label_{_metric_label_suffix(extra_label_buffer_m)}_"
        merged_metric_values.update(
            {f"{prefix}{key}": value for key, value in extra_metrics.values.items()}
        )
        merged_metric_values.update(
            {
                f"{prefix}{key}": value
                for key, value in _split_edge_metrics(
                    extra_labels.y,
                    probabilities,
                    threshold=threshold,
                    train_index=train_index,
                    val_index=val_index,
                    test_index=test_index,
                ).items()
            }
        )
    metrics = AnchorFreeMetrics(values=merged_metric_values)

    output_dir = _output_dir(config, output_root)
    _save_outputs(
        output_dir=output_dir,
        config=config,
        graph=graph,
        labels=labels,
        probabilities=probabilities,
        decoded=decoded,
        metrics=metrics,
        utility_truth=utility_truth,
    )

    return AnchorFreeRunResult(
        output_dir=output_dir,
        graph=graph,
        features=features,
        labels=labels,
        probabilities=probabilities,
        decoded=decoded,
        metrics=metrics,
        train_index=train_index,
        val_index=val_index,
        test_index=test_index,
        intersection_features=intersection_features,
    )
