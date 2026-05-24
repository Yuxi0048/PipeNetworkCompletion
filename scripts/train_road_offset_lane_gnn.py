"""Train a road-node GNN for presence + five-lane offset prediction.

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
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv, SAGEConv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipe_network_completion.anchor_free.config import load_anchor_free_config  # noqa: E402
from pipe_network_completion.anchor_free.evaluation import compute_edge_metrics  # noqa: E402
from pipe_network_completion.anchor_free.features import (  # noqa: E402
    assert_no_anchor_features,
    build_road_segment_features,
    standardize_features,
)
from pipe_network_completion.anchor_free.labels import (  # noqa: E402
    FIXED_ROAD_OFFSET_LANES,
    road_offset_lane_name,
)
from pipe_network_completion.anchor_free.model import (  # noqa: E402
    resolve_torch_device,
    torch_device_report,
)
from pipe_network_completion.anchor_free.pipeline import prepare_anchor_free_inputs  # noqa: E402


@dataclass(frozen=True)
class AOIRoadNodePart:
    aoi_id: str
    split: str
    config_path: Path
    features: pd.DataFrame
    labels: pd.DataFrame
    geometries: gpd.GeoDataFrame
    edge_pairs: np.ndarray


class RoadOffsetLaneGNN(torch.nn.Module):
    """Shared road-node encoder with binary presence and five-lane heads."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        layer_type: str = "sage",
        n_lanes: int = 5,
    ):
        super().__init__()
        self.dropout = float(dropout)
        self.encoder = torch.nn.Linear(input_dim, hidden_dim)
        layer = str(layer_type).lower()
        conv_cls = GraphConv if layer in {"graphconv", "graph_conv"} else SAGEConv
        self.convs = torch.nn.ModuleList(
            [conv_cls(hidden_dim, hidden_dim) for _ in range(max(int(num_layers), 1))]
        )
        self.presence_head = torch.nn.Linear(hidden_dim, 1)
        self.lane_head = torch.nn.Linear(hidden_dim, int(n_lanes))

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.encoder(data.x.float()))
        for conv in self.convs:
            x = F.dropout(
                F.relu(conv(x, data.edge_index)),
                p=self.dropout,
                training=self.training,
            )
        return self.presence_head(x).reshape(-1), self.lane_head(x)


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


def _feature_road_class_columns(config: dict, graph) -> list[str]:
    graph_config = dict(config.get("graph", {}))
    columns = graph_config.get("road_class_columns")
    if isinstance(columns, str):
        columns = [columns]
    else:
        columns = list(columns or [])
    if "candidate_source" not in columns:
        columns.append("candidate_source")
    if "road_offset_side" not in columns:
        columns.append("road_offset_side")
    return columns


def _read_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path)
    labels["aoi_id"] = labels["aoi_id"].astype(str)
    labels["source_index"] = labels["source_index"].astype(int)
    labels["y"] = labels["y"].astype(int)
    labels["lane_class"] = labels["lane_class"].astype(int)
    if "is_ambiguous" in labels.columns:
        labels["is_ambiguous"] = (
            labels["is_ambiguous"].astype(str).str.lower().isin({"true", "1", "yes"})
        )
    else:
        labels["is_ambiguous"] = False
    return labels


def _weighted_average_features(
    lane_segments: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    if lane_segments.empty:
        return np.zeros(len(feature_columns), dtype=float)
    values = lane_segments[feature_columns].to_numpy(dtype=float)
    weights = lane_segments["length_m"].to_numpy(dtype=float)
    weights = np.maximum(weights, 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        return values.mean(axis=0)
    return np.average(values, axis=0, weights=weights)


def _aggregate_source_road_features(
    graph,
    segment_features: pd.DataFrame,
) -> tuple[pd.DataFrame, gpd.GeoDataFrame, np.ndarray]:
    segments = graph.road_segments.copy()
    segments["offset_lane"] = segments.apply(road_offset_lane_name, axis=1)
    segments = segments[segments["offset_lane"].isin(FIXED_ROAD_OFFSET_LANES)].copy()
    feature_columns = list(segment_features.columns)
    assert_no_anchor_features(feature_columns)
    overlapping_feature_columns = [
        column
        for column in feature_columns
        if column in segments.columns and column != "segment_id"
    ]
    segment_base = segments.drop(columns=overlapping_feature_columns, errors="ignore")
    merged = segment_base.merge(
        segment_features.reset_index().rename(columns={"index": "segment_id"}),
        on="segment_id",
        how="left",
    )
    merged[feature_columns] = merged[feature_columns].fillna(0.0)

    records: list[dict] = []
    geometries: list[dict] = []
    for source_index, group in merged.groupby("source_index", dropna=False):
        record = {"source_index": int(source_index)}
        for lane_name in FIXED_ROAD_OFFSET_LANES:
            lane_group = group[group["offset_lane"] == lane_name]
            lane_values = _weighted_average_features(lane_group, feature_columns)
            for feature_name, value in zip(feature_columns, lane_values):
                record[f"{lane_name}__{feature_name}"] = float(value)
            record[f"{lane_name}__candidate_available"] = float(not lane_group.empty)
        records.append(record)

        center = group[group["offset_lane"] == "center"]
        geom_group = center if not center.empty else group
        geometries.append(
            {
                "source_index": int(source_index),
                "geometry": geom_group.geometry.unary_union,
            }
        )

    feature_frame = pd.DataFrame(records).sort_values("source_index").reset_index(drop=True)
    geometry_frame = gpd.GeoDataFrame(
        geometries,
        geometry="geometry",
        crs=segments.crs,
    ).sort_values("source_index").reset_index(drop=True)

    source_to_local = {
        int(source_index): node_index
        for node_index, source_index in enumerate(feature_frame["source_index"].astype(int))
    }
    seg_lookup = segments.set_index("segment_id")
    center_seg_ids = set(
        segments.loc[segments["offset_lane"] == "center", "segment_id"].astype(int).tolist()
    )
    edge_pairs: set[tuple[int, int]] = set()
    crosses = getattr(graph, "segment_crosses_segment", np.zeros((2, 0), dtype=np.int64))
    if crosses is not None and crosses.size:
        for a, b in zip(crosses[0], crosses[1]):
            a = int(a)
            b = int(b)
            if a not in center_seg_ids or b not in center_seg_ids:
                continue
            source_a = int(seg_lookup.loc[a, "source_index"])
            source_b = int(seg_lookup.loc[b, "source_index"])
            if source_a == source_b:
                continue
            if source_a in source_to_local and source_b in source_to_local:
                edge_pairs.add((source_to_local[source_a], source_to_local[source_b]))
    if edge_pairs:
        edge_index = np.asarray(sorted(edge_pairs), dtype=np.int64).T
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    return feature_frame, geometry_frame, edge_index


def _prepare_part(
    config_path: Path,
    labels: pd.DataFrame,
) -> AOIRoadNodePart:
    config = load_anchor_free_config(config_path)
    config.setdefault("graph", {})["candidate_graph_type"] = "road_offsets"
    graph_config = dict(config.get("graph", {}))
    graph, buildings, building_points, built_up, dem_path, _ = prepare_anchor_free_inputs(config)
    if graph_config.get("use_buildings", True) is False:
        buildings = None
    if graph_config.get("use_building_points", True) is False:
        building_points = None
    if graph_config.get("use_built_up", True) is False:
        built_up = None
    if graph_config.get("use_dem", True) is False:
        dem_path = None

    segment_features = build_road_segment_features(
        graph,
        buildings_gdf=buildings,
        building_points_gdf=building_points,
        built_up_gdf=built_up,
        dem_path=dem_path,
        road_class_columns=_feature_road_class_columns(config, graph),
        building_buffer_m=float(graph_config.get("building_buffer_m", 50.0)),
        building_point_buffer_m=float(graph_config.get("building_point_buffer_m", 50.0)),
        built_up_buffer_m=float(graph_config.get("built_up_buffer_m", 50.0)),
        road_density_buffer_m=float(graph_config.get("road_density_buffer_m", 100.0)),
        dem_sample_spacing_m=float(graph_config.get("dem_sample_spacing_m", 30.0)),
        dem_max_samples_per_edge=int(graph_config.get("dem_max_samples_per_edge", 64)),
    ).features

    features, geometries, edge_pairs = _aggregate_source_road_features(
        graph,
        segment_features,
    )
    aoi_config = dict(config.get("aoi", {}))
    aoi_id = str(aoi_config.get("aoi_id", config.get("experiment_name", config_path.stem)))
    split = str(aoi_config.get("split", "train")).lower()
    labels_part = labels[labels["aoi_id"] == aoi_id].copy()
    labels_part = features[["source_index"]].merge(
        labels_part,
        on="source_index",
        how="left",
    )
    if labels_part["y"].isna().any():
        missing = labels_part.loc[labels_part["y"].isna(), "source_index"].head(5).tolist()
        raise ValueError(f"Missing lane labels for {aoi_id} source_index examples {missing}")
    labels_part["aoi_id"] = aoi_id
    labels_part["aoi_split"] = split
    labels_part["y"] = labels_part["y"].astype(int)
    labels_part["lane_class"] = labels_part["lane_class"].astype(int)
    labels_part["is_ambiguous"] = labels_part["is_ambiguous"].astype(bool)

    features.insert(0, "aoi_id", aoi_id)
    features.insert(1, "aoi_split", split)
    geometries.insert(0, "aoi_id", aoi_id)
    geometries.insert(1, "aoi_split", split)
    return AOIRoadNodePart(
        aoi_id=aoi_id,
        split=split,
        config_path=config_path,
        features=features,
        labels=labels_part,
        geometries=geometries,
        edge_pairs=edge_pairs,
    )


def _combine_parts(parts: list[AOIRoadNodePart]) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    gpd.GeoDataFrame,
    np.ndarray,
    dict[str, np.ndarray],
]:
    feature_frames: list[pd.DataFrame] = []
    label_frames: list[pd.DataFrame] = []
    geometry_frames: list[gpd.GeoDataFrame] = []
    edge_arrays: list[np.ndarray] = []
    split_indices = {"train": [], "val": [], "test": []}
    node_offset = 0
    for part in parts:
        n = len(part.features)
        features = part.features.copy()
        labels = part.labels.copy()
        geoms = part.geometries.copy()
        features["node_id"] = np.arange(node_offset, node_offset + n, dtype=int)
        labels["node_id"] = np.arange(node_offset, node_offset + n, dtype=int)
        geoms["node_id"] = np.arange(node_offset, node_offset + n, dtype=int)
        feature_frames.append(features)
        label_frames.append(labels)
        geometry_frames.append(geoms)
        if part.edge_pairs.size:
            edge_arrays.append(part.edge_pairs + node_offset)
        split_key = "val" if part.split in {"val", "valid", "validation"} else part.split
        if split_key not in split_indices:
            split_key = "train"
        split_indices[split_key].extend(range(node_offset, node_offset + n))
        node_offset += n

    features = pd.concat(feature_frames, ignore_index=True)
    labels = pd.concat(label_frames, ignore_index=True)
    geometries = gpd.GeoDataFrame(
        pd.concat(geometry_frames, ignore_index=True),
        geometry="geometry",
        crs=geometry_frames[0].crs if geometry_frames else None,
    )
    feature_values = (
        features.drop(columns=["aoi_id", "aoi_split", "source_index", "node_id"])
        .reindex(sorted(features.drop(columns=["aoi_id", "aoi_split", "source_index", "node_id"]).columns), axis=1)
        .fillna(0.0)
    )
    assert_no_anchor_features(feature_values.columns)
    if edge_arrays:
        edge_index = np.concatenate(edge_arrays, axis=1)
        reverse = edge_index[[1, 0]]
        edge_index = np.unique(np.concatenate([edge_index, reverse], axis=1), axis=1)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    return (
        feature_values,
        labels.sort_values("node_id").reset_index(drop=True),
        geometries.sort_values("node_id").reset_index(drop=True),
        edge_index,
        {key: np.asarray(value, dtype=int) for key, value in split_indices.items()},
    )


def _select_threshold(
    y: np.ndarray,
    probabilities: np.ndarray,
    val_index: np.ndarray,
    thresholds: Iterable[float],
) -> tuple[float, pd.DataFrame]:
    if len(val_index) == 0:
        return 0.5, pd.DataFrame()
    rows = []
    for threshold in thresholds:
        rows.append(
            {
                "threshold": float(threshold),
                **compute_edge_metrics(y[val_index], probabilities[val_index], threshold=float(threshold)),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return 0.5, table
    best = table.sort_values(
        ["f1", "pr_auc", "balanced_accuracy"],
        ascending=[False, False, False],
    ).iloc[0]
    return float(best["threshold"]), table


def _lane_metrics(
    y_true_presence: np.ndarray,
    lane_true: np.ndarray,
    presence_prob: np.ndarray,
    lane_pred: np.ndarray,
    index: np.ndarray,
    *,
    threshold: float,
    ambiguous: np.ndarray,
    exclude_ambiguous: bool,
) -> dict[str, float]:
    index = np.asarray(index, dtype=int)
    if len(index) == 0:
        return {}
    mask = y_true_presence[index] == 1
    if exclude_ambiguous:
        mask = mask & (~ambiguous[index])
    lane_index = index[mask]
    out: dict[str, float] = {
        "lane_eval_count": int(len(lane_index)),
    }
    if len(lane_index) == 0:
        out.update(
            {
                "lane_accuracy": float("nan"),
                "lane_macro_f1": float("nan"),
                "lane_balanced_accuracy": float("nan"),
                "joint_presence_lane_accuracy": float("nan"),
            }
        )
        return out
    true_lane = lane_true[lane_index]
    pred_lane = lane_pred[lane_index]
    out["lane_accuracy"] = float(accuracy_score(true_lane, pred_lane))
    out["lane_macro_f1"] = float(f1_score(true_lane, pred_lane, average="macro", zero_division=0))
    out["lane_balanced_accuracy"] = float(
        balanced_accuracy_score(true_lane, pred_lane)
    )
    predicted_present = presence_prob[lane_index] >= float(threshold)
    out["joint_presence_lane_accuracy"] = float(
        np.mean((predicted_present) & (pred_lane == true_lane))
    )
    return out


def _metrics_table(
    labels: pd.DataFrame,
    presence_prob: np.ndarray,
    lane_prob: np.ndarray,
    split_indices: dict[str, np.ndarray],
    *,
    threshold: float,
    exclude_ambiguous_lane_eval: bool,
) -> pd.DataFrame:
    y = labels["y"].to_numpy(dtype=int)
    lane_true = labels["lane_class"].to_numpy(dtype=int)
    ambiguous = labels["is_ambiguous"].to_numpy(dtype=bool)
    lane_pred = lane_prob.argmax(axis=1).astype(int)
    rows: list[dict] = []
    for split, index in split_indices.items():
        if len(index) == 0:
            continue
        presence = compute_edge_metrics(y[index], presence_prob[index], threshold=threshold)
        row = {
            "split": split,
            "n_source_roads": int(len(index)),
            **{f"presence_{key}": value for key, value in presence.items()},
            **_lane_metrics(
                y,
                lane_true,
                presence_prob,
                lane_pred,
                index,
                threshold=threshold,
                ambiguous=ambiguous,
                exclude_ambiguous=exclude_ambiguous_lane_eval,
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gdf.empty:
        path.write_text('{"type":"FeatureCollection","features":[]}\n', encoding="utf-8")
    else:
        path.write_text(gdf.to_json(drop_id=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a road-node GNN for presence + fixed five-lane offset prediction."
    )
    parser.add_argument("--configs", nargs="+", default=["configs/aois_2km_gap500_115/*.yaml"])
    parser.add_argument(
        "--labels",
        type=Path,
        default=REPO_ROOT / "outputs" / "road_offset_lane_labels_5lane_aoi115" / "road_offset_lane_labels.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "road_offset_lane_gnn_5lane_aoi115_sage_no_xy",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lane-loss-weight", type=float, default=1.0)
    parser.add_argument("--gnn-layer-type", choices=["sage", "graphconv"], default="sage")
    parser.add_argument("--exclude-ambiguous-lane-loss", action="store_true", default=True)
    parser.add_argument("--include-ambiguous-lane-loss", action="store_false", dest="exclude_ambiguous_lane_loss")
    parser.add_argument("--exclude-ambiguous-lane-eval", action="store_true", default=True)
    parser.add_argument("--include-ambiguous-lane-eval", action="store_false", dest="exclude_ambiguous_lane_eval")
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--threshold-grid", nargs="+", type=float, default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    device_report = torch_device_report()
    print(f"GPU preflight: {device_report}", flush=True)
    if str(args.device).lower().startswith("cuda") and not device_report["cuda_available"]:
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    labels_all = _read_labels(_resolve(args.labels))
    config_paths = _expand_configs(args.configs)
    if int(args.max_configs) > 0:
        config_paths = config_paths[: int(args.max_configs)]

    parts: list[AOIRoadNodePart] = []
    for config_path in config_paths:
        part = _prepare_part(config_path, labels_all)
        parts.append(part)
        print(
            f"prepared {part.aoi_id} ({part.split}): "
            f"source_roads={len(part.labels)} "
            f"features={part.features.shape[1] - 4} "
            f"positives={int(part.labels['y'].sum())}/{len(part.labels)}",
            flush=True,
        )

    features, labels, geometries, edge_index, split_indices = _combine_parts(parts)
    train_index = split_indices["train"]
    val_index = split_indices["val"]
    test_index = split_indices["test"]
    scaled_features, mean, std = standardize_features(features, train_index=train_index)
    data = Data(
        x=torch.tensor(scaled_features.to_numpy(dtype=float), dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y_presence=torch.tensor(labels["y"].to_numpy(dtype=float), dtype=torch.float32),
        y_lane=torch.tensor(labels["lane_class"].to_numpy(dtype=int), dtype=torch.long),
        is_ambiguous=torch.tensor(labels["is_ambiguous"].to_numpy(dtype=bool), dtype=torch.bool),
    )

    torch.manual_seed(42)
    np.random.seed(42)
    device = resolve_torch_device(args.device)
    data = data.to(device)
    model = RoadOffsetLaneGNN(
        input_dim=int(data.x.shape[1]),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        layer_type=str(args.gnn_layer_type),
        n_lanes=len(FIXED_ROAD_OFFSET_LANES),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    train_tensor = torch.tensor(train_index, dtype=torch.long, device=device)
    train_y = data.y_presence[train_tensor]
    positives = torch.clamp(train_y.sum(), min=1.0)
    negatives = torch.clamp(train_y.numel() - train_y.sum(), min=1.0)
    presence_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=(negatives / positives))

    lane_train_mask = data.y_presence > 0.5
    if args.exclude_ambiguous_lane_loss:
        lane_train_mask = lane_train_mask & (~data.is_ambiguous)
    lane_train_mask = lane_train_mask & torch.zeros_like(data.y_presence, dtype=torch.bool)
    lane_train_mask[train_tensor] = (
        (data.y_presence[train_tensor] > 0.5)
        & (
            (~data.is_ambiguous[train_tensor])
            if args.exclude_ambiguous_lane_loss
            else torch.ones_like(data.is_ambiguous[train_tensor], dtype=torch.bool)
        )
    )
    lane_train_index = torch.where(lane_train_mask)[0]
    lane_counts = torch.bincount(
        data.y_lane[lane_train_index].clamp_min(0),
        minlength=len(FIXED_ROAD_OFFSET_LANES),
    ).float()
    lane_weights = lane_counts.sum() / torch.clamp(
        lane_counts * len(FIXED_ROAD_OFFSET_LANES),
        min=1.0,
    )
    lane_loss_fn = torch.nn.CrossEntropyLoss(weight=lane_weights.to(device))

    losses: list[dict[str, float]] = []
    for epoch in range(int(args.epochs)):
        model.train()
        optimizer.zero_grad()
        presence_logits, lane_logits = model(data)
        presence_loss = presence_loss_fn(
            presence_logits[train_tensor],
            data.y_presence[train_tensor],
        )
        if lane_train_index.numel() > 0:
            lane_loss = lane_loss_fn(
                lane_logits[lane_train_index],
                data.y_lane[lane_train_index],
            )
        else:
            lane_loss = torch.tensor(0.0, device=device)
        loss = presence_loss + float(args.lane_loss_weight) * lane_loss
        loss.backward()
        optimizer.step()
        losses.append(
            {
                "epoch": epoch + 1,
                "loss": float(loss.detach().cpu()),
                "presence_loss": float(presence_loss.detach().cpu()),
                "lane_loss": float(lane_loss.detach().cpu()),
            }
        )

    model.eval()
    with torch.no_grad():
        presence_logits, lane_logits = model(data)
        presence_prob = torch.sigmoid(presence_logits).detach().cpu().numpy()
        lane_prob = torch.softmax(lane_logits, dim=1).detach().cpu().numpy()

    best_threshold, threshold_table = _select_threshold(
        labels["y"].to_numpy(dtype=int),
        presence_prob,
        val_index,
        args.threshold_grid,
    )
    metrics = _metrics_table(
        labels,
        presence_prob,
        lane_prob,
        split_indices,
        threshold=best_threshold,
        exclude_ambiguous_lane_eval=bool(args.exclude_ambiguous_lane_eval),
    )

    predictions = geometries.merge(
        labels[
            [
                "node_id",
                "source_index",
                "y",
                "lane_class",
                "lane_name",
                "is_ambiguous",
                "assigned_truth_length_m",
                "lane_confidence",
            ]
        ],
        on=["node_id", "source_index"],
        how="left",
    )
    predictions["presence_probability"] = presence_prob
    predictions["predicted_presence"] = presence_prob >= best_threshold
    lane_pred = lane_prob.argmax(axis=1).astype(int)
    predictions["predicted_lane_class"] = lane_pred
    predictions["predicted_lane_name"] = [FIXED_ROAD_OFFSET_LANES[i] for i in lane_pred]
    for lane_index, lane_name in enumerate(FIXED_ROAD_OFFSET_LANES):
        predictions[f"prob_lane_{lane_name}"] = lane_prob[:, lane_index]
    predictions["lane_correct"] = (
        (predictions["y"].astype(int) == 1)
        & (predictions["lane_class"].astype(int) == predictions["predicted_lane_class"].astype(int))
    )

    threshold_table.to_csv(output_dir / "threshold_tuning_val.csv", index=False)
    metrics.to_csv(output_dir / "split_metrics.csv", index=False)
    pd.DataFrame(losses).to_csv(output_dir / "training_losses.csv", index=False)
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
    predictions.drop(columns=["geometry"], errors="ignore").to_csv(
        output_dir / "road_offset_lane_node_predictions.csv",
        index=False,
    )
    _write_geojson(predictions, output_dir / "road_offset_lane_node_predictions.geojson")
    torch.save(model.cpu().state_dict(), output_dir / "road_offset_lane_gnn_state.pt")

    summary = {
        "workstream": "Codex",
        "description": "road-node GraphSAGE presence + fixed five-lane offset classifier",
        "runtime_sec": time.perf_counter() - start,
        "device_report": device_report,
        "training_device": str(device),
        "n_aois": len(parts),
        "n_source_roads": int(len(labels)),
        "n_message_edges": int(edge_index.shape[1]),
        "n_features": int(features.shape[1]),
        "lane_names": list(FIXED_ROAD_OFFSET_LANES),
        "exclude_ambiguous_lane_loss": bool(args.exclude_ambiguous_lane_loss),
        "exclude_ambiguous_lane_eval": bool(args.exclude_ambiguous_lane_eval),
        "best_threshold_from_val": best_threshold,
        "gnn_layer_type": str(args.gnn_layer_type),
        "epochs": int(args.epochs),
        "source_config_paths": [_relative(path) for path in config_paths],
    }
    test_rows = metrics[metrics["split"] == "test"]
    if not test_rows.empty:
        summary["test_metrics"] = test_rows.iloc[0].to_dict()
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    print(f"Best threshold from val: {best_threshold:.3f}", flush=True)
    print("Split metrics:", flush=True)
    keep = [
        "split",
        "n_source_roads",
        "presence_roc_auc",
        "presence_pr_auc",
        "presence_f1",
        "presence_precision",
        "presence_recall",
        "lane_eval_count",
        "lane_accuracy",
        "lane_macro_f1",
        "joint_presence_lane_accuracy",
    ]
    print(metrics[[column for column in keep if column in metrics.columns]].to_string(index=False), flush=True)
    print(f"Wrote outputs to {_relative(output_dir)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
