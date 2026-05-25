"""Train a skeleton-building-road hetero GNN for sewer-main presence.

Workstream: Codex

This is the paper-principled anchor-free variant:

    SkeletonSegment -> prediction support
    Building        -> demand/context support
    RoadSegment     -> road-context support

The heterograph includes building-building, building-skeleton, building-road,
skeleton-road, road-road, and skeleton-skeleton relations. Utility truth is used
only for labels and evaluation.
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
from shapely.ops import unary_union
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, GraphConv, SAGEConv, to_hetero
import torch_geometric.transforms as T

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipe_network_completion.anchor_free.config import load_anchor_free_config  # noqa: E402
from pipe_network_completion.anchor_free.evaluation import compute_edge_metrics  # noqa: E402
from pipe_network_completion.anchor_free.features import (  # noqa: E402
    assert_no_anchor_features,
    standardize_features,
)
from pipe_network_completion.anchor_free.model import (  # noqa: E402
    resolve_torch_device,
    torch_device_report,
)
from pipe_network_completion.anchor_free.skeleton_context_graph import (  # noqa: E402
    SkeletonContextGraph,
    build_skeleton_context_features,
    build_skeleton_context_graph,
)


@dataclass(frozen=True)
class ContextPart:
    aoi_id: str
    split: str
    graph: SkeletonContextGraph
    features: dict[str, pd.DataFrame]
    labels: pd.DataFrame
    truth: gpd.GeoDataFrame


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
                raise FileNotFoundError(path)
            paths.append(path)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            deduped.append(path)
            seen.add(resolved)
    return deduped


def _read_layer(path_value, target_crs=None) -> gpd.GeoDataFrame:
    if path_value in (None, ""):
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)
    values = path_value if isinstance(path_value, (list, tuple)) else [path_value]
    frames: list[gpd.GeoDataFrame] = []
    for value in values:
        path = _resolve(value)
        if not path.exists():
            continue
        frame = gpd.read_file(path)
        if target_crs is not None and frame.crs is not None and str(frame.crs) != str(target_crs):
            frame = frame.to_crs(target_crs)
        frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
        frames.append(frame)
    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs)


def _label_skeleton_segments(
    skeleton: gpd.GeoDataFrame,
    truth: gpd.GeoDataFrame,
    *,
    label_buffer_m: float,
    min_truth_length_m: float,
    min_overlap_ratio: float,
) -> pd.DataFrame:
    labels = pd.DataFrame({"skeleton_id": skeleton["skeleton_id"].astype(int).to_numpy()})
    labels["overlap_length_m"] = 0.0
    if skeleton.empty or truth.empty:
        labels["overlap_ratio"] = 0.0
        labels["y"] = 0
        return labels
    if skeleton.crs is not None and truth.crs is not None and str(skeleton.crs) != str(truth.crs):
        truth = truth.to_crs(skeleton.crs)
    buffers = gpd.GeoDataFrame(
        {"skeleton_id": skeleton["skeleton_id"].astype(int).to_numpy()},
        geometry=skeleton.geometry.buffer(float(label_buffer_m)),
        crs=skeleton.crs,
    )
    joined = gpd.sjoin(truth[["geometry"]], buffers, predicate="intersects", how="inner")
    buffer_lookup = dict(zip(buffers["skeleton_id"].astype(int), buffers.geometry))
    overlaps: dict[int, float] = {}
    for row in joined.itertuples(index=False):
        skeleton_id = int(row.skeleton_id)
        length = float(row.geometry.intersection(buffer_lookup[skeleton_id]).length)
        overlaps[skeleton_id] = overlaps.get(skeleton_id, 0.0) + length
    labels["overlap_length_m"] = [overlaps.get(int(sid), 0.0) for sid in labels["skeleton_id"]]
    lengths = skeleton.set_index("skeleton_id")["length_m"].reindex(labels["skeleton_id"]).to_numpy(dtype=float)
    labels["overlap_ratio"] = labels["overlap_length_m"].to_numpy(dtype=float) / np.maximum(lengths, 1e-9)
    labels["y"] = (
        (labels["overlap_length_m"] >= float(min_truth_length_m))
        & (labels["overlap_ratio"] >= float(min_overlap_ratio))
    ).astype(int)
    return labels


def _shift_edges(edge_index: np.ndarray, source_offset: int, target_offset: int) -> np.ndarray:
    if edge_index.size == 0:
        return np.zeros((2, 0), dtype=np.int64)
    shifted = np.asarray(edge_index, dtype=np.int64).copy()
    shifted[0] += int(source_offset)
    shifted[1] += int(target_offset)
    return shifted


def _split_key(value: str) -> str:
    value = str(value).lower()
    return "val" if value in {"val", "valid", "validation"} else value if value in {"train", "test"} else "train"


def _add_train_quantile_bins(
    features: pd.DataFrame,
    *,
    column: str,
    train_index: np.ndarray,
    prefix: str,
    n_bins: int = 5,
) -> pd.DataFrame:
    if column not in features.columns or len(features) == 0:
        return features
    train_values = pd.to_numeric(features.iloc[list(train_index)][column], errors="coerce").dropna().to_numpy(dtype=float)
    if train_values.size < 2 or np.nanmax(train_values) <= np.nanmin(train_values):
        return features
    quantiles = np.unique(np.quantile(train_values, np.linspace(0.0, 1.0, int(n_bins) + 1)))
    if quantiles.size < 3:
        return features
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf
    values = pd.to_numeric(features[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    bins = np.digitize(values, quantiles[1:-1], right=True)
    for bin_id in range(int(n_bins)):
        features[f"{prefix}_bin_{bin_id}"] = (bins == bin_id).astype(float)
    return features


def _prepare_part(
    config_path: Path,
    *,
    include_drainage: bool,
    include_coords: bool,
    building_building_radius_m: float,
    building_skeleton_radius_m: float,
    building_road_radius_m: float,
    skeleton_road_radius_m: float,
    building_knn: int,
    context_knn: int,
    label_buffer_m: float,
    min_truth_length_m: float,
    min_overlap_ratio: float,
) -> ContextPart:
    config = load_anchor_free_config(config_path)
    data = dict(config.get("data", {}))
    graph_config = dict(config.get("graph", {}))
    aoi_config = dict(config.get("aoi", {}))
    target_crs = graph_config.get("target_crs", "EPSG:28356")

    roads = _read_layer(data.get("roads_path"), target_crs)
    buildings = _read_layer(data.get("buildings_path"), target_crs)
    building_points = _read_layer(data.get("building_points_path"), target_crs)
    truth = _read_layer(data.get("utility_truth_path"), target_crs)
    drainage = _read_layer(data.get("watercourse_drainage_lines_path"), target_crs)
    if drainage.empty:
        drainage = _read_layer(data.get("watercourse_corridor_centrelines_path"), target_crs)

    graph = build_skeleton_context_graph(
        roads=roads,
        building_points=building_points,
        buildings=buildings,
        drainage_lines=drainage,
        target_crs=target_crs,
        include_drainage=include_drainage,
        building_building_radius_m=building_building_radius_m,
        building_skeleton_radius_m=building_skeleton_radius_m,
        building_road_radius_m=building_road_radius_m,
        skeleton_road_radius_m=skeleton_road_radius_m,
        building_knn=building_knn,
        context_knn=context_knn,
    )
    features = build_skeleton_context_features(graph, include_coords=include_coords)
    labels = _label_skeleton_segments(
        graph.skeleton_segments,
        truth,
        label_buffer_m=label_buffer_m,
        min_truth_length_m=min_truth_length_m,
        min_overlap_ratio=min_overlap_ratio,
    )
    return ContextPart(
        aoi_id=str(aoi_config.get("aoi_id", config_path.stem)),
        split=_split_key(str(aoi_config.get("split", "train"))),
        graph=graph,
        features=features,
        labels=labels,
        truth=truth,
    )


def _concat_edges(edge_arrays: list[np.ndarray]) -> np.ndarray:
    non_empty = [edge for edge in edge_arrays if edge.size]
    return np.concatenate(non_empty, axis=1) if non_empty else np.zeros((2, 0), dtype=np.int64)


def _combine_parts(parts: list[ContextPart]) -> tuple[
    SkeletonContextGraph,
    dict[str, pd.DataFrame],
    pd.DataFrame,
    dict[str, dict[str, np.ndarray]],
]:
    skeleton_frames = []
    building_frames = []
    road_frames = []
    labels = []
    feature_frames = {"SkeletonSegment": [], "Building": [], "RoadSegment": []}
    edge_arrays = {
        "skeleton_adjacent_skeleton": [],
        "building_near_building": [],
        "building_near_skeleton": [],
        "building_near_road": [],
        "skeleton_near_road": [],
        "road_adjacent_road": [],
    }
    split_indices = {
        "SkeletonSegment": {"train": [], "val": [], "test": []},
        "Building": {"train": [], "val": [], "test": []},
        "RoadSegment": {"train": [], "val": [], "test": []},
    }

    skeleton_offset = 0
    building_offset = 0
    road_offset = 0
    for part in parts:
        graph = part.graph
        split = _split_key(part.split)
        n_skeleton = len(graph.skeleton_segments)
        n_building = len(graph.buildings)
        n_road = len(graph.road_segments)

        skeleton = graph.skeleton_segments.copy()
        skeleton["skeleton_id"] = skeleton["skeleton_id"].astype(int) + skeleton_offset
        skeleton["aoi_id"] = part.aoi_id
        skeleton["aoi_split"] = split
        skeleton_frames.append(skeleton)

        building = graph.buildings.copy()
        building["building_id"] = building["building_id"].astype(int) + building_offset
        building["aoi_id"] = part.aoi_id
        building["aoi_split"] = split
        building_frames.append(building)

        road = graph.road_segments.copy()
        road["road_id"] = road["road_id"].astype(int) + road_offset
        road["aoi_id"] = part.aoi_id
        road["aoi_split"] = split
        road_frames.append(road)

        label = part.labels.copy()
        label["skeleton_id"] = label["skeleton_id"].astype(int) + skeleton_offset
        label["aoi_id"] = part.aoi_id
        label["aoi_split"] = split
        labels.append(label)

        skeleton_features = part.features["SkeletonSegment"].copy()
        skeleton_features.index = skeleton_features.index.astype(int) + skeleton_offset
        feature_frames["SkeletonSegment"].append(skeleton_features)

        building_features = part.features["Building"].copy()
        building_features.index = building_features.index.astype(int) + building_offset
        feature_frames["Building"].append(building_features)

        road_features = part.features["RoadSegment"].copy()
        road_features.index = road_features.index.astype(int) + road_offset
        feature_frames["RoadSegment"].append(road_features)

        edge_arrays["skeleton_adjacent_skeleton"].append(
            _shift_edges(graph.skeleton_adjacent_skeleton, skeleton_offset, skeleton_offset)
        )
        edge_arrays["building_near_building"].append(
            _shift_edges(graph.building_near_building, building_offset, building_offset)
        )
        edge_arrays["building_near_skeleton"].append(
            _shift_edges(graph.building_near_skeleton, building_offset, skeleton_offset)
        )
        edge_arrays["building_near_road"].append(
            _shift_edges(graph.building_near_road, building_offset, road_offset)
        )
        edge_arrays["skeleton_near_road"].append(
            _shift_edges(graph.skeleton_near_road, skeleton_offset, road_offset)
        )
        edge_arrays["road_adjacent_road"].append(
            _shift_edges(graph.road_adjacent_road, road_offset, road_offset)
        )

        split_indices["SkeletonSegment"][split].extend(range(skeleton_offset, skeleton_offset + n_skeleton))
        split_indices["Building"][split].extend(range(building_offset, building_offset + n_building))
        split_indices["RoadSegment"][split].extend(range(road_offset, road_offset + n_road))

        skeleton_offset += n_skeleton
        building_offset += n_building
        road_offset += n_road

    features: dict[str, pd.DataFrame] = {}
    for node_type, frames in feature_frames.items():
        if frames:
            table = pd.concat(frames, axis=0).sort_index()
        else:
            table = pd.DataFrame()
        table = table.reindex(sorted(table.columns), axis=1).fillna(0.0)
        assert_no_anchor_features(table.columns)
        features[node_type] = table

    for node_type, by_split in split_indices.items():
        for split, values in by_split.items():
            by_split[split] = np.asarray(values, dtype=int)

    features["SkeletonSegment"] = _add_train_quantile_bins(
        features["SkeletonSegment"],
        column="length_m",
        train_index=split_indices["SkeletonSegment"]["train"],
        prefix="length",
    )
    features["RoadSegment"] = _add_train_quantile_bins(
        features["RoadSegment"],
        column="length_m",
        train_index=split_indices["RoadSegment"]["train"],
        prefix="length",
    )
    if "log1p_footprint_area_m2" in features["Building"].columns:
        features["Building"] = _add_train_quantile_bins(
            features["Building"],
            column="log1p_footprint_area_m2",
            train_index=split_indices["Building"]["train"],
            prefix="building_area",
        )

    skeleton_gdf = gpd.GeoDataFrame(pd.concat(skeleton_frames, ignore_index=True), geometry="geometry", crs=parts[0].graph.skeleton_segments.crs)
    building_gdf = gpd.GeoDataFrame(pd.concat(building_frames, ignore_index=True), geometry="geometry", crs=parts[0].graph.buildings.crs)
    road_gdf = gpd.GeoDataFrame(pd.concat(road_frames, ignore_index=True), geometry="geometry", crs=parts[0].graph.road_segments.crs)
    label_frame = pd.concat(labels, ignore_index=True).sort_values("skeleton_id").reset_index(drop=True)
    graph = SkeletonContextGraph(
        skeleton_segments=skeleton_gdf.sort_values("skeleton_id").reset_index(drop=True),
        buildings=building_gdf.sort_values("building_id").reset_index(drop=True),
        road_segments=road_gdf.sort_values("road_id").reset_index(drop=True),
        skeleton_adjacent_skeleton=_concat_edges(edge_arrays["skeleton_adjacent_skeleton"]),
        building_near_building=_concat_edges(edge_arrays["building_near_building"]),
        building_near_skeleton=_concat_edges(edge_arrays["building_near_skeleton"]),
        building_near_road=_concat_edges(edge_arrays["building_near_road"]),
        skeleton_near_road=_concat_edges(edge_arrays["skeleton_near_road"]),
        road_adjacent_road=_concat_edges(edge_arrays["road_adjacent_road"]),
        crs=parts[0].graph.crs,
        metadata={
            "n_aois": len(parts),
            "n_skeleton_segments": int(skeleton_offset),
            "n_buildings": int(building_offset),
            "n_road_segments": int(road_offset),
        },
    )
    return graph, features, label_frame, split_indices


def _scale_feature_tables(
    features: dict[str, pd.DataFrame],
    split_indices: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    scaled = {}
    scaling_rows = {}
    for node_type, table in features.items():
        train_index = split_indices[node_type]["train"]
        scaled_table, mean, std = standardize_features(table, train_index=train_index)
        scaled[node_type] = scaled_table
        scaling_rows[node_type] = pd.DataFrame(
            {"feature": table.columns, "mean": mean.reindex(table.columns).to_numpy(dtype=float), "std": std.reindex(table.columns).to_numpy(dtype=float)}
        )
    return scaled, scaling_rows


def _build_pyg_data(
    graph: SkeletonContextGraph,
    features: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
) -> HeteroData:
    data = HeteroData()
    for node_type, id_column in [
        ("SkeletonSegment", "skeleton_id"),
        ("Building", "building_id"),
        ("RoadSegment", "road_id"),
    ]:
        data[node_type].x = torch.tensor(features[node_type].to_numpy(dtype=float), dtype=torch.float32)
        data[node_type].node_id = torch.tensor(features[node_type].index.to_numpy(dtype=np.int64), dtype=torch.long)

    data["SkeletonSegment", "adjacent", "SkeletonSegment"].edge_index = torch.tensor(graph.skeleton_adjacent_skeleton, dtype=torch.long)
    data["Building", "near", "Building"].edge_index = torch.tensor(graph.building_near_building, dtype=torch.long)
    data["Building", "near", "SkeletonSegment"].edge_index = torch.tensor(graph.building_near_skeleton, dtype=torch.long)
    data["Building", "near", "RoadSegment"].edge_index = torch.tensor(graph.building_near_road, dtype=torch.long)
    data["SkeletonSegment", "near", "RoadSegment"].edge_index = torch.tensor(graph.skeleton_near_road, dtype=torch.long)
    data["RoadSegment", "adjacent", "RoadSegment"].edge_index = torch.tensor(graph.road_adjacent_road, dtype=torch.long)
    data = T.ToUndirected()(data)
    data["SkeletonSegment"].y = torch.tensor(labels.sort_values("skeleton_id")["y"].to_numpy(dtype=float), dtype=torch.float32)
    return data


def _normalise_layer_type(layer_type: str) -> str:
    layer = str(layer_type or "sage").lower().replace("-", "_")
    if layer in {"sage", "graphsage"}:
        return "sage"
    if layer in {"gat", "graph_attention"}:
        return "gat"
    if layer in {"graphconv", "graph_conv", "gcn_style", "gcn_like"}:
        return "graphconv"
    raise ValueError(f"Unsupported layer type: {layer_type}")


def _make_conv(layer_type: str, hidden_dim: int, dropout: float, gat_heads: int):
    layer = _normalise_layer_type(layer_type)
    if layer == "sage":
        return SAGEConv((-1, -1), hidden_dim)
    if layer == "gat":
        return GATConv((-1, -1), hidden_dim, heads=max(int(gat_heads), 1), concat=False, dropout=float(dropout), add_self_loops=False)
    if layer == "graphconv":
        return GraphConv((-1, -1), hidden_dim)
    raise AssertionError(layer)


class _BaseHeteroGNN(torch.nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float, layer_type: str, gat_heads: int):
        super().__init__()
        self.dropout = float(dropout)
        self.convs = torch.nn.ModuleList(
            [_make_conv(layer_type, hidden_dim, dropout, gat_heads) for _ in range(max(int(num_layers), 1))]
        )

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        return x


class SkeletonContextGNN(torch.nn.Module):
    def __init__(
        self,
        *,
        metadata,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        layer_type: str = "sage",
        gat_heads: int = 1,
    ):
        super().__init__()
        self.dropout = float(dropout)
        base = _BaseHeteroGNN(hidden_dim, num_layers, dropout, layer_type, gat_heads)
        self.gnn = to_hetero(base, metadata=metadata, aggr="sum")
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        h = self.gnn(data.x_dict, data.edge_index_dict)["SkeletonSegment"]
        return self.head(h).reshape(-1)


def _train_model(
    data: HeteroData,
    *,
    train_index: np.ndarray,
    seed: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    device: str,
    layer_type: str,
    gat_heads: int,
) -> tuple[SkeletonContextGNN, np.ndarray, list[float], str]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    resolved = resolve_torch_device(device)
    data = data.to(resolved)
    model = SkeletonContextGNN(
        metadata=data.metadata(),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        layer_type=layer_type,
        gat_heads=gat_heads,
    ).to(resolved)
    with torch.no_grad():
        _ = model(data)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_tensor = torch.tensor(train_index, dtype=torch.long, device=resolved)
    train_y = data["SkeletonSegment"].y[train_tensor]
    positives = torch.clamp(train_y.sum(), min=1.0)
    negatives = torch.clamp(train_y.numel() - train_y.sum(), min=1.0)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=(negatives / positives))
    losses = []
    for _ in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits[train_tensor], data["SkeletonSegment"].y[train_tensor])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(data)).detach().cpu().numpy()
    return model.cpu(), np.asarray(probabilities, dtype=float), losses, str(resolved)


def _select_threshold(y: np.ndarray, p: np.ndarray, val_index: np.ndarray, thresholds: Iterable[float]) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in thresholds:
        rows.append({"threshold": float(threshold), **compute_edge_metrics(y[val_index], p[val_index], threshold=float(threshold))})
    table = pd.DataFrame(rows)
    if table.empty:
        return 0.5, table
    best = table.sort_values(["f1", "pr_auc", "balanced_accuracy"], ascending=[False, False, False]).iloc[0]
    return float(best["threshold"]), table


def _split_metrics(
    graph: SkeletonContextGraph,
    labels: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    threshold: float,
    split_indices: dict[str, np.ndarray],
) -> pd.DataFrame:
    y = labels.sort_values("skeleton_id")["y"].to_numpy(dtype=int)
    lengths = graph.skeleton_segments.sort_values("skeleton_id")["length_m"].to_numpy(dtype=float)
    rows = []
    for split, idx in split_indices.items():
        idx = np.asarray(idx, dtype=int)
        if len(idx) == 0:
            continue
        metrics = compute_edge_metrics(y[idx], probabilities[idx], threshold=threshold)
        selected = probabilities[idx] >= float(threshold)
        tp_len = float(lengths[idx][selected & (y[idx] == 1)].sum())
        fp_len = float(lengths[idx][selected & (y[idx] == 0)].sum())
        fn_len = float(lengths[idx][(~selected) & (y[idx] == 1)].sum())
        precision_len = tp_len / (tp_len + fp_len) if tp_len + fp_len > 0 else 0.0
        recall_len = tp_len / (tp_len + fn_len) if tp_len + fn_len > 0 else 0.0
        f1_len = 2 * precision_len * recall_len / (precision_len + recall_len) if precision_len + recall_len > 0 else 0.0
        rows.append(
            {
                "split": split,
                "n_skeleton_segments": int(len(idx)),
                **{f"edge_{key}": value for key, value in metrics.items()},
                "length_precision": precision_len,
                "length_recall": recall_len,
                "length_f1": f1_len,
                "true_positive_length_m": tp_len,
                "false_positive_length_m": fp_len,
                "false_negative_length_m": fn_len,
            }
        )
    return pd.DataFrame(rows)


def _write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gdf.empty:
        path.write_text('{"type":"FeatureCollection","features":[]}\n', encoding="utf-8")
    else:
        path.write_text(gdf.to_json(drop_id=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gnn-layer-type", choices=["sage", "gat", "graphconv"], default="sage")
    parser.add_argument("--gat-heads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--include-drainage", action="store_true")
    parser.add_argument("--include-coords", action="store_true")
    parser.add_argument("--building-building-radius-m", type=float, default=80.0)
    parser.add_argument("--building-skeleton-radius-m", type=float, default=80.0)
    parser.add_argument("--building-road-radius-m", type=float, default=80.0)
    parser.add_argument("--skeleton-road-radius-m", type=float, default=30.0)
    parser.add_argument("--building-knn", type=int, default=4)
    parser.add_argument("--context-knn", type=int, default=3)
    parser.add_argument("--label-buffer-m", type=float, default=30.0)
    parser.add_argument("--min-truth-length-m", type=float, default=10.0)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.05)
    parser.add_argument("--threshold-grid", nargs="+", type=float, default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device_report = torch_device_report()
    print(f"GPU preflight: {device_report}", flush=True)

    config_paths = _expand_configs(args.configs)
    if args.max_configs is not None:
        config_paths = config_paths[: int(args.max_configs)]
    parts = []
    for config_path in config_paths:
        part = _prepare_part(
            config_path,
            include_drainage=bool(args.include_drainage),
            include_coords=bool(args.include_coords),
            building_building_radius_m=float(args.building_building_radius_m),
            building_skeleton_radius_m=float(args.building_skeleton_radius_m),
            building_road_radius_m=float(args.building_road_radius_m),
            skeleton_road_radius_m=float(args.skeleton_road_radius_m),
            building_knn=int(args.building_knn),
            context_knn=int(args.context_knn),
            label_buffer_m=float(args.label_buffer_m),
            min_truth_length_m=float(args.min_truth_length_m),
            min_overlap_ratio=float(args.min_overlap_ratio),
        )
        parts.append(part)
        print(
            f"prepared {part.aoi_id} ({part.split}): skeleton={len(part.graph.skeleton_segments)} "
            f"buildings={len(part.graph.buildings)} roads={len(part.graph.road_segments)} "
            f"positives={int(part.labels['y'].sum())}/{len(part.labels)}",
            flush=True,
        )

    graph, features, labels, split_indices_all = _combine_parts(parts)
    scaled_features, scaling = _scale_feature_tables(features, split_indices_all)
    data = _build_pyg_data(graph, scaled_features, labels)
    train_index = split_indices_all["SkeletonSegment"]["train"]
    val_index = split_indices_all["SkeletonSegment"]["val"]
    test_index = split_indices_all["SkeletonSegment"]["test"]

    model, probabilities, losses, training_device = _train_model(
        data,
        train_index=train_index,
        seed=int(args.seed),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        epochs=int(args.epochs),
        device=str(args.device),
        layer_type=str(args.gnn_layer_type),
        gat_heads=int(args.gat_heads),
    )

    y = labels.sort_values("skeleton_id")["y"].to_numpy(dtype=int)
    best_threshold, threshold_table = _select_threshold(y, probabilities, val_index, args.threshold_grid)
    split_metrics = _split_metrics(
        graph,
        labels,
        probabilities,
        threshold=best_threshold,
        split_indices={
            "train": train_index,
            "val": val_index,
            "test": test_index,
        },
    )

    predictions = graph.skeleton_segments.merge(labels, on="skeleton_id", how="left")
    predictions["presence_probability"] = probabilities
    predictions["predicted_presence"] = probabilities >= float(best_threshold)
    _write_geojson(predictions, output_dir / "skeleton_context_predictions.geojson")
    _write_geojson(
        predictions[predictions["predicted_presence"].astype(bool)].copy(),
        output_dir / "decoded_skeleton_context_network.geojson",
    )
    predictions.drop(columns=["geometry"], errors="ignore").to_csv(output_dir / "skeleton_context_predictions.csv", index=False)
    threshold_table.to_csv(output_dir / "threshold_tuning_val.csv", index=False)
    split_metrics.to_csv(output_dir / "split_metrics.csv", index=False)
    pd.DataFrame({"epoch": np.arange(1, len(losses) + 1), "loss": losses}).to_csv(output_dir / "training_losses.csv", index=False)
    for node_type, table in features.items():
        table.to_csv(output_dir / f"{node_type.lower()}_features_unscaled.csv")
        scaling[node_type].to_csv(output_dir / f"{node_type.lower()}_feature_scaling.csv", index=False)
    torch.save(model.state_dict(), output_dir / "skeleton_context_gnn_state.pt")

    summary = {
        "workstream": "Codex",
        "description": "anchor-free skeleton-building-road hetero GNN",
        "truth_target": "complete sewer mains: gravity mains plus pressure mains",
        "runtime_sec": time.perf_counter() - start,
        "device_report": device_report,
        "training_device": training_device,
        "epochs": int(args.epochs),
        "gnn_layer_type": str(args.gnn_layer_type),
        "include_coords": bool(args.include_coords),
        "include_drainage": bool(args.include_drainage),
        "best_threshold_from_val": float(best_threshold),
        "n_aois": len(parts),
        "n_skeleton_segments": int(len(graph.skeleton_segments)),
        "n_buildings": int(len(graph.buildings)),
        "n_road_segments": int(len(graph.road_segments)),
        "edge_counts": {
            "skeleton_adjacent_skeleton": int(graph.skeleton_adjacent_skeleton.shape[1]),
            "building_near_building": int(graph.building_near_building.shape[1]),
            "building_near_skeleton": int(graph.building_near_skeleton.shape[1]),
            "building_near_road": int(graph.building_near_road.shape[1]),
            "skeleton_near_road": int(graph.skeleton_near_road.shape[1]),
            "road_adjacent_road": int(graph.road_adjacent_road.shape[1]),
        },
        "feature_counts": {node_type: int(table.shape[1]) for node_type, table in features.items()},
        "source_config_paths": [_relative(path) for path in config_paths],
    }
    test_row = split_metrics[split_metrics["split"] == "test"]
    if not test_row.empty:
        summary["test_metrics"] = test_row.iloc[0].to_dict()
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    print(f"Best threshold from val: {best_threshold:.3f}", flush=True)
    print(
        split_metrics[
            [
                "split",
                "n_skeleton_segments",
                "edge_roc_auc",
                "edge_pr_auc",
                "edge_f1",
                "edge_precision",
                "edge_recall",
                "edge_positive_prevalence",
                "edge_all_positive_f1",
                "length_f1",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(f"Wrote outputs to {_relative(output_dir)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
