"""Hybrid U-Net support field to skeleton-graph GNN utilities.

Workstream: Codex

The functions in this module keep absolute map coordinates out of the model
features. Coordinates are only used to write inspectable output geometries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import distance_transform_edt, label as label_components
from shapely.geometry import LineString
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from pipe_network_completion.anchor_free.features import assert_no_anchor_features
from pipe_network_completion.anchor_free.unet_segmentation import (
    NpzSegmentationDataset,
    binary_pixel_metrics,
)


SPLIT_TO_CODE = {"train": 0, "val": 1, "valid": 1, "validation": 1, "test": 2}
CODE_TO_SPLIT = {0: "train", 1: "val", 2: "test"}


@dataclass(frozen=True)
class AoiRasterMosaic:
    aoi_id: str
    split: str
    x: np.ndarray
    y: np.ndarray
    probability: np.ndarray
    channel_names: list[str]
    bounds: tuple[float, float, float, float]
    pixel_size_m: float
    cnn_features: np.ndarray | None = None
    cnn_feature_names: list[str] | None = None


@dataclass(frozen=True)
class SkeletonGraphPart:
    aoi_id: str
    split: str
    data: Data
    edge_table: gpd.GeoDataFrame
    support_metrics: dict[str, float | int | str]
    node_feature_names: list[str]
    edge_feature_names: list[str]


def zhang_suen_thinning(mask: np.ndarray, *, max_iterations: int | None = None) -> np.ndarray:
    """Return a one-pixel skeleton using Zhang-Suen thinning."""

    image = np.asarray(mask, dtype=bool).copy()
    if image.size == 0 or not image.any():
        return image

    iteration = 0
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            padded = np.pad(image, 1, mode="constant", constant_values=False)
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            core = padded[1:-1, 1:-1]

            neighbor_count = (
                p2.astype(np.uint8)
                + p3.astype(np.uint8)
                + p4.astype(np.uint8)
                + p5.astype(np.uint8)
                + p6.astype(np.uint8)
                + p7.astype(np.uint8)
                + p8.astype(np.uint8)
                + p9.astype(np.uint8)
            )
            sequence = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            transitions = np.zeros_like(neighbor_count, dtype=np.uint8)
            for previous, current in zip(sequence[:-1], sequence[1:]):
                transitions += ((~previous) & current).astype(np.uint8)

            if step == 0:
                removable = (
                    core
                    & (neighbor_count >= 2)
                    & (neighbor_count <= 6)
                    & (transitions == 1)
                    & ~(p2 & p4 & p6)
                    & ~(p4 & p6 & p8)
                )
            else:
                removable = (
                    core
                    & (neighbor_count >= 2)
                    & (neighbor_count <= 6)
                    & (transitions == 1)
                    & ~(p2 & p4 & p8)
                    & ~(p2 & p6 & p8)
                )

            if removable.any():
                image[removable] = False
                changed = True

        iteration += 1
        if max_iterations is not None and iteration >= int(max_iterations):
            break
    return image


@torch.no_grad()
def predict_tile_probabilities(
    model: torch.nn.Module,
    index: pd.DataFrame,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> list[np.ndarray]:
    dataset = NpzSegmentationDataset(index, mean=mean, std=std)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    model.eval()
    probabilities: list[np.ndarray] = []
    for x, _ in loader:
        logits = model(x.to(device)).detach().cpu()
        for probability in torch.sigmoid(logits).numpy():
            probabilities.append(probability[0].astype("float32"))
    return probabilities


@torch.no_grad()
def predict_tile_cnn_outputs(
    model: torch.nn.Module,
    index: pd.DataFrame,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Return U-Net probabilities and full-resolution decoder CNN features."""

    dataset = NpzSegmentationDataset(index, mean=mean, std=std)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    model.eval()
    probabilities: list[np.ndarray] = []
    cnn_features: list[np.ndarray] = []
    feature_names: list[str] | None = None
    for x, _ in loader:
        x = x.to(device)
        features = model.encoder(x)
        decoder_features = model.decoder(*features)
        logits = model.segmentation_head(decoder_features).detach().cpu()
        decoder_np = decoder_features.detach().cpu().numpy().astype("float32")
        if feature_names is None:
            feature_names = [f"cnn_decoder_{idx:02d}" for idx in range(decoder_np.shape[1])]
        for probability, feature_map in zip(torch.sigmoid(logits).numpy(), decoder_np):
            probabilities.append(probability[0].astype("float32"))
            cnn_features.append(feature_map.astype("float32"))
    return probabilities, cnn_features, feature_names or []


def build_aoi_mosaic(
    aoi_index: pd.DataFrame,
    probabilities: list[np.ndarray],
    cnn_features: list[np.ndarray] | None = None,
    cnn_feature_names: list[str] | None = None,
) -> AoiRasterMosaic:
    """Mosaic one AOI in north-up order from grid tiles."""

    if aoi_index.empty:
        raise ValueError("Cannot mosaic an empty AOI index.")
    if len(aoi_index) != len(probabilities):
        raise ValueError("AOI index and probability list length mismatch.")
    if cnn_features is not None and len(aoi_index) != len(cnn_features):
        raise ValueError("AOI index and CNN feature list length mismatch.")

    rows = [int(value) for value in aoi_index["grid_row"]]
    cols = [int(value) for value in aoi_index["grid_col"]]
    if min(rows) < 0 or min(cols) < 0:
        rows = [0 for _ in rows]
        cols = [0 for _ in cols]
    n_rows = max(rows) + 1
    n_cols = max(cols) + 1

    first_path = Path(str(aoi_index.iloc[0]["tile_path"]))
    with np.load(first_path, allow_pickle=True) as first:
        tile_x = first["x"].astype("float32")
        channel_names = [str(value) for value in first["channel_names"].tolist()]
        tile_h, tile_w = tile_x.shape[-2:]
        pixel_size_m = float(first["pixel_size_m"])

    x_mosaic = np.zeros((tile_x.shape[0], n_rows * tile_h, n_cols * tile_w), dtype="float32")
    y_mosaic = np.zeros((n_rows * tile_h, n_cols * tile_w), dtype="float32")
    p_mosaic = np.zeros((n_rows * tile_h, n_cols * tile_w), dtype="float32")
    cnn_mosaic = None
    if cnn_features is not None and len(cnn_features) > 0:
        cnn_mosaic = np.zeros((cnn_features[0].shape[0], n_rows * tile_h, n_cols * tile_w), dtype="float32")
    bounds_values: list[np.ndarray] = []

    for tile_idx, ((_, row), probability) in enumerate(zip(aoi_index.iterrows(), probabilities)):
        grid_row = int(row["grid_row"])
        grid_col = int(row["grid_col"])
        display_row = (n_rows - 1) - grid_row
        path = Path(str(row["tile_path"]))
        with np.load(path, allow_pickle=True) as data:
            x = data["x"].astype("float32")
            y = data["y"].astype("float32")
            bounds_values.append(data["bounds"].astype("float64"))
        r0 = display_row * tile_h
        c0 = grid_col * tile_w
        x_mosaic[:, r0 : r0 + tile_h, c0 : c0 + tile_w] = x
        y_mosaic[r0 : r0 + tile_h, c0 : c0 + tile_w] = y
        p_mosaic[r0 : r0 + tile_h, c0 : c0 + tile_w] = probability
        if cnn_mosaic is not None and cnn_features is not None:
            cnn_mosaic[:, r0 : r0 + tile_h, c0 : c0 + tile_w] = cnn_features[tile_idx]

    bounds_array = np.vstack(bounds_values)
    bounds = (
        float(bounds_array[:, 0].min()),
        float(bounds_array[:, 1].min()),
        float(bounds_array[:, 2].max()),
        float(bounds_array[:, 3].max()),
    )
    return AoiRasterMosaic(
        aoi_id=str(aoi_index.iloc[0]["aoi_id"]),
        split=str(aoi_index.iloc[0]["split"]),
        x=x_mosaic,
        y=y_mosaic,
        probability=p_mosaic,
        channel_names=channel_names,
        bounds=bounds,
        pixel_size_m=pixel_size_m,
        cnn_features=cnn_mosaic,
        cnn_feature_names=list(cnn_feature_names or []),
    )


def _remove_small_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    if min_pixels <= 1 or not mask.any():
        return mask.astype(bool)
    labels, n_components = label_components(mask, structure=np.ones((3, 3), dtype=int))
    if n_components == 0:
        return mask.astype(bool)
    counts = np.bincount(labels.reshape(-1))
    keep = counts >= int(min_pixels)
    keep[0] = False
    return keep[labels]


def _world_xy(row: int, col: int, bounds: tuple[float, float, float, float], pixel_size_m: float) -> tuple[float, float]:
    xmin, _, _, ymax = bounds
    return (
        float(xmin) + (float(col) + 0.5) * float(pixel_size_m),
        float(ymax) - (float(row) + 0.5) * float(pixel_size_m),
    )


def _binary_metrics_from_masks(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_flat = np.asarray(y_true, dtype="float32").reshape(-1)
    y_pred_flat = np.asarray(y_pred, dtype="float32").reshape(-1)
    return binary_pixel_metrics(y_true_flat, y_pred_flat, threshold=0.5)


def build_skeleton_graph_from_mosaic(
    mosaic: AoiRasterMosaic,
    *,
    candidate_threshold: float = 0.2,
    min_component_pixels: int = 8,
    support_recall_tolerance_m: float = 10.0,
    include_cnn_features: bool = False,
) -> SkeletonGraphPart:
    """Build a PyG edge-classification graph from a U-Net support field."""

    support = mosaic.probability >= float(candidate_threshold)
    support = _remove_small_components(support, int(min_component_pixels))
    skeleton = zhang_suen_thinning(support)
    rows, cols = np.nonzero(skeleton)
    n_nodes = int(len(rows))

    cnn_feature_names = list(mosaic.cnn_feature_names or [])
    use_cnn = bool(include_cnn_features and mosaic.cnn_features is not None and cnn_feature_names)
    node_feature_names = ["unet_probability", *mosaic.channel_names]
    if use_cnn:
        node_feature_names.extend(cnn_feature_names)
    node_feature_names.extend(["skeleton_degree", "log_component_size"])
    edge_feature_names = [
        "edge_length_m",
        "edge_unet_probability_mean",
        "edge_unet_probability_min",
        "edge_unet_probability_max",
        "edge_unet_probability_absdiff",
        "edge_bearing_sin",
        "edge_bearing_cos",
        *[f"edge_mean_{name}" for name in mosaic.channel_names],
    ]
    if use_cnn:
        edge_feature_names.extend(f"edge_mean_{name}" for name in cnn_feature_names)
    assert_no_anchor_features(node_feature_names)
    assert_no_anchor_features(edge_feature_names)

    if n_nodes == 0:
        empty_data = Data(
            x=torch.zeros((0, len(node_feature_names)), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_label_index=torch.zeros((2, 0), dtype=torch.long),
            edge_label_attr=torch.zeros((0, len(edge_feature_names)), dtype=torch.float32),
            edge_label=torch.zeros((0,), dtype=torch.float32),
        )
        return SkeletonGraphPart(
            aoi_id=mosaic.aoi_id,
            split=mosaic.split,
            data=empty_data,
            edge_table=gpd.GeoDataFrame(geometry=[], crs="EPSG:28356"),
            support_metrics={
                "aoi_id": mosaic.aoi_id,
                "split": mosaic.split,
                "n_skeleton_nodes": 0,
                "n_skeleton_edges": 0,
            },
            node_feature_names=node_feature_names,
            edge_feature_names=edge_feature_names,
        )

    node_id = -np.ones(skeleton.shape, dtype=np.int64)
    node_id[rows, cols] = np.arange(n_nodes, dtype=np.int64)
    component_labels, _ = label_components(skeleton, structure=np.ones((3, 3), dtype=int))
    component_counts = np.bincount(component_labels.reshape(-1))
    component_size = component_counts[component_labels[rows, cols]]

    neighbor_offsets = [(0, 1), (1, 0), (1, 1), (1, -1)]
    edges: list[tuple[int, int, int, int, int, int]] = []
    h, w = skeleton.shape
    for row, col in zip(rows, cols):
        u = int(node_id[row, col])
        for dr, dc in neighbor_offsets:
            rr = int(row + dr)
            cc = int(col + dc)
            if 0 <= rr < h and 0 <= cc < w and skeleton[rr, cc]:
                v = int(node_id[rr, cc])
                edges.append((u, v, int(row), int(col), rr, cc))

    degree = np.zeros(n_nodes, dtype="float32")
    for u, v, *_ in edges:
        degree[u] += 1.0
        degree[v] += 1.0

    sampled_channels = mosaic.x[:, rows, cols].T
    node_feature_parts = [mosaic.probability[rows, cols][:, None], sampled_channels]
    if use_cnn and mosaic.cnn_features is not None:
        node_feature_parts.append(mosaic.cnn_features[:, rows, cols].T)
    node_feature_parts.extend(
        [
            degree[:, None],
            np.log1p(component_size.astype("float32"))[:, None],
        ]
    )
    node_features = np.column_stack(
        node_feature_parts
    ).astype("float32")

    edge_index_values: list[tuple[int, int]] = []
    edge_attr_values: list[list[float]] = []
    labels: list[float] = []
    records: list[dict] = []
    for edge_local_id, (u, v, r0, c0, r1, c1) in enumerate(edges):
        length = math.hypot(float(c1 - c0), float(r1 - r0)) * float(mosaic.pixel_size_m)
        p0 = float(mosaic.probability[r0, c0])
        p1 = float(mosaic.probability[r1, c1])
        dx = float(c1 - c0) * float(mosaic.pixel_size_m)
        dy = -float(r1 - r0) * float(mosaic.pixel_size_m)
        angle = math.atan2(dy, dx)
        mean_channels = ((mosaic.x[:, r0, c0] + mosaic.x[:, r1, c1]) / 2.0).astype("float32")
        edge_features = [
            float(length),
            float((p0 + p1) / 2.0),
            float(min(p0, p1)),
            float(max(p0, p1)),
            float(abs(p0 - p1)),
            float(math.sin(angle)),
            float(math.cos(angle)),
            *[float(value) for value in mean_channels],
        ]
        if use_cnn and mosaic.cnn_features is not None:
            mean_cnn = ((mosaic.cnn_features[:, r0, c0] + mosaic.cnn_features[:, r1, c1]) / 2.0).astype("float32")
            edge_features.extend(float(value) for value in mean_cnn)
        edge_attr_values.append(edge_features)
        y_value = float(max(float(mosaic.y[r0, c0]), float(mosaic.y[r1, c1])) >= 0.5)
        labels.append(y_value)
        edge_index_values.append((u, v))
        x0, y0 = _world_xy(r0, c0, mosaic.bounds, mosaic.pixel_size_m)
        x1, y1 = _world_xy(r1, c1, mosaic.bounds, mosaic.pixel_size_m)
        records.append(
            {
                "aoi_id": mosaic.aoi_id,
                "split": mosaic.split,
                "edge_local_id": edge_local_id,
                "u": u,
                "v": v,
                "y": int(y_value),
                "edge_length_m": float(length),
                "unet_probability_mean": float((p0 + p1) / 2.0),
                "geometry": LineString([(x0, y0), (x1, y1)]),
            }
        )

    if edge_index_values:
        edge_label_index = np.asarray(edge_index_values, dtype=np.int64).T
        message_edges = np.concatenate([edge_label_index, edge_label_index[::-1]], axis=1)
        edge_attr = np.asarray(edge_attr_values, dtype="float32")
        edge_label = np.asarray(labels, dtype="float32")
    else:
        edge_label_index = np.zeros((2, 0), dtype=np.int64)
        message_edges = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, len(edge_feature_names)), dtype="float32")
        edge_label = np.zeros((0,), dtype="float32")

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=torch.tensor(message_edges, dtype=torch.long),
        edge_label_index=torch.tensor(edge_label_index, dtype=torch.long),
        edge_label_attr=torch.tensor(edge_attr, dtype=torch.float32),
        edge_label=torch.tensor(edge_label, dtype=torch.float32),
    )

    truth_mask = mosaic.y >= 0.5
    dist_to_skeleton_m = distance_transform_edt(~skeleton) * float(mosaic.pixel_size_m)
    truth_pixels = int(truth_mask.sum())
    support_recall = (
        float((dist_to_skeleton_m[truth_mask] <= float(support_recall_tolerance_m)).mean())
        if truth_pixels > 0
        else 0.0
    )
    skeleton_precision = float(truth_mask[skeleton].mean()) if skeleton.any() else 0.0
    support_metrics = {
        "aoi_id": mosaic.aoi_id,
        "split": mosaic.split,
        "n_skeleton_nodes": int(n_nodes),
        "n_skeleton_edges": int(len(edge_index_values)),
        "truth_positive_pixel_fraction": float(truth_mask.mean()),
        "candidate_support_pixel_fraction": float(support.mean()),
        "candidate_support_recall": float(_binary_metrics_from_masks(truth_mask, support)["recall"]),
        "candidate_support_precision": float(_binary_metrics_from_masks(truth_mask, support)["precision"]),
        "candidate_support_f1": float(_binary_metrics_from_masks(truth_mask, support)["f1"]),
        "skeleton_truth_recall_within_tolerance": support_recall,
        "skeleton_node_precision_inside_truth": skeleton_precision,
        "support_recall_tolerance_m": float(support_recall_tolerance_m),
    }

    edge_table = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:28356")
    return SkeletonGraphPart(
        aoi_id=mosaic.aoi_id,
        split=mosaic.split,
        data=data,
        edge_table=edge_table,
        support_metrics=support_metrics,
        node_feature_names=node_feature_names,
        edge_feature_names=edge_feature_names,
    )


def combine_skeleton_graph_parts(parts: list[SkeletonGraphPart]) -> tuple[Data, gpd.GeoDataFrame]:
    """Combine per-AOI skeleton graphs into one disconnected PyG graph."""

    non_empty = [part for part in parts if int(part.data.edge_label.numel()) > 0]
    if not non_empty:
        raise ValueError("No skeleton candidate edges were generated.")

    node_features: list[torch.Tensor] = []
    message_edges: list[torch.Tensor] = []
    edge_label_indices: list[torch.Tensor] = []
    edge_attrs: list[torch.Tensor] = []
    edge_labels: list[torch.Tensor] = []
    node_split_codes: list[torch.Tensor] = []
    edge_split_codes: list[torch.Tensor] = []
    edge_tables: list[gpd.GeoDataFrame] = []
    node_offset = 0
    edge_offset = 0
    for part in non_empty:
        data = part.data
        split_code = SPLIT_TO_CODE.get(str(part.split).lower(), 0)
        n_nodes = int(data.x.shape[0])
        n_edges = int(data.edge_label.shape[0])
        node_features.append(data.x)
        node_split_codes.append(torch.full((n_nodes,), split_code, dtype=torch.long))
        edge_split_codes.append(torch.full((n_edges,), split_code, dtype=torch.long))
        message_edges.append(data.edge_index + node_offset)
        edge_label_indices.append(data.edge_label_index + node_offset)
        edge_attrs.append(data.edge_label_attr)
        edge_labels.append(data.edge_label)
        table = part.edge_table.copy()
        table["edge_global_id"] = np.arange(edge_offset, edge_offset + n_edges, dtype=int)
        table["split_code"] = split_code
        edge_tables.append(table)
        node_offset += n_nodes
        edge_offset += n_edges

    combined = Data(
        x=torch.cat(node_features, dim=0),
        edge_index=torch.cat(message_edges, dim=1),
        edge_label_index=torch.cat(edge_label_indices, dim=1),
        edge_label_attr=torch.cat(edge_attrs, dim=0),
        edge_label=torch.cat(edge_labels, dim=0),
        node_split_code=torch.cat(node_split_codes, dim=0),
        edge_split_code=torch.cat(edge_split_codes, dim=0),
    )
    edge_table = gpd.GeoDataFrame(pd.concat(edge_tables, ignore_index=True), geometry="geometry", crs=edge_tables[0].crs)
    return combined, edge_table


def standardize_graph_features(
    data: Data,
    *,
    train_split_code: int = 0,
) -> tuple[Data, dict[str, list[float]]]:
    """Standardize node and edge features using only training AOI features."""

    out = data.clone()
    train_nodes = out.node_split_code == int(train_split_code)
    train_edges = out.edge_split_code == int(train_split_code)
    if not bool(train_nodes.any()):
        train_nodes = torch.ones_like(out.node_split_code, dtype=torch.bool)
    if not bool(train_edges.any()):
        train_edges = torch.ones_like(out.edge_split_code, dtype=torch.bool)

    node_mean = out.x[train_nodes].mean(dim=0, keepdim=True)
    node_std = out.x[train_nodes].std(dim=0, keepdim=True)
    node_std = torch.where(node_std < 1e-6, torch.ones_like(node_std), node_std)
    edge_mean = out.edge_label_attr[train_edges].mean(dim=0, keepdim=True)
    edge_std = out.edge_label_attr[train_edges].std(dim=0, keepdim=True)
    edge_std = torch.where(edge_std < 1e-6, torch.ones_like(edge_std), edge_std)
    out.x = (out.x - node_mean) / node_std
    out.edge_label_attr = (out.edge_label_attr - edge_mean) / edge_std
    return out, {
        "node_mean": [float(value) for value in node_mean.reshape(-1)],
        "node_std": [float(value) for value in node_std.reshape(-1)],
        "edge_mean": [float(value) for value in edge_mean.reshape(-1)],
        "edge_std": [float(value) for value in edge_std.reshape(-1)],
    }
