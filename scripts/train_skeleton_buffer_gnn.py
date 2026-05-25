"""Train a single-support skeleton-buffer GNN for sewer-main presence.

Workstream: Codex

This is the replacement for the five-lane road-offset experiment when the
research story is region-to-network reconstruction:

  surface skeleton edge + context buffer -> sewer-main presence probability

The candidate graph is built from allowed surface supports only: road centerline
segments and optional drainage/watercourse centerlines. Sewer-main truth is used
only to create labels and evaluation metrics.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
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
from shapely.geometry import GeometryCollection, LineString, MultiLineString
from shapely.ops import unary_union
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv, SAGEConv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipe_network_completion.anchor_free.config import load_anchor_free_config  # noqa: E402
from pipe_network_completion.anchor_free.features import (  # noqa: E402
    assert_no_anchor_features,
    standardize_features,
)
from pipe_network_completion.anchor_free.model import (  # noqa: E402
    resolve_torch_device,
    torch_device_report,
)


@dataclass(frozen=True)
class SkeletonPart:
    aoi_id: str
    split: str
    candidates: gpd.GeoDataFrame
    features: pd.DataFrame
    labels: pd.DataFrame
    edge_pairs: np.ndarray
    truth: gpd.GeoDataFrame


class SkeletonBufferGNN(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        layer_type: str = "sage",
    ):
        super().__init__()
        self.dropout = float(dropout)
        self.encoder = torch.nn.Linear(input_dim, hidden_dim)
        conv_cls = GraphConv if str(layer_type).lower() in {"graphconv", "graph_conv"} else SAGEConv
        self.convs = torch.nn.ModuleList(
            [conv_cls(hidden_dim, hidden_dim) for _ in range(max(int(num_layers), 1))]
        )
        self.head = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data: Data) -> torch.Tensor:
        x = F.relu(self.encoder(data.x.float()))
        for conv in self.convs:
            x = F.dropout(F.relu(conv(x, data.edge_index)), p=self.dropout, training=self.training)
        return self.head(x).reshape(-1)


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
    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            deduped.append(path)
            seen.add(resolved)
    return deduped


def _read_layer(path_value: str | Path | None, target_crs=None) -> gpd.GeoDataFrame:
    if path_value in (None, ""):
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)
    path = _resolve(path_value)
    if not path.exists():
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)
    gdf = gpd.read_file(path)
    if target_crs is not None and gdf.crs is not None and str(gdf.crs) != str(target_crs):
        gdf = gdf.to_crs(target_crs)
    return gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()


def _line_parts(geometry) -> list[LineString]:
    if isinstance(geometry, LineString):
        return [geometry] if len(geometry.coords) >= 2 else []
    if isinstance(geometry, MultiLineString):
        return [part for part in geometry.geoms if len(part.coords) >= 2]
    return []


def _bearing_features(geometry) -> tuple[float, float]:
    parts = _line_parts(geometry)
    if not parts:
        return 0.0, 1.0
    coords = list(parts[0].coords)
    x0, y0 = coords[0][:2]
    x1, y1 = coords[-1][:2]
    angle = math.atan2(float(y1 - y0), float(x1 - x0))
    return math.sin(angle), math.cos(angle)


def _bearing_rad(geometry) -> float:
    parts = _line_parts(geometry)
    if not parts:
        return 0.0
    coords = list(parts[0].coords)
    x0, y0 = coords[0][:2]
    x1, y1 = coords[-1][:2]
    return math.atan2(float(y1 - y0), float(x1 - x0))


def _bearing_bin(angle_rad: float) -> int:
    degrees = math.degrees(float(angle_rad) % math.pi)
    if degrees < 15.0 or degrees >= 165.0:
        return 0
    return int((degrees - 15.0) / 30.0) + 1


def _fixed_distance_bins(values: np.ndarray, *, prefix: str, breaks: tuple[float, ...]) -> pd.DataFrame:
    values = np.asarray(values, dtype=float)
    bins = np.digitize(values, np.asarray(breaks[1:-1], dtype=float), right=True)
    frame = pd.DataFrame(index=np.arange(len(values)))
    for bin_id in range(len(breaks) - 1):
        frame[f"{prefix}_bin_{bin_id}"] = (bins == bin_id).astype(float)
    return frame


def _prepare_skeleton_candidates(
    roads: gpd.GeoDataFrame,
    drainage_lines: gpd.GeoDataFrame,
    *,
    include_drainage: bool,
) -> gpd.GeoDataFrame:
    frames = []
    if not roads.empty:
        road = roads.copy()
        road["candidate_source"] = "road"
        keep = [
            column
            for column in ["candidate_source", "ROUTE_TYPE", "OVL_CAT", "OVL2_CAT", "CLASS", "highway", "geometry"]
            if column in road.columns
        ]
        frames.append(road[keep])
    if include_drainage and not drainage_lines.empty:
        drainage = drainage_lines.copy()
        drainage["candidate_source"] = "drainage"
        drainage["ROUTE_TYPE"] = "drainage"
        keep = [
            column
            for column in ["candidate_source", "ROUTE_TYPE", "OVL_CAT", "OVL2_CAT", "CLASS", "highway", "geometry"]
            if column in drainage.columns
        ]
        frames.append(drainage[keep])
    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=roads.crs or drainage_lines.crs)
    candidates = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs)
    candidates = candidates.explode(index_parts=False).reset_index(drop=True)
    candidates = candidates[candidates.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    candidates["candidate_id"] = np.arange(len(candidates), dtype=int)
    candidates["length_m"] = candidates.geometry.length.astype(float)
    candidates = candidates[candidates["length_m"] > 1.0].copy().reset_index(drop=True)
    candidates["candidate_id"] = np.arange(len(candidates), dtype=int)
    return candidates


def _prepare_road_reference(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if roads.empty:
        return gpd.GeoDataFrame({"road_ref_id": [], "road_bearing_rad": []}, geometry=[], crs=roads.crs)
    records = []
    for source_index, row in roads.iterrows():
        for line in _line_parts(row.geometry):
            records.append(
                {
                    "road_ref_id": len(records),
                    "source_index": source_index,
                    "road_length_m": float(line.length),
                    "road_bearing_rad": _bearing_rad(line),
                    "geometry": line,
                }
            )
    if not records:
        return gpd.GeoDataFrame({"road_ref_id": [], "road_bearing_rad": []}, geometry=[], crs=roads.crs)
    return gpd.GeoDataFrame(records, geometry="geometry", crs=roads.crs)


def _nearest_distance(
    candidates: gpd.GeoDataFrame,
    context: gpd.GeoDataFrame,
    fallback: float,
) -> np.ndarray:
    if candidates.empty:
        return np.zeros(0, dtype=float)
    if context.empty:
        return np.full(len(candidates), fallback, dtype=float)
    joined = gpd.sjoin_nearest(
        candidates[["candidate_id", "geometry"]],
        context[["geometry"]],
        how="left",
        distance_col="distance_m",
    )
    distances = joined.groupby("candidate_id")["distance_m"].min()
    return distances.reindex(candidates["candidate_id"]).fillna(fallback).to_numpy(dtype=float)


def _count_context_in_buffer(
    candidates: gpd.GeoDataFrame,
    context: gpd.GeoDataFrame,
    buffer_m: float,
) -> np.ndarray:
    if candidates.empty:
        return np.zeros(0, dtype=float)
    if context.empty:
        return np.zeros(len(candidates), dtype=float)
    buffers = gpd.GeoDataFrame(
        {"candidate_id": candidates["candidate_id"].to_numpy(dtype=int)},
        geometry=candidates.geometry.buffer(buffer_m),
        crs=candidates.crs,
    )
    joined = gpd.sjoin(context[["geometry"]], buffers, predicate="intersects", how="inner")
    counts = joined.groupby("candidate_id").size()
    return counts.reindex(candidates["candidate_id"]).fillna(0.0).to_numpy(dtype=float)


def _polygon_area_in_buffer(
    candidates: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    buffer_m: float,
) -> np.ndarray:
    if candidates.empty:
        return np.zeros(0, dtype=float)
    if polygons.empty:
        return np.zeros(len(candidates), dtype=float)
    buffers = gpd.GeoDataFrame(
        {"candidate_id": candidates["candidate_id"].to_numpy(dtype=int)},
        geometry=candidates.geometry.buffer(buffer_m),
        crs=candidates.crs,
    )
    joined = gpd.sjoin(polygons[["geometry"]], buffers, predicate="intersects", how="inner")
    if joined.empty:
        return np.zeros(len(candidates), dtype=float)
    buffer_lookup = dict(zip(buffers["candidate_id"].astype(int), buffers.geometry))
    areas: dict[int, float] = {}
    for row in joined.itertuples():
        candidate_id = int(row.candidate_id)
        area = float(row.geometry.intersection(buffer_lookup[candidate_id]).area)
        areas[candidate_id] = areas.get(candidate_id, 0.0) + area
    return np.asarray([areas.get(int(cid), 0.0) for cid in candidates["candidate_id"]], dtype=float)


def _local_skeleton_density(candidates: gpd.GeoDataFrame, buffer_m: float) -> np.ndarray:
    if candidates.empty:
        return np.zeros(0, dtype=float)
    spatial_index = candidates.sindex
    lengths = candidates["length_m"].to_numpy(dtype=float)
    densities = []
    for geom in candidates.geometry:
        search = geom.buffer(buffer_m)
        area = max(float(search.area), 1e-9)
        try:
            idxs = spatial_index.query(search, predicate="intersects")
        except TypeError:
            idxs = list(spatial_index.intersection(search.bounds))
        densities.append(float(lengths[[int(i) for i in idxs]].sum()) / area if len(idxs) else 0.0)
    return np.asarray(densities, dtype=float)


def _same_type_graph_context(candidates: gpd.GeoDataFrame, edge_pairs: np.ndarray) -> pd.DataFrame:
    features = pd.DataFrame(index=candidates["candidate_id"].astype(int))
    n = len(candidates)
    degree = np.zeros(n, dtype=float)
    if edge_pairs.size:
        adjacency = {i: set() for i in range(n)}
        for src, dst in np.asarray(edge_pairs, dtype=int).T:
            if 0 <= src < n and 0 <= dst < n:
                adjacency[int(src)].add(int(dst))
        degree = np.asarray([len(adjacency[i]) for i in range(n)], dtype=float)
    features["same_type_degree"] = degree
    features["same_type_dead_end"] = (degree <= 1.0).astype(float)
    features["same_type_isolated"] = (degree <= 0.0).astype(float)
    return features


def _nearest_road_relation_features(
    candidates: gpd.GeoDataFrame,
    road_ref: gpd.GeoDataFrame,
) -> pd.DataFrame:
    features = pd.DataFrame(index=candidates["candidate_id"].astype(int))
    n = len(candidates)
    if candidates.empty:
        return features
    fallback = np.full(n, 1000.0, dtype=float)
    if road_ref.empty:
        features["nearest_road_distance_m"] = fallback
        features["nearest_road_pos_0_1"] = 0.0
        features["nearest_road_side_left"] = 0.0
        features["nearest_road_side_right"] = 0.0
        features["nearest_road_side_on"] = 1.0
        features["road_candidate_angle_diff_sin"] = 0.0
        features["road_candidate_angle_diff_cos"] = 1.0
    else:
        joined = gpd.sjoin_nearest(
            candidates[["candidate_id", "geometry"]],
            road_ref[["road_ref_id", "road_length_m", "road_bearing_rad", "geometry"]],
            how="left",
            distance_col="nearest_road_distance_m",
        )
        joined = joined.sort_values(["candidate_id", "nearest_road_distance_m", "road_ref_id"]).groupby("candidate_id").head(1)
        road_lookup = road_ref.set_index("road_ref_id")
        distances = []
        positions = []
        side_left = []
        side_right = []
        side_on = []
        diff_sin = []
        diff_cos = []
        candidate_lookup = candidates.set_index("candidate_id")
        for candidate_id in candidates["candidate_id"].astype(int):
            row = joined[joined["candidate_id"].astype(int) == int(candidate_id)]
            if row.empty or pd.isna(row.iloc[0].get("road_ref_id")):
                distances.append(1000.0)
                positions.append(0.0)
                side_left.append(0.0)
                side_right.append(0.0)
                side_on.append(1.0)
                diff_sin.append(0.0)
                diff_cos.append(1.0)
                continue
            road_id = int(row.iloc[0]["road_ref_id"])
            road_geom = road_lookup.loc[road_id].geometry
            candidate_geom = candidate_lookup.loc[int(candidate_id)].geometry
            candidate_centroid = candidate_geom.centroid
            projected_distance = float(road_geom.project(candidate_centroid))
            road_length = max(float(road_lookup.loc[road_id]["road_length_m"]), 1e-9)
            projected_point = road_geom.interpolate(projected_distance)
            coords = list(road_geom.coords)
            x0, y0 = coords[0][:2]
            x1, y1 = coords[-1][:2]
            road_dx = float(x1 - x0)
            road_dy = float(y1 - y0)
            rel_dx = float(candidate_centroid.x - projected_point.x)
            rel_dy = float(candidate_centroid.y - projected_point.y)
            cross = road_dx * rel_dy - road_dy * rel_dx
            distance = float(row.iloc[0]["nearest_road_distance_m"])
            candidate_angle = _bearing_rad(candidate_geom)
            road_angle = float(road_lookup.loc[road_id]["road_bearing_rad"])
            delta = abs((candidate_angle - road_angle + math.pi / 2.0) % math.pi - math.pi / 2.0)
            distances.append(distance)
            positions.append(projected_distance / road_length)
            side_left.append(1.0 if distance > 1e-6 and cross > 0 else 0.0)
            side_right.append(1.0 if distance > 1e-6 and cross < 0 else 0.0)
            side_on.append(1.0 if distance <= 1e-6 else 0.0)
            diff_sin.append(math.sin(delta))
            diff_cos.append(math.cos(delta))
        features["nearest_road_distance_m"] = np.asarray(distances, dtype=float)
        features["nearest_road_pos_0_1"] = np.clip(np.asarray(positions, dtype=float), 0.0, 1.0)
        features["nearest_road_side_left"] = np.asarray(side_left, dtype=float)
        features["nearest_road_side_right"] = np.asarray(side_right, dtype=float)
        features["nearest_road_side_on"] = np.asarray(side_on, dtype=float)
        features["road_candidate_angle_diff_sin"] = np.asarray(diff_sin, dtype=float)
        features["road_candidate_angle_diff_cos"] = np.asarray(diff_cos, dtype=float)
    distance_bins = _fixed_distance_bins(
        features["nearest_road_distance_m"].to_numpy(dtype=float),
        prefix="nearest_road_distance",
        breaks=(0.0, 5.0, 15.0, 30.0, 60.0, float("inf")),
    )
    distance_bins.index = features.index
    return pd.concat([features, distance_bins], axis=1)


def _building_group_context_features(
    candidates: gpd.GeoDataFrame,
    building_points: gpd.GeoDataFrame,
    buffer_m: float,
) -> pd.DataFrame:
    features = pd.DataFrame(index=candidates["candidate_id"].astype(int))
    if candidates.empty:
        return features
    group_columns = [column for column in building_points.columns if column.startswith("bt_group_")]
    if building_points.empty or not group_columns:
        for group in ["residential", "commercial", "industrial", "institutional", "unknown"]:
            features[f"building_{group}_count_buffer"] = 0.0
            features[f"building_{group}_density_buffer"] = 0.0
        return features
    buffers = gpd.GeoDataFrame(
        {"candidate_id": candidates["candidate_id"].to_numpy(dtype=int)},
        geometry=candidates.geometry.buffer(buffer_m),
        crs=candidates.crs,
    )
    joined = gpd.sjoin(
        building_points[[*group_columns, "geometry"]],
        buffers,
        predicate="intersects",
        how="inner",
    )
    buffer_area = np.maximum(candidates.geometry.buffer(buffer_m).area.to_numpy(dtype=float), 1e-9)
    area_by_candidate = pd.Series(buffer_area, index=candidates["candidate_id"].astype(int))
    grouped = joined.groupby("candidate_id")[group_columns].sum() if not joined.empty else pd.DataFrame()
    for column in group_columns:
        name = column.removeprefix("bt_group_")
        count = grouped[column].reindex(features.index).fillna(0.0) if column in grouped else pd.Series(0.0, index=features.index)
        features[f"building_{name}_count_buffer"] = count.to_numpy(dtype=float)
        features[f"building_{name}_density_buffer"] = (
            count / area_by_candidate.reindex(features.index).fillna(1.0)
        ).to_numpy(dtype=float)
    return features


def _build_features(
    candidates: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    building_points: gpd.GeoDataFrame,
    built_up: gpd.GeoDataFrame,
    edge_pairs: np.ndarray,
    *,
    feature_buffer_m: float,
    density_buffer_m: float,
    paper_analog_features: bool,
) -> pd.DataFrame:
    features = pd.DataFrame(index=candidates["candidate_id"].astype(int))
    features.index.name = "candidate_id"
    features["length_m"] = candidates["length_m"].astype(float).to_numpy()
    features["log1p_length_m"] = np.log1p(features["length_m"])
    bearing_rad = np.asarray([_bearing_rad(geom) for geom in candidates.geometry], dtype=float)
    bearing = np.asarray([_bearing_features(geom) for geom in candidates.geometry], dtype=float)
    features["bearing_sin"] = bearing[:, 0] if len(bearing) else []
    features["bearing_cos"] = bearing[:, 1] if len(bearing) else []
    if paper_analog_features:
        for bin_id in range(6):
            features[f"bearing_bin_{bin_id}"] = (np.asarray([_bearing_bin(value) for value in bearing_rad]) == bin_id).astype(float)
        features = pd.concat([features, _same_type_graph_context(candidates, edge_pairs)], axis=1)
        features = pd.concat([features, _nearest_road_relation_features(candidates, _prepare_road_reference(roads))], axis=1)
    features["skeleton_density_100m"] = _local_skeleton_density(candidates, density_buffer_m)

    buffer_area = np.maximum(candidates.geometry.buffer(feature_buffer_m).area.to_numpy(dtype=float), 1e-9)
    features["building_point_count_buffer"] = _count_context_in_buffer(candidates, building_points, feature_buffer_m)
    features["building_point_density_buffer"] = features["building_point_count_buffer"].to_numpy(dtype=float) / buffer_area
    features["nearest_building_point_m"] = _nearest_distance(candidates, building_points, fallback=1000.0)
    if paper_analog_features:
        building_distance_bins = _fixed_distance_bins(
            features["nearest_building_point_m"].to_numpy(dtype=float),
            prefix="nearest_building_point",
            breaks=(0.0, 10.0, 25.0, 50.0, 100.0, float("inf")),
        )
        building_distance_bins.index = features.index
        features = pd.concat(
            [
                features,
                building_distance_bins,
                _building_group_context_features(candidates, building_points, feature_buffer_m),
            ],
            axis=1,
        )
    features["building_polygon_count_buffer"] = _count_context_in_buffer(candidates, buildings, feature_buffer_m)
    features["nearest_building_polygon_m"] = _nearest_distance(candidates, buildings, fallback=1000.0)
    features["building_polygon_area_buffer_m2"] = _polygon_area_in_buffer(candidates, buildings, feature_buffer_m)
    features["building_polygon_area_fraction_buffer"] = (
        features["building_polygon_area_buffer_m2"].to_numpy(dtype=float) / buffer_area
    )
    features["built_up_area_buffer_m2"] = _polygon_area_in_buffer(candidates, built_up, feature_buffer_m)
    features["built_up_area_fraction_buffer"] = features["built_up_area_buffer_m2"].to_numpy(dtype=float) / buffer_area

    categorical = candidates[["candidate_source"]].copy()
    for optional in ["ROUTE_TYPE", "OVL_CAT", "OVL2_CAT", "CLASS", "highway"]:
        if optional in candidates.columns:
            categorical[optional] = candidates[optional].fillna("missing").astype(str)
    dummies = pd.get_dummies(categorical.fillna("missing").astype(str), prefix=categorical.columns, dtype=float)
    features = pd.concat([features, dummies.set_index(features.index)], axis=1)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    assert_no_anchor_features(features.columns)
    return features


def _label_candidates(
    candidates: gpd.GeoDataFrame,
    truth: gpd.GeoDataFrame,
    *,
    label_buffer_m: float,
    min_truth_length_m: float,
    min_overlap_ratio: float,
) -> pd.DataFrame:
    labels = pd.DataFrame({"candidate_id": candidates["candidate_id"].astype(int)})
    labels["overlap_length_m"] = 0.0
    if candidates.empty or truth.empty:
        labels["overlap_ratio"] = 0.0
        labels["y"] = 0
        return labels
    buffers = gpd.GeoDataFrame(
        {"candidate_id": candidates["candidate_id"].astype(int).to_numpy()},
        geometry=candidates.geometry.buffer(label_buffer_m),
        crs=candidates.crs,
    )
    joined = gpd.sjoin(truth[["geometry"]], buffers, predicate="intersects", how="inner")
    buffer_lookup = dict(zip(buffers["candidate_id"].astype(int), buffers.geometry))
    overlaps: dict[int, float] = {}
    for row in joined.itertuples():
        candidate_id = int(row.candidate_id)
        length = float(row.geometry.intersection(buffer_lookup[candidate_id]).length)
        overlaps[candidate_id] = overlaps.get(candidate_id, 0.0) + length
    labels["overlap_length_m"] = [
        overlaps.get(int(candidate_id), 0.0) for candidate_id in labels["candidate_id"]
    ]
    length = candidates.set_index("candidate_id")["length_m"].reindex(labels["candidate_id"]).to_numpy(dtype=float)
    labels["overlap_ratio"] = labels["overlap_length_m"].to_numpy(dtype=float) / np.maximum(length, 1e-9)
    labels["y"] = (
        (labels["overlap_length_m"] >= float(min_truth_length_m))
        & (labels["overlap_ratio"] >= float(min_overlap_ratio))
    ).astype(int)
    return labels


def _build_adjacency(candidates: gpd.GeoDataFrame, snap_tolerance_m: float) -> np.ndarray:
    if candidates.empty:
        return np.zeros((2, 0), dtype=np.int64)
    spatial_index = candidates.sindex
    pairs: set[tuple[int, int]] = set()
    geoms = list(candidates.geometry)
    for i, geom in enumerate(geoms):
        search = geom.buffer(float(snap_tolerance_m))
        try:
            idxs = spatial_index.query(search, predicate="intersects")
        except TypeError:
            idxs = list(spatial_index.intersection(search.bounds))
        for j_raw in idxs:
            j = int(j_raw)
            if i == j:
                continue
            if search.intersects(geoms[j]):
                pairs.add((i, j))
    if not pairs:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray(sorted(pairs), dtype=np.int64).T


def _load_part(
    config_path: Path,
    *,
    include_drainage: bool,
    feature_buffer_m: float,
    density_buffer_m: float,
    label_buffer_m: float,
    min_truth_length_m: float,
    min_overlap_ratio: float,
    snap_tolerance_m: float,
    paper_analog_features: bool,
) -> SkeletonPart:
    config = load_anchor_free_config(config_path)
    data = dict(config.get("data", {}))
    graph_cfg = dict(config.get("graph", {}))
    aoi_cfg = dict(config.get("aoi", {}))
    aoi_id = str(aoi_cfg.get("aoi_id", config_path.stem))
    split = str(aoi_cfg.get("split", "train")).lower()
    target_crs = graph_cfg.get("target_crs", "EPSG:28356")

    roads = _read_layer(data.get("roads_path"), target_crs)
    buildings = _read_layer(data.get("buildings_path"), target_crs)
    building_points = _read_layer(data.get("building_points_path"), target_crs)
    built_up = _read_layer(data.get("built_up_path"), target_crs)
    truth = _read_layer(data.get("utility_truth_path"), target_crs)
    drainage = _read_layer(data.get("watercourse_drainage_lines_path"), target_crs)
    if drainage.empty:
        drainage = _read_layer(data.get("watercourse_corridor_centrelines_path"), target_crs)

    candidates = _prepare_skeleton_candidates(roads, drainage, include_drainage=include_drainage)
    if candidates.empty:
        candidates = gpd.GeoDataFrame(geometry=[], crs=roads.crs)
    candidates.insert(0, "aoi_id", aoi_id)
    candidates.insert(1, "aoi_split", split)
    edge_pairs = _build_adjacency(candidates, snap_tolerance_m=snap_tolerance_m)

    features = _build_features(
        candidates,
        roads,
        buildings,
        building_points,
        built_up,
        edge_pairs,
        feature_buffer_m=feature_buffer_m,
        density_buffer_m=density_buffer_m,
        paper_analog_features=paper_analog_features,
    )
    labels = _label_candidates(
        candidates,
        truth,
        label_buffer_m=label_buffer_m,
        min_truth_length_m=min_truth_length_m,
        min_overlap_ratio=min_overlap_ratio,
    )
    return SkeletonPart(aoi_id, split, candidates, features, labels, edge_pairs, truth)


def _combine_parts(parts: list[SkeletonPart]) -> tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame, np.ndarray, dict[str, np.ndarray], dict[str, gpd.GeoDataFrame]]:
    feature_frames = []
    label_frames = []
    candidate_frames = []
    edge_arrays = []
    split_indices: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    truth_by_split: dict[str, list[gpd.GeoDataFrame]] = {"train": [], "val": [], "test": []}
    node_offset = 0
    for part in parts:
        n = len(part.candidates)
        features = part.features.copy()
        labels = part.labels.copy()
        candidates = part.candidates.copy()
        features["node_id"] = np.arange(node_offset, node_offset + n, dtype=int)
        labels["node_id"] = np.arange(node_offset, node_offset + n, dtype=int)
        candidates["node_id"] = np.arange(node_offset, node_offset + n, dtype=int)
        features.insert(0, "aoi_id", part.aoi_id)
        features.insert(1, "aoi_split", part.split)
        labels.insert(0, "aoi_id", part.aoi_id)
        labels.insert(1, "aoi_split", part.split)
        feature_frames.append(features)
        label_frames.append(labels)
        candidate_frames.append(candidates)
        split_key = "val" if part.split in {"val", "valid", "validation"} else part.split
        if split_key not in split_indices:
            split_key = "train"
        split_indices[split_key].extend(range(node_offset, node_offset + n))
        truth_by_split[split_key].append(part.truth)
        if part.edge_pairs.size:
            edge_arrays.append(part.edge_pairs + node_offset)
        node_offset += n

    features_all = pd.concat(feature_frames, ignore_index=True)
    labels_all = pd.concat(label_frames, ignore_index=True)
    candidates_all = gpd.GeoDataFrame(
        pd.concat(candidate_frames, ignore_index=True),
        geometry="geometry",
        crs=candidate_frames[0].crs if candidate_frames else None,
    )
    feature_values = (
        features_all.drop(columns=["aoi_id", "aoi_split", "candidate_id", "node_id"], errors="ignore")
        .reindex(
            sorted(features_all.drop(columns=["aoi_id", "aoi_split", "candidate_id", "node_id"], errors="ignore").columns),
            axis=1,
        )
        .fillna(0.0)
    )
    assert_no_anchor_features(feature_values.columns)
    if edge_arrays:
        edge_index = np.concatenate(edge_arrays, axis=1)
        reverse = edge_index[[1, 0]]
        edge_index = np.unique(np.concatenate([edge_index, reverse], axis=1), axis=1)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    truth_frames: dict[str, gpd.GeoDataFrame] = {}
    for split, frames in truth_by_split.items():
        non_empty = [frame for frame in frames if frame is not None and not frame.empty]
        if non_empty:
            truth_frames[split] = gpd.GeoDataFrame(pd.concat(non_empty, ignore_index=True), geometry="geometry", crs=non_empty[0].crs)
        else:
            truth_frames[split] = gpd.GeoDataFrame(geometry=[], crs=candidates_all.crs)
    return (
        feature_values,
        labels_all.sort_values("node_id").reset_index(drop=True),
        candidates_all.sort_values("node_id").reset_index(drop=True),
        edge_index,
        {key: np.asarray(value, dtype=int) for key, value in split_indices.items()},
        truth_frames,
    )


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
    out = features.copy()
    for bin_id in range(int(n_bins)):
        out[f"{prefix}_bin_{bin_id}"] = (bins == bin_id).astype(float)
    assert_no_anchor_features(out.columns)
    return out


def _safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, score)) if np.unique(y_true).size == 2 else float("nan")


def _edge_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = prob >= float(threshold)
    prevalence = float(np.mean(y_true)) if len(y_true) else float("nan")
    return {
        "positive_prevalence": prevalence,
        "roc_auc": _safe_auc(y_true, prob),
        "pr_auc": float(average_precision_score(y_true, prob)) if np.unique(y_true).size == 2 else float("nan"),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)) if np.unique(y_true).size == 2 else float("nan"),
        "brier_score": float(brier_score_loss(y_true, prob)),
        "all_positive_f1": float(2.0 * prevalence / (1.0 + prevalence)) if prevalence > 0 else 0.0,
    }


def _select_threshold(y: np.ndarray, prob: np.ndarray, val_index: np.ndarray, thresholds: Iterable[float]) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in thresholds:
        rows.append({"threshold": float(threshold), **_edge_metrics(y[val_index], prob[val_index], float(threshold))})
    table = pd.DataFrame(rows)
    if table.empty:
        return 0.5, table
    best = table.sort_values(["f1", "pr_auc", "balanced_accuracy"], ascending=[False, False, False]).iloc[0]
    return float(best["threshold"]), table


def _truth_coverage_metrics(
    candidates: gpd.GeoDataFrame,
    truth: gpd.GeoDataFrame,
    selected_mask: np.ndarray,
    *,
    buffer_m: float,
) -> dict[str, float]:
    if truth.empty:
        return {
            "selected_buffer_truth_recall": float("nan"),
            "selected_buffer_area_m2": 0.0,
        }
    selected = candidates[selected_mask].copy()
    total = float(truth.geometry.length.sum())
    if selected.empty:
        return {
            "selected_buffer_truth_recall": 0.0,
            "selected_buffer_area_m2": 0.0,
        }

    selected_buffers = gpd.GeoDataFrame(
        geometry=selected.geometry.buffer(buffer_m),
        crs=selected.crs,
    )
    buffer_geoms = list(selected_buffers.geometry)
    spatial_index = selected_buffers.sindex
    covered = 0.0
    for truth_geom in truth.geometry:
        if truth_geom is None or truth_geom.is_empty:
            continue
        try:
            idxs = spatial_index.query(truth_geom, predicate="intersects")
        except TypeError:
            idxs = list(spatial_index.intersection(truth_geom.bounds))
        local_buffers = [buffer_geoms[int(idx)] for idx in idxs]
        if not local_buffers:
            continue
        local_union = unary_union(local_buffers)
        covered += float(truth_geom.intersection(local_union).length)
    return {
        "selected_buffer_truth_recall": covered / total if total > 0.0 else float("nan"),
        "selected_buffer_covered_truth_length_m": covered,
        "selected_buffer_total_truth_length_m": total,
        "selected_buffer_area_m2": float(selected_buffers.geometry.area.sum()),
        "selected_buffer_area_note": "gross_area_sum_not_dissolved_union",
    }


def _write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gdf.empty:
        path.write_text('{"type":"FeatureCollection","features":[]}\n', encoding="utf-8")
    else:
        path.write_text(gdf.to_json(drop_id=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=["configs/aois_2km_gap500_115_osm_bpoints_all_mains/*.yaml"])
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "skeleton_buffer_gnn_osm_bpoints_all_mains")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gnn-layer-type", choices=["sage", "graphconv"], default="sage")
    parser.add_argument("--feature-buffer-m", type=float, default=50.0)
    parser.add_argument("--label-buffer-m", type=float, default=30.0)
    parser.add_argument("--min-truth-length-m", type=float, default=10.0)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.05)
    parser.add_argument("--density-buffer-m", type=float, default=100.0)
    parser.add_argument("--snap-tolerance-m", type=float, default=1.0)
    parser.add_argument("--include-drainage", action="store_true")
    parser.add_argument(
        "--paper-analog-features",
        action="store_true",
        help=(
            "Add paper-principled non-anchor features: length/bearing bins, "
            "skeleton-road relation bins, building group aggregates, and "
            "local dead-end/degree context."
        ),
    )
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--threshold-grid", nargs="+", type=float, default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_report = torch_device_report()
    print(f"GPU preflight: {device_report}", flush=True)
    if str(args.device).lower().startswith("cuda") and not device_report["cuda_available"]:
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    config_paths = _expand_configs(args.configs)
    if int(args.max_configs) > 0:
        config_paths = config_paths[: int(args.max_configs)]

    parts: list[SkeletonPart] = []
    for config_path in config_paths:
        part = _load_part(
            config_path,
            include_drainage=bool(args.include_drainage),
            feature_buffer_m=float(args.feature_buffer_m),
            density_buffer_m=float(args.density_buffer_m),
            label_buffer_m=float(args.label_buffer_m),
            min_truth_length_m=float(args.min_truth_length_m),
            min_overlap_ratio=float(args.min_overlap_ratio),
            snap_tolerance_m=float(args.snap_tolerance_m),
            paper_analog_features=bool(args.paper_analog_features),
        )
        parts.append(part)
        print(
            f"prepared {part.aoi_id} ({part.split}): skeleton_edges={len(part.candidates)} "
            f"positives={int(part.labels['y'].sum())}/{len(part.labels)}",
            flush=True,
        )

    features, labels, candidates, edge_index, split_indices, truth_by_split = _combine_parts(parts)
    train_index = split_indices["train"]
    val_index = split_indices["val"]
    test_index = split_indices["test"]
    if bool(args.paper_analog_features):
        features = _add_train_quantile_bins(
            features,
            column="length_m",
            train_index=train_index,
            prefix="length",
        )
    scaled_features, mean, std = standardize_features(features, train_index=train_index)

    data = Data(
        x=torch.tensor(scaled_features.to_numpy(dtype=float), dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(labels["y"].to_numpy(dtype=float), dtype=torch.float32),
    )
    torch.manual_seed(42)
    np.random.seed(42)
    device = resolve_torch_device(args.device)
    data = data.to(device)
    model = SkeletonBufferGNN(
        input_dim=int(data.x.shape[1]),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        layer_type=str(args.gnn_layer_type),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    train_tensor = torch.tensor(train_index, dtype=torch.long, device=device)
    train_y = data.y[train_tensor]
    positives = torch.clamp(train_y.sum(), min=1.0)
    negatives = torch.clamp(train_y.numel() - train_y.sum(), min=1.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=(negatives / positives))

    losses = []
    for epoch in range(int(args.epochs)):
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        loss = loss_fn(logits[train_tensor], data.y[train_tensor])
        loss.backward()
        optimizer.step()
        losses.append({"epoch": epoch + 1, "loss": float(loss.detach().cpu())})

    model.eval()
    with torch.no_grad():
        logits = model(data)
        prob = torch.sigmoid(logits).detach().cpu().numpy()

    y_np = labels["y"].to_numpy(dtype=int)
    best_threshold, threshold_table = _select_threshold(y_np, prob, val_index, args.threshold_grid)
    metric_rows = []
    for split, idx in split_indices.items():
        if len(idx) == 0:
            continue
        split_candidates = candidates.iloc[idx].copy()
        split_prob = prob[idx]
        selected = split_prob >= best_threshold
        row = {
            "split": split,
            "n_skeleton_edges": int(len(idx)),
            **{f"edge_{key}": value for key, value in _edge_metrics(y_np[idx], split_prob, best_threshold).items()},
            **_truth_coverage_metrics(
                split_candidates,
                truth_by_split[split],
                selected,
                buffer_m=float(args.label_buffer_m),
            ),
        }
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)

    predictions = candidates.merge(
        labels[["node_id", "candidate_id", "y", "overlap_length_m", "overlap_ratio"]],
        on=["node_id", "candidate_id"],
        how="left",
    )
    predictions["presence_probability"] = prob
    predictions["predicted_presence"] = prob >= best_threshold

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
    (output_dir / "feature_columns.json").write_text(json.dumps(list(features.columns), indent=2), encoding="utf-8")
    predictions.drop(columns=["geometry"], errors="ignore").to_csv(output_dir / "skeleton_buffer_predictions.csv", index=False)
    _write_geojson(predictions, output_dir / "skeleton_buffer_predictions.geojson")
    selected_predictions = predictions[predictions["predicted_presence"].astype(bool)].copy()
    _write_geojson(selected_predictions, output_dir / "decoded_skeleton_network.geojson")
    torch.save(model.cpu().state_dict(), output_dir / "skeleton_buffer_gnn_state.pt")

    summary = {
        "workstream": "Codex",
        "description": "single-support skeleton-buffer GraphSAGE sewer-main presence model",
        "truth_target": "complete sewer mains: gravity mains plus pressure mains",
        "runtime_sec": time.perf_counter() - start,
        "device_report": device_report,
        "training_device": str(device),
        "n_aois": len(parts),
        "n_skeleton_edges": int(len(labels)),
        "n_message_edges": int(edge_index.shape[1]),
        "n_features": int(features.shape[1]),
        "feature_buffer_m": float(args.feature_buffer_m),
        "label_buffer_m": float(args.label_buffer_m),
        "min_truth_length_m": float(args.min_truth_length_m),
        "min_overlap_ratio": float(args.min_overlap_ratio),
        "include_drainage": bool(args.include_drainage),
        "paper_analog_features": bool(args.paper_analog_features),
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
    keep = [
        "split",
        "n_skeleton_edges",
        "edge_roc_auc",
        "edge_pr_auc",
        "edge_f1",
        "edge_precision",
        "edge_recall",
        "edge_positive_prevalence",
        "edge_all_positive_f1",
        "selected_buffer_truth_recall",
    ]
    print(metrics[[column for column in keep if column in metrics.columns]].to_string(index=False), flush=True)
    print(f"Wrote outputs to {_relative(output_dir)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
