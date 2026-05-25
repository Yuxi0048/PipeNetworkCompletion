"""Evaluate post-filters on the best saved anchor-free road-offset model.

Workstream: Codex

This script does not train a model. It takes saved prediction probabilities from
the best current road-offset GNN and evaluates whether simple post-filters can
improve precision without sacrificing much recall.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

import geopandas as gpd
import pandas as pd


def _read_predictions(path: Path, split: str) -> gpd.GeoDataFrame:
    predictions = gpd.read_file(path)
    if "aoi_split" not in predictions.columns:
        raise ValueError(f"Prediction layer is missing aoi_split: {path}")
    predictions = predictions[predictions["aoi_split"].astype(str) == split].copy()
    predictions["candidate_length_m"] = predictions.geometry.length
    return predictions


def _merge_road_attributes(
    predictions: gpd.GeoDataFrame,
    aoi_root: Path,
) -> gpd.GeoDataFrame:
    frames = []
    for aoi_id in sorted(predictions["aoi_id"].astype(str).unique()):
        roads_path = aoi_root / aoi_id / "roads.geojson"
        if not roads_path.exists():
            continue
        roads = gpd.read_file(roads_path)
        attrs = roads.drop(columns="geometry").reset_index().rename(columns={"index": "source_index"})
        attrs["aoi_id"] = aoi_id
        keep_cols = [
            column
            for column in ["aoi_id", "source_index", "ROUTE_TYPE", "OVL_CAT", "OVL2_CAT"]
            if column in attrs.columns
        ]
        frames.append(attrs[keep_cols])
    if not frames:
        return predictions
    road_attrs = pd.concat(frames, ignore_index=True)
    return predictions.merge(road_attrs, on=["aoi_id", "source_index"], how="left")


def _optional_nearest_building_distance(
    predictions: gpd.GeoDataFrame,
    osm_footprints: Path,
    max_distance_m: float,
) -> gpd.GeoDataFrame:
    buildings = gpd.read_file(osm_footprints).to_crs(predictions.crs)
    nearest = gpd.sjoin_nearest(
        predictions[["aoi_id", "source_index", "geometry"]],
        buildings[["geometry"]],
        how="left",
        max_distance=max_distance_m,
        distance_col="building_distance_m",
    )
    distances = (
        nearest.groupby(["aoi_id", "source_index"])["building_distance_m"]
        .min()
        .reset_index()
    )
    out = predictions.merge(distances, on=["aoi_id", "source_index"], how="left")
    out["building_distance_m"] = out["building_distance_m"].fillna(float("inf"))
    return out


def _route_filter(name: str) -> Callable[[pd.DataFrame], pd.Series]:
    def all_rows(df: pd.DataFrame) -> pd.Series:
        return pd.Series(True, index=df.index)

    def not_motorway(df: pd.DataFrame) -> pd.Series:
        return ~df["ROUTE_TYPE"].fillna("").str.contains("Busway|Motorway", case=False, regex=True)

    def no_arterial(df: pd.DataFrame) -> pd.Series:
        return ~df["ROUTE_TYPE"].fillna("").str.contains("Arterial", case=False, regex=False)

    def local_only(df: pd.DataFrame) -> pd.Series:
        return df["ROUTE_TYPE"].fillna("").str.contains(
            "Neighbourhood / local",
            case=False,
            regex=False,
        )

    filters = {
        "all": all_rows,
        "not_motorway": not_motorway,
        "no_arterial": no_arterial,
        "local_only": local_only,
    }
    if name not in filters:
        raise ValueError(f"Unknown route filter {name!r}. Available: {sorted(filters)}")
    return filters[name]


def _evaluate(
    df: pd.DataFrame,
    threshold: float,
    filter_name: str,
    building_max_distance_m: float | None,
    baseline: dict[str, float] | None,
) -> dict[str, object]:
    y = df["y"].astype(bool)
    mask = df["presence_probability"] >= threshold
    mask = mask & _route_filter(filter_name)(df)
    if building_max_distance_m is not None:
        if "building_distance_m" not in df.columns:
            raise ValueError("building_distance_m is missing; rerun with --with-building-distance")
        mask = mask & (df["building_distance_m"] <= building_max_distance_m)

    tp = int((mask & y).sum())
    fp = int((mask & ~y).sum())
    fn = int((~mask & y).sum())
    tn = int((~mask & ~y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)

    total_positive_length = float(df.loc[y, "candidate_length_m"].sum())
    predicted_length = float(df.loc[mask, "candidate_length_m"].sum())
    true_positive_length = float(df.loc[mask & y, "candidate_length_m"].sum())
    false_positive_length = float(df.loc[mask & ~y, "candidate_length_m"].sum())
    length_precision = true_positive_length / predicted_length if predicted_length else 0.0
    length_recall = true_positive_length / total_positive_length if total_positive_length else 0.0
    length_f1 = (
        2.0 * length_precision * length_recall / (length_precision + length_recall)
        if length_precision + length_recall
        else 0.0
    )

    row: dict[str, object] = {
        "threshold": threshold,
        "route_filter": filter_name,
        "building_max_distance_m": building_max_distance_m if building_max_distance_m is not None else "",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_positive_count": int(mask.sum()),
        "length_precision_proxy": length_precision,
        "length_recall_proxy": length_recall,
        "length_f1_proxy": length_f1,
        "predicted_length_m": predicted_length,
        "true_positive_length_proxy_m": true_positive_length,
        "false_positive_length_proxy_m": false_positive_length,
    }
    if baseline is not None:
        row["precision_gain_pp"] = (precision - float(baseline["precision"])) * 100.0
        row["recall_loss_pp"] = (float(baseline["recall"]) - recall) * 100.0
        row["fp_reduction_pct"] = (
            (1.0 - fp / float(baseline["fp"])) * 100.0 if baseline["fp"] else 0.0
        )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "outputs/road_offset_lane_gnn_5lane_aoi115_sage_no_xy/"
            "road_offset_lane_node_predictions.geojson"
        ),
    )
    parser.add_argument(
        "--aoi-root",
        type=Path,
        default=Path("data/processed/aois/anchor_free_2km_gap500_115"),
    )
    parser.add_argument(
        "--osm-footprints",
        type=Path,
        default=Path("data/raw/context/buildings/osm_buildings_selected_aois.geojson"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "outputs/road_offset_lane_gnn_5lane_aoi115_sage_no_xy/"
            "best_model_postfilter_tradeoff.csv"
        ),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.25, 0.3, 0.35, 0.4, 0.5])
    parser.add_argument(
        "--route-filters",
        nargs="+",
        default=["all", "not_motorway", "no_arterial", "local_only"],
    )
    parser.add_argument("--with-building-distance", action="store_true")
    parser.add_argument("--building-max-distances-m", type=float, nargs="+", default=[80, 120, 160, 250])
    parser.add_argument("--nearest-building-max-distance-m", type=float, default=250.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions = _read_predictions(args.predictions, args.split)
    predictions = _merge_road_attributes(predictions, args.aoi_root)
    if args.with_building_distance:
        predictions = _optional_nearest_building_distance(
            predictions,
            args.osm_footprints,
            args.nearest_building_max_distance_m,
        )

    baseline = _evaluate(predictions, 0.2, "all", None, baseline=None)
    rows = []
    for threshold in args.thresholds:
        for route_filter in args.route_filters:
            rows.append(_evaluate(predictions, threshold, route_filter, None, baseline=baseline))
            if args.with_building_distance:
                for distance in args.building_max_distances_m:
                    rows.append(_evaluate(predictions, threshold, route_filter, distance, baseline=baseline))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = pd.DataFrame(rows).sort_values(["recall_loss_pp", "precision_gain_pp"], ascending=[True, False])
    print(summary.head(20).to_string(index=False))
    print(f"Wrote post-filter tradeoff: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
