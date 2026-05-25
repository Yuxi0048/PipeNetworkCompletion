"""Evaluate predicted utilities by ground-truth sewer segment type.

Workstream: Codex
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipe_network_completion.anchor_free.candidate_recall import iter_line_parts  # noqa: E402
from pipe_network_completion.anchor_free.config import load_anchor_free_config  # noqa: E402
from pipe_network_completion.anchor_free.labels import road_offset_lane_name  # noqa: E402
from pipe_network_completion.anchor_free.pipeline import prepare_anchor_free_inputs  # noqa: E402


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


def _sample_lines(
    lines: gpd.GeoDataFrame,
    *,
    sample_spacing_m: float,
    keep_columns: list[str],
) -> gpd.GeoDataFrame:
    records: list[dict] = []
    spacing = max(float(sample_spacing_m), 1.0)
    for row in lines.itertuples(index=False):
        geom = getattr(row, "geometry")
        attrs = {column: getattr(row, column, None) for column in keep_columns}
        for line in iter_line_parts(geom):
            length = float(line.length)
            if length <= 0.0:
                continue
            n_samples = max(1, int(round(length / spacing)))
            weight = length / n_samples
            for i in range(n_samples):
                distance = min((i + 0.5) * weight, length)
                records.append(
                    {
                        **attrs,
                        "weight_m": float(weight),
                        "geometry": line.interpolate(distance),
                    }
                )
    if not records:
        return gpd.GeoDataFrame({"weight_m": []}, geometry=[], crs=lines.crs)
    return gpd.GeoDataFrame(records, geometry="geometry", crs=lines.crs)


def _truth_type_class(row: pd.Series) -> str:
    segment_type = str(row.get("SEGMENTTYP", "") or "").strip().upper()
    try:
        diameter = float(row.get("DIAMETER", np.nan))
    except (TypeError, ValueError):
        diameter = np.nan
    if segment_type == "MS" or (np.isfinite(diameter) and diameter >= 375.0):
        return "trunk"
    if segment_type == "LM" or (np.isfinite(diameter) and 225.0 <= diameter < 375.0):
        return "collector"
    if segment_type == "RM" or (np.isfinite(diameter) and diameter <= 160.0):
        return "local_reticulation"
    return "other"


def _predicted_presence_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _reconstruct_estimated_lines(
    graph,
    predictions: pd.DataFrame,
    *,
    aoi_id: str,
) -> gpd.GeoDataFrame:
    segments = graph.road_segments.copy()
    segments["offset_lane"] = segments.apply(road_offset_lane_name, axis=1)
    pred = predictions[predictions["aoi_id"].astype(str) == str(aoi_id)].copy()
    if pred.empty:
        return gpd.GeoDataFrame(geometry=[], crs=segments.crs)
    pred["predicted_presence_bool"] = _predicted_presence_bool(pred["predicted_presence"])
    selected = pred[pred["predicted_presence_bool"]][
        ["source_index", "predicted_lane_name", "presence_probability"]
    ].copy()
    if selected.empty:
        return gpd.GeoDataFrame(geometry=[], crs=segments.crs)
    estimated = segments.merge(
        selected,
        left_on=["source_index", "offset_lane"],
        right_on=["source_index", "predicted_lane_name"],
        how="inner",
    )
    return gpd.GeoDataFrame(estimated, geometry="geometry", crs=segments.crs)


def _coverage_by_truth_type(
    truth: gpd.GeoDataFrame,
    estimated: gpd.GeoDataFrame,
    *,
    threshold_m: float,
    sample_spacing_m: float,
    aoi_id: str,
    split: str,
) -> pd.DataFrame:
    if truth.empty:
        return pd.DataFrame()
    truth = truth.copy()
    truth["SEGMENTTYP"] = truth.get("SEGMENTTYP", "unknown").fillna("unknown").astype(str)
    truth["truth_type_class"] = truth.apply(_truth_type_class, axis=1)
    truth_samples = _sample_lines(
        truth,
        sample_spacing_m=sample_spacing_m,
        keep_columns=["SEGMENTTYP", "truth_type_class", "DIAMETER"],
    )
    if truth_samples.empty:
        return pd.DataFrame()
    if estimated.empty:
        truth_samples["covered"] = False
    else:
        candidates = estimated[["geometry"]].copy()
        joined = gpd.sjoin_nearest(
            truth_samples,
            candidates,
            how="left",
            max_distance=float(threshold_m),
            distance_col="distance_to_prediction_m",
        )
        nearest = joined.groupby(level=0)["distance_to_prediction_m"].min()
        truth_samples["covered"] = nearest.reindex(truth_samples.index).notna()

    rows: list[dict] = []
    for keys, group in truth_samples.groupby(["SEGMENTTYP", "truth_type_class"], dropna=False):
        segment_type, type_class = keys
        total = float(group["weight_m"].sum())
        covered = float(group.loc[group["covered"], "weight_m"].sum())
        rows.append(
            {
                "aoi_id": aoi_id,
                "split": split,
                "SEGMENTTYP": segment_type,
                "truth_type_class": type_class,
                "truth_length_m": total,
                "covered_truth_length_m": covered,
                "truth_length_recall": covered / total if total > 0.0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _predicted_precision(
    estimated: gpd.GeoDataFrame,
    truth: gpd.GeoDataFrame,
    *,
    threshold_m: float,
    sample_spacing_m: float,
    aoi_id: str,
    split: str,
) -> dict[str, float]:
    total_length = float(estimated.geometry.length.sum()) if not estimated.empty else 0.0
    out = {
        "aoi_id": aoi_id,
        "split": split,
        "predicted_length_m": total_length,
        "covered_predicted_length_m": 0.0,
        "predicted_length_precision": 0.0,
    }
    if estimated.empty or truth.empty or total_length <= 0.0:
        return out
    pred_samples = _sample_lines(
        estimated,
        sample_spacing_m=sample_spacing_m,
        keep_columns=["presence_probability", "predicted_lane_name"],
    )
    if pred_samples.empty:
        return out
    candidates = truth[["geometry"]].copy()
    joined = gpd.sjoin_nearest(
        pred_samples,
        candidates,
        how="left",
        max_distance=float(threshold_m),
        distance_col="distance_to_truth_m",
    )
    nearest = joined.groupby(level=0)["distance_to_truth_m"].min()
    covered_index = nearest.index[nearest.notna()]
    covered = float(pred_samples.loc[covered_index, "weight_m"].sum())
    out["covered_predicted_length_m"] = covered
    out["predicted_length_precision"] = covered / total_length
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate predicted estimated utilities against truth by SEGMENTTYP "
            "using sampled length coverage at a spatial threshold."
        )
    )
    parser.add_argument("--configs", nargs="+", default=["configs/aois_2km_gap500_115/*.yaml"])
    parser.add_argument(
        "--predictions",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "road_offset_lane_gnn_5lane_aoi115_sage_no_xy"
        / "road_offset_lane_node_predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "road_offset_lane_gnn_5lane_aoi115_sage_no_xy"
        / "segment_type_eval_10m",
    )
    parser.add_argument("--threshold-m", type=float, default=10.0)
    parser.add_argument("--sample-spacing-m", type=float, default=10.0)
    parser.add_argument("--max-configs", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_paths = _expand_configs(args.configs)
    if int(args.max_configs) > 0:
        config_paths = config_paths[: int(args.max_configs)]
    predictions = pd.read_csv(_resolve(args.predictions))

    start = time.perf_counter()
    type_frames: list[pd.DataFrame] = []
    precision_rows: list[dict[str, float]] = []
    for config_path in config_paths:
        config = load_anchor_free_config(config_path)
        config.setdefault("graph", {})["candidate_graph_type"] = "road_offsets"
        graph, _, _, _, _, truth = prepare_anchor_free_inputs(config)
        aoi_config = dict(config.get("aoi", {}))
        aoi_id = str(aoi_config.get("aoi_id", config.get("experiment_name", config_path.stem)))
        split = str(aoi_config.get("split", "unknown")).lower()
        estimated = _reconstruct_estimated_lines(graph, predictions, aoi_id=aoi_id)
        if graph.road_segments.crs and truth.crs and str(graph.road_segments.crs) != str(truth.crs):
            truth = truth.to_crs(graph.road_segments.crs)
        type_metrics = _coverage_by_truth_type(
            truth,
            estimated,
            threshold_m=float(args.threshold_m),
            sample_spacing_m=float(args.sample_spacing_m),
            aoi_id=aoi_id,
            split=split,
        )
        if not type_metrics.empty:
            type_frames.append(type_metrics)
        precision_rows.append(
            _predicted_precision(
                estimated,
                truth,
                threshold_m=float(args.threshold_m),
                sample_spacing_m=float(args.sample_spacing_m),
                aoi_id=aoi_id,
                split=split,
            )
        )
        print(
            f"{aoi_id} ({split}): truth_lines={len(truth)} "
            f"predicted_segments={len(estimated)}",
            flush=True,
        )

    per_aoi_type = (
        pd.concat(type_frames, ignore_index=True)
        if type_frames
        else pd.DataFrame()
    )
    precision_by_aoi = pd.DataFrame(precision_rows)

    if not per_aoi_type.empty:
        by_segment_type = (
            per_aoi_type.groupby(["SEGMENTTYP", "truth_type_class"], dropna=False)[
                ["truth_length_m", "covered_truth_length_m"]
            ]
            .sum()
            .reset_index()
        )
        by_segment_type["truth_length_recall"] = (
            by_segment_type["covered_truth_length_m"]
            / by_segment_type["truth_length_m"]
        )
        by_class = (
            per_aoi_type.groupby(["truth_type_class"], dropna=False)[
                ["truth_length_m", "covered_truth_length_m"]
            ]
            .sum()
            .reset_index()
        )
        by_class["truth_length_recall"] = by_class["covered_truth_length_m"] / by_class["truth_length_m"]
    else:
        by_segment_type = pd.DataFrame()
        by_class = pd.DataFrame()

    total_pred = float(precision_by_aoi["predicted_length_m"].sum())
    covered_pred = float(precision_by_aoi["covered_predicted_length_m"].sum())
    summary = {
        "workstream": "Codex",
        "description": "spatial accuracy by sewer truth SEGMENTTYP",
        "threshold_m": float(args.threshold_m),
        "sample_spacing_m": float(args.sample_spacing_m),
        "n_configs": len(config_paths),
        "runtime_sec": time.perf_counter() - start,
        "predicted_length_m": total_pred,
        "covered_predicted_length_m": covered_pred,
        "predicted_length_precision": covered_pred / total_pred if total_pred > 0 else 0.0,
        "output_dir": _relative(output_dir),
    }

    per_aoi_type.to_csv(output_dir / "per_aoi_segment_type_recall.csv", index=False)
    by_segment_type.to_csv(output_dir / "segment_type_recall.csv", index=False)
    by_class.to_csv(output_dir / "segment_class_recall.csv", index=False)
    precision_by_aoi.to_csv(output_dir / "predicted_precision_by_aoi.csv", index=False)
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("Segment type recall:")
    if not by_segment_type.empty:
        print(
            by_segment_type.sort_values("truth_length_m", ascending=False).to_string(
                index=False,
                formatters={
                    "truth_length_m": "{:.1f}".format,
                    "covered_truth_length_m": "{:.1f}".format,
                    "truth_length_recall": "{:.4f}".format,
                },
            )
        )
    print(f"Predicted length precision @ {float(args.threshold_m):g}m: {summary['predicted_length_precision']:.4f}")
    print(f"Wrote outputs to {_relative(output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
