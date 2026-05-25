"""Evaluate skeleton-buffer predictions at multiple spatial tolerances.

Workstream: Codex

This is a post-training evaluator. It treats the saved predicted linework as a
line set and compares it against ground-truth sewer mains using symmetric
line-buffer length coverage:

- precision: predicted length within tolerance of any truth line;
- recall: truth length within tolerance of any predicted line.
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipe_network_completion.anchor_free.config import load_anchor_free_config  # noqa: E402


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


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


def _truth_paths_by_aoi(config_paths: list[Path]) -> dict[str, tuple[Path, str]]:
    out: dict[str, tuple[Path, str]] = {}
    for path in config_paths:
        cfg = load_anchor_free_config(path)
        aoi = cfg.get("aoi", {})
        aoi_id = str(aoi.get("aoi_id") or Path(path).stem)
        split = str(aoi.get("split") or "")
        truth_path = cfg.get("data", {}).get("utility_truth_path")
        if not truth_path:
            continue
        out[aoi_id] = (_resolve(truth_path), split)
    return out


def _truth_for_aoi(path: Path, crs) -> gpd.GeoDataFrame:
    truth = gpd.read_file(path)
    if truth.crs is None:
        truth = truth.set_crs(crs)
    elif crs is not None and truth.crs != crs:
        truth = truth.to_crs(crs)
    truth = truth[truth.geometry.notna() & ~truth.geometry.is_empty].copy()
    truth = truth[truth.geometry.length > 0.0].copy()
    return truth


def _prediction_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _split_column(predictions: gpd.GeoDataFrame) -> str:
    for column in ("aoi_split", "aoi_split_x", "split"):
        if column in predictions.columns:
            return column
    raise ValueError("Prediction layer is missing a split column")


def _aoi_column(predictions: gpd.GeoDataFrame) -> str:
    for column in ("aoi_id", "aoi_id_x", "aoi"):
        if column in predictions.columns:
            return column
    raise ValueError("Prediction layer is missing an AOI id column")


def _covered_length_by_buffered_support(
    target: gpd.GeoDataFrame,
    support: gpd.GeoDataFrame,
    *,
    buffer_m: float,
) -> tuple[float, float]:
    target_total = float(target.geometry.length.sum()) if not target.empty else 0.0
    if target.empty or support.empty or target_total <= 0.0:
        return 0.0, target_total

    support_buffers = gpd.GeoDataFrame(geometry=support.geometry.buffer(float(buffer_m)), crs=support.crs)
    support_buffers = support_buffers[support_buffers.geometry.notna() & ~support_buffers.geometry.is_empty].copy()
    if support_buffers.empty:
        return 0.0, target_total

    buffer_geoms = list(support_buffers.geometry)
    spatial_index = support_buffers.sindex
    covered = 0.0
    for geom in target.geometry:
        if geom is None or geom.is_empty:
            continue
        try:
            idxs = spatial_index.query(geom, predicate="intersects")
        except TypeError:
            idxs = list(spatial_index.intersection(geom.bounds))
        if len(idxs) == 0:
            continue
        local_union = unary_union([buffer_geoms[int(idx)] for idx in idxs])
        covered += float(geom.intersection(local_union).length)
    return covered, target_total


def _candidate_sets(predictions: gpd.GeoDataFrame) -> dict[str, gpd.GeoDataFrame]:
    sets = {"all_candidates": predictions}
    if "predicted_presence" in predictions.columns:
        selected = predictions[_prediction_bool(predictions["predicted_presence"])].copy()
        sets["selected_prediction"] = selected
    return sets


def evaluate(
    predictions: gpd.GeoDataFrame,
    truth_lookup: dict[str, tuple[Path, str]],
    *,
    split: str,
    tolerances_m: list[float],
) -> list[dict[str, object]]:
    aoi_col = _aoi_column(predictions)
    split_col = _split_column(predictions)
    predictions = predictions[predictions[split_col].astype(str) == split].copy()
    predictions = predictions[predictions.geometry.notna() & ~predictions.geometry.is_empty].copy()
    predictions = predictions[predictions.geometry.length > 0.0].copy()

    rows: list[dict[str, object]] = []
    for set_name, candidate_set in _candidate_sets(predictions).items():
        for tolerance in tolerances_m:
            predicted_total = 0.0
            predicted_covered = 0.0
            truth_total = 0.0
            truth_covered = 0.0
            n_segments = 0
            gross_buffer_area = 0.0
            n_aois = 0

            for aoi_id, group in candidate_set.groupby(aoi_col):
                aoi_id = str(aoi_id)
                truth_info = truth_lookup.get(aoi_id)
                if truth_info is None:
                    continue
                truth_path, _truth_split = truth_info
                truth = _truth_for_aoi(truth_path, group.crs)
                if truth.empty:
                    continue

                pred_group = gpd.GeoDataFrame(group.copy(), geometry="geometry", crs=group.crs)
                pred_cov, pred_len = _covered_length_by_buffered_support(
                    pred_group,
                    truth,
                    buffer_m=tolerance,
                )
                truth_cov, truth_len = _covered_length_by_buffered_support(
                    truth,
                    pred_group,
                    buffer_m=tolerance,
                )
                predicted_total += pred_len
                predicted_covered += pred_cov
                truth_total += truth_len
                truth_covered += truth_cov
                n_segments += int(len(pred_group))
                gross_buffer_area += float(pred_group.geometry.buffer(float(tolerance)).area.sum())
                n_aois += 1

            precision = predicted_covered / predicted_total if predicted_total > 0.0 else 0.0
            recall = truth_covered / truth_total if truth_total > 0.0 else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
            rows.append(
                {
                    "candidate_set": set_name,
                    "split": split,
                    "tolerance_m": float(tolerance),
                    "n_aois": n_aois,
                    "n_predicted_segments": n_segments,
                    "predicted_length_m": predicted_total,
                    "truth_length_m": truth_total,
                    "covered_predicted_length_m": predicted_covered,
                    "covered_truth_length_m": truth_covered,
                    "length_precision": precision,
                    "truth_length_recall": recall,
                    "length_f1": f1,
                    "gross_predicted_buffer_area_m2": gross_buffer_area,
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", default=["configs/aois_2km_gap500_115_osm_bpoints_all_mains/*.yaml"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--tolerances-m", nargs="+", type=float, default=[5, 10, 20, 30])
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = gpd.read_file(args.predictions)
    truth_lookup = _truth_paths_by_aoi(_expand_configs(args.configs))
    rows = evaluate(
        predictions,
        truth_lookup,
        split=args.split,
        tolerances_m=[float(value) for value in args.tolerances_m],
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
