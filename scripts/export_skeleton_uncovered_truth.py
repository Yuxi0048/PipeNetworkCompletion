"""Export truth sewer-main portions not covered by skeleton predictions.

Workstream: Codex

This script writes GeoJSON layers that answer: "where are the truth mains that
are not included within a given spatial tolerance?"

Two candidate sets are useful:

- all_candidates: support-generation misses, unrecoverable by the classifier;
- selected_prediction: misses after applying the trained model threshold.
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipe_network_completion.anchor_free.candidate_recall import iter_line_parts  # noqa: E402
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
        if truth_path:
            out[aoi_id] = (_resolve(truth_path), split)
    return out


def _prediction_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _aoi_column(predictions: gpd.GeoDataFrame) -> str:
    for column in ("aoi_id", "aoi_id_x", "aoi"):
        if column in predictions.columns:
            return column
    raise ValueError("Prediction layer is missing an AOI id column")


def _split_column(predictions: gpd.GeoDataFrame) -> str:
    for column in ("aoi_split", "aoi_split_x", "split"):
        if column in predictions.columns:
            return column
    raise ValueError("Prediction layer is missing a split column")


def _truth_for_aoi(path: Path, crs) -> gpd.GeoDataFrame:
    truth = gpd.read_file(path)
    if truth.crs is None:
        truth = truth.set_crs(crs)
    elif crs is not None and truth.crs != crs:
        truth = truth.to_crs(crs)
    truth = truth[truth.geometry.notna() & ~truth.geometry.is_empty].copy()
    truth = truth[truth.geometry.length > 0.0].copy()
    return truth


def _candidate_sets(predictions: gpd.GeoDataFrame, requested: list[str]) -> dict[str, gpd.GeoDataFrame]:
    out: dict[str, gpd.GeoDataFrame] = {}
    requested_set = set(requested)
    if "all_candidates" in requested_set:
        out["all_candidates"] = predictions
    if "selected_prediction" in requested_set:
        if "predicted_presence" not in predictions.columns:
            raise ValueError("selected_prediction requested but predicted_presence column is missing")
        out["selected_prediction"] = predictions[_prediction_bool(predictions["predicted_presence"])].copy()
    return out


def _truth_row_attrs(row: pd.Series) -> dict[str, object]:
    keep = [
        "SEGMENTTYP",
        "MATERIAL",
        "DIAMETER",
        "USIL",
        "DSIL",
        "Shape_Leng",
        "OBJECTID",
        "ASSETID",
    ]
    return {column: row.get(column) for column in keep if column in row.index}


def _uncovered_parts_for_aoi(
    truth: gpd.GeoDataFrame,
    candidates: gpd.GeoDataFrame,
    *,
    aoi_id: str,
    split: str,
    candidate_set: str,
    tolerance_m: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    truth_length = float(truth.geometry.length.sum()) if not truth.empty else 0.0
    candidate_length = float(candidates.geometry.length.sum()) if not candidates.empty else 0.0
    if truth.empty:
        return [], {
            "aoi_id": aoi_id,
            "split": split,
            "candidate_set": candidate_set,
            "tolerance_m": float(tolerance_m),
            "truth_length_m": 0.0,
            "uncovered_truth_length_m": 0.0,
            "covered_truth_length_m": 0.0,
            "truth_length_recall": 0.0,
            "candidate_length_m": candidate_length,
            "n_uncovered_parts": 0,
        }

    if candidates.empty:
        support_union = None
    else:
        support_union = unary_union(list(candidates.geometry.buffer(float(tolerance_m))))

    records: list[dict[str, object]] = []
    uncovered_total = 0.0
    for truth_index, row in truth.reset_index(drop=False).iterrows():
        geom = row.geometry
        uncovered = geom if support_union is None or support_union.is_empty else geom.difference(support_union)
        if uncovered is None or uncovered.is_empty:
            continue
        for part_index, part in enumerate(iter_line_parts(uncovered)):
            length = float(part.length)
            if length <= 0.0:
                continue
            uncovered_total += length
            records.append(
                {
                    "aoi_id": aoi_id,
                    "split": split,
                    "candidate_set": candidate_set,
                    "tolerance_m": float(tolerance_m),
                    "truth_row": int(row.get("index", truth_index)),
                    "part_index": int(part_index),
                    "uncovered_length_m": length,
                    **_truth_row_attrs(row),
                    "geometry": part,
                }
            )

    covered = max(truth_length - uncovered_total, 0.0)
    return records, {
        "aoi_id": aoi_id,
        "split": split,
        "candidate_set": candidate_set,
        "tolerance_m": float(tolerance_m),
        "truth_length_m": truth_length,
        "uncovered_truth_length_m": uncovered_total,
        "covered_truth_length_m": covered,
        "truth_length_recall": covered / truth_length if truth_length > 0.0 else 0.0,
        "candidate_length_m": candidate_length,
        "n_uncovered_parts": len(records),
    }


def export_uncovered(
    predictions: gpd.GeoDataFrame,
    truth_lookup: dict[str, tuple[Path, str]],
    *,
    split: str,
    tolerances_m: list[float],
    candidate_sets: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    aoi_col = _aoi_column(predictions)
    split_col = _split_column(predictions)
    predictions = predictions[predictions[split_col].astype(str) == split].copy()
    predictions = predictions[predictions.geometry.notna() & ~predictions.geometry.is_empty].copy()
    predictions = predictions[predictions.geometry.length > 0.0].copy()

    summary_rows: list[dict[str, object]] = []
    sets = _candidate_sets(predictions, candidate_sets)
    output_dir.mkdir(parents=True, exist_ok=True)

    for set_name, candidate_df in sets.items():
        for tolerance in tolerances_m:
            records: list[dict[str, object]] = []
            for aoi_id, group in candidate_df.groupby(aoi_col):
                aoi_id = str(aoi_id)
                truth_info = truth_lookup.get(aoi_id)
                if truth_info is None:
                    continue
                truth_path, truth_split = truth_info
                truth = _truth_for_aoi(truth_path, group.crs)
                candidates = gpd.GeoDataFrame(group.copy(), geometry="geometry", crs=group.crs)
                aoi_records, row = _uncovered_parts_for_aoi(
                    truth,
                    candidates,
                    aoi_id=aoi_id,
                    split=truth_split or split,
                    candidate_set=set_name,
                    tolerance_m=float(tolerance),
                )
                records.extend(aoi_records)
                summary_rows.append(row)

            suffix = f"{float(tolerance):g}m".replace(".", "p")
            layer_path = output_dir / f"uncovered_truth_{set_name}_{suffix}.geojson"
            if records:
                gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=predictions.crs)
                gdf.to_file(layer_path, driver="GeoJSON")
            else:
                layer_path.write_text('{"type":"FeatureCollection","features":[]}\n', encoding="utf-8")

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "uncovered_truth_summary_by_aoi.csv"
    summary.to_csv(summary_path, index=False)

    aggregate = (
        summary.groupby(["candidate_set", "tolerance_m"], as_index=False)
        .agg(
            n_aois=("aoi_id", "nunique"),
            truth_length_m=("truth_length_m", "sum"),
            uncovered_truth_length_m=("uncovered_truth_length_m", "sum"),
            covered_truth_length_m=("covered_truth_length_m", "sum"),
            candidate_length_m=("candidate_length_m", "sum"),
            n_uncovered_parts=("n_uncovered_parts", "sum"),
        )
        .sort_values(["candidate_set", "tolerance_m"])
    )
    aggregate["truth_length_recall"] = aggregate["covered_truth_length_m"] / aggregate["truth_length_m"]
    aggregate.to_csv(output_dir / "uncovered_truth_summary_aggregate.csv", index=False)

    largest = summary.sort_values("uncovered_truth_length_m", ascending=False).head(30)
    largest.to_csv(output_dir / "largest_uncovered_aoi_cases.csv", index=False)
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", default=["configs/aois_2km_gap500_115_osm_bpoints_all_mains/*.yaml"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--tolerances-m", nargs="+", type=float, default=[5, 10, 20, 30])
    parser.add_argument(
        "--candidate-sets",
        nargs="+",
        choices=["all_candidates", "selected_prediction"],
        default=["all_candidates", "selected_prediction"],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = gpd.read_file(args.predictions)
    truth_lookup = _truth_paths_by_aoi(_expand_configs(args.configs))
    aggregate = export_uncovered(
        predictions,
        truth_lookup,
        split=args.split,
        tolerances_m=[float(value) for value in args.tolerances_m],
        candidate_sets=list(args.candidate_sets),
        output_dir=args.output_dir,
    )
    print(aggregate.to_string(index=False))
    print(f"Wrote uncovered-truth layers to {args.output_dir}")


if __name__ == "__main__":
    main()
