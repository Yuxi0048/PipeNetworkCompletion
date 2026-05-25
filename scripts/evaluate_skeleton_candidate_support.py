"""Evaluate road/drainage skeleton candidate support without training.

Workstream: Codex

The candidate support used by ``train_skeleton_buffer_gnn.py`` is a simple line
set: road centerlines plus optional drainage/watercourse lines. This script
measures how much truth sewer-main length is spatially representable by that
support at several tolerances.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipe_network_completion.anchor_free.config import load_anchor_free_config  # noqa: E402
from scripts.evaluate_skeleton_spatial_tolerances import (  # noqa: E402
    _covered_length_by_buffered_support,
)
from scripts.train_skeleton_buffer_gnn import (  # noqa: E402
    _prepare_skeleton_candidates,
    _read_layer,
)


def _resolve(path: str | Path | None) -> Path | None:
    if path in (None, ""):
        return None
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
            if path is None or not path.exists():
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


def _write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gdf.empty:
        path.write_text('{"type":"FeatureCollection","features":[]}\n', encoding="utf-8")
    else:
        path.write_text(gdf.to_json(drop_id=True), encoding="utf-8")


def _load_candidate_support(
    config_path: Path,
    *,
    include_drainage: bool,
) -> tuple[str, str, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    config = load_anchor_free_config(config_path)
    data = dict(config.get("data", {}))
    graph = dict(config.get("graph", {}))
    aoi = dict(config.get("aoi", {}))
    aoi_id = str(aoi.get("aoi_id", config_path.stem))
    split = str(aoi.get("split", "unknown"))
    target_crs = str(graph.get("target_crs", "EPSG:28356"))

    roads = _read_layer(data.get("roads_path"), target_crs)
    drainage = _read_layer(data.get("watercourse_drainage_lines_path"), target_crs)
    if drainage.empty:
        drainage = _read_layer(data.get("watercourse_corridor_centrelines_path"), target_crs)
    truth = _read_layer(data.get("utility_truth_path"), target_crs)

    candidates = _prepare_skeleton_candidates(roads, drainage, include_drainage=include_drainage)
    candidates.insert(0, "aoi_id", aoi_id)
    candidates.insert(1, "aoi_split", split)
    return aoi_id, split, candidates, truth


def _evaluate_variant(
    config_paths: list[Path],
    *,
    variant: str,
    include_drainage: bool,
    tolerances_m: list[float],
    output_dir: Path,
    write_candidates: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    for config_path in config_paths:
        start = time.perf_counter()
        aoi_id, split, candidates, truth = _load_candidate_support(
            config_path,
            include_drainage=include_drainage,
        )
        if write_candidates:
            out = candidates.copy()
            out["variant"] = variant
            _write_geojson(out, output_dir / "candidate_graphs" / variant / f"{aoi_id}.geojson")
        for source, group in candidates.groupby("candidate_source", dropna=False):
            source_rows.append(
                {
                    "variant": variant,
                    "aoi_id": aoi_id,
                    "split": split,
                    "candidate_source": str(source),
                    "count": int(len(group)),
                    "length_m": float(group.geometry.length.sum()),
                }
            )

        truth_length = float(truth.geometry.length.sum()) if not truth.empty else 0.0
        candidate_length = float(candidates.geometry.length.sum()) if not candidates.empty else 0.0
        for tolerance in tolerances_m:
            pred_cov, pred_len = _covered_length_by_buffered_support(
                candidates,
                truth,
                buffer_m=float(tolerance),
            )
            truth_cov, truth_len = _covered_length_by_buffered_support(
                truth,
                candidates,
                buffer_m=float(tolerance),
            )
            precision = pred_cov / pred_len if pred_len > 0.0 else 0.0
            recall = truth_cov / truth_len if truth_len > 0.0 else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
            rows.append(
                {
                    "variant": variant,
                    "aoi_id": aoi_id,
                    "split": split,
                    "config_path": _relative(config_path),
                    "tolerance_m": float(tolerance),
                    "candidate_count": int(len(candidates)),
                    "candidate_length_m": candidate_length,
                    "truth_length_m": truth_length,
                    "covered_candidate_length_m": pred_cov,
                    "covered_truth_length_m": truth_cov,
                    "length_precision": precision,
                    "truth_length_recall": recall,
                    "length_f1": f1,
                    "runtime_sec": round(time.perf_counter() - start, 3),
                }
            )
        print(
            f"{variant} {aoi_id} ({split}): "
            f"candidates={len(candidates)} truth_km={truth_length / 1000.0:.2f}"
        )
    return rows, source_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["configs/aois_2km_gap500_112_osm_bpoints_all_mains_watercourses_complete/*.yaml"],
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "skeleton_candidate_support_watercourses")
    parser.add_argument("--tolerances-m", nargs="+", type=float, default=[5, 10, 20, 30])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["road_only", "road_drainage"],
        default=["road_only", "road_drainage"],
    )
    parser.add_argument("--write-candidates", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    if output_dir is None:
        raise ValueError("output-dir is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_paths = _expand_configs(args.configs)

    all_rows: list[dict[str, object]] = []
    all_source_rows: list[dict[str, object]] = []
    for variant in args.variants:
        rows, source_rows = _evaluate_variant(
            config_paths,
            variant=variant,
            include_drainage=variant == "road_drainage",
            tolerances_m=[float(value) for value in args.tolerances_m],
            output_dir=output_dir,
            write_candidates=bool(args.write_candidates),
        )
        all_rows.extend(rows)
        all_source_rows.extend(source_rows)

    by_aoi = pd.DataFrame(all_rows)
    by_source = pd.DataFrame(all_source_rows)
    by_aoi.to_csv(output_dir / "skeleton_candidate_support_by_aoi.csv", index=False)
    by_source.to_csv(output_dir / "skeleton_candidate_source_by_aoi.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    for (variant, tolerance), group in by_aoi.groupby(["variant", "tolerance_m"]):
        truth_total = float(group["truth_length_m"].sum())
        candidate_total = float(group["candidate_length_m"].sum())
        covered_truth = float(group["covered_truth_length_m"].sum())
        covered_candidate = float(group["covered_candidate_length_m"].sum())
        precision = covered_candidate / candidate_total if candidate_total > 0.0 else 0.0
        recall = covered_truth / truth_total if truth_total > 0.0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        summary_rows.append(
            {
                "variant": variant,
                "tolerance_m": float(tolerance),
                "n_aois": int(group["aoi_id"].nunique()),
                "candidate_count": int(group["candidate_count"].sum()),
                "candidate_length_m": candidate_total,
                "truth_length_m": truth_total,
                "covered_candidate_length_m": covered_candidate,
                "covered_truth_length_m": covered_truth,
                "length_precision": precision,
                "truth_length_recall": recall,
                "length_f1": f1,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["variant", "tolerance_m"])
    summary.to_csv(output_dir / "skeleton_candidate_support_summary.csv", index=False)

    manifest = {
        "workstream": "Codex",
        "description": "road-only versus road+drainage skeleton candidate support; no training",
        "n_configs": len(config_paths),
        "variants": list(args.variants),
        "tolerances_m": [float(value) for value in args.tolerances_m],
        "output_dir": _relative(output_dir),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(summary.to_string(index=False))
    print(f"Wrote {_relative(output_dir / 'skeleton_candidate_support_summary.csv')}")
    print("No model training was started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
