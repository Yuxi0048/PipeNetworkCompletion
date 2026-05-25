"""Evaluate candidate-graph truth-length recall without training.

Workstream: Codex
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipe_network_completion.anchor_free.candidate_recall import (  # noqa: E402
    candidate_representability_metrics,
    candidate_source_summary,
)
from pipe_network_completion.anchor_free.config import load_anchor_free_config  # noqa: E402
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
        resolved_pattern = str(_resolve(pattern))
        matches = sorted(Path(match) for match in glob.glob(resolved_pattern))
        if matches:
            paths.extend(matches)
        else:
            path = _resolve(pattern)
            if path.exists():
                paths.append(path)
            else:
                raise FileNotFoundError(pattern)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            deduped.append(path)
            seen.add(resolved)
    return deduped


def _write_geojson(gdf, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gdf.empty:
        path.write_text('{"type":"FeatureCollection","features":[]}\n', encoding="utf-8")
    else:
        path.write_text(gdf.to_json(drop_id=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build anchor-free candidate graphs from one or more configs and "
            "estimate truth-length recall at spatial tolerance buffers. No "
            "model training is run."
        )
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["configs/aois/anchor_free_small_aoi_*.yaml"],
        help="Config paths or glob patterns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "candidate_recall" / "small_aoi_hybrid",
    )
    parser.add_argument("--buffers-m", nargs="+", type=float, default=[5, 10, 20, 30, 50])
    parser.add_argument("--sample-spacing-m", type=float, default=50.0)
    parser.add_argument(
        "--candidate-graph-type",
        choices=["config", "road", "hybrid", "road_offsets"],
        default="config",
        help="Use config value, or override all configs to road/hybrid/road_offsets.",
    )
    parser.add_argument(
        "--write-candidates",
        action="store_true",
        help="Export each generated candidate graph as GeoJSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_paths = _expand_configs(args.configs)
    rows: list[dict] = []
    source_rows: list[dict] = []

    for config_path in config_paths:
        start = time.perf_counter()
        config = load_anchor_free_config(config_path)
        if args.candidate_graph_type != "config":
            config.setdefault("graph", {})["candidate_graph_type"] = args.candidate_graph_type
        graph, _, _, _, _, utility_truth = prepare_anchor_free_inputs(config)
        aoi_config = dict(config.get("aoi", {}))
        aoi_id = str(aoi_config.get("aoi_id", config.get("experiment_name", config_path.stem)))
        split = str(aoi_config.get("split", "unknown"))
        metrics = candidate_representability_metrics(
            graph,
            utility_truth,
            buffers_m=args.buffers_m,
            sample_spacing_m=float(args.sample_spacing_m),
        )
        row = {
            "aoi_id": aoi_id,
            "split": split,
            "config_path": _relative(config_path),
            "candidate_graph_type": str(graph.metadata.get("candidate_graph_type", "road")),
            "n_intersections": int(len(graph.intersections)),
            "runtime_sec": round(time.perf_counter() - start, 3),
        }
        row.update(metrics)
        rows.append(row)

        for source, summary in candidate_source_summary(graph).items():
            source_rows.append(
                {
                    "aoi_id": aoi_id,
                    "split": split,
                    "candidate_source": source,
                    "count": int(summary["count"]),
                    "length_m": float(summary["length_m"]),
                }
            )

        if args.write_candidates:
            candidates = graph.road_segments.copy()
            candidates["aoi_id"] = aoi_id
            candidates["split"] = split
            _write_geojson(candidates, output_dir / "candidate_graphs" / f"{aoi_id}.geojson")

        print(
            f"{aoi_id} ({split}): "
            f"segments={int(metrics.get('candidate_count', 0))} "
            f"truth_km={metrics.get('truth_total_length_sampled_m', 0.0) / 1000:.2f} "
            f"recall_10m={metrics.get('recall_10m', float('nan')):.3f} "
            f"recall_50m={metrics.get('recall_50m', float('nan')):.3f}"
        )

    results = pd.DataFrame(rows)
    source_summary = pd.DataFrame(source_rows)
    results.to_csv(output_dir / "candidate_recall_by_aoi.csv", index=False)
    source_summary.to_csv(output_dir / "candidate_source_summary_by_aoi.csv", index=False)

    aggregate = {
        "workstream": "Codex",
        "description": "candidate graph truth-length recall; no model training",
        "n_configs": len(config_paths),
        "buffers_m": [float(value) for value in args.buffers_m],
        "sample_spacing_m": float(args.sample_spacing_m),
        "write_candidates": bool(args.write_candidates),
        "output_dir": _relative(output_dir),
        "mean_recall": {
            column: float(results[column].mean())
            for column in results.columns
            if column.startswith("recall_")
        },
        "length_weighted_recall": {},
        "split_mean_recall": {},
    }
    if "truth_total_length_sampled_m" in results.columns:
        weights = results["truth_total_length_sampled_m"].astype(float)
        total_weight = float(weights.sum())
        if total_weight > 0:
            aggregate["length_weighted_recall"] = {
                column: float((results[column].astype(float) * weights).sum() / total_weight)
                for column in results.columns
                if column.startswith("recall_")
            }
    for split, group in results.groupby("split"):
        aggregate["split_mean_recall"][str(split)] = {
            column: float(group[column].mean())
            for column in group.columns
            if column.startswith("recall_")
        }
    (output_dir / "candidate_recall_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"Wrote {_relative(output_dir / 'candidate_recall_by_aoi.csv')}")
    print(f"Wrote {_relative(output_dir / 'candidate_recall_summary.json')}")
    print("No model training was started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
