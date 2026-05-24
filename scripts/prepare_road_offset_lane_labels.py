"""Prepare five-lane source-road labels for road_offsets candidate graphs.

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
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipe_network_completion.anchor_free.config import load_anchor_free_config  # noqa: E402
from pipe_network_completion.anchor_free.labels import (  # noqa: E402
    FIXED_ROAD_OFFSET_LANES,
    label_road_offset_lanes_from_utility_lines,
)
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


def _write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gdf.empty:
        path.write_text('{"type":"FeatureCollection","features":[]}\n', encoding="utf-8")
    else:
        path.write_text(gdf.to_json(drop_id=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build road_offsets candidate graphs and prepare one source-road "
            "label with five fixed lane classes from nearest sampled utility truth."
        )
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["configs/aois_2km_gap500_115/*.yaml"],
        help="Config paths or glob patterns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "road_offset_lane_labels",
    )
    parser.add_argument("--sample-spacing-m", type=float, default=10.0)
    parser.add_argument("--max-assignment-distance-m", type=float, default=50.0)
    parser.add_argument("--min-assigned-truth-length-m", type=float, default=10.0)
    parser.add_argument("--min-assigned-fraction", type=float, default=0.0)
    parser.add_argument("--ambiguous-weight-margin", type=float, default=0.1)
    parser.add_argument("--max-configs", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_paths = _expand_configs(args.configs)
    if int(args.max_configs) > 0:
        config_paths = config_paths[: int(args.max_configs)]

    start = time.perf_counter()
    label_frames: list[gpd.GeoDataFrame] = []
    rows: list[dict] = []

    for config_path in config_paths:
        config = load_anchor_free_config(config_path)
        config.setdefault("graph", {})["candidate_graph_type"] = "road_offsets"
        graph, _, _, _, _, utility_truth = prepare_anchor_free_inputs(config)
        lane_labels = label_road_offset_lanes_from_utility_lines(
            graph,
            utility_truth,
            sample_spacing_m=float(args.sample_spacing_m),
            max_assignment_distance_m=float(args.max_assignment_distance_m),
            min_assigned_truth_length_m=float(args.min_assigned_truth_length_m),
            min_assigned_fraction=float(args.min_assigned_fraction),
            ambiguous_weight_margin=float(args.ambiguous_weight_margin),
        )

        aoi_config = dict(config.get("aoi", {}))
        aoi_id = str(aoi_config.get("aoi_id", config.get("experiment_name", config_path.stem)))
        split = str(aoi_config.get("split", "unknown"))
        labels = lane_labels.labels.copy()
        labels.insert(0, "aoi_id", aoi_id)
        labels.insert(1, "aoi_split", split)
        labels.insert(2, "config_path", _relative(config_path))
        label_frames.append(labels)

        rows.append(
            {
                "aoi_id": aoi_id,
                "split": split,
                "n_source_roads": int(len(labels)),
                "positive_source_roads": int(labels["y"].sum()),
                "ambiguous_positive_source_roads": int(
                    labels.loc[labels["y"].astype(int) == 1, "is_ambiguous"].sum()
                ),
                "positive_rate": float(labels["y"].mean()) if len(labels) else 0.0,
            }
        )
        print(
            f"{aoi_id} ({split}): source_roads={len(labels)} "
            f"positive={int(labels['y'].sum())} "
            f"ambiguous={int(labels.loc[labels['y'].astype(int) == 1, 'is_ambiguous'].sum())}",
            flush=True,
        )

    if label_frames:
        labels_all = gpd.GeoDataFrame(
            pd.concat(label_frames, ignore_index=True),
            geometry="geometry",
            crs=label_frames[0].crs,
        )
    else:
        labels_all = gpd.GeoDataFrame(geometry=[], crs=None)

    _write_geojson(labels_all, output_dir / "road_offset_lane_labels.geojson")
    labels_all.drop(columns=["geometry"], errors="ignore").to_csv(
        output_dir / "road_offset_lane_labels.csv",
        index=False,
    )

    per_aoi = pd.DataFrame(rows)
    per_aoi.to_csv(output_dir / "road_offset_lane_labels_by_aoi.csv", index=False)
    if not labels_all.empty:
        (
            labels_all.groupby(["aoi_split", "lane_name", "y"], dropna=False)
            .size()
            .reset_index(name="count")
            .to_csv(output_dir / "road_offset_lane_label_summary.csv", index=False)
        )

    summary = {
        "workstream": "Codex",
        "description": "five-lane source-road labels from nearest sampled utility truth",
        "lane_names": list(FIXED_ROAD_OFFSET_LANES),
        "n_configs": len(config_paths),
        "n_source_roads": int(len(labels_all)),
        "positive_source_roads": int(labels_all["y"].sum()) if not labels_all.empty else 0,
        "sample_spacing_m": float(args.sample_spacing_m),
        "max_assignment_distance_m": float(args.max_assignment_distance_m),
        "min_assigned_truth_length_m": float(args.min_assigned_truth_length_m),
        "min_assigned_fraction": float(args.min_assigned_fraction),
        "ambiguous_weight_margin": float(args.ambiguous_weight_margin),
        "runtime_sec": time.perf_counter() - start,
        "output_dir": _relative(output_dir),
    }
    (output_dir / "road_offset_lane_label_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote labels to {_relative(output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
