"""Merge OSM/all-mains AOI configs with complete watercourse context configs.

Workstream: Codex

This preserves the complete-sewer-main target and OSM building point inputs,
while adding watercourse/drainage paths only for AOIs where watercourse context
was previously marked complete.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _find_config(config_dir: Path, aoi_id: str) -> Path:
    matches = sorted(config_dir.glob(f"*_{aoi_id}.yaml"))
    if not matches:
        matches = sorted(config_dir.glob(f"*{aoi_id}*.yaml"))
    if not matches:
        raise FileNotFoundError(f"No config for {aoi_id!r} in {config_dir}")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--osm-all-mains-config-dir",
        type=Path,
        default=Path("configs/aois_2km_gap500_115_osm_bpoints_all_mains"),
    )
    parser.add_argument(
        "--watercourse-config-dir",
        type=Path,
        default=Path("configs/aois_2km_gap500_115_watercourses_complete"),
    )
    parser.add_argument(
        "--watercourse-completeness-csv",
        type=Path,
        default=Path("data/processed/context/watercourses/watercourse_completeness_by_aoi.csv"),
    )
    parser.add_argument(
        "--output-config-dir",
        type=Path,
        default=Path("configs/aois_2km_gap500_112_osm_bpoints_all_mains_watercourses_complete"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_config_dir.mkdir(parents=True, exist_ok=True)
    root = Path.cwd()
    completeness = pd.read_csv(args.watercourse_completeness_csv)
    complete = completeness[completeness["watercourse_context_complete"].astype(bool)].copy()

    rows: list[dict[str, object]] = []
    config_paths: list[str] = []
    for row in complete.itertuples(index=False):
        aoi_id = str(row.aoi_id)
        base_path = _find_config(args.osm_all_mains_config_dir, aoi_id)
        watercourse_path = _find_config(args.watercourse_config_dir, aoi_id)
        config = json.loads(base_path.read_text(encoding="utf-8"))
        watercourse_config = json.loads(watercourse_path.read_text(encoding="utf-8"))
        wc_data = watercourse_config.get("data", {})

        config.setdefault("data", {}).update(
            {
                "watercourse_drainage_lines_path": wc_data.get("watercourse_drainage_lines_path"),
                "watercourse_corridor_centrelines_path": wc_data.get("watercourse_corridor_centrelines_path"),
                "watercourse_corridors_path": wc_data.get("watercourse_corridors_path"),
            }
        )
        config.setdefault("graph", {}).update(
            {
                "use_watercourses": True,
                "watercourse_context_complete": True,
                "watercourse_buffer_m": float(
                    watercourse_config.get("graph", {}).get("watercourse_buffer_m", 100.0)
                ),
            }
        )
        config.setdefault("aoi", {})["watercourse_context_complete"] = True
        config["experiment_name"] = (
            f"anchor_free_2km_gap500_aoi112_osm_bpoints_all_mains_watercourses_{aoi_id}"
        )

        output_path = args.output_config_dir / base_path.name
        output_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        rel = _relative(output_path, root)
        config_paths.append(rel)
        rows.append(
            {
                "aoi_id": aoi_id,
                "split": str(row.split),
                "config_path": rel,
                "drainage_line_count": int(row.drainage_line_count),
                "drainage_line_length_m": float(row.drainage_line_length_m),
                "corridor_centreline_count": int(row.corridor_centreline_count),
                "corridor_centreline_length_m": float(row.corridor_centreline_length_m),
                "corridor_polygon_count": int(row.corridor_polygon_count),
                "corridor_polygon_area_m2": float(row.corridor_polygon_area_m2),
            }
        )

    pd.DataFrame(rows).to_csv(args.output_config_dir / "config_summary.csv", index=False)
    (args.output_config_dir / "complete_config_paths.txt").write_text(
        "\n".join(config_paths) + ("\n" if config_paths else ""),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} configs to {args.output_config_dir}")
    print("No model training was started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
