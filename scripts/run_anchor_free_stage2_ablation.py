"""Stage 2 — Core ablation matrix for the anchor-free pipeline.

# Workstream: Claude

Implements the 7-variant ablation from
``docs/research_notes/audit_followup_implementation_plan.md`` §16 Stage 2,
itself derived from Codex's
``docs/research_notes/research_reward_upgrade_recommendation_codex.md``.

Variants:
    1. road-only RandomForest
    2. road + building-points RandomForest
    3. road + building-points + DEM RandomForest
    4. road-only GNN
    5. road + building-points GNN
    6. GNN without absolute road-node x/y (location-memorisation control)
    7. GNN with absolute road-node x/y (baseline pair for variant 6)

All variants share the ISARC-seeded buffer-invariant split from Stage 1
so per-edge metrics are comparable across rows. Writes
``ablation_stage2.csv`` plus per-variant ``metrics.json`` for review at
Checkpoint 2.

Usage:
    python scripts/run_anchor_free_stage2_ablation.py --epochs 100
    python scripts/run_anchor_free_stage2_ablation.py --synthetic --epochs 5
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the brisbane-script monkey-patches for the slow label loop and
# invalid-polygon repair, so this runner works on real data without
# touching Codex's modules.
from scripts.train_anchor_free_brisbane import apply_fast_patches  # noqa: E402


@dataclass(frozen=True)
class Variant:
    name: str
    model_type: str               # "random_forest" or "gnn"
    use_buildings: bool           # building polygons
    use_building_points: bool
    use_built_up: bool
    use_dem: bool
    include_node_coords: bool     # only meaningful for gnn variants


STAGE2_VARIANTS: tuple[Variant, ...] = (
    Variant("01_rf_road_only",            "random_forest", False, False, False, False, True),
    Variant("02_rf_road_bpoints",         "random_forest", False, True,  False, False, True),
    Variant("03_rf_road_bpoints_dem",     "random_forest", False, True,  False, True,  True),
    Variant("04_gnn_road_only",           "gnn",           False, False, False, False, True),
    Variant("05_gnn_road_bpoints",        "gnn",           False, True,  False, False, True),
    Variant("06_gnn_road_bpoints_noxy",   "gnn",           False, True,  False, False, False),
    Variant("07_gnn_road_bpoints_xy",     "gnn",           False, True,  False, False, True),
)


# Real Brisbane data defaults (same as train_anchor_free_brisbane.py).
DEFAULT_ROADS = "data/raw/gis/roads/Roads_ExportFeatures.shp"
DEFAULT_TRUTH = [
    "data/raw/gis/sewer/SewerGravityMa_ExportFeature1.shp",
    "data/raw/gis/sewer/SewerGravityMa_ExportFeature2.shp",
]
DEFAULT_BUILDINGS = "data/processed/context/study_area/building_areas_study_area.geojson"
DEFAULT_BUILDING_POINTS = "data/processed/context/study_area/building_points_study_area.geojson"
DEFAULT_BUILT_UP = "data/processed/context/study_area/build_up_areas_study_area.geojson"
DEFAULT_DEM = "data/processed/context/study_area/brisbane_dem_h_1sec_epsg28356.tif"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--epochs", type=int, default=100, help="GNN epochs.")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "cuda:0"]
    )
    p.add_argument("--target-crs", default="EPSG:28356")
    p.add_argument("--label-buffer-m", type=float, default=10.0)
    p.add_argument("--label-overlap-threshold", type=float, default=0.25)
    p.add_argument("--synthetic", action="store_true",
                   help="Use the synthetic fixture; skips real-data file reads.")
    p.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs")
    p.add_argument("--experiment-prefix", default="anchor_free_stage2",
                   help="Per-variant experiment_name will be '<prefix>_<variant.name>'.")
    p.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Subset of variant names to run (default: all 7).",
    )
    return p.parse_args()


def _build_variant_config(
    base: dict, variant: Variant, args: argparse.Namespace
) -> dict:
    cfg = copy.deepcopy(base)
    cfg["experiment_name"] = f"{args.experiment_prefix}_{variant.name}"
    cfg["mode"] = "anchor_free"
    cfg["seed"] = 42

    cfg["data"] = cfg.get("data", {})
    if not args.synthetic:
        cfg["data"]["roads_path"] = DEFAULT_ROADS
        cfg["data"]["utility_truth_path"] = DEFAULT_TRUTH
        cfg["data"]["buildings_path"] = DEFAULT_BUILDINGS if variant.use_buildings else ""
        cfg["data"]["building_points_path"] = (
            DEFAULT_BUILDING_POINTS if variant.use_building_points else ""
        )
        cfg["data"]["built_up_path"] = DEFAULT_BUILT_UP if variant.use_built_up else ""
        cfg["data"]["dem_path"] = DEFAULT_DEM if variant.use_dem else ""

    cfg["graph"] = cfg.get("graph", {})
    cfg["graph"]["target_crs"] = args.target_crs
    cfg["graph"]["road_class_columns"] = ["OVL2_CAT", "OVL_CAT"]
    cfg["graph"]["label_buffer_m"] = float(args.label_buffer_m)
    cfg["graph"]["label_overlap_threshold"] = float(args.label_overlap_threshold)
    cfg["graph"]["use_buildings"] = variant.use_buildings
    cfg["graph"]["use_building_points"] = variant.use_building_points
    cfg["graph"]["use_built_up"] = variant.use_built_up
    cfg["graph"]["use_dem"] = variant.use_dem

    cfg["model"] = cfg.get("model", {})
    cfg["model"]["type"] = variant.model_type
    cfg["model"]["device"] = args.device
    cfg["model"]["epochs"] = int(args.epochs)
    cfg["model"]["hidden_dim"] = int(args.hidden)
    cfg["model"]["num_layers"] = int(args.num_layers)
    cfg["model"]["include_node_coords"] = bool(variant.include_node_coords)

    cfg["decoder"] = cfg.get("decoder", {})
    cfg["decoder"]["type"] = "threshold"

    return cfg


def _summary_row(variant: Variant, result, runtime_sec: float) -> dict[str, Any]:
    m = result.metrics.values
    row: dict[str, Any] = {
        "variant": variant.name,
        "model_type": variant.model_type,
        "use_buildings": variant.use_buildings,
        "use_building_points": variant.use_building_points,
        "use_built_up": variant.use_built_up,
        "use_dem": variant.use_dem,
        "include_node_coords": variant.include_node_coords,
        "runtime_sec": runtime_sec,
        "n_features": int(result.features.features.shape[1]),
    }
    for key in (
        "test_positive_prevalence",
        "test_roc_auc",
        "test_pr_auc",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_brier_score",
        "test_all_positive_f1",
        "test_all_positive_brier_score",
        "train_roc_auc",
        "train_f1",
        "length_precision",
        "length_recall",
        "length_f1",
        "connected_component_count",
        "predicted_total_length",
        "true_total_length",
    ):
        row[key] = m.get(key)
    # Lift over the all-positive baseline — Codex's "is this a real win?" column.
    f1 = row.get("test_f1")
    f1_base = row.get("test_all_positive_f1")
    if isinstance(f1, float) and isinstance(f1_base, float):
        row["test_f1_lift_over_all_pos"] = f1 - f1_base
    return row


def main() -> int:
    args = parse_args()
    apply_fast_patches()

    from pipe_network_completion.anchor_free.config import (
        load_anchor_free_config,
    )
    from pipe_network_completion.anchor_free.pipeline import (
        run_anchor_free_experiment,
    )

    base_config = load_anchor_free_config(
        REPO_ROOT / "configs" / "anchor_free_isarc2024.yaml"
    )
    variants = STAGE2_VARIANTS
    if args.variants:
        wanted = set(args.variants)
        variants = tuple(v for v in STAGE2_VARIANTS if v.name in wanted)
        missing = wanted - {v.name for v in STAGE2_VARIANTS}
        if missing:
            raise SystemExit(f"Unknown variant names: {sorted(missing)}")

    print(f"==> Stage 2 ablation — {len(variants)} variant(s)")
    for v in variants:
        print(
            f"    {v.name}: model={v.model_type} buildings={v.use_buildings} "
            f"building_points={v.use_building_points} built_up={v.use_built_up} "
            f"dem={v.use_dem} coords={v.include_node_coords}"
        )
    print()

    rows: list[dict[str, Any]] = []
    for variant in variants:
        cfg = _build_variant_config(base_config, variant, args)
        print(f"==> running {variant.name} ({variant.model_type}) ...")
        t0 = time.time()
        try:
            result = run_anchor_free_experiment(
                cfg, synthetic=args.synthetic, output_root=args.output_root
            )
        except Exception as exc:  # noqa: BLE001 — log + skip across variants
            print(f"    [error] {variant.name} failed: {exc}")
            rows.append(
                {"variant": variant.name, "error": str(exc),
                 "model_type": variant.model_type}
            )
            continue
        runtime = time.time() - t0
        row = _summary_row(variant, result, runtime)
        rows.append(row)
        lift = row.get("test_f1_lift_over_all_pos")
        if isinstance(lift, float):
            print(
                f"    done in {runtime:.1f}s | test_f1={row.get('test_f1'):.4f} "
                f"(Δ over all-positive: {lift:+.4f})"
            )

    df = pd.DataFrame(rows)
    out_dir = Path(args.output_root) / args.experiment_prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ablation_stage2.csv"
    df.to_csv(csv_path, index=False)
    json_path = out_dir / "ablation_stage2.json"
    json_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print()
    print(f"==> Wrote {csv_path}")
    print(f"==> Wrote {json_path}")

    # Print the comparison-friendly subset.
    cols = [
        "variant", "model_type", "include_node_coords",
        "test_positive_prevalence", "test_f1", "test_all_positive_f1",
        "test_f1_lift_over_all_pos", "test_roc_auc", "test_pr_auc",
        "length_f1", "runtime_sec",
    ]
    print()
    print("Summary (post-split lift over all-positive baseline):")
    try:
        print(df[cols].to_string(index=False))
    except KeyError:
        print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
