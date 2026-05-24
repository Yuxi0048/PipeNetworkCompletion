"""Run anchor-free ablation variants and write a comparison table."""

# Workstream: Codex + Claude merge

from __future__ import annotations

import argparse
import csv
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipe_network_completion.anchor_free.config import load_anchor_free_config
from pipe_network_completion.anchor_free.model import torch_device_report
from pipe_network_completion.anchor_free.pipeline import run_anchor_free_experiment
from pipe_network_completion.paths import METRICS_DIR
from scripts.train_anchor_free_brisbane import apply_fast_patches


CONTEXT_FLAG_KEYS = (
    "use_buildings",
    "use_building_points",
    "use_built_up",
    "use_dem",
)


def _anchor_based_reference(metrics_csv: Path) -> dict[str, Any]:
    row = {
        "variant": "Reference: original ISARC-style anchor-based GNN",
        "status": "not_run",
        "comparison_scope": "reference_different_task_definition",
    }
    if not metrics_csv.exists():
        return row
    with metrics_csv.open(newline="") as handle:
        for candidate in csv.DictReader(handle):
            phase = str(candidate.get("Phase", "")).lower()
            if phase == "testing":
                row.update(
                    {
                        "status": "reported_from_results_csv",
                        "roc_auc": candidate.get("AUC"),
                        "f1": candidate.get("F1"),
                        "precision": candidate.get("Precision"),
                        "recall": candidate.get("Recall"),
                    }
                )
                break
    return row


def _slugify(value: str) -> str:
    return (
        value.lower()
        .replace("+", "plus")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def _variant_config(
    base: dict[str, Any],
    *,
    variant_name: str,
    model_type: str,
    context_flags: dict[str, bool],
    decoder_type: str,
) -> dict[str, Any]:
    config = deepcopy(base)
    base_name = str(base.get("experiment_name", "anchor_free_isarc2024"))
    config["experiment_name"] = f"{base_name}_{_slugify(variant_name)}"
    config["model"]["type"] = model_type
    for key in CONTEXT_FLAG_KEYS:
        config["graph"][key] = bool(context_flags.get(key, False))
    config["decoder"]["type"] = decoder_type
    return config


def _context_flags(
    *,
    use_buildings: bool,
    use_building_points: bool,
    use_built_up: bool,
    use_dem: bool,
) -> dict[str, bool]:
    return {
        "use_buildings": bool(use_buildings),
        "use_building_points": bool(use_building_points),
        "use_built_up": bool(use_built_up),
        "use_dem": bool(use_dem),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare anchor-free baseline/GNN variants."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "anchor_free_isarc2024.yaml",
    )
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--epochs", type=int, help="Override GNN epochs for all GNN variants.")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        help="Torch device for GNN variants. 'auto' uses CUDA only when PyTorch can see it.",
    )
    parser.add_argument(
        "--anchor-metrics",
        type=Path,
        default=METRICS_DIR / "model_metrics1212.csv",
        help="Optional recorded metrics CSV for the original anchor-based model.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    apply_fast_patches()
    base_config = load_anchor_free_config(args.config)
    if args.epochs is not None:
        base_config["model"]["epochs"] = args.epochs
    if args.device is not None:
        base_config["model"]["device"] = args.device

    device_report = torch_device_report()
    print(
        "PyTorch device check: "
        f"torch={device_report['torch_version']}, "
        f"cuda_available={device_report['cuda_available']}, "
        f"cuda_device_count={device_report['cuda_device_count']}"
    )
    if device_report["cuda_devices"]:
        print(f"CUDA devices: {', '.join(device_report['cuda_devices'])}")

    road_only = _context_flags(
        use_buildings=False,
        use_building_points=False,
        use_built_up=False,
        use_dem=False,
    )
    road_building = _context_flags(
        use_buildings=True,
        use_building_points=True,
        use_built_up=False,
        use_dem=False,
    )
    road_context = _context_flags(
        use_buildings=True,
        use_building_points=True,
        use_built_up=True,
        use_dem=True,
    )
    variants = [
        ("Anchor-free road-only Random Forest", "random_forest", road_only, "threshold"),
        ("Anchor-free road+building Random Forest", "random_forest", road_building, "threshold"),
        ("Anchor-free road+context Random Forest", "random_forest", road_context, "threshold"),
        ("Anchor-free road-only GNN", "gnn", road_only, "threshold"),
        ("Anchor-free road+building GNN", "gnn", road_building, "threshold"),
        ("Anchor-free road+context GNN + connected decoder", "gnn", road_context, "connected"),
    ]

    rows: list[dict[str, Any]] = [_anchor_based_reference(args.anchor_metrics)]
    for variant_name, model_type, context_flags, decoder_type in variants:
        config = _variant_config(
            base_config,
            variant_name=variant_name,
            model_type=model_type,
            context_flags=context_flags,
            decoder_type=decoder_type,
        )
        result = run_anchor_free_experiment(
            config,
            synthetic=args.synthetic,
            output_root=args.output_root,
        )
        row = {"variant": variant_name, "status": "completed"}
        row.update(context_flags)
        row["model_type"] = model_type
        row["decoder_type"] = decoder_type
        row.update(result.metrics.values)
        rows.append(row)

    output_root = args.output_root if args.output_root is not None else REPO_ROOT / "outputs"
    output_dir = Path(output_root) / str(base_config.get("experiment_name", "anchor_free_isarc2024"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ablation_results.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Ablation results: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
