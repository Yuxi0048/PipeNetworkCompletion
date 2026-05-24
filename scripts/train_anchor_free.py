"""Train and evaluate the anchor-free road-edge prediction experiment."""

# Workstream: Codex + Claude merge

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipe_network_completion.anchor_free.config import load_anchor_free_config
from pipe_network_completion.anchor_free.model import torch_device_report
from pipe_network_completion.anchor_free.pipeline import run_anchor_free_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run anchor-free road-constrained utility topology prediction."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "anchor_free_isarc2024.yaml",
        help="Anchor-free YAML config.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run against the tiny in-memory synthetic fixture.",
    )
    parser.add_argument(
        "--model-type",
        choices=["gnn", "logistic_regression", "random_forest"],
        help="Override model.type from the config.",
    )
    parser.add_argument(
        "--decoder-type",
        choices=["threshold", "connected", "steiner", "sewer", "water"],
        help="Override decoder.type from the config.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override model.epochs for quick smoke tests.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Root directory for experiment outputs.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        help="Torch device for GNN runs. 'auto' uses CUDA only when PyTorch can see it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_anchor_free_config(args.config)
    if args.model_type:
        config["model"]["type"] = args.model_type
    if args.decoder_type:
        config["decoder"]["type"] = args.decoder_type
    if args.epochs is not None:
        config["model"]["epochs"] = args.epochs
    if args.device is not None:
        config["model"]["device"] = args.device

    device_report = torch_device_report()
    print(
        "PyTorch device check: "
        f"torch={device_report['torch_version']}, "
        f"cuda_available={device_report['cuda_available']}, "
        f"cuda_device_count={device_report['cuda_device_count']}"
    )
    if device_report["cuda_devices"]:
        print(f"CUDA devices: {', '.join(device_report['cuda_devices'])}")

    result = run_anchor_free_experiment(
        config,
        synthetic=args.synthetic,
        output_root=args.output_root,
    )
    print(f"Anchor-free outputs: {result.output_dir}")

    # Phase C of audit_followup_implementation_plan.md:
    # print held-out test metrics beside trivial baselines with a Δ column,
    # so a small lift over prevalence cannot be misread as a strong result.
    metrics = result.metrics.values
    prev = metrics.get("test_positive_prevalence")
    if isinstance(prev, float):
        print(f"test_positive_prevalence: {prev:.6f}")
    print(f"{'metric':>20s} {'model':>10s} {'all_pos':>10s} {'Δ':>10s}")
    for model_key, baseline_key in [
        ("test_roc_auc", "test_all_positive_roc_auc"),
        ("test_pr_auc", "test_all_positive_pr_auc"),
        ("test_f1", "test_all_positive_f1"),
        ("test_precision", "test_all_positive_precision"),
        ("test_recall", "test_all_positive_recall"),
        ("test_brier_score", "test_all_positive_brier_score"),
    ]:
        m = metrics.get(model_key)
        b = metrics.get(baseline_key)
        if not isinstance(m, float) or not isinstance(b, float):
            continue
        print(f"{model_key:>20s} {m:>10.4f} {b:>10.4f} {m - b:>+10.4f}")
    print("Decoded network metrics, all candidate edges:")
    for key in ["length_precision", "length_recall", "length_f1"]:
        value = result.metrics.values.get(key)
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
