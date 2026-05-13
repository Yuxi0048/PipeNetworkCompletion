"""Evaluate a saved checkpoint against prepared graph data."""

from __future__ import annotations

import argparse
import csv
import pickle
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import LinkNeighborLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipe_network_completion.paths import CHECKPOINT_DIR, GRAPH_DATA_DIR, METRICS_DIR
from pipe_network_completion.evaluation import BinaryMetrics, evaluate_loader
from pipe_network_completion.model import (
    build_model_from_data,
    infer_architecture_from_state_dict,
    load_state_dict,
)

EDGE_TYPE = ("MH", "link", "MH")


def infer_hyperparams_from_checkpoint(path: Path) -> tuple[int, float]:
    match = re.search(r"hiddensize_(\d+)_drop_(\d+)", path.name)
    if not match:
        raise ValueError(
            "Could not infer hidden size/dropout from checkpoint name. "
            "Pass --hidden-channels and --dropout."
        )
    hidden_channels = int(match.group(1))
    dropout_code = match.group(2)
    dropout = int(dropout_code) / 10.0
    return hidden_channels, dropout


def load_expected_metrics(
    metrics_csv: Path,
    hidden_channels: int,
    dropout: float,
    phase: str,
) -> BinaryMetrics | None:
    if not metrics_csv.exists():
        return None
    with metrics_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                int(row["Model Size"]) == hidden_channels
                and abs(float(row["Dropout"]) - dropout) < 1e-12
                and row["Phase"].lower() == phase.lower()
            ):
                return BinaryMetrics(
                    auc=float(row["AUC"]),
                    f1=float(row["F1"]),
                    precision=float(row["Precision"]),
                    recall=float(row["Recall"]),
                    accuracy=float(row["Accuracy"]),
                    mcc=float(row["MCC"]),
                )
    return None


def print_metrics(title: str, metrics: BinaryMetrics) -> None:
    print(title)
    print(f"  AUC:       {metrics.auc:.6f}")
    print(f"  F1:        {metrics.f1:.6f}")
    print(f"  Precision: {metrics.precision:.6f}")
    print(f"  Recall:    {metrics.recall:.6f}")
    print(f"  Accuracy:  {metrics.accuracy:.6f}")
    print(f"  MCC:       {metrics.mcc:.6f}")


def max_metric_delta(left: BinaryMetrics, right: BinaryMetrics) -> float:
    return max(
        abs(getattr(left, field) - getattr(right, field))
        for field in ["auc", "f1", "precision", "recall", "accuracy", "mcc"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved model against the prepared data artifacts."
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Prepared split to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_DIR / "model1212_hiddensize_128_drop_00.pt",
        help="Model checkpoint to load.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=METRICS_DIR / "model_metrics1212.csv",
        help="Metrics CSV to compare against.",
    )
    parser.add_argument("--hidden-channels", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument(
        "--architecture",
        choices=["auto", "04", "22", "1111", "1212", "1212-skip"],
        default="auto",
        help="Model architecture variant. Auto reads the checkpoint state dict.",
    )
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-batches",
        type=int,
        help="Smoke-test mode; disables metrics comparison because it is partial.",
    )
    parser.add_argument("--tolerance", type=float, default=5e-3)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if args.hidden_channels is None or args.dropout is None:
        inferred_hidden, inferred_dropout = infer_hyperparams_from_checkpoint(
            args.checkpoint
        )
        hidden_channels = args.hidden_channels or inferred_hidden
        dropout = args.dropout if args.dropout is not None else inferred_dropout
    else:
        hidden_channels = args.hidden_channels
        dropout = args.dropout

    data_path = GRAPH_DATA_DIR / f"{args.split}_data.pkl"
    with data_path.open("rb") as handle:
        data = pickle.load(handle)

    edge_label_index = data[EDGE_TYPE].edge_label_index
    edge_label = data[EDGE_TYPE].edge_label
    loader = LinkNeighborLoader(
        data=data,
        num_neighbors=[20, 10],
        edge_label_index=(EDGE_TYPE, edge_label_index),
        edge_label=edge_label,
        batch_size=args.batch_size,
        shuffle=False,
    )

    state_dict = load_state_dict(args.checkpoint, map_location=args.device)
    if args.architecture == "auto":
        layers, skip = infer_architecture_from_state_dict(state_dict)
    else:
        architecture_map = {
            "04": (("sage", "sage", "sage", "sage"), False),
            "22": (("gat", "gat", "sage", "sage"), False),
            "1111": (("gat", "sage", "gat", "sage"), False),
            "1212": (("gat", "sage", "sage", "gat", "sage", "sage"), False),
            "1212-skip": (("gat", "sage", "sage", "gat", "sage", "sage"), True),
        }
        layers, skip = architecture_map[args.architecture]

    print(f"Architecture: layers={layers}, skip={skip}")
    model = build_model_from_data(
        data=data,
        hidden_channels=hidden_channels,
        dropout=dropout,
        device=args.device,
        layers=layers,
        skip=skip,
    )
    model.load_state_dict(state_dict)

    observed = evaluate_loader(
        model=model,
        loader=loader,
        device=args.device,
        threshold=args.threshold,
        max_batches=args.max_batches,
    )
    print_metrics("Observed metrics", observed)

    if args.max_batches is not None:
        print("Partial smoke test completed; metrics comparison skipped.")
        return 0

    phase = {"train": "Training", "val": "Validation", "test": "Testing"}[args.split]
    expected = load_expected_metrics(
        metrics_csv=args.metrics,
        hidden_channels=hidden_channels,
        dropout=dropout,
        phase=phase,
    )
    if expected is None:
        print("No matching expected metrics row found; evaluation completed.")
        return 0

    print_metrics("Expected metrics", expected)
    delta = max_metric_delta(observed, expected)
    print(f"Max metric delta: {delta:.6f}")
    if delta > args.tolerance:
        print(
            "Replication check failed. The original notebook used sampled "
            "neighborhoods, so exact metrics can vary unless the full software "
            "stack and random state match."
        )
        return 2

    print("Replication check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
