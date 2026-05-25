"""Train the original ISARC-style anchor-based GNN under node-feature ablations."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import LinkNeighborLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipe_network_completion.evaluation import BinaryMetrics, evaluate_loader
from pipe_network_completion.low_quality import apply_location_only_ablation
from pipe_network_completion.model import build_model_from_data
from pipe_network_completion.paths import GRAPH_DATA_DIR, REPO_ROOT


EDGE_TYPE = ("MH", "link", "MH")

ARCHITECTURES: dict[str, tuple[tuple[str, ...], bool]] = {
    "04": (("sage", "sage", "sage", "sage"), False),
    "22": (("gat", "gat", "sage", "sage"), False),
    "1111": (("gat", "sage", "gat", "sage"), False),
    "1212": (("gat", "sage", "sage", "gat", "sage", "sage"), False),
    "1212-skip": (("gat", "sage", "sage", "gat", "sage", "sage"), True),
}


def seed_everything(seed: int, device: str) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return requested


def load_split(split: str, *, node_feature_mode: str, zero_edge_attrs: bool):
    path = GRAPH_DATA_DIR / f"{split}_data.pkl"
    with path.open("rb") as handle:
        data = pickle.load(handle)
    report = None
    if node_feature_mode == "location-only":
        data, report = apply_location_only_ablation(
            data,
            zero_edge_attrs=zero_edge_attrs,
        )
    return data, report


def make_loader(data, *, batch_size: int, shuffle: bool) -> LinkNeighborLoader:
    edge_label_index = data[EDGE_TYPE].edge_label_index
    edge_label = data[EDGE_TYPE].edge_label
    return LinkNeighborLoader(
        data=data,
        num_neighbors=[20, 10],
        edge_label_index=(EDGE_TYPE, edge_label_index),
        edge_label=edge_label,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: LinkNeighborLoader,
    *,
    device: str,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    max_batches: int | None,
) -> float:
    model.train()
    losses: list[float] = []
    iterator = tqdm(loader, desc="Training", leave=False)
    for batch_index, batch in enumerate(iterator):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        labels = batch[EDGE_TYPE].edge_label.float()
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        iterator.set_postfix(loss=f"{loss_value:.4f}")
    if not losses:
        raise ValueError("No training batches were processed.")
    return float(np.mean(losses))


def metric_row(split: str, epoch: int, metrics: BinaryMetrics) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {"split": split, "epoch": int(epoch)}
    row.update(asdict(metrics))
    return row


def write_metrics_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain the original anchor-based ISARC GNN with full or "
            "location-only node features."
        )
    )
    parser.add_argument(
        "--node-feature-mode",
        choices=["full", "location-only"],
        default="location-only",
        help="Default reproduces the low-quality setting: only node locations remain.",
    )
    parser.add_argument(
        "--zero-edge-attrs",
        action="store_true",
        help="Also remove edge attributes for a stricter low-quality-data ablation.",
    )
    parser.add_argument("--architecture", choices=sorted(ARCHITECTURES), default="1212")
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Use auto to prefer CUDA when available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "isarc_low_quality_location_only",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    seed_everything(args.seed, device)
    if device == "cuda":
        print(f"CUDA available: True ({torch.cuda.get_device_name(0)})")
    else:
        print("CUDA available: False or not selected; training on CPU.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_data, ablation_report = load_split(
        "train",
        node_feature_mode=args.node_feature_mode,
        zero_edge_attrs=args.zero_edge_attrs,
    )
    val_data, _ = load_split(
        "val",
        node_feature_mode=args.node_feature_mode,
        zero_edge_attrs=args.zero_edge_attrs,
    )
    test_data, _ = load_split(
        "test",
        node_feature_mode=args.node_feature_mode,
        zero_edge_attrs=args.zero_edge_attrs,
    )

    train_loader = make_loader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = make_loader(test_data, batch_size=args.batch_size, shuffle=False)

    layers, skip = ARCHITECTURES[args.architecture]
    model = build_model_from_data(
        data=train_data,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        device=device,
        layers=layers,
        skip=skip,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_auc = -float("inf")
    best_epoch = 0
    best_checkpoint = args.output_dir / "best_model.pt"
    rows: list[dict[str, float | int | str]] = []
    losses: list[dict[str, float | int]] = []
    started = time.time()

    print(
        "Training ISARC ablation: "
        f"mode={args.node_feature_mode}, zero_edge_attrs={args.zero_edge_attrs}, "
        f"architecture={args.architecture}, epochs={args.epochs}"
    )
    if ablation_report is not None:
        print(f"Zeroed node attribute dims: {ablation_report.zeroed_node_attribute_dims}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
            max_batches=args.max_train_batches,
        )
        losses.append({"epoch": epoch, "train_loss": train_loss})
        should_eval = epoch == args.epochs or epoch % max(1, args.eval_every) == 0
        if not should_eval:
            print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f}")
            continue

        val_metrics = evaluate_loader(
            model=model,
            loader=val_loader,
            device=device,
            threshold=args.threshold,
            max_batches=args.max_eval_batches,
        )
        rows.append(metric_row("val", epoch, val_metrics))
        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.6f}, "
            f"val_auc={val_metrics.auc:.6f}, val_f1={val_metrics.f1:.6f}"
        )
        if np.isfinite(val_metrics.auc) and val_metrics.auc > best_val_auc:
            best_val_auc = val_metrics.auc
            best_epoch = epoch
            torch.save(model.state_dict(), best_checkpoint)

    if best_checkpoint.exists():
        state_dict = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(state_dict)

    test_metrics = evaluate_loader(
        model=model,
        loader=test_loader,
        device=device,
        threshold=args.threshold,
        max_batches=args.max_eval_batches,
    )
    rows.append(metric_row("test", best_epoch or args.epochs, test_metrics))

    write_metrics_csv(args.output_dir / "metrics.csv", rows)
    write_metrics_csv(args.output_dir / "training_losses.csv", losses)

    summary = {
        "node_feature_mode": args.node_feature_mode,
        "zero_edge_attrs": bool(args.zero_edge_attrs),
        "architecture": args.architecture,
        "layers": list(layers),
        "skip": bool(skip),
        "hidden_channels": int(args.hidden_channels),
        "dropout": float(args.dropout),
        "epochs": int(args.epochs),
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_val_auc),
        "test_metrics": asdict(test_metrics),
        "device": device,
        "runtime_seconds": float(time.time() - started),
        "ablation": asdict(ablation_report) if ablation_report is not None else None,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "config_resolved.json").write_text(
        json.dumps(vars(args) | {"device_resolved": device}, default=str, indent=2),
        encoding="utf-8",
    )

    print("Final test metrics")
    print(f"  AUC:       {test_metrics.auc:.6f}")
    print(f"  F1:        {test_metrics.f1:.6f}")
    print(f"  Precision: {test_metrics.precision:.6f}")
    print(f"  Recall:    {test_metrics.recall:.6f}")
    print(f"  Accuracy:  {test_metrics.accuracy:.6f}")
    print(f"  MCC:       {test_metrics.mcc:.6f}")
    print(f"Wrote outputs to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
