"""Train a U-Net-support skeleton-graph GNN hybrid.

Workstream: Codex

Pipeline:
1. Run the trained U-Net over each raster tile.
2. Mosaic each AOI and skeletonize the predicted support field.
3. Build a candidate graph from skeleton pixels.
4. Train a GNN edge classifier using U-Net/context features sampled on the graph.

No absolute coordinate or location encoder is used as a model input. Coordinates
are written only to the output GeoJSON for QGIS inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_GDAL_DATA = Path(sys.prefix) / "Library" / "share" / "gdal"
if "GDAL_DATA" not in os.environ and DEFAULT_GDAL_DATA.exists():
    os.environ["GDAL_DATA"] = str(DEFAULT_GDAL_DATA)

from pipe_network_completion.anchor_free.model import (  # noqa: E402
    RoadEdgeGNN,
    resolve_torch_device,
    torch_device_report,
)
from pipe_network_completion.anchor_free.unet_segmentation import load_tile_index, make_unet_model  # noqa: E402
from pipe_network_completion.anchor_free.unet_skeleton_hybrid import (  # noqa: E402
    CODE_TO_SPLIT,
    SPLIT_TO_CODE,
    build_aoi_mosaic,
    build_skeleton_graph_from_mosaic,
    combine_skeleton_graph_parts,
    predict_tile_cnn_outputs,
    predict_tile_probabilities,
    standardize_graph_features,
)


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(resolved)


def _split_indices(edge_split_code: torch.Tensor, split: str) -> np.ndarray:
    code = SPLIT_TO_CODE[split]
    return torch.nonzero(edge_split_code.cpu() == int(code), as_tuple=False).reshape(-1).numpy()


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, *, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= float(threshold)).astype(int)
    out = {
        "threshold": float(threshold),
        "n_edges": int(y_true.size),
        "positive_fraction": float(y_true.mean()) if y_true.size else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)) if y_true.size else 0.0,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)) if y_true.size else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if y_true.size else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if np.unique(y_true).size == 2 else float("nan"),
        "brier": float(brier_score_loss(y_true, y_prob)) if y_true.size else float("nan"),
    }
    if np.unique(y_true).size == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def _threshold_rows(y_true: np.ndarray, y_prob: np.ndarray, *, split: str, thresholds: list[float]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    positive_fraction = float(np.asarray(y_true, dtype=int).mean()) if len(y_true) else 0.0
    rows.append(
        {
            "split": split,
            "threshold": "all_positive",
            "n_edges": int(len(y_true)),
            "positive_fraction": positive_fraction,
            "precision": positive_fraction,
            "recall": 1.0 if len(y_true) else 0.0,
            "f1": (2.0 * positive_fraction / (1.0 + positive_fraction)) if positive_fraction > 0 else 0.0,
            "balanced_accuracy": 0.5 if len(y_true) else float("nan"),
            "brier": float("nan"),
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
        }
    )
    for threshold in thresholds:
        row = _binary_metrics(y_true, y_prob, threshold=float(threshold))
        row["split"] = split
        rows.append(row)
    return rows


def _train_gnn(
    data,
    *,
    train_index: np.ndarray,
    val_index: np.ndarray,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    lr: float,
    epochs: int,
    device: torch.device,
) -> tuple[RoadEdgeGNN, list[dict[str, float]], np.ndarray]:
    model = RoadEdgeGNN(
        node_input_dim=int(data.x.shape[1]),
        edge_input_dim=int(data.edge_label_attr.shape[1]),
        hidden_dim=int(hidden_dim),
        num_layers=int(num_layers),
        dropout=float(dropout),
    ).to(device)
    data = data.to(device)
    train_tensor = torch.tensor(train_index, dtype=torch.long, device=device)
    val_tensor = torch.tensor(val_index, dtype=torch.long, device=device)
    positives = torch.clamp(data.edge_label[train_tensor].sum(), min=1.0)
    negatives = torch.clamp(torch.tensor(float(len(train_index)), device=device) - data.edge_label[train_tensor].sum(), min=1.0)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=negatives / positives)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))
    history: list[dict[str, float]] = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(data)
        loss = criterion(logits[train_tensor], data.edge_label[train_tensor])
        loss.backward()
        optimizer.step()

        row = {"epoch": float(epoch), "train_loss": float(loss.detach().cpu())}
        if len(val_index):
            model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(model(data)[val_tensor]).detach().cpu().numpy()
            y_val = data.edge_label[val_tensor].detach().cpu().numpy()
            row.update({f"val_{key}": value for key, value in _binary_metrics(y_val, probs, threshold=0.5).items()})
        history.append(row)
        print(
            f"epoch={epoch:03d} loss={row['train_loss']:.4f} "
            f"val_f1={row.get('val_f1', float('nan')):.4f}"
        )

    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(data)).detach().cpu().numpy()
    return model.cpu(), history, probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-index", type=Path, default=REPO_ROOT / "data" / "processed" / "unet_tiles" / "aoi24_grid128_cadastral_perf" / "tiles_index.csv")
    parser.add_argument("--unet-checkpoint", type=Path, default=REPO_ROOT / "outputs" / "unet_segmentation_aoi24_grid128_perf" / "unet_checkpoint.pt")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "unet_skeleton_hybrid_gnn_aoi24_grid128")
    parser.add_argument("--candidate-threshold", type=float, default=0.2)
    parser.add_argument("--min-component-pixels", type=int, default=8)
    parser.add_argument("--support-recall-tolerance-m", type=float, default=10.0)
    parser.add_argument(
        "--include-cnn-features",
        action="store_true",
        help="Sample full-resolution U-Net decoder CNN features onto skeleton graph nodes/edges.",
    )
    parser.add_argument("--unet-batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.time()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_torch_device(str(args.device))
    print(json.dumps({**torch_device_report(), "selected_device": str(device)}, indent=2))

    tile_index_path = args.tile_index if args.tile_index.is_absolute() else REPO_ROOT / args.tile_index
    unet_checkpoint_path = args.unet_checkpoint if args.unet_checkpoint.is_absolute() else REPO_ROOT / args.unet_checkpoint
    tile_index = load_tile_index(tile_index_path)
    checkpoint = torch.load(unet_checkpoint_path, map_location="cpu")
    unet = make_unet_model(
        in_channels=len(checkpoint["channel_names"]),
        encoder_name=checkpoint["encoder_name"],
        encoder_weights=checkpoint["encoder_weights"],
        classes=1,
    )
    unet.load_state_dict(checkpoint["model_state_dict"])
    unet.to(device)

    parts = []
    for aoi_id, aoi_index in tile_index.groupby("aoi_id", sort=True):
        aoi_index = aoi_index.sort_values(["grid_row", "grid_col"]).reset_index(drop=True)
        if args.include_cnn_features:
            probs, cnn_features, cnn_feature_names = predict_tile_cnn_outputs(
                unet,
                aoi_index,
                mean=checkpoint["mean"],
                std=checkpoint["std"],
                device=device,
                batch_size=int(args.unet_batch_size),
            )
            mosaic = build_aoi_mosaic(
                aoi_index,
                probs,
                cnn_features=cnn_features,
                cnn_feature_names=cnn_feature_names,
            )
        else:
            probs = predict_tile_probabilities(
                unet,
                aoi_index,
                mean=checkpoint["mean"],
                std=checkpoint["std"],
                device=device,
                batch_size=int(args.unet_batch_size),
            )
            mosaic = build_aoi_mosaic(aoi_index, probs)
        part = build_skeleton_graph_from_mosaic(
            mosaic,
            candidate_threshold=float(args.candidate_threshold),
            min_component_pixels=int(args.min_component_pixels),
            support_recall_tolerance_m=float(args.support_recall_tolerance_m),
            include_cnn_features=bool(args.include_cnn_features),
        )
        parts.append(part)
        print(
            f"{aoi_id} ({part.split}): nodes={part.support_metrics['n_skeleton_nodes']} "
            f"edges={part.support_metrics['n_skeleton_edges']} "
            f"candidate_recall={part.support_metrics.get('candidate_support_recall', float('nan')):.3f} "
            f"skeleton_recall={part.support_metrics.get('skeleton_truth_recall_within_tolerance', float('nan')):.3f}"
        )

    data, edge_table = combine_skeleton_graph_parts(parts)
    data, standardization = standardize_graph_features(data)
    train_index = _split_indices(data.edge_split_code, "train")
    val_index = _split_indices(data.edge_split_code, "val")
    test_index = _split_indices(data.edge_split_code, "test")

    print(
        f"combined graph: nodes={int(data.x.shape[0])} "
        f"message_edges={int(data.edge_index.shape[1])} "
        f"candidate_edges={int(data.edge_label.shape[0])} "
        f"train/val/test={len(train_index)}/{len(val_index)}/{len(test_index)}"
    )

    model, history, probabilities = _train_gnn(
        data,
        train_index=train_index,
        val_index=val_index,
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        lr=float(args.lr),
        epochs=int(args.epochs),
        device=device,
    )

    labels = data.edge_label.cpu().numpy()
    split_codes = data.edge_split_code.cpu().numpy()
    threshold_rows = []
    split_metrics = {}
    for code, split in CODE_TO_SPLIT.items():
        mask = split_codes == int(code)
        y_split = labels[mask]
        p_split = probabilities[mask]
        split_metrics[split] = _binary_metrics(y_split, p_split, threshold=0.5)
        threshold_rows.extend(_threshold_rows(y_split, p_split, split=split, thresholds=[float(v) for v in args.thresholds]))

    threshold_table = pd.DataFrame(threshold_rows)
    threshold_table.to_csv(output_dir / "threshold_sweep.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    support_metrics = pd.DataFrame([part.support_metrics for part in parts])
    support_metrics.to_csv(output_dir / "support_metrics_by_aoi.csv", index=False)
    support_summary = (
        support_metrics.groupby("split")
        .agg(
            n_aois=("aoi_id", "count"),
            n_skeleton_nodes=("n_skeleton_nodes", "sum"),
            n_skeleton_edges=("n_skeleton_edges", "sum"),
            truth_positive_pixel_fraction=("truth_positive_pixel_fraction", "mean"),
            candidate_support_pixel_fraction=("candidate_support_pixel_fraction", "mean"),
            candidate_support_recall=("candidate_support_recall", "mean"),
            candidate_support_precision=("candidate_support_precision", "mean"),
            candidate_support_f1=("candidate_support_f1", "mean"),
            skeleton_truth_recall_within_tolerance=("skeleton_truth_recall_within_tolerance", "mean"),
            skeleton_node_precision_inside_truth=("skeleton_node_precision_inside_truth", "mean"),
        )
        .reset_index()
    )
    support_summary.to_csv(output_dir / "support_metrics_by_split.csv", index=False)

    edge_table = edge_table.copy()
    edge_table["gnn_probability"] = probabilities
    edge_table["gnn_pred_05"] = (probabilities >= 0.5).astype(int)
    best_val = (
        threshold_table[(threshold_table["split"] == "val") & (threshold_table["threshold"] != "all_positive")]
        .sort_values("f1", ascending=False)
        .head(1)
    )
    best_val_threshold = float(best_val.iloc[0]["threshold"]) if not best_val.empty else 0.5
    edge_table["gnn_pred_best_val"] = (probabilities >= best_val_threshold).astype(int)
    edge_table.to_file(output_dir / "edge_predictions.geojson", driver="GeoJSON")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "node_feature_names": parts[0].node_feature_names,
            "edge_feature_names": parts[0].edge_feature_names,
            "standardization": standardization,
            "candidate_threshold": float(args.candidate_threshold),
            "best_val_threshold": best_val_threshold,
        },
        output_dir / "hybrid_gnn_checkpoint.pt",
    )

    metrics = {
        "workstream": "Codex",
        "description": "U-Net support field skeletonized into a candidate graph, followed by GNN edge classification.",
        "uses_absolute_location_channels": False,
        "tile_index": _relative(tile_index_path),
        "unet_checkpoint": _relative(unet_checkpoint_path),
        "output_dir": _relative(output_dir),
        "device": {**torch_device_report(), "selected_device": str(device)},
        "candidate_threshold": float(args.candidate_threshold),
        "min_component_pixels": int(args.min_component_pixels),
        "support_recall_tolerance_m": float(args.support_recall_tolerance_m),
        "include_cnn_features": bool(args.include_cnn_features),
        "best_val_threshold": best_val_threshold,
        "n_aois": int(len(parts)),
        "n_nodes": int(data.x.shape[0]),
        "n_candidate_edges": int(data.edge_label.shape[0]),
        "n_train_edges": int(len(train_index)),
        "n_val_edges": int(len(val_index)),
        "n_test_edges": int(len(test_index)),
        "node_feature_names": parts[0].node_feature_names,
        "edge_feature_names": parts[0].edge_feature_names,
        "metrics_at_0_5": split_metrics,
        "support_summary_by_split": support_summary.to_dict(orient="records"),
        "runtime_seconds": float(time.time() - start),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"best_val_threshold={best_val_threshold:.3f}")
    print(json.dumps({"val": split_metrics["val"], "test": split_metrics["test"]}, indent=2))
    print(f"wrote {_relative(output_dir / 'metrics.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
