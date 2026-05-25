"""Plot U-Net / CNN-GNN comparison figures.

Workstream: Codex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


MODEL_FILES = [
    {
        "model": "U-Net 14ch",
        "task": "pixel",
        "path": "outputs/unet_segmentation_aoi112_watercourses/threshold_sweep.csv",
        "color": "#2f6f9f",
    },
    {
        "model": "U-Net 21ch soft",
        "task": "pixel",
        "path": "outputs/unet_segmentation_aoi112_watercourses_soft30/threshold_sweep.csv",
        "color": "#7aa6c2",
    },
    {
        "model": "CNN-GNN 14ch",
        "task": "graph edge",
        "path": "outputs/cnn_gnn_hybrid_aoi112_watercourses_thr02/threshold_sweep.csv",
        "color": "#9a5b1f",
    },
    {
        "model": "CNN-GNN 21ch soft",
        "task": "graph edge",
        "path": "outputs/cnn_gnn_hybrid_aoi112_watercourses_soft30_thr02/threshold_sweep.csv",
        "color": "#d18f35",
    },
]


def _read_threshold_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    table["threshold_num"] = pd.to_numeric(table["threshold"], errors="coerce")
    return table


def _best_val_test_rows() -> pd.DataFrame:
    rows = []
    for spec in MODEL_FILES:
        table = _read_threshold_table(REPO_ROOT / spec["path"])
        valid = table[(table["split"] == "val") & table["threshold_num"].notna()].copy()
        best = valid.sort_values("f1", ascending=False).iloc[0]
        test = table[
            (table["split"] == "test")
            & (table["threshold_num"].round(6) == round(float(best["threshold_num"]), 6))
        ].iloc[0]
        rows.append(
            {
                "model": spec["model"],
                "task": spec["task"],
                "color": spec["color"],
                "threshold": float(best["threshold_num"]),
                "val_f1": float(best["f1"]),
                "test_precision": float(test["precision"]),
                "test_recall": float(test["recall"]),
                "test_f1": float(test["f1"]),
                "test_pr_auc": float(test["pr_auc"]),
                "test_roc_auc": float(test["roc_auc"]),
            }
        )
    return pd.DataFrame(rows)


def plot(output: Path) -> None:
    summary = _best_val_test_rows()
    fig = plt.figure(figsize=(13.5, 8.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0])
    ax_metrics = fig.add_subplot(gs[0, :])
    ax_f1 = fig.add_subplot(gs[1, 0])
    ax_pr = fig.add_subplot(gs[1, 1])

    metrics = ["test_precision", "test_recall", "test_f1", "test_pr_auc"]
    metric_labels = ["Precision", "Recall", "F1", "PR AUC"]
    x = range(len(summary))
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    palette = ["#5b6f82", "#7aa457", "#c96e4b", "#7f63a8"]
    for metric, label, offset, color in zip(metrics, metric_labels, offsets, palette):
        values = summary[metric].to_numpy(dtype=float)
        ax_metrics.bar([i + offset for i in x], values, width=width, label=label, color=color)
        for i, value in enumerate(values):
            ax_metrics.text(i + offset, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)

    labels = [f"{row.model}\n{row.task}\nth={row.threshold:g}" for row in summary.itertuples()]
    ax_metrics.set_xticks(list(x))
    ax_metrics.set_xticklabels(labels, fontsize=9)
    ax_metrics.set_ylim(0.0, 1.05)
    ax_metrics.set_ylabel("Score")
    ax_metrics.set_title("Test Metrics at Validation-Selected Threshold")
    ax_metrics.grid(axis="y", alpha=0.25)
    ax_metrics.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.03), frameon=False)

    for spec in MODEL_FILES:
        table = _read_threshold_table(REPO_ROOT / spec["path"])
        test = table[(table["split"] == "test") & table["threshold_num"].notna()].sort_values("threshold_num")
        ax_f1.plot(test["threshold_num"], test["f1"], marker="o", linewidth=2.0, label=spec["model"], color=spec["color"])
        ax_pr.plot(test["threshold_num"], test["precision"], marker="o", linewidth=2.0, label=spec["model"], color=spec["color"])
        ax_pr.plot(test["threshold_num"], test["recall"], marker="x", linewidth=1.7, linestyle="--", color=spec["color"], alpha=0.75)

    ax_f1.set_title("Test F1 Across Thresholds")
    ax_f1.set_xlabel("Threshold")
    ax_f1.set_ylabel("F1")
    ax_f1.set_ylim(0.0, 0.78)
    ax_f1.grid(alpha=0.25)
    ax_f1.legend(fontsize=8, frameon=False)

    ax_pr.set_title("Test Precision/Recall Tradeoff")
    ax_pr.set_xlabel("Threshold")
    ax_pr.set_ylabel("Score")
    ax_pr.set_ylim(0.0, 1.05)
    ax_pr.grid(alpha=0.25)
    ax_pr.text(
        0.02,
        0.04,
        "solid: precision, dashed: recall",
        transform=ax_pr.transAxes,
        fontsize=9,
        color="#444444",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.8},
    )

    fig.suptitle("Anchor-Free Sewer Prediction: U-Net and CNN-GNN Comparison", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "model_comparison" / "unet_cnn_gnn_comparison.png",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    plot(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
