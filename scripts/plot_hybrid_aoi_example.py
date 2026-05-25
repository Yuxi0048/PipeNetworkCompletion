"""Plot one AOI example comparing U-Net and CNN-GNN variants.

Workstream: Codex
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_GDAL_DATA = Path(sys.prefix) / "Library" / "share" / "gdal"
if "GDAL_DATA" not in os.environ and DEFAULT_GDAL_DATA.exists():
    os.environ["GDAL_DATA"] = str(DEFAULT_GDAL_DATA)

from pipe_network_completion.anchor_free.model import resolve_torch_device  # noqa: E402
from pipe_network_completion.anchor_free.unet_segmentation import load_tile_index, make_unet_model  # noqa: E402
from pipe_network_completion.anchor_free.unet_skeleton_hybrid import (  # noqa: E402
    build_aoi_mosaic,
    predict_tile_probabilities,
)


def _load_unet_mosaic(tile_index_path: Path, checkpoint_path: Path, aoi_id: str, device: torch.device):
    index = load_tile_index(tile_index_path)
    aoi_index = index[index["aoi_id"].astype(str) == str(aoi_id)].copy()
    if aoi_index.empty:
        raise ValueError(f"AOI {aoi_id!r} not found in {tile_index_path}")
    aoi_index = aoi_index.sort_values(["grid_row", "grid_col"]).reset_index(drop=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = make_unet_model(
        in_channels=len(checkpoint["channel_names"]),
        encoder_name=checkpoint["encoder_name"],
        encoder_weights=checkpoint["encoder_weights"],
        classes=1,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    probabilities = predict_tile_probabilities(
        model,
        aoi_index,
        mean=checkpoint["mean"],
        std=checkpoint["std"],
        device=device,
        batch_size=8,
    )
    return build_aoi_mosaic(aoi_index, probabilities)


def _context_rgb(mosaic) -> np.ndarray:
    names = list(mosaic.channel_names)
    idx = {name: pos for pos, name in enumerate(names)}
    h, w = mosaic.y.shape
    rgb = np.ones((h, w, 3), dtype="float32")

    building = mosaic.x[idx.get("building_area", 0)] > 0.0 if "building_area" in idx else np.zeros((h, w), dtype=bool)
    road = mosaic.x[idx.get("road_line", 0)] > 0.0 if "road_line" in idx else np.zeros((h, w), dtype=bool)
    drainage = mosaic.x[idx.get("watercourse_drainage_line", 0)] > 0.0 if "watercourse_drainage_line" in idx else np.zeros((h, w), dtype=bool)
    corridor = mosaic.x[idx.get("watercourse_corridor_area", 0)] > 0.0 if "watercourse_corridor_area" in idx else np.zeros((h, w), dtype=bool)
    address_density = mosaic.x[idx.get("address_point_density_50m", 0)] if "address_point_density_50m" in idx else np.zeros((h, w), dtype="float32")

    rgb[building] = [0.72, 0.72, 0.72]
    rgb[corridor] = [0.73, 0.90, 0.83]
    rgb[drainage] = [0.10, 0.45, 0.95]
    density_alpha = np.clip(address_density, 0.0, 1.0)[..., None] * 0.35
    rgb = rgb * (1.0 - density_alpha) + np.array([0.95, 0.62, 0.20], dtype="float32") * density_alpha
    rgb[road] = [0.03, 0.03, 0.03]
    return np.clip(rgb, 0.0, 1.0)


def _extent(mosaic) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = mosaic.bounds
    return xmin, xmax, ymin, ymax


def _plot_prob(ax, mosaic, threshold: float, title: str):
    extent = _extent(mosaic)
    ax.imshow(mosaic.probability, extent=extent, origin="upper", cmap="magma", vmin=0.0, vmax=1.0)
    truth = np.ma.masked_where(mosaic.y < 0.5, mosaic.y)
    pred = np.ma.masked_where(mosaic.probability < float(threshold), mosaic.probability)
    ax.imshow(pred, extent=extent, origin="upper", cmap="Greens", alpha=0.35, vmin=0.0, vmax=1.0)
    ax.imshow(truth, extent=extent, origin="upper", cmap="Blues", alpha=0.35, vmin=0.0, vmax=1.0)
    ax.set_title(title)


def _edge_colors(edges: gpd.GeoDataFrame) -> dict[str, gpd.GeoDataFrame]:
    pred = edges["gnn_pred_best_val"].astype(int) == 1
    truth = edges["y"].astype(int) == 1
    return {
        "candidate": edges,
        "tp": edges[pred & truth],
        "fp": edges[pred & ~truth],
        "fn": edges[~pred & truth],
    }


def _plot_edges(ax, context_rgb: np.ndarray, mosaic, edge_path: Path, aoi_id: str, title: str):
    extent = _extent(mosaic)
    ax.imshow(context_rgb, extent=extent, origin="upper")
    truth = np.ma.masked_where(mosaic.y < 0.5, mosaic.y)
    ax.imshow(truth, extent=extent, origin="upper", cmap="Blues", alpha=0.25, vmin=0.0, vmax=1.0)
    edges = gpd.read_file(edge_path)
    edges = edges[edges["aoi_id"].astype(str) == str(aoi_id)].copy()
    groups = _edge_colors(edges)
    if not groups["candidate"].empty:
        groups["candidate"].plot(ax=ax, color="#999999", linewidth=0.25, alpha=0.25)
    if not groups["fn"].empty:
        groups["fn"].plot(ax=ax, color="#1d5fd1", linewidth=0.9, alpha=0.95)
    if not groups["fp"].empty:
        groups["fp"].plot(ax=ax, color="#d62728", linewidth=0.9, alpha=0.95)
    if not groups["tp"].empty:
        groups["tp"].plot(ax=ax, color="#2ca02c", linewidth=1.0, alpha=0.95)
    precision = len(groups["tp"]) / max(len(groups["tp"]) + len(groups["fp"]), 1)
    recall = len(groups["tp"]) / max(len(groups["tp"]) + len(groups["fn"]), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    ax.set_title(f"{title}\nAOI edge P/R/F1={precision:.2f}/{recall:.2f}/{f1:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi-id", default="small_aoi_08_13")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "model_comparison" / "aoi_example_small_aoi_08_13.png",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_torch_device(args.device)
    watercourse_mosaic = _load_unet_mosaic(
        REPO_ROOT / "data" / "processed" / "unet_tiles" / "aoi112_grid128_cadastral_watercourses" / "tiles_index.csv",
        REPO_ROOT / "outputs" / "unet_segmentation_aoi112_watercourses" / "unet_checkpoint.pt",
        args.aoi_id,
        device,
    )
    soft_mosaic = _load_unet_mosaic(
        REPO_ROOT / "data" / "processed" / "unet_tiles" / "aoi112_grid128_watercourses_soft30" / "tiles_index.csv",
        REPO_ROOT / "outputs" / "unet_segmentation_aoi112_watercourses_soft30" / "unet_checkpoint.pt",
        args.aoi_id,
        device,
    )
    context_rgb = _context_rgb(soft_mosaic)
    extent = _extent(soft_mosaic)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    axes[0, 0].imshow(context_rgb, extent=extent, origin="upper")
    truth = np.ma.masked_where(soft_mosaic.y < 0.5, soft_mosaic.y)
    axes[0, 0].imshow(truth, extent=extent, origin="upper", cmap="Blues", alpha=0.35, vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("Context + ground-truth sewer buffer")

    _plot_prob(axes[0, 1], watercourse_mosaic, 0.7, "U-Net 14ch watercourse\np>=0.7 green, truth blue")
    _plot_prob(axes[0, 2], soft_mosaic, 0.7, "U-Net 21ch soft heatmaps\np>=0.7 green, truth blue")

    _plot_edges(
        axes[1, 0],
        context_rgb,
        soft_mosaic,
        REPO_ROOT / "outputs" / "cnn_gnn_hybrid_aoi112_watercourses_thr02" / "edge_predictions.geojson",
        args.aoi_id,
        "CNN-GNN 14ch graph",
    )
    _plot_edges(
        axes[1, 1],
        context_rgb,
        soft_mosaic,
        REPO_ROOT / "outputs" / "cnn_gnn_hybrid_aoi112_watercourses_soft30_thr02" / "edge_predictions.geojson",
        args.aoi_id,
        "CNN-GNN 21ch soft graph",
    )
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.03,
        0.95,
        "Legend\n\n"
        "Context panel:\n"
        "black = roads\n"
        "gray = building footprints\n"
        "blue/green = watercourse layers\n"
        "orange haze = address density\n"
        "blue overlay = truth sewer buffer\n\n"
        "GNN panels:\n"
        "gray = candidate skeleton edge\n"
        "green = true positive predicted edge\n"
        "red = false positive predicted edge\n"
        "blue = false negative truth edge\n\n"
        "This is one held-out test AOI, not an aggregate result.",
        va="top",
        fontsize=11,
    )

    for ax in axes.ravel()[:5]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"AOI Example: {args.aoi_id}", fontsize=16)
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
