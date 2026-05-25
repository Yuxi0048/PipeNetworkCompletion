"""Plot U-Net sewer-corridor prediction for one AOI.

Workstream: Codex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipe_network_completion.anchor_free.model import resolve_torch_device  # noqa: E402
from pipe_network_completion.anchor_free.unet_segmentation import (  # noqa: E402
    NpzSegmentationDataset,
    load_tile_index,
    make_unet_model,
)


def _mosaic(values: list[np.ndarray], rows: list[int], cols: list[int]) -> np.ndarray:
    tile_h, tile_w = values[0].shape[-2:]
    n_rows = max(rows) + 1
    n_cols = max(cols) + 1
    if values[0].ndim == 3:
        out = np.zeros((values[0].shape[0], n_rows * tile_h, n_cols * tile_w), dtype=values[0].dtype)
        for value, row, col in zip(values, rows, cols):
            out[:, row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = value
    else:
        out = np.zeros((n_rows * tile_h, n_cols * tile_w), dtype=values[0].dtype)
        for value, row, col in zip(values, rows, cols):
            out[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = value
    return out


@torch.no_grad()
def _predict(model: torch.nn.Module, dataset: NpzSegmentationDataset, device: torch.device, batch_size: int) -> list[np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    probs: list[np.ndarray] = []
    for x, _ in loader:
        logits = model(x.to(device)).detach().cpu()
        for prob in torch.sigmoid(logits).numpy():
            probs.append(prob[0].astype("float32"))
    return probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-index", type=Path, default=REPO_ROOT / "data" / "processed" / "unet_tiles" / "aoi24_grid128_cadastral_perf" / "tiles_index.csv")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "outputs" / "unet_segmentation_aoi24_grid128_perf" / "unet_checkpoint.pt")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "unet_segmentation_aoi24_grid128_perf" / "aoi_prediction_small_aoi_02_05.png")
    parser.add_argument("--aoi-id", default="small_aoi_02_05")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index = load_tile_index(args.tile_index if args.tile_index.is_absolute() else REPO_ROOT / args.tile_index)
    aoi_index = index[index["aoi_id"].astype(str) == str(args.aoi_id)].copy()
    if aoi_index.empty:
        available = ", ".join(sorted(index["aoi_id"].astype(str).unique())[:12])
        raise ValueError(f"AOI {args.aoi_id!r} not found. Examples: {available}")
    aoi_index = aoi_index.sort_values(["grid_row", "grid_col"]).reset_index(drop=True)

    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = make_unet_model(
        in_channels=len(checkpoint["channel_names"]),
        encoder_name=checkpoint["encoder_name"],
        encoder_weights=checkpoint["encoder_weights"],
        classes=1,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = resolve_torch_device(str(args.device))
    model.to(device)

    dataset = NpzSegmentationDataset(aoi_index, mean=checkpoint["mean"], std=checkpoint["std"])
    probs = _predict(model, dataset, device, batch_size=int(args.batch_size))

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for path in aoi_index["tile_path"]:
        with np.load(path, allow_pickle=True) as data:
            xs.append(data["x"].astype("float32"))
            ys.append(data["y"].astype("float32"))
    rows = [int(value) for value in aoi_index["grid_row"]]
    cols = [int(value) for value in aoi_index["grid_col"]]
    x = _mosaic(xs, rows, cols)
    y = _mosaic(ys, rows, cols)
    prob = _mosaic(probs, rows, cols)
    pred = prob >= float(args.threshold)

    road = x[0] > 0
    building = x[2] > 0
    building_density = x[4]
    cadastral_road_near = x[5] < 0.15
    natural_near = x[7] < 0.15

    context = np.ones((*y.shape, 3), dtype="float32")
    context[building] = [0.72, 0.72, 0.72]
    context[natural_near] = [0.62, 0.84, 0.70]
    context[cadastral_road_near] = [0.82, 0.82, 0.82]
    context[road] = [0.02, 0.02, 0.02]
    density_alpha = np.clip(building_density, 0.0, 1.0)[..., None] * 0.45
    context = context * (1.0 - density_alpha) + np.array([0.95, 0.56, 0.18], dtype="float32") * density_alpha

    overlay = context.copy()
    truth_only = (y >= 0.5) & ~pred
    pred_only = pred & (y < 0.5)
    both = pred & (y >= 0.5)
    overlay[truth_only] = [0.12, 0.35, 0.95]
    overlay[pred_only] = [0.92, 0.22, 0.18]
    overlay[both] = [0.18, 0.70, 0.25]

    split = str(aoi_index["split"].iloc[0])
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    axes[0, 0].imshow(context)
    axes[0, 0].set_title("Surface/context channels")
    axes[0, 1].imshow(y, cmap="Blues", vmin=0, vmax=1)
    axes[0, 1].set_title("Ground-truth sewer buffer mask")
    im = axes[1, 0].imshow(prob, cmap="magma", vmin=0, vmax=1)
    axes[1, 0].set_title("U-Net predicted sewer probability")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title(f"Threshold overlay, p >= {float(args.threshold):.2f}")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{args.aoi_id} ({split}) - no location channels", fontsize=14)
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
