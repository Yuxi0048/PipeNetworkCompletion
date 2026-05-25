"""Train a standard U-Net on prepared raster sewer-corridor tiles.

Workstream: Codex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipe_network_completion.anchor_free.model import resolve_torch_device, torch_device_report  # noqa: E402
from pipe_network_completion.anchor_free.unet_segmentation import (  # noqa: E402
    NpzSegmentationDataset,
    bce_dice_loss,
    binary_pixel_metrics,
    collect_predictions,
    compute_channel_stats,
    load_tile_index,
    make_unet_model,
    positive_class_weight,
    save_stats,
    split_index,
)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _channel_names(index: pd.DataFrame) -> list[str]:
    with np.load(index.iloc[0]["tile_path"], allow_pickle=True) as data:
        return [str(value) for value in data["channel_names"].tolist()]


def _make_loader(dataset: NpzSegmentationDataset, *, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle, num_workers=0)


def _evaluate_split(model, dataset, device, *, batch_size: int, threshold: float) -> dict[str, float]:
    if len(dataset) == 0:
        return {}
    loader = _make_loader(dataset, batch_size=batch_size, shuffle=False)
    y_true, y_prob = collect_predictions(model, loader, device)
    return binary_pixel_metrics(y_true, y_prob, threshold=threshold)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-index", type=Path, default=REPO_ROOT / "data" / "processed" / "unet_tiles" / "aoi24_cadastral" / "tiles_index.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "unet_segmentation_aoi24")
    parser.add_argument("--encoder-name", default="resnet18")
    parser.add_argument("--encoder-weights", default="none", help="Use 'none' for no pretrained weights.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    index = load_tile_index(args.tile_index if args.tile_index.is_absolute() else REPO_ROOT / args.tile_index)
    train_index = split_index(index, "train")
    val_index = split_index(index, "val")
    test_index = split_index(index, "test")
    if train_index.empty:
        raise ValueError("Tile index has no train split.")

    channel_names = _channel_names(index)
    mean, std = compute_channel_stats(train_index)
    save_stats(output_dir / "channel_stats.json", mean=mean, std=std, channel_names=channel_names)

    train_dataset = NpzSegmentationDataset(train_index, mean=mean, std=std)
    val_dataset = NpzSegmentationDataset(val_index, mean=mean, std=std)
    test_dataset = NpzSegmentationDataset(test_index, mean=mean, std=std)
    train_loader = _make_loader(train_dataset, batch_size=args.batch_size, shuffle=True)

    encoder_weights = None if str(args.encoder_weights).lower() in {"none", "", "null"} else str(args.encoder_weights)
    model = make_unet_model(
        in_channels=len(channel_names),
        encoder_name=str(args.encoder_name),
        encoder_weights=encoder_weights,
        classes=1,
    )
    device = resolve_torch_device(str(args.device))
    model.to(device)
    pos_weight_value = positive_class_weight(train_index)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))

    history: list[dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: list[float] = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = bce_dice_loss(logits, y, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": float(epoch), "train_loss": float(np.mean(losses)) if losses else float("nan")}
        if len(val_dataset) > 0:
            val_metrics = _evaluate_split(model, val_dataset, device, batch_size=args.batch_size, threshold=float(args.threshold))
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        print(
            f"epoch={epoch:03d} loss={row['train_loss']:.4f} "
            f"val_f1={row.get('val_f1', float('nan')):.4f}"
        )

    metrics = {
        "workstream": "Codex",
        "description": "standard segmentation-models-pytorch U-Net for sewer-corridor masks",
        "tile_index": _relative(args.tile_index if args.tile_index.is_absolute() else REPO_ROOT / args.tile_index),
        "encoder_name": str(args.encoder_name),
        "encoder_weights": encoder_weights,
        "device": {
            **torch_device_report(),
            "selected_device": str(device),
        },
        "channel_names": channel_names,
        "n_train_tiles": int(len(train_dataset)),
        "n_val_tiles": int(len(val_dataset)),
        "n_test_tiles": int(len(test_dataset)),
        "pos_weight": float(pos_weight_value),
        "threshold": float(args.threshold),
        "val": _evaluate_split(model, val_dataset, device, batch_size=args.batch_size, threshold=float(args.threshold)),
        "test": _evaluate_split(model, test_dataset, device, batch_size=args.batch_size, threshold=float(args.threshold)),
    }
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "channel_names": channel_names,
            "mean": mean,
            "std": std,
            "encoder_name": str(args.encoder_name),
            "encoder_weights": encoder_weights,
        },
        output_dir / "unet_checkpoint.pt",
    )
    print(f"wrote {_relative(output_dir / 'metrics.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
