# Anchor-Free U-Net Segmentation Setup

Workstream: Codex

This setup adds a raster segmentation baseline for the anchor-free sewer
prediction story. It is separate from the road/skeleton GNNs.

## Framing

The U-Net predicts a sewer-corridor probability field on raster tiles:

```text
surface/context raster channels -> P(sewer corridor at pixel)
```

Ground-truth sewer geometry is used only to create the target mask. It is not an
input channel.

The current performance setup tiles each existing AOI into adjacent
non-overlapping windows with no gap. The saved tile bounds are metadata for
mapping/evaluation only; no absolute coordinate, `x/y`, or location-encoding
channel is written to the model input tensor.

## Input Channels

The initial tile generator writes these channels:

1. road centerline mask
2. road distance field, clipped at 100 m
3. building-area mask
4. building-area distance field, clipped at 100 m
5. building-point density field, Gaussian-smoothed at 50 m
6. cadastral-road distance field, clipped at 100 m
7. address-point density field, Gaussian-smoothed at 50 m
8. natural-boundary distance field, clipped at 100 m

When `--include-watercourses` is passed, the generator adds six additional
channels and requires the AOI config to be marked `watercourse_context_complete`
unless `--allow-incomplete-watercourses` is explicitly set:

9. watercourse drainage-line mask
10. watercourse drainage-line distance field, clipped at 100 m
11. watercourse corridor-centreline mask
12. watercourse corridor-centreline distance field, clipped at 100 m
13. watercourse corridor polygon mask
14. watercourse corridor polygon distance field, clipped at 100 m

When `--include-soft-context` is also passed, it adds Gaussian-smoothed support
heatmaps. With `--soft-context-sigma-m 30`, these are:

15. road-line heatmap
16. cadastral-road heatmap
17. natural-boundary heatmap
18. building-area heatmap
19. watercourse drainage-line heatmap
20. watercourse corridor-centreline heatmap
21. watercourse corridor-area heatmap

The heatmaps are max-normalized to `[0, 1]`. Training still standardizes every
channel using train-set mean and standard deviation, so the model receives
variance-normalized inputs. Tile generation also writes `channel_summary.csv`
so per-channel variance can be inspected before training.

The cadastral parcel polygon layer is not used in this first setup. The
non-parcel cadastral layers are treated as soft context fields.

## Target

The target is a binary mask created by buffering the utility-truth line raster
by `label_buffer_m`, default `10 m`. This target is a corridor/support target,
not exact legal utility location.

## Commands

Prepare tiles:

```powershell
& '.\.conda\pipe-network-completion-cuda\python.exe' -B scripts\prepare_unet_tiles.py `
  --configs configs\aois_24\*.yaml `
  --cadastral-dir data\processed\context\study_area\cadastral_sewer_extent_exact_epsg28356 `
  --output-dir data\processed\unet_tiles\aoi24_grid128_cadastral_perf `
  --pixel-size-m 10 `
  --tile-size-px 128 `
  --label-buffer-m 10 `
  --grid-tiles
```

Train:

```powershell
& '.\.conda\pipe-network-completion-cuda\python.exe' -B scripts\train_unet_segmentation.py `
  --tile-index data\processed\unet_tiles\aoi24_grid128_cadastral_perf\tiles_index.csv `
  --output-dir outputs\unet_segmentation_aoi24_grid128_perf `
  --encoder-name resnet18 `
  --encoder-weights none `
  --epochs 20 `
  --batch-size 8
```

For the current 24 AOIs this writes 384 tiles: 224 train, 64 validation, and
96 test tiles. The split is inherited from the parent AOI, so tiles from one AOI
do not cross train/validation/test.

Prepare the watercourse-complete version:

```powershell
& '.\.conda\pipe-network-completion-cuda\python.exe' -B scripts\prepare_unet_tiles.py `
  --configs configs\aois_2km_gap500_115_watercourses_complete\*.yaml `
  --cadastral-dir data\processed\context\study_area\cadastral_sewer_extent_exact_epsg28356 `
  --output-dir data\processed\unet_tiles\aoi112_grid128_cadastral_watercourses `
  --pixel-size-m 10 `
  --tile-size-px 128 `
  --label-buffer-m 10 `
  --grid-tiles `
  --include-watercourses
```

Train the watercourse-complete U-Net:

```powershell
& '.\.conda\pipe-network-completion-cuda\python.exe' -B scripts\train_unet_segmentation.py `
  --tile-index data\processed\unet_tiles\aoi112_grid128_cadastral_watercourses\tiles_index.csv `
  --output-dir outputs\unet_segmentation_aoi112_watercourses `
  --encoder-name resnet18 `
  --encoder-weights none `
  --epochs 20 `
  --batch-size 8
```

Prepare the soft-context watercourse-complete version:

```powershell
& '.\.conda\pipe-network-completion-cuda\python.exe' -B scripts\prepare_unet_tiles.py `
  --configs configs\aois_2km_gap500_115_watercourses_complete\*.yaml `
  --output-dir data\processed\unet_tiles\aoi112_grid128_watercourses_soft30 `
  --pixel-size-m 10 `
  --tile-size-px 128 `
  --label-buffer-m 10 `
  --grid-tiles `
  --include-watercourses `
  --include-soft-context `
  --soft-context-sigma-m 30
```

Train the soft-context U-Net:

```powershell
& '.\.conda\pipe-network-completion-cuda\python.exe' -B scripts\train_unet_segmentation.py `
  --tile-index data\processed\unet_tiles\aoi112_grid128_watercourses_soft30\tiles_index.csv `
  --output-dir outputs\unet_segmentation_aoi112_watercourses_soft30 `
  --encoder-name resnet18 `
  --encoder-weights none `
  --epochs 20 `
  --batch-size 8
```

## Dependency Note

The model uses `segmentation_models_pytorch.Unet`. Keep
`encoder_weights=none` unless the experiment explicitly wants pretrained image
weights. The CUDA environment is pinned to `torch==2.1.0+cu121`; avoid normal
`pip install` commands that allow pip to upgrade torch.

## Outputs

Tile preparation writes:

```text
data/processed/unet_tiles/aoi24_grid128_cadastral_perf/
  manifest.json
  tiles_index.csv
  tiles/*.npz
```

Training writes:

```text
outputs/unet_segmentation_aoi24_grid128_perf/
  channel_stats.json
  threshold_sweep.csv
  training_history.csv
  metrics.json
  unet_checkpoint.pt
```
