"""Prepare raster tiles for anchor-free U-Net sewer-corridor segmentation.

Workstream: Codex

The output target mask is derived from utility truth lines. All input channels
come from surface/context layers only.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_GDAL_DATA = Path(sys.prefix) / "Library" / "share" / "gdal"
if "GDAL_DATA" not in os.environ and DEFAULT_GDAL_DATA.exists():
    os.environ["GDAL_DATA"] = str(DEFAULT_GDAL_DATA)

from osgeo import gdal, ogr, osr  # noqa: E402

from pipe_network_completion.anchor_free.config import load_anchor_free_config  # noqa: E402

gdal.UseExceptions()


BASE_CHANNEL_NAMES = [
    "road_line",
    "road_distance_100m",
    "building_area",
    "building_distance_100m",
    "building_point_density_50m",
    "cadastral_road_distance_100m",
    "address_point_density_50m",
    "natural_boundary_distance_100m",
]

WATERCOURSE_CHANNEL_NAMES = [
    "watercourse_drainage_line",
    "watercourse_drainage_distance_100m",
    "watercourse_corridor_centreline",
    "watercourse_corridor_centreline_distance_100m",
    "watercourse_corridor_area",
    "watercourse_corridor_distance_100m",
]


def _sigma_tag(sigma_m: float) -> str:
    value = float(sigma_m)
    return str(int(value)) if value.is_integer() else str(value).replace(".", "p")


def _soft_channel_names(*, include_watercourses: bool, soft_sigma_m: float) -> list[str]:
    tag = _sigma_tag(soft_sigma_m)
    names = [
        f"road_line_heatmap_{tag}m",
        f"cadastral_road_heatmap_{tag}m",
        f"natural_boundary_heatmap_{tag}m",
        f"building_area_heatmap_{tag}m",
    ]
    if include_watercourses:
        names.extend(
            [
                f"watercourse_drainage_heatmap_{tag}m",
                f"watercourse_corridor_centreline_heatmap_{tag}m",
                f"watercourse_corridor_area_heatmap_{tag}m",
            ]
        )
    return names


def _channel_names(*, include_watercourses: bool, include_soft_context: bool = False, soft_sigma_m: float = 30.0) -> list[str]:
    names = list(BASE_CHANNEL_NAMES)
    if include_watercourses:
        names.extend(WATERCOURSE_CHANNEL_NAMES)
    if include_soft_context:
        names.extend(_soft_channel_names(include_watercourses=include_watercourses, soft_sigma_m=soft_sigma_m))
    return names


def _resolve(path: str | Path | None) -> Path | None:
    if path in (None, "", "null"):
        return None
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _relative(path: Path, base: Path = REPO_ROOT) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _expand_configs(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(match) for match in glob.glob(str(_resolve(pattern))))
        if matches:
            out.extend(matches)
        else:
            path = _resolve(pattern)
            if path is None or not path.exists():
                raise FileNotFoundError(pattern)
            out.append(path)
    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in out:
        resolved = path.resolve()
        if resolved not in seen:
            deduped.append(path)
            seen.add(resolved)
    return deduped


def _dataset(bounds: tuple[float, float, float, float], pixel_size_m: float, target_crs: str, dtype) -> gdal.Dataset:
    xmin, ymin, xmax, ymax = bounds
    width = int(round((xmax - xmin) / float(pixel_size_m)))
    height = int(round((ymax - ymin) / float(pixel_size_m)))
    ds = gdal.GetDriverByName("MEM").Create("", width, height, 1, dtype)
    ds.SetGeoTransform((xmin, float(pixel_size_m), 0.0, ymax, 0.0, -float(pixel_size_m)))
    srs = osr.SpatialReference()
    srs.SetFromUserInput(target_crs)
    ds.SetProjection(srs.ExportToWkt())
    return ds


def _rasterize(path: Path | None, bounds: tuple[float, float, float, float], pixel_size_m: float, target_crs: str) -> np.ndarray:
    ds = _dataset(bounds, pixel_size_m, target_crs, gdal.GDT_Byte)
    if path is not None and path.exists():
        src = ogr.Open(str(path))
        if src is None:
            raise RuntimeError(f"OGR failed to open {path}")
        xmin, ymin, xmax, ymax = bounds
        for layer_idx in range(src.GetLayerCount()):
            layer = src.GetLayerByIndex(layer_idx)
            layer.SetSpatialFilterRect(float(xmin), float(ymin), float(xmax), float(ymax))
            try:
                result = gdal.RasterizeLayer(
                    ds,
                    [1],
                    layer,
                    burn_values=[1],
                    options=["ALL_TOUCHED=TRUE"],
                )
            finally:
                layer.SetSpatialFilter(None)
            if result != 0:
                raise RuntimeError(f"GDAL failed to rasterize {path}")
    arr = ds.GetRasterBand(1).ReadAsArray()
    return (arr > 0).astype("float32")


def _distance_channel(mask: np.ndarray, pixel_size_m: float, max_distance_m: float) -> np.ndarray:
    distance = distance_transform_edt(mask <= 0.0) * float(pixel_size_m)
    return np.clip(distance, 0.0, float(max_distance_m)).astype("float32") / float(max_distance_m)


def _density_channel(mask: np.ndarray, pixel_size_m: float, sigma_m: float) -> np.ndarray:
    sigma_px = max(float(sigma_m) / float(pixel_size_m), 0.5)
    density = gaussian_filter(mask.astype("float32"), sigma=sigma_px)
    max_value = float(density.max())
    if max_value <= 0.0:
        return density.astype("float32")
    return (density / max_value).astype("float32")


def _write_channel_summary(index: pd.DataFrame, output_dir: Path, channel_names: list[str]) -> None:
    sums = np.zeros(len(channel_names), dtype="float64")
    sq_sums = np.zeros(len(channel_names), dtype="float64")
    mins = np.full(len(channel_names), np.inf, dtype="float64")
    maxs = np.full(len(channel_names), -np.inf, dtype="float64")
    count = 0
    for value in index["tile_path"]:
        tile_path = output_dir / str(value)
        with np.load(tile_path, allow_pickle=True) as data:
            x = data["x"].astype("float64")
        flat = x.reshape(x.shape[0], -1)
        sums += flat.sum(axis=1)
        sq_sums += np.square(flat).sum(axis=1)
        mins = np.minimum(mins, flat.min(axis=1))
        maxs = np.maximum(maxs, flat.max(axis=1))
        count += flat.shape[1]
    if count == 0:
        return
    mean = sums / float(count)
    variance = np.maximum((sq_sums / float(count)) - np.square(mean), 0.0)
    summary = pd.DataFrame(
        {
            "channel": channel_names,
            "mean": mean,
            "std": np.sqrt(variance),
            "variance": variance,
            "min": mins,
            "max": maxs,
        }
    )
    summary.to_csv(output_dir / "channel_summary.csv", index=False)


def _source_bounds_from_aoi(cfg: dict, *, target_crs: str) -> tuple[float, float, float, float]:
    data = dict(cfg.get("data", {}))
    roads_path = _resolve(data.get("roads_path"))
    if roads_path is None or not roads_path.exists():
        raise FileNotFoundError("roads_path is required to derive U-Net tile bounds")

    import geopandas as gpd

    aoi_path = roads_path.parent / "aoi.geojson"
    if aoi_path.exists():
        aoi = gpd.read_file(aoi_path).to_crs(target_crs)
        return tuple(float(value) for value in aoi.total_bounds)

    roads = gpd.read_file(roads_path).to_crs(target_crs)
    return tuple(float(value) for value in roads.total_bounds)


def _center_tile_bounds_from_aoi(cfg: dict, *, pixel_size_m: float, tile_size_px: int) -> tuple[float, float, float, float]:
    graph = dict(cfg.get("graph", {}))
    target_crs = str(graph.get("target_crs", "EPSG:28356"))
    xmin, ymin, xmax, ymax = _source_bounds_from_aoi(cfg, target_crs=target_crs)
    cx = float((xmin + xmax) / 2.0)
    cy = float((ymin + ymax) / 2.0)
    half = float(pixel_size_m) * int(tile_size_px) / 2.0
    return (cx - half, cy - half, cx + half, cy + half)


def _grid_tile_bounds(
    source_bounds: tuple[float, float, float, float],
    *,
    pixel_size_m: float,
    tile_size_px: int,
) -> list[tuple[int, int, tuple[float, float, float, float]]]:
    xmin, ymin, xmax, ymax = source_bounds
    tile_width_m = float(pixel_size_m) * int(tile_size_px)
    n_cols = max(1, int(math.ceil((float(xmax) - float(xmin)) / tile_width_m)))
    n_rows = max(1, int(math.ceil((float(ymax) - float(ymin)) / tile_width_m)))
    tiles: list[tuple[int, int, tuple[float, float, float, float]]] = []
    for row in range(n_rows):
        for col in range(n_cols):
            tile_xmin = float(xmin) + col * tile_width_m
            tile_ymin = float(ymin) + row * tile_width_m
            tiles.append(
                (
                    row,
                    col,
                    (
                        tile_xmin,
                        tile_ymin,
                        tile_xmin + tile_width_m,
                        tile_ymin + tile_width_m,
                    ),
                )
            )
    return tiles


def _prepare_tile(
    config_path: Path,
    *,
    bounds: tuple[float, float, float, float],
    tile_suffix: str | None,
    grid_row: int | None,
    grid_col: int | None,
    include_watercourses: bool,
    require_watercourse_complete: bool,
    include_soft_context: bool,
    soft_sigma_m: float,
    cadastral_dir: Path,
    output_dir: Path,
    pixel_size_m: float,
    tile_size_px: int,
    label_buffer_m: float,
) -> dict[str, object]:
    cfg = load_anchor_free_config(config_path)
    data = dict(cfg.get("data", {}))
    graph = dict(cfg.get("graph", {}))
    aoi = dict(cfg.get("aoi", {}))
    aoi_id = str(aoi.get("aoi_id") or cfg.get("experiment_name") or config_path.stem)
    split = str(aoi.get("split", "unknown"))
    target_crs = str(graph.get("target_crs", "EPSG:28356"))
    tile_id = aoi_id if tile_suffix is None else f"{aoi_id}_{tile_suffix}"
    watercourse_complete = bool(
        aoi.get("watercourse_context_complete", False)
        or graph.get("watercourse_context_complete", False)
    )
    if include_watercourses and require_watercourse_complete and not watercourse_complete:
        raise ValueError(f"{aoi_id} requested watercourse channels but watercourse_context_complete is false")

    roads = _rasterize(_resolve(data.get("roads_path")), bounds, pixel_size_m, target_crs)
    buildings = _rasterize(_resolve(data.get("buildings_path")), bounds, pixel_size_m, target_crs)
    building_points = _rasterize(_resolve(data.get("building_points_path")), bounds, pixel_size_m, target_crs)
    cadastral_roads = _rasterize(cadastral_dir / "cadastral_roads_epsg28356.fgb", bounds, pixel_size_m, target_crs)
    address_points = _rasterize(cadastral_dir / "address_points_epsg28356.fgb", bounds, pixel_size_m, target_crs)
    natural_boundaries = _rasterize(cadastral_dir / "natural_boundaries_epsg28356.fgb", bounds, pixel_size_m, target_crs)
    truth_line = _rasterize(_resolve(data.get("utility_truth_path")), bounds, pixel_size_m, target_crs)
    channels = [
        roads,
        _distance_channel(roads, pixel_size_m, 100.0),
        buildings,
        _distance_channel(buildings, pixel_size_m, 100.0),
        _density_channel(building_points, pixel_size_m, 50.0),
        _distance_channel(cadastral_roads, pixel_size_m, 100.0),
        _density_channel(address_points, pixel_size_m, 50.0),
        _distance_channel(natural_boundaries, pixel_size_m, 100.0),
    ]
    watercourse_masks: dict[str, np.ndarray] = {}
    if include_watercourses:
        drainage_lines = _rasterize(_resolve(data.get("watercourse_drainage_lines_path")), bounds, pixel_size_m, target_crs)
        corridor_centrelines = _rasterize(
            _resolve(data.get("watercourse_corridor_centrelines_path")),
            bounds,
            pixel_size_m,
            target_crs,
        )
        corridors = _rasterize(_resolve(data.get("watercourse_corridors_path")), bounds, pixel_size_m, target_crs)
        watercourse_masks = {
            "drainage_lines": drainage_lines,
            "corridor_centrelines": corridor_centrelines,
            "corridors": corridors,
        }
        channels.extend(
            [
                drainage_lines,
                _distance_channel(drainage_lines, pixel_size_m, 100.0),
                corridor_centrelines,
                _distance_channel(corridor_centrelines, pixel_size_m, 100.0),
                corridors,
                _distance_channel(corridors, pixel_size_m, 100.0),
            ]
        )
    if include_soft_context:
        channels.extend(
            [
                _density_channel(roads, pixel_size_m, soft_sigma_m),
                _density_channel(cadastral_roads, pixel_size_m, soft_sigma_m),
                _density_channel(natural_boundaries, pixel_size_m, soft_sigma_m),
                _density_channel(buildings, pixel_size_m, soft_sigma_m),
            ]
        )
        if include_watercourses:
            channels.extend(
                [
                    _density_channel(watercourse_masks["drainage_lines"], pixel_size_m, soft_sigma_m),
                    _density_channel(watercourse_masks["corridor_centrelines"], pixel_size_m, soft_sigma_m),
                    _density_channel(watercourse_masks["corridors"], pixel_size_m, soft_sigma_m),
                ]
            )

    channel_names = _channel_names(
        include_watercourses=include_watercourses,
        include_soft_context=include_soft_context,
        soft_sigma_m=soft_sigma_m,
    )

    x = np.stack(channels, axis=0).astype("float32")
    y = np.zeros_like(truth_line, dtype="float32")
    if truth_line.max() > 0:
        y = ((distance_transform_edt(truth_line <= 0.0) * float(pixel_size_m)) <= float(label_buffer_m)).astype("float32")

    tile_path = output_dir / "tiles" / f"{tile_id}.npz"
    tile_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        tile_path,
        x=x,
        y=y,
        channel_names=np.array(channel_names),
        aoi_id=aoi_id,
        tile_id=tile_id,
        split=split,
        bounds=np.array(bounds, dtype="float64"),
        grid_row=-1 if grid_row is None else int(grid_row),
        grid_col=-1 if grid_col is None else int(grid_col),
        pixel_size_m=float(pixel_size_m),
        label_buffer_m=float(label_buffer_m),
    )
    return {
        "tile_id": tile_id,
        "aoi_id": aoi_id,
        "split": split,
        "tile_path": _relative(tile_path, output_dir),
        "config_path": _relative(config_path),
        "grid_row": -1 if grid_row is None else int(grid_row),
        "grid_col": -1 if grid_col is None else int(grid_col),
        "pixel_size_m": float(pixel_size_m),
        "tile_size_px": int(tile_size_px),
        "label_buffer_m": float(label_buffer_m),
        "positive_pixel_fraction": float(y.mean()),
        "bounds_xmin": bounds[0],
        "bounds_ymin": bounds[1],
        "bounds_xmax": bounds[2],
        "bounds_ymax": bounds[3],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=["configs/aois_24/*.yaml"])
    parser.add_argument(
        "--cadastral-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "context" / "study_area" / "cadastral_sewer_extent_exact_epsg28356",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "processed" / "unet_tiles" / "aoi24_cadastral")
    parser.add_argument("--pixel-size-m", type=float, default=10.0)
    parser.add_argument("--tile-size-px", type=int, default=256)
    parser.add_argument("--label-buffer-m", type=float, default=10.0)
    parser.add_argument(
        "--grid-tiles",
        action="store_true",
        help="Tile each AOI into adjacent non-overlapping windows instead of one center crop.",
    )
    parser.add_argument(
        "--include-watercourses",
        action="store_true",
        help="Add explicit watercourse drainage/corridor mask and distance channels.",
    )
    parser.add_argument(
        "--allow-incomplete-watercourses",
        action="store_true",
        help="Allow watercourse channels even when the AOI config is not marked complete.",
    )
    parser.add_argument(
        "--include-soft-context",
        action="store_true",
        help="Add Gaussian-smoothed heatmap channels for line/polygon context supports.",
    )
    parser.add_argument(
        "--soft-context-sigma-m",
        type=float,
        default=30.0,
        help="Gaussian sigma in metres for soft context heatmap channels.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    cadastral_dir = args.cadastral_dir if args.cadastral_dir.is_absolute() else REPO_ROOT / args.cadastral_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for config_path in _expand_configs(args.configs):
        cfg = load_anchor_free_config(config_path)
        graph = dict(cfg.get("graph", {}))
        target_crs = str(graph.get("target_crs", "EPSG:28356"))
        if args.grid_tiles:
            tile_specs = [
                (row, col, bounds, f"r{row:02d}_c{col:02d}")
                for row, col, bounds in _grid_tile_bounds(
                    _source_bounds_from_aoi(cfg, target_crs=target_crs),
                    pixel_size_m=float(args.pixel_size_m),
                    tile_size_px=int(args.tile_size_px),
                )
            ]
        else:
            tile_specs = [
                (
                    None,
                    None,
                    _center_tile_bounds_from_aoi(cfg, pixel_size_m=float(args.pixel_size_m), tile_size_px=int(args.tile_size_px)),
                    None,
                )
            ]

        for grid_row, grid_col, bounds, suffix in tile_specs:
            row = _prepare_tile(
                config_path,
                bounds=bounds,
                tile_suffix=suffix,
                grid_row=grid_row,
                grid_col=grid_col,
                include_watercourses=bool(args.include_watercourses),
                require_watercourse_complete=not bool(args.allow_incomplete_watercourses),
                include_soft_context=bool(args.include_soft_context),
                soft_sigma_m=float(args.soft_context_sigma_m),
                cadastral_dir=cadastral_dir,
                output_dir=output_dir,
                pixel_size_m=float(args.pixel_size_m),
                tile_size_px=int(args.tile_size_px),
                label_buffer_m=float(args.label_buffer_m),
            )
            rows.append(row)
            print(f"{row['tile_id']} ({row['split']}): positive_pixel_fraction={row['positive_pixel_fraction']:.4f}")

    index = pd.DataFrame(rows)
    index.to_csv(output_dir / "tiles_index.csv", index=False)
    channel_names = _channel_names(
        include_watercourses=bool(args.include_watercourses),
        include_soft_context=bool(args.include_soft_context),
        soft_sigma_m=float(args.soft_context_sigma_m),
    )
    _write_channel_summary(index, output_dir, channel_names)
    manifest = {
        "workstream": "Codex",
        "description": "U-Net raster tiles. Inputs are surface/context layers; utility truth is used only for target masks.",
        "channel_names": channel_names,
        "n_tiles": int(len(index)),
        "pixel_size_m": float(args.pixel_size_m),
        "tile_size_px": int(args.tile_size_px),
        "label_buffer_m": float(args.label_buffer_m),
        "grid_tiles": bool(args.grid_tiles),
        "uses_absolute_location_channels": False,
        "include_watercourses": bool(args.include_watercourses),
        "require_watercourse_complete": not bool(args.allow_incomplete_watercourses),
        "include_soft_context": bool(args.include_soft_context),
        "soft_context_sigma_m": float(args.soft_context_sigma_m),
        "cadastral_dir": _relative(cadastral_dir),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {_relative(output_dir / 'tiles_index.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
