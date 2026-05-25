"""Audit line features excluded by the two-anchor endpoint assumption.

For each sewer line group, a line is treated as represented by an anchor graph
only if both line endpoints have at least one selected anchor point within the
chosen tolerance. Lines that fail this test are "orphans" under the two-anchor
connection assumption.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Iterable

import fiona
import numpy as np
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree


ANCHOR_POLICIES = {
    "manholes": ["SewerManholes_ExportFeatures.shp"],
    "visible_surface_assets": [
        "SewerManholes_ExportFeatures.shp",
        "SewersqlSewerP_ExportFeature.shp",
        "SewerPumpStati_ExportFeature.shp",
        "SewerVent_ExportFeatures.shp",
        "SewerControlVa_ExportFeature.shp",
        "SewerSystemVal_ExportFeature.shp",
        "SewerDevice_ExportFeatures.shp",
        "UUSewertreatme_ExportFeature.shp",
    ],
    "visible_plus_hidden_fittings": [
        "SewerManholes_ExportFeatures.shp",
        "SewersqlSewerP_ExportFeature.shp",
        "SewerPumpStati_ExportFeature.shp",
        "SewerVent_ExportFeatures.shp",
        "SewerControlVa_ExportFeature.shp",
        "SewerSystemVal_ExportFeature.shp",
        "SewerDevice_ExportFeatures.shp",
        "UUSewertreatme_ExportFeature.shp",
        "SewerFitting_ExportFeatures.shp",
    ],
}

LINE_GROUPS = {
    "gravity_all": [
        "SewerGravityMa_ExportFeature1.shp",
        "SewerGravityMa_ExportFeature2.shp",
    ],
    "pressure_mains": ["SewerPressureM_ExportFeature.shp"],
    "service_laterals": ["SewerService_ExportFeatures.shp"],
    "vent_pipes": ["SewerVentPipe_ExportFeatures.shp"],
}


def transformer_for(src: fiona.Collection, target_crs: CRS) -> Transformer:
    source = CRS.from_user_input(src.crs_wkt or src.crs)
    return Transformer.from_crs(source, target_crs, always_xy=True)


def point_xy(geometry: dict) -> tuple[float, float] | None:
    if geometry is None:
        return None
    coords = geometry.get("coordinates")
    if not coords:
        return None
    geom_type = geometry.get("type")
    if geom_type == "Point":
        return float(coords[0]), float(coords[1])
    if geom_type == "MultiPoint" and coords:
        return float(coords[0][0]), float(coords[0][1])
    return None


def line_endpoints(geometry: dict) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if geometry is None:
        return None
    coords = geometry.get("coordinates")
    if not coords:
        return None
    geom_type = geometry.get("type")
    if geom_type == "LineString" and len(coords) >= 2:
        return tuple(coords[0]), tuple(coords[-1])
    if geom_type == "MultiLineString":
        parts = [part for part in coords if len(part) >= 2]
        if not parts:
            return None
        return tuple(parts[0][0]), tuple(parts[-1][-1])
    return None


def load_anchor_coords(sewer_root: Path, files: Iterable[str], target_crs: CRS) -> np.ndarray:
    coords: list[tuple[float, float]] = []
    for filename in files:
        path = sewer_root / filename
        with fiona.open(path) as src:
            transformer = transformer_for(src, target_crs)
            for feature in src:
                xy = point_xy(feature.get("geometry"))
                if xy is None:
                    continue
                coords.append(transformer.transform(*xy))
    return np.asarray(coords, dtype=float)


def load_line_endpoints(
    sewer_root: Path,
    files: Iterable[str],
    target_crs: CRS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts: list[tuple[float, float]] = []
    ends: list[tuple[float, float]] = []
    subtypes: list[str] = []
    for filename in files:
        path = sewer_root / filename
        with fiona.open(path) as src:
            transformer = transformer_for(src, target_crs)
            for feature in src:
                endpoints = line_endpoints(feature.get("geometry"))
                if endpoints is None:
                    continue
                start, end = endpoints
                starts.append(transformer.transform(*start))
                ends.append(transformer.transform(*end))
                props = feature.get("properties") or {}
                subtypes.append(str(props.get("SUBTYPECD", "")))
    return np.asarray(starts, dtype=float), np.asarray(ends, dtype=float), np.asarray(subtypes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sewer-root", type=Path, default=Path("data/raw/gis/sewer"))
    parser.add_argument("--target-crs", default="EPSG:28356")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/anchor_orphan_audit/line_orphans_by_anchor_policy.csv"),
    )
    parser.add_argument(
        "--output-subtype-csv",
        type=Path,
        default=Path("outputs/anchor_orphan_audit/line_orphans_by_subtype.csv"),
    )
    parser.add_argument("--tolerances-m", type=float, nargs="+", default=[1.0, 5.0])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_crs = CRS.from_user_input(args.target_crs)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_subtype_csv.parent.mkdir(parents=True, exist_ok=True)

    trees = {}
    anchor_counts = {}
    for policy, files in ANCHOR_POLICIES.items():
        coords = load_anchor_coords(args.sewer_root, files, target_crs)
        trees[policy] = cKDTree(coords)
        anchor_counts[policy] = int(len(coords))
        print(f"Loaded anchor policy {policy}: {len(coords)} points")

    summary_rows = []
    subtype_rows = []
    for line_group, files in LINE_GROUPS.items():
        starts, ends, subtypes = load_line_endpoints(args.sewer_root, files, target_crs)
        total = int(len(starts))
        print(f"Loaded line group {line_group}: {total} lines")
        subtype_total = Counter(subtypes)
        for policy, tree in trees.items():
            for tolerance in args.tolerances_m:
                start_dist, _ = tree.query(starts, k=1, distance_upper_bound=tolerance)
                end_dist, _ = tree.query(ends, k=1, distance_upper_bound=tolerance)
                start_hit = np.isfinite(start_dist)
                end_hit = np.isfinite(end_dist)
                both = start_hit & end_hit
                one_or_more = start_hit | end_hit
                orphan = ~both
                summary_rows.append(
                    {
                        "line_group": line_group,
                        "anchor_policy": policy,
                        "anchor_count": anchor_counts[policy],
                        "tolerance_m": float(tolerance),
                        "total_lines": total,
                        "both_anchored": int(both.sum()),
                        "orphan_lines": int(orphan.sum()),
                        "one_or_more_endpoint": int(one_or_more.sum()),
                        "zero_endpoint": int((~one_or_more).sum()),
                        "orphan_fraction": float(orphan.mean()) if total else 0.0,
                    }
                )
                for subtype, subtype_count in sorted(subtype_total.items()):
                    mask = subtypes == subtype
                    subtype_rows.append(
                        {
                            "line_group": line_group,
                            "subtype": subtype,
                            "anchor_policy": policy,
                            "tolerance_m": float(tolerance),
                            "total_lines": int(subtype_count),
                            "both_anchored": int((both & mask).sum()),
                            "orphan_lines": int((orphan & mask).sum()),
                            "orphan_fraction": float((orphan & mask).sum() / subtype_count)
                            if subtype_count
                            else 0.0,
                        }
                    )

    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with args.output_subtype_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(subtype_rows[0]))
        writer.writeheader()
        writer.writerows(subtype_rows)

    print(f"Wrote summary: {args.output_csv}")
    print(f"Wrote subtype detail: {args.output_subtype_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
