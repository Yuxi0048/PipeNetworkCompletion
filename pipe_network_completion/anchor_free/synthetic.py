"""Synthetic smoke-test fixture for the anchor-free pipeline."""

# Workstream: Codex

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
from shapely.geometry import LineString, Point


@dataclass(frozen=True)
class SyntheticAnchorFreeData:
    roads: gpd.GeoDataFrame
    buildings: gpd.GeoDataFrame
    utility_truth: gpd.GeoDataFrame
    sources_sinks: gpd.GeoDataFrame


def make_synthetic_anchor_free_data(
    *,
    grid_size: int = 4,
    spacing_m: float = 100.0,
    crs: str = "EPSG:3857",
) -> SyntheticAnchorFreeData:
    """Create a small road grid with a truth network following road edges."""

    roads = []
    road_id = 0
    for y in range(grid_size):
        for x in range(grid_size - 1):
            p1 = (x * spacing_m, y * spacing_m)
            p2 = ((x + 1) * spacing_m, y * spacing_m)
            roads.append(
                {
                    "road_id": road_id,
                    "road_class": "collector" if y == grid_size // 2 else "local",
                    "geometry": LineString([p1, p2]),
                }
            )
            road_id += 1
    for x in range(grid_size):
        for y in range(grid_size - 1):
            p1 = (x * spacing_m, y * spacing_m)
            p2 = (x * spacing_m, (y + 1) * spacing_m)
            roads.append(
                {
                    "road_id": road_id,
                    "road_class": "arterial" if x == 1 else "local",
                    "geometry": LineString([p1, p2]),
                }
            )
            road_id += 1

    buildings = []
    for y in range(grid_size):
        for x in range(grid_size):
            buildings.append(
                {
                    "building_id": y * grid_size + x,
                    "demand_proxy": 1.0,
                    "geometry": Point(x * spacing_m + 20.0, y * spacing_m + 18.0),
                }
            )

    middle_y = (grid_size // 2) * spacing_m
    truth_lines = [
        LineString([(0.0, middle_y), ((grid_size - 1) * spacing_m, middle_y)]),
        LineString([(spacing_m, 0.0), (spacing_m, (grid_size - 1) * spacing_m)]),
    ]

    sources_sinks = [
        {"facility_type": "outlet", "geometry": Point(0.0, middle_y)},
        {"facility_type": "source", "geometry": Point(spacing_m, 0.0)},
    ]

    return SyntheticAnchorFreeData(
        roads=gpd.GeoDataFrame(roads, geometry="geometry", crs=crs),
        buildings=gpd.GeoDataFrame(buildings, geometry="geometry", crs=crs),
        utility_truth=gpd.GeoDataFrame(
            [{"truth_id": i, "geometry": geom} for i, geom in enumerate(truth_lines)],
            geometry="geometry",
            crs=crs,
        ),
        sources_sinks=gpd.GeoDataFrame(sources_sinks, geometry="geometry", crs=crs),
    )
