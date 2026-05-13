from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from pipe_network_completion.paths import (
    INTERIM_DATA_DIR,
    RAW_MH_ROAD_DIR,
    RAW_ROADS_DIR,
    RAW_SEWER_DIR,
)


def spatial_join_intersects(left: gpd.GeoDataFrame, right: gpd.GeoDataFrame):
    """Support both old and new GeoPandas spatial-join APIs."""

    try:
        return gpd.sjoin(left, right, how="left", predicate="intersects").dropna()
    except TypeError:
        return gpd.sjoin(left, right, how="left", op="intersects").dropna()


def split_list_by_ratio(lst, ratio, seed: int):
    items = list(lst)
    random.Random(seed).shuffle(items)

    total_len = len(items)
    ratio_sum = sum(ratio)
    split1_end = total_len * ratio[0] // ratio_sum
    split2_end = split1_end + total_len * ratio[1] // ratio_sum

    return (
        items[:split1_end],
        items[split1_end:split2_end],
        items[split2_end:],
    )


def angle_between(p1, p2):
    angle = np.arctan2(p2.x - p1.x, p2.y - p1.y)
    return np.rad2deg(angle % np.pi)


def angle2bin(angle):
    if angle < 15 or angle >= 165:
        return 0
    return int((angle - 15) / 30) + 1


def calculate_angles(gdf_road):
    angles = []
    idx = []
    angle_bin = []

    for i in range(len(gdf_road)):
        if gdf_road["geometry"][i].boundary.is_empty:
            continue
        p1 = gdf_road["geometry"][i].boundary.geoms[0]
        p2 = gdf_road["geometry"][i].boundary.geoms[1]
        angle = angle_between(p1, p2)
        angles.append(angle)
        angle_bin.append(angle2bin(angle))
        idx.append(i)

    return angles, angle_bin, idx


def extract_coordinates(point):
    return pd.Series([point.x, point.y])


def dump_pickle(value, output_dir: Path, filename: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / filename).open("wb") as file:
        pickle.dump(value, file)


def build_preprocessed_data(raw_root: Path, output_dir: Path, seed: int = 42):
    sewer_root = raw_root / "gis" / "sewer"
    road_root = raw_root / "gis" / "roads"
    mh_road_path = raw_root / "mh_road" / "MH_Road.pkl"

    if not sewer_root.exists() and RAW_SEWER_DIR.exists():
        sewer_root = RAW_SEWER_DIR
    if not road_root.exists() and RAW_ROADS_DIR.exists():
        road_root = RAW_ROADS_DIR
    if not mh_road_path.exists() and (RAW_MH_ROAD_DIR / "MH_Road.pkl").exists():
        mh_road_path = RAW_MH_ROAD_DIR / "MH_Road.pkl"

    gdf_mh = gpd.read_file(sewer_root / "SewerManholes_ExportFeatures.shp")
    gdf_ml2 = gpd.read_file(sewer_root / "SewerGravityMa_ExportFeature2.shp")
    gdf_ml2 = gdf_ml2.drop_duplicates()
    gdf_ml1 = gpd.read_file(sewer_root / "SewerGravityMa_ExportFeature1.shp")
    gdf_pump = gpd.read_file(sewer_root / "SewersqlSewerP_ExportFeature.shp")
    gdf_road = gpd.read_file(road_root / "Roads_ExportFeatures.shp")
    gdf_ml = gpd.GeoDataFrame(
        pd.concat([gdf_ml1, gdf_ml2], ignore_index=True),
        crs=gdf_ml1.crs,
    )

    with mh_road_path.open("rb") as file:
        mh_road_list = pickle.load(file)

    df_mh_road = pd.DataFrame(
        data=np.array(mh_road_list),
        columns=["OBJECTID", "NEAR_FID", "NEAR_POS", "NEAR_DIST", "SIDE"],
    )
    gdf_mh = gdf_mh.merge(df_mh_road, left_index=True, right_index=True, how="inner")

    # Append pump points into the manhole/anchor-point dataset.
    gdf_mh["SUBTYPECD"] = gdf_mh["SUBTYPECD"].astype("str") + gdf_mh["MANHOLEUSE"]
    gdf_ap = gpd.GeoDataFrame(
        pd.concat([gdf_mh, gdf_pump], ignore_index=True),
        crs=gdf_mh.crs,
    ).dropna().reset_index(drop=True)

    gdf_ap["geo"] = gdf_ap.geometry
    gdf_ml["index"] = gdf_ml.index

    edge_search = spatial_join_intersects(gdf_ml, gdf_mh)
    line_mh = edge_search[["index", "index_right"]].values

    grouped = {}
    for line, point in line_mh:
        grouped.setdefault(line, []).append(point)

    connected_mh = []
    for line, points in grouped.items():
        if len(points) >= 2:
            connected_mh.extend(
                [
                    (points[i], points[j])
                    for i in range(len(points))
                    for j in range(i + 1, len(points))
                ]
            )

    raw_graph = Data()
    raw_graph.node_id = torch.arange(len(gdf_ap))
    raw_graph.edge_index = torch.from_numpy(np.array(connected_mh).astype(int)).T
    raw_g = to_networkx(raw_graph).to_undirected()
    components = sorted(nx.connected_components(raw_g), key=len)

    lst_d = [element for set_ in components for element in set_]
    gdf_ap = gdf_ap.iloc[lst_d].reset_index(drop=True)

    gdf_ap["geo"] = gdf_ap.geometry
    gdf_ml["index"] = gdf_ml.index
    edge_search = spatial_join_intersects(gdf_ml, gdf_ap)
    line_mh = edge_search[["index", "index_right"]].values

    grouped = {}
    for line, point in line_mh:
        grouped.setdefault(line, []).append(point)

    connected_mh = []
    for line, points in grouped.items():
        if len(points) >= 2:
            connected_mh.extend(
                [
                    (points[i], points[j], line)
                    for i in range(len(points))
                    for j in range(i + 1, len(points))
                ]
            )

    connected_mh_array = np.array(connected_mh)
    gdf_ml = gdf_ml.iloc[connected_mh_array[:, 2].astype(int)].copy()
    gdf_ml[["MH1_index", "MH2_index"]] = connected_mh_array[:, 0:2].astype(int)
    gdf_ml = gdf_ml.reset_index(drop=True)

    raw_graph = Data()
    raw_graph.node_id = torch.arange(len(gdf_ap))
    raw_graph.edge_index = torch.from_numpy(connected_mh_array[:, :2].astype(int)).T
    raw_g = to_networkx(raw_graph).to_undirected()
    components = list(nx.connected_components(raw_g))

    d_train, d_val, d_test = split_list_by_ratio(components, (6, 2, 2), seed=seed)
    split_mask = {"train": d_train, "val": d_val, "test": d_test}

    gdf_road["FID"] = gdf_road.index + 1
    gdf_road = gdf_road[gdf_road["FID"].isin(gdf_ap["NEAR_FID"].values)].reset_index(
        drop=True
    )

    angles, angle_bin, idx = calculate_angles(gdf_road)

    gdf_road_filtered = gdf_road.iloc[idx].reset_index(drop=True).copy()
    gdf_road_filtered["angle"] = angles
    gdf_road_filtered["angle_bin"] = angle_bin
    gdf_road_filtered["len_bins"], _ = pd.qcut(
        gdf_road_filtered["Shape_Leng"], q=5, labels=False, retbins=True
    )
    gdf_road_filtered["centroid"] = (
        gdf_road_filtered.geometry.to_crs("9295").centroid.to_crs("4326")
    )

    merged_df = pd.merge(
        gdf_road_filtered.reset_index(),
        gdf_ap.reset_index(),
        how="inner",
        left_on="FID",
        right_on="NEAR_FID",
    )
    merged_df = merged_df[merged_df["NEAR_DIST"] < 100]
    merged_df = merged_df.rename(
        columns={
            "index_x": "index_Road",
            "index_y": "index_MH",
        }
    )

    df_mh_r_edge = merged_df[
        ["index_MH", "index_Road", "NEAR_POS", "NEAR_DIST", "SIDE"]
    ].copy()
    df_mh_r_edge["dist_bins"], _ = pd.qcut(
        df_mh_r_edge["NEAR_DIST"], q=5, labels=False, retbins=True
    )
    del df_mh_r_edge["NEAR_DIST"]

    gdf_ap[["x_coordinate", "y_coordinate"]] = gdf_ap.geometry.apply(
        extract_coordinates
    )
    gdf_road_filtered[["x_coordinate", "y_coordinate"]] = gdf_road_filtered[
        "centroid"
    ].apply(extract_coordinates)
    del gdf_road_filtered["centroid"]

    edge_search = spatial_join_intersects(gdf_road_filtered, gdf_road_filtered)
    edges = np.array(
        [edge_search[["index_right"]].values.reshape(-1), np.array(edge_search.index)]
    )
    edges = edges[:, np.where(edges[0] != edges[1])[0]]

    dump_pickle(gdf_ap, output_dir, "MH_proc.pkl")
    dump_pickle(gdf_road_filtered, output_dir, "Road_proc.pkl")
    dump_pickle(gdf_ml, output_dir, "Line_proc.pkl")
    dump_pickle(df_mh_r_edge, output_dir, "MH_R_RL_proc.pkl")
    dump_pickle(edges, output_dir, "R_R_proc.pkl")
    dump_pickle(split_mask, output_dir, "split_mask.pkl")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess raw sewer and road GIS artifacts into model pickles."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw"),
        help=(
            "Raw data root containing gis/sewer, gis/roads, and "
            "mh_road/MH_Road.pkl."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=INTERIM_DATA_DIR,
        help="Directory for generated *_proc.pkl and split_mask.pkl artifacts.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    build_preprocessed_data(
        raw_root=args.raw_root.resolve(),
        output_dir=args.output_dir.resolve(),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
