from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.utils import negative_sampling


MH_EDGE_TYPE = ("MH", "link", "MH")


def split_2d_array(array, indices):
    index_set = set(indices)
    included_pairs = []
    line_index = []

    for i, pair in enumerate(array):
        if pair[0] in index_set or pair[1] in index_set:
            included_pairs.append(pair)
            line_index.append(i)

    return np.array(line_index), np.array(included_pairs)


def splitdata(line, mh, road, mh_r_rl, r_r_rl, split_set):
    split_set = set(split_set)
    array_2d = line[["MH1_index", "MH2_index"]].values
    new_line_index, _ = split_2d_array(array_2d, split_set)
    new_line = line.iloc[new_line_index].reset_index(drop=True).copy()
    new_mh = mh.iloc[sorted(split_set)].copy()
    new_mh_r_rl = mh_r_rl[mh_r_rl["index_MH"].isin(split_set)].copy()
    new_road = road[road.index.isin(set(new_mh_r_rl["index_Road"].values))].copy()
    new_r_r_rl = pd.DataFrame(
        data=r_r_rl[:, np.isin(r_r_rl, list(new_road.index)).all(axis=0)].T,
        columns=["ind1", "ind2"],
    )
    new_mh["old"] = new_mh.index
    new_mh = new_mh.reset_index()
    mapping_mh = pd.Series(new_mh.index, index=new_mh["old"]).to_dict()
    new_road["old"] = new_road.index
    new_road = new_road.reset_index()
    mapping_road = pd.Series(new_road.index, index=new_road["old"]).to_dict()
    new_mh_r_rl["index_MH"] = new_mh_r_rl["index_MH"].map(mapping_mh)
    new_line["MH1_index"] = new_line["MH1_index"].map(mapping_mh)
    new_line["MH2_index"] = new_line["MH2_index"].map(mapping_mh)
    new_mh_r_rl["index_Road"] = new_mh_r_rl["index_Road"].map(mapping_road)
    new_r_r_rl["ind1"] = new_r_r_rl["ind1"].map(mapping_road)
    new_r_r_rl["ind2"] = new_r_r_rl["ind2"].map(mapping_road)
    del new_mh["old"]
    del new_road["old"]
    return new_line, new_line_index, new_mh, new_road, new_mh_r_rl, new_r_r_rl


def data_transform(data, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    new_data = data.clone()
    pos_edge_sample = data[MH_EDGE_TYPE].edge_index
    neg_edge_sample = negative_sampling(pos_edge_sample)
    pos_edge_label = torch.from_numpy(np.ones(pos_edge_sample.shape[1]))
    neg_edge_label = torch.from_numpy(np.zeros(neg_edge_sample.shape[1]))
    new_data[MH_EDGE_TYPE].edge_index = torch.concat(
        [pos_edge_sample, neg_edge_sample], axis=1
    )
    new_data[MH_EDGE_TYPE].edge_label_index = torch.concat(
        [pos_edge_sample, neg_edge_sample], axis=1
    )
    new_data[MH_EDGE_TYPE].edge_label = torch.concat([pos_edge_label, neg_edge_label])
    return new_data


def dataset(mh, line, road, mh_r_rl, r_r_rl, split_set, seed=None):
    encoded_mh = pd.get_dummies(mh, columns=["SUBTYPECD"])
    encoded_ml = pd.get_dummies(line, columns=["SEGMENTTYP", "MATERIAL"])
    encoded_r = pd.get_dummies(road, columns=["OVL2_CAT", "angle_bin", "len_bins"])
    encoded_mh_r = pd.get_dummies(mh_r_rl, columns=["dist_bins"])
    encoded_ml, line_index, encoded_mh, encoded_r, encoded_mh_r, r_r_rl = splitdata(
        encoded_ml, encoded_mh, encoded_r, encoded_mh_r, r_r_rl, split_set
    )
    # Column-offset slicing matches the notebook layout exactly. The leading
    # columns are identifiers/geometry that the model does not consume; the
    # remaining columns are the one-hot feature block.
    # encoded_r:    cols 0-6  = id/geometry,  cols 7+  = one-hot road features
    # encoded_mh:   cols 0-10 = id/geometry,  cols 11+ = one-hot MH features
    # encoded_mh_r: cols 0-1  = (index_MH, index_Road), cols 2+ = edge features
    road_features = torch.from_numpy(encoded_r[encoded_r.columns[7:]].values)
    mh_features = torch.from_numpy(encoded_mh[encoded_mh.columns[11:]].values)
    mh_road_edge_features = torch.from_numpy(
        encoded_mh_r[encoded_mh_r.columns[2:]].values
    )
    mh_mh_edge_idx = torch.from_numpy(encoded_ml[["MH1_index", "MH2_index"]].values.T)
    mh_road_edge_idx = torch.from_numpy(
        encoded_mh_r[encoded_mh_r.columns[:2]].values.astype("int")
    ).T
    road_road_edge_idx = torch.from_numpy(r_r_rl.values.T.astype("int"))

    data = HeteroData()
    data["MH"].node_id = torch.arange(len(encoded_mh))
    data["MH"].x = mh_features.float()
    data["Road"].node_id = torch.arange(len(encoded_r))
    data["Road"].x = road_features.float()
    data["Road", "link", "Road"].edge_index = road_road_edge_idx
    data["MH", "near", "Road"].edge_index = mh_road_edge_idx
    data["MH", "near", "Road"].edge_attr = mh_road_edge_features
    data = T.ToUndirected()(data)
    data[MH_EDGE_TYPE].edge_index = mh_mh_edge_idx
    data = data_transform(data, seed=seed)
    return data, line_index

