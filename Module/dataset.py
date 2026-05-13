import numpy as np
import torch
import pandas as pd
from torch_geometric.data import HeteroData
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T
def split_2d_array(array, indices):
    included_pairs = []
    line_index = []

    for i, pair in enumerate(array):
        if pair[0] in indices or pair[1] in indices:
            included_pairs.append(pair)
            line_index.append(i)

    return np.array(line_index), np.array(included_pairs)

def splitdata(Line, MH, Road, MH_R_RL, R_R_RL, split_set):
    array_2d = Line[['MH1_index', 'MH2_index']].values
    new_line_index, new_edge_index = split_2d_array(array_2d, split_set)
    new_line = Line.iloc[new_line_index].reset_index()
    new_MH = MH.iloc[list(split_set)]
    new_MH_R_RL = MH_R_RL[MH_R_RL['index_MH'].isin(split_set)]
    new_road = Road[Road.index.isin(set(new_MH_R_RL['index_Road'].values))]
    new_R_R_RL = pd.DataFrame(data = R_R_RL[:,np.isin(R_R_RL, list(new_road.index)).all(axis = 0)].T, columns = ['ind1', 'ind2'])
    new_MH['old'] = new_MH.index
    new_MH = new_MH.reset_index()
    mapping_MH = pd.Series(new_MH.index, index=new_MH['old']).to_dict()
    new_road['old'] = new_road.index
    new_road = new_road.reset_index()
    mapping_road = pd.Series(new_road.index, index=new_road['old']).to_dict()
    new_MH_R_RL['index_MH'] = new_MH_R_RL['index_MH'].map(mapping_MH)
    new_line['MH1_index']= new_line['MH1_index'].map(mapping_MH)
    new_line['MH2_index']= new_line['MH2_index'].map(mapping_MH)
    new_MH_R_RL['index_Road'] = new_MH_R_RL['index_Road'].map(mapping_road)
    new_R_R_RL['ind1'] = new_R_R_RL['ind1'].map(mapping_road)
    new_R_R_RL['ind2'] = new_R_R_RL['ind2'].map(mapping_road)
    del new_MH['old']
    del new_road['old']
    return new_line, new_line_index, new_MH, new_road, new_MH_R_RL, new_R_R_RL

def data_transform(data):
    new_data = data.clone()
    pos_edge_sample = data['MH','link','MH'].edge_index
    neg_edge_sample = negative_sampling(pos_edge_sample)
    pos_edge_label = torch.from_numpy(np.ones(pos_edge_sample.shape[1]))
    neg_edge_label = torch.from_numpy(np.zeros(neg_edge_sample.shape[1]))
    new_data['MH','link','MH'].edge_index = torch.concat([pos_edge_sample, neg_edge_sample], axis= 1)
    new_data['MH','link','MH'].edge_label_index = torch.concat([pos_edge_sample, neg_edge_sample ], axis= 1)
    new_data['MH','link','MH'].edge_label = torch.concat([pos_edge_label, neg_edge_label ])
    return new_data

def dataset(MH, Line, Road, MH_R_RL, R_R_RL, split_set):
    encoded_MH = pd.get_dummies(MH, columns = ['SUBTYPECD'])
    encoded_ML = pd.get_dummies(Line, columns = ['SEGMENTTYP','MATERIAL'])
    encoded_R = pd.get_dummies(Road, columns = ['OVL2_CAT','angle_bin','len_bins'])
    encoded_MH_R = pd.get_dummies(MH_R_RL, columns = ['dist_bins'])
    encoded_ML, line_index, encoded_MH, encoded_R, encoded_MH_R, R_R_RL  = splitdata(encoded_ML, encoded_MH, encoded_R, encoded_MH_R, R_R_RL, split_set)
    Road_features = torch.from_numpy(encoded_R[encoded_R.columns[7:]].values)
    MH_features = torch.from_numpy(encoded_MH[encoded_MH.columns[11:]].values)
    MH_Road_edge_features = torch.from_numpy(encoded_MH_R[encoded_MH_R.columns[2:]].values)
    MH_MH_edge_idx = torch.from_numpy(encoded_ML[['MH1_index','MH2_index']].values.T)
    MH_Road_edge_idx = torch.from_numpy(encoded_MH_R[encoded_MH_R.columns[:2]].values.astype('int')).T
    Road_Road_edge_idx = torch.from_numpy(R_R_RL.values.T.astype('int'))

    data = HeteroData()
    data['MH'].node_id = torch.arange(len(encoded_MH))
    data['MH'].x = MH_features.float()
    data['Road'].node_id = torch.arange(len(encoded_R))
    data['Road'].x = Road_features.float()
    data['Road','link','Road'].edge_index = Road_Road_edge_idx
    data['MH','near','Road'].edge_index = MH_Road_edge_idx
    data['MH','near','Road'].edge_attr = MH_Road_edge_features
    data = T.ToUndirected()(data)
    data['MH','link','MH'].edge_index = MH_MH_edge_idx
    data = data_transform(data)
    return data, line_index