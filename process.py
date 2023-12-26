from torch_geometric.data import Data
import pandas as pd
import numpy as np
import torch
import pickle
from torch_geometric.utils import to_networkx
import geopandas as gpd
import networkx as nx
import random
# load shape files from arcgis
gdf_MH = gpd.read_file("/content/drive/MyDrive/0Network_PipeLine_Predcition/Sewer_shp/SewerManholes_ExportFeatures.shp")
gdf_ML2 = gpd.read_file("/content/drive/MyDrive/0Network_PipeLine_Predcition/Sewer_shp/SewerGravityMa_ExportFeature2.shp")
gdf_ML2 = gdf_ML2.drop_duplicates()
gdf_ML1 = gpd.read_file("/content/drive/MyDrive/0Network_PipeLine_Predcition/Sewer_shp/SewerGravityMa_ExportFeature1.shp")
gdf_Pump = gpd.read_file("/content/drive/MyDrive/0Network_PipeLine_Predcition/Sewer_shp/SewersqlSewerP_ExportFeature.shp")
gdf_Road = gpd.read_file("/content/drive/MyDrive/0Network_PipeLine_Predcition/Roads_shp/Roads_ExportFeatures.shp")
gdf_ML = gdf_ML1.append(gdf_ML2).reset_index(drop=True)

with open('/content/drive/MyDrive/0Network_PipeLine_Predcition/MH_Road.pkl', 'rb') as file:
    MH_Road_list = pickle.load(file)

df_MH_Road = pd.DataFrame(data = np.array(MH_Road_list), columns = ['OBJECTID', 'NEAR_FID','NEAR_POS','NEAR_DIST','SIDE'])
gdf_MH = gdf_MH.merge(df_MH_Road, left_index=True, right_index=True, how='inner')

# Append pumps points into the MH dataset 
gdf_MH['SUBTYPECD'] = gdf_MH['SUBTYPECD'].astype('str')+ gdf_MH['MANHOLEUSE']
gdf_AP = gdf_MH.append(gdf_Pump)
gdf_AP=gdf_AP.reset_index(drop=True).dropna()

# Calculate MH-MH edge by checking the intersection between MH and Main Sewer Lines
gdf_AP['geo'] = gdf_AP.geometry
gdf_ML['index'] = gdf_ML.index

edge_search = gpd.sjoin(gdf_ML, gdf_MH, how = 'left', op='intersects').dropna()
L_MH = edge_search[['index','index_right']].values

# Group connections by line index (first column)
grouped = {}
for line, point in L_MH:
    if line not in grouped:
        grouped[line] = []
    grouped[line].append(point)

# Extract pairs of connected points based on line index
connected_MH = []
for line, points in grouped.items():
    if len(points) >= 2:
        connected_MH.extend([(points[i], points[j]) for i in range(len(points)) for j in range(i + 1, len(points))])

# create a networkx graph and extract the biggest graph, them store the original index and reindex the MH dataset --> 198351 MHs
raw_graph = Data()
raw_graph.node_id = torch.arange(len(gdf_AP))
raw_graph.edge_index = torch.from_numpy(np.array(connected_MH).astype(int)).T
raw_g = to_networkx(raw_graph).to_undirected()
d = list(nx.connected_components(raw_g))

# select those graphs greater than or equal to 100
d = sorted(d, key=len)
lst_d = [element for set_ in d for element in set_]
gdf_AP = gdf_AP.iloc[lst_d]
gdf_AP = gdf_AP.reset_index(drop=True)

# Recalculate MH-MH edge by checking the intersection between MH and Main Sewer Lines
gdf_AP['geo'] = gdf_AP.geometry
gdf_ML['index'] = gdf_ML.index
edge_search = gpd.sjoin(gdf_ML, gdf_AP, how = 'left', op='intersects').dropna()
L_MH = edge_search[['index','index_right']].values

# Group connections by line index (first column)
grouped = {}
for line, point in L_MH:
    if line not in grouped:
        grouped[line] = []
    grouped[line].append(point)

# Extract pairs of connected points based on line index
connected_MH = []
for line, points in grouped.items():
    if len(points) >= 2:
        connected_MH.extend([(points[i], points[j], line) for i in range(len(points)) for j in range(i + 1, len(points))])

gdf_ML = gdf_ML.iloc[np.array(connected_MH)[:,2].astype(int)]
gdf_ML[['MH1_index','MH2_index']] = np.array(connected_MH)[:,0:2].astype(int)
gdf_ML = gdf_ML.reset_index(drop=True)

# create a networkx graph
raw_graph = Data()
raw_graph.node_id = torch.arange(len(gdf_AP))
raw_graph.edge_index = torch.from_numpy(np.array(connected_MH).astype(int)[:,:2]).T
raw_g = to_networkx(raw_graph).to_undirected()
d = list(nx.connected_components(raw_g))


# split graph datasets
def split_list_by_ratio(lst, ratio):
    # Shuffle the list
    random.shuffle(lst)

    total_len = len(lst)
    ratio_sum = sum(ratio)

    # Calculate the split indices directly
    split1_end = total_len * ratio[0] // ratio_sum
    split2_end = split1_end + total_len * ratio[1] // ratio_sum

    # Split the list using calculated indices
    split1 = lst[:split1_end]
    split2 = lst[split1_end:split2_end]
    split3 = lst[split2_end:]

    return split1, split2, split3


# Split the list in 8:1:1 ratio
d_train, d_val, d_test = split_list_by_ratio(d, (6, 2, 2))
split_mask = {
    'train': d_train,
    'val': d_val,
    'test' :d_test
              }
lst_d = [element for set_ in d for element in set_]

# Exclude those roads not near MH
gdf_Road['FID'] = gdf_Road.index + 1
gdf_Road = gdf_Road[gdf_Road['FID'].isin(gdf_AP['NEAR_FID'].values)].reset_index(drop=True)

# process the road data
def angle_between(p1, p2):
    ang = np.arctan2(p2.x - p1.x, p2.y - p1.y)
    return np.rad2deg(ang % np.pi)

def angle2bin(angle):
    if angle < 15 or angle >= 165:
        angle_bin = 0
    else:
        angle_bin=int((angle - 15) / 30) + 1
    return angle_bin

def calculate_angles(gdf_Road):
    angles = []
    idx = []
    angle_bin = []

    for i in range(len(gdf_Road)):
        if gdf_Road['geometry'][i].boundary.is_empty:
            pass
        else:
            p1 = gdf_Road['geometry'][i].boundary.geoms[0]
            p2 = gdf_Road['geometry'][i].boundary.geoms[1]
            angle = angle_between(p1, p2)
            angles.append(angle)
            angle_bin.append(angle2bin(angle))
            idx.append(i)

    return angles, angle_bin, idx

# Usage:
angles, angle_bin, idx = calculate_angles(gdf_Road)

gdf_Road_filtered = gdf_Road.iloc[idx].reset_index(drop=True)
gdf_Road_filtered['angle'] = angles
gdf_Road_filtered['angle_bin'] = angle_bin
gdf_Road_filtered['len_bins'], edges = pd.qcut(gdf_Road_filtered['Shape_Leng'], q=5, labels=False, retbins=True)
gdf_Road_filtered['centroid'] = gdf_Road_filtered.geometry.to_crs('9295').centroid.to_crs('4326')

# calculate MH-Road edge
# Merge DataFrames A and B based on the common columns 'FID' and 'NEAR_FID'
merged_df = pd.merge(gdf_Road_filtered.reset_index(), gdf_AP.reset_index(), how='inner', left_on='FID', right_on='NEAR_FID')
merged_df = merged_df[merged_df['NEAR_DIST']<100]
# Rename columns
merged_df =merged_df.rename(columns={
    'index_x': 'index_Road',
    'index_y': 'index_MH',
})
# Extract pairs of original indexes based on the merged DataFrame
df_MH_R_edge = merged_df[['index_MH','index_Road', 'NEAR_POS', 'NEAR_DIST', 'SIDE']]
df_MH_R_edge['dist_bins'], edges = pd.qcut(df_MH_R_edge['NEAR_DIST'], q=5, labels=False, retbins=True)
del df_MH_R_edge['NEAR_DIST']

# add spatial coordinates
# Function to extract x and y coordinates
def extract_coordinates(point):
    return pd.Series([point.x, point.y])

# Apply function to create separate columns for x and y coordinates
gdf_AP[['x_coordinate', 'y_coordinate']] = gdf_AP.geometry.apply(extract_coordinates)
gdf_Road_filtered[['x_coordinate', 'y_coordinate']] = gdf_Road_filtered['centroid'].apply(extract_coordinates)
del gdf_Road_filtered['centroid']
# Calculate Road-Road edge by checking the intersection between Road and Road Lines
edge_search = gpd.sjoin(gdf_Road_filtered, gdf_Road_filtered, how = 'left', op='intersects').dropna()
edges = np.array([edge_search[['index_right']].values.reshape(-1), np.array(edge_search.index)])
# Delete same road matches
edges = edges[:, np.where(edges[0] != edges[1])[0]]

with open('/content/drive/MyDrive/0Network_PipeLine_Predcition/MH_proc.pkl', 'wb') as file:
    pickle.dump(gdf_AP, file)

with open('/content/drive/MyDrive/0Network_PipeLine_Predcition/Road_proc.pkl', 'wb') as file:
    pickle.dump(gdf_Road_filtered, file)

with open('/content/drive/MyDrive/0Network_PipeLine_Predcition/Line_proc.pkl', 'wb') as file:
    pickle.dump(gdf_ML, file)

with open('/content/drive/MyDrive/0Network_PipeLine_Predcition/MH_R_RL_proc.pkl', 'wb') as file:
    pickle.dump(df_MH_R_edge, file)

# with open('/content/drive/MyDrive/0Network_PipeLine_Predcition/MH_MH_proc.pkl', 'wb') as file:
#     pickle.dump(connected_MH, file) 

with open('/content/drive/MyDrive/0Network_PipeLine_Predcition/R_R_proc.pkl', 'wb') as file:
    pickle.dump(edges, file)  

with open('/content/drive/MyDrive/0Network_PipeLine_Predcition/split_mask.pkl', 'wb') as file:
    pickle.dump(split_mask, file)       