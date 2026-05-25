import numpy as np
import torch

from pipe_network_completion.anchor_free.unet_skeleton_hybrid import (
    AoiRasterMosaic,
    build_skeleton_graph_from_mosaic,
    zhang_suen_thinning,
)


def test_zhang_suen_thinning_keeps_connected_line():
    mask = np.zeros((9, 9), dtype=bool)
    mask[3:6, 1:8] = True

    skeleton = zhang_suen_thinning(mask)

    assert skeleton.sum() < mask.sum()
    assert skeleton.sum() > 0
    assert skeleton[3:6, :].any()


def test_skeleton_graph_uses_no_coordinate_features():
    x = np.zeros((2, 16, 16), dtype="float32")
    x[0, 8, 2:14] = 1.0
    cnn_features = np.zeros((3, 16, 16), dtype="float32")
    cnn_features[0, 7:10, 2:14] = 0.25
    cnn_features[1, 7:10, 2:14] = 0.50
    cnn_features[2, 7:10, 2:14] = 0.75
    probability = np.zeros((16, 16), dtype="float32")
    probability[7:10, 2:14] = 0.8
    y = np.zeros((16, 16), dtype="float32")
    y[8, 7:9] = 1.0
    mosaic = AoiRasterMosaic(
        aoi_id="synthetic",
        split="train",
        x=x,
        y=y,
        probability=probability,
        channel_names=["road_line", "building_area"],
        bounds=(0.0, 0.0, 160.0, 160.0),
        pixel_size_m=10.0,
        cnn_features=cnn_features,
        cnn_feature_names=["cnn_decoder_00", "cnn_decoder_01", "cnn_decoder_02"],
    )

    part = build_skeleton_graph_from_mosaic(
        mosaic,
        candidate_threshold=0.2,
        min_component_pixels=1,
        include_cnn_features=True,
    )

    assert part.data.x.shape[1] == len(part.node_feature_names)
    assert part.data.edge_label_attr.shape[1] == len(part.edge_feature_names)
    assert "x" not in part.node_feature_names
    assert "y" not in part.node_feature_names
    assert "cnn_decoder_00" in part.node_feature_names
    assert "edge_mean_cnn_decoder_00" in part.edge_feature_names
    assert torch.unique(part.data.edge_label).numel() == 2
