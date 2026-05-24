from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from pipe_network_completion.low_quality import apply_location_only_ablation


def _toy_isarc_graph() -> HeteroData:
    data = HeteroData()
    data["MH"].x = torch.tensor(
        [
            [153.0, -27.0, 1.0, 0.0],
            [153.1, -27.1, 0.0, 1.0],
        ]
    )
    data["Road"].x = torch.tensor(
        [
            [153.0, -27.2, 2.0, 3.0, 4.0],
            [153.2, -27.3, 5.0, 6.0, 7.0],
        ]
    )
    data["MH", "near", "Road"].edge_index = torch.tensor([[0, 1], [0, 1]])
    data["MH", "near", "Road"].edge_attr = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    data["Road", "rev_near", "MH"].edge_index = torch.tensor([[0, 1], [0, 1]])
    data["Road", "rev_near", "MH"].edge_attr = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    data["MH", "link", "MH"].edge_index = torch.tensor([[0], [1]])
    data["MH", "link", "MH"].edge_label_index = torch.tensor([[0], [1]])
    data["MH", "link", "MH"].edge_label = torch.tensor([1.0])
    return data


def test_location_only_ablation_preserves_locations_and_shape() -> None:
    original = _toy_isarc_graph()
    transformed, report = apply_location_only_ablation(original)

    assert transformed["MH"].x.shape == original["MH"].x.shape
    assert transformed["Road"].x.shape == original["Road"].x.shape
    torch.testing.assert_close(transformed["MH"].x[:, :2], original["MH"].x[:, :2])
    torch.testing.assert_close(transformed["Road"].x[:, :2], original["Road"].x[:, :2])
    assert torch.count_nonzero(transformed["MH"].x[:, 2:]) == 0
    assert torch.count_nonzero(transformed["Road"].x[:, 2:]) == 0
    torch.testing.assert_close(
        transformed["MH", "near", "Road"].edge_attr,
        original["MH", "near", "Road"].edge_attr,
    )
    assert report.zeroed_node_attribute_dims == {"MH": 2, "Road": 3}


def test_location_only_ablation_can_zero_edge_attributes() -> None:
    transformed, report = apply_location_only_ablation(
        _toy_isarc_graph(),
        zero_edge_attrs=True,
    )

    assert torch.count_nonzero(transformed["MH", "near", "Road"].edge_attr) == 0
    assert torch.count_nonzero(transformed["Road", "rev_near", "MH"].edge_attr) == 0
    assert report.zeroed_edge_attribute_dims == {
        "MH|near|Road": 2,
        "Road|rev_near|MH": 2,
    }
