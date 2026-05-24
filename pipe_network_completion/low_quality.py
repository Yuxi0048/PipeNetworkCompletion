from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import HeteroData


DEFAULT_NODE_TYPES = ("MH", "Road")


@dataclass(frozen=True)
class LowQualityAblationReport:
    """Summary of a location-only graph ablation."""

    node_feature_dims_before: dict[str, int]
    node_feature_dims_after: dict[str, int]
    zeroed_node_attribute_dims: dict[str, int]
    zeroed_edge_attribute_dims: dict[str, int]


def apply_location_only_ablation(
    data: HeteroData,
    *,
    keep_location_dims: int = 2,
    node_types: tuple[str, ...] = DEFAULT_NODE_TYPES,
    zero_edge_attrs: bool = False,
    clone: bool = True,
) -> tuple[HeteroData, LowQualityAblationReport]:
    """Keep node locations and remove other node attributes.

    The ISARC model treats the first two columns of ``MH.x`` and ``Road.x`` as
    coordinates. To evaluate a low-quality-data setting while preserving saved
    checkpoint tensor shapes, this transform keeps those columns unchanged and
    fills all remaining node feature columns with zeros.

    Edge attributes are not node attributes, so they are retained by default.
    Set ``zero_edge_attrs=True`` for a stricter context-poor ablation.
    """

    if keep_location_dims < 1:
        raise ValueError("keep_location_dims must be at least 1.")

    transformed = data.clone() if clone else data
    dims_before: dict[str, int] = {}
    dims_after: dict[str, int] = {}
    zeroed_node_dims: dict[str, int] = {}

    for node_type in node_types:
        if node_type not in transformed.node_types:
            continue
        x = transformed[node_type].x
        if x is None:
            raise ValueError(f"Node type {node_type!r} has no x tensor.")
        if x.ndim != 2:
            raise ValueError(f"Node type {node_type!r} x tensor must be 2-D.")
        if x.shape[1] < keep_location_dims:
            raise ValueError(
                f"Node type {node_type!r} has only {x.shape[1]} features; "
                f"cannot keep {keep_location_dims} location dimensions."
            )
        dims_before[node_type] = int(x.shape[1])
        if x.shape[1] > keep_location_dims:
            x = x.clone()
            x[:, keep_location_dims:] = 0
            transformed[node_type].x = x
        dims_after[node_type] = int(transformed[node_type].x.shape[1])
        zeroed_node_dims[node_type] = max(0, dims_before[node_type] - keep_location_dims)

    zeroed_edge_dims: dict[str, int] = {}
    if zero_edge_attrs:
        for edge_type in transformed.edge_types:
            store = transformed[edge_type]
            if "edge_attr" not in store or store.edge_attr is None:
                continue
            edge_attr = store.edge_attr
            if not torch.is_tensor(edge_attr):
                continue
            store.edge_attr = torch.zeros_like(edge_attr)
            zeroed_edge_dims["|".join(edge_type)] = int(edge_attr.shape[-1])

    return transformed, LowQualityAblationReport(
        node_feature_dims_before=dims_before,
        node_feature_dims_after=dims_after,
        zeroed_node_attribute_dims=zeroed_node_dims,
        zeroed_edge_attribute_dims=zeroed_edge_dims,
    )
