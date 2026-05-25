import importlib.util

import numpy as np
import pytest
import torch


@pytest.mark.skipif(
    importlib.util.find_spec("segmentation_models_pytorch") is None,
    reason="segmentation_models_pytorch is not installed",
)
def test_standard_smp_unet_forward_pass():
    from pipe_network_completion.anchor_free.unet_segmentation import make_unet_model

    model = make_unet_model(
        in_channels=8,
        encoder_name="resnet18",
        encoder_weights=None,
        classes=1,
    )
    output = model(torch.randn(1, 8, 128, 128))
    assert output.shape == (1, 1, 128, 128)


def test_binary_pixel_metrics_handles_single_class():
    from pipe_network_completion.anchor_free.unet_segmentation import binary_pixel_metrics

    y_true = np.zeros(16, dtype="float32")
    y_prob = np.linspace(0.0, 1.0, 16, dtype="float32")
    metrics = binary_pixel_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert np.isnan(metrics["roc_auc"])
