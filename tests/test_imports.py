"""Confirm the runnable package modules import cleanly.

These tests do not touch GPU, PyG samplers, or large pickles.
"""

from __future__ import annotations

import importlib

import pytest


PACKAGE_MODULES = [
    "pipe_network_completion",
    "pipe_network_completion.paths",
    "pipe_network_completion.dataset",
    "pipe_network_completion.evaluation",
    "pipe_network_completion.location_encoder",
    "pipe_network_completion.model",
]


@pytest.mark.parametrize("module_name", PACKAGE_MODULES)
def test_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)


def test_package_version_exposed() -> None:
    package = importlib.import_module("pipe_network_completion")
    assert isinstance(package.__version__, str)
    assert package.__version__
