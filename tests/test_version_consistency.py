"""Lock the three version sources together so a release bump cannot drift.

If this test fails, the release-bump checklist in RELEASE.md was not fully
followed -- the version was updated in one place but not the others.
"""

from __future__ import annotations

import re
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

from pipe_network_completion import __version__ as package_version
from pipe_network_completion import paths


def test_pyproject_matches_package_version() -> None:
    with (paths.REPO_ROOT / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    assert data["project"]["version"] == package_version


def test_citation_cff_matches_package_version() -> None:
    citation = (paths.REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r"^version:\s*\"?([^\"\n]+)\"?", citation, flags=re.MULTILINE)
    assert match is not None, "CITATION.cff has no 'version' field"
    assert match.group(1).strip() == package_version
