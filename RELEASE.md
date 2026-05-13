# Release Process

Maintainer playbook for cutting a formal tagged release.

## Versioning

The project follows [Semantic Versioning](https://semver.org/):

| Bump | Trigger |
| --- | --- |
| `MAJOR` (`1.0.0` → `2.0.0`) | Breaking change to CLI flags, data layout, or model state-dict format |
| `MINOR` (`1.0.0` → `1.1.0`) | New checkpoints, new replication targets, additive CLI flags |
| `PATCH` (`1.0.0` → `1.0.1`) | Docs, lint, dependency updates that do not change results |

Three places must stay in sync:

1. [pyproject.toml](pyproject.toml) `version`
2. [pipe_network_completion/\_\_init\_\_.py](pipe_network_completion/__init__.py) `__version__`
3. [CITATION.cff](CITATION.cff) `version`

`tests/test_version_consistency.py` enforces this.

## Pre-release Checklist

```bash
# 1. Working tree is clean
git status

# 2. Tests + lint pass
pip install -e ".[test]"
pytest
ruff check pipe_network_completion scripts process.py tests

# 3. End-to-end replication smoke test
python scripts/verify_environment.py --load-data
python scripts/replicate_results.py --max-batches 2

# 4. CHANGELOG.md has an entry for the new version (move [Unreleased] -> [X.Y.Z])
```

## Cut The Release

```bash
# 1. Bump version in pyproject.toml, __init__.py, CITATION.cff
# 2. Commit the bump
git add pyproject.toml pipe_network_completion/__init__.py CITATION.cff CHANGELOG.md
git commit -m "Release v1.0.0"

# 3. Tag
git tag -a v1.0.0 -m "v1.0.0 - ISARC 2024 replication"
git push origin Final
git push origin v1.0.0

# 4. Bundle large assets
python scripts/bundle_release_assets.py --version v1.0.0 --format zip

# 5. Create the GitHub release with notes from CHANGELOG.md
gh release create v1.0.0 \
  --title "v1.0.0 - ISARC 2024 replication" \
  --notes-file CHANGELOG.md \
  release_assets/pipe-network-artifacts-v1.0.0.zip \
  release_assets/pipe-network-artifacts-v1.0.0.zip.sha256
```

`bundle_release_assets.py` writes a SHA-256 sidecar so users can verify
downloads end to end. Add `--include raw` only if the raw GIS data can be
redistributed publicly.

## Zenodo Archival (One-Time Setup)

For a citable DOI alongside the paper:

1. Log in to [zenodo.org](https://zenodo.org) and link your GitHub account.
2. Flip the toggle for `Yuxi0048/PipeNetworkCompletion` in the Zenodo
   GitHub-integration page.
3. Cut a release on GitHub (as above). Zenodo auto-archives it and mints a
   DOI within a few minutes.
4. Add the Zenodo DOI badge to the top of [README.md](README.md) once the
   DOI is allocated.

After the first release the integration is automatic for every subsequent
tag.
