# Release Process

Maintainer playbook for cutting a formal tagged release.

## Versioning

The project follows [Semantic Versioning](https://semver.org/):

| Bump | Trigger |
| --- | --- |
| `MAJOR` (`1.0.0` → `2.0.0`) | Breaking change to CLI flags, data layout, or model state-dict format |
| `MINOR` (`1.0.0` -> `1.1.0`) | New checkpoints, new evaluation targets, additive CLI flags |
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

# 3. End-to-end evaluation smoke test, when authorized artifacts are available
python scripts/verify_environment.py --load-data
python scripts/evaluate_checkpoint.py --max-batches 2

# 4. CHANGELOG.md has an entry for the new version (move [Unreleased] -> [X.Y.Z])
```

## Cut The Release

```bash
# 1. Bump version in pyproject.toml, __init__.py, CITATION.cff
# 2. Commit the bump
git add pyproject.toml pipe_network_completion/__init__.py CITATION.cff CHANGELOG.md
git commit -m "Release v1.0.0"

# 3. Tag
git tag -a v1.0.0 -m "v1.0.0 - ISARC 2024 source release"
git push origin main
git push origin v1.0.0

# 4. Create the GitHub source release with notes from CHANGELOG.md
gh release create v1.0.0 \
  --title "v1.0.0 - ISARC 2024 source release" \
  --notes-file CHANGELOG.md
```

Do not attach raw GIS files, processed graph pickles, model checkpoints, or
other utility-network artifacts to a public GitHub release unless redistribution
permission is documented.

Create a private artifact archive only after written redistribution permission
or an equivalent provider-compliant agreement is in place:

```bash
python scripts/bundle_release_assets.py --version v1.0.0 --format zip --confirm-authorized-distribution
```

`bundle_release_assets.py` writes a SHA-256 sidecar so recipients can verify
downloads end to end.

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
