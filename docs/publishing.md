# Publishing and citation

RejuvenationKit releases are built by GitHub Actions and published to PyPI with Trusted
Publishing. The workflow uses short-lived OpenID Connect credentials and requires no stored PyPI
API token.

## One-time maintainer setup

Create a pending Trusted Publisher for the `rejuvenationkit` project on PyPI with these exact
values:

| Field | Value |
|---|---|
| Owner | `weston-wang` |
| Repository | `RejuvenationKit` |
| Workflow filename | `publish.yml` |
| Environment | `pypi` |

Create a GitHub environment named `pypi`. Requiring a reviewer for that environment is strongly
recommended so publishing remains a deliberate action.

Enable the repository in Zenodo before publishing the GitHub release. Zenodo reads
`CITATION.cff` and archives each enabled release; after the first archive, add its concept DOI to
the README and package metadata.

## Release checklist

1. Update the version in `pyproject.toml`, `src/rejuvenationkit/__init__.py`, and `CITATION.cff`.
2. Add a dated changelog entry and update `date-released` in `CITATION.cff`.
3. Run Ruff, formatting, strict mypy, pytest, the documentation build, and `python -m build`.
4. Confirm `python -m twine check dist/*` passes.
5. Commit and push the release preparation.
6. Create a GitHub release whose tag is exactly `v<package-version>`.
7. Approve the protected `pypi` environment deployment, if configured.
8. Verify installation in a clean environment with
   `python -m pip install rejuvenationkit==<package-version>`.
9. Verify the Zenodo archive and update citation links when necessary.

The publishing workflow rejects a GitHub release whose tag does not match the version declared in
`pyproject.toml`. PyPI versions are immutable: never reuse a version after it has been uploaded.

## Installing an unpublished source release

Before a version appears on PyPI, install it directly from its Git tag:

```bash
python -m pip install \
  "rejuvenationkit[visualization] @ git+https://github.com/weston-wang/RejuvenationKit.git@v0.2.0a1"
```

## Citation

Use GitHub's **Cite this repository** control or the metadata in `CITATION.cff`. Once Zenodo has
archived the project, prefer the DOI for the exact software version used in an analysis. Record
the RejuvenationKit version in every exported audit bundle and research report.
