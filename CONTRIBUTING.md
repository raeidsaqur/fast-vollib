# Contributing to fast-vollib

Thank you for your interest in contributing! This document outlines the
process for reporting bugs, proposing features, and submitting pull requests.

---

## Reporting issues

Please open a [GitHub issue](https://github.com/raeidsaqur/fast-vollib/issues)
and include:

- A minimal reproducible example
- Your Python version and platform
- The installed versions of `fast-vollib` and relevant extras (`torch`, `jax`)

---

## Development setup

```bash
git clone https://github.com/raeidsaqur/fast-vollib.git
cd fast-vollib

# Install all dev dependencies (requires Python >=3.10 and uv)
uv sync --all-groups

# Run the test suite
uv run pytest tests/ -v

# Optional backends normally skip when absent. Name the ones this run must
# actually exercise and a missing or skipped backend fails instead.
uv sync --all-groups --extra torch --extra jax
FV_REQUIRE_BACKENDS=torch,jax uv run pytest tests/ -q

# Lint and format
uv run ruff check . --fix
uv run ruff format .

# Type-check against the repository's accepted diagnostic baseline
uv run python scripts/check_mypy_baseline.py
```

---

## Pull request guidelines

1. **Open an issue first** for any non-trivial change so we can discuss the
   approach before you invest time implementing it.
2. **Keep PRs focused** — one logical change per pull request.
3. **Add or update tests** for every bug fix and new feature.
4. **Ensure all checks pass** (ruff lint, ruff format, pytest) before
   requesting review.
5. **Update the changelog** (`docs/changelog.md`) under `[Unreleased]`.

---

## Code style

- Formatter: `ruff format` (Black-compatible, line length 100)
- Linter: `ruff check` (E/F/I rules; see `ruff.toml` for full config)
- Type hints: encouraged throughout; the package ships `py.typed`

---

## Release procedure

The Git tag identifies the source commit; the GitHub Release supplies the
announcement and downloads for that tag. The `Release` workflow publishes to
PyPI on a pushed `v*` tag, not on publication of a GitHub Release. Hatch VCS
derives the distribution version from the tag: `v0.2.3` produces `0.2.3`.

1. Prepare `release/vX.Y.Z` from up-to-date `main`. Promote the changelog's
   unreleased changes to the planned version and date, update `CITATION.cff`
   and the README release summary, and prepare public release notes. Preserve
   the concept DOI; do not invent a version-specific DOI or edit the generated
   `src/fast_vollib/_version.py`.
2. Run lint, formatting, the type-check baseline, tests, schema and reference
   checks, a strict documentation build, and distribution metadata checks.
   The untagged branch still has a VCS-derived development version.
3. Submit the release branch for review and merge it into `main`. If publication
   moves to another day, update the release date and affected documentation
   links before merging. Verify CI on the intended merged commit.
4. Create an annotated `vX.Y.Z` tag on that verified commit in `main`, then push
   **only that tag**. Check that the `Release` workflow builds the intended
   version and successfully publishes both distributions to PyPI. A `main`
   push publishes to TestPyPI separately; that is not the stable release.
5. Create the GitHub Release using the **existing** `vX.Y.Z` tag, a matching
   version in the title, and the reviewed notes. With the GitHub CLI, use
   `gh release create vX.Y.Z --verify-tag --title "vX.Y.Z — <summary>"`
   together with `--notes-file <notes-file>`. The `--verify-tag` option prevents
   accidental creation of a tag at a different commit. If attaching wheels or
   sdists, use the distributions from the successful tag workflow, not a
   separate local rebuild.

Do not move a published tag to fix a release, and do not use the production
workflow's manual dispatch on an untagged branch. Keep the version, tag target,
and published distributions consistent; fixes after publication need a new
version. Creating the GitHub Release after PyPI succeeds makes the announcement
accurate, but the two publications are not an atomic operation.

---

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE) that covers this project.
