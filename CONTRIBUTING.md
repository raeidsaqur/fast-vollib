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

# Install all dev dependencies (requires Python >=3.11 and uv)
uv sync --all-groups

# Run the test suite
uv run pytest tests/ -v

# Lint and format
uv run ruff check . --fix
uv run ruff format .

# Type-check
uv run mypy src/fast_vollib --ignore-missing-imports
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

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE) that covers this project.
