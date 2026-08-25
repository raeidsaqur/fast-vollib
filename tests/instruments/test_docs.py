"""Documentation that claims a capability must be derived from the registry."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = REPO_ROOT / "docs" / "instruments.md"
SCRIPTS = REPO_ROOT / "scripts"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from generate_instrument_docs import BEGIN, END, render_table, splice  # noqa: E402


def test_capability_table_is_up_to_date() -> None:
    """A stale table is a documentation bug the suite should catch, not ship."""
    document = DOC_PATH.read_text(encoding="utf-8")
    assert document == splice(document, render_table()), (
        "docs/instruments.md is stale; run `uv run python scripts/generate_instrument_docs.py`"
    )


def test_generated_block_is_delimited() -> None:
    document = DOC_PATH.read_text(encoding="utf-8")
    assert document.count(BEGIN) == 1
    assert document.count(END) == 1


def test_table_lists_every_registered_type() -> None:
    from fast_vollib.instruments import instrument_types

    table = render_table()
    for type_id, info in instrument_types().items():
        assert f"`{type_id}`" in table
        assert f"`{info.python_type.__name__}`" in table


def test_docs_are_wired_into_the_navigation() -> None:
    nav = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "instruments.md" in nav


def test_changelog_records_the_feature() -> None:
    changelog = (REPO_ROOT / "docs" / "changelog.md").read_text(encoding="utf-8")
    unreleased = changelog.split("## [Unreleased]", 1)[1].split("## [0.1.8]", 1)[0]
    assert "instruments" in unreleased
    assert "v0.2.0" in unreleased


@pytest.mark.parametrize(
    "claim",
    [
        "price_instrument",
        "greeks_instrument",
        "implied_volatility_instrument",
        "payoff",
        "instrument_types",
        "capabilities",
    ],
)
def test_documented_names_exist(claim: str) -> None:
    import fast_vollib.instruments as instruments

    assert claim in instruments.__all__
    assert hasattr(instruments, claim)


def test_documented_differentiability_matches_the_capability_set() -> None:
    """The prose table and the machine-readable record must not disagree."""
    from fast_vollib.instruments import EuropeanOption, IVSolver, capabilities

    document = DOC_PATH.read_text(encoding="utf-8")
    assert "**No — do not rely on it**" in document

    caps = capabilities(EuropeanOption)
    for _model, solver, _backend in caps.native_autodiff:
        assert solver is IVSolver.JACKEL, "docs claim only the Jäckel route carries gradients"
