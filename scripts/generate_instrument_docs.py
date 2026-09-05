"""Regenerate the instrument type/capability table in ``docs/instruments.md``.

The table is a projection of ``fast_vollib.instruments.instrument_types()``,
not a hand-maintained list, so documentation cannot claim a capability the
registry does not have.  Run after adding an instrument type or changing what
one supports:

    uv run python scripts/generate_instrument_docs.py

Only the block between the BEGIN/END markers is rewritten.  A test regenerates
it and compares, so a stale table fails the suite.

Deliberately excluded: the ``native_autodiff`` and ``simulation_autodiff``
sets, which depend on which optional backends are installed and would make the
checked-in file environment-dependent.  Differentiability is documented as
prose in the same page.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fast_vollib.instruments import IVSolver, PricingModel, instrument_types

DOC_PATH = PROJECT_ROOT / "docs" / "instruments.md"
BEGIN = "<!-- BEGIN generated: instrument capability table -->"
END = "<!-- END generated: instrument capability table -->"

_MODEL_ABBREVIATION = {
    PricingModel.BLACK: "black",
    PricingModel.BLACK_SCHOLES: "bs",
    PricingModel.BLACK_SCHOLES_MERTON: "bsm",
}


def _models(models: frozenset[PricingModel]) -> str:
    if not models:
        return "—"
    ordered = [m for m in PricingModel if m in models]
    return ", ".join(_MODEL_ABBREVIATION[m] for m in ordered)


def _solvers(mapping: object) -> str:
    assert hasattr(mapping, "items")
    if not mapping:
        return "—"
    parts = []
    for model in PricingModel:
        solvers = mapping.get(model)  # type: ignore[attr-defined]
        if not solvers:
            continue
        names = ", ".join(s.value for s in IVSolver if s in solvers)
        parts.append(f"{_MODEL_ABBREVIATION[model]}: {names}")
    return "; ".join(parts) if parts else "—"


def render_table() -> str:
    lines = [
        BEGIN,
        "",
        "| Type | `type_id` | Payoff | Payoff needs | Price | Greeks | "
        "Implied volatility | Monte Carlo | Cashflows | Present value |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for info in instrument_types().values():
        caps = info.capabilities
        requirement = info.payoff_requirement.value if info.payoff_requirement is not None else "—"
        lines.append(
            f"| `{info.python_type.__name__}` "
            f"| `{info.type_id}` "
            f"| {'yes' if caps.payoff else 'no'} "
            f"| {requirement} "
            f"| {_models(caps.price)} "
            f"| {_models(caps.greeks)} "
            f"| {_solvers(caps.implied_volatility)} "
            f"| {'yes' if caps.simulate else 'no'} "
            f"| {'yes' if caps.cashflows else 'no'} "
            f"| {'yes' if caps.present_value else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Model abbreviations: `black` = Black-76, `bs` = Black-Scholes, "
            "`bsm` = Black-Scholes-Merton. A dash means the operation is not "
            "available for that type, and asking for it raises rather than "
            "returning an approximation. **Monte Carlo** is a type-level "
            "answer: an individual contract is additionally eligible only with "
            "a strictly positive maturity, which "
            "`MonteCarloEngine.supports(instrument)` applies and is "
            "authoritative for an actual request. **Cashflows** and **present "
            "value** are the fixed-income route: a security with dated payments "
            "has no payoff and no option-pricing model, so `cashflows()` reads "
            "its schedule and `fast_vollib.pricing.present_value()` values it "
            "against a `DiscountCurve`.",
            "",
            END,
        ]
    )
    return "\n".join(lines)


def splice(document: str, table: str) -> str:
    if BEGIN not in document or END not in document:
        raise SystemExit(f"{DOC_PATH} is missing the generated-table markers.")
    head = document[: document.index(BEGIN)]
    tail = document[document.index(END) + len(END) :]
    return head + table + tail


def main() -> int:
    document = DOC_PATH.read_text(encoding="utf-8")
    updated = splice(document, render_table())
    if "--check" in sys.argv[1:]:
        if updated != document:
            sys.stderr.write("The instrument capability table is stale.\n")
            return 1
        return 0
    if updated == document:
        sys.stdout.write(f"{DOC_PATH} is up to date.\n")
        return 0
    DOC_PATH.write_text(updated, encoding="utf-8")
    sys.stdout.write(f"Updated the capability table in {DOC_PATH}.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
