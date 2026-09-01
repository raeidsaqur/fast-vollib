"""Generate the ``diagnostics-v1`` JSON schema and its golden fixture.

The schema and the fixture are build products, not hand-maintained files: they
are rendered from the same field tables the codec encodes with, so the wire
contract and its documentation cannot drift apart.  Run with ``--check`` in CI
to fail when a code change would alter either artifact without regenerating it.

    python scripts/generate_diagnostics_schema.py          # write
    python scripts/generate_diagnostics_schema.py --check  # verify bytes
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from fast_vollib.diagnostics import (  # noqa: E402
    Box,
    DiagnosticRecord,
    DiagnosticReport,
    DiagnosticsConfig,
    RaggedArbitrageConfig,
    SampleDiagnostics,
    SurfaceQuotes,
    diagnose_fit,
    diagnose_surface,
    normalized_option_price,
    quotes_to_surface,
    serialization,  # noqa: E402
)

SCHEMA_PATH = REPOSITORY_ROOT / "docs" / "schemas" / "diagnostics-v1.schema.json"
GOLDEN_PATH = REPOSITORY_ROOT / "tests" / "fixtures" / "diagnostics" / "golden-v1.json"


def _smile(surface: str, maturity: float, strikes: list[float], vols: list[float]) -> SurfaceQuotes:
    return SurfaceQuotes(
        k=np.array(strikes),
        T=np.full(len(strikes), maturity),
        iv=np.array(vols),
        surface_id=surface,
        point_id=[f"{surface}-{maturity}-{index}" for index in range(len(strikes))],
    )


def _quoted_sample() -> tuple[SurfaceQuotes, np.ndarray]:
    """A two-maturity surface carrying bid/ask, with one unpredicted target."""
    strikes = [-0.2, -0.1, 0.0, 0.1, 0.2]
    k = np.array(strikes * 2)
    T = np.array([0.25] * 5 + [0.50] * 5)
    iv = np.array([0.24, 0.22, 0.21, 0.22, 0.24, 0.23, 0.21, 0.20, 0.21, 0.23])
    is_call = np.array([k_value >= 0.0 for k_value in k])
    predicted = iv + np.array([0.004, -0.002, 0.001, 0.0, -0.003, 0.002, 0.0, -0.001, 0.003, 0.0])
    predicted[7] = -0.5  # one invalid prediction: the sample is `partial`

    # Keep the golden payload independent of last-bit differences between
    # platform normal-CDF implementations.  The spread arithmetic itself has
    # dedicated numerical tests; this wire fixture needs only to exercise a
    # present spread block.  Valid predictions therefore sit exactly at their
    # quoted price.  The invalid row is quoted at the zero-volatility price but
    # remains unpriced by the evaluator.
    quoted_iv = np.maximum(predicted, 0.0)
    quoted_price = normalized_option_price(quoted_iv, k, T, is_call)
    quotes = SurfaceQuotes(
        k=k,
        T=T,
        iv=iv,
        surface_id="AAA",
        point_id=list(range(10)),
        bid=quoted_price,
        ask=quoted_price,
        is_call=is_call,
    )
    return quotes, predicted


def _duplicate_sample() -> tuple[SurfaceQuotes, np.ndarray]:
    """A smile with an exactly duplicated strike and a zero-variance node."""
    k = np.array([-0.1, 0.0, 0.0, 0.1, 0.2])
    quotes = SurfaceQuotes(
        k=k,
        T=np.full(5, 0.75),
        iv=np.array([0.20, 0.19, 0.21, 0.20, 0.22]),
        surface_id="BBB",
        point_id=list(range(5)),
    )
    return quotes, np.array([0.20, 0.19, 0.21, 0.0, 0.22])


def _grid_sample() -> SampleDiagnostics:
    """A rectangular grid, so the record carries a grid block."""
    strikes = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
    maturities = np.array([0.25, 0.5])
    k_grid, T_grid = np.meshgrid(strikes, maturities, indexing="ij")
    vols = 0.20 + 0.10 * k_grid**2
    quotes = SurfaceQuotes(
        k=k_grid.ravel(),
        T=T_grid.ravel(),
        iv=vols.ravel(),
        surface_id="CCC",
        point_id=list(range(k_grid.size)),
    )
    sample = diagnose_fit(quotes.iv, quotes)
    surface = quotes_to_surface(quotes)
    return SampleDiagnostics(
        fit=sample.fit,
        spread=sample.spread,
        ragged=sample.ragged,
        grid=diagnose_surface(surface),
    )


def build_golden_report() -> DiagnosticReport:
    """A fixed report exercising every branch of the wire contract."""
    config = DiagnosticsConfig(
        regions=(
            Box(
                name="liquid",
                complement_name="illiquid",
                k_min=-0.2,
                k_max=0.2,
                T_max=0.5,
                closed="both",
            ),
            Box(name="wings", complement_name=None, k_min=0.15, closed="left"),
        ),
        ragged=RaggedArbitrageConfig(),
        metadata={"fixture": "golden-v1", "purpose": "byte-compared wire contract"},
    )
    quoted, quoted_pred = _quoted_sample()
    duplicated, duplicated_pred = _duplicate_sample()
    flat = _smile("DDD", 1.0, [-0.05, 0.05], [0.2, 0.2])  # too few nodes: zero checks
    records = [
        DiagnosticRecord(
            model="prior",
            split="unif_10",
            sample_id=900,
            diagnostics=diagnose_fit(quoted_pred, quoted, regions=config.regions),
        ),
        DiagnosticRecord(
            # A different split: this sample carries no bid/ask, and a group may not
            # mix samples that do with samples that do not.
            model="prior",
            split="extrap_10",
            sample_id=901,
            diagnostics=diagnose_fit(duplicated_pred, duplicated, regions=config.regions),
        ),
        DiagnosticRecord(
            model="bspline",
            split="extrap_5",
            sample_id="d-902",
            diagnostics=diagnose_fit(flat.iv, flat, regions=config.regions),
        ),
        DiagnosticRecord(
            model="ssvi",
            split="grid",
            sample_id=1,
            diagnostics=_grid_sample(),
        ),
    ]
    return DiagnosticReport.from_records(records, config=config)


def render() -> dict[Path, str]:
    """The generated artifacts keyed by destination path."""
    return {
        SCHEMA_PATH: serialization.render_schema(),
        GOLDEN_PATH: serialization.dumps(build_golden_report(), indent=2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed artifacts match byte-for-byte instead of writing them",
    )
    arguments = parser.parse_args(argv)
    stale: list[Path] = []
    for path, content in render().items():
        if arguments.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                stale.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path.relative_to(REPOSITORY_ROOT)}")
    if stale:
        for path in stale:
            print(f"stale: {path.relative_to(REPOSITORY_ROOT)}", file=sys.stderr)
        print(
            "Regenerate with: python scripts/generate_diagnostics_schema.py",
            file=sys.stderr,
        )
        return 1
    if arguments.check:
        print("diagnostics-v1 schema and golden fixture are up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
