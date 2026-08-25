"""Regenerate the checked-in instrument JSON Schema.

The schema at ``docs/schemas/instrument-v1.schema.json`` is a build product of
``fast_vollib.instruments.serialization``, not a hand-maintained file.  Run
this after changing the instrument field table:

    uv run python scripts/generate_instrument_schema.py

``tests/instruments/test_serialization.py`` regenerates it in memory and
compares bytes, so a stale artifact fails the suite rather than shipping.
"""

from __future__ import annotations

from pathlib import Path
import sys

from fast_vollib.instruments.serialization import render_instrument_json_schema

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "docs" / "schemas" / "instrument-v1.schema.json"


def main() -> int:
    text = render_instrument_json_schema()
    previous = SCHEMA_PATH.read_text(encoding="utf-8") if SCHEMA_PATH.exists() else None
    if previous == text:
        sys.stdout.write(f"{SCHEMA_PATH} is up to date.\n")
        return 0
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(text, encoding="utf-8")
    sys.stdout.write(f"Wrote {SCHEMA_PATH}.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
