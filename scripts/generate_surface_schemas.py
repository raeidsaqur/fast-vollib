"""Regenerate the checked-in JSON Schemas for the surface layer's wire contracts.

Three closed Draft 2020-12 schemas, each built from the same shape its codec
emits, so the checked-in artifact can never describe a different format from the
code that produces it:

* ``fast-vollib-surface-capabilities-v1`` -- what algorithms this build offers;
* ``fast-vollib-surface-evaluation-v1`` -- how one predicted surface scored;
* ``fast-vollib-generative-arbitrage-v1`` -- how a distribution's draws scored.

A test regenerates all three and compares them byte for byte with the files.

Usage::

    python scripts/generate_surface_schemas.py          # write
    python scripts/generate_surface_schemas.py --check  # verify bytes
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from fast_vollib.surface.capabilities import (  # noqa: E402
    render_capabilities_json_schema,
)
from fast_vollib.surface.evaluation import render_evaluation_json_schema  # noqa: E402
from fast_vollib.surface.generative import render_generative_json_schema  # noqa: E402

SCHEMA_DIRECTORY = REPOSITORY_ROOT / "docs" / "schemas"
CAPABILITIES_PATH = SCHEMA_DIRECTORY / "fast-vollib-surface-capabilities-v1.schema.json"
EVALUATION_PATH = SCHEMA_DIRECTORY / "fast-vollib-surface-evaluation-v1.schema.json"
GENERATIVE_PATH = SCHEMA_DIRECTORY / "fast-vollib-generative-arbitrage-v1.schema.json"


def render() -> dict[Path, str]:
    """Every generated artifact, keyed by the path it belongs at."""
    return {
        CAPABILITIES_PATH: render_capabilities_json_schema(),
        EVALUATION_PATH: render_evaluation_json_schema(),
        GENERATIVE_PATH: render_generative_json_schema(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the checked-in files match, rather than rewriting them",
    )
    arguments = parser.parse_args(argv)

    stale = []
    for path, content in render().items():
        relative = path.relative_to(REPOSITORY_ROOT)
        if arguments.check:
            current = path.read_text(encoding="utf-8") if path.exists() else None
            if current != content:
                stale.append(relative)
                print(f"stale: {relative}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"wrote {relative}")

    if arguments.check:
        if stale:
            print(
                "Regenerate with: python scripts/generate_surface_schemas.py",
                file=sys.stderr,
            )
            return 1
        print("surface schemas are up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
