"""Fail on *new* type-checker diagnostics, not on the ones already there.

``mypy src/fast_vollib`` is not clean, and making it clean is a separate piece
of work from anything this checker guards. What matters in the meantime is that
the count does not grow -- and a count alone is not enough, because a new error
can hide behind a removed one. So the baseline records diagnostic *identities*
(path, line, level, message, code) and the check reports the set difference.

Diagnostics that disappear are fine and are reported as such: they are progress,
and the baseline is refreshed at whatever moment someone chooses to. New ones
fail.

    uv run python scripts/check_mypy_baseline.py           # check
    uv run python scripts/check_mypy_baseline.py --update  # refresh after fixing

The baseline is environment-stable: it is identical under Python 3.10 with no
optional backends and under 3.13 with torch and jax installed, because
``[tool.mypy] python_version`` pins the analysis version and
``--ignore-missing-imports`` makes an absent backend indistinguishable from a
present one here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = Path(__file__).resolve().parent / "mypy_baseline.txt"
COMMAND = ("mypy", "src/fast_vollib", "--ignore-missing-imports")

#: A diagnostic line, as opposed to mypy's summary or a blank.
DIAGNOSTIC = re.compile(r"^src/.*: (error|note):")


def collect() -> set[str]:
    """Run mypy and return its diagnostic identities."""
    completed = subprocess.run(
        [sys.executable, "-m", *COMMAND],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    lines = {line.rstrip() for line in completed.stdout.splitlines() if DIAGNOSTIC.match(line)}
    if not lines and completed.returncode not in (0, 1):
        sys.stderr.write(completed.stdout + completed.stderr)
        raise SystemExit(f"mypy could not be run (exit {completed.returncode}).")
    return lines


def read_baseline() -> set[str]:
    if not BASELINE_PATH.exists():
        raise SystemExit(
            f"No baseline at {BASELINE_PATH}. Create one with --update after "
            f"confirming the current diagnostics are the ones you mean to accept."
        )
    return {
        line.rstrip()
        for line in BASELINE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update", action="store_true", help="overwrite the baseline with what mypy reports now"
    )
    args = parser.parse_args()

    current = collect()
    if args.update:
        BASELINE_PATH.write_text("\n".join(sorted(current)) + "\n", encoding="utf-8")
        sys.stdout.write(f"Wrote {len(current)} diagnostics to {BASELINE_PATH}.\n")
        return 0

    baseline = read_baseline()
    added = sorted(current - baseline)
    removed = sorted(baseline - current)

    if removed:
        sys.stdout.write(f"{len(removed)} diagnostic(s) no longer reported:\n")
        for line in removed:
            sys.stdout.write(f"  - {line}\n")
        sys.stdout.write(
            "Refresh the baseline with `--update` when you are ready to hold the new level.\n"
        )
    if added:
        sys.stderr.write(f"\n{len(added)} new type-checker diagnostic(s):\n")
        for line in added:
            sys.stderr.write(f"  + {line}\n")
        sys.stderr.write(
            "\nFix them, or -- if they are genuinely acceptable -- accept them "
            "deliberately with `--update`.\n"
        )
        return 1

    sys.stdout.write(f"No new diagnostics ({len(current)} at baseline).\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
