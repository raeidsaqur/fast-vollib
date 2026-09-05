"""Every example in the pricing package is executed.

The instruments, simulation, and surface packages each run their doctests; this
package did not, so its eleven worked examples were prose that happened to look
like code.  They pass -- but nothing was checking, and the fixed income and
BCC97 work adds several modules of examples here.

Non-recursive, matching the instruments and simulation runners: the package is
flat and a subpackage would be silently skipped, which
``test_the_package_is_flat_so_nothing_is_skipped`` catches.
"""

from __future__ import annotations

import doctest
import importlib
import pkgutil
from types import ModuleType

import pytest

import fast_vollib.pricing as pricing


def _package_modules() -> list[ModuleType]:
    modules = [pricing]
    for info in pkgutil.iter_modules(pricing.__path__):
        modules.append(importlib.import_module(f"{pricing.__name__}.{info.name}"))
    return modules


def test_the_package_is_flat_so_nothing_is_skipped() -> None:
    """``iter_modules`` does not recurse; a subpackage would go unexamined."""
    subpackages = [info.name for info in pkgutil.iter_modules(pricing.__path__) if info.ispkg]
    assert subpackages == [], (
        f"{subpackages} would not have their doctests run; switch this module to "
        f"pkgutil.walk_packages, as tests/test_surface/test_doctests.py does."
    )


def test_the_examples_are_not_vacuous() -> None:
    """A runner that found no examples would pass while checking nothing."""
    total = sum(len(doctest.DocTestFinder().find(module)) for module in _package_modules())
    assert total > 0


@pytest.mark.parametrize("module", _package_modules(), ids=lambda m: m.__name__)
def test_module_doctests(module: ModuleType) -> None:
    results = doctest.testmod(module, verbose=False, raise_on_error=True)
    del results  # raise_on_error turns any failure into an exception
