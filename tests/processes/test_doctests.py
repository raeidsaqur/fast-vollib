"""Execute all process-package examples, including modules in subpackages."""

from __future__ import annotations

import doctest
import importlib
import pkgutil
from types import ModuleType

import pytest

import fast_vollib.processes as processes


def _package_modules() -> list[ModuleType]:
    modules = [processes]
    for info in pkgutil.walk_packages(processes.__path__, prefix=f"{processes.__name__}."):
        modules.append(importlib.import_module(info.name))
    return modules


def test_the_examples_are_not_vacuous() -> None:
    """Count executable examples, not empty docstrings returned by the finder."""
    examples = [
        example
        for module in _package_modules()
        for test in doctest.DocTestFinder().find(module)
        for example in test.examples
    ]
    assert examples


@pytest.mark.parametrize("module", _package_modules(), ids=lambda m: m.__name__)
def test_module_doctests(module: ModuleType) -> None:
    doctest.testmod(module, verbose=False, raise_on_error=True)
