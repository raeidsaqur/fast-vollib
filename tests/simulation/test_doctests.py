"""Every doctest in the process and simulation packages is executed."""

from __future__ import annotations

import doctest
import importlib
import pkgutil
from types import ModuleType

import pytest

import fast_vollib.processes as processes
import fast_vollib.simulation as simulation


def _package_modules() -> list[ModuleType]:
    modules: list[ModuleType] = []
    for package in (processes, simulation):
        modules.append(package)
        for info in pkgutil.iter_modules(package.__path__):
            modules.append(importlib.import_module(f"{package.__name__}.{info.name}"))
    modules.append(importlib.import_module("fast_vollib._random_api"))
    return modules


@pytest.mark.parametrize("module", _package_modules(), ids=lambda m: m.__name__)
def test_module_doctests(module: ModuleType) -> None:
    results = doctest.testmod(module, verbose=False, raise_on_error=True)
    del results  # raise_on_error turns any failure into an exception
