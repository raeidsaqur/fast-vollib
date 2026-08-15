# `testcapi-compat`

A four-line development shim, installed only via the `dev` dependency group.

`py_lets_be_rational` 1.0.x — one of the reference implementations `fast-vollib`
benchmarks and parity-tests against, and the version `py_vollib_vectorized` jits
against with numba — imports two float limits from CPython's private
`_testcapi` module:

```python
# py_lets_be_rational/constants.py
from _testcapi import DBL_MIN, DBL_MAX
```

`_testcapi` is a build artefact of CPython's own test suite. Most distributions
ship it, but some deliberately do not — the `install_only`
python-build-standalone builds that `uv python install` provides strip the test
modules. On those interpreters the import raises `ModuleNotFoundError` and takes
`py_vollib_vectorized` down with it. This is worth caring about wherever a
uv-managed interpreter is the only route to a recent CPython, e.g. on platforms
that ship no system CPython >= 3.12.

Upgrading past the problem does not work: `py_lets_be_rational` 1.1.x removes
the `_testcapi` dependency but restructures the module such that numba's
nopython type inference fails inside `py_vollib_vectorized`'s IV solver. That
raises a `TypingError` at call time rather than a clean `ImportError`, so the
lazy-import guards in `fast_vollib.compat` cannot fall back cleanly. Hence the
`py-lets-be-rational>=1.0.1,<1.1` pin plus this shim.

## Safety

`lib-dynload` precedes `site-packages` on `sys.path`, so on any interpreter that
ships the real `_testcapi` this package is shadowed and never imported. It is
inert there.

The exported values come straight from `sys.float_info`, which is what the C
module reports.

## Scope

Development and benchmarking only. No `fast_vollib` runtime module imports
`_testcapi`, and this package is not part of the published wheel's dependency
metadata — it exists so the third-party reference implementations used in parity
tests can be imported.
