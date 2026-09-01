"""What surface algorithms this installation can actually run, as data.

A client -- a benchmark runner, a web API, a notebook -- needs to know which
algorithms exist, what they take, what they produce, and which of them are
unavailable here and why.  Reading that off a source tree is how a UI ends up
advertising a model that was deleted two releases ago.  This module answers it
as immutable, serializable metadata computed against what is installed.

Two distinctions are load-bearing.

*A family is not an output kind.*  A forecaster may return a point forecast or
a predictive distribution; a generative model always returns a distribution; a
calibrator always returns a definite surface.  ``family`` says which lifecycle
the algorithm has and ``output`` says what a caller gets back, and neither one
implies the other.  A client that dispatched on family alone would have to
inspect a returned object to find out what it asked for.

*Unavailable is not absent.*  An algorithm appears here only when it is
implemented.  It is reported unavailable when an optional dependency, a
compatible backend, or a required checkpoint is missing -- with a
machine-readable reason a client can render.  A model that has never been
written is not listed at all, because listing it would be advertising.

The registry is **built once from a fixed list and cannot be mutated**.  There
is no ``register()``, so a public identifier such as ``'svi'`` cannot change
meaning with import order, and importing one algorithm's module cannot change
another algorithm's availability.

Examples
--------
>>> from fast_vollib.surface import list_algorithms
>>> ids = [entry.spec.public_id for entry in list_algorithms()]
>>> "flat" in ids
True
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
import importlib.util
import json
from types import MappingProxyType
from typing import Any

from .errors import SurfaceAlgorithmUnavailableError, SurfaceValidationError

__all__ = [
    "FAMILIES",
    "OUTPUT_KINDS",
    "SCHEMA_VERSION",
    "UNAVAILABLE_CODES",
    "AlgorithmAvailability",
    "BackendSupport",
    "SurfaceAlgorithmSpec",
    "build_algorithm",
    "capabilities_document",
    "capabilities_json_schema",
    "get_algorithm",
    "list_algorithms",
    "render_capabilities_json_schema",
    "validate_configuration",
]

_SCHEMA_ID = (
    "https://raeidsaqur.github.io/fast-vollib/schemas/"
    "fast-vollib-surface-capabilities-v1.schema.json"
)

#: The wire identifier of the serialized capability document.
SCHEMA_VERSION = "fast-vollib-surface-capabilities-v1"

#: Lifecycles an algorithm can have.  See
#: :mod:`fast_vollib.surface.protocols` for what each one promises.
FAMILIES = ("calibrator", "conditional", "forecaster", "generative")

#: What a caller gets back from the algorithm's principal method.
OUTPUT_KINDS = ("definite", "distribution")

#: Machine-readable reasons an implemented algorithm can be unavailable.
UNAVAILABLE_CODES = ("optional_dependency", "backend", "checkpoint")


@dataclass(frozen=True, slots=True)
class BackendSupport:
    """Which array backends, dtypes, and devices an algorithm really supports.

    Declared honestly rather than optimistically.  A SciPy-based calibrator runs
    on the host in float64 and says so; claiming a torch backend because the
    surface it returns can be evaluated on tensors would make "does this fit on
    GPU" unanswerable.

    Attributes
    ----------
    backends : tuple[str, ...]
        Backend names the algorithm's own computation runs on.
    dtypes : tuple[str, ...]
        Floating dtypes it is validated for.
    devices : tuple[str, ...]
        Devices it runs on; ``('host',)`` for anything that does not leave the CPU.
    gradients : bool
        Whether gradients flow through the algorithm's own computation.  A
        surface whose *evaluation* is differentiable while its *calibration* is
        not sets this ``False`` and says so in the summary.
    """

    backends: tuple[str, ...] = ("numpy",)
    dtypes: tuple[str, ...] = ("float64",)
    devices: tuple[str, ...] = ("host",)
    gradients: bool = False

    def to_dict(self) -> dict[str, Any]:
        """The support matrix as a JSON-safe mapping."""
        return {
            "backends": list(self.backends),
            "dtypes": list(self.dtypes),
            "devices": list(self.devices),
            "gradients": bool(self.gradients),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class SurfaceAlgorithmSpec:
    """Immutable description of one surface algorithm.

    Attributes
    ----------
    public_id : str
        Stable identifier a client configures against.  Renaming one is a
        breaking change to every stored experiment configuration.
    display_name : str
        Human-readable name.
    family : str
        One of :data:`FAMILIES`.
    output : str
        One of :data:`OUTPUT_KINDS`.
    summary : str
        One paragraph: what it fits, and what it does not claim.
    implementation_version : str
        Version of *this implementation*, bumped when its numerics change.
        Distinct from the library version, which the capability document
        carries separately: a bundle needs both to say what produced a number.
    configuration_schema : Mapping[str, Any]
        A closed JSON Schema object for the algorithm's constructor keywords.
    support : BackendSupport
    requires_training : bool
        Whether the algorithm needs a training run before it can be conditioned.
    requires_checkpoint : bool
        Whether it needs weights loaded from somewhere.
    optional_dependencies : tuple[str, ...]
        Distributions that must be installed for it to run.
    supports_arbitrary_points : bool
        Whether it evaluates off its fitting grid.
    supports_temporal_context : bool
        Whether it consumes a history rather than a single observation set.
    supports_uncertainty : bool
        Whether its output carries a standard deviation, quantiles, or samples.
    references : tuple[str, ...]
        Sources the implementation was written from.
    """

    public_id: str
    display_name: str
    family: str
    output: str
    summary: str
    implementation_version: str
    configuration_schema: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    support: BackendSupport = BackendSupport()
    requires_training: bool = False
    requires_checkpoint: bool = False
    optional_dependencies: tuple[str, ...] = ()
    supports_arbitrary_points: bool = True
    supports_temporal_context: bool = False
    supports_uncertainty: bool = False
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise SurfaceValidationError(f"family must be one of {FAMILIES}; got {self.family!r}.")
        if self.output not in OUTPUT_KINDS:
            raise SurfaceValidationError(
                f"output must be one of {OUTPUT_KINDS}; got {self.output!r}."
            )
        if not self.public_id or not self.public_id.replace("-", "").replace("_", "").isalnum():
            raise SurfaceValidationError(
                f"public_id must be a non-empty alphanumeric identifier "
                f"(hyphens and underscores allowed); got {self.public_id!r}."
            )
        object.__setattr__(
            self, "configuration_schema", MappingProxyType(dict(self.configuration_schema))
        )

    def to_dict(self) -> dict[str, Any]:
        """The specification as a JSON-safe mapping."""
        return {
            "public_id": self.public_id,
            "display_name": self.display_name,
            "family": self.family,
            "output": self.output,
            "summary": self.summary,
            "implementation_version": self.implementation_version,
            "configuration_schema": _plain(self.configuration_schema),
            "support": self.support.to_dict(),
            "requires_training": bool(self.requires_training),
            "requires_checkpoint": bool(self.requires_checkpoint),
            "optional_dependencies": list(self.optional_dependencies),
            "supports_arbitrary_points": bool(self.supports_arbitrary_points),
            "supports_temporal_context": bool(self.supports_temporal_context),
            "supports_uncertainty": bool(self.supports_uncertainty),
            "references": list(self.references),
        }


@dataclass(frozen=True, slots=True)
class AlgorithmAvailability:
    """An algorithm specification plus whether it can run here.

    Attributes
    ----------
    spec : SurfaceAlgorithmSpec
    available : bool
    unavailable_code : str | None
        One of :data:`UNAVAILABLE_CODES` when unavailable; ``None`` otherwise.
    unavailable_reason : str | None
        Human-readable explanation, safe to render in a UI.
    """

    spec: SurfaceAlgorithmSpec
    available: bool = True
    unavailable_code: str | None = None
    unavailable_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """The entry as a JSON-safe mapping."""
        return {
            **self.spec.to_dict(),
            "available": bool(self.available),
            "unavailable_code": self.unavailable_code,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True, slots=True)
class _Provider:
    """One built-in algorithm: its specification and how to construct it."""

    spec: SurfaceAlgorithmSpec
    factory: Callable[..., Any]


def _module_installed(name: str) -> bool:
    """Whether ``name`` can be imported, without importing it.

    ``find_spec`` walks the finders and stops; it does not execute the module.
    That matters here: describing an algorithm must not pull torch into a
    process that only wanted the list.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken installation
        return False


@lru_cache(maxsize=1)
def _providers() -> tuple[_Provider, ...]:
    """The built-in algorithm table, built once, in a fixed order.

    Imported lazily so that ``import fast_vollib.surface`` does not pay for the
    fitting modules, and constructed from one explicit list so that the set of
    identifiers cannot depend on which modules a caller happened to import.
    """
    from .fitting import builtin_providers

    return tuple(builtin_providers())


def _availability(provider: _Provider) -> AlgorithmAvailability:
    missing = [name for name in provider.spec.optional_dependencies if not _module_installed(name)]
    if missing:
        return AlgorithmAvailability(
            spec=provider.spec,
            available=False,
            unavailable_code="optional_dependency",
            unavailable_reason=(
                f"Requires the optional dependency "
                f"{', '.join(repr(name) for name in missing)}, which is not installed."
            ),
        )
    if provider.spec.requires_checkpoint:
        return AlgorithmAvailability(
            spec=provider.spec,
            available=False,
            unavailable_code="checkpoint",
            unavailable_reason=(
                "Requires trained weights, and this installation ships none. "
                "Load a checkpoint explicitly to use it."
            ),
        )
    return AlgorithmAvailability(spec=provider.spec)


def list_algorithms(
    *, family: str | None = None, available_only: bool = False
) -> tuple[AlgorithmAvailability, ...]:
    """Every built-in surface algorithm, with its availability here.

    Parameters
    ----------
    family:
        Restrict to one of :data:`FAMILIES`.
    available_only:
        Drop entries that cannot run in this installation.  Off by default: a
        client that shows why something is unavailable is more useful than one
        that pretends it never existed.

    Examples
    --------
    >>> from fast_vollib.surface import list_algorithms
    >>> calibrators = list_algorithms(family="calibrator")
    >>> all(entry.spec.family == "calibrator" for entry in calibrators)
    True
    """
    if family is not None and family not in FAMILIES:
        raise SurfaceValidationError(f"family must be one of {FAMILIES}; got {family!r}.")
    entries = tuple(_availability(provider) for provider in _providers())
    if family is not None:
        entries = tuple(entry for entry in entries if entry.spec.family == family)
    if available_only:
        entries = tuple(entry for entry in entries if entry.available)
    return entries


def get_algorithm(public_id: str) -> AlgorithmAvailability:
    """The entry for ``public_id``.

    Raises
    ------
    SurfaceAlgorithmUnavailableError
        If no algorithm carries that identifier.  The message lists the ones
        that do.
    """
    for provider in _providers():
        if provider.spec.public_id == public_id:
            return _availability(provider)
    known = ", ".join(sorted(provider.spec.public_id for provider in _providers()))
    raise SurfaceAlgorithmUnavailableError(
        f"Unknown surface algorithm {public_id!r}. Known algorithms: {known}."
    )


def build_algorithm(public_id: str, config: Mapping[str, Any] | None = None) -> Any:
    """Construct the algorithm ``public_id`` from a validated configuration.

    The configuration is checked against the algorithm's own closed schema
    before construction, so a typo in a benchmark configuration file is an error
    naming the field rather than a silently defaulted parameter.

    Raises
    ------
    SurfaceAlgorithmUnavailableError
        If the algorithm is unknown, or is implemented but cannot run here.
    SurfaceValidationError
        If the configuration does not satisfy the algorithm's schema.
    """
    entry = get_algorithm(public_id)
    if not entry.available:
        raise SurfaceAlgorithmUnavailableError(
            f"{public_id!r} is implemented but unavailable here: {entry.unavailable_reason}"
        )
    validated = validate_configuration(entry.spec, config or {})
    for provider in _providers():
        if provider.spec.public_id == public_id:
            return provider.factory(**validated)
    raise AssertionError(f"provider for {public_id!r} vanished")  # pragma: no cover


def validate_configuration(spec: SurfaceAlgorithmSpec, config: Mapping[str, Any]) -> dict[str, Any]:
    """Check ``config`` against ``spec``'s closed schema and return it as a plain dict.

    Deliberately a small validator rather than a schema library: the schemas
    this package emits are closed objects of scalars, the library is a runtime
    dependency nobody should have to install to fit a smile, and a validator
    that accepts more than the schema says would defeat the point.
    """
    schema = spec.configuration_schema
    properties: Mapping[str, Any] = schema.get("properties", {})
    required: Sequence[str] = schema.get("required", ())
    unknown = sorted(set(config) - set(properties))
    if unknown:
        raise SurfaceValidationError(
            f"Unknown configuration field(s) {', '.join(repr(name) for name in unknown)} "
            f"for {spec.public_id!r}. Known fields: {', '.join(sorted(properties)) or '(none)'}."
        )
    missing = [name for name in required if name not in config]
    if missing:
        raise SurfaceValidationError(
            f"Missing required configuration field(s) "
            f"{', '.join(repr(name) for name in missing)} for {spec.public_id!r}."
        )
    validated: dict[str, Any] = {}
    for name, value in config.items():
        validated[name] = _validate_field(spec.public_id, name, value, properties[name])
    return validated


_JSON_TYPES: Mapping[str, tuple[type, ...]] = {
    "number": (int, float),
    "integer": (int,),
    "string": (str,),
    "boolean": (bool,),
    "array": (list, tuple),
}


def _validate_field(algorithm: str, name: str, value: Any, schema: Mapping[str, Any]) -> Any:
    kind = schema.get("type")
    if kind is not None:
        expected = _JSON_TYPES.get(kind)
        if expected is None:
            raise SurfaceValidationError(
                f"{algorithm}.{name} declares an unsupported schema type {kind!r}."
            )
        if kind != "boolean" and isinstance(value, bool):
            raise SurfaceValidationError(
                f"{algorithm}.{name} must be a {kind}; got a boolean. A truthy value "
                f"must not silently stand in for a number."
            )
        if not isinstance(value, expected):
            raise SurfaceValidationError(
                f"{algorithm}.{name} must be a {kind}; got {type(value).__name__}."
            )
    choices = schema.get("enum")
    if choices is not None and value not in choices:
        raise SurfaceValidationError(
            f"{algorithm}.{name} must be one of {tuple(choices)}; got {value!r}."
        )
    for bound, ok, description in (
        ("minimum", lambda v, b: v >= b, "at least"),
        ("maximum", lambda v, b: v <= b, "at most"),
        ("exclusiveMinimum", lambda v, b: v > b, "greater than"),
        ("exclusiveMaximum", lambda v, b: v < b, "less than"),
    ):
        limit = schema.get(bound)
        if limit is not None and not ok(value, limit):
            raise SurfaceValidationError(
                f"{algorithm}.{name} must be {description} {limit}; got {value!r}."
            )
    return value


def capabilities_document() -> dict[str, Any]:
    """The full capability listing as a JSON-safe mapping.

    Carries the library version alongside each algorithm's own implementation
    version: a stored result needs both to say what produced it, because the
    library can change around an algorithm whose numerics did not.

    Examples
    --------
    >>> from fast_vollib.surface import capabilities_document
    >>> document = capabilities_document()
    >>> document["schema"]
    'fast-vollib-surface-capabilities-v1'
    """
    from .. import __version__

    return {
        "schema": SCHEMA_VERSION,
        "library": "fast-vollib",
        "library_version": str(__version__),
        "algorithms": [entry.to_dict() for entry in list_algorithms()],
    }


def _plain(value: Any) -> Any:
    """A plain, JSON-safe copy of a possibly proxied nested mapping."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


# --- the wire schema ----------------------------------------------------------


def _closed(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(properties),
        "properties": properties,
    }


def capabilities_json_schema() -> dict[str, Any]:
    """The closed Draft 2020-12 schema for ``fast-vollib-surface-capabilities-v1``.

    Built from the same shape :func:`capabilities_document` emits, so a client
    that validates against the checked-in artifact is validating against what
    this build actually produces.  ``configuration_schema`` is deliberately left
    as a free object: it is itself a JSON Schema, and each algorithm's is its
    own.
    """
    strings = {"type": "array", "items": {"type": "string"}}
    algorithm = _closed(
        {
            "public_id": {"type": "string", "minLength": 1},
            "display_name": {"type": "string", "minLength": 1},
            "family": {"type": "string", "enum": list(FAMILIES)},
            "output": {"type": "string", "enum": list(OUTPUT_KINDS)},
            "summary": {"type": "string"},
            "implementation_version": {"type": "string", "minLength": 1},
            "configuration_schema": {"type": "object"},
            "support": _closed(
                {
                    "backends": strings,
                    "dtypes": strings,
                    "devices": strings,
                    "gradients": {"type": "boolean"},
                }
            ),
            "requires_training": {"type": "boolean"},
            "requires_checkpoint": {"type": "boolean"},
            "optional_dependencies": strings,
            "supports_arbitrary_points": {"type": "boolean"},
            "supports_temporal_context": {"type": "boolean"},
            "supports_uncertainty": {"type": "boolean"},
            "references": strings,
            "available": {"type": "boolean"},
            "unavailable_code": {
                "type": ["string", "null"],
                "enum": list(UNAVAILABLE_CODES) + [None],
            },
            "unavailable_reason": {"type": ["string", "null"]},
        }
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": _SCHEMA_ID,
        "title": "fast-vollib surface-capabilities-v1",
        **_closed(
            {
                "schema": {"const": SCHEMA_VERSION},
                "library": {"const": "fast-vollib"},
                "library_version": {"type": "string", "minLength": 1},
                "algorithms": {"type": "array", "items": algorithm},
            }
        ),
    }


def render_capabilities_json_schema() -> str:
    """The schema as the exact text checked in at ``docs/schemas``."""
    return (
        json.dumps(capabilities_json_schema(), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
