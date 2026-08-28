"""Read-only discovery: what instrument types exist, and what they support.

The registry answers "what does this library know about?" and nothing else.
It is not an extension point: there is no public ``register()``, and the
mapping it returns cannot be mutated.

Immutability prevents type identifiers such as ``"european_option"`` from
changing meaning with import order. An external engine can dispatch directly
on an :class:`~fast_vollib.instruments.Instrument` subclass without modifying
this registry.

The registry is the one source the codec, the JSON Schema, the capability
tables, and the generated documentation all read, so those four cannot drift
apart.

Examples
--------
>>> from fast_vollib.instruments import instrument_type, instrument_types
>>> sorted(instrument_types())  # doctest: +NORMALIZE_WHITESPACE
['asian_option', 'asset', 'barrier_option', 'binary_option', 'european_option',
 'forward', 'future', 'lookback_option', 'variance_swap']
>>> info = instrument_type("european_option")
>>> info.python_type.__name__, info.schema_version
('EuropeanOption', 1)
>>> info.payoff_requirement.value
'terminal'
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping

from ._metadata import SCHEMA_VERSION, TYPE_SPECS
from .base import Instrument
from .capabilities import CapabilitySet, capabilities
from .enums import PayoffRequirement
from .errors import UnsupportedInstrumentError

__all__ = ["InstrumentTypeInfo", "instrument_type", "instrument_types", "type_id_for"]


@dataclass(frozen=True, slots=True)
class InstrumentTypeInfo:
    """Everything the registry knows about one instrument type.

    Attributes
    ----------
    type_id : str
        The stable wire identifier, also the record's ``instrument_type``
        discriminator.
    python_type : type[Instrument]
        The contract class.  Engines dispatch on this directly.
    schema_version : int
        Version of the serialized record format for this type.
    payoff_requirement : PayoffRequirement or None
        What state a payoff evaluator needs; ``None`` for types with no payoff
        (an asset is a description, not a contract).
    capabilities : CapabilitySet
        What can be computed for this type, given what is installed.
    description : str
        One-line summary, used by the generated documentation table.
    """

    type_id: str
    python_type: type[Instrument]
    schema_version: int
    payoff_requirement: PayoffRequirement | None
    capabilities: CapabilitySet
    description: str


@lru_cache(maxsize=1)
def _registry() -> Mapping[str, InstrumentTypeInfo]:
    """Build the type table once and hand out an unmodifiable view of it."""
    table = {
        spec.type_id: InstrumentTypeInfo(
            type_id=spec.type_id,
            python_type=spec.python_type,
            schema_version=SCHEMA_VERSION,
            payoff_requirement=spec.payoff_requirement,
            capabilities=capabilities(spec.python_type),
            description=spec.description,
        )
        for spec in TYPE_SPECS
    }
    return MappingProxyType(table)


def instrument_types() -> Mapping[str, InstrumentTypeInfo]:
    """Every instrument type in the public registry, keyed by ``type_id``.

    Returns
    -------
    Mapping[str, InstrumentTypeInfo]
        A read-only view.  Attempting to add, replace, or delete an entry
        raises :class:`TypeError`.

    Examples
    --------
    >>> from fast_vollib.instruments import instrument_types
    >>> types = instrument_types()
    >>> types["forward"].payoff_requirement.value
    'terminal'
    >>> types["asset"].payoff_requirement is None
    True
    """
    return _registry()


def instrument_type(type_id: str) -> InstrumentTypeInfo:
    """Look up one instrument type by its wire identifier.

    Parameters
    ----------
    type_id : str
        For example ``"european_option"``.

    Returns
    -------
    InstrumentTypeInfo

    Raises
    ------
    UnsupportedInstrumentError
        If no such type is defined.  The message lists the ones that are.
    """
    try:
        return _registry()[type_id]
    except KeyError:
        raise UnsupportedInstrumentError(
            f"No instrument type {type_id!r} is defined. "
            f"Known types: {', '.join(sorted(_registry()))}."
        ) from None


def type_id_for(instrument: Instrument | type[Instrument]) -> str:
    """The wire identifier for an instrument class or instance.

    Raises
    ------
    UnsupportedInstrumentError
        If the class is not part of the public registry.
    """
    cls = instrument if isinstance(instrument, type) else type(instrument)
    for info in _registry().values():
        if info.python_type is cls:
            return info.type_id
    raise UnsupportedInstrumentError(
        f"{cls.__name__} is not an instrument type known to fast_vollib.instruments. "
        f"Known types: {', '.join(sorted(_registry()))}."
    )
