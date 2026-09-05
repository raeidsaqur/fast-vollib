"""A strict, versioned codec for instrument records -- and the schema for it.

An instrument is a value, so it has a wire format: a flat JSON object carrying
its terms, a ``schema_version``, and an ``instrument_type`` discriminator.

The codec is strict in the direction that matters.  It rejects unknown schema
versions, unknown instrument types, unknown fields, missing required fields,
and non-canonical enum spellings -- all of them errors rather than input to be
ignored.  Leniency here would be the expensive kind: a record whose
``option_type`` was silently dropped decodes into a valid-looking option that
is not the one that was sent, and nothing downstream can tell.  The
constructors accept ``'c'`` and ``'p'`` as a convenience; the wire format has
exactly one spelling per value.

The JSON Schema in ``docs/schemas/instrument-v1.schema.json`` is generated from
the same field table this codec reads, and a test regenerates it and compares
bytes.  It therefore cannot describe a record the codec would reject, or omit a
constraint the codec enforces.  Generating it is a development step:
``jsonschema`` is not a runtime dependency and nothing here validates against
it at call time.

The schema is experimental until explicitly declared stable. Incompatible
record-shape changes increment ``schema_version``. Pickle is not a wire format
for these objects.

Examples
--------
>>> from fast_vollib.instruments import EuropeanOption, instrument_from_dict, instrument_to_dict
>>> option = EuropeanOption(
...     instrument_id="SPX-C-5000", underlier="SPX",
...     option_type="c", strike=5000.0, maturity=0.75,
... )
>>> record = instrument_to_dict(option)
>>> record["instrument_type"], record["option_type"]
('european_option', 'call')
>>> instrument_from_dict(record) == option
True
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from enum import Enum
import json
from typing import Any

import numpy as np

from ._metadata import (
    INSTRUMENT_REF_FIELDS,
    SCHEMA_VERSION,
    TYPE_SPECS,
    FieldSpec,
    TypeSpec,
    spec_for_type,
    spec_for_type_id,
)
from .base import Instrument, InstrumentRef
from .errors import SerializationError

__all__ = [
    "SCHEMA_VERSION",
    "instrument_from_dict",
    "instrument_from_json",
    "instrument_json_schema",
    "instrument_to_dict",
    "instrument_to_json",
    "render_instrument_json_schema",
]

_SCHEMA_ID = "https://raeidsaqur.github.io/fast-vollib/schemas/instrument-v1.schema.json"
_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
_REF_DEF = "instrument_ref"


# --- encoding ----------------------------------------------------------------


def _encode_ref(ref: InstrumentRef) -> dict[str, Any]:
    return {
        "identifier": ref.identifier,
        "asset_class": None if ref.asset_class is None else ref.asset_class.value,
        "currency": ref.currency,
    }


def _encode_value(value: object) -> Any:
    if isinstance(value, InstrumentRef):
        return _encode_ref(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        # A schedule. JSON has no tuple, and the decoder turns the list back
        # into one, so the round trip is closed rather than lossy.
        return [_encode_value(item) for item in value]
    if isinstance(value, float):
        return value
    return value


def instrument_to_dict(instrument: Instrument) -> dict[str, Any]:
    """Encode an instrument as a plain, JSON-ready dictionary.

    Parameters
    ----------
    instrument : Instrument
        Any registered contract type.

    Returns
    -------
    dict
        ``schema_version`` and ``instrument_type`` first, then the contract's
        terms in a fixed order.  Optional fields are always written, so the
        record shape is constant for a given type.

    Raises
    ------
    SerializationError
        If the object is not a registered instrument type.
    """
    spec = spec_for_type(type(instrument))
    if spec is None:
        raise SerializationError(
            f"{type(instrument).__name__} is not a serializable instrument type. "
            f"Known types: {', '.join(sorted(s.python_type.__name__ for s in TYPE_SPECS))}."
        )
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "instrument_type": spec.type_id,
    }
    for field_spec in spec.fields:
        record[field_spec.name] = _encode_value(getattr(instrument, field_spec.name))
    return record


def instrument_to_json(instrument: Instrument, **json_kwargs: Any) -> str:
    """Encode an instrument as a JSON string.

    Parameters
    ----------
    instrument : Instrument
    **json_kwargs
        Passed to :func:`json.dumps`.  ``allow_nan`` is forced to ``False``:
        ``NaN`` and the infinities are not JSON, and a contract cannot hold
        them anyway.

    Examples
    --------
    >>> import json
    >>> from fast_vollib.instruments import Forward, instrument_to_json
    >>> text = instrument_to_json(Forward(underlier="CL", delivery_price=75.0, maturity=0.25))
    >>> json.loads(text)["instrument_type"]
    'forward'
    """
    json_kwargs["allow_nan"] = False
    return json.dumps(instrument_to_dict(instrument), **json_kwargs)


# --- decoding ----------------------------------------------------------------


def _decode_number(value: object, *, field: str, type_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise SerializationError(
            f"{type_id}.{field} must be a JSON number; got {type(value).__name__}."
        )
    return float(value)


def _decode_string(value: object, *, field: str, type_id: str) -> str:
    if not isinstance(value, str):
        raise SerializationError(
            f"{type_id}.{field} must be a JSON string; got {type(value).__name__}."
        )
    return value


def _decode_enum(value: object, spec: FieldSpec, *, type_id: str) -> Enum:
    assert spec.enum_cls is not None
    text = _decode_string(value, field=spec.name, type_id=type_id)
    try:
        return spec.enum_cls(text)
    except ValueError:
        valid = ", ".join(repr(m.value) for m in spec.enum_cls)
        raise SerializationError(
            f"{type_id}.{spec.name} must be one of {valid}; got {text!r}. "
            f"The wire format uses one canonical spelling per value."
        ) from None


def _decode_field(value: object, spec: FieldSpec, *, type_id: str) -> object:
    if value is None:
        if spec.nullable:
            return None
        raise SerializationError(f"{type_id}.{spec.name} must not be null.")
    if spec.json_type == "object":
        return _decode_ref(value, field=spec.name, type_id=type_id)
    if spec.enum_cls is not None:
        return _decode_enum(value, spec, type_id=type_id)
    if spec.json_type == "number":
        return _decode_number(value, field=spec.name, type_id=type_id)
    if spec.json_type == "array":
        return _decode_array(value, spec, type_id=type_id)
    return _decode_string(value, field=spec.name, type_id=type_id)


def _decode_array(value: object, spec: FieldSpec, *, type_id: str) -> tuple[Any, ...]:
    """A JSON array as a tuple, each element decoded by the item spec.

    A ``list`` and nothing else. A string is a sequence of characters and a
    mapping is a sequence of keys, and either would decode into a schedule of
    the wrong thing rather than failing.
    """
    assert spec.items is not None  # guaranteed by the import-time metadata guard
    if not isinstance(value, list):
        raise SerializationError(
            f"{type_id}.{spec.name} must be a JSON array; got {type(value).__name__}."
        )
    if spec.min_items is not None and len(value) < spec.min_items:
        raise SerializationError(
            f"{type_id}.{spec.name} must have at least {spec.min_items} entr"
            f"{'y' if spec.min_items == 1 else 'ies'}; got {len(value)}."
        )
    return tuple(
        _decode_field(item, spec.items, type_id=f"{type_id}.{spec.name}[{index}]")
        for index, item in enumerate(value)
    )


def _decode_ref(value: object, *, field: str, type_id: str) -> InstrumentRef:
    if not isinstance(value, MappingABC):
        raise SerializationError(
            f"{type_id}.{field} must be a JSON object; got {type(value).__name__}."
        )
    nested_id = f"{type_id}.{field}"
    known = {spec.name for spec in INSTRUMENT_REF_FIELDS}
    _reject_unknown_keys(value, known, type_id=nested_id)
    kwargs: dict[str, object] = {}
    for spec in INSTRUMENT_REF_FIELDS:
        if spec.name not in value:
            if spec.required:
                raise SerializationError(f"{nested_id} is missing required field {spec.name!r}.")
            continue
        kwargs[spec.name] = _decode_field(value[spec.name], spec, type_id=nested_id)
    return InstrumentRef(**kwargs)  # type: ignore[arg-type]


def _reject_unknown_keys(record: MappingABC, known: set[str], *, type_id: str) -> None:
    unknown = sorted(str(key) for key in record if key not in known)
    if unknown:
        raise SerializationError(
            f"Unknown field(s) {', '.join(repr(u) for u in unknown)} in a {type_id} record. "
            f"Known fields: {', '.join(sorted(known))}."
        )


def _resolve_spec(record: MappingABC) -> TypeSpec:
    if "schema_version" not in record:
        raise SerializationError("Instrument record is missing 'schema_version'.")
    version = record["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != SCHEMA_VERSION:
        raise SerializationError(
            f"Unsupported instrument schema_version {version!r}; this build reads "
            f"version {SCHEMA_VERSION} only."
        )
    if "instrument_type" not in record:
        raise SerializationError("Instrument record is missing 'instrument_type'.")
    type_id = record["instrument_type"]
    spec = spec_for_type_id(type_id) if isinstance(type_id, str) else None
    if spec is None:
        known = ", ".join(sorted(s.type_id for s in TYPE_SPECS))
        raise SerializationError(f"Unknown instrument_type {type_id!r}. Known types: {known}.")
    return spec


def instrument_from_dict(record: MappingABC) -> Instrument:
    """Decode an instrument from a dictionary produced by :func:`instrument_to_dict`.

    Parameters
    ----------
    record : Mapping
        Must carry ``schema_version`` and ``instrument_type``.

    Returns
    -------
    Instrument
        An instance equal to the one that was encoded.

    Raises
    ------
    SerializationError
        For a structural problem: an unknown schema version, an unknown
        instrument type, an unknown or missing field, a value of the wrong JSON
        type, or a non-canonical enum spelling.
    InstrumentValidationError
        If the record is structurally sound but a term is out of range -- a
        negative strike, say.  The contract's own validation raises it, so
        decoding cannot produce an instrument the constructor would refuse.

    Examples
    --------
    >>> from fast_vollib.instruments import instrument_from_dict
    >>> instrument_from_dict({
    ...     "schema_version": 1, "instrument_type": "forward",
    ...     "underlier": {"identifier": "CL"},
    ...     "delivery_price": 75.0, "maturity": 0.25,
    ... }).delivery_price
    75.0
    """
    if not isinstance(record, MappingABC):
        raise SerializationError(
            f"An instrument record must be a mapping; got {type(record).__name__}."
        )
    spec = _resolve_spec(record)
    known = set(spec.field_names) | {"schema_version", "instrument_type"}
    _reject_unknown_keys(record, known, type_id=spec.type_id)

    kwargs: dict[str, object] = {}
    for field_spec in spec.fields:
        if field_spec.name not in record:
            if field_spec.required:
                raise SerializationError(
                    f"A {spec.type_id} record is missing required field {field_spec.name!r}."
                )
            continue
        kwargs[field_spec.name] = _decode_field(
            record[field_spec.name], field_spec, type_id=spec.type_id
        )
    return spec.python_type(**kwargs)  # type: ignore[arg-type]


def instrument_from_json(text: str | bytes | bytearray) -> Instrument:
    """Decode an instrument from a JSON string.

    Raises
    ------
    SerializationError
        If the text is not valid JSON, or the record is not decodable.
    """
    try:
        record = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise SerializationError(f"Instrument record is not valid JSON: {exc}") from exc
    return instrument_from_dict(record)


# --- schema generation (development-time) ------------------------------------


def _field_schema(spec: FieldSpec) -> dict[str, Any]:
    if spec.json_type == "object":
        schema: dict[str, Any] = {"$ref": f"#/$defs/{_REF_DEF}"}
        if spec.description:
            schema["description"] = spec.description
        return schema

    if spec.json_type == "array":
        assert spec.items is not None  # guaranteed by the import-time metadata guard
        array_schema: dict[str, Any] = {"type": "array", "items": _field_schema(spec.items)}
        if spec.min_items is not None:
            array_schema["minItems"] = spec.min_items
        if spec.description:
            array_schema["description"] = spec.description
        return array_schema

    schema = {}
    types: list[str] = [spec.json_type]
    if spec.nullable:
        types.append("null")
    schema["type"] = types[0] if len(types) == 1 else types
    if spec.enum_cls is not None:
        values: list[Any] = [member.value for member in spec.enum_cls]
        if spec.nullable:
            values.append(None)
        schema["enum"] = values
    if spec.min_length is not None:
        schema["minLength"] = spec.min_length
    if spec.minimum is not None:
        schema["minimum"] = spec.minimum
    if spec.exclusive_minimum is not None:
        schema["exclusiveMinimum"] = spec.exclusive_minimum
    if spec.non_zero:
        schema["not"] = {"const": 0}
    if spec.description:
        schema["description"] = spec.description
    return schema


def _ref_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "title": "Underlier reference",
        "description": (
            "An immutable pointer to an underlier. Carries identity and descriptive "
            "metadata only -- never market data."
        ),
        "additionalProperties": False,
        "properties": {spec.name: _field_schema(spec) for spec in INSTRUMENT_REF_FIELDS},
        "required": [spec.name for spec in INSTRUMENT_REF_FIELDS if spec.required],
    }


def _type_schema(spec: TypeSpec) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "schema_version": {
            "type": "integer",
            "const": SCHEMA_VERSION,
            "description": "Version of the instrument record format.",
        },
        "instrument_type": {
            "type": "string",
            "const": spec.type_id,
            "description": "Discriminator naming the instrument type.",
        },
    }
    for field_spec in spec.fields:
        properties[field_spec.name] = _field_schema(field_spec)
    required = ["schema_version", "instrument_type"]
    required.extend(field_spec.name for field_spec in spec.fields if field_spec.required)
    schema: dict[str, Any] = {
        "type": "object",
        "title": spec.python_type.__name__,
        "description": spec.description,
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }
    # Fixed- and floating-strike records have a relation between two fields
    # that neither field can express alone. The constructor enforces the same
    # rule: fixed requires a numeric strike; floating permits an omitted strike
    # (the decoder's default is None) and requires null when it is present.
    if {"strike", "strike_convention"}.issubset(spec.field_names):
        schema["allOf"] = [
            {
                "if": {"properties": {"strike_convention": {"const": "fixed"}}},
                "then": {
                    "required": ["strike"],
                    "properties": {"strike": {"type": "number"}},
                },
            },
            {
                "if": {"properties": {"strike_convention": {"const": "floating"}}},
                "then": {"properties": {"strike": {"type": "null"}}},
            },
        ]
    return schema


def instrument_json_schema() -> dict[str, Any]:
    """The JSON Schema describing every instrument record, as a dictionary.

    Built from the same field table the codec reads, so the two cannot
    disagree.  Objects are closed (``additionalProperties: false``), the union
    is discriminated by ``instrument_type``, and the runtime numeric and string
    constraints are mirrored keyword for keyword.

    Examples
    --------
    >>> from fast_vollib.instruments.serialization import instrument_json_schema
    >>> schema = instrument_json_schema()
    >>> "european_option" in schema["$defs"], "instrument_ref" in schema["$defs"]
    (True, True)
    >>> schema["$defs"]["european_option"]["properties"]["strike"]["exclusiveMinimum"]
    0.0
    """
    defs: dict[str, Any] = {_REF_DEF: _ref_schema()}
    for spec in TYPE_SPECS:
        defs[spec.type_id] = _type_schema(spec)
    return {
        "$schema": _SCHEMA_DIALECT,
        "$id": _SCHEMA_ID,
        "title": "fast-vollib instrument record",
        "description": (
            "A single fast-vollib instrument, serialized. Records are discriminated by "
            "'instrument_type' and versioned by 'schema_version'. Objects are closed: an "
            "unknown field is an error, not ignored input. Generated from the same field "
            "table the runtime codec uses; do not edit by hand. Experimental until "
            "explicitly declared stable."
        ),
        "oneOf": [{"$ref": f"#/$defs/{spec.type_id}"} for spec in TYPE_SPECS],
        "$defs": dict(sorted(defs.items())),
    }


def render_instrument_json_schema() -> str:
    """The schema as the exact text checked in at ``docs/schemas``.

    A test regenerates this and compares it byte for byte with the file, so the
    checked-in artifact can never describe a different format from the code.
    """
    return (
        json.dumps(instrument_json_schema(), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
