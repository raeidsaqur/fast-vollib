"""The codec is strict, round-trips exactly, and cannot drift from the schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fast_vollib.instruments import (
    AsianOption,
    Asset,
    AssetClass,
    EuropeanOption,
    Forward,
    Future,
    InstrumentRef,
    InstrumentValidationError,
    LookbackOption,
    SerializationError,
    instrument_from_dict,
    instrument_from_json,
    instrument_to_dict,
    instrument_to_json,
    instrument_types,
)
from fast_vollib.instruments._metadata import SCHEMA_VERSION, TYPE_SPECS
from fast_vollib.instruments.serialization import (
    instrument_json_schema,
    render_instrument_json_schema,
)

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "docs" / "schemas" / "instrument-v1.schema.json"

FULLY_POPULATED = [
    Asset(
        instrument_id="asset-1",
        identifier="SPX",
        asset_class=AssetClass.INDEX,
        currency="USD",
    ),
    Forward(
        instrument_id="fwd-1",
        underlier=InstrumentRef(identifier="CL", asset_class=AssetClass.COMMODITY, currency="USD"),
        delivery_price=-37.63,
        maturity=0.25,
        settlement="physical",
        notional=-1000.0,
    ),
    Future(
        instrument_id="fut-1",
        underlier=InstrumentRef(identifier="ES", asset_class=AssetClass.INDEX, currency="USD"),
        contract_price=5000.0,
        maturity=0.0,
        settlement="cash",
        notional=50.0,
    ),
    EuropeanOption(
        instrument_id="SPX-2027-06-18-P-5000",
        underlier=InstrumentRef(identifier="SPX", asset_class=AssetClass.INDEX, currency="USD"),
        option_type="put",
        strike=5000.0,
        maturity=0.8164383562,
        settlement="cash",
        notional=100.0,
    ),
]

MINIMAL = [
    Asset(identifier="SPX", asset_class="index"),
    Forward(underlier="CL", delivery_price=75.0, maturity=0.25),
    Future(underlier="ES", contract_price=5000.0, maturity=0.25),
    EuropeanOption(underlier="SPX", option_type="call", strike=5000.0, maturity=0.75),
]

ALL_INSTRUMENTS = FULLY_POPULATED + MINIMAL


def ids(items: list[Any]) -> list[str]:
    return [f"{type(i).__name__}-{'full' if i.instrument_id else 'minimal'}" for i in items]


def option_record(**overrides: Any) -> dict[str, Any]:
    record = instrument_to_dict(MINIMAL[-1])
    record.update(overrides)
    return record


# --- round trips -------------------------------------------------------------


@pytest.mark.parametrize("instrument", ALL_INSTRUMENTS, ids=ids(ALL_INSTRUMENTS))
def test_dict_round_trip_preserves_equality(instrument: Any) -> None:
    assert instrument_from_dict(instrument_to_dict(instrument)) == instrument


@pytest.mark.parametrize("instrument", ALL_INSTRUMENTS, ids=ids(ALL_INSTRUMENTS))
def test_json_round_trip_preserves_equality(instrument: Any) -> None:
    assert instrument_from_json(instrument_to_json(instrument)) == instrument


@pytest.mark.parametrize("instrument", ALL_INSTRUMENTS, ids=ids(ALL_INSTRUMENTS))
def test_record_is_json_serializable_and_versioned(instrument: Any) -> None:
    record = instrument_to_dict(instrument)
    assert record["schema_version"] == SCHEMA_VERSION
    assert record["instrument_type"] in instrument_types()
    assert json.loads(json.dumps(record, allow_nan=False)) == record


@pytest.mark.parametrize("instrument", ALL_INSTRUMENTS, ids=ids(ALL_INSTRUMENTS))
def test_record_contains_no_engines_arrays_or_market_data(instrument: Any) -> None:
    record = instrument_to_dict(instrument)
    flat = json.dumps(record)
    for forbidden in ("volatility", "rate", "spot", "engine", "backend", "device", "dtype"):
        assert forbidden not in flat


def test_optional_fields_may_be_omitted_and_take_their_defaults() -> None:
    decoded = instrument_from_dict(
        {
            "schema_version": 1,
            "instrument_type": "european_option",
            "underlier": {"identifier": "SPX"},
            "option_type": "call",
            "strike": 100.0,
            "maturity": 1.0,
        }
    )
    assert decoded == EuropeanOption(
        underlier="SPX", option_type="call", strike=100.0, maturity=1.0
    )


def test_enum_spellings_are_canonical_on_the_wire() -> None:
    record = instrument_to_dict(
        EuropeanOption(underlier="SPX", option_type="c", strike=100.0, maturity=1.0)
    )
    assert record["option_type"] == "call"
    assert record["settlement"] == "cash"


def test_int_json_numbers_decode_as_floats() -> None:
    decoded = instrument_from_dict(option_record(strike=100, maturity=1))
    assert isinstance(decoded.strike, float)
    assert decoded.strike == 100.0


# --- strictness --------------------------------------------------------------


@pytest.mark.parametrize("version", [0, 2, 99, "1", 1.0, None, True], ids=str)
def test_unknown_schema_version_is_rejected(version: Any) -> None:
    with pytest.raises(SerializationError, match="schema_version"):
        instrument_from_dict(option_record(schema_version=version))


@pytest.mark.parametrize("type_id", ["american_option", "cliquet_option", "", 7, None], ids=str)
def test_unknown_instrument_type_is_rejected(type_id: Any) -> None:
    with pytest.raises(SerializationError) as excinfo:
        instrument_from_dict(option_record(instrument_type=type_id))
    assert "european_option" in str(excinfo.value)


def test_unknown_field_is_rejected_and_named() -> None:
    with pytest.raises(SerializationError) as excinfo:
        instrument_from_dict(option_record(barrier=90.0))
    message = str(excinfo.value)
    assert "'barrier'" in message
    assert "strike" in message


def test_unknown_nested_field_is_rejected() -> None:
    with pytest.raises(SerializationError, match="'sector'"):
        instrument_from_dict(option_record(underlier={"identifier": "SPX", "sector": "tech"}))


@pytest.mark.parametrize("missing", ["underlier", "option_type", "strike", "maturity"])
def test_missing_required_field_is_rejected(missing: str) -> None:
    record = option_record()
    del record[missing]
    with pytest.raises(SerializationError, match=missing):
        instrument_from_dict(record)


@pytest.mark.parametrize("key", ["schema_version", "instrument_type"])
def test_missing_discriminators_are_rejected(key: str) -> None:
    record = option_record()
    del record[key]
    with pytest.raises(SerializationError, match=key):
        instrument_from_dict(record)


def test_short_flag_spelling_is_not_wire_format() -> None:
    with pytest.raises(SerializationError) as excinfo:
        instrument_from_dict(option_record(option_type="c"))
    assert "'call'" in str(excinfo.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("strike", "100"),
        ("strike", True),
        ("maturity", [1.0]),
        ("option_type", 1),
        ("underlier", "SPX"),
        ("instrument_id", 3),
    ],
    ids=["str-strike", "bool-strike", "list-maturity", "int-option-type", "str-ref", "int-id"],
)
def test_wrong_json_types_are_rejected(field: str, value: Any) -> None:
    with pytest.raises(SerializationError, match=field):
        instrument_from_dict(option_record(**{field: value}))


@pytest.mark.parametrize("field", ["strike", "maturity", "option_type", "underlier"])
def test_null_is_rejected_for_non_nullable_fields(field: str) -> None:
    with pytest.raises(SerializationError, match=field):
        instrument_from_dict(option_record(**{field: None}))


def test_nullable_fields_accept_null() -> None:
    decoded = instrument_from_dict(
        option_record(
            instrument_id=None,
            underlier={"identifier": "SPX", "asset_class": None, "currency": None},
        )
    )
    assert decoded.instrument_id is None
    assert decoded.underlier.currency is None


def test_out_of_range_terms_still_fail_the_contract_validation() -> None:
    with pytest.raises(InstrumentValidationError, match="strike"):
        instrument_from_dict(option_record(strike=-1.0))
    with pytest.raises(InstrumentValidationError, match="notional"):
        instrument_from_dict(option_record(notional=0.0))


def test_non_mapping_input_is_rejected() -> None:
    with pytest.raises(SerializationError, match="mapping"):
        instrument_from_dict(["european_option"])  # type: ignore[arg-type]


def test_invalid_json_text_is_rejected() -> None:
    with pytest.raises(SerializationError, match="not valid JSON"):
        instrument_from_json("{not json")


def test_encoding_a_foreign_object_is_rejected() -> None:
    with pytest.raises(SerializationError, match="not a serializable instrument type"):
        instrument_to_dict(object())  # type: ignore[arg-type]


def test_nan_and_infinity_are_not_json() -> None:
    option = EuropeanOption(underlier="SPX", option_type="call", strike=1.0, maturity=1.0)
    record = instrument_to_dict(option)
    record["strike"] = float("nan")
    with pytest.raises(ValueError):
        json.dumps(record, allow_nan=False)


# --- the checked-in schema artifact ------------------------------------------


def test_schema_file_regenerates_byte_for_byte() -> None:
    assert SCHEMA_PATH.exists(), (
        f"{SCHEMA_PATH} is missing; run scripts/generate_instrument_schema.py"
    )
    assert SCHEMA_PATH.read_text(encoding="utf-8") == render_instrument_json_schema()


def test_schema_is_valid_json_and_self_describing() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"].endswith("instrument-v1.schema.json")


def test_schema_objects_are_closed() -> None:
    schema = instrument_json_schema()
    for definition in schema["$defs"].values():
        assert definition["additionalProperties"] is False


def test_schema_union_is_discriminated_by_instrument_type() -> None:
    schema = instrument_json_schema()
    refs = {entry["$ref"].rsplit("/", 1)[-1] for entry in schema["oneOf"]}
    assert refs == set(instrument_types())
    for type_id in refs:
        discriminator = schema["$defs"][type_id]["properties"]["instrument_type"]
        assert discriminator["const"] == type_id


# --- registry / codec / schema consistency ------------------------------------


@pytest.mark.parametrize("spec", TYPE_SPECS, ids=lambda s: s.type_id)
def test_registry_codec_and_schema_agree_on_every_type(spec: Any) -> None:
    schema = instrument_json_schema()["$defs"][spec.type_id]
    info = instrument_types()[spec.type_id]

    assert info.python_type is spec.python_type
    assert info.schema_version == SCHEMA_VERSION

    # The codec writes exactly the fields the schema declares.
    example_fields = set(spec.field_names)
    assert set(schema["properties"]) == example_fields | {"schema_version", "instrument_type"}

    # Required-ness matches field-by-field.
    expected_required = {"schema_version", "instrument_type"} | {
        f.name for f in spec.fields if f.required
    }
    assert set(schema["required"]) == expected_required


@pytest.mark.parametrize("spec", TYPE_SPECS, ids=lambda s: s.type_id)
def test_schema_mirrors_the_runtime_numeric_and_string_constraints(spec: Any) -> None:
    properties = instrument_json_schema()["$defs"][spec.type_id]["properties"]
    for field_spec in spec.fields:
        declared = properties[field_spec.name]
        if field_spec.exclusive_minimum is not None:
            assert declared["exclusiveMinimum"] == field_spec.exclusive_minimum
        if field_spec.minimum is not None:
            assert declared["minimum"] == field_spec.minimum
        if field_spec.non_zero:
            assert declared["not"] == {"const": 0}
        if field_spec.min_length is not None:
            assert declared["minLength"] == field_spec.min_length
        if field_spec.enum_cls is not None:
            expected = [m.value for m in field_spec.enum_cls]
            if field_spec.nullable:
                expected = [*expected, None]
            assert declared["enum"] == expected


@pytest.mark.parametrize("instrument", ALL_INSTRUMENTS, ids=ids(ALL_INSTRUMENTS))
def test_every_encoded_record_validates_against_the_checked_in_schema(instrument: Any) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(instrument_to_dict(instrument), schema)


def test_a_record_the_codec_rejects_also_fails_the_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    for bad in (option_record(barrier=90.0), option_record(strike=-1.0), option_record(notional=0)):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)


@pytest.mark.parametrize(
    "instrument",
    [
        AsianOption(
            underlier="ACME",
            option_type="call",
            strike=100.0,
            averaging_method="arithmetic",
            strike_convention="fixed",
            maturity=1.0,
        ),
        LookbackOption(
            underlier="ACME",
            option_type="call",
            strike=100.0,
            strike_convention="fixed",
            maturity=1.0,
        ),
    ],
    ids=["asian", "lookback"],
)
def test_schema_enforces_the_strike_convention_relation(instrument: Any) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = instrument_json_schema()
    fixed_without_strike = instrument_to_dict(instrument)
    fixed_without_strike.pop("strike")
    floating_with_strike = instrument_to_dict(instrument)
    floating_with_strike["strike_convention"] = "floating"
    for bad in (fixed_without_strike, floating_with_strike):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)
