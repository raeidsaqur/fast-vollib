"""Batches: one validation core, three doors, and no per-row objects."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fast_vollib.instruments import (
    Asset,
    AssetClass,
    EuropeanOption,
    EuropeanOptionBatch,
    Forward,
    InstrumentRef,
    InstrumentValidationError,
    OptionType,
    SettlementType,
    VanillaMarketInputs,
    forward_price,
    log_moneyness,
    moneyness,
    time_to_maturity,
)

FLAGS = ["c", "p", "c"]
STRIKES = [95.0, 100.0, 105.0]
MATURITIES = [0.25, 0.5, 1.0]
NOTIONALS = [1.0, -2.0, 100.0]
IDS = ["row-0", "row-1", "row-2"]
REF = InstrumentRef(identifier="SPX", asset_class=AssetClass.INDEX, currency="USD")


def arrays_batch(**overrides: object) -> EuropeanOptionBatch:
    kwargs: dict[str, object] = {
        "option_type": FLAGS,
        "strike": STRIKES,
        "maturity": MATURITIES,
        "notional": NOTIONALS,
        "instrument_id": IDS,
        "underlier": REF,
    }
    kwargs.update(overrides)
    return EuropeanOptionBatch.from_arrays(**kwargs)  # type: ignore[arg-type]


def scalar_options() -> list[EuropeanOption]:
    return [
        EuropeanOption(
            instrument_id=identifier,
            underlier=REF,
            option_type=flag,
            strike=strike,
            maturity=maturity,
            notional=notional,
        )
        for flag, strike, maturity, notional, identifier in zip(
            FLAGS, STRIKES, MATURITIES, NOTIONALS, IDS
        )
    ]


def contract_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cp": FLAGS,
            "K": STRIKES,
            "T": MATURITIES,
            "qty": NOTIONALS,
            "sym": ["SPX"] * 3,
            "row_id": IDS,
        }
    )


def frame_batch(**overrides: object) -> EuropeanOptionBatch:
    kwargs: dict[str, object] = {
        "option_type_col": "cp",
        "strike_col": "K",
        "maturity_col": "T",
        "underlier_col": "sym",
        "notional_col": "qty",
        "instrument_id_col": "row_id",
        "asset_class": AssetClass.INDEX,
        "currency": "USD",
    }
    kwargs.update(overrides)
    return EuropeanOptionBatch.from_frame(contract_frame(), **kwargs)  # type: ignore[arg-type]


# --- the three constructors agree --------------------------------------------


def test_all_three_paths_produce_equal_columns() -> None:
    from_arrays = arrays_batch()
    from_instruments = EuropeanOptionBatch.from_instruments(scalar_options())
    from_frame = frame_batch()
    assert from_arrays.equals(from_instruments)
    assert from_arrays.equals(from_frame)


def test_columns_are_host_numpy_with_the_kernel_conventions() -> None:
    batch = arrays_batch()
    assert batch.flag.dtype.kind == "U"
    assert batch.flag.tolist() == FLAGS
    for name in ("strike", "maturity", "notional"):
        column = getattr(batch, name)
        assert isinstance(column, np.ndarray)
        assert column.dtype == np.float64


def test_columns_are_read_only() -> None:
    batch = arrays_batch()
    with pytest.raises(ValueError):
        batch.strike[0] = 1.0


def test_length_and_repr() -> None:
    batch = arrays_batch()
    assert len(batch) == 3
    assert "n=3" in repr(batch)
    assert "SPX" in repr(batch)


# --- broadcasting and normalization ------------------------------------------


def test_scalar_columns_broadcast() -> None:
    batch = arrays_batch(maturity=0.5, notional=1.0, underlier="SPX", instrument_id=None)
    assert batch.maturity.tolist() == [0.5] * 3
    assert batch.notional.tolist() == [1.0] * 3
    assert batch.instrument_id.tolist() == ["", "", ""]


def test_a_single_underlier_broadcasts_across_the_book() -> None:
    batch = arrays_batch(underlier="SPX")
    assert batch.underlier_identifier.tolist() == ["SPX"] * 3


def test_per_row_underliers_are_kept() -> None:
    batch = arrays_batch(underlier=["SPX", "NDX", "SPX"])
    assert batch.underlier_identifier.tolist() == ["SPX", "NDX", "SPX"]


@pytest.mark.parametrize(
    "spelling",
    [["call", "put", "CALL"], ["c", "P", "c"], [OptionType.CALL, OptionType.PUT, "call"]],
    ids=["long", "short", "enums"],
)
def test_option_type_spellings_normalize_to_kernel_flags(spelling: list[object]) -> None:
    assert arrays_batch(option_type=spelling).flag.tolist() == ["c", "p", "c"]


def test_settlement_defaults_and_parses() -> None:
    assert arrays_batch().settlement.tolist() == ["cash"] * 3
    assert arrays_batch(settlement=SettlementType.PHYSICAL).settlement.tolist() == ["physical"] * 3
    assert arrays_batch(settlement=["cash", "physical", "cash"]).settlement.tolist() == [
        "cash",
        "physical",
        "cash",
    ]


def test_underlier_metadata_is_carried_and_overridable() -> None:
    batch = arrays_batch(underlier=REF)
    assert batch.underlier_asset_class.tolist() == ["index"] * 3
    assert batch.underlier_currency.tolist() == ["USD"] * 3
    plain = arrays_batch(underlier="SPX")
    assert plain.underlier_asset_class.tolist() == [""] * 3
    assert plain.underliers[0].asset_class is None


def test_asset_underliers_are_converted_to_references() -> None:
    asset = Asset(identifier="SPX", asset_class=AssetClass.INDEX, currency="USD")
    assert arrays_batch(underlier=asset).equals(arrays_batch(underlier=asset.ref()))


def test_underliers_property_rebuilds_references() -> None:
    batch = arrays_batch(underlier=REF)
    assert batch.underliers == (REF, REF, REF)


# --- row-local validation ----------------------------------------------------


@pytest.mark.parametrize(
    ("column", "values", "match"),
    [
        ("strike", [95.0, -1.0, 105.0], "'strike'.*row 1"),
        ("strike", [95.0, 0.0, 105.0], "'strike'.*row 1"),
        ("strike", [95.0, np.nan, 105.0], "'strike'.*row 1"),
        ("maturity", [0.25, 0.5, -0.1], "'maturity'.*row 2"),
        ("maturity", [np.inf, 0.5, 1.0], "'maturity'.*row 0"),
        ("notional", [1.0, 0.0, 1.0], "'notional'.*row 1"),
    ],
)
def test_invalid_values_name_the_row(column: str, values: list[float], match: str) -> None:
    with pytest.raises(InstrumentValidationError, match=match):
        arrays_batch(**{column: values})


def test_invalid_option_type_names_the_row() -> None:
    with pytest.raises(InstrumentValidationError, match="row 1"):
        arrays_batch(option_type=["c", "straddle", "p"])


def test_blank_underlier_names_the_row() -> None:
    with pytest.raises(InstrumentValidationError, match="row 2"):
        arrays_batch(underlier=["SPX", "NDX", "  "])


def test_null_underlier_is_rejected() -> None:
    with pytest.raises(InstrumentValidationError, match="row 1"):
        arrays_batch(underlier=["SPX", None, "SPX"])


def test_mismatched_column_lengths_are_rejected() -> None:
    with pytest.raises(InstrumentValidationError, match="length"):
        arrays_batch(strike=[95.0, 100.0])


def test_two_dimensional_columns_are_rejected() -> None:
    with pytest.raises(InstrumentValidationError, match="one-dimensional"):
        arrays_batch(strike=np.array([[95.0, 100.0], [105.0, 110.0]]))


def test_batch_validation_matches_the_scalar_constructor() -> None:
    """The same value is accepted or refused by both paths."""
    for bad_strike in (-1.0, 0.0, float("nan")):
        with pytest.raises(InstrumentValidationError):
            EuropeanOption(underlier="SPX", option_type="c", strike=bad_strike, maturity=1.0)
        with pytest.raises(InstrumentValidationError):
            arrays_batch(strike=[bad_strike, 100.0, 105.0])


# --- homogeneity -------------------------------------------------------------


def test_mixed_instrument_types_are_rejected() -> None:
    mixed = [*scalar_options(), Forward(underlier="SPX", delivery_price=100.0, maturity=1.0)]
    with pytest.raises(InstrumentValidationError, match="homogeneous"):
        EuropeanOptionBatch.from_instruments(mixed)  # type: ignore[arg-type]


def test_empty_input_is_rejected() -> None:
    with pytest.raises(InstrumentValidationError, match="empty"):
        EuropeanOptionBatch.from_instruments([])
    with pytest.raises(InstrumentValidationError, match="empty"):
        EuropeanOptionBatch.from_frame(
            pd.DataFrame({"cp": [], "K": [], "T": [], "sym": []}),
            option_type_col="cp",
            strike_col="K",
            maturity_col="T",
            underlier_col="sym",
        )


# --- the frame bridge --------------------------------------------------------


def test_frame_round_trip_preserves_order_and_columns() -> None:
    batch = arrays_batch()
    frame = batch.to_frame()
    assert list(frame["strike"]) == STRIKES
    rebuilt = EuropeanOptionBatch.from_frame(
        frame,
        option_type_col="option_type",
        strike_col="strike",
        maturity_col="maturity",
        underlier_col="underlier",
        notional_col="notional",
        instrument_id_col="instrument_id",
        asset_class=AssetClass.INDEX,
        currency="USD",
    )
    assert rebuilt.equals(batch)


def test_to_frame_renders_unset_optionals_as_none() -> None:
    frame = arrays_batch(underlier="SPX", instrument_id=None).to_frame()
    assert frame["instrument_id"].tolist() == [None] * 3
    assert frame["asset_class"].tolist() == [None] * 3


def test_missing_column_names_what_is_available() -> None:
    with pytest.raises(InstrumentValidationError) as excinfo:
        frame_batch(strike_col="strike")
    message = str(excinfo.value)
    assert "'strike'" in message
    assert "'K'" in message


@pytest.mark.parametrize("market_column", ["sigma", "rate", "underlying_price", "price"])
def test_market_columns_are_refused_by_name(market_column: str) -> None:
    frame = contract_frame().rename(columns={"K": market_column})
    with pytest.raises(InstrumentValidationError, match="market data"):
        EuropeanOptionBatch.from_frame(
            frame,
            option_type_col="cp",
            strike_col=market_column,
            maturity_col="T",
            underlier_col="sym",
        )


def test_nothing_is_guessed_from_column_names() -> None:
    """Every mapping is required; there is no fallback to a conventional name."""
    with pytest.raises(TypeError):
        EuropeanOptionBatch.from_frame(contract_frame())  # type: ignore[call-arg]


def test_frame_row_errors_report_the_row_index() -> None:
    frame = contract_frame()
    frame.loc[1, "K"] = -5.0
    with pytest.raises(InstrumentValidationError, match="row 1"):
        EuropeanOptionBatch.from_frame(
            frame,
            option_type_col="cp",
            strike_col="K",
            maturity_col="T",
            underlier_col="sym",
        )


# --- no per-row object construction ------------------------------------------


@pytest.mark.parametrize("path", ["from_arrays", "from_frame"], ids=["arrays", "frame"])
def test_array_paths_build_no_per_row_objects(path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point of a columnar batch is not paying for N objects.

    Counted rather than benchmarked: a timing assertion would be flaky, while
    a constructor call count is exact.
    """
    counts = {"option": 0, "ref": 0}

    original_option_init = EuropeanOption.__post_init__
    original_ref_init = InstrumentRef.__post_init__

    def spy_option(self: EuropeanOption) -> None:
        counts["option"] += 1
        original_option_init(self)

    def spy_ref(self: InstrumentRef) -> None:
        counts["ref"] += 1
        original_ref_init(self)

    monkeypatch.setattr(EuropeanOption, "__post_init__", spy_option)
    monkeypatch.setattr(InstrumentRef, "__post_init__", spy_ref)

    n = 500
    if path == "from_arrays":
        batch = EuropeanOptionBatch.from_arrays(
            option_type=np.array(["c"] * n),
            strike=np.linspace(90.0, 110.0, n),
            maturity=np.full(n, 0.5),
            underlier=np.array([f"SYM{i % 7}" for i in range(n)]),
        )
    else:
        batch = EuropeanOptionBatch.from_frame(
            pd.DataFrame(
                {
                    "cp": ["c"] * n,
                    "K": np.linspace(90.0, 110.0, n),
                    "T": np.full(n, 0.5),
                    "sym": [f"SYM{i % 7}" for i in range(n)],
                }
            ),
            option_type_col="cp",
            strike_col="K",
            maturity_col="T",
            underlier_col="sym",
        )

    assert len(batch) == n
    assert counts["option"] == 0, "the array path materialized scalar contracts"
    assert counts["ref"] == 0, "the array path materialized underlier references"


def test_instruments_round_trips_back_to_scalar_contracts() -> None:
    options = scalar_options()
    assert EuropeanOptionBatch.from_instruments(options).instruments() == tuple(options)


# --- equality semantics ------------------------------------------------------


def test_batches_have_no_equality_operator() -> None:
    a, b = arrays_batch(), arrays_batch()
    assert a is not b
    assert a != b  # identity comparison, deliberately
    assert a.equals(b)


def test_batches_are_identity_hashable() -> None:
    batch = arrays_batch()
    assert len({batch, batch}) == 1


def test_equals_detects_a_differing_column() -> None:
    assert not arrays_batch().equals(arrays_batch(strike=[95.0, 100.0, 106.0]))
    assert not arrays_batch().equals("not a batch")


# --- coordinates --------------------------------------------------------------


def test_time_to_maturity_for_a_scalar_and_a_batch() -> None:
    option = scalar_options()[0]
    assert time_to_maturity(option) == option.maturity
    np.testing.assert_array_equal(time_to_maturity(arrays_batch()), np.array(MATURITIES))


@pytest.mark.parametrize(
    ("model", "expected"),
    [("black", 100.0), ("black_scholes", 100.0 * np.exp(0.05))],
    ids=["black", "black_scholes"],
)
def test_forward_price_depends_on_the_explicit_model(model: str, expected: float) -> None:
    market = VanillaMarketInputs(underlying=100.0, rate=0.05)
    np.testing.assert_allclose(float(forward_price(market, 1.0, model=model)), expected)


def test_forward_price_under_bsm_uses_the_dividend_yield() -> None:
    market = VanillaMarketInputs(underlying=100.0, rate=0.05, dividend_yield=0.02)
    np.testing.assert_allclose(
        float(forward_price(market, 1.0, model="black_scholes_merton")),
        100.0 * np.exp(0.03),
    )


def test_forward_price_without_a_dividend_yield_says_so() -> None:
    from fast_vollib.instruments import MissingMarketInputError

    market = VanillaMarketInputs(underlying=100.0, rate=0.05)
    with pytest.raises(MissingMarketInputError, match="dividend_yield"):
        forward_price(market, 1.0, model="black_scholes_merton")


def test_log_moneyness_is_the_surface_convention() -> None:
    """fast_vollib.surface is parametrized in k = log(K/F)."""
    market = VanillaMarketInputs(underlying=100.0, rate=0.0)
    batch = arrays_batch()
    np.testing.assert_allclose(
        log_moneyness(batch, market, model="black_scholes"),
        np.log(np.array(STRIKES) / 100.0),
    )
    np.testing.assert_allclose(
        log_moneyness(batch, market, model="black_scholes"),
        np.log(moneyness(batch, market, model="black_scholes")),
    )


def test_moneyness_uses_the_forward_not_the_spot() -> None:
    market = VanillaMarketInputs(underlying=100.0, rate=0.05)
    option = EuropeanOption(underlier="SPX", option_type="c", strike=100.0, maturity=1.0)
    np.testing.assert_allclose(
        float(moneyness(option, market, model="black_scholes")), np.exp(-0.05)
    )
    assert float(moneyness(option, market, model="black")) == 1.0


def test_moneyness_is_undefined_without_a_strike() -> None:
    market = VanillaMarketInputs(underlying=100.0, rate=0.0)
    forward = Forward(underlier="SPX", delivery_price=100.0, maturity=1.0)
    with pytest.raises(InstrumentValidationError, match="strike"):
        moneyness(forward, market, model="black_scholes")


def test_coordinates_are_pure_functions_of_their_arguments() -> None:
    batch = arrays_batch()
    market = VanillaMarketInputs(underlying=100.0, rate=0.01)
    first = np.array(log_moneyness(batch, market, model="black_scholes"))
    second = np.array(log_moneyness(batch, market, model="black_scholes"))
    np.testing.assert_array_equal(first, second)
    assert not hasattr(batch, "_cached_moneyness")
