"""SurfaceQuotes: ownership, validation, conversion, and explicit alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fast_vollib.diagnostics import SurfaceQuotes, align_predictions


def test_arrays_are_owned_copies_and_read_only():
    k = np.array([-0.1, 0.0, 0.1])
    quotes = SurfaceQuotes(k=k, T=[0.5, 0.5, 0.5], iv=[0.2, 0.2, 0.2])
    k[0] = 99.0  # mutating the caller's array must not change the container
    assert quotes.k[0] == pytest.approx(-0.1)
    for name in ("k", "T", "iv"):
        array = getattr(quotes, name)
        assert array.flags.writeable is False
        assert array.dtype == np.float64
        with pytest.raises(ValueError):
            array[0] = 1.0


def test_scalar_fields_broadcast_to_every_row():
    quotes = SurfaceQuotes(k=[0.0, 0.1], T=1.0, iv=0.2, surface_id="AAA", is_call=True)
    assert quotes.T.tolist() == [1.0, 1.0]
    assert quotes.iv.tolist() == [0.2, 0.2]
    assert quotes.surface_id.tolist() == ["AAA", "AAA"]
    assert quotes.is_call.tolist() == [True, True]


def test_empty_quote_set_is_valid():
    quotes = SurfaceQuotes(k=[], T=[], iv=[])
    assert quotes.n == 0
    assert len(quotes) == 0
    assert quotes.surface_ids() == []
    assert list(quotes.smiles()) == []


def test_missing_iv_is_allowed_and_infinite_iv_is_not():
    assert np.isnan(SurfaceQuotes(k=[0.0], T=[1.0], iv=[np.nan]).iv[0])
    with pytest.raises(ValueError, match="infinities are rejected"):
        SurfaceQuotes(k=[0.0], T=[1.0], iv=[np.inf])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"k": [np.inf], "T": [1.0], "iv": [0.2]}, "k must be finite"),
        ({"k": [0.0], "T": [np.nan], "iv": [0.2]}, "T must be finite"),
        ({"k": [0.0], "T": [0.0], "iv": [0.2]}, "T must be strictly positive"),
        ({"k": [0.0], "T": [-1.0], "iv": [0.2]}, "T must be strictly positive"),
        ({"k": [0.0], "T": [1.0], "iv": [-0.01]}, "non-negative"),
        ({"k": [0.0, 0.0], "T": [1.0], "iv": [0.2, 0.2]}, "same length"),
    ],
)
def test_coordinate_validation_is_unconditional(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SurfaceQuotes(**kwargs)


def test_coordinates_are_validated_even_where_iv_is_missing():
    # A missing observation does not excuse an unusable coordinate.
    with pytest.raises(ValueError, match="T must be strictly positive"):
        SurfaceQuotes(k=[0.0], T=[0.0], iv=[np.nan])


@pytest.mark.parametrize("labels", [[True, False], [1.5, 2.5], [None, "a"]])
def test_labels_must_be_json_scalars(labels):
    with pytest.raises((TypeError, ValueError)):
        SurfaceQuotes(k=[0.0, 0.0], T=[1.0, 1.0], iv=[0.2, 0.2], surface_id=labels)


def test_integer_labels_keep_an_integer_dtype():
    quotes = SurfaceQuotes(k=[0.0, 0.0], T=[1.0, 1.0], iv=[0.2, 0.2], surface_id=[7, 7])
    assert quotes.surface_id.dtype == np.int64
    assert quotes.surface_ids() == [7]


def test_quote_ids_must_be_unique_within_a_surface_but_may_repeat_across_them():
    SurfaceQuotes(
        k=[0.0, 0.0],
        T=[1.0, 1.0],
        iv=[0.2, 0.2],
        surface_id=["A", "B"],
        point_id=["q1", "q1"],
    )
    with pytest.raises(ValueError, match="unique within each surface_id"):
        SurfaceQuotes(
            k=[0.0, 0.0],
            T=[1.0, 1.0],
            iv=[0.2, 0.2],
            surface_id=["A", "A"],
            point_id=["q1", "q1"],
        )


def test_duplicate_coordinates_are_preserved_as_separate_observations():
    quotes = SurfaceQuotes(k=[0.0, 0.0], T=[1.0, 1.0], iv=[0.20, 0.22])
    assert quotes.n == 2
    assert quotes.iv.tolist() == [0.20, 0.22]


@pytest.mark.parametrize(
    "value, expected",
    [
        ([True, False], [True, False]),
        ([1, 0], [True, False]),
        ([1.0, 0.0], [True, False]),
        (["true", "false"], [True, False]),
        (["Call", "PUT"], [True, False]),
        (["c", "p"], [True, False]),
        ([" C ", "P"], [True, False]),
    ],
)
def test_is_call_accepts_only_canonical_encodings(value, expected):
    quotes = SurfaceQuotes(k=[0.0, 0.0], T=[1.0, 1.0], iv=[0.2, 0.2], is_call=value)
    assert quotes.is_call.tolist() == expected


@pytest.mark.parametrize("value", [["yes", "no"], [2, 0], [0.5, 0.0], [None, None]])
def test_is_call_rejects_arbitrary_truthiness(value):
    with pytest.raises((TypeError, ValueError)):
        SurfaceQuotes(k=[0.0, 0.0], T=[1.0, 1.0], iv=[0.2, 0.2], is_call=value)


def test_spread_rules():
    SurfaceQuotes(k=[0.0], T=[1.0], iv=[0.2], bid=[np.nan], ask=[np.nan])  # paired missing
    with pytest.raises(ValueError, match="given together"):
        SurfaceQuotes(k=[0.0], T=[1.0], iv=[0.2], bid=[0.1])
    with pytest.raises(ValueError, match="missing together"):
        SurfaceQuotes(k=[0.0], T=[1.0], iv=[0.2], bid=[np.nan], ask=[0.2])
    with pytest.raises(ValueError, match="non-negative"):
        SurfaceQuotes(k=[0.0], T=[1.0], iv=[0.2], bid=[-0.1], ask=[0.2])
    with pytest.raises(ValueError, match="must not exceed ask"):
        SurfaceQuotes(k=[0.0], T=[1.0], iv=[0.2], bid=[0.3], ask=[0.2])


def test_from_strikes_validates_positivity_and_computes_log_moneyness():
    quotes = SurfaceQuotes.from_strikes([90.0, 110.0], [1.0, 1.0], [0.2, 0.2], forward=100.0)
    assert quotes.k.tolist() == pytest.approx(np.log(np.array([0.9, 1.1])).tolist())
    with pytest.raises(ValueError, match="K must be finite and strictly positive"):
        SurfaceQuotes.from_strikes([0.0], [1.0], [0.2], forward=100.0)
    with pytest.raises(ValueError, match="forward must be finite and strictly positive"):
        SurfaceQuotes.from_strikes([100.0], [1.0], [0.2], forward=-1.0)


def test_dataframe_round_trip_preserves_order_labels_ids_and_booleans():
    quotes = SurfaceQuotes(
        k=[0.1, -0.1, 0.0],
        T=[1.0, 1.0, 2.0],
        iv=[0.2, np.nan, 0.25],
        surface_id=["B", "B", "A"],
        point_id=[3, 1, 2],
        bid=[0.01, np.nan, 0.02],
        ask=[0.02, np.nan, 0.03],
        is_call=["call", "put", "c"],
    )
    frame = quotes.to_dataframe()
    assert frame["is_call"].dtype == bool
    restored = SurfaceQuotes.from_dataframe(frame)
    assert restored.k.tolist() == quotes.k.tolist()
    assert restored.surface_id.tolist() == quotes.surface_id.tolist()
    assert restored.point_id.tolist() == quotes.point_id.tolist()
    assert restored.is_call.tolist() == quotes.is_call.tolist()
    assert np.isnan(restored.iv[1])


def test_from_dataframe_reads_the_surface_label_from_a_named_index():
    frame = pd.DataFrame(
        {"logmoneyness": [0.0, 0.1], "maturity": [1.0, 1.0], "iv": [0.2, 0.21]},
        index=pd.Index(["S", "S"], name="id"),
    )
    assert SurfaceQuotes.from_dataframe(frame).surface_ids() == ["S"]


def test_smiles_group_by_rounded_maturity_and_sort_strikes():
    quotes = SurfaceQuotes(
        k=[0.1, -0.1, 0.0],
        T=[0.5000001, 0.5, 0.5],
        iv=[0.21, 0.22, np.nan],
        surface_id="A",
    )
    smiles = list(quotes.smiles(maturity_decimals=3))
    assert len(smiles) == 1
    _, maturity, strikes, vols = smiles[0]
    assert maturity == pytest.approx(0.5)
    assert strikes.tolist() == [-0.1, 0.1]  # the NaN row is dropped, k ascending
    assert vols.tolist() == [0.22, 0.21]


# -- alignment --------------------------------------------------------------
def _truth():
    return SurfaceQuotes(
        k=[-0.1, 0.0, 0.1],
        T=[0.5, 0.5, 0.5],
        iv=[0.22, 0.20, 0.21],
        surface_id="A",
        point_id=["a", "b", "c"],
    )


def _shuffled_predictions():
    return SurfaceQuotes(
        k=[0.1, -0.1, 0.0],
        T=[0.5, 0.5, 0.5],
        iv=[np.nan, np.nan, np.nan],
        surface_id="A",
        point_id=["c", "a", "b"],
    )


def test_alignment_by_stable_ids_reorders_raw_values():
    raw = np.array([0.31, 0.11, -0.5])
    aligned = align_predictions(_truth(), _shuffled_predictions(), values=raw)
    assert aligned.tolist() == [0.11, -0.5, 0.31]


def test_alignment_returns_unsanitized_values():
    # A negative prediction survives alignment so the evaluator can count it.
    raw = np.array([0.31, 0.11, -0.5])
    assert align_predictions(_truth(), _shuffled_predictions(), values=raw).min() < 0.0


def test_alignment_by_unique_coordinates_matches_id_alignment():
    raw = np.array([0.31, 0.11, -0.5])
    by_id = align_predictions(_truth(), _shuffled_predictions(), values=raw)
    by_coordinates = align_predictions(
        _truth(), _shuffled_predictions(), values=raw, on="coordinates"
    )
    assert by_id.tolist() == by_coordinates.tolist()


def test_alignment_defaults_to_ids_and_requires_them():
    without_ids = SurfaceQuotes(k=[-0.1, 0.0, 0.1], T=[0.5] * 3, iv=[0.2] * 3, surface_id="A")
    with pytest.raises(ValueError, match="requires point_id on both sides"):
        align_predictions(_truth(), without_ids)


def test_coordinate_alignment_refuses_duplicate_coordinates():
    truth = SurfaceQuotes(k=[0.0, 0.0], T=[0.5, 0.5], iv=[0.2, 0.22], surface_id="A")
    predictions = SurfaceQuotes(k=[0.0, 0.0], T=[0.5, 0.5], iv=[0.2, 0.22], surface_id="A")
    with pytest.raises(ValueError, match="unique .* keys on both sides"):
        align_predictions(truth, predictions, on="coordinates")


def test_alignment_rejects_missing_and_extra_rows():
    truth = _truth()
    short = SurfaceQuotes(
        k=[-0.1, 0.0], T=[0.5, 0.5], iv=[0.2, 0.2], surface_id="A", point_id=["a", "b"]
    )
    with pytest.raises(ValueError, match="no prediction"):
        align_predictions(truth, short)
    long = SurfaceQuotes(
        k=[-0.1, 0.0, 0.1, 0.2],
        T=[0.5] * 4,
        iv=[0.2] * 4,
        surface_id="A",
        point_id=["a", "b", "c", "d"],
    )
    with pytest.raises(ValueError, match="match no truth row"):
        align_predictions(truth, long)


def test_alignment_never_falls_back_to_row_order():
    # Same coordinates, different ids: alignment must fail rather than pair by position.
    predictions = SurfaceQuotes(
        k=[-0.1, 0.0, 0.1],
        T=[0.5] * 3,
        iv=[0.2] * 3,
        surface_id="A",
        point_id=["x", "y", "z"],
    )
    with pytest.raises(ValueError, match="no prediction"):
        align_predictions(_truth(), predictions)


def test_alignment_rejects_an_unknown_key():
    with pytest.raises(ValueError, match="on must be one of"):
        align_predictions(_truth(), _shuffled_predictions(), on="nearest")


def test_alignment_rejects_mismatched_values_length():
    with pytest.raises(ValueError, match="values must have shape"):
        align_predictions(_truth(), _shuffled_predictions(), values=np.array([1.0, 2.0]))
