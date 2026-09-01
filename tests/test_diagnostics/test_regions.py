"""Region descriptors: explicit bounds, honest serialization, strict validation."""

from __future__ import annotations

import numpy as np
import pytest

from fast_vollib.diagnostics import DEFAULT_REGIONS, Box, NamedRegion, Region, liquid_mask


def test_the_default_split_is_the_inclusive_liquid_box():
    (box,) = DEFAULT_REGIONS
    assert (box.name, box.complement_name) == ("liquid", "illiquid")
    assert (box.k_min, box.k_max, box.T_min, box.T_max) == (-0.2, 0.2, None, 0.5)
    assert box.closed == "both"


def test_the_default_box_agrees_with_the_liquid_mask_helper():
    rng = np.random.default_rng(0)
    k = rng.uniform(-0.6, 0.6, 200)
    T = rng.uniform(0.01, 2.0, 200)
    assert DEFAULT_REGIONS[0].mask(k, T).tolist() == liquid_mask(k, T).tolist()


@pytest.mark.parametrize(
    "closed, expected",
    [
        ("both", [True, True, True]),
        ("left", [True, True, False]),
        ("right", [False, True, True]),
        ("neither", [False, True, False]),
    ],
)
def test_bound_inclusion_follows_the_closed_policy(closed, expected):
    box = Box(name="band", k_min=-0.1, k_max=0.1, closed=closed)
    k = np.array([-0.1, 0.0, 0.1])
    T = np.full(3, 0.5)
    assert box.mask(k, T).tolist() == expected


def test_an_unbounded_side_is_left_open():
    box = Box(name="long_dated", T_min=1.0)
    assert box.mask(np.array([-5.0, 5.0]), np.array([1.0, 2.0])).tolist() == [True, True]


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"name": ""}, ValueError),
        ({"name": "a", "complement_name": "a"}, ValueError),
        ({"name": "a", "closed": "diagonal"}, ValueError),
        ({"name": "a", "k_min": np.inf}, ValueError),
        ({"name": "a", "k_min": 0.5, "k_max": 0.1}, ValueError),
        ({"name": "a", "T_min": 2.0, "T_max": 1.0}, ValueError),
        ({"name": "a", "k_min": True}, TypeError),
        ({"name": "a", "k_min": "0.1"}, TypeError),
    ],
)
def test_box_validation(kwargs, error):
    with pytest.raises(error):
        Box(**kwargs)


def test_a_box_round_trips_through_its_descriptor():
    box = Box(name="core", complement_name="wings", k_min=-0.3, T_max=1.0, closed="right")
    descriptor = box.describe()
    assert descriptor["kind"] == "box"
    assert Box.from_describe(descriptor) == box


@pytest.mark.parametrize(
    "descriptor",
    [
        {
            "kind": "wedge",
            "name": "a",
            "complement_name": None,
            "k_min": None,
            "k_max": None,
            "T_min": None,
            "T_max": None,
            "closed": "both",
        },
        {"kind": "box", "name": "a"},
    ],
)
def test_from_describe_is_strict(descriptor):
    with pytest.raises(ValueError):
        Box.from_describe(descriptor)


def test_a_named_callable_region_is_usable_but_undescribable():
    region = NamedRegion(
        name="short", complement_name="long", predicate=lambda k, T: np.asarray(T) < 1.0
    )
    assert region.mask(np.array([0.0, 0.0]), np.array([0.5, 2.0])).tolist() == [True, False]
    assert region.describe() is None


def test_a_named_region_validates_its_predicate_output():
    region = NamedRegion(name="bad", complement_name=None, predicate=lambda k, T: np.array([True]))
    with pytest.raises(ValueError, match="must return shape"):
        region.mask(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    with pytest.raises(TypeError, match="predicate must be callable"):
        NamedRegion(name="bad", complement_name=None, predicate=42)


def test_mask_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="same length"):
        Box(name="a").mask(np.array([0.0, 0.0]), np.array([1.0]))


def test_both_region_kinds_satisfy_the_protocol():
    assert isinstance(Box(name="a"), Region)
    assert isinstance(NamedRegion(name="a", complement_name=None, predicate=lambda k, T: k), Region)
