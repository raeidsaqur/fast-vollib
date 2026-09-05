"""The Poisson, gamma, and split primitives.

``test_random.py`` covers ``standard_normal`` and the stream plumbing.  These
three are separate because they break one of that module's assumptions on
purpose: a Poisson count cannot be mirrored and a gamma variate cannot be
negated, so antithetic sampling here *duplicates* the first half rather than
reflecting it.  Most of what follows is about that difference and about
``split``, which exists because a JAX key must never be drawn from twice.
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_vollib._random_api import (
    RandomStream,
    UnsupportedProcessError,
    gamma,
    poisson,
    random_stream,
    split,
    standard_normal,
)
from fast_vollib._simulation_errors import SimulationValidationError

SEED = 20260904


def numpy_stream(dtype=np.float64) -> RandomStream:
    return random_stream(SEED, namespace="numpy", dtype=dtype)


# --- the laws ------------------------------------------------------------------


@pytest.mark.parametrize("rate", [0.25, 1.0, 7.5])
def test_poisson_has_the_mean_and_variance_of_its_rate(rate) -> None:
    """Both equal ``lambda``, which is a property no other law on hand shares."""
    draws = poisson(numpy_stream(), (200_000,), rate)
    assert draws.mean() == pytest.approx(rate, rel=0.02)
    assert draws.var() == pytest.approx(rate, rel=0.03)


def test_poisson_returns_whole_numbers_in_a_floating_dtype() -> None:
    """Counts, but typed for the arithmetic they enter rather than for storage."""
    draws = poisson(numpy_stream(), (5_000,), 3.0)
    assert draws.dtype == np.float64
    assert np.all(draws == np.floor(draws))
    assert np.all(draws >= 0.0)


def test_a_per_path_poisson_rate_is_honoured_path_by_path() -> None:
    """The shape a short-rate transition needs: one rate per path, per step."""
    rates = np.concatenate([np.full(50_000, 0.5), np.full(50_000, 20.0)])
    draws = poisson(numpy_stream(), (100_000,), rates)
    assert draws[:50_000].mean() == pytest.approx(0.5, rel=0.05)
    assert draws[50_000:].mean() == pytest.approx(20.0, rel=0.02)


@pytest.mark.parametrize("alpha", [0.3, 1.0, 4.0])
def test_gamma_has_the_mean_and_variance_of_its_shape_at_unit_scale(alpha) -> None:
    """``mean = var = alpha`` is what pins the scale as one rather than two."""
    draws = gamma(numpy_stream(), (200_000,), alpha)
    assert draws.mean() == pytest.approx(alpha, rel=0.02)
    assert draws.var() == pytest.approx(alpha, rel=0.04)
    assert np.all(draws > 0.0)


def test_twice_a_gamma_is_the_chi_square_it_is_used_as() -> None:
    """``chi2_k = 2 * Gamma(k/2)`` -- the identity the exact transition rests on.

    Checked against the chi-square moments (mean ``k``, variance ``2k``)
    rather than against another gamma, so it is a statement about the law and
    not about this function agreeing with itself.
    """
    k = 5.0
    draws = 2.0 * gamma(numpy_stream(), (200_000,), 0.5 * k)
    assert draws.mean() == pytest.approx(k, rel=0.02)
    assert draws.var() == pytest.approx(2.0 * k, rel=0.05)


# --- antithetic duplication ----------------------------------------------------


@pytest.mark.parametrize("draw", [poisson, gamma], ids=["poisson", "gamma"])
def test_the_second_half_is_a_duplicate_and_not_a_mirror(draw) -> None:
    """Negating either law leaves its support, so mirroring is unavailable."""
    values = draw(numpy_stream(), (8, 3), 2.0, antithetic=True)
    np.testing.assert_array_equal(values[:4], values[4:])


@pytest.mark.parametrize("draw", [poisson, gamma], ids=["poisson", "gamma"])
def test_the_antithetic_half_matches_an_ordinary_draw_of_half_the_size(draw) -> None:
    """The reproducibility claim ``standard_normal`` makes, kept here.

    Only half the numbers are drawn, so a stateful generator advances by half
    as much and an antithetic run of ``n`` paths reproduces an ordinary run of
    ``n / 2``.
    """
    doubled = draw(numpy_stream(), (10, 2), 1.5, antithetic=True)
    plain = draw(numpy_stream(), (5, 2), 1.5)
    np.testing.assert_array_equal(doubled[:5], plain)


@pytest.mark.parametrize("draw", [poisson, gamma], ids=["poisson", "gamma"])
def test_an_odd_leading_axis_is_refused(draw) -> None:
    with pytest.raises(SimulationValidationError, match="even number of paths"):
        draw(numpy_stream(), (7,), 1.0, antithetic=True)


def test_a_per_path_parameter_is_cut_down_to_the_paths_actually_drawn() -> None:
    """Lossless, because the two halves of any per-path quantity are equal here.

    Every draw taken before this one was mirrored or duplicated, so a rate
    array's second half repeats its first. Slicing it is what lets a caller
    pass the full-width state without knowing the antithetic rule.
    """
    rates = np.array([0.5, 20.0, 0.5, 20.0])
    drawn = poisson(numpy_stream(), (4,), rates, antithetic=True)
    expected = poisson(numpy_stream(), (2,), np.array([0.5, 20.0]))
    np.testing.assert_array_equal(drawn, np.concatenate([expected, expected]))


def test_a_scalar_parameter_is_left_alone_under_antithetic() -> None:
    values = gamma(numpy_stream(), (6,), 2.0, antithetic=True)
    np.testing.assert_array_equal(values[:3], values[3:])


# --- backends ------------------------------------------------------------------


def test_torch_draws_poisson_from_the_generator_it_was_given() -> None:
    """Reproducible, which is the whole reason gamma is refused below."""
    torch = pytest.importorskip("torch")
    rates = torch.full((6,), 3.0, dtype=torch.float64)
    first = poisson(random_stream(SEED, namespace="torch", dtype=torch.float64), (6,), rates)
    second = poisson(random_stream(SEED, namespace="torch", dtype=torch.float64), (6,), rates)
    assert torch.equal(first, second)
    assert first.dtype == torch.float64


def test_torch_gamma_is_refused_by_name_rather_than_drawn_globally() -> None:
    """torch publishes no generator-bound gamma sampler.

    ``torch.distributions.Gamma.sample`` takes no generator and reads the
    global torch stream, so honouring the request would silently break the
    reproducibility every other draw in this library guarantees. Refusing is
    the only answer that does not lie.
    """
    torch = pytest.importorskip("torch")
    stream = random_stream(SEED, namespace="torch", dtype=torch.float64)
    with pytest.raises(UnsupportedProcessError, match="no public generator-bound gamma"):
        gamma(stream, (4,), 2.0)


def test_torch_gamma_is_refused_before_the_generator_is_advanced() -> None:
    """A refusal that consumed randomness would not be free to retry after."""
    torch = pytest.importorskip("torch")
    stream = random_stream(SEED, namespace="torch", dtype=torch.float64)
    before = stream.handle.get_state().clone()
    with pytest.raises(UnsupportedProcessError):
        gamma(stream, (4,), 2.0)
    assert torch.equal(stream.handle.get_state(), before)


@pytest.mark.parametrize("draw", [poisson, gamma], ids=["poisson", "gamma"])
def test_jax_draws_both_laws_from_a_key(draw) -> None:
    jax = pytest.importorskip("jax")
    stream = random_stream(jax.random.key(SEED), namespace="jax")
    values = draw(stream, (2_000,), 2.0)
    assert values.shape == (2_000,)
    assert float(values.mean()) == pytest.approx(2.0, rel=0.1)


# --- split ---------------------------------------------------------------------


def test_split_hands_a_stateful_generator_back_unchanged() -> None:
    """NumPy and torch need no children: two draws from one handle differ already."""
    stream = numpy_stream()
    children = split(stream, 3)
    assert len(children) == 3
    assert all(child.handle is stream.handle for child in children)


def test_split_consumes_nothing_from_a_stateful_generator() -> None:
    stream = numpy_stream()
    before = stream.handle.bit_generator.state
    split(stream, 4)
    assert stream.handle.bit_generator.state == before


def test_split_gives_jax_children_that_draw_differently() -> None:
    """The property the same handle would not have: a key is not stateful."""
    jax = pytest.importorskip("jax")
    stream = random_stream(jax.random.key(SEED), namespace="jax")
    children = split(stream, 3)
    drawn = [float(standard_normal(child, (1,))[0]) for child in children]
    assert len(set(drawn)) == 3, drawn


@pytest.mark.parametrize("build", ["key", "PRNGKey"])
def test_a_split_child_is_a_single_key_for_both_key_flavours(build) -> None:
    """A batch is not a key; indexing it is what turns it back into one.

    A typed batch has shape ``(n,)`` with scalar elements and a legacy uint32
    batch has shape ``(n, 2)`` with ``(2,)`` elements, and ``jax.random``
    refuses the batch itself.
    """
    jax = pytest.importorskip("jax")
    from fast_vollib._random_api import _is_jax_key

    stream = random_stream(getattr(jax.random, build)(SEED), namespace="jax")
    for child in split(stream, 3):
        assert _is_jax_key(child.handle), child.handle.shape
        assert standard_normal(child, (2,)).shape == (2,)


def test_split_carries_the_stream_device_and_dtype_to_its_children() -> None:
    jax = pytest.importorskip("jax")

    stream = random_stream(jax.random.key(SEED), namespace="jax", dtype=None)
    for child in split(stream, 2):
        assert child.namespace == "jax"
        assert child.device == stream.device
        assert child.dtype == stream.dtype


def test_splitting_the_same_key_twice_gives_the_same_children() -> None:
    """Determinism, which is the reason to split rather than to reseed."""
    jax = pytest.importorskip("jax")
    stream = random_stream(jax.random.key(SEED), namespace="jax")
    first = [float(standard_normal(c, (1,))[0]) for c in split(stream, 3)]
    second = [float(standard_normal(c, (1,))[0]) for c in split(stream, 3)]
    assert first == second


@pytest.mark.parametrize("n", [0, -1, 1.5, True, "2"])
def test_split_refuses_a_count_that_is_not_a_positive_integer(n) -> None:
    with pytest.raises(SimulationValidationError, match="positive integer"):
        split(numpy_stream(), n)
