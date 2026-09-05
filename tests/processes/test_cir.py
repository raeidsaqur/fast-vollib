"""``CIRShortRate``: the contract, the refusals, and the three schemes.

The numerical claim -- that the simulated rate reproduces the analytic bond
price -- lives in ``test_cir_convergence.py``, which measures the two error
sources apart.  This file is about everything that has to be right before that
measurement means anything: the shape and dtype contract, which parameters are
refused and when, that a refusal costs no randomness, and that the JAX key
discipline is actually followed rather than merely intended.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fast_vollib._simulation_errors import (
    SimulationValidationError,
    UnsupportedProcessError,
)
from fast_vollib.processes import CIR_SCHEMES, CIRShortRate, StochasticProcess

PARAMETERS = {"kappa": 0.3, "theta": 0.04, "volatility": 0.1}
INITIAL = {"short_rate": 0.04}
GRID = np.linspace(0.0, 1.0, 5)
SEED = 20260904

#: Feller-violating *and* below one degree of freedom, which is the branch of
#: the exact transition that has no normal in it.  ``d = 4*0.3*0.04/0.25**2``.
FEW_DEGREES = {"kappa": 0.3, "theta": 0.04, "volatility": 0.25}


def sample(process=None, *, scheme="quadratic_exponential", n_paths=64, **kwargs):
    process = process or CIRShortRate(**PARAMETERS)
    return process.sample(
        initial_state=INITIAL,
        time_grid=GRID,
        n_paths=n_paths,
        rng=SEED,
        scheme=scheme,
        **kwargs,
    )


# --- the contract --------------------------------------------------------------


def test_the_process_satisfies_the_structural_protocol() -> None:
    assert isinstance(CIRShortRate(**PARAMETERS), StochasticProcess)


def test_it_evolves_one_named_state() -> None:
    assert CIRShortRate.state_names == ("short_rate",)


def test_params_returns_the_original_objects_read_only() -> None:
    kappa = np.float64(0.3)
    process = CIRShortRate(kappa=kappa, theta=0.04, volatility=0.1)
    assert process.params()["kappa"] is kappa
    with pytest.raises(TypeError):
        process.params()["kappa"] = 1.0  # type: ignore[index]


@pytest.mark.parametrize("scheme", CIR_SCHEMES)
def test_every_scheme_produces_the_documented_shape(scheme) -> None:
    """The trailing axis has length one, matching ``state_names``."""
    assert sample(scheme=scheme).shape == (64, len(GRID), 1)


@pytest.mark.parametrize("scheme", CIR_SCHEMES)
def test_column_zero_is_the_initial_state_exactly(scheme) -> None:
    """Exactly: the sampler must not arrive at ``r0`` by computing with it."""
    paths = sample(scheme=scheme)
    assert np.all(paths[:, 0, 0] == INITIAL["short_rate"])


@pytest.mark.parametrize("scheme", ["quadratic_exponential", "exact_transition"])
def test_the_rate_never_goes_negative_under_the_non_euler_schemes(scheme) -> None:
    """CIR cannot produce a negative rate; two of the three schemes preserve that."""
    assert np.all(sample(scheme=scheme, n_paths=4_000) >= 0.0)


def test_full_truncation_euler_may_go_negative_and_that_is_its_definition() -> None:
    """Named rather than hidden: the state is truncated in the coefficients only.

    Documented as the scheme's behaviour, so a reader who sees a negative rate
    in a path knows which scheme produced it and why, rather than filing a bug.
    """
    paths = sample(
        CIRShortRate(kappa=0.3, theta=0.04, volatility=0.5),
        scheme="full_truncation_euler",
        n_paths=20_000,
    )
    assert paths.min() < 0.0


def test_a_process_is_a_value_holding_no_simulation_state() -> None:
    process = CIRShortRate(**PARAMETERS)
    with pytest.raises(Exception):
        process.kappa = 1.0  # type: ignore[misc]
    assert process == CIRShortRate(**PARAMETERS)
    forbidden = {"rng", "paths", "scenario", "device", "engine", "instrument"}
    assert forbidden.isdisjoint(set(process.__dataclass_fields__))


def test_sampling_twice_from_one_seed_gives_one_answer() -> None:
    np.testing.assert_array_equal(sample(), sample())


# --- the Feller ratio, reported and not enforced -------------------------------


def test_the_feller_ratio_is_reported_and_a_violation_still_prices() -> None:
    assert CIRShortRate(**PARAMETERS).feller_ratio == pytest.approx(2.4)
    assert CIRShortRate(**PARAMETERS).satisfies_feller

    violating = CIRShortRate(**FEW_DEGREES)
    assert not violating.satisfies_feller
    assert sample(violating).shape == (64, len(GRID), 1)


# --- the one-way curve ---------------------------------------------------------


def test_the_process_hands_back_the_analytic_curve_it_implies() -> None:
    from fast_vollib.rates import CIRDiscountCurve

    process = CIRShortRate(**PARAMETERS)
    curve = process.discount_curve(initial_rate=0.05)
    assert curve == CIRDiscountCurve(**PARAMETERS, initial_rate=0.05)


def test_the_initial_rate_is_an_argument_because_it_is_state_not_a_parameter() -> None:
    """The same dynamics from a different rate are the same process."""
    process = CIRShortRate(**PARAMETERS)
    assert "initial_rate" not in process.__dataclass_fields__
    with pytest.raises(TypeError):
        process.discount_curve(0.05)  # type: ignore[misc]


def test_importing_a_process_does_not_import_the_rates_package() -> None:
    """The curve import is function-local; the boundary test pins the source.

    This one pins the *behaviour*, which is what a user observes: a process
    module that reached into rates at import time would make every simulation
    pay for a valuation layer it may never call.
    """
    import subprocess
    import sys

    code = (
        "import sys; import fast_vollib.processes as p; "
        "before = 'fast_vollib.rates' in sys.modules; "
        "c = p.CIRShortRate(kappa=0.3, theta=0.04, volatility=0.1)"
        ".discount_curve(initial_rate=0.04); "
        "print(before, 'fast_vollib.rates' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "False True", out


# --- refusals ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kappa", 0.0, "kappa must be strictly positive"),
        ("theta", 0.0, "theta must be strictly positive"),
        ("volatility", 0.0, "volatility must be strictly positive"),
        ("volatility", -0.1, "volatility must be strictly positive"),
        ("kappa", float("nan"), "kappa must be finite"),
    ],
)
def test_an_invalid_parameter_is_refused_at_construction(field, value, message) -> None:
    with pytest.raises(SimulationValidationError, match=message):
        CIRShortRate(**{**PARAMETERS, field: value})


def test_a_zero_volatility_is_refused_and_the_limit_is_named() -> None:
    """The deterministic limit exists and is analytic; simulating it is not it.

    The curve accepts ``volatility=0`` and evaluates the limit exactly, so the
    process refusing it removes nothing a caller can do.
    """
    from fast_vollib.rates import CIRDiscountCurve

    with pytest.raises(SimulationValidationError):
        CIRShortRate(kappa=0.3, theta=0.04, volatility=0.0)
    exact = CIRDiscountCurve(kappa=0.3, theta=0.04, volatility=0.0, initial_rate=0.04)
    assert 0.0 < float(exact.discount_factor(1.0)) < 1.0


def test_an_unknown_scheme_is_refused_naming_the_ones_that_exist() -> None:
    with pytest.raises(SimulationValidationError, match="scheme must be one of"):
        sample(scheme="milstein")


def test_a_missing_initial_state_is_refused_naming_the_state() -> None:
    process = CIRShortRate(**PARAMETERS)
    with pytest.raises(SimulationValidationError, match="short_rate"):
        process.sample(initial_state={"spot": 100.0}, time_grid=GRID, n_paths=8, rng=SEED)


def test_a_negative_initial_rate_is_refused() -> None:
    process = CIRShortRate(**PARAMETERS)
    with pytest.raises(SimulationValidationError, match="non-negative"):
        process.sample(initial_state={"short_rate": -0.01}, time_grid=GRID, n_paths=8, rng=SEED)


# --- the exact transition's own refusals ---------------------------------------


def test_exact_transition_is_refused_on_torch_naming_the_reason() -> None:
    torch = pytest.importorskip("torch")
    process = CIRShortRate(**PARAMETERS)
    with pytest.raises(UnsupportedProcessError, match="generator"):
        process.sample(
            initial_state={"short_rate": torch.tensor(0.04, dtype=torch.float64)},
            time_grid=torch.as_tensor(GRID),
            n_paths=8,
            rng=SEED,
            scheme="exact_transition",
        )


def test_the_torch_refusal_happens_before_the_generator_is_touched() -> None:
    """ "Refused before any draw" is a promise about the caller's generator too."""
    torch = pytest.importorskip("torch")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    before = generator.get_state().clone()
    with pytest.raises(UnsupportedProcessError):
        CIRShortRate(**PARAMETERS).sample(
            initial_state={"short_rate": torch.tensor(0.04, dtype=torch.float64)},
            time_grid=torch.as_tensor(GRID),
            n_paths=8,
            rng=generator,
            scheme="exact_transition",
        )
    assert torch.equal(generator.get_state(), before)


@pytest.mark.parametrize("scheme", ["quadratic_exponential", "full_truncation_euler"])
def test_the_two_discretized_schemes_do_run_on_torch(scheme) -> None:
    """So the refusal above is about one scheme, not about the backend."""
    torch = pytest.importorskip("torch")
    paths = CIRShortRate(**PARAMETERS).sample(
        initial_state={"short_rate": torch.tensor(0.04, dtype=torch.float64)},
        time_grid=torch.as_tensor(GRID),
        n_paths=64,
        rng=SEED,
        scheme=scheme,
    )
    assert paths.shape == (64, len(GRID), 1)
    assert paths.dtype == torch.float64


def test_antithetic_exact_transition_is_refused_below_one_degree_of_freedom() -> None:
    """There is no normal in that construction, so a pair would be two copies.

    Refusing is the honest answer: the run would cost twice the draws and
    reduce no variance, and the alternative -- reporting ``n / 2`` effective
    samples for ``n`` identical paths -- would be arithmetically defensible
    and practically a waste nobody asked for.
    """
    process = CIRShortRate(**FEW_DEGREES)
    assert 4.0 * 0.3 * 0.04 / 0.25**2 < 1.0
    with pytest.raises(UnsupportedProcessError, match="degrees of freedom"):
        sample(process, scheme="exact_transition", antithetic=True)


def test_the_refused_antithetic_run_is_available_as_half_an_ordinary_one() -> None:
    """The refusal message says so, so it has to be true."""
    process = CIRShortRate(**FEW_DEGREES)
    half = sample(process, scheme="exact_transition", n_paths=32)
    assert half.shape == (32, len(GRID), 1)


def test_antithetic_exact_transition_is_allowed_above_one_degree_of_freedom() -> None:
    """There the construction contains a normal, and mirroring it is genuine."""
    paths = sample(scheme="exact_transition", n_paths=64, antithetic=True)
    assert not np.array_equal(paths[:32], paths[32:])


def test_exact_transition_is_refused_for_traced_parameters_naming_alternatives() -> None:
    """The branch is a choice between two formulas, not a rounding detail."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    def run(kappa):
        return CIRShortRate(kappa=kappa, theta=0.04, volatility=0.1).sample(
            initial_state={"short_rate": 0.04},
            time_grid=jnp.asarray(GRID),
            n_paths=8,
            rng=jax.random.key(SEED),
            scheme="exact_transition",
        )

    with pytest.raises(UnsupportedProcessError, match="traced parameters"):
        jax.jit(run)(jnp.asarray(0.3))


@pytest.mark.parametrize("scheme", ["quadratic_exponential", "full_truncation_euler"])
def test_the_discretized_schemes_trace_because_they_branch_on_nothing(scheme) -> None:
    """The alternative the refusal above points at has to actually work."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    def run(kappa):
        return CIRShortRate(kappa=kappa, theta=0.04, volatility=0.1).sample(
            initial_state={"short_rate": 0.04},
            time_grid=jnp.asarray(GRID),
            n_paths=16,
            rng=jax.random.key(SEED),
            scheme=scheme,
        )

    assert jax.jit(run)(jnp.asarray(0.3)).shape == (16, len(GRID), 1)


# --- JAX key discipline --------------------------------------------------------


@pytest.mark.parametrize("scheme", ["quadratic_exponential", "full_truncation_euler"])
def test_a_single_block_sampler_draws_from_the_key_it_was_given(scheme) -> None:
    """Not from a split of it: splitting would change the numbers for nothing.

    Verified against the primitive rather than asserted about the source: the
    normals a one-block scheme uses are exactly ``standard_normal`` on the
    caller's own stream.
    """
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from fast_vollib._array_api import get_namespace
    from fast_vollib._random_api import random_stream, standard_normal
    from fast_vollib.processes._square_root import (
        full_truncation_step,
        quadratic_exponential_step,
    )

    key = jax.random.key(SEED)
    grid = jnp.asarray(GRID)
    process = CIRShortRate(**PARAMETERS)
    produced = process.sample(
        initial_state=INITIAL, time_grid=grid, n_paths=8, rng=key, scheme=scheme
    )

    stream = random_stream(key, namespace="jax", dtype=produced.dtype)
    normals = standard_normal(stream, (8, len(GRID) - 1))
    xp = get_namespace(produced)
    r = xp.zeros((8,), like=normals) + xp.asarray(0.04, like=normals)
    step = quadratic_exponential_step if scheme == "quadratic_exponential" else None
    for index in range(len(GRID) - 1):
        dt = xp.asarray(grid[index + 1] - grid[index], like=normals)
        args = dict(
            kappa=xp.asarray(0.3, like=normals),
            theta=xp.asarray(0.04, like=normals),
            xi=xp.asarray(0.1, like=normals),
            dt=dt,
            z=normals[:, index],
        )
        r = step(xp, r, **args) if step else full_truncation_step(xp, r, **args)[0]
        np.testing.assert_array_equal(np.asarray(produced[:, index + 1, 0]), np.asarray(r))


def test_a_multi_block_sampler_never_draws_from_the_parent_key() -> None:
    """The children are derived from the parent's own output.

    So a block drawn from the parent alongside blocks drawn from its children
    is reusing the randomness the children were built from. This pins that the
    exact transition's two blocks come from a split rather than from the key it
    was handed -- a failure mode invisible to any distributional test, because
    every block is marginally correct however it was derived.

    The first step is reconstructed outright, which makes the check an equality
    rather than an inequality:
    ``r_1 = c((Z + sqrt(lambda))^2 + 2 Gamma((d-1)/2))``.
    """
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from fast_vollib._random_api import gamma, random_stream, split, standard_normal

    key = jax.random.key(SEED)
    produced = CIRShortRate(**PARAMETERS).sample(
        initial_state=INITIAL,
        time_grid=jnp.asarray(GRID),
        n_paths=8,
        rng=key,
        scheme="exact_transition",
    )

    kappa, theta, sigma, r0 = 0.3, 0.04, 0.1, 0.04
    d = 4.0 * kappa * theta / (sigma * sigma)
    assert d > 1.0, "this test is about the branch that draws two blocks up front"
    dt = float(GRID[1] - GRID[0])
    decay = math.exp(-kappa * dt)
    c = sigma * sigma * (1.0 - decay) / (4.0 * kappa)
    root_lambda = math.sqrt(r0 * decay / c)

    def first_step(normal_stream, gamma_stream):
        z = np.asarray(standard_normal(normal_stream, (8, len(GRID) - 1)))[:, 0]
        chi = 2.0 * np.asarray(gamma(gamma_stream, (8, len(GRID) - 1), 0.5 * (d - 1.0)))[:, 0]
        return c * ((z + root_lambda) ** 2 + chi)

    stream = random_stream(key, namespace="jax", dtype=produced.dtype)
    children = split(stream, 2)
    expected = np.asarray(produced[:, 1, 0])
    # Not bitwise: the reconstruction forms ``lambda`` from host constants where
    # the sampler forms it in the backend, so the two agree to rounding rather
    # than to the bit. The claim being made is which key each block came from.
    np.testing.assert_allclose(first_step(*children), expected, rtol=1e-10)

    # Substituting the parent for either child must not reproduce the output,
    # and neither must swapping the two children -- otherwise the test would
    # pass for a sampler that split correctly and then used the keys the other
    # way round, or one that ignored the split entirely.
    for wrong in (
        (stream, children[1]),
        (children[0], stream),
        (children[1], children[0]),
    ):
        assert not np.allclose(first_step(*wrong), expected, rtol=1e-3)


def test_the_poisson_branch_draws_each_child_key_exactly_once() -> None:
    """The ``d <= 1`` branch, whose first step is peeled for this reason.

    Every later step forms its Poisson rate from the state it starts from and
    so needs an array drawn already; the first does not, because every path
    starts at the same rate and ``lambda_0`` is a scalar. Drawing a throwaway
    template would consume ``streams[0]`` twice, which no distributional test
    would notice. The first step is reconstructed here instead.
    """
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from fast_vollib._random_api import gamma, poisson, random_stream, split

    process = CIRShortRate(**FEW_DEGREES)
    kappa, theta, sigma = FEW_DEGREES.values()
    d = 4.0 * kappa * theta / (sigma * sigma)
    assert d <= 1.0, "this test is about the Poisson-mixture branch"

    key = jax.random.key(SEED)
    produced = process.sample(
        initial_state=INITIAL,
        time_grid=jnp.asarray(GRID),
        n_paths=8,
        rng=key,
        scheme="exact_transition",
    )

    r0 = INITIAL["short_rate"]
    dt = float(GRID[1] - GRID[0])
    decay = math.exp(-kappa * dt)
    c = sigma * sigma * (1.0 - decay) / (4.0 * kappa)

    stream = random_stream(key, namespace="jax", dtype=produced.dtype)
    streams = split(stream, 2 * (len(GRID) - 1))

    def first_step(count_stream, gamma_stream):
        counts = poisson(count_stream, (8,), 0.5 * r0 * (decay / c))
        return c * 2.0 * np.asarray(gamma(gamma_stream, (8,), 0.5 * d + counts))

    expected = np.asarray(produced[:, 1, 0])
    np.testing.assert_allclose(first_step(streams[0], streams[1]), expected, rtol=1e-10)
    # Reusing one child for both draws, or taking the pair in the other order,
    # must not reproduce it -- which is what makes the check above load-bearing.
    for wrong in ((streams[0], streams[0]), (streams[1], streams[0])):
        assert not np.allclose(first_step(*wrong), expected, rtol=1e-3)


def test_a_traced_time_grid_is_refused_by_the_exact_transition() -> None:
    """The step lengths are read as host floats to form ``c`` and the decay.

    Without this the failure surfaces as a raw ``ConcretizationTypeError`` from
    inside the loop, after the refusal point the scheme promises.
    """
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    def run(grid):
        return CIRShortRate(**PARAMETERS).sample(
            initial_state=INITIAL,
            time_grid=grid,
            n_paths=8,
            rng=jax.random.key(SEED),
            scheme="exact_transition",
        )

    with pytest.raises(UnsupportedProcessError, match="traced time grid"):
        jax.jit(run)(jnp.asarray(GRID))


@pytest.mark.parametrize("scheme", CIR_SCHEMES)
def test_a_one_point_grid_is_the_initial_state_under_every_scheme(scheme) -> None:
    """A grid with no step has no transition, and the scheme must not change
    what the grid means. ``Heston`` and ``GBM`` behave the same way."""
    paths = CIRShortRate(**PARAMETERS).sample(
        initial_state=INITIAL,
        time_grid=np.array([0.0]),
        n_paths=4,
        rng=SEED,
        scheme=scheme,
    )
    assert paths.shape == (4, 1, 1)
    assert np.all(paths[:, 0, 0] == INITIAL["short_rate"])
