"""Backend selection, RNG validation, and the antithetic draw layout."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fast_vollib._random_api import (
    RandomStream,
    namespace_of,
    random_stream,
    resolve_device,
    resolve_namespace,
    standard_normal,
)
from fast_vollib.simulation.errors import SimulationValidationError


def numpy_stream(seed: int = 11) -> RandomStream:
    return random_stream(seed, namespace="numpy")


# --- which backend a value selects --------------------------------------------


@pytest.mark.parametrize(
    "value",
    [None, 1, 2.5, -3, [1.0, 2.0], (1.0, 2.0), np.float64(1.0), np.int64(3)],
    ids=["none", "int", "float", "negative", "list", "tuple", "np-scalar", "np-int"],
)
def test_python_and_numpy_scalars_are_neutral(value: Any) -> None:
    assert namespace_of(value) is None


def test_a_numpy_array_selects_numpy() -> None:
    assert namespace_of(np.zeros(3)) == "numpy"
    assert namespace_of(np.zeros(())) == "numpy", "a zero-dimensional array is still a buffer"


def test_a_numpy_generator_selects_numpy() -> None:
    assert namespace_of(np.random.default_rng(0)) == "numpy"


def test_a_torch_tensor_and_generator_select_torch() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    assert namespace_of(torch.zeros(3)) == "torch"
    assert namespace_of(torch.Generator()) == "torch"


def test_a_jax_array_and_key_select_jax() -> None:
    jax = pytest.importorskip("jax", reason="jax not installed")
    assert namespace_of(jax.numpy.zeros(3)) == "jax"
    assert namespace_of(jax.random.key(0)) == "jax"
    assert namespace_of(jax.random.PRNGKey(0)) == "jax"


# --- resolution ---------------------------------------------------------------


def test_all_neutral_input_defaults_to_numpy() -> None:
    assert resolve_namespace({"spot": 100.0, "grid": [0.0, 1.0], "rng": 7}) == "numpy"


def test_one_native_input_selects_its_namespace() -> None:
    assert resolve_namespace({"spot": 100.0, "grid": np.array([0.0, 1.0])}) == "numpy"


def test_mixed_namespaces_raise_before_anything_is_drawn() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    jax = pytest.importorskip("jax", reason="jax not installed")
    with pytest.raises(SimulationValidationError) as excinfo:
        resolve_namespace({"spot": torch.tensor(1.0), "grid": jax.numpy.zeros(2)})
    message = str(excinfo.value)
    assert "spot is torch" in message
    assert "grid is jax" in message


def test_a_torch_rng_conflicts_with_a_numpy_array() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    with pytest.raises(SimulationValidationError, match="more than one array namespace"):
        resolve_namespace({"grid": np.zeros(2), "rng": torch.Generator()})


def test_device_is_none_for_numpy_and_jax() -> None:
    assert resolve_device("numpy", {"spot": np.zeros(3)}) is None
    assert resolve_device("jax", {"spot": np.zeros(3)}) is None


def test_a_single_torch_device_is_reported() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    device = resolve_device("torch", {"spot": torch.zeros(3)})
    assert device == torch.device("cpu")


def test_a_torch_generator_supplies_the_device_when_there_are_no_tensors() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    generator = torch.Generator(device="cpu")
    assert resolve_device("torch", {"rng": generator, "spot": 100.0}) == torch.device("cpu")


def test_two_torch_devices_raise() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    with pytest.raises(SimulationValidationError, match="more than one torch device"):
        resolve_device("torch", {"a": torch.zeros(3), "b": torch.zeros(3, device="cuda")})


# --- RNG contracts ------------------------------------------------------------


def test_a_numpy_seed_creates_a_local_generator() -> None:
    """A seed must not read or advance any global random state."""
    np.random.seed(0)
    first = standard_normal(numpy_stream(5), (4,))
    np.random.seed(0)
    second = standard_normal(numpy_stream(5), (4,))
    np.testing.assert_array_equal(first, second)


def test_a_supplied_numpy_generator_advances() -> None:
    generator = np.random.default_rng(3)
    first = standard_normal(RandomStream("numpy", generator), (4,))
    second = standard_normal(RandomStream("numpy", generator), (4,))
    assert not np.array_equal(first, second)


@pytest.mark.parametrize("bad", [True, np.bool_(True)], ids=["bool", "np-bool"])
def test_a_bool_is_not_a_seed(bad: Any) -> None:
    with pytest.raises(SimulationValidationError, match="not a seed"):
        random_stream(bad, namespace="numpy")


def test_a_negative_seed_is_refused() -> None:
    with pytest.raises(SimulationValidationError, match="non-negative"):
        random_stream(-1, namespace="numpy")


def test_a_legacy_random_state_is_refused() -> None:
    with pytest.raises(SimulationValidationError, match="RandomState is not accepted"):
        random_stream(np.random.RandomState(0), namespace="numpy")


def test_an_unusable_rng_object_is_refused() -> None:
    with pytest.raises(SimulationValidationError, match="numpy.random.Generator"):
        random_stream("seed", namespace="numpy")


def test_torch_seed_and_generator_agree() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    from_seed = standard_normal(random_stream(9, namespace="torch"), (3, 2))
    generator = torch.Generator()
    generator.manual_seed(9)
    from_generator = standard_normal(random_stream(generator, namespace="torch"), (3, 2))
    torch.testing.assert_close(from_seed, from_generator, rtol=0, atol=0)


def test_a_torch_generator_on_the_wrong_device_raises() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    generator = torch.Generator(device="cpu")
    with pytest.raises(SimulationValidationError, match="torch generator is on"):
        random_stream(generator, namespace="torch", device=torch.device("cuda"))


def test_jax_refuses_an_integer_seed_and_says_why() -> None:
    pytest.importorskip("jax", reason="jax not installed")
    with pytest.raises(SimulationValidationError) as excinfo:
        random_stream(0, namespace="jax")
    assert "jax.random.key" in str(excinfo.value)


def test_jax_accepts_typed_and_legacy_keys() -> None:
    jax = pytest.importorskip("jax", reason="jax not installed")
    for key in (jax.random.key(0), jax.random.PRNGKey(0)):
        stream = random_stream(key, namespace="jax")
        assert stream.namespace == "jax"
        assert standard_normal(stream, (3,)).shape == (3,)


def test_jax_refuses_a_batch_of_typed_keys_before_drawing() -> None:
    jax = pytest.importorskip("jax", reason="jax not installed")
    keys = jax.random.split(jax.random.key(0), 2)
    with pytest.raises(SimulationValidationError, match="one PRNG key"):
        random_stream(keys, namespace="jax")


def test_a_jax_key_reproduces_rather_than_advancing() -> None:
    """Keys are immutable: the same key is the same draw, by design."""
    jax = pytest.importorskip("jax", reason="jax not installed")
    stream = random_stream(jax.random.key(4), namespace="jax")
    np.testing.assert_array_equal(
        np.asarray(standard_normal(stream, (5,))),
        np.asarray(standard_normal(stream, (5,))),
    )


def test_an_ordinary_jax_array_is_not_a_key() -> None:
    jax = pytest.importorskip("jax", reason="jax not installed")
    with pytest.raises(SimulationValidationError, match="PRNG key"):
        random_stream(jax.numpy.zeros(3), namespace="jax")


def test_an_unknown_namespace_is_refused() -> None:
    with pytest.raises(SimulationValidationError, match="Unknown array namespace"):
        random_stream(0, namespace="cupy")


# --- the antithetic layout ----------------------------------------------------


def test_antithetic_draws_are_negated_by_halves() -> None:
    drawn = standard_normal(numpy_stream(2), (6, 4), antithetic=True)
    assert drawn.shape == (6, 4)
    np.testing.assert_array_equal(drawn[3:], -drawn[:3])


def test_antithetic_first_half_equals_an_ordinary_half_size_draw() -> None:
    """The pairing is exactly "draw m, mirror m", never a reordering."""
    ordinary = standard_normal(numpy_stream(2), (3, 4))
    paired = standard_normal(numpy_stream(2), (6, 4), antithetic=True)
    np.testing.assert_array_equal(paired[:3], ordinary)


def test_antithetic_consumes_only_the_half_sample_draw() -> None:
    """A stateful generator must advance by m paths, not 2m."""
    generator = np.random.default_rng(17)
    standard_normal(RandomStream("numpy", generator), (5, 4), antithetic=False)
    after_ordinary = generator.standard_normal(size=(2,))

    generator = np.random.default_rng(17)
    standard_normal(RandomStream("numpy", generator), (10, 4), antithetic=True)
    after_antithetic = generator.standard_normal(size=(2,))

    np.testing.assert_array_equal(after_ordinary, after_antithetic)


def test_antithetic_requires_an_even_leading_axis() -> None:
    with pytest.raises(SimulationValidationError, match="even number of paths"):
        standard_normal(numpy_stream(), (5, 3), antithetic=True)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_numpy_dtype_is_honoured(dtype_name: str) -> None:
    dtype = getattr(np, dtype_name)
    stream = random_stream(1, namespace="numpy", dtype=dtype)
    assert standard_normal(stream, (4, 2)).dtype == dtype


def test_torch_dtype_and_device_are_honoured() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    stream = random_stream(1, namespace="torch", device=torch.device("cpu"), dtype=torch.float32)
    drawn = standard_normal(stream, (4, 2), antithetic=True)
    assert drawn.dtype == torch.float32
    assert drawn.device == torch.device("cpu")
    torch.testing.assert_close(drawn[2:], -drawn[:2], rtol=0, atol=0)


def test_jax_antithetic_layout_and_dtype() -> None:
    jax = pytest.importorskip("jax", reason="jax not installed")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    stream = random_stream(jax.random.key(1), namespace="jax", dtype=jnp.float64)
    drawn = standard_normal(stream, (4, 3), antithetic=True)
    assert drawn.dtype == jnp.float64
    np.testing.assert_array_equal(np.asarray(drawn[2:]), -np.asarray(drawn[:2]))


def test_streams_are_not_compared_across_backends() -> None:
    """Reproducibility is per backend. The library must not imply otherwise."""
    pytest.importorskip("torch", reason="torch not installed")
    from_numpy = np.asarray(standard_normal(random_stream(0, namespace="numpy"), (8,)))
    from_torch = standard_normal(random_stream(0, namespace="torch"), (8,)).numpy()
    assert not np.allclose(from_numpy, from_torch)


# --- precision --------------------------------------------------------------


def test_a_dtype_numpy_cannot_draw_directly_is_still_honoured() -> None:
    """``Generator.standard_normal`` takes only float32 and float64.

    Anything else is drawn at double precision and cast, rather than refused:
    the dtype is the caller's statement about their buffer, not about which
    sampler kernel exists.
    """
    stream = random_stream(1, namespace="numpy", dtype=np.float16)
    drawn = standard_normal(stream, (4, 2))
    assert drawn.dtype == np.float16
    assert np.all(np.isfinite(drawn))


def test_mixed_native_precisions_promote_rather_than_pick_one() -> None:
    from fast_vollib._random_api import resolve_dtype

    assert (
        resolve_dtype(
            "numpy", {"a": np.zeros(3, dtype=np.float32), "b": np.zeros(3, dtype=np.float64)}
        )
        == np.float64
    )


def test_torch_mixed_precisions_promote() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    from fast_vollib._random_api import resolve_dtype

    promoted = resolve_dtype(
        "torch",
        {"a": torch.zeros(3, dtype=torch.float32), "b": torch.zeros(3, dtype=torch.float64)},
    )
    assert promoted == torch.float64


def test_jax_mixed_precisions_promote() -> None:
    jax = pytest.importorskip("jax", reason="jax not installed")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    from fast_vollib._random_api import resolve_dtype

    promoted = resolve_dtype(
        "jax", {"a": jnp.zeros(3, dtype=jnp.float32), "b": jnp.zeros(3, dtype=jnp.float64)}
    )
    assert promoted == jnp.float64


def test_a_numpy_scalar_promotes_only_where_the_backend_promotes() -> None:
    """numpy and jax treat a NumPy scalar as strong; torch treats it as weak."""
    torch = pytest.importorskip("torch", reason="torch not installed")
    from fast_vollib._random_api import resolve_dtype

    single_numpy = {"grid": np.zeros(3, dtype=np.float32), "spot": np.float64(1.0)}
    assert resolve_dtype("numpy", single_numpy) == np.float64

    single_torch = {"grid": torch.zeros(3, dtype=torch.float32), "spot": np.float64(1.0)}
    assert resolve_dtype("torch", single_torch) == torch.float32


def test_an_integer_numpy_scalar_does_not_set_a_floating_dtype() -> None:
    from fast_vollib._random_api import resolve_dtype

    assert resolve_dtype("numpy", {"n": np.int64(3)}) is None
