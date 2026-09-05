"""Backend-native randomness, normalized once for every sampler.

Random number generation is the one part of a simulation that cannot be
written against a single array namespace.  NumPy advances a stateful
:class:`numpy.random.Generator`; torch advances a :class:`torch.Generator`
that is bound to a device; JAX has no state at all and threads an immutable
key, so reusing one reproduces a draw rather than continuing a stream.  Those
are three different contracts, not three spellings of one, which is why they
are here instead of in :mod:`fast_vollib._array_api`.

What this module fixes is the part that *is* shared: which backend a call is
running in, on which device, in which dtype, and whether the caller's RNG is
usable there.  Both :mod:`fast_vollib.processes` and
:mod:`fast_vollib.simulation` depend on it and neither depends on the other.

Backend selection
-----------------
Native arrays and native generators/keys *select* a backend; Python numbers,
lists, tuples, and integer seeds are neutral.  If nothing selects one the
result is NumPy.  If two things select different ones that is an error, raised
before any draw: silently promoting a torch tensor and a jax array into one of
the two would move a caller's data off its device without saying so.

Reproducibility is promised within one backend and library version.  It is
never promised *across* backends: three generators with the same seed produce
three different streams, and nothing here pretends otherwise.

Importing this module pulls in nothing beyond numpy.  torch and jax are
imported lazily, and only once a value has already identified itself as
belonging to one of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ._simulation_errors import SimulationValidationError, UnsupportedProcessError

__all__ = [
    "NAMESPACES",
    "RandomStream",
    "namespace_of",
    "normalize_device",
    "random_stream",
    "resolve_device",
    "resolve_dtype",
    "resolve_namespace",
    "gamma",
    "poisson",
    "split",
    "standard_normal",
]

#: The array namespaces a simulation can run in, in preference order.
NAMESPACES = ("numpy", "torch", "jax")


def _module_root(value: Any) -> str:
    return type(value).__module__.partition(".")[0]


def _is_torch(value: Any) -> bool:
    return _module_root(value) == "torch"


def _is_jax(value: Any) -> bool:
    return _module_root(value) in {"jax", "jaxlib"}


def namespace_of(value: Any) -> str | None:
    """The backend ``value`` selects, or ``None`` when it is neutral.

    Neutral means "usable from any backend": ``None``, Python numbers and
    containers, integer seeds, and NumPy scalar types.  A NumPy *array* --
    including a zero-dimensional one -- selects NumPy, because it is a real
    buffer whose namespace a caller can observe.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib._random_api import namespace_of
    >>> namespace_of(1.0) is None
    True
    >>> namespace_of(np.zeros(3))
    'numpy'
    """
    if value is None:
        return None
    if _is_torch(value):
        return "torch"
    if _is_jax(value):
        return "jax"
    if isinstance(value, (np.ndarray, np.random.Generator, np.random.RandomState)):
        return "numpy"
    return None


def resolve_namespace(inputs: Mapping[str, Any]) -> str:
    """The one backend every selecting input agrees on; NumPy if none do.

    Parameters
    ----------
    inputs : Mapping[str, Any]
        Named values, so a conflict can say which two disagreed.

    Raises
    ------
    SimulationValidationError
        If two inputs select different backends.
    """
    chosen: dict[str, str] = {}
    for label, value in inputs.items():
        name = namespace_of(value)
        if name is not None and name not in chosen.values():
            chosen[label] = name
    if len(chosen) > 1:
        detail = ", ".join(f"{label} is {name}" for label, name in chosen.items())
        raise SimulationValidationError(
            f"Simulation inputs span more than one array namespace ({detail}). "
            f"Convert them to a single namespace first; nothing is moved between "
            f"backends implicitly, because that would silently relocate a caller's "
            f"data and drop its autograd tape."
        )
    return next(iter(chosen.values()), "numpy")


def resolve_device(namespace: str, inputs: Mapping[str, Any]) -> Any | None:
    """The torch data device, or a generator's device when no tensor selects one.

    NumPy and JAX return ``None``: NumPy has no device, and JAX placement is
    decided by its own committal rules rather than by this library.

    Raises
    ------
    SimulationValidationError
        If torch tensors, or otherwise the supplied generators, span more than
        one device.
    """
    if namespace != "torch":
        return None
    import torch

    devices: dict[str, Any] = {}
    for label, value in inputs.items():
        # Data takes precedence. A generator is checked *against* the data
        # rather than voting on where it is, unless there is no tensor below.
        if not torch.is_tensor(value):
            continue
        device = normalize_device(value.device)
        devices.setdefault(str(device), (label, device))
    if len(devices) > 1:
        detail = ", ".join(f"{label} on {name}" for name, (label, _d) in devices.items())
        raise SimulationValidationError(
            f"Simulation inputs span more than one torch device ({detail}). "
            f"Move them onto one device first."
        )
    for _name, (_label, device) in devices.items():
        return device

    # With no tensor to decide placement, a supplied generator is the only
    # native object in the request and therefore the only device information
    # available. Honour it so a CUDA generator plus otherwise-Python inputs
    # builds the grid on CUDA instead of manufacturing a CPU mismatch itself.
    generator_devices: dict[str, Any] = {}
    for label, value in inputs.items():
        if isinstance(value, torch.Generator):
            device = normalize_device(value.device)
            generator_devices.setdefault(str(device), (label, device))
    if len(generator_devices) > 1:
        detail = ", ".join(
            f"{label} on {name}" for name, (label, _device) in generator_devices.items()
        )
        raise SimulationValidationError(
            f"Simulation generators span more than one torch device ({detail})."
        )
    for _name, (_label, device) in generator_devices.items():
        return device
    return None


def resolve_dtype(namespace: str, inputs: Mapping[str, Any]) -> Any | None:
    """The floating dtype the native inputs promote to, or ``None``.

    ``None`` means "use the backend's own default", which is what an
    all-Python-scalar call gets.  When natives disagree -- a float32 spot with
    a float64 volatility -- the promoted type is used, matching what the
    arithmetic would have produced anyway.  Choosing one input's dtype instead
    would make the *draws* single-precision while everything they multiply is
    double, which is a silent precision loss rather than a promotion.
    """
    dtypes = [
        dtype
        for value in inputs.values()
        if (dtype := _contributing_dtype(namespace, value)) is not None
    ]
    if not dtypes:
        return None
    promoted = dtypes[0]
    for dtype in dtypes[1:]:
        promoted = _promote(namespace, promoted, dtype)
    return promoted


#: Backends whose arithmetic treats a NumPy scalar as a strong type that
#: promotes the array it meets, rather than as a weak wrapped number. torch is
#: absent deliberately: ``float32_tensor * np.float64(2.0)`` stays float32
#: there, so counting the scalar would make the draws double the precision of
#: everything they multiply -- the very mismatch this function prevents.
_NUMPY_SCALARS_PROMOTE = frozenset({"numpy", "jax"})


def _contributing_dtype(namespace: str, value: Any) -> Any | None:
    """The floating dtype ``value`` contributes to ``namespace``'s arithmetic.

    Native arrays of the namespace contribute their own dtype. NumPy scalars
    are backend-neutral for *selection* -- they do not choose a backend -- but
    they are not neutral for *promotion* in numpy and jax, so they contribute
    there and not in torch. Anything else contributes nothing, which is what
    makes an all-Python-scalar call run at the backend's default precision.
    """
    if namespace_of(value) == namespace:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and _is_floating(namespace, dtype):
            return dtype
        return None
    if (
        namespace in _NUMPY_SCALARS_PROMOTE
        and isinstance(value, np.generic)
        and np.issubdtype(value.dtype, np.floating)
    ):
        return value.dtype
    return None


def _is_floating(namespace: str, dtype: Any) -> bool:
    if namespace == "torch":
        import torch

        return bool(torch.is_floating_point(torch.empty(0, dtype=dtype)))
    if namespace == "jax":
        import jax.numpy as jnp

        return bool(jnp.issubdtype(dtype, jnp.floating))
    return bool(np.issubdtype(dtype, np.floating))


def _promote(namespace: str, left: Any, right: Any) -> Any:
    if left == right:
        return left
    if namespace == "torch":
        import torch

        return torch.promote_types(left, right)
    if namespace == "jax":
        import jax.numpy as jnp

        return jnp.promote_types(left, right)
    return np.promote_types(left, right)


def normalize_device(device: Any) -> Any:
    """A torch device with its index resolved, so ``cuda`` equals ``cuda:0``.

    ``torch.device("cuda")`` means "the current CUDA device" and compares
    unequal to the ``cuda:0`` a tensor reports, even when they are the same
    piece of hardware. Comparing the resolved forms keeps a generator built the
    convenient way from being rejected as a device mismatch.
    """
    import torch

    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return resolved


@dataclass(frozen=True, slots=True)
class RandomStream:
    """A validated source of standard normal draws for one backend.

    Attributes
    ----------
    namespace : {"numpy", "torch", "jax"}
    handle : object
        A :class:`numpy.random.Generator`, a :class:`torch.Generator`, or a
        JAX PRNG key.  NumPy and torch handles are stateful and advance as they
        are drawn from; a JAX key does not, so drawing from the same stream
        twice reproduces the same numbers.  Splitting a key is the caller's
        job, exactly as it is in JAX itself.
    device : object or None
        The torch device draws are placed on; ``None`` for the other backends.
    dtype : object or None
        The floating dtype draws are produced in; ``None`` means the
        backend's own default.
    """

    namespace: str
    handle: Any
    device: Any | None = None
    dtype: Any | None = None


def _reject_bool(seed: Any) -> None:
    if isinstance(seed, (bool, np.bool_)):
        raise SimulationValidationError(
            "rng must be a generator, a PRNG key, or a non-negative integer seed; "
            "a bool is not a seed."
        )


def _as_seed(value: Any) -> int | None:
    """``value`` as a non-negative integer seed, or ``None`` if it is not one."""
    _reject_bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        seed = int(value)
        if seed < 0:
            raise SimulationValidationError(
                f"An integer seed must be non-negative; got {seed}. Pass a generator "
                f"if you need to continue an existing stream."
            )
        return seed
    return None


def _is_jax_key(value: Any) -> bool:
    """Whether ``value`` is a JAX PRNG key, typed or legacy ``uint32[2]``."""
    if not _is_jax(value):
        return False
    import jax
    import jax.numpy as jnp

    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return False
    if jnp.issubdtype(dtype, jax.dtypes.prng_key):
        # A typed key has scalar shape. ``jax.random.split`` returns a batch of
        # typed keys, which random.normal refuses unless the caller vmaps it.
        return getattr(value, "shape", None) == ()
    return dtype == jnp.uint32 and getattr(value, "shape", None) == (2,)


def random_stream(
    rng: Any,
    *,
    namespace: str,
    device: Any | None = None,
    dtype: Any | None = None,
) -> RandomStream:
    """Validate ``rng`` for ``namespace`` and bind it to a device and dtype.

    Parameters
    ----------
    rng : object
        A generator or key native to ``namespace``, or a non-negative integer
        seed for the two backends that accept one.  A seed always creates a
        *local* generator: no global random state is read or advanced.
    namespace : {"numpy", "torch", "jax"}
    device, dtype : object, optional
        Where draws are placed and in what precision.  Defaults come from the
        backend.

    Raises
    ------
    SimulationValidationError
        If the RNG is not usable for this backend, or a torch generator is on
        a different device from the data.
    """
    if namespace == "numpy":
        return _numpy_stream(rng, dtype=dtype)
    if namespace == "torch":
        return _torch_stream(rng, device=device, dtype=dtype)
    if namespace == "jax":
        return _jax_stream(rng, dtype=dtype)
    raise SimulationValidationError(
        f"Unknown array namespace {namespace!r}; expected one of {', '.join(NAMESPACES)}."
    )


def _numpy_stream(rng: Any, *, dtype: Any | None) -> RandomStream:
    if isinstance(rng, np.random.Generator):
        return RandomStream("numpy", rng, None, dtype)
    seed = _as_seed(rng)
    if seed is not None:
        return RandomStream("numpy", np.random.default_rng(seed), None, dtype)
    raise SimulationValidationError(
        f"The NumPy backend needs a numpy.random.Generator or a non-negative integer "
        f"seed; got {type(rng).__name__}. A legacy RandomState is not accepted: its "
        f"stream guarantees differ from Generator's."
    )


def _torch_stream(rng: Any, *, device: Any | None, dtype: Any | None) -> RandomStream:
    import torch

    if isinstance(rng, torch.Generator):
        if device is not None and normalize_device(rng.device) != normalize_device(device):
            raise SimulationValidationError(
                f"The torch generator is on {rng.device} but the simulation inputs are "
                f"on {device}. torch draws from a generator bound to one device; move "
                f"one of them rather than having the library guess which was meant."
            )
        return RandomStream("torch", rng, rng.device, dtype)
    seed = _as_seed(rng)
    if seed is not None:
        generator = torch.Generator(device=device if device is not None else "cpu")
        generator.manual_seed(seed)
        return RandomStream("torch", generator, generator.device, dtype)
    raise SimulationValidationError(
        f"The torch backend needs a torch.Generator or a non-negative integer seed; "
        f"got {type(rng).__name__}."
    )


def _jax_stream(rng: Any, *, dtype: Any | None) -> RandomStream:
    if _is_jax_key(rng):
        return RandomStream("jax", rng, None, dtype)
    if _is_jax(rng):
        import jax
        import jax.numpy as jnp

        key_dtype = getattr(rng, "dtype", None)
        if key_dtype is not None and jnp.issubdtype(key_dtype, jax.dtypes.prng_key):
            raise SimulationValidationError(
                f"The JAX backend needs one PRNG key; got a batch with shape "
                f"{getattr(rng, 'shape', None)}. Split and select one key per simulation, "
                f"or vmap the simulation explicitly."
            )
    if _as_seed(rng) is not None:
        raise SimulationValidationError(
            "The JAX backend needs an explicit PRNG key, not an integer seed. Build "
            "one with jax.random.key(seed) and split it yourself for independent "
            "streams -- a key is immutable, so the library cannot advance it for you "
            "and silently reusing one would repeat a draw."
        )
    raise SimulationValidationError(
        f"The JAX backend needs a PRNG key from jax.random.key or jax.random.PRNGKey; "
        f"got {type(rng).__name__}."
    )


def standard_normal(
    stream: RandomStream, shape: tuple[int, ...], *, antithetic: bool = False
) -> Any:
    """Standard normal draws of ``shape``, in the stream's own namespace.

    With ``antithetic=True`` the leading axis must be even: the first half is
    drawn and the second half is its exact negative, in order, so row ``i`` and
    row ``i + n/2`` are a matched pair.  Only half the numbers are drawn, so a
    stateful generator advances by half as much as an ordinary call of the same
    shape -- which is what makes an antithetic run reproducible against a
    half-size ordinary run.

    Raises
    ------
    SimulationValidationError
        If ``antithetic`` is requested with an odd leading axis.
    """
    if antithetic:
        if not shape or shape[0] % 2 != 0:
            raise SimulationValidationError(
                f"Antithetic sampling needs an even number of paths; got shape {shape}."
            )
        half = (shape[0] // 2,) + tuple(shape[1:])
        drawn = _draw(stream, half)
        return _concatenate_with_negative(stream, drawn)
    return _draw(stream, tuple(shape))


def _draw(stream: RandomStream, shape: tuple[int, ...]) -> Any:
    if stream.namespace == "numpy":
        dtype = stream.dtype if stream.dtype in (np.float32, np.float64) else None
        if dtype is None:
            values = stream.handle.standard_normal(size=shape)
            if stream.dtype is not None:
                return values.astype(stream.dtype, copy=False)
            return values
        return stream.handle.standard_normal(size=shape, dtype=dtype)
    if stream.namespace == "torch":
        import torch

        return torch.randn(
            shape,
            generator=stream.handle,
            dtype=stream.dtype,
            device=stream.device if stream.device is not None else stream.handle.device,
        )
    import jax

    if stream.dtype is None:
        return jax.random.normal(stream.handle, shape)
    return jax.random.normal(stream.handle, shape, dtype=stream.dtype)


def _concatenate_with_negative(stream: RandomStream, drawn: Any) -> Any:
    if stream.namespace == "torch":
        import torch

        return torch.cat((drawn, -drawn), dim=0)
    if stream.namespace == "jax":
        import jax.numpy as jnp

        return jnp.concatenate((drawn, -drawn), axis=0)
    return np.concatenate((drawn, -drawn), axis=0)


# --- non-Gaussian draws --------------------------------------------------------
#
# Two more laws, added for the exact CIR transition, which needs a Poisson count
# and a gamma variate where every other sampler in the library needs only
# normals.  They follow ``standard_normal``'s contract in every respect but one,
# and the exception is mathematical rather than stylistic.
#
# ``standard_normal`` implements antithetic sampling by *mirroring*: the second
# half is the exact negative of the first.  A negated Poisson count is not a
# Poisson count and a negated gamma variate is not a gamma variate, so mirroring
# is unavailable here and the second half is a *duplicate* of the first instead.
#
# That is not a weaker version of the same idea; it is what makes the idea work
# for a mixed draw.  In the exact transition the non-central chi-square is a
# normal plus an independent chi-square, and a pair of paths that shares its
# chi-square component while mirroring its normal is a genuine antithetic pair:
# the shared part cancels out of the difference exactly, leaving the mirrored
# part to do the variance reduction.  Duplicating rather than redrawing is what
# makes the sharing exact.


def _antithetic_half(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape or shape[0] % 2 != 0:
        raise SimulationValidationError(
            f"Antithetic sampling needs an even number of paths; got shape {shape}."
        )
    return (shape[0] // 2,) + tuple(shape[1:])


def _leading_half(parameter: Any, full_leading: int) -> Any:
    """A per-path parameter restricted to the paths that are actually drawn.

    Under antithetic sampling only the first half of the rows is drawn, so a
    parameter carrying one value per path has to be cut down to match.  Cutting
    it is lossless: every draw taken before this one was either mirrored or
    duplicated, so the two halves of any per-path quantity are equal by
    construction and the first half is the whole of the information.

    A scalar, or an array whose leading axis is not the path axis, is left
    alone -- it broadcasts against either width.
    """
    shape = getattr(parameter, "shape", None)
    if shape and shape[0] == full_leading:
        return parameter[: full_leading // 2]
    return parameter


def _duplicate(stream: "RandomStream", drawn: Any) -> Any:
    """The antithetic counterpart of ``_concatenate_with_negative``."""
    if stream.namespace == "torch":
        import torch

        return torch.cat((drawn, drawn), dim=0)
    if stream.namespace == "jax":
        import jax.numpy as jnp

        return jnp.concatenate((drawn, drawn), axis=0)
    return np.concatenate((drawn, drawn), axis=0)


def poisson(
    stream: RandomStream, shape: tuple[int, ...], rate: Any, *, antithetic: bool = False
) -> Any:
    """Poisson counts of ``shape`` at ``rate``, as floating point.

    The values are whole numbers and the dtype is the stream's floating dtype,
    not an integer one.  They are counts that go on to be multiplied by
    non-integers -- a chi-square degrees-of-freedom in the exact CIR transition
    -- and the three backends disagree about what integer division and integer
    promotion mean, while they agree exactly about float64.

    Parameters
    ----------
    rate : array-like
        The mean, broadcast against ``shape``.  A per-path rate is ordinary:
        in a short-rate transition it depends on the state the step starts
        from.
    antithetic : bool, default False
        Duplicate the first half of the rows into the second, per the module
        note above.  The leading axis must be even.

    Raises
    ------
    SimulationValidationError
        If ``antithetic`` is requested with an odd leading axis.
    """
    if antithetic:
        half = _antithetic_half(tuple(shape))
        drawn = _draw_poisson(stream, half, _leading_half(rate, shape[0]))
        return _duplicate(stream, drawn)
    return _draw_poisson(stream, tuple(shape), rate)


def gamma(
    stream: RandomStream, shape: tuple[int, ...], alpha: Any, *, antithetic: bool = False
) -> Any:
    """Gamma variates of ``shape`` with shape parameter ``alpha`` and unit scale.

    Unit scale, because every scale this library needs is applied afterwards by
    a multiplication the caller can see.  A chi-square with ``k`` degrees of
    freedom is ``2 * gamma(k / 2)``.

    Parameters
    ----------
    alpha : array-like
        The shape parameter, strictly positive, broadcast against ``shape``.
    antithetic : bool, default False
        Duplicate the first half of the rows into the second.

    Raises
    ------
    UnsupportedProcessError
        On torch, which has no public generator-bound gamma sampler.  Raised
        rather than silently drawing from the global torch stream, which would
        make a seeded run irreproducible.
    SimulationValidationError
        If ``antithetic`` is requested with an odd leading axis.
    """
    if stream.namespace == "torch":
        raise UnsupportedProcessError(
            "torch has no public generator-bound gamma sampler, so a gamma draw here "
            "could not honour the generator it was given: torch.distributions.Gamma.sample "
            "takes no generator and reads the global torch stream, which would make a "
            "seeded simulation silently irreproducible. Draw on NumPy or JAX, or use a "
            "scheme that needs only normals ('quadratic_exponential', "
            "'full_truncation_euler')."
        )
    if antithetic:
        half = _antithetic_half(tuple(shape))
        drawn = _draw_gamma(stream, half, _leading_half(alpha, shape[0]))
        return _duplicate(stream, drawn)
    return _draw_gamma(stream, tuple(shape), alpha)


def _float_dtype(stream: RandomStream) -> Any:
    """The stream's floating dtype, or the backend's default when it has none."""
    if stream.dtype is not None:
        return stream.dtype
    if stream.namespace == "torch":
        import torch

        return torch.get_default_dtype()
    if stream.namespace == "jax":
        import jax.numpy as jnp

        return jnp.zeros(()).dtype
    return np.float64


def _draw_poisson(stream: RandomStream, shape: tuple[int, ...], rate: Any) -> Any:
    dtype = _float_dtype(stream)
    if stream.namespace == "numpy":
        return stream.handle.poisson(lam=rate, size=shape).astype(dtype, copy=False)
    if stream.namespace == "torch":
        import torch

        device = stream.device if stream.device is not None else stream.handle.device
        rates = torch.as_tensor(rate, dtype=dtype, device=device)
        # ``torch.poisson`` takes the rates rather than a size, so the shape has
        # to be materialized. ``expand`` would share storage and the sampler
        # writes an output of the same shape, so it is broadcast into a real
        # tensor first.
        rates = torch.broadcast_to(rates, shape).contiguous()
        return torch.poisson(rates, generator=stream.handle)
    import jax
    import jax.numpy as jnp

    counts = jax.random.poisson(stream.handle, jnp.asarray(rate), shape=shape)
    return counts.astype(dtype)


def _draw_gamma(stream: RandomStream, shape: tuple[int, ...], alpha: Any) -> Any:
    dtype = _float_dtype(stream)
    if stream.namespace == "numpy":
        native = dtype if dtype in (np.float32, np.float64) else None
        if native is None:
            return stream.handle.standard_gamma(shape=alpha, size=shape).astype(dtype, copy=False)
        return stream.handle.standard_gamma(shape=alpha, size=shape, dtype=native)
    import jax
    import jax.numpy as jnp

    return jax.random.gamma(stream.handle, jnp.asarray(alpha), shape=shape, dtype=dtype)


def split(stream: RandomStream, n: int) -> tuple[RandomStream, ...]:
    """``n`` streams that draw independently of one another.

    The three backends need this for opposite reasons, which is why it is one
    function rather than a JAX special case at every call site.

    NumPy and torch generators are *stateful*: two draws from one generator are
    already independent, so the same handle is handed back ``n`` times and
    nothing is consumed here.  A JAX key is not: drawing twice from one key
    reproduces the draw, so real child keys are derived with
    ``jax.random.split``.  The parent must not be drawn from afterwards -- the
    children are derived from the parent's own output, so a draw from the
    parent would reuse the randomness the children were built from.

    A sampler that needs more than one block of draws therefore calls this once
    and uses one child per block.  A sampler that needs exactly one block does
    not call it at all, and uses the caller's stream directly.

    Raises
    ------
    SimulationValidationError
        If ``n`` is not a positive integer.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 1:
        raise SimulationValidationError(f"split needs a positive integer count; got {n!r}.")
    n = int(n)
    if stream.namespace != "jax":
        return tuple(stream for _ in range(n))
    import jax

    keys = jax.random.split(stream.handle, n)
    # Indexing the batch is what turns it back into single keys: a typed batch
    # has shape ``(n,)`` and each element has scalar shape, while a legacy
    # uint32 batch has shape ``(n, 2)`` and each element has shape ``(2,)``.
    # Both are what ``_is_jax_key`` accepts and what ``jax.random`` will draw
    # from; the batch itself is neither.
    return tuple(
        RandomStream("jax", keys[index], stream.device, stream.dtype) for index in range(n)
    )
