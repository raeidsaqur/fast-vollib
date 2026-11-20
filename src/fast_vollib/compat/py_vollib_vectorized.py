from __future__ import annotations

import functools
import inspect
from typing import Any, Callable

import numpy as np

from ..api import get_all_greeks, price_dataframe
from ..greeks import delta, gamma, rho, theta, vega
from ..implied_volatility import fast_implied_volatility, fast_implied_volatility_black
from ..models import fast_black, fast_black_scholes, fast_black_scholes_merton


def patch_py_vollib_vectorized() -> None:
    """Replace py_vollib_vectorized functions with fast_vollib implementations.

    https://github.com/marcdemers/py_vollib_vectorized

    After calling this, any code that calls ``py_vollib_vectorized.vectorized_black_scholes(...)``
    (or any other exported function) will transparently use the fast_vollib backend.
    Signatures are identical, so no call-site changes are needed.

    Raises
    ------
    ImportError
        If ``py_vollib_vectorized`` is not installed.
    """
    try:
        import py_vollib_vectorized
        import py_vollib_vectorized.api
        import py_vollib_vectorized.greeks
        import py_vollib_vectorized.implied_volatility
        import py_vollib_vectorized.models
    except ImportError as exc:
        raise ImportError(
            "You must have py_vollib_vectorized installed to use patch_py_vollib_vectorized()."
        ) from exc

    # --- Pricing models ---
    py_vollib_vectorized.models.vectorized_black = fast_black
    py_vollib_vectorized.models.vectorized_black_scholes = fast_black_scholes
    py_vollib_vectorized.models.vectorized_black_scholes_merton = fast_black_scholes_merton
    py_vollib_vectorized.vectorized_black = fast_black
    py_vollib_vectorized.vectorized_black_scholes = fast_black_scholes
    py_vollib_vectorized.vectorized_black_scholes_merton = fast_black_scholes_merton

    # --- Implied volatility ---
    py_vollib_vectorized.implied_volatility.vectorized_implied_volatility = fast_implied_volatility
    py_vollib_vectorized.implied_volatility.vectorized_implied_volatility_black = (
        fast_implied_volatility_black
    )
    py_vollib_vectorized.vectorized_implied_volatility = fast_implied_volatility
    py_vollib_vectorized.vectorized_implied_volatility_black = fast_implied_volatility_black

    # --- Greeks ---
    py_vollib_vectorized.greeks.delta = delta
    py_vollib_vectorized.greeks.gamma = gamma
    py_vollib_vectorized.greeks.rho = rho
    py_vollib_vectorized.greeks.theta = theta
    py_vollib_vectorized.greeks.vega = vega
    py_vollib_vectorized.vectorized_delta = delta
    py_vollib_vectorized.vectorized_gamma = gamma
    py_vollib_vectorized.vectorized_rho = rho
    py_vollib_vectorized.vectorized_theta = theta
    py_vollib_vectorized.vectorized_vega = vega

    # --- High-level API ---
    py_vollib_vectorized.api.get_all_greeks = get_all_greeks
    py_vollib_vectorized.api.price_dataframe = price_dataframe
    py_vollib_vectorized.get_all_greeks = get_all_greeks
    py_vollib_vectorized.price_dataframe = price_dataframe


_PATCH_MARK = "__fast_vollib_adapter__"
#: fast-vollib keyword arguments accepted by the py_vollib adapters in addition to the upstream
#: signature (keyword-only).
_EXTRA_KWARGS = ("backend", "on_error", "dtype")


def _adapter(original: Callable[..., Any], call: Callable[[dict, dict], Any]) -> Callable[..., Any]:
    """Wrap ``call(arguments, extras)`` behind ``original``'s own signature.

    Arguments are bound against the upstream signature, so positional and keyword calls mean
    what they mean in py_vollib. The adapter carries ``__wrapped__ = original`` (so
    ``inspect.signature`` reports the upstream signature) and a marker that makes patching
    idempotent. Scalar inputs return a Python float, as py_vollib does; array inputs return a
    NumPy array.
    """
    if getattr(original, _PATCH_MARK, False):
        return original
    signature = inspect.signature(original)

    @functools.wraps(original)
    def adapted(*args: Any, **kwargs: Any) -> Any:
        extras = {key: kwargs.pop(key) for key in _EXTRA_KWARGS if key in kwargs}
        arguments = signature.bind(*args, **kwargs).arguments
        values = np.asarray(call(arguments, extras), dtype=np.float64)
        if all(np.ndim(v) == 0 for v in arguments.values()):
            return float(values.reshape(-1)[0])
        return values

    setattr(adapted, _PATCH_MARK, True)
    return adapted


def _greek(
    function: Callable[..., Any], model: str, underlying: str
) -> Callable[[dict, dict], Any]:
    def call(a: dict, extras: dict) -> Any:
        q = a.get("q") if model == "black_scholes_merton" else None
        return function(
            a["flag"],
            a[underlying],
            a["K"],
            a["t"],
            a["r"],
            a["sigma"],
            q,
            model=model,
            return_as="numpy",
            **extras,
        )

    return call


def patch_py_vollib() -> None:
    """Replace py_vollib (scalar) functions with fast_vollib implementations.

    Each replacement keeps py_vollib's signature and argument order (including
    ``black_scholes_merton.implied_volatility(price, S, K, t, r, q, flag)``), routes to the
    matching model (Black-76 greeks use the Black-76 model; Black-Scholes-Merton entries honour
    ``q`` on every backend), and returns a float for scalar inputs. fast_vollib's own functions
    are not modified, and calling this more than once has no further effect.

    The replacements also accept fast-vollib's ``backend=``, ``on_error=`` and ``dtype=`` as
    keywords.

    Raises
    ------
    ImportError
        If ``py_vollib`` is not installed.
    """
    try:
        import py_vollib.black
        import py_vollib.black.greeks.numerical
        import py_vollib.black.implied_volatility
        import py_vollib.black_scholes
        import py_vollib.black_scholes.greeks.numerical
        import py_vollib.black_scholes.implied_volatility
        import py_vollib.black_scholes_merton
        import py_vollib.black_scholes_merton.greeks.numerical
        import py_vollib.black_scholes_merton.implied_volatility
    except ImportError as exc:
        raise ImportError("You must have py_vollib installed to use patch_py_vollib().") from exc

    def patch(module: Any, name: str, call: Callable[[dict, dict], Any]) -> None:
        setattr(module, name, _adapter(getattr(module, name), call))

    numpy_out: dict[str, Any] = {"return_as": "numpy"}

    # --- Pricing ---
    patch(
        py_vollib.black,
        "black",
        lambda a, x: fast_black(
            a["flag"], a["F"], a["K"], a["t"], a["r"], a["sigma"], **numpy_out, **x
        ),
    )
    patch(
        py_vollib.black_scholes,
        "black_scholes",
        lambda a, x: fast_black_scholes(
            a["flag"], a["S"], a["K"], a["t"], a["r"], a["sigma"], **numpy_out, **x
        ),
    )
    patch(
        py_vollib.black_scholes_merton,
        "black_scholes_merton",
        lambda a, x: fast_black_scholes_merton(
            a["flag"], a["S"], a["K"], a["t"], a["r"], a["sigma"], a["q"], **numpy_out, **x
        ),
    )

    # --- Implied volatility ---
    patch(
        py_vollib.black.implied_volatility,
        "implied_volatility",
        lambda a, x: fast_implied_volatility_black(
            a["discounted_option_price"],
            a["F"],
            a["K"],
            a["r"],
            a["t"],
            a["flag"],
            **numpy_out,
            **x,
        ),
    )
    patch(
        py_vollib.black_scholes.implied_volatility,
        "implied_volatility",
        lambda a, x: fast_implied_volatility(
            a["price"],
            a["S"],
            a["K"],
            a["t"],
            a["r"],
            a["flag"],
            model="black_scholes",
            **numpy_out,
            **x,
        ),
    )
    patch(
        py_vollib.black_scholes_merton.implied_volatility,
        "implied_volatility",
        lambda a, x: fast_implied_volatility(
            a["price"],
            a["S"],
            a["K"],
            a["t"],
            a["r"],
            a["flag"],
            a["q"],
            model="black_scholes_merton",
            **numpy_out,
            **x,
        ),
    )

    # --- Greeks (py_vollib.<model>.greeks.numerical) ---
    for package, model, underlying in (
        (py_vollib.black, "black", "F"),
        (py_vollib.black_scholes, "black_scholes", "S"),
        (py_vollib.black_scholes_merton, "black_scholes_merton", "S"),
    ):
        numerical = package.greeks.numerical
        for name, function in (
            ("delta", delta),
            ("gamma", gamma),
            ("rho", rho),
            ("theta", theta),
            ("vega", vega),
        ):
            if model == "black" and name == "rho":
                continue  # below
            patch(numerical, name, _greek(function, model, underlying))

    # py_vollib's Black-76 rho is the derivative at a fixed forward, -t * price per 1%.
    # fast_vollib's rho(model="black") follows a different convention, so the adapter computes
    # py_vollib's from the Black-76 price.
    patch(
        py_vollib.black.greeks.numerical,
        "rho",
        lambda a, x: (
            -0.01
            * np.asarray(a["t"], dtype=np.float64)
            * fast_black(a["flag"], a["F"], a["K"], a["t"], a["r"], a["sigma"], **numpy_out, **x)
        ),
    )
