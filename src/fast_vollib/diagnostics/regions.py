"""Regions of the ``(k, T)`` plane a fit error can be split by.

The headline diagnostic of surface-fitting evaluations is how a model does in
the region it was *not* shown: quotes are dense near the money and at short
maturities (the *liquid* box) and sparse in the wings and at long maturities
(*illiquid*), so the error is reported for the whole surface and for each
region separately.

A region is a **named, serializable** predicate on ``(k, T)`` with an optional
name for its complement.  :class:`Box` is the serializable form -- explicit
inclusive/exclusive bounds that round-trip through JSON unchanged -- and
:data:`DEFAULT_REGIONS` is the liquid / illiquid split at ``|k| <= 0.2`` and
``T <= 0.5`` years.  :class:`NamedRegion` wraps an arbitrary callable for
low-level use; it deliberately has no serializable descriptor, so a report can
never claim to describe a predicate it cannot reproduce.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "DEFAULT_REGIONS",
    "LIQUID_BOX",
    "Box",
    "NamedRegion",
    "Region",
    "liquid_mask",
]

#: Bound-inclusion policies accepted by :class:`Box`.
CLOSED_POLICIES: tuple[str, ...] = ("both", "left", "right", "neither")


@runtime_checkable
class Region(Protocol):
    """A named subset of the ``(k, T)`` plane.

    ``name`` labels the metrics computed *inside* the region
    (``rmse_liquid``); ``complement_name`` labels the metrics computed outside
    it (``rmse_illiquid``), or is ``None`` to report the region alone.
    ``describe()`` returns a JSON-serializable descriptor, or ``None`` when the
    predicate cannot be described -- such a region may still be evaluated but
    is refused by the serialized contract.
    """

    @property
    def name(self) -> str: ...

    @property
    def complement_name(self) -> str | None: ...

    def mask(self, k: Any, T: Any) -> np.ndarray: ...

    def describe(self) -> dict[str, Any] | None: ...


def _as_float_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")
    return array


def liquid_mask(k: Any, T: Any, *, k_liq: float = 0.2, T_liq: float = 0.5) -> np.ndarray:
    """``True`` where ``|k| <= k_liq`` and ``T <= T_liq`` (both bounds inclusive)."""
    k_arr = np.asarray(k, dtype=np.float64)
    T_arr = np.asarray(T, dtype=np.float64)
    return (np.abs(k_arr) <= k_liq) & (T_arr <= T_liq)


@dataclass(frozen=True, slots=True)
class Box:
    """An axis-aligned, named, JSON-serializable box in the ``(k, T)`` plane.

    Parameters
    ----------
    name:
        Suffix of the metrics computed inside the box (``rmse_liquid``).
    complement_name:
        Suffix of the metrics computed outside it (``rmse_illiquid``), or
        ``None`` to report the box alone.
    k_min, k_max, T_min, T_max:
        Bounds; ``None`` means unbounded on that side.  Every supplied bound
        must be finite and ``k_min <= k_max`` / ``T_min <= T_max``.
    closed:
        Which bounds are inclusive: ``"both"`` (default), ``"left"`` (lower
        bounds inclusive, upper exclusive), ``"right"``, or ``"neither"``.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.diagnostics import Box
    >>> box = Box(name="liquid", complement_name="illiquid", k_min=-0.2, k_max=0.2, T_max=0.5)
    >>> box.mask(np.array([0.0, 0.5]), np.array([0.25, 0.25])).tolist()
    [True, False]
    """

    name: str
    complement_name: str | None = None
    k_min: float | None = None
    k_max: float | None = None
    T_min: float | None = None
    T_max: float | None = None
    closed: str = "both"

    def __post_init__(self) -> None:
        for label in ("name", "complement_name"):
            value = getattr(self, label)
            if label == "complement_name" and value is None:
                continue
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a non-empty string; got {value!r}.")
        if self.name == self.complement_name:
            raise ValueError("name and complement_name must differ.")
        if self.closed not in CLOSED_POLICIES:
            raise ValueError(f"closed must be one of {CLOSED_POLICIES}; got {self.closed!r}.")
        for label in ("k_min", "k_max", "T_min", "T_max"):
            value = getattr(self, label)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{label} must be a real number or None; got {value!r}.")
            coerced = float(value)
            if not np.isfinite(coerced):
                raise ValueError(f"{label} must be finite; got {value!r}.")
            object.__setattr__(self, label, coerced)
        if self.k_min is not None and self.k_max is not None and self.k_min > self.k_max:
            raise ValueError(f"k_min ({self.k_min}) must not exceed k_max ({self.k_max}).")
        if self.T_min is not None and self.T_max is not None and self.T_min > self.T_max:
            raise ValueError(f"T_min ({self.T_min}) must not exceed T_max ({self.T_max}).")

    @property
    def lower_closed(self) -> bool:
        """Whether ``k_min`` / ``T_min`` are inclusive."""
        return self.closed in ("both", "left")

    @property
    def upper_closed(self) -> bool:
        """Whether ``k_max`` / ``T_max`` are inclusive."""
        return self.closed in ("both", "right")

    def mask(self, k: Any, T: Any) -> np.ndarray:
        """``True`` at the points inside the box."""
        k_arr = _as_float_array(k, "k")
        T_arr = _as_float_array(T, "T")
        if k_arr.size != T_arr.size:
            raise ValueError(
                f"k and T must have the same length; got {k_arr.size} and {T_arr.size}."
            )
        inside = np.ones(k_arr.shape, dtype=bool)
        lower = np.greater_equal if self.lower_closed else np.greater
        upper = np.less_equal if self.upper_closed else np.less
        for array, low, high in ((k_arr, self.k_min, self.k_max), (T_arr, self.T_min, self.T_max)):
            if low is not None:
                inside &= lower(array, low)
            if high is not None:
                inside &= upper(array, high)
        return inside

    def __call__(self, k: Any, T: Any) -> np.ndarray:
        return self.mask(k, T)

    def describe(self) -> dict[str, Any]:
        """The JSON-serializable descriptor stored in a diagnostics report."""
        return {
            "kind": "box",
            "name": self.name,
            "complement_name": self.complement_name,
            "k_min": self.k_min,
            "k_max": self.k_max,
            "T_min": self.T_min,
            "T_max": self.T_max,
            "closed": self.closed,
        }

    @classmethod
    def from_describe(cls, descriptor: dict[str, Any]) -> Box:
        """Rebuild a box from :meth:`describe` output (strict about extra keys)."""
        expected = {
            "kind",
            "name",
            "complement_name",
            "k_min",
            "k_max",
            "T_min",
            "T_max",
            "closed",
        }
        keys = set(descriptor)
        if keys != expected:
            raise ValueError(f"Box descriptor keys {sorted(keys)} do not match {sorted(expected)}.")
        if descriptor["kind"] != "box":
            raise ValueError(f"Unsupported region kind {descriptor['kind']!r}; expected 'box'.")
        return cls(
            name=descriptor["name"],
            complement_name=descriptor["complement_name"],
            k_min=descriptor["k_min"],
            k_max=descriptor["k_max"],
            T_min=descriptor["T_min"],
            T_max=descriptor["T_max"],
            closed=descriptor["closed"],
        )


@dataclass(frozen=True, slots=True)
class NamedRegion:
    """A named region backed by an arbitrary callable.

    Accepted by the low-level :func:`~fast_vollib.diagnostics.fit_error_by_region`
    so callers can score a bespoke predicate, but :meth:`describe` returns
    ``None``: a serialized report never claims to describe a predicate it
    cannot reproduce, so a report carrying one is refused by the codec.
    """

    name: str
    complement_name: str | None
    predicate: Callable[[Any, Any], Any]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"name must be a non-empty string; got {self.name!r}.")
        if self.complement_name is not None and (
            not isinstance(self.complement_name, str) or not self.complement_name
        ):
            raise ValueError(
                f"complement_name must be a non-empty string or None; got {self.complement_name!r}."
            )
        if self.name == self.complement_name:
            raise ValueError("name and complement_name must differ.")
        if not callable(self.predicate):
            raise TypeError("predicate must be callable.")

    def mask(self, k: Any, T: Any) -> np.ndarray:
        k_arr = _as_float_array(k, "k")
        T_arr = _as_float_array(T, "T")
        if k_arr.size != T_arr.size:
            raise ValueError(
                f"k and T must have the same length; got {k_arr.size} and {T_arr.size}."
            )
        result = np.asarray(self.predicate(k_arr, T_arr), dtype=bool)
        if result.shape != k_arr.shape:
            raise ValueError(f"predicate must return shape {k_arr.shape}; got {result.shape}.")
        return result

    def __call__(self, k: Any, T: Any) -> np.ndarray:
        return self.mask(k, T)

    def describe(self) -> None:
        """``None`` -- a callable predicate has no serializable descriptor."""
        return None


#: Default liquid-box bounds, kept as a mapping for callers that pass
#: ``**LIQUID_BOX`` into :func:`liquid_mask`.
LIQUID_BOX: dict[str, float] = {"k_liq": 0.2, "T_liq": 0.5}

#: The liquid / illiquid split used by default: inclusive ``|k| <= 0.2``, ``T <= 0.5``.
DEFAULT_REGIONS: tuple[Box, ...] = (
    Box(
        name="liquid",
        complement_name="illiquid",
        k_min=-LIQUID_BOX["k_liq"],
        k_max=LIQUID_BOX["k_liq"],
        T_max=LIQUID_BOX["T_liq"],
        closed="both",
    ),
)
