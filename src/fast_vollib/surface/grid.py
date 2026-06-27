"""Backend-agnostic IV-surface container and constructors (design §3).

:class:`IVSurface` is parametrized internally in **forward log-moneyness**
``k = log(K / F)`` × **maturity** ``T`` (year fractions) — the natural
coordinates for the no-arbitrage conditions — while accepting strike / DTE
inputs via constructors.

Two grid topologies are supported and tracked by :attr:`IVSurface.shared_k`:

* **Shared-moneyness** (``k`` is 1-D, one forward-log-moneyness axis common to
  every maturity).  This is the natural output of most surface generators.
  Calendar arbitrage is checked as total-variance monotonicity ``∂_T w ≥ 0``.
* **Fixed-strike** (``k`` is 2-D, derived from a shared strike vector under a
  term-varying forward).  Calendar arbitrage is checked as undiscounted-call
  monotonicity in ``T`` at fixed strike — the coordinate-correct form here.

Inputs may be numpy / torch / jax arrays; the container preserves dtype and
device and never silently moves tensors off-device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ._xp import ArrayNS, get_namespace

if TYPE_CHECKING:
    from .._typing import ArrayLike  # noqa: F401


def _as_array(x: Any) -> Any:
    """Pass native arrays through untouched; coerce python scalars/sequences."""
    mod = type(x).__module__
    if mod.startswith(("torch", "jax")) or isinstance(x, np.ndarray):
        return x
    return np.asarray(x, dtype=np.float64)


@dataclass
class IVSurface:
    """An implied-volatility surface on a ``(Nk, Nt)`` mesh.

    Construct via :meth:`from_logmoneyness`, :meth:`from_strikes`,
    :meth:`from_total_variance`, or directly.  See module docstring for the
    coordinate convention.
    """

    k: Any  # log-moneyness, shape (Nk,) [shared] or (Nk, Nt) [fixed-strike]
    T: Any  # maturities (year fractions), shape (Nt,)
    iv: Any  # implied vols, shape (Nk, Nt); NaN allowed = no quote
    forward: Any = 1.0  # forward curve F(T), shape (Nt,) or scalar
    r: Any = 0.0  # discount rate(s), shape (Nt,) or scalar
    q: Any = 0.0  # carry / dividend yield
    native_mask: Any = None  # True where value came from a generator node
    t_index: int | None = None  # optional calendar-time step (animation axis)
    shared_k: bool = True  # whether k is a single shared moneyness axis

    _ns: ArrayNS = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self.k = _as_array(self.k)
        self.T = _as_array(self.T)
        self.iv = _as_array(self.iv)
        self.forward = _as_array(self.forward)
        self.r = _as_array(self.r)
        self.q = _as_array(self.q)
        if self.native_mask is not None:
            self.native_mask = np.asarray(self.native_mask, dtype=bool)
        self.shared_k = getattr(self.k, "ndim", 1) == 1
        if self.iv.ndim != 2:
            raise ValueError(f"iv must be 2-D (Nk, Nt); got shape {self.iv.shape}.")
        self._ns = get_namespace(self.iv)

    # -- geometry ------------------------------------------------------------
    @property
    def Nk(self) -> int:
        return int(self.iv.shape[0])

    @property
    def Nt(self) -> int:
        return int(self.iv.shape[1])

    def namespace(self) -> ArrayNS:
        """The :class:`ArrayNS` matching this surface's array backend."""
        return self._ns

    def broadcast(self, xp: ArrayNS | None = None):
        """Return geometry broadcast to ``(Nk, Nt)`` in the array namespace.

        Returns
        -------
        (k2d, T2d, w, forward2d, discount2d):
            Log-moneyness, maturity, total variance ``w = σ²T``, forward, and
            discount factor ``e^{−rT}``, each shape ``(Nk, Nt)``.
        """
        xp = xp or self._ns
        iv = self.iv
        k = xp.asarray(self.k, like=iv)
        T = xp.asarray(self.T, like=iv)
        fwd = xp.asarray(self.forward, like=iv)
        r = xp.asarray(self.r, like=iv)

        Nk, Nt = self.Nk, self.Nt
        T2d = self._row_to_2d(T, Nk, Nt, xp)
        forward2d = self._row_to_2d(fwd, Nk, Nt, xp)
        r2d = self._row_to_2d(r, Nk, Nt, xp)
        k2d = k if getattr(k, "ndim", 1) == 2 else (k[:, None] + xp.zeros((Nk, Nt), like=k))
        w = iv * iv * T2d
        discount2d = xp.exp(-r2d * T2d)
        return k2d, T2d, w, forward2d, discount2d

    @staticmethod
    def _row_to_2d(v, Nk: int, Nt: int, xp: ArrayNS):
        """Broadcast a scalar or per-maturity (Nt,) vector to (Nk, Nt)."""
        v_nd = getattr(v, "ndim", 0)
        zeros = xp.zeros((Nk, Nt), like=v)
        if v_nd == 0:
            return v + zeros
        if v_nd == 1:
            return v[None, :] + zeros
        return v

    # -- derived views -------------------------------------------------------
    def total_variance(self):
        """Total implied variance ``w = σ²T`` on the mesh."""
        _, _, w, _, _ = self.broadcast()
        return w

    def call_prices(self, *, undiscounted: bool = False):
        """Black call prices on the mesh (discounted unless ``undiscounted``)."""
        from .transforms import discounted_call, undiscounted_call

        xp = self._ns
        k2d, _, w, forward2d, discount2d = self.broadcast(xp)
        if undiscounted:
            return undiscounted_call(k2d, w, forward2d, xp)
        return discounted_call(k2d, w, forward2d, discount2d, xp)

    def density(self):
        """Breeden–Litzenberger risk-neutral density and interior strikes."""
        from .density import bl_density

        xp = self._ns
        k2d, _, w, forward2d, _ = self.broadcast(xp)
        return bl_density(k2d, w, forward2d, xp)

    def validate(self, **kwargs):
        """Run the full arbitrage harness; see :func:`validate_surface`."""
        from .metrics import validate_surface

        return validate_surface(self, **kwargs)

    # -- constructors --------------------------------------------------------
    @classmethod
    def from_logmoneyness(
        cls,
        k: ArrayLike,
        T: ArrayLike,
        iv: ArrayLike,
        *,
        forward: ArrayLike = 1.0,
        r: ArrayLike = 0.0,
        q: ArrayLike = 0.0,
        native_mask: Any = None,
        t_index: int | None = None,
    ) -> IVSurface:
        """Build from a shared forward-log-moneyness axis ``k = log(K/F)``."""
        return cls(
            k=k,
            T=T,
            iv=iv,
            forward=forward,
            r=r,
            q=q,
            native_mask=native_mask,
            t_index=t_index,
        )

    @classmethod
    def from_strikes(
        cls,
        K: ArrayLike,
        T: ArrayLike,
        iv: ArrayLike,
        *,
        spot: ArrayLike,
        r: ArrayLike = 0.0,
        q: ArrayLike = 0.0,
        native_mask: Any = None,
        t_index: int | None = None,
    ) -> IVSurface:
        """Build from a shared strike vector ``K`` and spot.

        The forward curve is ``F(T) = spot · e^{(r−q)T}``.  When the forward is
        flat across maturities the moneyness axis is shared (1-D ``k``);
        otherwise ``k`` becomes 2-D and calendar checks use the fixed-strike
        (undiscounted-call) form.  ``K`` and ``T`` are matched to ``iv`` of
        shape ``(len(K), len(T))``.
        """
        K_a = np.asarray(K, dtype=np.float64)
        T_a = np.asarray(T, dtype=np.float64)
        r_a = np.asarray(r, dtype=np.float64)
        q_a = np.asarray(q, dtype=np.float64)
        spot_a = np.asarray(spot, dtype=np.float64)
        forward = spot_a * np.exp((r_a - q_a) * T_a)  # (Nt,) or scalar
        forward = np.broadcast_to(forward, T_a.shape) if T_a.ndim else forward
        # k_ij = log(K_i / F_j)
        F_row = np.atleast_1d(forward)
        k2d = np.log(K_a[:, None] / F_row[None, :])  # (Nk, Nt)
        flat_forward = bool(np.allclose(k2d, k2d[:, :1]))
        k = k2d[:, 0] if flat_forward else k2d
        return cls(
            k=k,
            T=T_a,
            iv=iv,
            forward=forward,
            r=r_a,
            q=q_a,
            native_mask=native_mask,
            t_index=t_index,
        )

    @classmethod
    def from_call_prices(
        cls,
        K: ArrayLike,
        T: ArrayLike,
        call_prices: ArrayLike,
        *,
        spot: ArrayLike,
        r: ArrayLike = 0.0,
        q: ArrayLike = 0.0,
        discounted: bool = True,
        native_mask: Any = None,
        t_index: int | None = None,
    ) -> IVSurface:
        """Build from a discounted (or undiscounted) call-price grid.

        Prices are inverted to implied vol via the library solver
        (:func:`fast_vollib.fast_implied_volatility`, Black-Scholes-Merton), so
        the resulting surface is analysed in the same IV-centric coordinates as
        every other constructor.  ``call_prices`` has shape ``(len(K), len(T))``.

        Notes
        -----
        Because each node is inverted independently, single-node *price-box* and
        *slope* pathologies in the raw prices are normalized away by the
        inversion; the multi-node butterfly/calendar conditions (the ones
        generators actually break) are preserved.  Pass raw IVs via
        :meth:`from_strikes` / :meth:`from_logmoneyness` when available.
        """
        from ..implied_volatility import fast_implied_volatility

        K_a = np.asarray(K, dtype=np.float64)
        T_a = np.asarray(T, dtype=np.float64)
        r_a = np.asarray(r, dtype=np.float64)
        q_a = np.asarray(q, dtype=np.float64)
        spot_a = np.asarray(spot, dtype=np.float64)
        prices = np.asarray(call_prices, dtype=np.float64)
        Nk, Nt = prices.shape
        K2d = np.broadcast_to(K_a[:, None], (Nk, Nt))
        T2d = np.broadcast_to(np.atleast_1d(T_a)[None, :], (Nk, Nt))
        r2d = np.broadcast_to(np.atleast_1d(r_a), (Nt,))[None, :] * np.ones((Nk, 1))
        q2d = np.broadcast_to(np.atleast_1d(q_a), (Nt,))[None, :] * np.ones((Nk, 1))
        px = prices if discounted else prices * np.exp(-r2d * T2d)
        iv = fast_implied_volatility(
            px.ravel(),
            np.broadcast_to(spot_a, (Nk * Nt,)),
            K2d.ravel(),
            T2d.ravel(),
            r2d.ravel(),
            "c",
            q=q2d.ravel(),
            model="black_scholes_merton",
            on_error="ignore",
            return_as="numpy",
            backend="numpy",
        ).reshape(Nk, Nt)
        return cls.from_strikes(
            K_a,
            T_a,
            iv,
            spot=spot_a,
            r=r_a,
            q=q_a,
            native_mask=native_mask,
            t_index=t_index,
        )

    @classmethod
    def from_total_variance(
        cls,
        k: ArrayLike,
        T: ArrayLike,
        w: ArrayLike,
        *,
        forward: ArrayLike = 1.0,
        r: ArrayLike = 0.0,
        q: ArrayLike = 0.0,
        native_mask: Any = None,
        t_index: int | None = None,
    ) -> IVSurface:
        """Build from a total-variance grid ``w = σ²T`` (σ recovered as √(w/T))."""
        w_a = _as_array(w)
        T_a = _as_array(T)
        ns = get_namespace(w_a)
        iv = ns.sqrt(w_a / (T_a[None, :] if getattr(T_a, "ndim", 1) == 1 else T_a))
        return cls(
            k=k,
            T=T,
            iv=iv,
            forward=forward,
            r=r,
            q=q,
            native_mask=native_mask,
            t_index=t_index,
        )


@dataclass
class SurfaceSequence:
    """An ordered stack of :class:`IVSurface` sharing a mesh (design §3).

    Represents a generative surface evolving over calendar time, indexed by
    ``t_index``.  This is what the UI animates and what calendar-of-surfaces
    evaluation consumes.
    """

    surfaces: list[IVSurface]

    def __post_init__(self):
        if not self.surfaces:
            raise ValueError("SurfaceSequence requires at least one surface.")

    def __len__(self) -> int:
        return len(self.surfaces)

    def __getitem__(self, i: int) -> IVSurface:
        return self.surfaces[i]

    def __iter__(self):
        return iter(self.surfaces)

    def validate(self, **kwargs):
        """Validate every frame; returns a list of reports indexed by time."""
        from .metrics import validate_surface

        return [validate_surface(s, **kwargs) for s in self.surfaces]
