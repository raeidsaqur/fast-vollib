"""No-arbitrage condition kernels (design §4).

The four condition families are computed as **normalized, dimensionless slack
fields** on the ``(Nk, Nt)`` mesh:

* ``butterfly`` — price-space convexity (Breeden–Litzenberger density ≥ 0),
  the robust primary check; Durrleman's ``g`` is carried alongside as the
  secondary IV-space view.
* ``calendar``  — total variance non-decreasing in maturity (``∂_T w ≥ 0``).
* ``vertical``  — call slope bound ``−1 ≤ ∂c̃/∂K ≤ 0`` (subsumes monotonicity).
* ``bound``     — discounted-call price box.

Each field is ``≥ 0`` where the condition holds; the *violation magnitude* is
``relu(−slack)`` normalized to be comparable across meshes and underlyings.
A single :func:`compute_fields` produces every array namespace-generically, so
the numpy report (:func:`~fast_vollib.surface.metrics.build_report`) and the
differentiable penalty (:func:`~fast_vollib.surface.penalty.arbitrage_penalty`)
consume *identical* math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .density import bl_density, durrleman_g
from .transforms import discounted_call, iv_to_total_variance, undiscounted_call

if TYPE_CHECKING:
    from ._xp import ArrayNS

_FLOOR = 1e-300


@dataclass
class SurfaceFields:
    """All per-node arbitrage diagnostic arrays for one surface.

    Shapes are stated relative to the ``(Nk, Nt)`` mesh.  ``*_mag`` arrays are
    normalized violation magnitudes (0 where the condition holds).  Coordinate
    arrays (``*_k``, ``*_T``) accompany each field for localization.
    """

    # geometry (Nk, Nt)
    k2d: Any
    T2d: Any
    w: Any
    forward2d: Any
    # butterfly — price-space density (interior, Nk-2, Nt)
    density: Any
    density_strikes: Any
    bfly_mag: Any
    g: Any  # Durrleman g (Nk-2, Nt), secondary view
    # calendar — adjacent maturities (Nk, Nt-1)
    cal_depth: Any
    cal_rel: Any
    # vertical — adjacent strikes (Nk-1, Nt)
    vert_mag: Any
    slope: Any
    # bounds — every node (Nk, Nt)
    bound_mag: Any


def compute_fields(
    k2d, w, forward2d, discount2d, xp: "ArrayNS", *, shared_k: bool = True, valid_mask=None
) -> SurfaceFields:
    """Compute all normalized slack / violation fields for a surface.

    Parameters
    ----------
    k2d, w, forward2d, discount2d:
        Log-moneyness, total variance, forward curve and discount factor
        ``e^{−rT}``, each broadcast to shape ``(Nk, Nt)``.
    shared_k:
        ``True`` when rows are a shared forward-log-moneyness axis (calendar
        checked as ``∂_T w ≥ 0``); ``False`` for a fixed-strike grid (calendar
        checked as undiscounted-call monotonicity at fixed strike — the
        coordinate-correct form when the forward is term-varying).
    valid_mask:
        Reserved for future native-only field computation; the caller handles
        NaN-quote masking.  Fields propagate NaN naturally.
    """
    c_tilde = undiscounted_call(k2d, w, forward2d, xp)

    # -- butterfly: price-space convexity (BL density ≥ 0) -------------------
    density, density_strikes = bl_density(k2d, w, forward2d, xp)
    abs_mass = xp.sum(xp.abs(density), axis=0)  # (Nt,) per-slice scale
    scale = xp.maximum(abs_mass, xp.asarray(_FLOOR, like=abs_mass))
    bfly_mag = xp.relu(-density) / scale
    g = durrleman_g(k2d, w, xp)

    # -- calendar -----------------------------------------------------------
    if shared_k:
        # ∂_T w ≥ 0 at fixed forward-log-moneyness.
        raw_cal = w[:, :-1] - w[:, 1:]  # > 0 ⟺ crossing
        cal_ref = xp.maximum(w[:, :-1], xp.asarray(_FLOOR, like=w))
    else:
        # Undiscounted call non-decreasing in T at fixed strike.
        raw_cal = c_tilde[:, :-1] - c_tilde[:, 1:]
        cal_ref = xp.maximum(c_tilde[:, :-1], xp.asarray(_FLOOR, like=c_tilde))
    cal_depth = xp.relu(raw_cal)
    cal_rel = cal_depth / cal_ref

    # -- vertical: slope bound −1 ≤ ∂c̃/∂K ≤ 0 -------------------------------
    strikes = forward2d * xp.exp(k2d)
    dK = strikes[1:] - strikes[:-1]
    slope = (c_tilde[1:] - c_tilde[:-1]) / xp.maximum(dK, xp.asarray(_FLOOR, like=dK))
    # ∂c̃/∂K is already dimensionless and bounded in [−1, 0] for a no-arb surface.
    upper = xp.relu(slope)  # slope > 0  (calls increasing — also breaks monotonicity)
    lower = xp.relu(-1.0 - slope)  # slope < −1
    vert_mag = upper + lower

    # -- bounds: max(0,(F−K)e^{−rT}) ≤ C ≤ F·e^{−rT} -----------------------
    C = discounted_call(k2d, w, forward2d, discount2d, xp)
    cap = forward2d * discount2d
    intrinsic = cap * xp.relu(1.0 - xp.exp(k2d))
    over = xp.relu(C - cap)
    under = xp.relu(intrinsic - C)
    bound_mag = (over + under) / xp.maximum(cap, xp.asarray(_FLOOR, like=cap))

    del valid_mask
    return SurfaceFields(
        k2d=k2d,
        T2d=None,
        w=w,
        forward2d=forward2d,
        density=density,
        density_strikes=density_strikes,
        bfly_mag=bfly_mag,
        g=g,
        cal_depth=cal_depth,
        cal_rel=cal_rel,
        vert_mag=vert_mag,
        slope=slope,
        bound_mag=bound_mag,
    )


def total_variance_field(iv, T2d, xp: "ArrayNS"):
    """Convenience wrapper: total variance ``w = σ²T`` for a surface."""
    return iv_to_total_variance(iv, T2d, xp)
