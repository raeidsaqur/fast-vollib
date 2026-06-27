"""The six diagnostic figures of design §8.

All functions take an :class:`~fast_vollib.surface.grid.IVSurface` (and, where
useful, a precomputed :class:`~fast_vollib.surface.report.ArbitrageReport`),
return a :class:`matplotlib.figure.Figure`, and never call ``plt.show`` — the
caller owns display / saving.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..surface._xp import numpy_namespace
from ..surface.density import bl_density, durrleman_g

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ..surface.grid import IVSurface
    from ..surface.report import ArbitrageReport

_MISSING_MPL = (
    "fast_vollib.diagnostics requires matplotlib — install the viz extra: "
    '`pip install "fast-vollib[viz]"`.'
)


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without mpl
        raise ImportError(_MISSING_MPL) from exc
    return plt


def _geometry(surf: "IVSurface"):
    xp = numpy_namespace()
    k2d, T2d, w, fwd2d, disc2d = surf.broadcast(xp)
    to_np = xp.to_numpy
    return (to_np(k2d), to_np(T2d), to_np(w), to_np(fwd2d), to_np(disc2d), xp)


def plot_total_variance_slices(surf: "IVSurface", *, ax=None) -> "Figure":
    """Total variance ``w(k, ·)`` over maturity — calendar crossings are visible
    directly as slices that touch or cross."""
    plt = _plt()
    k2d, T2d, w, *_ = _geometry(surf)
    fig, ax = _fig_ax(plt, ax)
    Tvals = T2d[0]
    cmap = plt.get_cmap("viridis")
    for j in range(w.shape[1]):
        ax.plot(
            k2d[:, j],
            w[:, j],
            color=cmap(j / max(w.shape[1] - 1, 1)),
            label=f"T={Tvals[j]:.3g}",
            lw=1.6,
        )
    ax.set_xlabel("log-moneyness $k$")
    ax.set_ylabel("total variance $w = \\sigma^2 T$")
    ax.set_title("Total-variance slices")
    ax.legend(fontsize=8, ncol=2)
    return fig


def plot_durrleman_g(surf: "IVSurface", *, ax=None) -> "Figure":
    """Durrleman's ``g(k)`` per slice, with ``g < 0`` (butterfly-arbitrage)
    regions shaded red."""
    plt = _plt()
    k2d, T2d, w, _, _, xp = _geometry(surf)
    g = xp.to_numpy(durrleman_g(k2d, w, xp))
    ki = k2d[1:-1]
    fig, ax = _fig_ax(plt, ax)
    Tvals = T2d[0]
    cmap = plt.get_cmap("viridis")
    for j in range(g.shape[1]):
        ax.plot(
            ki[:, j],
            g[:, j],
            color=cmap(j / max(g.shape[1] - 1, 1)),
            label=f"T={Tvals[j]:.3g}",
            lw=1.4,
        )
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    gmin = np.nanmin(g)
    if gmin < 0:
        ax.fill_between(
            ki[:, 0],
            gmin,
            0.0,
            where=np.nanmin(g, axis=1) < 0,
            color="red",
            alpha=0.15,
            label="$g<0$",
        )
    ax.set_xlabel("log-moneyness $k$")
    ax.set_ylabel("Durrleman $g(k)$")
    ax.set_title("Durrleman butterfly function")
    ax.legend(fontsize=8, ncol=2)
    return fig


def plot_density(surf: "IVSurface", *, t_index: int = 0, ax=None) -> "Figure":
    """Breeden–Litzenberger risk-neutral density ``f(K)`` for one slice, with the
    negative-mass region shaded and the ``ndm`` metric annotated."""
    plt = _plt()
    k2d, T2d, w, fwd2d, _, xp = _geometry(surf)
    f, K = bl_density(k2d, w, fwd2d, xp)
    f = xp.to_numpy(f)[:, t_index]
    K = xp.to_numpy(K)[:, t_index]
    order = np.argsort(K)
    f, K = f[order], K[order]
    fig, ax = _fig_ax(plt, ax)
    ax.plot(K, f, color="C0", lw=1.6)
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.fill_between(K, f, 0.0, where=f < 0, color="red", alpha=0.3, label="negative mass")
    trapz = getattr(np, "trapezoid", None) or np.trapz
    tot = trapz(np.abs(f), K)
    ndm = trapz(np.maximum(-f, 0.0), K) / tot if tot > 0 else 0.0
    ax.set_xlabel("strike $K$")
    ax.set_ylabel("RND $f(K)$")
    ax.set_title(f"Risk-neutral density (T={T2d[0, t_index]:.3g}, ndm={ndm:.3g})")
    if (f < 0).any():
        ax.legend(fontsize=8)
    return fig


def plot_violation_heatmap(
    surf: "IVSurface", report: "ArbitrageReport | None" = None, *, ax=None
) -> "Figure":
    """Per-node maximum normalized violation magnitude over the ``(k, T)`` mesh."""
    plt = _plt()
    from ..surface.metrics import validate_surface

    report = report or validate_surface(surf, compute_trust=False)
    k2d, T2d, *_ = _geometry(surf)
    heat = np.zeros((surf.Nk, surf.Nt))
    for v in report.violations:
        if v.index is not None:
            i, j = v.index
            heat[i, j] = max(heat[i, j], v.value)
    fig, ax = _fig_ax(plt, ax)
    mesh = ax.pcolormesh(T2d, k2d, heat, shading="nearest", cmap="inferno")
    fig.colorbar(mesh, ax=ax, label="max normalized violation")
    ax.set_xlabel("maturity $T$")
    ax.set_ylabel("log-moneyness $k$")
    ax.set_title("Violation heatmap")
    return fig


def plot_calendar_map(surf: "IVSurface", *, ax=None) -> "Figure":
    """Calendar crossing depth over ``(k, T_j → T_{j+1})`` adjacent-maturity pairs."""
    plt = _plt()
    from ..surface.arbitrage import compute_fields

    k2d, T2d, w, fwd2d, disc2d, xp = _geometry(surf)
    fields = compute_fields(k2d, w, fwd2d, disc2d, xp, shared_k=surf.shared_k)
    depth = xp.to_numpy(fields.cal_rel)
    fig, ax = _fig_ax(plt, ax)
    Tmid = 0.5 * (T2d[:, :-1] + T2d[:, 1:])
    mesh = ax.pcolormesh(Tmid, k2d[:, :-1], depth, shading="nearest", cmap="magma")
    fig.colorbar(mesh, ax=ax, label="relative crossing depth")
    ax.set_xlabel("maturity midpoint")
    ax.set_ylabel("log-moneyness $k$")
    ax.set_title("Calendar crossing map")
    return fig


def plot_trust_map(
    surf: "IVSurface", report: "ArbitrageReport | None" = None, *, ax=None
) -> "Figure":
    """Round-trip LBR fixed-point trust mask — where ``σ→C→σ'`` is machine-tight."""
    plt = _plt()
    from ..surface.metrics import validate_surface

    report = report or validate_surface(surf)
    if report.trust_mask is None:
        report = validate_surface(surf, compute_trust=True)
    k2d, T2d, *_ = _geometry(surf)
    fig, ax = _fig_ax(plt, ax)
    mesh = ax.pcolormesh(
        T2d,
        k2d,
        np.asarray(report.trust_mask, dtype=float),
        shading="nearest",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
    )
    fig.colorbar(mesh, ax=ax, label="trusted (1) / low-confidence (0)")
    ax.set_xlabel("maturity $T$")
    ax.set_ylabel("log-moneyness $k$")
    ax.set_title("Round-trip trust map")
    return fig


def _fig_ax(plt, ax) -> tuple[Any, Any]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    else:
        fig = ax.get_figure()
    return fig, ax
