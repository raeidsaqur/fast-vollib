"""The six diagnostic figures of design §8.

All functions take an :class:`~fast_vollib.surface.grid.IVSurface` (and, where
useful, a precomputed :class:`~fast_vollib.surface.report.ArbitrageReport`),
return a :class:`matplotlib.figure.Figure`, and never call ``plt.show`` — the
caller owns display / saving.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._array_api import numpy_namespace
from ..surface.density import bl_density, durrleman_g

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ..surface.grid import IVSurface
    from ..surface.report import ArbitrageReport
    from .quotes import SurfaceQuotes

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
    """Per-node maximum normalized violation magnitude over the ``(k, T)`` mesh.

    Magnitudes are recomputed from the arbitrage fields (gated at the report's
    tolerance), so the map is complete even when ``report.violations`` was
    truncated by ``max_violations``.
    """
    plt = _plt()
    from ..surface.arbitrage import compute_fields
    from ..surface.metrics import DEFAULT_TOLERANCE

    tolerance = report.tolerance if report is not None else DEFAULT_TOLERANCE
    k2d, T2d, w, fwd2d, disc2d, xp = _geometry(surf)
    fields = compute_fields(k2d, w, fwd2d, disc2d, xp, shared_k=surf.shared_k)
    heat = np.zeros((surf.Nk, surf.Nt))

    def _scatter(mag, i0: int, j0: int) -> None:
        m = np.nan_to_num(xp.to_numpy(mag), nan=0.0)
        m = np.where(m > tolerance, m, 0.0)
        view = heat[i0 : i0 + m.shape[0], j0 : j0 + m.shape[1]]
        np.maximum(view, m, out=view)

    # Anchor nodes match metrics._collect: butterfly interior rows shift by 1;
    # calendar/vertical pairs anchor on the earlier/lower node.
    _scatter(fields.bfly_mag, 1, 0)
    _scatter(fields.cal_rel, 0, 0)
    _scatter(fields.vert_mag, 0, 0)
    _scatter(fields.bound_mag, 0, 0)
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


def plot_smile_fit(
    truth: "SurfaceQuotes",
    pred_iv: Any,
    *,
    surface_id: Any = None,
    maturity: float | None = None,
    maturity_decimals: int = 6,
    ax=None,
) -> "Figure":
    """Observed quotes against a model's prediction on one smile.

    The scattered-quote counterpart of the grid figures above: it plots the
    truth as markers and the prediction as a line over the same strikes, so a
    reader can see *where* on the smile a fit error sits rather than only how
    large it is.

    Parameters
    ----------
    truth:
        The observed quotes.  Rows with a missing observation are dropped.
    pred_iv:
        Predicted implied volatilities aligned row-for-row with ``truth``.
        Values the evaluator would classify as invalid (non-finite or
        negative) are omitted from the prediction line rather than drawn as a
        break in it.
    surface_id, maturity:
        Which smile to draw.  Both default to the first one present, and the
        chosen pair is named in the title.
    maturity_decimals:
        Decimal places maturities are rounded to when bucketing into smiles.
    ax:
        Optional existing axes to draw on.
    """
    from .quotes import SurfaceQuotes  # noqa: F401  (documented parameter type)

    plt = _plt()
    pred = np.asarray(pred_iv, dtype=np.float64)
    if pred.shape != (truth.n,):
        raise ValueError(f"pred_iv must have shape ({truth.n},); got {pred.shape}.")

    labels = truth.surface_ids()
    if not labels:
        raise ValueError("Cannot plot a smile from an empty quote set.")
    label = labels[0] if surface_id is None else surface_id
    if label not in labels:
        raise ValueError(f"No surface {label!r} in these quotes; have {labels[:5]}.")

    in_surface = np.asarray(truth.surface_id) == label
    observed = ~np.isnan(np.asarray(truth.iv, dtype=np.float64))
    buckets = np.round(np.asarray(truth.T, dtype=np.float64), maturity_decimals)
    available = np.unique(buckets[in_surface & observed])
    if available.size == 0:
        raise ValueError(f"Surface {label!r} carries no observed quotes to plot.")
    chosen = float(available[0]) if maturity is None else round(float(maturity), maturity_decimals)
    if not np.any(np.isclose(available, chosen, rtol=0.0, atol=0.0)):
        raise ValueError(
            f"No maturity {chosen!r} on surface {label!r}; have {available.tolist()[:5]}."
        )

    rows = in_surface & observed & (buckets == chosen)
    strikes = np.asarray(truth.k, dtype=np.float64)[rows]
    order = np.argsort(strikes, kind="stable")
    strikes = strikes[order]
    observed_iv = np.asarray(truth.iv, dtype=np.float64)[rows][order]
    predicted_iv = pred[rows][order]

    fig, ax = _fig_ax(plt, ax)
    ax.plot(strikes, observed_iv, "o", label="observed", color="#1f77b4")
    usable = np.isfinite(predicted_iv) & (predicted_iv >= 0.0)
    if usable.any():
        ax.plot(strikes[usable], predicted_iv[usable], "-", label="predicted", color="#d62728")
    if not usable.all():
        ax.plot(
            strikes[~usable],
            observed_iv[~usable],
            "x",
            label="unpredicted",
            color="#7f7f7f",
        )
    ax.set_xlabel("log-moneyness $k$")
    ax.set_ylabel("implied volatility")
    ax.set_title(f"Smile fit — surface {label}, T = {chosen:g}")
    ax.legend(loc="best", frameon=False)
    return fig


def _fig_ax(plt, ax) -> tuple[Any, Any]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    else:
        fig = ax.get_figure()
    return fig, ax
