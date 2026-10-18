"""Optional P&L figures with persistent valuation-basis captions."""

from __future__ import annotations

from typing import Any


def _plt() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            'fast_vollib.pnl requires matplotlib — install the viz extra: `pip install "fast-vollib[viz]"`.'
        ) from exc
    return plt


def plot_pnl(
    profiles: Any,
    *,
    mark: Any = None,
    ax: Any = None,
    show_legs: bool = False,
    show_breakevens: bool = True,
) -> Any:
    """Plot explicit expiry/model curves and quote indications; never calls show."""
    from .errors import PnLError
    from .results import PnLProfile

    profiles = tuple(profiles)
    all_profiles = profiles + (() if mark is None else (mark,))
    if not all_profiles or any(not isinstance(p, PnLProfile) for p in all_profiles):
        raise PnLError("plot_pnl requires typed profiles")
    if len({p.meta.currency for p in all_profiles}) != 1:
        raise PnLError("one P&L axis requires a common currency")
    if mark is not None and mark.meta.basis != "quote_mark":
        raise PnLError("mark overlay requires an observed quote mark")
    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    caveats = []
    for j, profile in enumerate(all_profiles):
        if profile.meta.basis == "quote_mark":
            if profile.valid[0, 0]:
                ax.scatter(
                    profile.spot,
                    profile.total[0],
                    marker="D",
                    color="black",
                    label="Quote indication",
                )
                ax.axvline(profile.spot[0], color="0.4", linestyle="--", linewidth=0.8)
            else:
                ax.annotate("Quote total unavailable", xy=(0.02, 0.88), xycoords="axes fraction")
        else:
            for h, horizon in enumerate(profile.horizons):
                label = (
                    (f"Expiry {profile.target_date}" if profile.target_date else "Closed trade")
                    if profile.meta.basis == "expiry_intrinsic"
                    else f"{'European proxy' if profile.meta.basis == 'european_proxy' else 'European model'} at +{horizon:g} years"
                )
                style = (
                    "-."
                    if profile.meta.basis == "european_proxy"
                    else "--"
                    if profile.meta.basis == "model"
                    else ("-", "--", "-.", ":")[j % 4]
                )
                ax.plot(
                    profile.spot,
                    profile.total[h],
                    label=f"{label} ({profile.meta.origin})",
                    linestyle=style,
                )
                if show_legs:
                    for i, name in enumerate(profile.leg_ids):
                        ax.plot(
                            profile.spot,
                            profile.leg_values[h, :, i],
                            linestyle=":",
                            alpha=0.6,
                            label=f"{name} at {horizon}",
                        )
            for strike in set(profile.strikes):
                ax.axvline(float(strike), color="0.6", linestyle=":", linewidth=0.6)
            if show_breakevens and profile.breakevens is not None:
                for root in profile.breakevens:
                    ax.plot(float(root), 0, marker="x", color="black")
            if profile.metrics is not None:
                ax.annotate(
                    f"Minimum P&L: {profile.minimum_pnl}; maximum P&L: {profile.maximum_pnl}",
                    xy=(0.02, 0.98 - 0.08 * j),
                    xycoords="axes fraction",
                    va="top",
                )
        if profile.meta.caveat and profile.meta.caveat not in caveats:
            caveats.append(profile.meta.caveat)
    ax.axhline(0, color="0.3", linewidth=0.8)
    ax.set(
        xlabel="Spot at target",
        ylabel=f"Nominal P&L ({all_profiles[0].meta.currency})",
        title="Position P&L by declared valuation basis",
    )
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    if caveats:
        import textwrap

        fig.text(
            0.02, 0.01, "\n".join(textwrap.fill(x, 100) for x in caveats), fontsize=8, va="bottom"
        )
        fig.subplots_adjust(bottom=min(0.5, 0.18 + 0.05 * len(caveats)))
    return fig


def plot_pnl_band(
    band: Any, *, ax: Any = None, horizon_index: int = 0, quantiles: Any = (0.05, 0.5, 0.95)
) -> Any:
    """Surface-only sensitivity quantiles at fixed spots, without probability claims."""
    from .bands import SurfaceSensitivityBand
    from .errors import PnLError

    if (
        not isinstance(band, SurfaceSensitivityBand)
        or type(horizon_index) is not int
        or not 0 <= horizon_index < len(band.horizons)
    ):
        raise PnLError("typed surface band and a valid horizon index required")
    if len(quantiles) != 3:
        raise PnLError("lower, center and upper surface quantiles required")
    values = band.quantiles(quantiles)[:, horizon_index, :]
    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    label = "European proxy" if band.meta.basis == "european_proxy" else "European model"
    ax.fill_between(
        band.spot, values[0], values[2], alpha=0.2, label="Surface-sample quantile range"
    )
    ax.plot(
        band.spot,
        values[1],
        linestyle="-." if band.meta.basis == "european_proxy" else "--",
        label=label,
    )
    ax.axhline(0, color="0.3", linewidth=0.8)
    ax.set(
        xlabel="Fixed spot",
        ylabel=f"Nominal P&L ({band.meta.currency})",
        title=f"Surface sensitivity at +{band.horizons[horizon_index]:g} years",
    )
    ax.legend()
    import textwrap

    assert band.meta.caveat is not None
    fig.text(0.02, 0.01, textwrap.fill(band.meta.caveat, 100), fontsize=8, va="bottom")
    fig.subplots_adjust(bottom=0.28)
    return fig
