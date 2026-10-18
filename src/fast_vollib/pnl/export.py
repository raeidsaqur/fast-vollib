"""Explicit file export of report data, metadata and an optional figure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import PnLError
from .results import PnLProfile


def export_pnl(profile: PnLProfile, path: Any, *, figure: Any = None) -> tuple[Path, ...]:
    """Write strict JSON and CSV beside a caller-selected filename prefix.

    Pass the returned figure from plot_pnl to additionally write a PNG. CSV rows
    retain valuation basis and caveats; JSON retains full metadata and status.
    """
    if not isinstance(profile, PnLProfile):
        raise PnLError("export requires a typed P&L profile")
    prefix = Path(path)
    json_path = Path(str(prefix) + ".json")
    csv_path = Path(str(prefix) + ".csv")
    json_path.write_text(profile.to_json(), encoding="utf-8")
    profile.to_frame().to_csv(csv_path, index=False)
    if figure is None:
        return json_path, csv_path
    png_path = Path(str(prefix) + ".png")
    figure.savefig(png_path, bbox_inches="tight")
    return json_path, csv_path, png_path


def export_surface_band(band: Any, path: Any, *, figure: Any = None) -> tuple[Path, ...]:
    """Export sensitivity data retaining proxy type and non-probability caveats."""
    from .bands import SurfaceSensitivityBand

    if not isinstance(band, SurfaceSensitivityBand):
        raise PnLError("surface-band export requires a typed sensitivity result")
    prefix = Path(path)
    json_path = Path(str(prefix) + ".json")
    csv_path = Path(str(prefix) + ".csv")
    json_path.write_text(band.to_json(), encoding="utf-8")
    band.to_frame().to_csv(csv_path, index=False)
    if figure is None:
        return json_path, csv_path
    png_path = Path(str(prefix) + ".png")
    figure.savefig(png_path, bbox_inches="tight")
    return json_path, csv_path, png_path
