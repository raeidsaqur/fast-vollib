"""Separately typed European illustrations of American-listed positions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .bands import SurfaceSensitivityBand
from .errors import PnLError
from .results import PnLProfile


@dataclass(frozen=True, slots=True, kw_only=True)
class EuropeanProxyProfile(PnLProfile):
    def __post_init__(self) -> None:
        from .model import PROXY_CAVEAT

        PnLProfile.__post_init__(self)
        if (
            self.meta.basis != "european_proxy"
            or self.meta.caveat != PROXY_CAVEAT
            or not any(c.exercise == "american" for c in self.contracts)
        ):
            raise PnLError("European proxy must retain American source terms and its fixed caveat")


@dataclass(frozen=True, slots=True, kw_only=True)
class EuropeanProxySensitivityBand(SurfaceSensitivityBand):
    """Sampled European surrogate, without American risk/probability capabilities."""


def european_proxy_profile(snapshot: Any, spot: Any, **kwargs: Any) -> EuropeanProxyProfile:
    from .model import _evaluate

    if "proxy" in kwargs:
        raise PnLError("proxy policy is fixed by the separately typed route")
    result = _evaluate(snapshot, spot, proxy=True, **kwargs)
    assert isinstance(result, EuropeanProxyProfile)
    return result


def european_proxy_samples(snapshot: Any, spot: Any, **kwargs: Any) -> EuropeanProxySensitivityBand:
    from .surface_samples import _sampled

    if "proxy" in kwargs:
        raise PnLError("proxy policy is fixed by the separately typed route")
    result = _sampled(snapshot, spot, proxy=True, **kwargs)
    assert isinstance(result, EuropeanProxySensitivityBand)
    return result
