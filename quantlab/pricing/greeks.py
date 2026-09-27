# quantlab: Eugene (Yoogeun) Song. MIT licensed; see LICENSE.
"""Engine-agnostic Greeks by bump-and-reprice.

Any engine that can price can be differentiated numerically. The analytic
engine has closed-form Greeks and the PDE engine reads delta and gamma off
its surface for free; this module is for everything else, and for checking
those two against an independent route.

Step sizes are chosen by derivative order, which matters more than it looks.
A central first difference has truncation error O(h^2) and rounding error
O(eps/h); the optimum is h ~ eps^(1/3), about 1e-5 relative. A central
second difference has rounding error O(eps/h^2) and needs h ~ eps^(1/4),
about 1e-4 relative. Use a first-derivative step for gamma and the
cancellation error swamps the answer -- a 1% discrepancy that looks exactly
like a wrong formula and is nothing of the kind.
"""

from __future__ import annotations

import numpy as np

from .engines.base import Engine
from .instruments import Instrument
from .models import BlackScholes

__all__ = ["bump_greeks"]

_EPS = np.finfo(float).eps


def bump_greeks(engine: Engine, instrument: Instrument, model: BlackScholes,
                which: tuple[str, ...] = ("delta", "gamma", "vega", "theta", "rho")) -> dict[str, float]:
    """Finite-difference Greeks through any engine.

    For stochastic engines, fix the seed on the engine's generator so the
    bumped and unbumped runs share random numbers; otherwise the difference
    is dominated by sampling noise (Glasserman 7.1, "common random numbers").
    """
    out: dict[str, float] = {}
    base = engine.price(instrument, model).value

    if "delta" in which or "gamma" in which:
        h1 = model.spot * _EPS ** (1 / 3)
        up, dn = (engine.price(instrument, model.with_spot(model.spot + h1)).value,
                  engine.price(instrument, model.with_spot(model.spot - h1)).value)
        if "delta" in which:
            out["delta"] = (up - dn) / (2 * h1)
        if "gamma" in which:
            h2 = model.spot * _EPS ** 0.25
            up2, dn2 = (engine.price(instrument, model.with_spot(model.spot + h2)).value,
                        engine.price(instrument, model.with_spot(model.spot - h2)).value)
            out["gamma"] = (up2 - 2 * base + dn2) / h2**2

    if "vega" in which:
        h = max(model.vol * _EPS ** (1 / 3), 1e-6)
        out["vega"] = (engine.price(instrument, model.with_vol(model.vol + h)).value
                       - engine.price(instrument, model.with_vol(model.vol - h)).value) / (2 * h)

    if "rho" in which:
        h = 1e-5
        out["rho"] = (engine.price(instrument, model.with_rate(model.rate + h)).value
                      - engine.price(instrument, model.with_rate(model.rate - h)).value) / (2 * h)

    if "theta" in which:
        # Shorten expiry: dV/dt = -dV/dT. One-sided to avoid negative expiry.
        h = min(1e-4, instrument.expiry / 10)
        shorter = _with_expiry(instrument, instrument.expiry - h)
        out["theta"] = -(base - engine.price(shorter, model).value) / h

    return out


def _with_expiry(instrument: Instrument, expiry: float) -> Instrument:
    from dataclasses import replace
    return replace(instrument, expiry=expiry)
