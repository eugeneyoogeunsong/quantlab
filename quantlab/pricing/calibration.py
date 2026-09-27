# quantlab: Eugene (Yoogeun) Song. MIT licensed; see LICENSE.
"""Implied volatility: running Black-Scholes backwards.

Sources: Hull Ch. 20 (the volatility smile and what implied vol measures);
Brent (1973) for the root-finder; Jaeckel, "Let's be rational" (2015) for the
observation that arbitrage bounds should be checked before any iteration.

The problem is one-dimensional root-finding on a monotone function -- the
Black-Scholes price is strictly increasing in volatility -- so a bracketing
method cannot fail once a bracket exists. Newton-Raphson on vega converges
faster near the money but diverges where vega is tiny, and that is precisely
the deep in/out-of-the-money region where market quotes are least reliable.
Brent's method gets superlinear convergence where Newton would, and falls back
to bisection where Newton would not. There is no reason to use anything else
here.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .engines.analytic import AnalyticEngine
from .instruments import Exercise, OptionType, VanillaOption
from .models import BlackScholes

__all__ = ["implied_volatility", "implied_vol_smile", "implied_vol_surface"]

_VOL_LO, _VOL_HI = 1e-6, 10.0


def _arbitrage_bounds(opt: VanillaOption, model: BlackScholes) -> tuple[float, float]:
    """Prices outside [intrinsic (discounted), asset or strike bound] have no
    implied vol, whatever the solver claims."""
    T = opt.expiry
    D, Dq = model.discount(T), np.exp(-model.dividend_yield * T)
    if opt.is_call:
        return max(model.spot * Dq - opt.strike * D, 0.0), model.spot * Dq
    return max(opt.strike * D - model.spot * Dq, 0.0), opt.strike * D


def implied_volatility(price: float, opt: VanillaOption, model: BlackScholes,
                       tol: float = 1e-10) -> float:
    """Volatility at which Black-Scholes reproduces `price`. NaN if none exists.

    `model.vol` is ignored; every other field of the model is used.
    """
    if opt.exercise is not Exercise.EUROPEAN:
        raise ValueError("implied volatility is defined for European options")
    lo, hi = _arbitrage_bounds(opt, model)
    if not (lo - 1e-12 <= price <= hi + 1e-12):
        return float("nan")
    if price <= lo + 1e-12:
        return 0.0  # at intrinsic: the zero-vol limit

    engine = AnalyticEngine()

    def gap(vol: float) -> float:
        return engine.price(opt, model.with_vol(vol)).value - price

    try:
        return float(brentq(gap, _VOL_LO, _VOL_HI, xtol=tol, maxiter=200))
    except ValueError:
        return float("nan")  # bracket failed -- price is numerically at a bound


def implied_vol_smile(prices, strikes, expiry: float, model: BlackScholes,
                      option_type: OptionType | str = OptionType.CALL) -> np.ndarray:
    """Implied vol across strikes at one expiry. The smile, if it is one."""
    prices, strikes = np.asarray(prices, dtype=float), np.asarray(strikes, dtype=float)
    return np.array([
        implied_volatility(p, VanillaOption(expiry, k, option_type), model)
        for p, k in zip(prices, strikes)
    ])


def implied_vol_surface(prices, strikes, expiries, model: BlackScholes,
                        option_type: OptionType | str = OptionType.CALL) -> np.ndarray:
    """Implied vol on a strikes x expiries grid. `prices` has that shape.

    A flat plane would mean Black-Scholes were exactly right about the
    underlying. It never is; the surface's curvature across strikes is the
    market's estimate of how wrong, and its term structure is the same across
    time. Hull Ch. 20.
    """
    prices = np.asarray(prices, dtype=float)
    strikes, expiries = np.asarray(strikes, dtype=float), np.asarray(expiries, dtype=float)
    if prices.shape != (len(strikes), len(expiries)):
        raise ValueError(f"prices has shape {prices.shape}; expected {(len(strikes), len(expiries))}")
    out = np.full(prices.shape, np.nan)
    for i, k in enumerate(strikes):
        for j, t in enumerate(expiries):
            out[i, j] = implied_volatility(prices[i, j], VanillaOption(t, k, option_type), model)
    return out
