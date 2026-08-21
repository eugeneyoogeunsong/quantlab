# quantlab: Eugene (Yoogeun) Song (https://www.linkedin.com/in/yoogeunsong)
# Ported from Adrian's (Adrian.ph689) independent work, used with permission;
# see CREDITS.md. Independent side project. MIT licensed; see LICENSE.

"""Implied volatility by Newton-Raphson inversion.

Original implementation by Adrian (Adrian.ph689), 2025, following Wilmott,
*Paul Wilmott Introduces Quantitative Finance*, Ch. 8. Refactored into library
form with a bisection fallback added for robustness.

Every other function in this package maps volatility to a price; this one runs
the map backwards: given the price the market is actually quoting, what
volatility does Black-Scholes need in order to agree?

That number is not a forecast. It is the market's price expressed in different
units, and the fact that it differs across strikes (the smile) is precisely the
market telling us Black-Scholes is wrong about lognormal returns.
"""

from __future__ import annotations

import numpy as np

from .analytic import black_scholes_call, black_scholes_put, vega

__all__ = ["implied_volatility", "implied_vol_surface"]


def _intrinsic_bounds(S, K, T, r, q, option):
    """No-arbitrage bounds; a price outside these has no implied vol at all."""
    disc, div = np.exp(-r * T), np.exp(-q * T)
    if option == "call":
        return max(S * div - K * disc, 0.0), S * div
    return max(K * disc - S * div, 0.0), K * disc


def implied_volatility(price: float, S: float, K: float, T: float, r: float,
                       q: float = 0.0, option: str = "call",
                       tol: float = 1e-8, max_iter: int = 100,
                       initial_guess: float | None = None) -> float:
    """Invert Black-Scholes for volatility.

    Returns NaN when no solution exists, rather than a misleading number.

    Method
    ------
    Newton-Raphson first: it converges quadratically, using vega as the
    derivative. Vega collapses to nearly zero for deep in- or out-of-the-money
    options, however, and dividing by a near-zero derivative sends the iteration
    somewhere useless; when that happens we fall back to bisection, which is
    slower but cannot diverge.

    The original implementation handled the same instability by retrying from
    several random starting guesses. Bisection is deterministic and provably
    convergent on a bracketed root, so it is preferred here; the
    Brenner-Subrahmanyam approximation then supplies a sensible starting point
    instead of a random one.
    """
    if option not in ("call", "put"):
        raise ValueError(f"option must be 'call' or 'put', got {option!r}")
    if T <= 0 or S <= 0 or K <= 0:
        return float("nan")

    lower, upper = _intrinsic_bounds(S, K, T, r, q, option)
    if price < lower - 1e-10 or price > upper + 1e-10:
        # Outside the no-arbitrage bounds: no volatility reproduces this price,
        # so we return NaN rather than a number that would look like an answer.
        return float("nan")

    pricer = black_scholes_call if option == "call" else black_scholes_put

    if initial_guess is not None:
        sigma = initial_guess
    else:
        # Brenner-Subrahmanyam: accurate near the money, adequate elsewhere, and
        # clipped below so the first vega evaluation is never degenerate.
        sigma = max(np.sqrt(2 * np.pi / T) * price / S, 1e-3)
    sigma = float(np.clip(sigma, 1e-4, 5.0))

    # --- Newton-Raphson ---
    for _ in range(max_iter):
        diff = float(pricer(S, K, T, r, sigma, q)) - price
        if abs(diff) < tol:
            return float(sigma)
        v = float(vega(S, K, T, r, sigma, q))
        if v < 1e-10:
            break  # derivative too flat to divide by; hand over to bisection
        step = diff / v
        step = float(np.clip(step, -0.5, 0.5))  # damp jumps a flat vega would produce
        sigma_new = sigma - step
        if sigma_new <= 0 or sigma_new > 5.0:
            break
        if abs(sigma_new - sigma) < 1e-12:
            return float(sigma_new)
        sigma = sigma_new

    # --- Bisection fallback ---
    lo, hi = 1e-6, 5.0
    f_lo = float(pricer(S, K, T, r, lo, q)) - price
    f_hi = float(pricer(S, K, T, r, hi, q)) - price
    if f_lo * f_hi > 0:
        return float("nan")  # root not bracketed on [1e-6, 5.0]
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = float(pricer(S, K, T, r, mid, q)) - price
        if abs(f_mid) < tol or (hi - lo) < 1e-12:
            return float(mid)
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return float(0.5 * (lo + hi))


def implied_vol_surface(prices, spots, expiries, K: float, r: float,
                        q: float = 0.0, option: str = "call") -> np.ndarray:
    """Implied vol across a grid of spots and expiries.

    Parameters
    ----------
    prices   : 2-D array (len(spots), len(expiries)) of observed option prices
    spots    : 1-D array of underlying prices
    expiries : 1-D array of times to expiry, in years

    Returns a matching 2-D array of implied vols, NaN where no solution exists.

    Plotted as a surface this is the standard visualisation of the volatility
    smile/skew: a flat plane would mean Black-Scholes were exactly right, and it
    never is.
    """
    prices = np.asarray(prices, dtype=float)
    spots = np.asarray(spots, dtype=float)
    expiries = np.asarray(expiries, dtype=float)

    if prices.shape != (len(spots), len(expiries)):
        raise ValueError(
            f"prices has shape {prices.shape}, expected {(len(spots), len(expiries))}")

    surface = np.full(prices.shape, np.nan)
    for i, S in enumerate(spots):
        for j, T in enumerate(expiries):
            surface[i, j] = implied_volatility(prices[i, j], S, K, T, r, q, option)
    return surface
