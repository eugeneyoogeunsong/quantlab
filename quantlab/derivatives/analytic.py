"""Closed-form option prices.

Original implementations by Adrian (Adrian.ph689), 2025.
Refactored into library form; formulas unchanged.

These are the reference values everything else is measured against. A numerical
method that cannot reproduce Black-Scholes on a vanilla European call is not
going to be trusted on an exotic one.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = [
    "black_scholes_call",
    "black_scholes_put",
    "black_scholes_greeks",
    "vega",
    "up_and_out_call_closed_form",
]


def _d1_d2(S, K, T, r, sigma, q=0.0):
    """The two arguments of the normal CDF in Black-Scholes.

    `T` here is time to expiry, already net of any current time t.
    """
    S = np.asarray(S, dtype=float)
    T = np.maximum(np.asarray(T, dtype=float), 1e-12)  # avoid /0 at expiry
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    return d1, d1 - sigma * sqrt_T


def black_scholes_call(S, K, T, r, sigma, q: float = 0.0):
    """European call price under Black-Scholes-Merton.

    Parameters
    ----------
    S : float or array   spot price of the underlying
    K : float            strike
    T : float or array   time to expiry in YEARS
    r : float            continuously-compounded risk-free rate
    sigma : float        annualised volatility
    q : float            continuous dividend yield

    Notes
    -----
    The formula assumes constant volatility and a lognormal terminal
    distribution. Real option markets visibly disagree with the constant-vol
    part -- that disagreement *is* the volatility smile, and inverting this
    formula to measure it is what `implied_vol.py` does.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    T = np.maximum(np.asarray(T, dtype=float), 1e-12)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S, K, T, r, sigma, q: float = 0.0):
    """European put price. Derived from the same d1/d2."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    T = np.maximum(np.asarray(T, dtype=float), 1e-12)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def vega(S, K, T, r, sigma, q: float = 0.0):
    """Sensitivity of price to volatility, dV/dsigma.

    Identical for calls and puts (put-call parity has no sigma dependence).
    This is the derivative Newton-Raphson uses to invert for implied vol, and
    it vanishes deep in- or out-of-the-money -- which is exactly where that
    inversion becomes numerically unstable.
    """
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    T = np.maximum(np.asarray(T, dtype=float), 1e-12)
    return S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)


def black_scholes_greeks(S, K, T, r, sigma, q: float = 0.0, option: str = "call") -> dict:
    """Delta, gamma, vega, theta, rho.

    Theta is returned per YEAR. Divide by 365 for the more commonly quoted
    per-calendar-day decay.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    T = np.maximum(np.asarray(T, dtype=float), 1e-12)
    sqrt_T = np.sqrt(T)
    disc, div = np.exp(-r * T), np.exp(-q * T)
    pdf_d1 = norm.pdf(d1)

    if option not in ("call", "put"):
        raise ValueError(f"option must be 'call' or 'put', got {option!r}")

    if option == "call":
        delta = div * norm.cdf(d1)
        theta = (-S * div * pdf_d1 * sigma / (2 * sqrt_T)
                 - r * K * disc * norm.cdf(d2) + q * S * div * norm.cdf(d1))
        rho = K * T * disc * norm.cdf(d2)
    else:
        delta = -div * norm.cdf(-d1)
        theta = (-S * div * pdf_d1 * sigma / (2 * sqrt_T)
                 + r * K * disc * norm.cdf(-d2) - q * S * div * norm.cdf(-d1))
        rho = -K * T * disc * norm.cdf(-d2)

    return {
        "delta": float(delta),
        "gamma": float(div * pdf_d1 / (S * sigma * sqrt_T)),
        "vega": float(S * div * sqrt_T * pdf_d1),
        "theta": float(theta),
        "rho": float(rho),
    }


def up_and_out_call_closed_form(S0, K, B, T, r, sigma):
    """Up-and-out barrier call: pays like a call unless the barrier is touched.

    If the underlying ever trades at or above `B` before expiry, the option is
    knocked out and pays nothing. Continuous monitoring is assumed.

    The price decomposes into four terms: the two you would recognise from
    vanilla Black-Scholes, and two reflection terms that subtract the value of
    paths which breached the barrier. The reflection principle for Brownian
    motion is what makes the closed form possible at all.

    A barrier option is always worth less than the equivalent vanilla, and the
    test suite asserts exactly that. As B → infinity the knockout becomes
    unreachable and the price converges to the vanilla call.

    Original implementation by Adrian (Adrian.ph689), 2025.
    """
    if B <= K:
        # Knocked out before it can finish in the money: worthless.
        return 0.0
    if S0 >= B:
        return 0.0  # already knocked out

    def d(t, s, sign):
        drift = (r + 0.5 * sigma**2) if sign == "+" else (r - 0.5 * sigma**2)
        return (np.log(s) - drift * t) / (sigma * np.sqrt(t))

    N = norm.cdf
    nu = 2 * r / sigma**2

    I1 = S0 * (N(d(T, B / S0, "+")) - N(d(T, K / S0, "+")))
    I2 = -np.exp(-r * T) * K * (N(d(T, B / S0, "-")) - N(d(T, K / S0, "-")))
    I3 = -B * (B / S0) ** nu * (N(d(T, S0 / B, "+")) - N(d(T, K * S0 / B**2, "+")))
    I4 = np.exp(-r * T) * K * (B / S0) ** (nu - 1) * (
        N(d(T, S0 / B, "-")) - N(d(T, K * S0 / B**2, "-")))

    return float(max(I1 + I2 + I3 + I4, 0.0))
