# quantlab: Eugene (Yoogeun) Song (https://www.linkedin.com/in/yoogeunsong)
# Independent side project. MIT licensed; see LICENSE.

"""Closed-form option prices.

These are the reference values everything else is measured against: a numerical
method that cannot reproduce Black-Scholes on a vanilla European call will not
be trusted on an exotic one.

Where the formulas come from
----------------------------
Three routes reach the same expression, and each one explains a different part
of it.

(i) *The risk-neutral expectation.* Girsanov's theorem lets us tilt the measure
so that the discounted price `exp(-r*t) * S_t` is a martingale (Shreve II,
Ch. 5): the market price of risk is absorbed into the Brownian motion, the drift
of `S` becomes `(r - q)`, and the value of a European claim is just
`exp(-r*T) * E_Q[payoff]`. For lognormal `S_T` that integral is elementary,
which is where the two normal CDFs come from: `N(d2)` is the risk-neutral
probability of finishing in the money, and `N(d1)` is the same probability
computed under the share measure (i.e., with `S` itself as numeraire), which is
why `exp(-q*T) * N(d1)` doubles as delta.

(ii) *The PDE.* Feynman-Kac (Shreve II, Ch. 6) turns that expectation into the
Black-Scholes-Merton equation, `V_t + 0.5*sigma^2*S^2*V_SS + (r - q)*S*V_S =
r*V`, and back again. The two views are equivalent, but they are discretised by
different modules: `finite_difference.py` solves the PDE, `monte_carlo.py`
samples the expectation, and the fact that they agree is the package's central
correctness argument.

(iii) *The reflection principle.* A barrier price needs the joint law of the
terminal value and the running maximum. Reflecting a Brownian path about its
first-passage level supplies exactly that joint density (Shreve II, Ch. 7); one
Girsanov tilt then handles the drift, and the up-and-out call collapses to four
terms built from normal CDFs: two vanilla terms, and two reflected terms that
strip out the paths which touched the barrier.

Conventions
-----------
`T` is in years, `r` and `q` are continuously compounded, the risk-neutral drift
of `S` is `(r - q)`, and discounting is always at `r`, never at `r - q`. We write
the cost of carry as `b = r - q` where a formula is more readable that way.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = [
    "black_scholes_call",
    "black_scholes_put",
    "black_scholes_greeks",
    "black_scholes_put_call_parity_residual",
    "digital_call",
    "digital_put",
    "vega",
    "up_and_out_call_closed_form",
]


def _d1_d2(S, K, T, r, sigma, q=0.0):
    """The two arguments of the normal CDF in Black-Scholes.

    Here `T` is time to expiry (i.e., already net of any current time t).
    """
    S = np.asarray(S, dtype=float)
    T = np.maximum(np.asarray(T, dtype=float), 1e-12)  # floor T: no division by zero at expiry
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
    We assume constant volatility and a lognormal terminal distribution. Real
    option markets visibly disagree with the constant-vol part: that
    disagreement *is* the volatility smile, and inverting this formula to
    measure it is what `implied_vol.py` does.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    T = np.maximum(np.asarray(T, dtype=float), 1e-12)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S, K, T, r, sigma, q: float = 0.0):
    """European put price, built from the same d1/d2."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    T = np.maximum(np.asarray(T, dtype=float), 1e-12)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def black_scholes_put_call_parity_residual(S, K, T, r, sigma, q: float = 0.0):
    """Parity residual `C - P - (S*exp(-q*T) - K*exp(-r*T))`.

    Parity is not a Black-Scholes result: it is a static replication argument
    (long call, short put, and the forward is reproduced exactly), so it holds
    for any arbitrage-free model, any volatility, and any smile. That makes it
    the sharpest available check on the two pricers and on the dividend
    convention: the residual should be zero to floating-point rounding, i.e.
    order `1e-14` at spot 100, and any larger number localises a sign error or a
    discount factor applied at the wrong rate.

    The invariant is latent in `black_scholes_call` and `black_scholes_put`
    already; exposing it here makes it something tests and callers can assert on
    rather than something a reader has to rederive.
    """
    T = np.maximum(np.asarray(T, dtype=float), 1e-12)
    call = black_scholes_call(S, K, T, r, sigma, q)
    put = black_scholes_put(S, K, T, r, sigma, q)
    forward = S * np.exp(-q * T) - K * np.exp(-r * T)
    return call - put - forward


def digital_call(S, K, T, r, sigma, q: float = 0.0, cash: float = 1.0):
    """Cash-or-nothing digital call: pays `cash` if `S_T > K`, nothing otherwise.

    The price is `cash * exp(-r*T) * N(d2)`, i.e. the discounted risk-neutral
    probability of expiring in the money. Two consequences are worth stating.

    First, the digital is the negative strike-derivative of the vanilla:
    `digital_call = -dC/dK` at `cash = 1`, so a tight call spread
    `(C(K - h) - C(K + h)) / (2*h)` converges to it as `h -> 0`. The test suite
    uses precisely that as the independent cross-check.

    Second, the payoff is discontinuous at `K`, so delta and gamma blow up as
    expiry approaches: near the strike a digital cannot be hedged with any
    bounded position, which is why desks quote and risk-manage it as the call
    spread rather than as the limit.
    """
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    T = np.maximum(np.asarray(T, dtype=float), 1e-12)
    return cash * np.exp(-r * T) * norm.cdf(d2)


def digital_put(S, K, T, r, sigma, q: float = 0.0, cash: float = 1.0):
    """Cash-or-nothing digital put: pays `cash` if `S_T < K`.

    Complementary to `digital_call` by construction: the two together always
    pay `cash`, so their prices sum to `cash * exp(-r*T)` to machine precision.
    """
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    T = np.maximum(np.asarray(T, dtype=float), 1e-12)
    return cash * np.exp(-r * T) * norm.cdf(-d2)


def vega(S, K, T, r, sigma, q: float = 0.0):
    """Sensitivity of price to volatility, dV/dsigma.

    Identical for calls and puts (put-call parity has no sigma dependence).
    This is the derivative Newton-Raphson uses to invert for implied vol, and it
    vanishes deep in- or out-of-the-money, which is exactly where that inversion
    becomes numerically unstable.
    """
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    T = np.maximum(np.asarray(T, dtype=float), 1e-12)
    return S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)


def black_scholes_greeks(S, K, T, r, sigma, q: float = 0.0, option: str = "call",
                         second_order: bool = False) -> dict:
    """Delta, gamma, vega, theta, rho, and optionally the cross-derivatives.

    Theta is returned per YEAR; divide by 365 for the more commonly quoted
    per-calendar-day decay. All five first-order Greeks are spot Greeks under the
    same constant-vol assumptions as the price itself.

    Parameters
    ----------
    second_order : bool
        When True the dict additionally carries `vanna`, `volga` and `charm`.
        The default is False, so the returned keys are unchanged for every
        existing caller.

    Second-order Greeks
    -------------------
    These are the three a volatility desk actually watches, and each answers a
    question the first-order Greeks cannot:

    - `vanna` = d2V/dS dsigma = d(vega)/dS = d(delta)/dsigma. How the vega
      hedge drifts as spot moves, equivalently how delta responds to a
      repricing of volatility. It is the term that makes a risk-reversal
      position directional in vol, and it is what the skew is charging for.
    - `volga` (also called vomma) = d2V/dsigma2. The convexity of the position
      in volatility: long volga means a vega hedge that is systematically too
      small when vol moves, so it is the natural price of the wings, and it is
      zero exactly at `d1*d2 = 0`.
    - `charm` = d(delta)/dt, the decay of delta as calendar time passes (note
      the sign: `charm = -d(delta)/dT`, since `T` shrinks as `t` advances).
      This is what forces a delta rebalance over a weekend even when nothing
      trades, and it is largest near the strike close to expiry.

    Vanna and volga are identical for calls and puts, for the same reason gamma
    and vega are: parity is linear in `S` and free of `sigma`. Charm is not, and
    the two differ by exactly `q * exp(-q*T)`.

    Formulas follow the standard catalogue (Haug 2007), specialised to a
    continuous dividend yield with cost of carry `b = r - q`.
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

    greeks = {
        "delta": float(delta),
        "gamma": float(div * pdf_d1 / (S * sigma * sqrt_T)),
        "vega": float(S * div * sqrt_T * pdf_d1),
        "theta": float(theta),
        "rho": float(rho),
    }

    if second_order:
        b = r - q
        # Charm's shared term; the two option types then differ only by the
        # dividend piece, which is what the parity relation predicts.
        charm_shared = div * pdf_d1 * (2 * b * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T)
        if option == "call":
            charm = q * div * norm.cdf(d1) - charm_shared
        else:
            charm = -q * div * norm.cdf(-d1) - charm_shared
        greeks.update({
            "vanna": float(-div * pdf_d1 * d2 / sigma),
            "volga": float(S * div * sqrt_T * pdf_d1 * d1 * d2 / sigma),
            "charm": float(charm),
        })

    return greeks


def up_and_out_call_closed_form(S0, K, B, T, r, sigma, q: float = 0.0):
    """Up-and-out barrier call: pays like a call unless the barrier is touched.

    If the underlying ever trades at or above `B` before expiry the option is
    knocked out and pays nothing. We assume continuous monitoring.

    The price decomposes into four terms: the two you would recognise from
    vanilla Black-Scholes, and two reflection terms that subtract the value of
    paths which breached the barrier. The reflection principle for Brownian
    motion is what makes the closed form possible at all.

    A barrier option is always worth less than the equivalent vanilla, and the
    test suite asserts exactly that. As B grows without bound the knockout
    becomes unreachable and the price converges to the vanilla call.

    Parameters
    ----------
    q : float
        Continuous dividend yield, appended last so that `q = 0.0` reproduces
        the previous signature and the previous number bit-for-bit.

    Notes
    -----
    The dividend generalisation is Merton's, in the Reiner-Rubinstein (1991)
    presentation: replace the drift `r` by the cost of carry `b = r - q`
    everywhere it appears in the passage-time algebra, keep discounting at `r`,
    and the reflection exponent becomes `nu = 2*b/sigma**2`. Concretely the two
    `S`-denominated terms pick up `exp(-q*T)` and the two `K`-denominated terms
    keep `exp(-r*T)`, exactly as in the vanilla formula.

    A useful identity for testing, and the one the test suite uses to cross-check
    against a lattice that does not yet carry `q`: since a dividend yield only
    changes the drift, pricing with drift `b` and discounting at `r` equals
    pricing with drift `b` and discounting at `b`, rescaled. Therefore
    `price(S0, K, B, T, r, sigma, q) == exp(-q*T) * price(S0, K, B, T, r-q,
    sigma, 0.0)`.
    """
    if B <= K:
        # Barrier at or below the strike: every in-the-money path has already
        # knocked out, so the contract is worthless by construction.
        return 0.0
    if S0 >= B:
        return 0.0  # spot is already at or through the barrier

    b = r - q  # cost of carry; q = 0.0 recovers the pure Black-Scholes drift

    def d(t, s, sign):
        drift = (b + 0.5 * sigma**2) if sign == "+" else (b - 0.5 * sigma**2)
        return (np.log(s) - drift * t) / (sigma * np.sqrt(t))

    N = norm.cdf
    nu = 2 * b / sigma**2
    div = np.exp(-q * T)

    I1 = div * S0 * (N(d(T, B / S0, "+")) - N(d(T, K / S0, "+")))
    I2 = -np.exp(-r * T) * K * (N(d(T, B / S0, "-")) - N(d(T, K / S0, "-")))
    I3 = -div * B * (B / S0) ** nu * (N(d(T, S0 / B, "+")) - N(d(T, K * S0 / B**2, "+")))
    I4 = np.exp(-r * T) * K * (B / S0) ** (nu - 1) * (
        N(d(T, S0 / B, "-")) - N(d(T, K * S0 / B**2, "-")))

    return float(max(I1 + I2 + I3 + I4, 0.0))
