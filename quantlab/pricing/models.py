# quantlab: Eugene (Yoogeun) Song. MIT licensed; see LICENSE.
"""Models: how the underlying moves.

A model owns the market data and the dynamics, and exposes the quantities every
engine needs -- forwards, discount factors, the standard `d+`/`d-` terms, and
the ability to simulate itself. It does not know what a call is.

Only geometric Brownian motion is implemented. The abstraction exists so a
local-vol or stochastic-vol model could be dropped in later without touching
the engines that only use `forward`, `discount`, and `evolve`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

__all__ = ["BlackScholes"]


@dataclass(frozen=True)
class BlackScholes:
    """Lognormal dynamics under the risk-neutral measure (Hull, Ch. 15).

        dS/S = (r - q) dt + sigma dW

    Parameters
    ----------
    spot : current price of the underlying
    rate : continuously compounded risk-free rate
    vol  : annualised volatility
    dividend_yield : continuous yield; the drift is `rate - dividend_yield`
                     while discounting stays at `rate`
    """

    spot: float
    rate: float
    vol: float
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError(f"spot must be positive, got {self.spot}")
        if self.vol <= 0:
            raise ValueError(f"vol must be positive, got {self.vol}")

    # -- deterministic quantities --------------------------------------------

    @property
    def carry(self) -> float:
        """Cost of carry, b = r - q. The drift of S under Q."""
        return self.rate - self.dividend_yield

    def discount(self, t: float) -> float:
        return float(np.exp(-self.rate * t))

    def forward(self, t: float) -> float:
        return float(self.spot * np.exp(self.carry * t))

    def total_variance(self, t: float) -> float:
        return self.vol**2 * t

    def d_plus_minus(self, strike, t):
        """The two standard-normal arguments in every Black-Scholes formula.

        d+ = [ln(F/K) + sigma^2 t / 2] / (sigma sqrt t)
        d- = d+ - sigma sqrt t

        Written in terms of the forward so the dividend yield needs no special
        handling anywhere downstream.
        """
        strike = np.asarray(strike, dtype=float)
        t = np.maximum(np.asarray(t, dtype=float), 1e-12)
        s = self.vol * np.sqrt(t)
        d_plus = (np.log(self.forward(t) / strike) + 0.5 * s**2) / s
        return d_plus, d_plus - s

    @staticmethod
    def N(x):
        return norm.cdf(x)

    @staticmethod
    def n(x):
        return norm.pdf(x)

    def with_spot(self, spot: float) -> "BlackScholes":
        return BlackScholes(spot, self.rate, self.vol, self.dividend_yield)

    def with_vol(self, vol: float) -> "BlackScholes":
        return BlackScholes(self.spot, self.rate, vol, self.dividend_yield)

    def with_rate(self, rate: float) -> "BlackScholes":
        return BlackScholes(self.spot, rate, self.vol, self.dividend_yield)

    # -- simulation ---------------------------------------------------------

    def evolve(self, log_spot: np.ndarray, dt: float, z: np.ndarray) -> np.ndarray:
        """One exact step of the log-price given standard-normal draws `z`.

        Exact rather than Euler: the lognormal SDE has a closed-form solution,
        so there is no time-discretisation error in the marginal distribution
        at any step -- only sampling error. Glasserman, Ch. 3.2.
        """
        return log_spot + (self.carry - 0.5 * self.vol**2) * dt + self.vol * np.sqrt(dt) * z

    def terminal_moments(self, t: float) -> tuple[float, float]:
        """E[S_t] and Var[ln S_t]. Used to validate any simulator of this model."""
        return self.forward(t), self.total_variance(t)
