# quantlab: Eugene (Yoogeun) Song. MIT licensed; see LICENSE.
"""Finite-difference engine: the Black-Scholes PDE on a spot/time grid.

Sources: Wilmott, *Paul Wilmott on Quantitative Finance* Ch. 77-78 (theta
schemes, boundary conditions, projected SOR for American options); Hull Ch. 21
(the explicit/implicit correspondence with trinomial trees); Rannacher (1984)
for the start-up smoothing.

Design
------
One time-stepping routine, parameterised by theta:

    theta = 0    explicit          (conditionally stable, first order)
    theta = 1/2  Crank-Nicolson    (unconditionally stable, second order)
    theta = 1    fully implicit    (unconditionally stable, first order)

Written in backward time tau = T - t, the PDE is

    dV/dtau = (1/2) sigma^2 S^2 V_SS + b S V_S - r V

and on a uniform grid S_i = i * dS the spatial operator L is tridiagonal with

    lower_i = (1/2) dtau (sigma^2 i^2 - b i)
    diag_i  =      -dtau (sigma^2 i^2 + r)
    upper_i = (1/2) dtau (sigma^2 i^2 + b i)

The theta step is then  (I - theta L) V_{n+1} = (I + (1 - theta) L) V_n,
a banded solve per step. Boundary values come from the instrument: a call
grows linearly at large S, a put is worth the discounted strike at S = 0, a
knock-out is worth its rebate on the barrier. Placing the grid's edge *on* a
barrier is what makes the PDE the natural home for barrier options -- no
correction terms, just a Dirichlet condition.

American exercise turns the linear system into a linear complementarity
problem: V >= payoff everywhere, with equality wherever exercise is optimal.
Projected SOR solves it properly; simple projection (solve, then take the max
with intrinsic) is cheaper and adequate for most uses. Both are offered.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from ..instruments import BarrierOption, DigitalOption, Exercise, Instrument, OptionType, VanillaOption
from ..models import BlackScholes
from ..results import PricingResult
from .base import UnsupportedInstrument

__all__ = ["PDEEngine"]


@dataclass
class PDEEngine:
    """Theta-scheme finite differences.

    Parameters
    ----------
    n_space         : spatial nodes (excluding the two boundaries)
    n_time          : time steps
    theta           : 0 explicit, 0.5 Crank-Nicolson, 1 implicit
    s_max_multiple  : upper grid edge as a multiple of max(spot, strike);
                      ignored when a barrier fixes the edge
    rannacher       : replace the first two steps with fully implicit ones.
                      Crank-Nicolson rings for a few steps near a payoff kink
                      or discontinuity; implicit Euler damps the ringing and
                      the second-order accuracy is recovered thereafter.
    american_solver : 'projection' or 'psor'
    psor_omega      : SOR relaxation parameter (1 < omega < 2)
    psor_tol        : convergence tolerance for PSOR
    """

    n_space: int = 400
    n_time: int = 400
    theta: float = 0.5
    s_max_multiple: float = 4.0
    rannacher: bool = True
    american_solver: str = "projection"
    psor_omega: float = 1.4
    psor_tol: float = 1e-8
    psor_max_iter: int = 200

    name = "pde"

    def __post_init__(self) -> None:
        if not 0.0 <= self.theta <= 1.0:
            raise ValueError("theta must lie in [0, 1]")
        if self.american_solver not in ("projection", "psor"):
            raise ValueError("american_solver must be 'projection' or 'psor'")

    def supports(self, instrument: Instrument) -> bool:
        return isinstance(instrument, (VanillaOption, DigitalOption, BarrierOption))

    # -- grid and boundaries ----------------------------------------------------

    def _grid(self, instrument: Instrument, model: BlackScholes) -> np.ndarray:
        strike = getattr(instrument, "strike", model.spot)
        lo, hi = 0.0, self.s_max_multiple * max(model.spot, strike)
        if isinstance(instrument, BarrierOption) and instrument.barrier_type.is_knock_out:
            if instrument.barrier_type.is_up:
                hi = instrument.barrier
            else:
                lo = instrument.barrier
        return np.linspace(lo, hi, self.n_space + 2)

    @staticmethod
    def _boundary_values(instrument: Instrument, model: BlackScholes,
                         S_lo: float, S_hi: float, tau: float) -> tuple[float, float]:
        """Dirichlet values at the two grid edges, at backward time tau."""
        D, Dq = model.discount(tau), np.exp(-model.dividend_yield * tau)

        if isinstance(instrument, BarrierOption) and instrument.barrier_type.is_knock_out:
            rebate = instrument.rebate * D
            base = instrument.underlying_vanilla
            if instrument.barrier_type.is_up:
                # lower edge is S=0; upper edge is the barrier
                lo = base.strike * D if not base.is_call else 0.0
                return lo, rebate
            hi = S_hi * Dq - base.strike * D if base.is_call else 0.0
            return rebate, hi

        if isinstance(instrument, VanillaOption):
            if instrument.is_call:
                return 0.0, max(S_hi * Dq - instrument.strike * D, 0.0)
            return instrument.strike * D, 0.0

        if isinstance(instrument, DigitalOption):
            return (0.0, instrument.cash * D) if instrument.is_call else (instrument.cash * D, 0.0)

        raise UnsupportedInstrument(type(instrument).__name__)

    # -- the operator -----------------------------------------------------------

    @staticmethod
    def _operator(model: BlackScholes, S: np.ndarray, dtau: float):
        """Tridiagonal coefficients of dtau * L on interior nodes (Wilmott Ch. 77)."""
        dS = S[1] - S[0]
        i = S[1:-1] / dS
        sig2 = model.vol**2
        lower = 0.5 * dtau * (sig2 * i**2 - model.carry * i)
        diag = -dtau * (sig2 * i**2 + model.rate)
        upper = 0.5 * dtau * (sig2 * i**2 + model.carry * i)
        return lower, diag, upper

    # -- time stepping ----------------------------------------------------------

    def _solve(self, instrument: Instrument, model: BlackScholes, want_surface: bool = False):
        S = self._grid(instrument, model)
        S_lo, S_hi = S[0], S[-1]
        T = instrument.expiry
        dtau = T / self.n_time
        lower, diag, upper = self._operator(model, S, dtau)
        n_int = len(S) - 2

        payoff = np.asarray(instrument.payoff(S), dtype=float)
        if isinstance(instrument, BarrierOption) and instrument.barrier_type.is_knock_out:
            # The barrier edge is dead at expiry too.
            payoff[-1 if instrument.barrier_type.is_up else 0] = instrument.rebate
        V = payoff.copy()
        intrinsic = payoff[1:-1]
        american = instrument.exercise is Exercise.AMERICAN

        for step in range(self.n_time):
            tau = (step + 1) * dtau
            # Rannacher: two fully implicit steps to damp Crank-Nicolson ringing.
            th = 1.0 if (self.rannacher and step < 2) else self.theta

            # Right-hand side: (I + (1 - th) L) V_n on interior, plus boundary terms.
            rhs = V[1:-1].copy()
            if th < 1.0:
                w = 1.0 - th
                rhs += w * (lower * V[:-2] + diag * V[1:-1] + upper * V[2:])

            b_lo, b_hi = self._boundary_values(instrument, model, S_lo, S_hi, tau)

            if th == 0.0:
                V_new_int = rhs
            else:
                # Left-hand side (I - th L) in scipy banded form.
                ab = np.zeros((3, n_int))
                ab[0, 1:] = -th * upper[:-1]
                ab[1, :] = 1.0 - th * diag
                ab[2, :-1] = -th * lower[1:]
                rhs[0] += th * lower[0] * b_lo
                rhs[-1] += th * upper[-1] * b_hi
                if american and self.american_solver == "psor":
                    V_new_int = self._psor(ab, rhs, intrinsic, V[1:-1])
                else:
                    V_new_int = solve_banded((1, 1), ab, rhs)

            V = np.empty_like(V)
            V[0], V[-1] = b_lo, b_hi
            V[1:-1] = V_new_int
            if american and self.american_solver == "projection":
                V[1:-1] = np.maximum(V[1:-1], intrinsic)

        return (S, V) if want_surface else (S, V)

    def _psor(self, ab: np.ndarray, rhs: np.ndarray, intrinsic: np.ndarray,
              guess: np.ndarray) -> np.ndarray:
        """Projected successive over-relaxation for the LCP  A x = b,  x >= g.

        Gauss-Seidel is inherently sequential, so this is a Python loop. Grids
        of a few hundred nodes converge in tens of sweeps; it is the price of
        an exact early-exercise boundary rather than a projected approximation.
        """
        n = len(rhs)
        upper, main, lower = ab[0], ab[1], ab[2]
        x = np.maximum(guess.copy(), intrinsic)
        omega = self.psor_omega
        for _ in range(self.psor_max_iter):
            max_change = 0.0
            for i in range(n):
                s = rhs[i]
                if i > 0:
                    s -= lower[i - 1] * x[i - 1]
                if i < n - 1:
                    s -= upper[i + 1] * x[i + 1]
                gs = s / main[i]
                new = max(intrinsic[i], x[i] + omega * (gs - x[i]))
                max_change = max(max_change, abs(new - x[i]))
                x[i] = new
            if max_change < self.psor_tol:
                break
        return x

    # -- public -----------------------------------------------------------------

    def price(self, instrument: Instrument, model: BlackScholes) -> PricingResult:
        if not self.supports(instrument):
            raise UnsupportedInstrument(f"PDEEngine cannot price {type(instrument).__name__}")

        if isinstance(instrument, BarrierOption) and not instrument.barrier_type.is_knock_out:
            # Knock-in by parity.
            from .analytic import AnalyticEngine  # local import to avoid cycles
            v_out = self.price(instrument.with_barrier_type(instrument.barrier_type.complement), model).value
            v_van = self.price(instrument.underlying_vanilla, model).value
            return PricingResult(float(v_van - v_out), self.name,
                                 diagnostics={"via": "in-out parity"})

        if isinstance(instrument, BarrierOption) and instrument.already_triggered(model.spot):
            return PricingResult(float(instrument.rebate * model.discount(instrument.expiry)), self.name)

        S, V = self._solve(instrument, model)
        value = float(np.interp(model.spot, S, V))
        greeks = self._greeks_from_surface(S, V, model.spot)
        return PricingResult(value, self.name, greeks=greeks,
                             diagnostics={"n_space": self.n_space, "n_time": self.n_time,
                                          "theta": self.theta})

    def surface(self, instrument: Instrument, model: BlackScholes):
        """The full price-vs-spot curve at t = 0. Free by-product of the solve."""
        return self._solve(instrument, model, want_surface=True)

    @staticmethod
    def _greeks_from_surface(S: np.ndarray, V: np.ndarray, spot: float) -> dict[str, float]:
        """Delta and gamma by central differences, evaluated *at spot*.

        Spot rarely lands on a grid node. Differencing at the nearest node
        gives the slope somewhere else -- at S = 100 on a grid with dS = 1.09
        that produced a delta 4% off. Interpolating V to spot +/- h first
        and differencing that removes the error.
        """
        dS = S[1] - S[0]
        h = dS
        v_p, v_0, v_m = (np.interp(spot + h, S, V), np.interp(spot, S, V),
                         np.interp(spot - h, S, V))
        return {"delta": float((v_p - v_m) / (2 * h)),
                "gamma": float((v_p - 2 * v_0 + v_m) / h**2)}
