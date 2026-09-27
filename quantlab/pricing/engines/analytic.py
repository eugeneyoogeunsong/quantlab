# quantlab: Eugene (Yoogeun) Song. MIT licensed; see LICENSE.
"""Closed-form prices under Black-Scholes. The reference the other engines are
scored against.

Sources: Hull, *Options, Futures, and Other Derivatives* -- vanilla and Greeks
Ch. 15 and 19; digitals and barriers Ch. 26. Barrier formulas follow Hull's
presentation of Reiner & Rubinstein (1991).

Design
------
Barriers are priced through **in-out parity** rather than by four separate
formulas. Hull gives closed forms for the knock-*in* contracts; a knock-out is
then simply the vanilla minus its knock-in twin. One formula family, one code
path, and the parity relationship is enforced by construction rather than
asserted afterwards.
"""

from __future__ import annotations

import numpy as np

from ..instruments import (
    BarrierOption,
    BarrierType,
    DigitalOption,
    Exercise,
    Instrument,
    OptionType,
    VanillaOption,
)
from ..models import BlackScholes
from ..results import PricingResult
from .base import UnsupportedInstrument

__all__ = ["AnalyticEngine"]


class AnalyticEngine:
    name = "analytic"

    def supports(self, instrument: Instrument) -> bool:
        if isinstance(instrument, VanillaOption):
            return instrument.exercise is Exercise.EUROPEAN
        return isinstance(instrument, (DigitalOption, BarrierOption))

    def price(self, instrument: Instrument, model: BlackScholes) -> PricingResult:
        if isinstance(instrument, VanillaOption):
            if instrument.exercise is Exercise.AMERICAN:
                raise UnsupportedInstrument(
                    "No closed form exists for American options under Black-Scholes; "
                    "use LatticeEngine, PDEEngine or MonteCarloEngine (Longstaff-Schwartz).")
            value = self._vanilla(instrument, model)
            greeks = self._vanilla_greeks(instrument, model)
        elif isinstance(instrument, DigitalOption):
            value = self._digital(instrument, model)
            greeks = {}
        elif isinstance(instrument, BarrierOption):
            value = self._barrier(instrument, model)
            greeks = {}
        else:
            raise UnsupportedInstrument(f"{type(instrument).__name__} has no closed form here")
        return PricingResult(float(value), self.name, greeks=greeks)

    # -- vanilla ----------------------------------------------------------------

    @staticmethod
    def _vanilla(opt: VanillaOption, m: BlackScholes) -> float:
        """Black-Scholes-Merton via the forward: Hull eq. (17.4)/(17.5).

            V = phi * D(T) * [ F N(phi d+) - K N(phi d-) ],   phi = +1 call, -1 put

        Writing it with the forward and one sign variable collapses the call
        and put formulas into a single line and makes the dividend yield free.
        """
        phi = opt.option_type.sign
        d_plus, d_minus = m.d_plus_minus(opt.strike, opt.expiry)
        F, D = m.forward(opt.expiry), m.discount(opt.expiry)
        return phi * D * (F * m.N(phi * d_plus) - opt.strike * m.N(phi * d_minus))

    @staticmethod
    def _vanilla_greeks(opt: VanillaOption, m: BlackScholes) -> dict[str, float]:
        """Hull, Ch. 19. Theta is per year."""
        phi = opt.option_type.sign
        T, K = opt.expiry, opt.strike
        d_plus, d_minus = m.d_plus_minus(K, T)
        sqrt_T = np.sqrt(T)
        D = m.discount(T)
        Dq = np.exp(-m.dividend_yield * T)
        S = m.spot
        n_dp = m.n(d_plus)

        delta = phi * Dq * m.N(phi * d_plus)
        gamma = Dq * n_dp / (S * m.vol * sqrt_T)
        vega = S * Dq * n_dp * sqrt_T
        theta = (-S * Dq * n_dp * m.vol / (2 * sqrt_T)
                 - phi * m.rate * K * D * m.N(phi * d_minus)
                 + phi * m.dividend_yield * S * Dq * m.N(phi * d_plus))
        rho = phi * K * T * D * m.N(phi * d_minus)
        return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega),
                "theta": float(theta), "rho": float(rho)}

    # -- digital ----------------------------------------------------------------

    @staticmethod
    def _digital(opt: DigitalOption, m: BlackScholes) -> float:
        """Cash-or-nothing: Hull, Ch. 26.  V = cash * D(T) * N(phi d-)."""
        phi = opt.option_type.sign
        _, d_minus = m.d_plus_minus(opt.strike, opt.expiry)
        return opt.cash * m.discount(opt.expiry) * m.N(phi * d_minus)

    # -- barrier ----------------------------------------------------------------

    def _barrier(self, opt: BarrierOption, m: BlackScholes) -> float:
        """Knock-out = vanilla - knock-in (in-out parity); knock-ins from Hull Ch. 26."""
        if opt.already_triggered(m.spot):
            # A knock-out at or past its barrier is dead; a knock-in is now vanilla.
            return opt.rebate if opt.barrier_type.is_knock_out else self._vanilla(
                opt.underlying_vanilla, m)

        vanilla = self._vanilla(opt.underlying_vanilla, m)
        knock_in = self._knock_in(opt, m)

        if opt.barrier_type.is_knock_out:
            value = vanilla - knock_in
            if opt.rebate:
                value += opt.rebate * m.discount(opt.expiry) * self._touch_probability(opt, m)
            return max(value, 0.0)
        return max(knock_in, 0.0)

    @staticmethod
    def _knock_in(opt: BarrierOption, m: BlackScholes) -> float:
        """Hull's knock-in formulas, all four cases, using his lambda / x1 / y / y1.

        lambda = (b + sigma^2/2) / sigma^2
        y      = ln(H^2 / (S K)) / (sigma sqrt T) + lambda sigma sqrt T
        x1     = ln(S / H) / (sigma sqrt T) + lambda sigma sqrt T
        y1     = ln(H / S) / (sigma sqrt T) + lambda sigma sqrt T
        """
        S, K, H, T = m.spot, opt.strike, opt.barrier, opt.expiry
        sig, b = m.vol, m.carry
        st = sig * np.sqrt(T)
        D, Dq = m.discount(T), np.exp(-m.dividend_yield * T)
        N = m.N

        lam = (b + 0.5 * sig**2) / sig**2
        y = np.log(H**2 / (S * K)) / st + lam * st
        x1 = np.log(S / H) / st + lam * st
        y1 = np.log(H / S) / st + lam * st
        ratio_2lam = (H / S) ** (2 * lam)
        ratio_2lam_2 = (H / S) ** (2 * lam - 2)

        kind, is_call = opt.barrier_type, opt.option_type is OptionType.CALL

        if is_call:
            if kind in (BarrierType.DOWN_AND_IN, BarrierType.DOWN_AND_OUT):
                if H <= K:
                    return (S * Dq * ratio_2lam * N(y)
                            - K * D * ratio_2lam_2 * N(y - st))
                # H > K: down-and-in call = vanilla - down-and-out call, where
                # down-and-out call = S Dq N(x1) - K D N(x1 - st) - S Dq (H/S)^2lam N(y1) + K D (H/S)^(2lam-2) N(y1 - st)
                vanilla = AnalyticEngine._vanilla(opt.underlying_vanilla, m)
                down_out = (S * Dq * N(x1) - K * D * N(x1 - st)
                            - S * Dq * ratio_2lam * N(y1)
                            + K * D * ratio_2lam_2 * N(y1 - st))
                return vanilla - down_out
            # up barriers, call
            if H <= K:
                return AnalyticEngine._vanilla(opt.underlying_vanilla, m)  # up-and-out is worthless
            return (S * Dq * N(x1) - K * D * N(x1 - st)
                    - S * Dq * ratio_2lam * (N(-y) - N(-y1))
                    + K * D * ratio_2lam_2 * (N(-y + st) - N(-y1 + st)))

        # puts
        if kind in (BarrierType.UP_AND_IN, BarrierType.UP_AND_OUT):
            if H >= K:
                return (-S * Dq * ratio_2lam * N(-y)
                        + K * D * ratio_2lam_2 * N(-y + st))
            vanilla = AnalyticEngine._vanilla(opt.underlying_vanilla, m)
            up_out = (-S * Dq * N(-x1) + K * D * N(-x1 + st)
                      + S * Dq * ratio_2lam * N(-y1)
                      - K * D * ratio_2lam_2 * N(-y1 + st))
            return vanilla - up_out
        # down barriers, put
        if H >= K:
            return AnalyticEngine._vanilla(opt.underlying_vanilla, m)  # down-and-out put worthless
        return (-S * Dq * N(-x1) + K * D * N(-x1 + st)
                + S * Dq * ratio_2lam * (N(y) - N(y1))
                - K * D * ratio_2lam_2 * (N(y - st) - N(y1 - st)))

    @staticmethod
    def _touch_probability(opt: BarrierOption, m: BlackScholes) -> float:
        """Risk-neutral probability of touching the barrier before expiry.

        Reflection principle for Brownian motion with drift (Hull, Ch. 26
        discussion of rebates). Only used when a rebate is specified.
        """
        S, H, T, sig = m.spot, opt.barrier, opt.expiry, m.vol
        mu = m.carry - 0.5 * sig**2           # drift of ln S
        st = sig * np.sqrt(T)
        h = np.log(H / S)                      # > 0 for up, < 0 for down
        # P(max_{t<=T} ln S_t >= ln H) for an up-barrier; mirror for down.
        # Both cases collapse to one expression in terms of |h| and the
        # drift *towards* the barrier.
        drift_toward = mu if opt.barrier_type.is_up else -mu
        a = np.abs(h)
        return float(m.N((-a + drift_toward * T) / st)
                     + np.exp(2 * drift_toward * a / sig**2) * m.N((-a - drift_toward * T) / st))
