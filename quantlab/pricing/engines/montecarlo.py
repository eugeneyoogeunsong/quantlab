# quantlab: Eugene (Yoogeun) Song. MIT licensed; see LICENSE.
"""Monte Carlo engine.

Sources: Glasserman, *Monte Carlo Methods in Financial Engineering* -- path
generation Ch. 3, variance reduction Ch. 4, barrier crossing probability
Ch. 6.4, Longstaff-Schwartz Ch. 8.6, pathwise Greeks Ch. 7. Hull Ch. 21 for
the control-variate technique.

Design
------
Glasserman's decomposition, made literal:

    PathGenerator     draws paths from the model
    payoff evaluation  is the instrument's job
    estimator          averages, with variance reduction applied as plugins

The engine composes these. A barrier option does not get its own simulation
routine; it gets the standard paths plus a survival weight. An American option
does not get its own routine either; it gets the standard paths plus a
backward regression for the continuation value. What varies is small and
named, what is shared is shared.

On error bars
------------
Every Monte Carlo result here carries a standard error, and the test suite
checks the *error estimate* -- that the true value lies within a few standard
errors -- not merely that the price is close. An estimator whose reported
uncertainty is wrong is worse than one that is merely noisy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..instruments import BarrierOption, DigitalOption, Exercise, Instrument, VanillaOption
from ..models import BlackScholes
from ..results import PricingResult
from .base import UnsupportedInstrument

__all__ = ["PathGenerator", "MonteCarloEngine"]


@dataclass
class PathGenerator:
    """Draws log-price paths from a model.

    antithetic : pair every draw z with -z. The two paths' errors are
                 negatively correlated, so their average has lower variance
                 than two independent paths. Requires even n_paths.
                 Glasserman 4.2.
    """

    n_paths: int = 50_000
    n_steps: int = 252
    antithetic: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.antithetic and self.n_paths % 2:
            raise ValueError("antithetic sampling needs an even n_paths")

    def normals(self, rng: np.random.Generator) -> np.ndarray:
        if self.antithetic:
            half = rng.standard_normal((self.n_paths // 2, self.n_steps))
            return np.concatenate([half, -half])
        return rng.standard_normal((self.n_paths, self.n_steps))

    def paths(self, model: BlackScholes, expiry: float,
              rng: np.random.Generator | None = None) -> np.ndarray:
        """Spot paths, shape (n_paths, n_steps + 1), column 0 = spot."""
        rng = rng or np.random.default_rng(self.seed)
        dt = expiry / self.n_steps
        z = self.normals(rng)
        log_s = np.empty((self.n_paths, self.n_steps + 1))
        log_s[:, 0] = np.log(model.spot)
        # Exact lognormal increments, vectorised across paths (Glasserman 3.2).
        incr = (model.carry - 0.5 * model.vol**2) * dt + model.vol * np.sqrt(dt) * z
        log_s[:, 1:] = log_s[:, [0]] + np.cumsum(incr, axis=1)
        return np.exp(log_s)


@dataclass
class MonteCarloEngine:
    """Simulation pricing for European, barrier and American contracts.

    control_variate : for vanillas, use the discounted terminal spot as a
                      control. E[S_T e^{-rT}] = S0 e^{-qT} is known exactly,
                      so the regression-adjusted estimator removes the part
                      of the payoff's variance that is explained by S_T.
                      Glasserman 4.1; Hull 21.7.
    lsm_basis       : polynomial degree for the Longstaff-Schwartz regression
    """

    generator: PathGenerator = field(default_factory=PathGenerator)
    control_variate: bool = True
    lsm_basis: int = 3

    name = "monte-carlo"

    def supports(self, instrument: Instrument) -> bool:
        return isinstance(instrument, (VanillaOption, DigitalOption, BarrierOption))

    # -- helpers ------------------------------------------------------------------

    def _estimate(self, discounted: np.ndarray) -> tuple[float, float]:
        """Mean and standard error, respecting antithetic pairing.

        With antithetic sampling the 2m draws are m *pairs*; sample j and
        sample j+m share a random number and are not independent. Treating
        them as 2m independent observations reports a standard error that
        ignores the variance reduction entirely -- the test that checks
        antithetic SE < plain SE caught exactly this. The correct estimator
        (Glasserman 4.2) averages within each pair first, then takes the SE
        over the m pair-averages.
        """
        if self.generator.antithetic and len(discounted) % 2 == 0:
            m = len(discounted) // 2
            paired = 0.5 * (discounted[:m] + discounted[m:])
            return float(paired.mean()), float(paired.std(ddof=1) / np.sqrt(m))
        n = len(discounted)
        return float(discounted.mean()), float(discounted.std(ddof=1) / np.sqrt(n))

    @staticmethod
    def _control_adjust(y: np.ndarray, x: np.ndarray, x_mean: float) -> np.ndarray:
        """Regression-based control variate: y - beta (x - E[x])."""
        cov = np.cov(y, x, ddof=1)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
        return y - beta * (x - x_mean)

    @staticmethod
    def _survival_weights(paths: np.ndarray, instrument: BarrierOption,
                          model: BlackScholes, dt: float) -> np.ndarray:
        """Per-path probability of NOT having crossed the barrier, continuous time.

        Checking only the simulated points misses crossings between them, so
        a discrete check systematically underestimates knock-outs. Given two
        consecutive spots both on the safe side of barrier H, the probability
        that the Brownian bridge between them touched H has a closed form
        (Glasserman 6.4):

            P(touch) = exp( -2 ln(S_i / H) ln(S_{i+1} / H) / (sigma^2 dt) )

        The survival weight is the product over steps of (1 - P(touch)),
        accumulated in log space to avoid underflow, and forced to zero on any
        path that lands at or beyond the barrier outright.
        """
        H, sig2 = instrument.barrier, model.vol**2
        touched = instrument.is_touched(paths).any(axis=1)

        a = np.log(paths[:, :-1] / H)
        b = np.log(paths[:, 1:] / H)
        with np.errstate(over="ignore", invalid="ignore"):
            p_touch = np.exp(-2.0 * a * b / (sig2 * dt))
        p_touch = np.clip(np.nan_to_num(p_touch, nan=1.0, posinf=1.0), 0.0, 1.0)

        # A path that sits exactly on the barrier has p_touch = 1 for that step,
        # and log1p(-1) = -inf, which exponentiates to exactly zero survival --
        # the right answer, so the warning is silenced rather than avoided.
        with np.errstate(divide="ignore"):
            log_survive = np.log1p(-p_touch).sum(axis=1)
        survive = np.exp(log_survive)
        survive[touched] = 0.0
        return survive

    # -- pricing --------------------------------------------------------------------

    def price(self, instrument: Instrument, model: BlackScholes) -> PricingResult:
        if not self.supports(instrument):
            raise UnsupportedInstrument(f"MonteCarloEngine cannot price {type(instrument).__name__}")

        if isinstance(instrument, VanillaOption) and instrument.exercise is Exercise.AMERICAN:
            return self._longstaff_schwartz(instrument, model)

        T = instrument.expiry
        rng = np.random.default_rng(self.generator.seed)
        paths = self.generator.paths(model, T, rng)
        S_T = paths[:, -1]
        D = model.discount(T)

        if isinstance(instrument, BarrierOption):
            if instrument.already_triggered(model.spot):
                v = instrument.rebate * D if instrument.barrier_type.is_knock_out else \
                    self.price(instrument.underlying_vanilla, model).value
                return PricingResult(float(v), self.name, stderr=0.0)
            dt = T / self.generator.n_steps
            survive = self._survival_weights(paths, instrument, model, dt)
            base = instrument.payoff(S_T)
            if instrument.barrier_type.is_knock_out:
                payoff = base * survive + instrument.rebate * (1.0 - survive)
            else:
                payoff = base * (1.0 - survive) + instrument.rebate * survive
            disc = D * payoff
            value, se = self._estimate(disc)
            return PricingResult(value, self.name, stderr=se,
                                 diagnostics={"n_paths": self.generator.n_paths,
                                              "brownian_bridge": True})

        disc = D * instrument.payoff(S_T)
        diag = {"n_paths": self.generator.n_paths, "antithetic": self.generator.antithetic}
        if self.control_variate and isinstance(instrument, VanillaOption):
            control = D * S_T
            disc = self._control_adjust(disc, control, model.spot * np.exp(-model.dividend_yield * T))
            diag["control_variate"] = "discounted terminal spot"
        value, se = self._estimate(disc)
        return PricingResult(value, self.name, stderr=se, diagnostics=diag)

    # -- American: Longstaff-Schwartz ------------------------------------------------

    def _longstaff_schwartz(self, opt: VanillaOption, model: BlackScholes) -> PricingResult:
        """Least-squares Monte Carlo (Glasserman 8.6; Longstaff & Schwartz 2001).

        Walk backwards through time. At each step, among the paths that are in
        the money, regress the discounted value of continuing on a polynomial
        in spot; exercise wherever intrinsic beats the fitted continuation.
        Only in-the-money paths enter the regression -- fitting the others
        wastes the basis on a region where the decision is trivially "hold".

        The estimate is biased low: the exercise rule is fitted on the same
        paths that are then priced with it, and an imperfect rule can only
        lose value relative to the optimum. Glasserman's remedy is a second,
        independent set of paths to evaluate the rule; here the bias is
        simply reported as a diagnostic and the lattice serves as the check.
        """
        T = opt.expiry
        gen = self.generator
        rng = np.random.default_rng(gen.seed)
        paths = gen.paths(model, T, rng)
        n_paths, n_steps = paths.shape[0], gen.n_steps
        dt = T / n_steps
        disc = model.discount(dt)

        cashflow = opt.payoff(paths[:, -1])
        for t in range(n_steps - 1, 0, -1):
            S_t = paths[:, t]
            intrinsic = opt.payoff(S_t)
            itm = intrinsic > 0
            cashflow = cashflow * disc
            if itm.sum() < self.lsm_basis + 2:
                continue
            x = S_t[itm] / opt.strike           # normalise for conditioning
            A = np.vander(x, self.lsm_basis + 1)
            coef, *_ = np.linalg.lstsq(A, cashflow[itm], rcond=None)
            continuation = A @ coef
            exercise = intrinsic[itm] > continuation
            idx = np.flatnonzero(itm)[exercise]
            cashflow[idx] = intrinsic[itm][exercise]

        disc_cf = cashflow * disc
        value, se = self._estimate(disc_cf)
        return PricingResult(value, f"{self.name}:lsm", stderr=se,
                             diagnostics={"n_paths": n_paths, "basis_degree": self.lsm_basis,
                                          "note": "LSM estimate is biased low"})

    # -- Greeks -----------------------------------------------------------------------

    def pathwise_delta(self, opt: VanillaOption, model: BlackScholes) -> tuple[float, float]:
        """Pathwise derivative estimate of delta for a European vanilla.

        For S_T = S0 * exp(...), dS_T/dS0 = S_T/S0 pathwise, and the payoff's
        derivative is the indicator of finishing in the money. So

            delta = E[ e^{-rT} 1{ITM} S_T / S0 ]

        Unbiased, and far lower variance than bumping and re-simulating.
        Glasserman 7.2. Not applicable to digitals, whose payoff has zero
        derivative almost everywhere -- use likelihood-ratio for those.
        """
        if opt.exercise is not Exercise.EUROPEAN:
            raise UnsupportedInstrument("pathwise delta is for European vanillas")
        T = opt.expiry
        paths = self.generator.paths(model, T)
        S_T = paths[:, -1]
        phi = opt.option_type.sign
        itm = (phi * (S_T - opt.strike)) > 0
        sample = model.discount(T) * phi * itm * S_T / model.spot
        return self._estimate(sample)

    def likelihood_ratio_delta(self, instrument: Instrument, model: BlackScholes) -> tuple[float, float]:
        """Likelihood-ratio delta: works for any payoff, including digitals.

        Differentiate the density rather than the payoff:

            delta = E[ e^{-rT} payoff(S_T) * Z / (S0 sigma sqrt T) ]

        where Z is the standard normal that generated ln S_T. Unbiased and
        payoff-agnostic, at the cost of variance that grows like 1/T.
        Glasserman 7.3.
        """
        T = instrument.expiry
        rng = np.random.default_rng(self.generator.seed)
        n = self.generator.n_paths
        z = rng.standard_normal(n)
        S_T = model.spot * np.exp((model.carry - 0.5 * model.vol**2) * T + model.vol * np.sqrt(T) * z)
        score = z / (model.spot * model.vol * np.sqrt(T))
        sample = model.discount(T) * instrument.payoff(S_T) * score
        # Draws here are independent (no antithetic pairing), so plain SE.
        return float(sample.mean()), float(sample.std(ddof=1) / np.sqrt(n))
