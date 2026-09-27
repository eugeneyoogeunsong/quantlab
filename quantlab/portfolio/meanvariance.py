# quantlab: Eugene (Yoogeun) Song. MIT licensed; see LICENSE.
"""Layer 4 -- mean-variance portfolio construction with shrinkage estimators.

Sources: Markowitz (1952); Ledoit & Wolf (2003, 2004) for covariance
shrinkage; Jorion (1986) for Bayes-Stein shrinkage of the mean; Sharpe (1964)
for CAPM-implied returns; DeMiguel, Garlappi & Uppal (2009) for why all of
this is necessary.

The problem this module is designed around
------------------------------------------
Mean-variance optimisation is an *error maximiser*. Hand it noisy inputs and
it will find the asset whose expected return is most overstated and load up
on it, because that is exactly what "attractive" looks like to the objective.
DeMiguel et al. tested fourteen optimised strategies against naive 1/N across
seven datasets and found none reliably beat it out of sample. The estimation
error swamps the optimisation gain.

So the estimators here are not an afterthought. Ledoit-Wolf shrinkage pulls
the sample covariance toward a structured target by an intensity chosen to
minimise expected loss -- it is the difference between a covariance matrix
that is well-conditioned and one that is not. James-Stein shrinkage does the
same for the mean vector, which is estimated far worse than the covariance
(a mean needs decades of data to pin down; a variance needs months).

Design
------
Three interchangeable parts:

    CovarianceEstimator   returns -> Sigma        (Sample, EWMA, LedoitWolf)
    ReturnEstimator       returns -> mu           (Sample, EWMA, CAPM, JamesStein)
    Objective             (w, mu, Sigma) -> scalar to minimise

and a `MeanVariance` optimiser that takes one of each. `RollingMeanVariance`
wraps it for quantlab's backtester.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.optimize import minimize

log = logging.getLogger(__name__)

TRADING_DAYS = 252

__all__ = [
    "CovarianceEstimator", "SampleCovariance", "EWMACovariance", "LedoitWolf",
    "ReturnEstimator", "SampleMean", "EWMAMean", "CAPMImplied", "JamesStein",
    "Objective", "MinVariance", "MaxSharpe", "MaxUtility", "TargetReturn",
    "MeanVariance", "RollingMeanVariance", "efficient_frontier",
]


# ===========================================================================
# Covariance estimators
# ===========================================================================

class CovarianceEstimator(Protocol):
    def fit(self, returns: pd.DataFrame) -> pd.DataFrame: ...


@dataclass(frozen=True)
class SampleCovariance:
    """The textbook estimator. Unbiased, and badly conditioned when the number
    of assets approaches the number of observations."""

    annualise: bool = True

    def fit(self, returns: pd.DataFrame) -> pd.DataFrame:
        cov = returns.dropna(how="all").cov()
        return cov * TRADING_DAYS if self.annualise else cov


@dataclass(frozen=True)
class EWMACovariance:
    """Exponentially weighted, parameterised by half-life in trading days.

    Half-life is the number the practitioner actually has an intuition for:
    "how far back does a shock matter?" The smoothing factor
    alpha = 1 - 2^(-1/halflife) follows from it. RiskMetrics' lambda = 0.94
    corresponds to a half-life of about 11 days.
    """

    halflife: float = 30.0
    annualise: bool = True

    def fit(self, returns: pd.DataFrame) -> pd.DataFrame:
        if self.halflife <= 0:
            raise ValueError("halflife must be positive")
        r = returns.dropna(how="all")
        ewm = r.ewm(halflife=self.halflife).cov(pairwise=True)
        latest = ewm.xs(ewm.index.get_level_values(0)[-1], level=0)
        return latest * TRADING_DAYS if self.annualise else latest


@dataclass(frozen=True)
class LedoitWolf:
    """Shrink the sample covariance toward a constant-correlation target.

    Ledoit & Wolf (2003), "Honey, I shrunk the sample covariance matrix".

        Sigma_hat = delta * F + (1 - delta) * S

    where S is the sample covariance, F keeps S's variances but sets every
    correlation to the average sample correlation, and delta in [0, 1] is
    chosen analytically to minimise the expected Frobenius loss. When the
    sample is large relative to the number of assets, delta -> 0 and you get
    S back; when it is small, delta -> 1 and the structure of F dominates.

    The intensity formula is reproduced from the paper's appendix. It is the
    part that makes this an estimator rather than a heuristic: the amount of
    shrinkage is *estimated*, not chosen.
    """

    annualise: bool = True

    def fit(self, returns: pd.DataFrame) -> pd.DataFrame:
        r = returns.dropna(how="any")
        X = r.to_numpy(dtype=float)
        n, p = X.shape
        if n < 3 or p < 2:
            return SampleCovariance(self.annualise).fit(returns)

        Xc = X - X.mean(axis=0)
        S = Xc.T @ Xc / n                          # MLE sample covariance
        var = np.diag(S)
        sd = np.sqrt(var)
        corr = S / np.outer(sd, sd)
        r_bar = (corr.sum() - p) / (p * (p - 1))   # mean off-diagonal correlation

        # Target F: constant correlation
        F = r_bar * np.outer(sd, sd)
        np.fill_diagonal(F, var)

        # pi: sum of asymptotic variances of the sample covariance entries
        Y = Xc**2
        pi_mat = (Y.T @ Y) / n - S**2
        pi_hat = pi_mat.sum()

        # rho: sum of asymptotic covariances between S entries and F entries
        term = (Xc**3).T @ Xc / n                  # theta_{ii,ij}
        theta = term - np.outer(var, np.ones(p)) * S
        np.fill_diagonal(theta, 0.0)
        rho_hat = (np.trace(pi_mat)
                   + r_bar * ((np.outer(1 / sd, sd) * theta).sum()))

        # gamma: misspecification of the target
        gamma_hat = np.linalg.norm(S - F, "fro") ** 2

        kappa = (pi_hat - rho_hat) / gamma_hat if gamma_hat > 0 else 0.0
        delta = float(np.clip(kappa / n, 0.0, 1.0))

        sigma = delta * F + (1 - delta) * S
        out = pd.DataFrame(sigma, index=r.columns, columns=r.columns)
        out.attrs["shrinkage"] = delta
        return out * TRADING_DAYS if self.annualise else out


# ===========================================================================
# Expected-return estimators
# ===========================================================================

class ReturnEstimator(Protocol):
    def fit(self, returns: pd.DataFrame, **context) -> pd.Series: ...


@dataclass(frozen=True)
class SampleMean:
    annualise: bool = True

    def fit(self, returns: pd.DataFrame, **context) -> pd.Series:
        mu = returns.dropna(how="all").mean()
        return mu * TRADING_DAYS if self.annualise else mu


@dataclass(frozen=True)
class EWMAMean:
    halflife: float = 60.0
    annualise: bool = True

    def fit(self, returns: pd.DataFrame, **context) -> pd.Series:
        mu = returns.dropna(how="all").ewm(halflife=self.halflife).mean().iloc[-1]
        return mu * TRADING_DAYS if self.annualise else mu


@dataclass(frozen=True)
class CAPMImplied:
    """E[R_i] = r_f + beta_i * (E[R_M] - r_f).  Sharpe (1964).

    The case for it: betas are estimated far more precisely than means. N
    noisy mean estimates become N stable betas plus one market premium --
    a large reduction in badly-estimated inputs. Requires `market` returns
    passed as context. Regression alphas are discarded by design; historical
    alpha is mostly noise, and feeding it back in reintroduces the error
    CAPM was brought in to remove.
    """

    risk_free: float = 0.02
    market_premium: float | None = None

    def fit(self, returns: pd.DataFrame, **context) -> pd.Series:
        market = context.get("market")
        if market is None:
            raise ValueError("CAPMImplied needs market=<Series> in context")
        df = returns.join(market.rename("_mkt"), how="inner").dropna()
        if len(df) < 30:
            return SampleMean().fit(returns)
        m = df["_mkt"]
        premium = (self.market_premium if self.market_premium is not None
                   else float(m.mean() * TRADING_DAYS - self.risk_free))
        betas = df.drop(columns="_mkt").apply(lambda col: np.cov(col, m, ddof=1)[0, 1] / m.var(ddof=1))
        return self.risk_free + betas * premium


@dataclass(frozen=True)
class JamesStein:
    """Shrink another estimator's means toward their grand mean.

    Jorion (1986) applied Stein's insight to portfolio inputs: with p > 2
    assets, shrinking every mean toward a common value has lower expected
    squared error than using the raw sample means, no matter what the true
    means are. The intensity

        phi = (p - 2) / ((p - 2) + n * (mu - mu_bar)' Sigma^{-1} (mu - mu_bar))

    is large when the sample means barely differ relative to their noise and
    small when the differences are statistically real.
    """

    base: ReturnEstimator = field(default_factory=SampleMean)

    def fit(self, returns: pd.DataFrame, **context) -> pd.Series:
        mu = self.base.fit(returns, **context)
        r = returns.dropna(how="any")
        n, p = r.shape
        if p <= 2 or n < 10:
            return mu
        sigma = SampleCovariance().fit(r).reindex(index=mu.index, columns=mu.index).to_numpy()
        sigma = sigma + np.eye(p) * 1e-10
        grand = float(mu.mean())
        diff = (mu - grand).to_numpy()
        lam = float(diff @ np.linalg.solve(sigma, diff))
        phi = (p - 2) / ((p - 2) + n * lam) if lam > 0 else 1.0
        phi = float(np.clip(phi, 0.0, 1.0))
        shrunk = (1 - phi) * mu + phi * grand
        shrunk.attrs["shrinkage"] = phi
        return shrunk


# ===========================================================================
# Objectives
# ===========================================================================

class Objective(Protocol):
    name: str

    def __call__(self, w: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float: ...

    def extra_constraints(self, mu: np.ndarray) -> list[dict]: ...


@dataclass(frozen=True)
class MinVariance:
    """w' Sigma w. Uses no expected-return input at all.

    Since expected returns are the least reliable input, dropping them removes
    the dominant error, and minimum-variance portfolios routinely post better
    out-of-sample Sharpe ratios than maximum-Sharpe ones. Optimising Sharpe
    directly tends to produce a worse Sharpe -- the objective is evaluated on
    estimates, not truth.
    """

    name: str = "min-variance"

    def __call__(self, w, mu, sigma):
        return float(w @ sigma @ w)

    def extra_constraints(self, mu):
        return []


@dataclass(frozen=True)
class MaxSharpe:
    risk_free: float = 0.02
    name: str = "max-sharpe"

    def __call__(self, w, mu, sigma):
        vol = np.sqrt(max(w @ sigma @ w, 1e-16))
        return -float((w @ mu - self.risk_free) / vol)

    def extra_constraints(self, mu):
        return []


@dataclass(frozen=True)
class MaxUtility:
    """Quadratic utility: mu' w - (gamma / 2) w' Sigma w.

    Every point on the efficient frontier is the maximiser for some gamma,
    which makes this the cleanest way to parameterise risk appetite. gamma
    around 2-4 is a conventional "moderate" investor.
    """

    risk_aversion: float = 3.0
    name: str = "max-utility"

    def __call__(self, w, mu, sigma):
        return -float(w @ mu - 0.5 * self.risk_aversion * (w @ sigma @ w))

    def extra_constraints(self, mu):
        return []


@dataclass(frozen=True)
class TargetReturn:
    """Minimum variance subject to mu' w = target. One frontier point."""

    target: float
    name: str = "target-return"

    def __call__(self, w, mu, sigma):
        return float(w @ sigma @ w)

    def extra_constraints(self, mu):
        return [{"type": "eq", "fun": lambda w, m=mu, t=self.target: w @ m - t}]


# ===========================================================================
# Optimiser
# ===========================================================================

@dataclass
class MeanVariance:
    """Solve for weights given an objective and box constraints.

    Long-only with a per-asset cap by default. The cap is the single most
    effective guard against the error-maximisation problem: it stops the
    optimiser from betting the book on one overstated input.
    """

    objective: Objective = field(default_factory=MinVariance)
    covariance: CovarianceEstimator = field(default_factory=LedoitWolf)
    expected_return: ReturnEstimator = field(default_factory=SampleMean)
    lower: float = 0.0
    upper: float = 0.35
    n_restarts: int = 6
    seed: int = 0

    def solve(self, mu: pd.Series, sigma: pd.DataFrame) -> pd.Series:
        names = list(mu.index)
        sigma = sigma.reindex(index=names, columns=names)
        m, S = mu.to_numpy(dtype=float), sigma.to_numpy(dtype=float)
        p = len(m)
        S = S + np.eye(p) * 1e-10

        if p * self.lower > 1 + 1e-9 or p * self.upper < 1 - 1e-9:
            raise ValueError(
                f"box [{self.lower}, {self.upper}] with {p} assets cannot sum to 1")

        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}] + self.objective.extra_constraints(m)
        bounds = [(self.lower, self.upper)] * p
        rng = np.random.default_rng(self.seed)

        starts = [np.full(p, 1 / p)]
        starts += [np.clip(rng.dirichlet(np.ones(p)), self.lower, self.upper) for _ in range(self.n_restarts - 1)]

        best, best_f = None, np.inf
        for w0 in starts:
            w0 = w0 / w0.sum()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(self.objective, w0, args=(m, S), method="SLSQP",
                               bounds=bounds, constraints=cons,
                               options={"maxiter": 500, "ftol": 1e-12})
            if res.success and res.fun < best_f:
                best, best_f = res.x, res.fun

        if best is None:
            log.warning("%s failed to converge from %d starts; using 1/N", self.objective.name, len(starts))
            best = np.full(p, 1 / p)
        w = np.clip(best, self.lower, self.upper)
        return pd.Series(w / w.sum(), index=names)

    def fit_solve(self, returns: pd.DataFrame, **context) -> pd.Series:
        """Estimate inputs from returns, then solve."""
        mu = self.expected_return.fit(returns, **context)
        sigma = self.covariance.fit(returns)
        return self.solve(mu, sigma)


def efficient_frontier(mu: pd.Series, sigma: pd.DataFrame, n_points: int = 25,
                       lower: float = 0.0, upper: float = 1.0) -> pd.DataFrame:
    """Trace minimum variance across a range of target returns."""
    mv = MeanVariance(MinVariance(), lower=lower, upper=upper)
    w_min = mv.solve(mu, sigma)
    lo = float(w_min @ mu)
    hi = float(mu.max()) if upper >= 1 else float(
        MeanVariance(MaxUtility(1e-6), lower=lower, upper=upper).solve(mu, sigma) @ mu)
    rows = []
    for t in np.linspace(lo, hi, n_points):
        try:
            w = MeanVariance(TargetReturn(t), lower=lower, upper=upper).solve(mu, sigma)
            rows.append({"target_return": t, "volatility": float(np.sqrt(w @ sigma @ w)),
                         "weights": w.to_numpy()})
        except Exception:
            continue
    return pd.DataFrame(rows)


# ===========================================================================
# quantlab Layer 4 adapter
# ===========================================================================

@dataclass
class RollingMeanVariance:
    """Re-optimise on a calendar, hold in between. Causal by construction.

    At each rebalance date the estimators see returns strictly *before* that
    date. The final day of a truncated history counts as its partial period's
    last session and triggers a rebalance -- intended live behaviour, and not
    a leak, since the inputs are still prior data. The causality test pins
    divergence under truncation to exactly that row.
    """

    optimiser: MeanVariance = field(default_factory=MeanVariance)
    lookback: int = 252
    rebalance: str = "QE"

    def generate_weights(self, prices: pd.DataFrame,
                         market_prices: pd.Series | None = None) -> pd.DataFrame:
        returns = prices.pct_change(fill_method=None)
        market = market_prices.pct_change(fill_method=None) if market_prices is not None else None

        marks = (pd.Series(prices.index, index=prices.index)
                 .groupby(pd.Grouper(freq=self.rebalance)).max().dropna())
        rebalance_days = set(pd.DatetimeIndex(marks.values))

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        current = pd.Series(0.0, index=prices.columns)

        for day in prices.index:
            if day in rebalance_days:
                window = returns.loc[:day].iloc[:-1].tail(self.lookback).dropna(axis=1, how="all")
                if len(window) >= max(30, self.lookback // 4) and window.shape[1] >= 2:
                    ctx = {"market": market.reindex(window.index)} if market is not None else {}
                    try:
                        w = self.optimiser.fit_solve(window, **ctx)
                        current = w.reindex(prices.columns).fillna(0.0)
                    except Exception as exc:
                        log.warning("rebalance %s failed (%s); holding", day.date(), exc)
            weights.loc[day] = current
        return weights
