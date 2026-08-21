# quantlab: Eugene (Yoogeun) Song (https://www.linkedin.com/in/yoogeunsong)
# Independent side project. MIT licensed; see LICENSE.

"""Layer 4: mean-variance optimisation (Markowitz MPT).

CAPM- and EWMA-informed mean-variance portfolios, benchmarked against the
S&P 500 and an equal-weight book with quarterly rebalancing: Sharpe maximisation
against variance minimisation on cumulative growth, drawdowns, turnover, and
annual Sharpe.

Where this sits
---------------
`sizing.py` holds the heuristic weighting rules (equal weight, inverse
volatility, risk parity), none of which require a return forecast. This module is
the optimisation-based alternative: it takes an expected-return vector and a
covariance matrix, then solves for the weights that maximise Sharpe or minimise
variance.

An honest warning before you use it
-----------------------------------
Mean-variance optimisation is fragile in practice. It is a maximiser, and what it
mostly maximises is estimation error: it piles into whichever asset has the most
overstated expected return, because that is exactly what "attractive" looks like
to the objective. Small changes in the inputs therefore produce large changes in
the weights.

Michaud called it an "error maximiser", and the empirical literature repeatedly
finds naive 1/N hard to beat out of sample. That is why the constraints here
matter more than they look, and why `min_weight`/`max_weight` default to
something restrictive rather than unbounded.

The analogy is fitting a curve through data points that each carry error bars,
then reporting the fit to four decimal places: the arithmetic is exact, the
confidence is not.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

log = logging.getLogger(__name__)

TRADING_DAYS = 252

__all__ = [
    "ewma_covariance",
    "ewma_drift",
    "capm_regression",
    "capm_expected_returns",
    "optimise_portfolio",
    "efficient_frontier",
    "MeanVarianceOptimiser",
]


# ---------------------------------------------------------------------------
# Input estimation: EWMA
# ---------------------------------------------------------------------------

def ewma_covariance(returns: pd.DataFrame, alpha: float = 0.06,
                    annualise: bool = True) -> pd.DataFrame:
    """Exponentially-weighted covariance, taking the most recent estimate.

    Recent observations carry more weight, so the estimate tracks changing market
    conditions instead of averaging a crisis together with the calm years around
    it. `alpha` is the smoothing factor: 0.06 corresponds to a half-life of
    roughly 11 days, close to the RiskMetrics lambda = 0.94 convention.

    Causal by construction, since `ewm` only ever looks backwards.
    """
    if not 0 < alpha <= 1:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    clean = returns.dropna(how="all")
    cov = clean.ewm(alpha=alpha).cov(pairwise=True)
    last = cov.index.get_level_values(0).max()
    matrix = cov.loc[last]
    return matrix * TRADING_DAYS if annualise else matrix


def ewma_drift(returns: pd.DataFrame, alpha: float = 0.06,
               annualise: bool = True) -> pd.Series:
    """Exponentially-weighted mean return, most recent estimate.

    Treat this sceptically as an expected-return forecast: mean returns are
    estimated far less precisely than covariances, since pinning a mean down to
    useful accuracy takes decades of data whilst a covariance stabilises in
    months. This is the weakest input in the whole pipeline.
    """
    mu = returns.dropna(how="all").ewm(alpha=alpha).mean().iloc[-1]
    return mu * TRADING_DAYS if annualise else mu


# ---------------------------------------------------------------------------
# Input estimation: CAPM
# ---------------------------------------------------------------------------

def capm_regression(asset_returns: pd.Series, market_returns: pd.Series,
                    alpha_ewma: float | None = 0.06) -> dict:
    """Regress an asset on the market: R_i = alpha + beta*R_M + e.

    With `alpha_ewma` set, the regression is weighted so that recent observations
    count for more (WLS), letting beta drift as the relationship changes; set it
    to None for ordinary least squares over the whole sample.

    We return alpha and beta annualised, together with the residual variance.
    """
    df = pd.concat([asset_returns, market_returns], axis=1).dropna()
    if len(df) < 30:
        return {"alpha": 0.0, "beta": 1.0, "residual_var": float("nan"), "n_obs": len(df)}

    y = df.iloc[:, 0].to_numpy()
    x = df.iloc[:, 1].to_numpy()
    X = np.column_stack([np.ones(len(x)), x])

    if alpha_ewma is not None:
        # Exponentially decaying weights; the most recent observation weighs 1.
        w = (1 - alpha_ewma) ** np.arange(len(x))[::-1]
        w = w / w.sum() * len(w)
        W = np.sqrt(w)[:, None]
        beta_hat, *_ = np.linalg.lstsq(X * W, y * np.sqrt(w), rcond=None)
    else:
        beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)

    resid = y - X @ beta_hat
    return {
        "alpha": float(beta_hat[0] * TRADING_DAYS),
        "beta": float(beta_hat[1]),
        "residual_var": float(resid.var(ddof=2) * TRADING_DAYS),
        "n_obs": int(len(df)),
    }


def capm_expected_returns(returns: pd.DataFrame, market_returns: pd.Series,
                          risk_free: float = 0.02,
                          market_premium: float | None = None,
                          alpha_ewma: float | None = 0.06,
                          include_alpha: bool = False) -> pd.Series:
    """Expected returns from CAPM: E[R_i] = rf + beta_i * (E[R_M] - rf).

    We route expected returns through CAPM rather than through historical
    averages because betas are estimated far more precisely than means: N noisy
    mean estimates are replaced by N reasonably-stable betas plus ONE market
    premium estimate. That is a large reduction in the count of badly-estimated
    quantities, and mean-variance optimisation is acutely sensitive to exactly
    those.

    `include_alpha=False` by default, deliberately: historical alpha is mostly
    noise, and feeding it in reintroduces precisely the estimation error CAPM was
    brought in to avoid.
    """
    if market_premium is None:
        market_premium = float(market_returns.mean() * TRADING_DAYS - risk_free)

    out = {}
    for col in returns.columns:
        reg = capm_regression(returns[col], market_returns, alpha_ewma)
        er = risk_free + reg["beta"] * market_premium
        if include_alpha:
            er += reg["alpha"]
        out[col] = er
    return pd.Series(out)


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------

def optimise_portfolio(expected_returns, cov_matrix, risk_free: float = 0.02,
                       objective: str = "sharpe",
                       min_weight: float = 0.0, max_weight: float = 0.40,
                       n_starts: int = 8, seed: int | None = 42) -> pd.Series:
    """Solve for optimal weights subject to box constraints and full investment.

    Parameters
    ----------
    objective : 'sharpe'   maximise (mu - rf) / sigma
                'variance' minimise w' Sigma w  (ignores expected returns)
                'return'   maximise expected return (degenerate without bounds)

    Notes
    -----
    `objective='variance'` deserves attention: the minimum-variance portfolio
    uses NO expected-return input at all. Since expected returns are the least
    reliable input, dropping them removes the dominant source of error, and
    minimum-variance portfolios have historically delivered better out-of-sample
    Sharpe ratios than max-Sharpe ones. Optimising for Sharpe tends to produce a
    worse Sharpe; that is not a paradox, merely a reminder that the objective is
    evaluated on estimates rather than on truth.

    SLSQP is run from several starting points because the Sharpe objective is not
    convex and can have local optima.
    """
    mu = np.asarray(expected_returns, dtype=float)
    Sigma = np.asarray(cov_matrix, dtype=float)
    n = len(mu)

    if Sigma.shape != (n, n):
        raise ValueError(f"cov_matrix is {Sigma.shape}, expected {(n, n)}")
    if n * min_weight > 1.0 + 1e-9:
        raise ValueError(
            f"Infeasible: {n} assets at min_weight={min_weight} require "
            f"{n*min_weight:.2f} > 1.0 of capital.")
    if n * max_weight < 1.0 - 1e-9:
        raise ValueError(
            f"Infeasible: {n} assets at max_weight={max_weight} allow at most "
            f"{n*max_weight:.2f} < 1.0 of capital.")

    # Ridge the diagonal: a singular covariance (i.e., collinear assets) breaks SLSQP.
    Sigma = Sigma + np.eye(n) * 1e-10

    def neg_sharpe(w):
        vol = np.sqrt(max(w @ Sigma @ w, 1e-16))
        return -(w @ mu - risk_free) / vol

    objectives = {
        "sharpe": neg_sharpe,
        "variance": lambda w: w @ Sigma @ w,
        "return": lambda w: -(w @ mu),
    }
    if objective not in objectives:
        raise ValueError(f"objective must be one of {sorted(objectives)}, got {objective!r}")
    fn = objectives[objective]

    bounds = [(min_weight, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    rng = np.random.default_rng(seed)
    best, best_val = None, np.inf

    starts = [np.full(n, 1.0 / n)]
    for _ in range(max(0, n_starts - 1)):
        w = rng.dirichlet(np.ones(n))
        starts.append(np.clip(w, min_weight, max_weight))

    for w0 in starts:
        s = w0.sum()
        if s > 0:
            w0 = w0 / s
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(fn, w0, method="SLSQP", bounds=bounds,
                           constraints=constraints,
                           options={"maxiter": 500, "ftol": 1e-10})
        if res.success and res.fun < best_val:
            best_val, best = res.fun, res.x

    if best is None:
        log.warning("Optimisation failed from all %d starts; falling back to equal weight",
                    len(starts))
        best = np.full(n, 1.0 / n)

    best = np.clip(best, min_weight, max_weight)
    best = best / best.sum()

    index = (expected_returns.index if isinstance(expected_returns, pd.Series)
             else range(n))
    return pd.Series(best, index=index)


def efficient_frontier(expected_returns, cov_matrix, n_points: int = 25,
                       min_weight: float = 0.0, max_weight: float = 1.0) -> pd.DataFrame:
    """Trace the efficient frontier: minimum variance at each target return.

    We return a frame of target_return, volatility, and weights, ready for
    plotting the classic risk/return curve.
    """
    mu = np.asarray(expected_returns, dtype=float)
    Sigma = np.asarray(cov_matrix, dtype=float) + np.eye(len(mu)) * 1e-10
    n = len(mu)

    lo = optimise_portfolio(expected_returns, cov_matrix, objective="variance",
                            min_weight=min_weight, max_weight=max_weight)
    lo_ret = float(np.asarray(lo) @ mu)
    hi_ret = float(mu.max()) if max_weight >= 1.0 else float(
        np.asarray(optimise_portfolio(expected_returns, cov_matrix, objective="return",
                                      min_weight=min_weight, max_weight=max_weight)) @ mu)

    rows = []
    for target in np.linspace(lo_ret, hi_ret, n_points):
        cons = [
            {"type": "eq", "fun": lambda w: w.sum() - 1.0},
            {"type": "eq", "fun": lambda w, t=target: w @ mu - t},
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(lambda w: w @ Sigma @ w, np.full(n, 1.0 / n),
                           method="SLSQP", bounds=[(min_weight, max_weight)] * n,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-10})
        if res.success:
            rows.append({"target_return": target,
                         "volatility": float(np.sqrt(max(res.fun, 0.0))),
                         "weights": res.x})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# quantlab Layer 4 interface
# ---------------------------------------------------------------------------

@dataclass
class MeanVarianceOptimiser:
    """Rolling mean-variance sizing, wired into quantlab's backtester.

    Produces a date x symbol weight matrix from a price history, re-optimising on
    a fixed calendar and holding the weights in between.

    CAUSALITY: at each rebalance date, only returns strictly BEFORE that date are
    used; the optimiser never sees the period it is about to be graded on.

    Calendar note: rebalance dates are the last AVAILABLE session in each period,
    not the calendar period end, since otherwise a quarter ending on a weekend or
    a holiday would silently lose its rebalance. One consequence is that if the
    price history stops mid-period, the final day counts as that partial period's
    last session and triggers a rebalance. That is intended (live, we want to act
    on the latest close) and is not look-ahead, because the estimate still uses
    only prior data. `test_optimiser_weights_are_causal` pins this down:
    divergence under truncation is permitted on that final row and nowhere else.
    """

    lookback: int = 252
    rebalance: str = "QE"           # quarterly, as in the original study
    objective: str = "sharpe"       # 'sharpe' | 'variance'
    inputs: str = "ewma"            # 'ewma' | 'capm' | 'historical'
    risk_free: float = 0.02
    ewma_alpha: float = 0.06
    min_weight: float = 0.0
    max_weight: float = 0.40

    def _estimate(self, window: pd.DataFrame, market: pd.Series | None):
        if self.inputs == "ewma":
            mu = ewma_drift(window, self.ewma_alpha)
            cov = ewma_covariance(window, self.ewma_alpha)
        elif self.inputs == "capm":
            if market is None:
                raise ValueError("inputs='capm' requires market returns")
            mkt = market.reindex(window.index).dropna()
            mu = capm_expected_returns(window, mkt, self.risk_free,
                                       alpha_ewma=self.ewma_alpha)
            cov = ewma_covariance(window, self.ewma_alpha)
        elif self.inputs == "historical":
            mu = window.mean() * TRADING_DAYS
            cov = window.cov() * TRADING_DAYS
        else:
            raise ValueError(f"inputs must be 'ewma', 'capm' or 'historical', "
                             f"got {self.inputs!r}")
        return mu, cov

    def generate_weights(self, prices: pd.DataFrame,
                         market_prices: pd.Series | None = None) -> pd.DataFrame:
        returns = prices.pct_change(fill_method=None)
        market = (market_prices.pct_change(fill_method=None)
                  if market_prices is not None else None)

        rebal_dates = (pd.Series(prices.index, index=prices.index)
                       .groupby(pd.Grouper(freq=self.rebalance)).max().dropna())
        rebal_set = set(pd.DatetimeIndex(rebal_dates.values))

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        current = pd.Series(0.0, index=prices.columns)

        for dt in prices.index:
            if dt in rebal_set:
                # Strictly before dt: the rebalance cannot use its own bar.
                window = returns.loc[:dt].iloc[:-1].tail(self.lookback)
                window = window.dropna(axis=1, how="all")
                if len(window) >= max(30, self.lookback // 4) and window.shape[1] >= 2:
                    try:
                        mu, cov = self._estimate(window, market)
                        w = optimise_portfolio(
                            mu, cov, self.risk_free, self.objective,
                            self.min_weight, self.max_weight)
                        current = w.reindex(prices.columns).fillna(0.0)
                    except Exception as exc:
                        log.warning("optimisation failed at %s: %s -- holding previous",
                                    dt.date(), exc)
            weights.loc[dt] = current
        return weights
