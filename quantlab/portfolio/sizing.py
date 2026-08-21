"""Layer 4: position sizing (QA Section E).

A signal says *what* to hold; sizing says *how much*, and it is usually the
larger determinant of the equity curve's shape. Equal-weighting a portfolio that
contains both a Treasury ETF and a leveraged commodity fund means the commodity
supplies nearly all the risk and nearly all the P&L, at which point the careful
signal work is mostly decoration.

Every function here is causal: all volatility and correlation estimates use
trailing windows only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equal_weight(signal_mask: pd.DataFrame) -> pd.DataFrame:
    """1/N across the selected names. The honest baseline.

    Hard to beat out of sample: optimisation-based schemes have more parameters
    to estimate and therefore more ways to be wrong.
    """
    sel = signal_mask.astype(bool)
    counts = sel.sum(axis=1)
    return sel.astype(float).div(counts.replace(0, np.nan), axis=0).fillna(0.0)


def inverse_volatility(signal_mask: pd.DataFrame, prices: pd.DataFrame,
                       window: int = 63, min_vol: float = 1e-4) -> pd.DataFrame:
    """Weight inversely to trailing volatility (i.e., naive risk parity).

    Each position then contributes roughly equal risk, assuming correlations are
    similar. Cheap, stable, and it captures most of what full risk parity offers
    without ever forming a covariance matrix.
    """
    vol = prices.pct_change(fill_method=None).rolling(window).std() * np.sqrt(TRADING_DAYS)
    vol = vol.clip(lower=min_vol)
    raw = signal_mask.astype(float) / vol
    total = raw.sum(axis=1)
    return raw.div(total.replace(0, np.nan), axis=0).fillna(0.0)


def volatility_target(weights: pd.DataFrame, prices: pd.DataFrame,
                      target_vol: float = 0.10, window: int = 63,
                      max_leverage: float = 1.5, min_leverage: float = 0.0) -> pd.DataFrame:
    """Scale the whole book so that forecast portfolio vol hits a target.

    Position size becomes inversely proportional to recent volatility: the book
    shrinks automatically in turbulent markets and expands in calm ones. This is
    the single most effective risk control in the library, because volatility is
    genuinely persistent (calm weeks cluster, turbulent weeks cluster).

    The catch is that it is backward-looking: it de-risks *after* volatility has
    already risen, so it offers no protection against a one-day gap. What it does
    protect against is the sustained turbulence that usually follows.
    """
    rets = prices.pct_change(fill_method=None)
    # Portfolio vol from yesterday's weights; the shift is what keeps it causal.
    port_ret = (weights.shift(1) * rets).sum(axis=1)
    realized = port_ret.rolling(window).std() * np.sqrt(TRADING_DAYS)
    scale = (target_vol / realized.replace(0, np.nan)).clip(min_leverage, max_leverage)
    scale = scale.ffill().fillna(1.0)
    return weights.mul(scale, axis=0)


def risk_parity(signal_mask: pd.DataFrame, prices: pd.DataFrame,
                window: int = 126, n_iter: int = 60) -> pd.DataFrame:
    """Equal risk contribution from the trailing covariance matrix.

    Solved by fixed-point iteration rather than by an optimiser: fewer
    dependencies, and it converges reliably for long-only weights.

    Honest caveat: a covariance estimated from 126 daily observations across N
    assets is noisy, and it degrades as N grows. Below roughly 10 assets this is
    reasonable; above roughly 30 you are mostly estimating noise, and inverse-vol
    is the more defensible choice.
    """
    out = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    rets = prices.pct_change(fill_method=None)

    # Recompute monthly: daily re-solves add turnover without adding information.
    # (The covariance barely moves day to day, so the extra trading buys nothing.)
    rebal_dates = pd.Series(prices.index, index=prices.index).groupby(
        pd.Grouper(freq="ME")).max().dropna()

    last_w = None
    for dt in prices.index:
        if dt in set(rebal_dates.values) or last_w is None:
            sel = signal_mask.loc[dt]
            names = list(sel[sel.astype(bool)].index)
            if len(names) == 0:
                last_w = pd.Series(0.0, index=prices.columns)
            elif len(names) == 1:
                last_w = pd.Series(0.0, index=prices.columns)
                last_w[names[0]] = 1.0
            else:
                hist = rets.loc[:dt, names].tail(window).dropna(how="all")
                if len(hist) < window // 2:
                    w = pd.Series(1.0 / len(names), index=names)
                else:
                    cov = hist.cov().to_numpy() * TRADING_DAYS
                    w_vec = _solve_risk_parity(cov, n_iter)
                    w = pd.Series(w_vec, index=names)
                last_w = pd.Series(0.0, index=prices.columns)
                last_w[w.index] = w.values
        out.loc[dt] = last_w
    return out.fillna(0.0)


def _solve_risk_parity(cov: np.ndarray, n_iter: int = 60) -> np.ndarray:
    """Fixed-point iteration towards equal risk contribution."""
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(n_iter):
        port_vol = np.sqrt(max(1e-16, w @ cov @ w))
        mrc = cov @ w / port_vol            # marginal risk contribution
        rc = w * mrc                        # risk contribution
        target = port_vol / n
        w = w * (target / np.maximum(rc, 1e-12)) ** 0.5
        w = np.maximum(w, 0.0)
        s = w.sum()
        if s <= 0:
            return np.ones(n) / n
        w /= s
    return w


def kelly_fraction(returns: pd.Series, window: int = 252,
                   fraction: float = 0.25, cap: float = 1.0) -> pd.Series:
    """Fractional Kelly leverage from the trailing mean and variance.

    Full Kelly maximises long-run growth but produces drawdowns almost nobody
    tolerates in practice, and it assumes you know the true mean return, which
    you emphatically do not. Quarter-Kelly is the usual compromise: roughly 75%
    of the growth at a fraction of the volatility.
    """
    mu = returns.rolling(window).mean() * TRADING_DAYS
    var = returns.rolling(window).var() * TRADING_DAYS
    k = (mu / var.replace(0, np.nan)) * fraction
    return k.clip(0, cap).fillna(0.0)


def mean_variance(signal_mask: pd.DataFrame, prices: pd.DataFrame,
                  objective: str = "sharpe", inputs: str = "ewma",
                  lookback: int = 252, max_weight: float = 0.40,
                  **kwargs) -> pd.DataFrame:
    """Markowitz mean-variance weights, restricted to the selected names.

    Delegates to `portfolio.optimisation`. Unlike the heuristics above, this needs
    an expected-return estimate, which is the least reliable input in quantitative
    finance: read that module's docstring before trusting the output.

    `objective='variance'` sidesteps the problem by ignoring expected returns
    entirely, and historically tends to do better out of sample.
    """
    from .optimisation import MeanVarianceOptimiser

    opt = MeanVarianceOptimiser(
        lookback=lookback, objective=objective, inputs=inputs,
        max_weight=max_weight,
        **{k: v for k, v in kwargs.items()
           if k in {"rebalance", "risk_free", "ewma_alpha", "min_weight"}},
    )
    raw = opt.generate_weights(prices)
    # Respect the strategy's selection: zero out anything not signalled.
    masked = raw.where(signal_mask.astype(bool), 0.0)
    total = masked.sum(axis=1)
    return masked.div(total.replace(0, np.nan), axis=0).fillna(0.0)


SIZERS = {
    "equal_weight": "equal_weight",
    "inverse_vol": "inverse_volatility",
    "risk_parity": "risk_parity",
    "mean_variance": "mean_variance",
    "min_variance": "mean_variance (objective='variance')",
}


def apply_sizing(signal_mask: pd.DataFrame, prices: pd.DataFrame,
                 method: str = "equal_weight", **kwargs) -> pd.DataFrame:
    """Dispatch to a sizing scheme by name; see SIZERS for what is available."""
    if method == "equal_weight":
        return equal_weight(signal_mask)
    if method == "inverse_vol":
        return inverse_volatility(signal_mask, prices, **kwargs)
    if method == "risk_parity":
        return risk_parity(signal_mask, prices, **kwargs)
    if method == "mean_variance":
        return mean_variance(signal_mask, prices, objective="sharpe", **kwargs)
    if method == "min_variance":
        return mean_variance(signal_mask, prices, objective="variance", **kwargs)
    raise KeyError(f"Unknown sizing method {method!r}. Available: {sorted(SIZERS)}")
