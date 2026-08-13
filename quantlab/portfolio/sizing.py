"""Layer 4 - Position sizing (QA Section E).

A signal says *what* to hold. Sizing says *how much*, and it is usually the
larger determinant of the equity curve's shape. Equal-weighting a portfolio
containing both a Treasury ETF and a leveraged commodity fund means the
commodity supplies nearly all the risk and nearly all the P&L -- your careful
signal work is then mostly decoration.

Every function here is causal: all volatility and correlation estimates use
trailing windows only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equal_weight(signal_mask: pd.DataFrame) -> pd.DataFrame:
    """1/N across selected names. The honest baseline.

    Hard to beat out-of-sample. Optimisation-based schemes have more parameters
    to estimate and therefore more ways to be wrong.
    """
    sel = signal_mask.astype(bool)
    counts = sel.sum(axis=1)
    return sel.astype(float).div(counts.replace(0, np.nan), axis=0).fillna(0.0)


def inverse_volatility(signal_mask: pd.DataFrame, prices: pd.DataFrame,
                       window: int = 63, min_vol: float = 1e-4) -> pd.DataFrame:
    """Weight inversely to trailing volatility -- naive risk parity.

    Each position contributes roughly equal risk, assuming correlations are
    similar. Cheap, stable, and it captures most of what full risk parity offers
    without needing a covariance matrix.
    """
    vol = prices.pct_change(fill_method=None).rolling(window).std() * np.sqrt(TRADING_DAYS)
    vol = vol.clip(lower=min_vol)
    raw = signal_mask.astype(float) / vol
    total = raw.sum(axis=1)
    return raw.div(total.replace(0, np.nan), axis=0).fillna(0.0)


def volatility_target(weights: pd.DataFrame, prices: pd.DataFrame,
                      target_vol: float = 0.10, window: int = 63,
                      max_leverage: float = 1.5, min_leverage: float = 0.0) -> pd.DataFrame:
    """Scale the whole book so forecast portfolio vol hits a target.

    Position size becomes inversely proportional to recent volatility: the book
    shrinks automatically in turbulent markets and expands in calm ones. This is
    the single most effective risk control in the library, because volatility is
    genuinely persistent -- calm weeks cluster, turbulent weeks cluster.

    The catch is that it is backward-looking. It de-risks *after* volatility has
    already risen, so it does not protect against a one-day gap. It protects
    against the sustained turbulence that usually follows.
    """
    rets = prices.pct_change(fill_method=None)
    # Portfolio vol using yesterday's weights -- shift to keep it causal.
    port_ret = (weights.shift(1) * rets).sum(axis=1)
    realized = port_ret.rolling(window).std() * np.sqrt(TRADING_DAYS)
    scale = (target_vol / realized.replace(0, np.nan)).clip(min_leverage, max_leverage)
    scale = scale.ffill().fillna(1.0)
    return weights.mul(scale, axis=0)


def risk_parity(signal_mask: pd.DataFrame, prices: pd.DataFrame,
                window: int = 126, n_iter: int = 60) -> pd.DataFrame:
    """Equal risk contribution using the trailing covariance matrix.

    Solved iteratively rather than by an optimiser -- fewer dependencies and it
    converges reliably for long-only weights.

    Honest caveat: covariance estimated from 126 daily observations across N
    assets is noisy, and it gets worse as N grows. Below ~10 assets this is
    reasonable; above ~30 you are mostly estimating noise and inverse-vol is the
    more defensible choice.
    """
    out = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    rets = prices.pct_change(fill_method=None)

    # Recompute monthly: daily re-solves add turnover without adding information.
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
    """Fixed-point iteration toward equal risk contribution."""
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
    """Fractional Kelly leverage from trailing mean and variance.

    Full Kelly maximises long-run growth but produces drawdowns almost nobody
    tolerates in practice -- and it assumes you know the true mean return, which
    you emphatically do not. Quarter-Kelly is the common compromise: roughly 75%
    of the growth at a fraction of the volatility.
    """
    mu = returns.rolling(window).mean() * TRADING_DAYS
    var = returns.rolling(window).var() * TRADING_DAYS
    k = (mu / var.replace(0, np.nan)) * fraction
    return k.clip(0, cap).fillna(0.0)


SIZERS = {
    "equal_weight": "equal_weight",
    "inverse_vol": "inverse_volatility",
    "risk_parity": "risk_parity",
}


def apply_sizing(signal_mask: pd.DataFrame, prices: pd.DataFrame,
                 method: str = "equal_weight", **kwargs) -> pd.DataFrame:
    """Dispatch to a sizing scheme by name."""
    if method == "equal_weight":
        return equal_weight(signal_mask)
    if method == "inverse_vol":
        return inverse_volatility(signal_mask, prices, **kwargs)
    if method == "risk_parity":
        return risk_parity(signal_mask, prices, **kwargs)
    raise KeyError(f"Unknown sizing method {method!r}. Available: {sorted(SIZERS)}")
