"""Layer 2 - Feature engineering.

THE ONE RULE IN THIS MODULE
---------------------------
Every feature computed at row `t` may use data from rows `<= t` only.

pandas makes it very easy to break this by accident. `.rolling()` and `.shift(k)`
for k > 0 are safe. `.shift(-k)`, `.pct_change(-k)`, `.rolling(center=True)`,
`.interpolate()`, `.bfill()`, and any full-sample statistic (`.mean()` over the
whole column, `StandardScaler().fit_transform` on all data) are not.

`assert_no_lookahead` at the bottom empirically tests any feature function for
this property by truncating the input and checking the output is unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Period-over-period simple returns. Safe: uses t and t-1."""
    return prices.pct_change(fill_method=None)


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1))


def forward_returns(prices: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Returns from t to t+horizon.

    DANGEROUS BY DESIGN. This is the prediction target for research/IC analysis.
    It must never appear as a model input. Kept in this module deliberately, and
    loudly named, so its use is greppable.
    """
    return prices.shift(-horizon) / prices - 1.0


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def momentum(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Total return over [t-lookback, t-skip]. The classic "12-1".

    The skip drops the most recent month. Jegadeesh & Titman (1993) established
    the convention because very short-horizon returns exhibit reversal from
    microstructure effects -- bid-ask bounce and liquidity provision -- which
    contaminates the momentum signal. Without the skip you are partly betting on
    one-month reversal, which points the other way.

    Analogy: judging a runner's form over the last year, but ignoring yesterday's
    sprint because they might just have been dodging traffic.
    """
    if skip < 0 or lookback <= skip:
        raise ValueError(f"need 0 <= skip < lookback, got skip={skip}, lookback={lookback}")
    return prices.shift(skip) / prices.shift(lookback) - 1.0


def time_series_momentum(prices: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """Own past return, no skip. The trend-following signal.

    Moskowitz, Ooi & Pedersen (2012) found the 12-month own-return sign predicts
    the next month across 58 futures markets. Note this is *absolute*, not
    relative: it compares an asset to zero, not to its peers, so the whole
    portfolio can be flat in a bear market. That is the point -- it is what gives
    trend following its crisis convexity.
    """
    return prices / prices.shift(lookback) - 1.0


def moving_average_crossover(prices: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """(fast MA / slow MA) - 1. Positive = uptrend. A smoothed trend proxy."""
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow})")
    return prices.rolling(fast).mean() / prices.rolling(slow).mean() - 1.0


# ---------------------------------------------------------------------------
# Risk / volatility
# ---------------------------------------------------------------------------

def realized_vol(prices: pd.DataFrame, window: int = 63, annualize: bool = True) -> pd.DataFrame:
    """Trailing realized volatility of simple returns."""
    vol = simple_returns(prices).rolling(window).std()
    return vol * np.sqrt(TRADING_DAYS) if annualize else vol


def downside_vol(prices: pd.DataFrame, window: int = 63, annualize: bool = True) -> pd.DataFrame:
    """Volatility of negative returns only -- the half investors actually mind."""
    rets = simple_returns(prices)
    neg = rets.where(rets < 0)
    vol = neg.rolling(window, min_periods=max(2, window // 4)).std()
    return vol * np.sqrt(TRADING_DAYS) if annualize else vol


def rolling_beta(prices: pd.DataFrame, market: pd.Series, window: int = 252) -> pd.DataFrame:
    """Trailing OLS beta of each asset against a market proxy."""
    rets = simple_returns(prices)
    mkt = market.pct_change(fill_method=None).reindex(rets.index)
    mkt_var = mkt.rolling(window).var()
    out = {}
    for col in rets.columns:
        cov = rets[col].rolling(window).cov(mkt)
        out[col] = cov / mkt_var
    return pd.DataFrame(out, index=rets.index)


def max_drawdown_rolling(prices: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Worst peak-to-trough decline inside a trailing window."""
    roll_max = prices.rolling(window, min_periods=2).max()
    return prices / roll_max - 1.0


# ---------------------------------------------------------------------------
# Mean reversion
# ---------------------------------------------------------------------------

def zscore(frame: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Trailing z-score. Rolling, never full-sample -- a full-sample mean leaks."""
    mu = frame.rolling(window).mean()
    sd = frame.rolling(window).std()
    return (frame - mu) / sd.replace(0, np.nan)


def rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Wilder's RSI, 0-100. Below 30 oversold, above 70 overbought (by convention)."""
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ---------------------------------------------------------------------------
# Cross-sectional transforms
# ---------------------------------------------------------------------------

def cross_sectional_rank(frame: pd.DataFrame, pct: bool = True) -> pd.DataFrame:
    """Rank across symbols within each date. Row-wise -- no time leakage possible."""
    return frame.rank(axis=1, pct=pct, na_option="keep")


def cross_sectional_zscore(frame: pd.DataFrame, winsor: float | None = 3.0) -> pd.DataFrame:
    """Demean and scale across symbols within each date.

    Winsorizing matters more than it looks: one stock up 400% on a takeover will
    otherwise dominate the z-score for every other name on that date.
    """
    mu = frame.mean(axis=1)
    sd = frame.std(axis=1).replace(0, np.nan)
    z = frame.sub(mu, axis=0).div(sd, axis=0)
    return z.clip(-winsor, winsor) if winsor else z


def winsorize(frame: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Clip to cross-sectional quantiles per date."""
    lo = frame.quantile(lower, axis=1)
    hi = frame.quantile(upper, axis=1)
    return frame.clip(lo, hi, axis=0)


# ---------------------------------------------------------------------------
# Look-ahead test harness
# ---------------------------------------------------------------------------

def assert_no_lookahead(feature_fn, prices: pd.DataFrame, truncate_at: float = 0.7, tol: float = 1e-9) -> None:
    """Empirically prove a feature function is causal.

    Method: compute the feature on the full history, then on history truncated
    to `truncate_at` of its length. If any overlapping value differs, the
    function consulted the future.

    This catches leakage that visual code review misses -- a stray `bfill()`
    three calls deep, a scaler fit on the whole sample, a `center=True` window.
    """
    cut = int(len(prices) * truncate_at)
    if cut < 2:
        raise ValueError("Not enough rows to truncate meaningfully")

    full = feature_fn(prices)
    part = feature_fn(prices.iloc[:cut])

    common_idx = full.index[:cut]
    a = full.loc[common_idx]
    b = part.loc[common_idx]
    if isinstance(a, pd.Series):
        a, b = a.to_frame(), b.to_frame()
    b = b.reindex(columns=a.columns)

    both_nan = a.isna() & b.isna()
    diff = (a - b).abs()
    violations = (diff > tol) & ~both_nan
    nan_mismatch = (a.isna() != b.isna())
    bad = violations | nan_mismatch

    if bad.to_numpy().any():
        n = int(bad.to_numpy().sum())
        first = bad.stack()
        first = first[first]
        d, s = first.index[0]
        raise AssertionError(
            f"LOOK-AHEAD BIAS in {getattr(feature_fn, '__name__', feature_fn)}: "
            f"{n} value(s) changed when future data was withheld. "
            f"First divergence at {d:%Y-%m-%d} / {s}: "
            f"full={a.loc[d, s]!r} vs truncated={b.loc[d, s]!r}"
        )
