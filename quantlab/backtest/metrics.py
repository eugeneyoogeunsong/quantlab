"""Layer 3 - Performance and risk metrics (QA Section F).

Includes the deflated Sharpe ratio, which is the metric most likely to change
your mind about a strategy. If you tried 200 parameter combinations and kept the
best, its Sharpe is inflated by selection alone -- even if every candidate was
pure noise. The DSR asks: given that I ran N trials, how surprising is this
result really?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252

# Standard deviations below this are treated as zero. A literal `sd == 0` test
# does not work: np.std of a constant array returns ~1e-19 rather than exactly
# 0.0, which produced a Sharpe of 3.7e16 for a flat return series. Any dispersion
# below 1e-12 daily (~1.6e-10 annualised) is numerically indistinguishable from
# none, and dividing by it is meaningless.
_ZERO_VAR_TOL = 1e-12


# ---------------------------------------------------------------------------
# Return / risk
# ---------------------------------------------------------------------------

def total_return(returns: pd.Series) -> float:
    return float((1 + returns).prod() - 1)


def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    n = len(returns)
    if n == 0:
        return 0.0
    growth = (1 + returns).prod()
    if growth <= 0:
        return -1.0
    return float(growth ** (periods_per_year / n) - 1)


def volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised Sharpe. Note the standard error is roughly 1/sqrt(years):
    over 5 years a Sharpe of 0.5 has a standard error near 0.45."""
    excess = returns - rf / periods_per_year
    sd = excess.std(ddof=1)
    if not np.isfinite(sd) or sd < _ZERO_VAR_TOL:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    """Like Sharpe but penalises only downside deviation."""
    excess = returns - rf / periods_per_year
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    dd = downside.std(ddof=1)
    if not np.isfinite(dd) or dd < _ZERO_VAR_TOL:
        return 0.0
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def drawdown_series(returns: pd.Series) -> pd.Series:
    equity = (1 + returns).cumprod()
    return equity / equity.cummax() - 1


def max_drawdown(returns: pd.Series) -> float:
    dd = drawdown_series(returns)
    return float(dd.min()) if len(dd) else 0.0


def calmar_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = abs(max_drawdown(returns))
    return float(cagr(returns, periods_per_year) / mdd) if mdd > 1e-12 else 0.0


def max_drawdown_duration(returns: pd.Series) -> int:
    """Longest stretch, in periods, spent below a prior peak.

    Often the more decision-relevant number than depth: a 25% drawdown that
    recovers in four months is survivable; the same depth lasting three years
    is what makes people abandon a system at the worst possible moment.
    """
    dd = drawdown_series(returns)
    if len(dd) == 0:
        return 0
    underwater = dd < -1e-12
    best = run = 0
    for flag in underwater:
        run = run + 1 if flag else 0
        best = max(best, run)
    return int(best)


def var_historical(returns: pd.Series, level: float = 0.05) -> float:
    return float(returns.quantile(level)) if len(returns) else 0.0


def cvar_historical(returns: pd.Series, level: float = 0.05) -> float:
    """Expected shortfall -- mean loss given you are already in the worst tail."""
    if len(returns) == 0:
        return 0.0
    cutoff = returns.quantile(level)
    tail = returns[returns <= cutoff]
    return float(tail.mean()) if len(tail) else float(cutoff)


def skewness(returns: pd.Series) -> float:
    return float(stats.skew(returns.dropna())) if len(returns.dropna()) > 2 else 0.0


def kurtosis(returns: pd.Series) -> float:
    """Excess kurtosis. Normal = 0. Equity returns typically 3-8."""
    return float(stats.kurtosis(returns.dropna())) if len(returns.dropna()) > 3 else 0.0


# ---------------------------------------------------------------------------
# Benchmark-relative
# ---------------------------------------------------------------------------

def _align(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    idx = a.index.intersection(b.index)
    return a.loc[idx], b.loc[idx]


def beta_alpha(returns: pd.Series, benchmark: pd.Series, periods_per_year: int = TRADING_DAYS):
    r, b = _align(returns, benchmark)
    if len(r) < 3 or b.var() == 0:
        return 0.0, 0.0
    beta = float(np.cov(r, b, ddof=1)[0, 1] / b.var(ddof=1))
    alpha = float((r.mean() - beta * b.mean()) * periods_per_year)
    return beta, alpha


def information_ratio(returns: pd.Series, benchmark: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    r, b = _align(returns, benchmark)
    active = r - b
    sd = active.std(ddof=1)
    if not np.isfinite(sd) or sd < _ZERO_VAR_TOL:
        return 0.0
    return float(active.mean() / sd * np.sqrt(periods_per_year))


# ---------------------------------------------------------------------------
# Overfitting-aware statistics
# ---------------------------------------------------------------------------

def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sr: float = 0.0,
                               periods_per_year: int = TRADING_DAYS) -> float:
    """P(true Sharpe > benchmark_sr), adjusting for skew and fat tails.

    Bailey & Lopez de Prado. A Sharpe of 1.0 from strongly negatively skewed,
    fat-tailed returns is much weaker evidence than the same number from
    well-behaved ones -- PSR makes that explicit.
    """
    r = returns.dropna()
    n = len(r)
    if n < 10:
        return float("nan")
    sr = sharpe_ratio(r, periods_per_year=periods_per_year) / np.sqrt(periods_per_year)
    bsr = benchmark_sr / np.sqrt(periods_per_year)
    g = skewness(r)
    k = kurtosis(r) + 3.0
    denom = np.sqrt(max(1e-12, 1 - g * sr + (k - 1) / 4 * sr**2))
    z = (sr - bsr) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int,
                          trial_sr_std: float | None = None,
                          periods_per_year: int = TRADING_DAYS) -> float:
    """PSR against the Sharpe you'd expect from the BEST of `n_trials` noise strategies.

    Bailey & Lopez de Prado (2014). The mechanism: the expected maximum of N
    draws from a zero-mean distribution grows with N. Test 100 variants of a
    worthless strategy and the winner will still look good. DSR subtracts that
    expected maximum before judging.

    Interpretation: DSR > 0.95 is the usual bar. Below ~0.5, the result is
    consistent with having found nothing at all.

    `n_trials` must be the HONEST count -- every parameter set you evaluated,
    including the ones you discarded and the ones you tried before lunch.
    """
    r = returns.dropna()
    n = len(r)
    if n < 10 or n_trials < 1:
        return float("nan")

    if trial_sr_std is None:
        # Fall back to the asymptotic SE of an uninformative Sharpe estimate.
        trial_sr_std = 1.0 / np.sqrt(n - 1)

    euler = 0.5772156649015329
    if n_trials > 1:
        z1 = stats.norm.ppf(1 - 1 / n_trials)
        z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))
        expected_max_sr = trial_sr_std * ((1 - euler) * z1 + euler * z2)
    else:
        expected_max_sr = 0.0

    return probabilistic_sharpe_ratio(
        r, benchmark_sr=expected_max_sr * np.sqrt(periods_per_year),
        periods_per_year=periods_per_year,
    )


def min_track_record_length(returns: pd.Series, target_sr: float = 0.0,
                            confidence: float = 0.95,
                            periods_per_year: int = TRADING_DAYS) -> float:
    """How many periods you'd need to be `confidence` sure the Sharpe beats target.

    Frequently sobering. A Sharpe of 0.6 often needs 8-10 years of data before
    you can distinguish it from zero at 95% confidence.
    """
    r = returns.dropna()
    if len(r) < 10:
        return float("nan")
    sr = sharpe_ratio(r, periods_per_year=periods_per_year) / np.sqrt(periods_per_year)
    tsr = target_sr / np.sqrt(periods_per_year)
    if sr <= tsr:
        return float("inf")
    g, k = skewness(r), kurtosis(r) + 3.0
    z = stats.norm.ppf(confidence)
    return float(1 + (1 - g * sr + (k - 1) / 4 * sr**2) * (z / (sr - tsr)) ** 2)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(returns: pd.Series, benchmark: pd.Series | None = None,
              n_trials: int = 1, rf: float = 0.0,
              periods_per_year: int = TRADING_DAYS) -> dict:
    """Full metric block. Everything QA Section F asks for, plus the honesty checks."""
    r = returns.dropna()
    out = {
        "n_periods": int(len(r)),
        "years": round(len(r) / periods_per_year, 2),
        "total_return": total_return(r),
        "cagr": cagr(r, periods_per_year),
        "volatility": volatility(r, periods_per_year),
        "sharpe": sharpe_ratio(r, rf, periods_per_year),
        "sortino": sortino_ratio(r, rf, periods_per_year),
        "max_drawdown": max_drawdown(r),
        "max_dd_duration_days": max_drawdown_duration(r),
        "calmar": calmar_ratio(r, periods_per_year),
        "var_95": var_historical(r, 0.05),
        "cvar_95": cvar_historical(r, 0.05),
        "skew": skewness(r),
        "excess_kurtosis": kurtosis(r),
        "win_rate": float((r > 0).mean()) if len(r) else 0.0,
        "best_day": float(r.max()) if len(r) else 0.0,
        "worst_day": float(r.min()) if len(r) else 0.0,
        "psr": probabilistic_sharpe_ratio(r, 0.0, periods_per_year),
        "deflated_sharpe": deflated_sharpe_ratio(r, n_trials, periods_per_year=periods_per_year),
        "n_trials_assumed": n_trials,
        "min_track_record_years": round(
            min_track_record_length(r, 0.0, 0.95, periods_per_year) / periods_per_year, 2
        ) if len(r) >= 10 else float("nan"),
    }
    if benchmark is not None and len(benchmark.dropna()) > 2:
        b = benchmark.dropna()
        beta, alpha = beta_alpha(r, b, periods_per_year)
        out.update({
            "benchmark_cagr": cagr(b, periods_per_year),
            "benchmark_sharpe": sharpe_ratio(b, rf, periods_per_year),
            "benchmark_max_dd": max_drawdown(b),
            "beta": beta,
            "alpha": alpha,
            "information_ratio": information_ratio(r, b, periods_per_year),
            "excess_cagr": cagr(r, periods_per_year) - cagr(b, periods_per_year),
        })
    return out


def monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    """Year x month table of compounded returns. QA Section F."""
    if len(returns) == 0:
        return pd.DataFrame()
    m = (1 + returns).resample("ME").prod() - 1
    tbl = pd.DataFrame({"year": m.index.year, "month": m.index.month, "ret": m.values})
    return tbl.pivot(index="year", columns="month", values="ret")
