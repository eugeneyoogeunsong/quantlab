# quantlab: Eugene (Yoogeun) Song (https://www.linkedin.com/in/yoogeunsong)
# Independent side project. MIT licensed; see LICENSE.

"""Layer 3: validation (QA Checklist Section C).

Three tests, in increasing order of how often they kill a strategy:

1. Walk-forward   : does it work on data it was not fitted to?
2. Regime split   : does it work in more than one kind of market?
3. Param sweep    : does it work at parameters you did not pick?

Section G of the checklist says it plainly: if performance collapses once you
add costs, test out-of-sample, or change parameters, it is probably not robust.
These functions exist to produce that collapse early, on a laptop, rather than
later, with money.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from . import metrics as M
from .engine import BacktestEngine, BacktestResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardWindow:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict = field(default_factory=dict)
    train_sharpe: float = float("nan")
    test_sharpe: float = float("nan")
    test_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


@dataclass
class WalkForwardResult:
    windows: list[WalkForwardWindow]
    oos_returns: pd.Series
    embargo_days: int = 0

    @property
    def degradation(self) -> float:
        """Mean train Sharpe minus mean test Sharpe.

        Some decay is normal and expected. A large gap means the optimiser was
        fitting noise: it found patterns specific to the training window, and
        those patterns do not survive contact with the next one.
        """
        tr = np.nanmean([w.train_sharpe for w in self.windows])
        te = np.nanmean([w.test_sharpe for w in self.windows])
        return float(tr - te)

    @property
    def consistency(self) -> float:
        """Fraction of test folds with a positive Sharpe.

        More informative than the average: one enormous fold can carry a mean
        whilst five of the six folds lost money.
        """
        sr = [w.test_sharpe for w in self.windows if not np.isnan(w.test_sharpe)]
        return float(np.mean([s > 0 for s in sr])) if sr else 0.0

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "fold": w.fold,
                "train": f"{w.train_start:%Y-%m} to {w.train_end:%Y-%m}",
                "test": f"{w.test_start:%Y-%m} to {w.test_end:%Y-%m}",
                "train_sharpe": round(w.train_sharpe, 3),
                "test_sharpe": round(w.test_sharpe, 3),
                "best_params": w.best_params,
            }
            for w in self.windows
        ])


def walk_forward(
    prices: pd.DataFrame,
    strategy_factory: Callable[..., Any],
    param_grid: dict[str, Sequence],
    engine: BacktestEngine | None = None,
    n_folds: int = 5,
    train_frac: float = 0.6,
    anchored: bool = False,
    embargo_days: int = 5,
) -> WalkForwardResult:
    """Rolling (or anchored) walk-forward with an embargo gap.

    The embargo is the subtle part. If the signal uses a 252-day lookback and the
    test set begins the day after training ends, the first test observations were
    computed mostly from training data; they are not really out-of-sample, and a
    walk-forward that ignores this reports leakage as skill. The embargo drops a
    gap between the two windows so that the leakage window closes.

    `anchored=True` holds the training start fixed and grows the window, which is
    closer to how we would actually re-fit a live system over time.
    """
    engine = engine or BacktestEngine()
    idx = pd.DatetimeIndex(prices.index)
    n = len(idx)
    if n < 200:
        raise ValueError(f"Need at least 200 bars for walk-forward, got {n}")

    train_len = int(n * train_frac)
    remaining = n - train_len
    test_len = max(21, remaining // n_folds)

    keys = list(param_grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*param_grid.values())]
    log.info("Walk-forward: %d folds x %d param combos", n_folds, len(combos))

    windows: list[WalkForwardWindow] = []
    oos_parts: list[pd.Series] = []

    for fold in range(n_folds):
        test_start_i = train_len + fold * test_len
        test_end_i = min(test_start_i + test_len, n)
        if test_end_i - test_start_i < 21:
            break

        train_start_i = 0 if anchored else max(0, test_start_i - train_len)
        train_end_i = max(0, test_start_i - embargo_days)
        if train_end_i - train_start_i < 100:
            continue

        train_px = prices.iloc[train_start_i:train_end_i]
        test_px = prices.iloc[test_start_i:test_end_i]

        # --- select params on TRAIN ONLY ---
        best_sr, best_params = -np.inf, {}
        for combo in combos:
            try:
                strat = strategy_factory(**combo)
                res = engine.run_strategy(strat, train_px)
                sr = M.sharpe_ratio(res.returns)
            except Exception as exc:
                log.debug("combo %s failed on train: %s", combo, exc)
                continue
            if np.isfinite(sr) and sr > best_sr:
                best_sr, best_params = sr, combo

        if not best_params:
            continue

        # --- evaluate on TEST, params frozen ---
        # Prepend the train tail so that the signal's lookback is warmed up, then
        # keep only the test-period returns. Without this, the first months of
        # every fold are contaminated by warm-up NaNs and read as flat.
        warm = min(300, train_end_i - train_start_i)
        ctx = prices.iloc[test_start_i - warm : test_end_i]
        try:
            strat = strategy_factory(**best_params)
            res = engine.run_strategy(strat, ctx)
            test_ret = res.returns.loc[res.returns.index >= test_px.index[0]]
            test_sr = M.sharpe_ratio(test_ret)
        except Exception as exc:
            log.warning("fold %d test failed: %s", fold, exc)
            continue

        windows.append(WalkForwardWindow(
            fold=fold,
            train_start=prices.index[train_start_i], train_end=prices.index[train_end_i - 1],
            test_start=test_px.index[0], test_end=test_px.index[-1],
            best_params=best_params, train_sharpe=best_sr, test_sharpe=test_sr,
            test_returns=test_ret,
        ))
        oos_parts.append(test_ret)

    oos = pd.concat(oos_parts).sort_index() if oos_parts else pd.Series(dtype=float)
    oos = oos[~oos.index.duplicated(keep="first")]
    return WalkForwardResult(windows, oos, embargo_days)


# ---------------------------------------------------------------------------
# Parameter sensitivity
# ---------------------------------------------------------------------------

def parameter_sensitivity(
    prices: pd.DataFrame,
    strategy_factory: Callable[..., Any],
    param_grid: dict[str, Sequence],
    engine: BacktestEngine | None = None,
    metric: str = "sharpe",
) -> pd.DataFrame:
    """Backtest every parameter combination (QA Section C).

    What we want to see is a broad plateau: a whole neighbourhood of parameters
    that all work. What should worry us is a lone spike, one setting that shines
    whilst its immediate neighbours do not. Real edges are not that fussy about
    whether the lookback is 250 days or 255.
    """
    engine = engine or BacktestEngine()
    keys = list(param_grid)
    rows = []
    for values in itertools.product(*param_grid.values()):
        combo = dict(zip(keys, values))
        try:
            res = engine.run_strategy(strategy_factory(**combo), prices)
            stats = M.summarize(res.returns)
            rows.append({**combo, **{k: stats[k] for k in
                                     ("sharpe", "cagr", "max_drawdown", "volatility")},
                         "turnover": res.annual_turnover})
        except Exception as exc:
            log.debug("combo %s failed: %s", combo, exc)
            rows.append({**combo, "sharpe": np.nan, "cagr": np.nan,
                         "max_drawdown": np.nan, "volatility": np.nan, "turnover": np.nan})
    df = pd.DataFrame(rows)
    return df.sort_values(metric, ascending=False).reset_index(drop=True)


def sensitivity_verdict(sweep: pd.DataFrame, metric: str = "sharpe") -> dict:
    """Turn a parameter sweep into a robustness judgement, with the reasoning attached."""
    vals = sweep[metric].dropna()
    if len(vals) < 3:
        return {"verdict": "insufficient", "detail": "Fewer than 3 successful runs."}

    best, median = float(vals.max()), float(vals.median())
    frac_pos = float((vals > 0).mean())
    spread = best - median

    if frac_pos < 0.5:
        verdict, detail = "fragile", (
            f"Only {frac_pos:.0%} of parameter settings are profitable. The winning "
            "configuration looks like a lucky draw rather than a stable effect."
        )
    elif spread > 1.0 and median < 0.3:
        verdict, detail = "fragile", (
            f"Best {metric} {best:.2f} vs median {median:.2f}. A narrow spike with a "
            "weak neighbourhood is the classic signature of curve-fitting."
        )
    elif frac_pos > 0.8 and median > 0.3:
        verdict, detail = "robust", (
            f"{frac_pos:.0%} of settings profitable, median {metric} {median:.2f}. "
            "Broad plateau -- performance does not hinge on one lucky setting."
        )
    else:
        verdict, detail = "mixed", (
            f"{frac_pos:.0%} profitable, median {metric} {median:.2f}, best {best:.2f}. "
            "Some signal, but not a wide plateau."
        )
    return {
        "verdict": verdict, "detail": detail, "best": best, "median": median,
        "frac_profitable": frac_pos, "n_combos": int(len(vals)),
    }


# ---------------------------------------------------------------------------
# Regime analysis
# ---------------------------------------------------------------------------

def split_regimes(benchmark_returns: pd.Series, vol_window: int = 63) -> pd.Series:
    """Label each date bull/bear x calm/volatile from the benchmark.

    Deliberately crude and, more importantly, causal: it uses trailing windows
    only, so the labels are ones we could have known at the time. A regime
    classifier that peeks at the full sample would flatter every strategy it
    touched, which defeats the purpose of splitting by regime in the first place.
    """
    eq = (1 + benchmark_returns).cumprod()
    trend = eq / eq.rolling(126, min_periods=30).mean() - 1
    vol = benchmark_returns.rolling(vol_window, min_periods=20).std()
    vol_med = vol.expanding(min_periods=60).median()

    bull = trend > 0
    calm = vol <= vol_med
    labels = pd.Series("unknown", index=benchmark_returns.index, dtype=object)
    labels[bull & calm] = "bull_calm"
    labels[bull & ~calm] = "bull_volatile"
    labels[~bull & calm] = "bear_calm"
    labels[~bull & ~calm] = "bear_volatile"
    return labels


def regime_performance(returns: pd.Series, regimes: pd.Series) -> pd.DataFrame:
    """Performance broken out by regime (QA Section C)."""
    idx = returns.index.intersection(regimes.index)
    r, g = returns.loc[idx], regimes.loc[idx]
    rows = []
    for label in ["bull_calm", "bull_volatile", "bear_calm", "bear_volatile"]:
        sub = r[g == label]
        if len(sub) < 20:
            continue
        rows.append({
            "regime": label, "n_days": len(sub),
            "pct_of_sample": round(100 * len(sub) / len(r), 1),
            "ann_return": round(M.cagr(sub), 4),
            "sharpe": round(M.sharpe_ratio(sub), 3),
            "max_dd": round(M.max_drawdown(sub), 4),
            "win_rate": round(float((sub > 0).mean()), 3),
        })
    return pd.DataFrame(rows)


def regime_verdict(table: pd.DataFrame) -> dict:
    if table.empty:
        return {"verdict": "insufficient", "detail": "No regime had enough observations."}
    n_pos = int((table["sharpe"] > 0).sum())
    n_tot = len(table)
    worst = table.loc[table["sharpe"].idxmin()]
    if n_pos == n_tot:
        v, d = "robust", f"Positive Sharpe in all {n_tot} regimes tested."
    elif n_pos >= n_tot - 1:
        v, d = "acceptable", (
            f"Positive in {n_pos}/{n_tot} regimes. Weakest: {worst['regime']} "
            f"(Sharpe {worst['sharpe']:.2f}). One weak regime is normal."
        )
    else:
        v, d = "regime_dependent", (
            f"Positive in only {n_pos}/{n_tot} regimes. This strategy is a bet on a "
            f"particular market state, not an all-weather system. Weakest: "
            f"{worst['regime']} (Sharpe {worst['sharpe']:.2f})."
        )
    return {"verdict": v, "detail": d, "n_positive": n_pos, "n_regimes": n_tot}


# ---------------------------------------------------------------------------
# Cost sensitivity (QA Section G)
# ---------------------------------------------------------------------------

def cost_sensitivity(
    prices: pd.DataFrame,
    strategy,
    bps_levels: Sequence[float] = (0, 5, 10, 20, 40),
    config=None,
) -> pd.DataFrame:
    """Re-run at escalating cost levels and locate the break-even.

    This answers Section G directly: if the Sharpe falls to zero somewhere
    between 5 and 10 bps, the strategy works only for someone whose execution is
    better than ours, which is a fact about them and not about the edge.
    """
    from .costs import FixedBpsCost
    from .engine import BacktestConfig

    cfg = config or BacktestConfig()
    rows = []
    for bps in bps_levels:
        eng = BacktestEngine(cfg, FixedBpsCost(commission_bps=bps / 2, spread_bps=bps / 2))
        res = eng.run_strategy(strategy, prices)
        rows.append({
            "one_way_bps": bps,
            "round_trip_bps": 2 * bps,
            "sharpe": round(M.sharpe_ratio(res.returns), 3),
            "cagr": round(M.cagr(res.returns), 4),
            "annual_turnover": round(res.annual_turnover, 2),
            "annual_cost_drag": round(res.cost_drag_annual, 4),
        })
    return pd.DataFrame(rows)


def cost_verdict(table: pd.DataFrame, min_sharpe: float = 0.3) -> dict:
    ok = table[table["sharpe"] >= min_sharpe]
    if ok.empty:
        return {"verdict": "fails_on_costs",
                "detail": f"Sharpe never reaches {min_sharpe} at any cost level, including zero."}
    breakeven = float(ok["one_way_bps"].max())
    realistic = 10.0  # retail large-cap one-way
    if breakeven >= 20:
        v, d = "robust", (f"Holds Sharpe >= {min_sharpe} up to {breakeven:.0f}bps one-way, "
                          "well past realistic retail equity costs.")
    elif breakeven >= realistic:
        v, d = "acceptable", (f"Break-even at {breakeven:.0f}bps one-way -- above the ~10bps "
                              "retail large-cap assumption, but without much margin.")
    else:
        v, d = "cost_fragile", (f"Only survives below {breakeven:.0f}bps one-way. Realistic "
                                "retail equity costs are ~10bps, so this strategy is likely "
                                "unprofitable as executed.")
    return {"verdict": v, "detail": d, "breakeven_bps": breakeven}
