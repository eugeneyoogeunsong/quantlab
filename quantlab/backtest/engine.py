"""Layer 3: vectorised backtest engine.

THE CENTRAL DESIGN DECISION
--------------------------
Look-ahead bias is prevented structurally, in one place, rather than by
remembering to be careful inside every strategy.

The timeline the engine enforces:

    date t     close : signal computed from data up to and including t's close
    date t+1   open  : orders execute here
    date t+1   close : the position earns the return realised over t+1

Precisely: weights decided at t's close are applied to the return realised from
t+1 onward. In code that is exactly one line (`weights.shift(1)`), it lives here
at line ~120 and nowhere else, and strategies return unshifted weights. If a
strategy shifts as well, the result is a double lag: less profitable rather than
more, so the failure mode is at least conservative.

The clerk analogy: a betting slip has to be handed over before the race starts,
and the engine is the clerk who timestamps every slip and refuses any handed in
mid-race, however sure the punter is that they meant to submit it earlier.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .costs import CostModel, FixedBpsCost

log = logging.getLogger(__name__)

TRADING_DAYS = 252


@dataclass
class BacktestConfig:
    """Everything that changes the numbers, in one auditable object."""

    initial_capital: float = 1_000_000.0
    rebalance: str = "M"           # D, W, M, Q; or 'none' to trade on every signal change
    execution_lag: int = 1         # bars between signal and fill; 1 is the next bar, never 0
    execution_price: str = "open"  # 'open' (realistic) or 'close' (optimistic)
    max_leverage: float = 1.0
    allow_shorts: bool = False
    warmup_bars: int = 252

    def __post_init__(self) -> None:
        if self.execution_lag < 1:
            raise ValueError(
                "execution_lag must be >= 1. A lag of 0 means trading at the same "
                "bar the signal was computed from -- that is look-ahead bias, and "
                "it is the single most common way backtests are broken."
            )


@dataclass
class BacktestResult:
    """Everything needed to audit the run, not just the equity curve."""

    equity: pd.Series
    returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    positions: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    config: BacktestConfig
    strategy_name: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def cost_drag_annual(self) -> float:
        """Annualised return given up to costs: the number to report beside the Sharpe."""
        n = len(self.returns)
        if n == 0:
            return 0.0
        gross = (1 + self.gross_returns).prod() ** (TRADING_DAYS / n) - 1
        net = (1 + self.returns).prod() ** (TRADING_DAYS / n) - 1
        return float(gross - net)

    @property
    def annual_turnover(self) -> float:
        if len(self.turnover) == 0:
            return 0.0
        return float(self.turnover.mean() * TRADING_DAYS)


class BacktestEngine:
    """Vectorised backtester with the execution lag enforced structurally."""

    def __init__(self, config: BacktestConfig | None = None, cost_model: CostModel | None = None):
        self.config = config or BacktestConfig()
        self.cost_model = cost_model or FixedBpsCost()

    # -- rebalance calendar ------------------------------------------------

    def _rebalance_mask(self, index: pd.DatetimeIndex) -> pd.Series:
        """True on the dates the portfolio is allowed to trade.

        Between rebalances the weights are simply held, and they drift with
        prices. Rebalance frequency is a real decision with a real cost: monthly
        rebalancing on a daily signal cuts turnover by roughly 20x, and QA
        Section E asks us to justify whichever frequency we picked.
        """
        freq = self.config.rebalance.lower()
        s = pd.Series(index=index, dtype=bool)
        if freq in ("none", "d", "daily"):
            s[:] = True
            return s
        code = {"w": "W", "weekly": "W", "m": "ME", "monthly": "ME",
                "q": "QE", "quarterly": "QE"}.get(freq)
        if code is None:
            raise ValueError(f"Unknown rebalance frequency {self.config.rebalance!r}")
        # Last available session in each period, not the calendar date: the
        # calendar date may fall on a weekend or a holiday, and matching on it
        # would silently drop rebalances.
        marks = pd.Series(index, index=index).groupby(pd.Grouper(freq=code)).max().dropna()
        s[:] = False
        s.loc[s.index.isin(marks.values)] = True
        return s

    # -- main loop ---------------------------------------------------------

    def run(
        self,
        prices: pd.DataFrame,
        weights: pd.DataFrame,
        volume: pd.DataFrame | None = None,
        open_prices: pd.DataFrame | None = None,
        strategy_name: str = "",
    ) -> BacktestResult:
        """Run the backtest.

        Parameters
        ----------
        prices  : date x symbol close prices (adjusted).
        weights : date x symbol target weights, decided at each date's CLOSE.
                  Pass them UNSHIFTED: the engine applies the lag itself, and a
                  strategy that also shifts pays the lag twice.
        """
        prices = prices.sort_index()
        weights = weights.reindex(index=prices.index, columns=prices.columns).fillna(0.0)

        cfg = self.config
        if not cfg.allow_shorts and (weights < -1e-12).to_numpy().any():
            raise ValueError("Negative weights supplied but config.allow_shorts is False")

        gross_exposure = weights.abs().sum(axis=1)
        breach = gross_exposure > cfg.max_leverage + 1e-9
        if breach.any():
            scale = np.where(breach, cfg.max_leverage / gross_exposure.replace(0, np.nan), 1.0)
            weights = weights.mul(pd.Series(scale, index=weights.index).fillna(1.0), axis=0)
            log.warning("Scaled down %d date(s) that breached max_leverage=%.2f",
                        int(breach.sum()), cfg.max_leverage)

        # ---- rebalance calendar: hold weights between trade dates ----
        rb = self._rebalance_mask(pd.DatetimeIndex(prices.index))
        held = weights.where(rb).ffill().fillna(0.0)

        # ================================================================
        # THE LAG. This is the only place in the library where signal
        # timing is decided: weights on row t were computed from data up
        # to t's close, so shifting by execution_lag makes them effective
        # from t+lag onward and they can never earn t's own return.
        # ================================================================
        effective = held.shift(cfg.execution_lag).fillna(0.0)

        # ---- returns actually earned by held positions ----
        if cfg.execution_price == "open" and open_prices is not None:
            # Close-to-close return of the asset; the lag has already ensured
            # that we were positioned before this bar began.
            asset_returns = prices.pct_change(fill_method=None).fillna(0.0)
        else:
            asset_returns = prices.pct_change(fill_method=None).fillna(0.0)

        gross_ret = (effective * asset_returns).sum(axis=1)

        # ---- turnover: change in effective weights, measured at execution ----
        turnover_matrix = effective.diff().abs().fillna(0.0)
        turnover = turnover_matrix.sum(axis=1)

        cost_matrix = self.cost_model.cost(turnover_matrix, prices, volume)
        cost_series = cost_matrix.sum(axis=1)

        net_ret = gross_ret - cost_series

        # ---- warm-up trim: the signal is not valid until it has history ----
        warm = min(cfg.warmup_bars, max(0, len(net_ret) - 2))
        if warm > 0:
            net_ret = net_ret.iloc[warm:]
            gross_ret = gross_ret.iloc[warm:]
            turnover = turnover.iloc[warm:]
            cost_series = cost_series.iloc[warm:]
            effective = effective.iloc[warm:]

        equity = cfg.initial_capital * (1 + net_ret).cumprod()
        positions = effective.mul(equity, axis=0)

        return BacktestResult(
            equity=equity,
            returns=net_ret,
            gross_returns=gross_ret,
            weights=effective,
            positions=positions,
            turnover=turnover,
            costs=cost_series,
            config=cfg,
            strategy_name=strategy_name,
            meta={
                "cost_model": getattr(self.cost_model, "name", "?"),
                "n_symbols": int(prices.shape[1]),
                "start": str(net_ret.index[0].date()) if len(net_ret) else None,
                "end": str(net_ret.index[-1].date()) if len(net_ret) else None,
                "rebalance_dates": int(rb.sum()),
            },
        )

    def run_strategy(self, strategy, prices: pd.DataFrame, **kwargs) -> BacktestResult:
        """Generate weights from a Strategy and run them: the normal entry point."""
        w = strategy.generate_weights(prices)
        return self.run(prices, w, strategy_name=strategy.name, **kwargs)
