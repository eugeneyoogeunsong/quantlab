# quantlab: Eugene (Yoogeun) Song (https://www.linkedin.com/in/yoogeunsong)
# Independent side project. MIT licensed; see LICENSE.

"""The five layers, wired end to end.

Data -> Research -> Backtest -> Portfolio/Risk -> Execution/Ops

This module is where the blueprint becomes one callable object. ``Pipeline.run()``
executes each layer in order and returns a ``PipelineResult`` carrying both the equity
curve and the QA report. The coupling is deliberate: reading the performance without
also reading whether it is trustworthy should take extra effort, not less.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .backtest import metrics as M
from .backtest import validation as V
from .backtest.costs import get_cost_model
from .backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from .data.loaders import SyntheticSource, YFinanceSource, load_prices, to_wide
from .data.universe import Universe, get_universe
from .data.validate import clean_prices, run_data_checks
from .portfolio.risk import RiskLimits, RiskManager
from .qa import checklist as QA
from .research.strategies import Strategy, get_strategy

log = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Every decision that moves the output, in one place: if it is not here, it is not a knob."""

    # Layer 1
    universe: str = "sector_etfs"
    start: str = "2010-01-01"
    end: str = "2024-12-31"
    cache_dir: str = "./.cache/prices"
    data_source: str = "yfinance"   # 'yfinance' (network) | 'synthetic' (offline, seeded)
    synthetic_seed: int = 42

    # Layer 2
    strategy: str = "xs_momentum"
    strategy_params: dict[str, Any] = field(default_factory=dict)

    # Layer 3
    cost_preset: str = "large_cap_equity"
    rebalance: str = "M"
    execution_lag: int = 1
    execution_price: str = "open"
    initial_capital: float = 1_000_000.0
    warmup_bars: int = 252

    # Layer 4
    sizing: str = "equal_weight"
    max_position: float = 0.35
    max_gross_exposure: float = 1.0
    vol_target: float | None = None
    max_drawdown_stop: float = 0.25

    # Validation
    benchmark_symbol: str = "SPY"
    n_trials: int = 1
    run_walk_forward: bool = True
    run_param_sweep: bool = True
    run_cost_sensitivity: bool = True
    param_grid: dict[str, Sequence] | None = None


@dataclass
class PipelineResult:
    result: BacktestResult
    stats: dict
    qa: QA.QAReport
    benchmark_returns: pd.Series | None = None
    walk_forward: Any = None
    param_sweep: pd.DataFrame | None = None
    param_verdict: dict | None = None
    regime_table: pd.DataFrame | None = None
    regime_verdict: dict | None = None
    cost_table: pd.DataFrame | None = None
    cost_verdict: dict | None = None
    prices: pd.DataFrame | None = None
    config: PipelineConfig | None = None

    @property
    def tradeable(self) -> bool:
        """True when every blocking QA check passed: a floor to clear, not a recommendation."""
        return self.qa.passed


class Pipeline:
    """Runs the five layers in order and keeps what each one produced."""

    def __init__(self, config: PipelineConfig | None = None):
        self.cfg = config or PipelineConfig()

    # -- Layer 1 -----------------------------------------------------------

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, Universe, list]:
        cfg = self.cfg
        uni = get_universe(cfg.universe)

        if cfg.data_source == "synthetic":
            source = SyntheticSource(seed=cfg.synthetic_seed)
        elif cfg.data_source == "yfinance":
            source = YFinanceSource()
        else:
            raise KeyError(f"Unknown data_source {cfg.data_source!r}")

        symbols = list(uni.symbols)
        if cfg.benchmark_symbol and cfg.benchmark_symbol not in symbols:
            symbols.append(cfg.benchmark_symbol)

        log.info("Layer 1: loading %d symbols %s..%s", len(symbols), cfg.start, cfg.end)
        raw = load_prices(symbols, cfg.start, cfg.end, source=source, cache_dir=cfg.cache_dir)

        close = to_wide(raw, "close")
        volume = to_wide(raw, "volume")

        bench_px = close[cfg.benchmark_symbol].copy() if cfg.benchmark_symbol in close else None
        px = close[[c for c in uni.symbols if c in close.columns]]
        px = clean_prices(px)
        vol = volume[[c for c in px.columns if c in volume.columns]]

        strat_tmp = get_strategy(cfg.strategy, **cfg.strategy_params)
        checks = run_data_checks(
            px,
            source_name=source.name,
            auto_adjust=source.auto_adjust,
            source_survivorship_safe=source.survivorship_safe,
            universe_is_pit=uni.is_point_in_time,
            lookback=strat_tmp.min_history,
        )
        self._bench_px = bench_px
        return px, vol, uni, checks

    # -- Layers 2-4 --------------------------------------------------------

    def build_weights(self, strategy: Strategy, prices: pd.DataFrame,
                      risk: RiskManager) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Signal, then sizing, then risk limits; returns (raw, risk-adjusted) weights."""
        raw = strategy.generate_weights(prices)

        if self.cfg.sizing != "equal_weight":
            from .portfolio.sizing import apply_sizing
            mask = raw > 0
            raw = apply_sizing(mask, prices, self.cfg.sizing)

        # The first pass has no equity curve, so drawdown control is skipped:
        # drawdowns are computed from an equity curve, and the equity curve
        # is computed from weights. One extra iteration resolves the circle.
        adjusted = risk.apply(raw, prices, equity=None)
        return raw, adjusted

    # -- Main --------------------------------------------------------------

    def run(self, verbose: bool = True) -> PipelineResult:
        cfg = self.cfg
        qa = QA.QAReport()

        # ---- Layer 1 ----
        prices, volume, uni, data_checks = self.load_data()
        qa.add(*data_checks)
        if verbose:
            log.info("Layer 1 done: %s, %d symbols, %d sessions",
                     uni.name, prices.shape[1], prices.shape[0])

        # ---- Layer 2 ----
        strategy = get_strategy(cfg.strategy, **cfg.strategy_params)

        # ---- Layer 3 + 4 ----
        bt_cfg = BacktestConfig(
            initial_capital=cfg.initial_capital,
            rebalance=cfg.rebalance,
            execution_lag=cfg.execution_lag,
            execution_price=cfg.execution_price,
            max_leverage=cfg.max_gross_exposure,
            warmup_bars=min(cfg.warmup_bars, max(1, len(prices) // 4)),
        )
        cost_model = get_cost_model(cfg.cost_preset)
        engine = BacktestEngine(bt_cfg, cost_model)

        limits = RiskLimits(
            max_position=cfg.max_position,
            max_gross_exposure=cfg.max_gross_exposure,
            max_net_exposure=cfg.max_gross_exposure,
            vol_target=cfg.vol_target,
            max_drawdown_stop=cfg.max_drawdown_stop,
        )
        risk = RiskManager(limits)

        raw_w, adj_w = self.build_weights(strategy, prices, risk)
        result = engine.run(prices, adj_w, volume=volume, strategy_name=strategy.name)

        # Second pass: with an equity curve in hand, drawdown control can bind.
        if cfg.max_drawdown_stop:
            adj_w2 = risk.apply(raw_w, prices, equity=result.equity)
            result = engine.run(prices, adj_w2, volume=volume, strategy_name=strategy.name)
            adj_w = adj_w2

        # ---- Benchmark ----
        bench_ret = None
        if self._bench_px is not None:
            bench_ret = self._bench_px.pct_change(fill_method=None).reindex(result.returns.index)
            bench_ret = bench_ret.fillna(0.0)

        stats = M.summarize(result.returns, bench_ret, n_trials=cfg.n_trials)

        # ---- Validation ----
        wf = param_sweep = param_verdict = None
        regime_tbl = regime_verdict = cost_tbl = cost_verdict = None

        grid = cfg.param_grid or _default_grid(cfg.strategy)

        if cfg.run_param_sweep and grid:
            def factory(**kw):
                return get_strategy(cfg.strategy, **kw)
            param_sweep = V.parameter_sensitivity(prices, factory, grid, engine)
            param_verdict = V.sensitivity_verdict(param_sweep)
            # Honest trial count: every combination we evaluated, not the one we kept.
            stats["n_trials_assumed"] = max(cfg.n_trials, len(param_sweep))
            stats["deflated_sharpe"] = M.deflated_sharpe_ratio(
                result.returns, stats["n_trials_assumed"])

        if cfg.run_walk_forward and grid and len(prices) >= 500:
            def factory(**kw):
                return get_strategy(cfg.strategy, **kw)
            try:
                wf = V.walk_forward(prices, factory, grid, engine, n_folds=5)
            except Exception as exc:
                log.warning("walk-forward failed: %s", exc)

        if bench_ret is not None and len(bench_ret) > 200:
            regimes = V.split_regimes(bench_ret)
            regime_tbl = V.regime_performance(result.returns, regimes)
            regime_verdict = V.regime_verdict(regime_tbl)

        if cfg.run_cost_sensitivity:
            try:
                cost_tbl = V.cost_sensitivity(prices, strategy, config=bt_cfg)
                cost_verdict = V.cost_verdict(cost_tbl)
            except Exception as exc:
                log.warning("cost sensitivity failed: %s", exc)

        # ---- QA: sections B through G ----
        asset_rets = prices.pct_change(fill_method=None)
        oos_check = QA.check_oos_reported(wf)

        qa.add(
            QA.check_execution_lag(bt_cfg),
            QA.check_weight_causality(result.weights, asset_rets),
            QA.check_no_future_features(strategy, prices),
            QA.check_rebalance_timing(bt_cfg),
            oos_check,
            QA.check_parameter_sensitivity(param_verdict),
            QA.check_regimes(regime_verdict),
            QA.check_benchmark(stats),
            QA.check_multiple_testing(stats),
            QA.check_costs_included(getattr(cost_model, "name", "?"), result),
            QA.check_turnover_sanity(result),
            QA.check_execution_specified(bt_cfg),
            QA.check_risk_limits(limits, risk.audit(adj_w)),
            QA.check_rebalance_justified(bt_cfg, result),
            QA.check_reporting_complete(stats),
            QA.check_sample_adequacy(stats),
            QA.check_robustness_triad(cost_verdict, oos_check, param_verdict),
        )
        qa.context = {
            "universe": uni.describe(),
            "strategy": strategy.describe(),
            "config": cfg.__dict__,
        }

        return PipelineResult(
            result=result, stats=stats, qa=qa, benchmark_returns=bench_ret,
            walk_forward=wf, param_sweep=param_sweep, param_verdict=param_verdict,
            regime_table=regime_tbl, regime_verdict=regime_verdict,
            cost_table=cost_tbl, cost_verdict=cost_verdict,
            prices=prices, config=cfg,
        )


def _default_grid(strategy_name: str) -> dict[str, Sequence]:
    """Default sweep ranges per strategy, kept modest on purpose.

    Every extra combination inflates the multiple-testing penalty, and the deflated
    Sharpe will charge us for it: correctly, since a grid searched is a set of trials
    run whether or not we choose to report them.
    """
    return {
        "xs_momentum": {"lookback": [126, 189, 252], "skip": [0, 21], "top_n": [2, 3, 4]},
        "ts_momentum": {"lookback": [126, 189, 252, 315]},
        "low_vol": {"vol_window": [63, 126, 252], "top_n": [2, 3, 4]},
        "mean_reversion": {"zscore_window": [10, 21, 42], "top_n": [2, 3]},
        "dual_momentum": {"lookback": [126, 189, 252], "top_n": [1, 2, 3]},
        "buy_and_hold": {},
    }.get(strategy_name, {})
