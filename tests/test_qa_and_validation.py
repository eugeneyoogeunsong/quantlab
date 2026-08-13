"""QA checklist behaviour, validation machinery, strategies, and execution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest import metrics as M
from quantlab.backtest import validation as V
from quantlab.backtest.costs import FixedBpsCost, ZeroCost
from quantlab.backtest.engine import BacktestConfig, BacktestEngine
from quantlab.data.universe import Universe, get_universe
from quantlab.data.validate import (check_bad_ticks, check_missing_dates,
                                    check_monotonic_unique, check_survivorship,
                                    clean_prices, run_data_checks)
from quantlab.execution.broker import PaperBroker, generate_orders
from quantlab.execution.monitor import StrategyMonitor
from quantlab.qa import checklist as QA
from quantlab.research.strategies import STRATEGY_REGISTRY, get_strategy


# ---------------------------------------------------------------------------
# Section A -- data integrity
# ---------------------------------------------------------------------------

def test_gap_detection(gappy_prices):
    check = check_missing_dates(gappy_prices)
    assert check.status == "warn"
    assert "gaps" in check.detail


def test_bad_tick_detection(gappy_prices):
    check = check_bad_ticks(gappy_prices, threshold=0.5)
    assert check.status == "warn"
    assert check.evidence["n_outliers"] >= 1


def test_unsorted_index_fails(synthetic_prices):
    shuffled = synthetic_prices.sample(frac=1.0, random_state=0)
    assert check_monotonic_unique(shuffled).status == "fail"


def test_survivorship_always_flagged_for_static_universe():
    """A static list of today's names can never quietly pass."""
    assert check_survivorship(False, False).status == "warn"
    assert check_survivorship(True, True).status == "pass"


def test_clean_prices_never_backfills():
    idx = pd.bdate_range("2020-01-01", periods=10)
    px = pd.DataFrame({"A": [np.nan, np.nan, 100.0, 101, np.nan, 102, 103, 104, 105, 106]}, index=idx)
    out = clean_prices(px)
    # Leading NaNs must remain NaN. Filling them would invent history.
    assert out["A"].iloc[:2].isna().all()
    assert out["A"].iloc[4] == 101.0  # forward-filled from the previous value


def test_universe_requires_rationale():
    with pytest.raises(ValueError, match="rationale"):
        Universe(name="x", symbols=["AAA"], rationale="   ")


def test_point_in_time_membership_masks_correctly():
    idx = pd.bdate_range("2020-01-01", periods=5)
    membership = pd.DataFrame({"AAA": [True] * 5, "BBB": [False, False, True, True, True]},
                              index=idx)
    u = Universe("t", ["AAA", "BBB"], "test", membership=membership)
    assert u.is_point_in_time
    assert u.active_on(idx[0]) == ["AAA"]
    assert set(u.active_on(idx[3])) == {"AAA", "BBB"}


def test_run_data_checks_returns_all_sections(synthetic_prices):
    checks = run_data_checks(synthetic_prices, source_name="test", lookback=252)
    assert len(checks) == 7
    assert all(c.section == "A" for c in checks)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(STRATEGY_REGISTRY))
def test_every_strategy_produces_valid_weights(name, synthetic_prices):
    w = get_strategy(name).generate_weights(synthetic_prices)
    assert w.shape[1] == synthetic_prices.shape[1]
    assert not w.isna().any().any(), "Strategy emitted NaN weights"
    assert (w >= -1e-9).all().all(), "Long-only strategy emitted a negative weight"
    assert w.sum(axis=1).max() <= 1.0 + 1e-9, "Weights sum above 100%"


def test_momentum_picks_the_uptrend(trending_prices):
    """A basic sanity check: momentum must prefer UP over DOWN."""
    w = get_strategy("xs_momentum", top_n=1).generate_weights(trending_prices)
    late = w.iloc[300:]
    active = late[late.sum(axis=1) > 0]
    assert active["UP"].mean() > active["DOWN"].mean(), \
        "Momentum favoured the downtrend over the uptrend"


def test_ts_momentum_goes_to_cash_in_downtrend(trending_prices):
    """Absolute momentum must hold nothing when everything is falling."""
    only_down = trending_prices[["DOWN"]]
    w = get_strategy("ts_momentum").generate_weights(only_down)
    assert w.iloc[300:].sum(axis=1).mean() < 0.5, \
        "Time-series momentum stayed invested through a sustained downtrend"


def test_dual_momentum_gate_is_stricter(trending_prices):
    """Dual momentum holds at most what cross-sectional momentum holds."""
    xs = get_strategy("xs_momentum", top_n=2).generate_weights(trending_prices)
    dual = get_strategy("dual_momentum", top_n=2).generate_weights(trending_prices)
    assert dual.sum(axis=1).sum() <= xs.sum(axis=1).sum() + 1e-9


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_walk_forward_windows_do_not_overlap(synthetic_prices):
    def factory(**kw):
        return get_strategy("xs_momentum", **kw)

    wf = V.walk_forward(synthetic_prices, factory,
                        {"lookback": [126, 252], "top_n": [2, 3]},
                        BacktestEngine(BacktestConfig(warmup_bars=100), ZeroCost()),
                        n_folds=3)
    assert len(wf.windows) >= 2
    for a, b in zip(wf.windows, wf.windows[1:]):
        assert a.test_end <= b.test_start, "Walk-forward test windows overlap"


def test_walk_forward_embargo_gap_exists(synthetic_prices):
    def factory(**kw):
        return get_strategy("xs_momentum", **kw)

    wf = V.walk_forward(synthetic_prices, factory, {"lookback": [126, 252]},
                        BacktestEngine(BacktestConfig(warmup_bars=100), ZeroCost()),
                        n_folds=3, embargo_days=10)
    for w in wf.windows:
        assert w.train_end < w.test_start, "No embargo gap between train and test"


def test_parameter_sweep_covers_grid(synthetic_prices):
    def factory(**kw):
        return get_strategy("xs_momentum", **kw)

    grid = {"lookback": [126, 252], "top_n": [2, 3]}
    sweep = V.parameter_sensitivity(synthetic_prices, factory, grid,
                                    BacktestEngine(BacktestConfig(warmup_bars=300), ZeroCost()))
    assert len(sweep) == 4
    assert {"lookback", "top_n", "sharpe", "cagr"}.issubset(sweep.columns)


def test_sensitivity_verdict_flags_fragility():
    fragile = pd.DataFrame({"sharpe": [2.5, -0.3, -0.4, -0.2, -0.5, -0.1]})
    assert V.sensitivity_verdict(fragile)["verdict"] == "fragile"

    robust = pd.DataFrame({"sharpe": [0.9, 0.85, 0.8, 0.78, 0.75, 0.7]})
    assert V.sensitivity_verdict(robust)["verdict"] == "robust"


def test_regime_labels_are_causal(synthetic_prices):
    """Regime labels must not change when future data is appended."""
    bench = synthetic_prices.mean(axis=1).pct_change(fill_method=None).dropna()
    full = V.split_regimes(bench)
    part = V.split_regimes(bench.iloc[:1000])
    pd.testing.assert_series_equal(full.iloc[:1000], part, check_names=False)


def test_cost_sensitivity_declines(synthetic_prices):
    tbl = V.cost_sensitivity(synthetic_prices, get_strategy("xs_momentum"),
                             bps_levels=(0, 10, 25, 50),
                             config=BacktestConfig(warmup_bars=300))
    assert tbl["sharpe"].is_monotonic_decreasing or tbl["sharpe"].iloc[0] >= tbl["sharpe"].iloc[-1]
    assert tbl["annual_cost_drag"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# QA report
# ---------------------------------------------------------------------------

def test_zero_cost_model_fails_qa(synthetic_prices):
    eng = BacktestEngine(BacktestConfig(warmup_bars=300), ZeroCost())
    res = eng.run_strategy(get_strategy("xs_momentum"), synthetic_prices)
    assert QA.check_costs_included("zero", res).status == "fail"


def test_missing_benchmark_fails_qa():
    assert QA.check_benchmark({"cagr": 0.1}).status == "fail"


def test_missing_walk_forward_fails_qa():
    assert QA.check_oos_reported(None).status == "fail"


def test_qa_gate_raises_on_failure():
    report = QA.QAReport()
    report.add(QA.check_benchmark({"cagr": 0.1}))  # a guaranteed failure
    assert not report.passed
    with pytest.raises(AssertionError, match="QA GATE FAILED"):
        report.gate()


def test_qa_gate_strict_mode_rejects_warnings():
    from quantlab.data.validate import Check
    report = QA.QAReport()
    report.add(Check("A", "test", "warn", "a warning"))
    report.gate(allow_warnings=True)  # tolerated
    with pytest.raises(AssertionError, match="strict mode"):
        report.gate(allow_warnings=False)


def test_robustness_triad_fails_when_any_leg_fails():
    from quantlab.data.validate import Check
    oos_pass = Check("C", "Out-of-sample", "pass", "ok")
    bad_costs = {"verdict": "cost_fragile", "detail": "dies at 5bps"}
    good_params = {"verdict": "robust", "detail": "wide plateau"}
    result = QA.check_robustness_triad(bad_costs, oos_pass, good_params)
    assert result.status == "fail"
    assert "costs" in result.detail


def test_deflated_sharpe_check_fails_on_low_dsr():
    assert QA.check_multiple_testing({"deflated_sharpe": 0.2, "n_trials_assumed": 500}).status == "fail"
    assert QA.check_multiple_testing({"deflated_sharpe": 0.97, "n_trials_assumed": 10}).status == "pass"


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def test_order_generation_respects_threshold():
    prices = pd.Series({"AAA": 100.0, "BBB": 50.0})
    target = pd.Series({"AAA": 0.5, "BBB": 0.5})
    # Already at target -> no orders.
    current = {"AAA": 500.0, "BBB": 1000.0}
    assert generate_orders(target, current, prices, 100_000) == []

    # Well away from target -> orders appear.
    orders = generate_orders(target, {}, prices, 100_000)
    assert len(orders) == 2
    assert all(o.side == "buy" for o in orders)
    assert sum(o.notional for o in orders) == pytest.approx(100_000, rel=0.01)


def test_paper_broker_slippage_hurts_both_ways():
    from quantlab.execution.broker import Order
    b = PaperBroker(cash=100_000, slippage_bps=10, commission_bps=0)
    b.mark({"AAA": 100.0})

    buy = b.submit(Order("AAA", "buy", 100))
    assert buy.price > 100.0, "Buy did not fill above the mark"

    sell = b.submit(Order("AAA", "sell", 100))
    assert sell.price < 100.0, "Sell did not fill below the mark"


def test_paper_broker_rejects_overspend():
    from quantlab.execution.broker import Order
    b = PaperBroker(cash=1_000)
    b.mark({"AAA": 100.0})
    assert b.submit(Order("AAA", "buy", 1_000)) is None


def test_monitor_flags_worse_than_backtest_drawdown():
    rng = np.random.default_rng(3)
    baseline = pd.Series(rng.normal(0.0005, 0.008, 1500))
    monitor = StrategyMonitor(baseline)

    catastrophe = pd.Series([-0.02] * 100)
    alerts = monitor.check(catastrophe)
    assert any(a.level == "alert" and a.metric == "drawdown" for a in alerts), \
        [str(a) for a in alerts]


def test_monitor_quiet_when_healthy():
    rng = np.random.default_rng(4)
    baseline = pd.Series(rng.normal(0.0005, 0.008, 1500))
    live = pd.Series(rng.normal(0.0005, 0.008, 300))
    alerts = StrategyMonitor(baseline).check(live)
    assert all(a.level != "alert" for a in alerts), [str(a) for a in alerts]
