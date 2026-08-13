"""Engine mechanics, cost models, metrics, sizing and risk limits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest import metrics as M
from quantlab.backtest.costs import (CompositeCost, FixedBpsCost, SlippageModel,
                                     SquareRootImpactCost, ZeroCost, get_cost_model)
from quantlab.backtest.engine import BacktestConfig, BacktestEngine
from quantlab.portfolio.risk import (RiskLimits, RiskManager, apply_exposure_limits,
                                     apply_position_limits, drawdown_control)
from quantlab.portfolio.sizing import equal_weight, inverse_volatility, risk_parity
from quantlab.research.strategies import BuyAndHold, get_strategy


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def test_buy_and_hold_matches_equal_weight_return(synthetic_prices):
    """Ground truth: 1/N buy-and-hold with zero costs must track the naive average."""
    engine = BacktestEngine(BacktestConfig(warmup_bars=5, rebalance="M"), ZeroCost())
    res = engine.run_strategy(BuyAndHold(), synthetic_prices)

    naive = synthetic_prices.pct_change(fill_method=None).mean(axis=1).iloc[6:]
    # Monthly rebalancing lets weights drift, so allow a modest divergence.
    assert M.total_return(res.returns) == pytest.approx(M.total_return(naive), rel=0.35)
    assert res.equity.iloc[-1] > 0


def test_costs_strictly_reduce_returns(synthetic_prices):
    strat = get_strategy("xs_momentum")
    cfg = BacktestConfig(warmup_bars=300)

    free = BacktestEngine(cfg, ZeroCost()).run_strategy(strat, synthetic_prices)
    paid = BacktestEngine(cfg, FixedBpsCost(10, 10)).run_strategy(strat, synthetic_prices)

    assert M.total_return(paid.returns) < M.total_return(free.returns)
    assert paid.cost_drag_annual > 0
    assert free.cost_drag_annual == pytest.approx(0.0, abs=1e-12)


def test_higher_costs_monotonically_worse(synthetic_prices):
    strat = get_strategy("mean_reversion")
    cfg = BacktestConfig(warmup_bars=200)
    prev = np.inf
    for bps in (0, 5, 10, 25, 50):
        res = BacktestEngine(cfg, FixedBpsCost(bps / 2, bps / 2)).run_strategy(strat, synthetic_prices)
        cur = M.total_return(res.returns)
        assert cur <= prev + 1e-12, f"Return increased when costs rose to {bps}bps"
        prev = cur


def test_rebalance_frequency_changes_turnover(synthetic_prices):
    strat = get_strategy("xs_momentum")
    turnovers = {}
    for freq in ("D", "W", "M", "Q"):
        eng = BacktestEngine(BacktestConfig(rebalance=freq, warmup_bars=300), ZeroCost())
        turnovers[freq] = eng.run_strategy(strat, synthetic_prices).annual_turnover
    assert turnovers["D"] > turnovers["M"], "Daily rebalancing should trade more than monthly"
    assert turnovers["M"] >= turnovers["Q"], "Monthly should trade at least as much as quarterly"


def test_leverage_cap_enforced(synthetic_prices):
    prices = synthetic_prices
    over = pd.DataFrame(0.5, index=prices.index, columns=prices.columns)  # gross = 2.5
    eng = BacktestEngine(BacktestConfig(max_leverage=1.0, warmup_bars=5), ZeroCost())
    res = eng.run(prices, over)
    assert res.weights.abs().sum(axis=1).max() <= 1.0 + 1e-9


def test_shorts_rejected_when_disallowed(synthetic_prices):
    w = pd.DataFrame(-0.2, index=synthetic_prices.index, columns=synthetic_prices.columns)
    eng = BacktestEngine(BacktestConfig(allow_shorts=False), ZeroCost())
    with pytest.raises(ValueError, match="allow_shorts"):
        eng.run(synthetic_prices, w)


# ---------------------------------------------------------------------------
# Cost models
# ---------------------------------------------------------------------------

def test_fixed_bps_arithmetic():
    model = FixedBpsCost(commission_bps=5, spread_bps=5)
    assert model.one_way_bps == 10
    assert model.round_trip_bps == 20
    turnover = pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2020-01-02"]))
    # Trading 100% of the book one way at 10bps costs exactly 10bps.
    assert model.cost(turnover).iloc[0, 0] == pytest.approx(0.0010)


def test_sqrt_impact_per_unit_grows_as_sqrt():
    """Doubling order size raises the PER-UNIT impact by sqrt(2), not 2.

    Total cost still rises superlinearly (size^1.5) -- that is what capacity
    limits look like. The sublinear part is the price paid per share traded,
    which is the actual claim of the square-root model.
    """
    idx = pd.to_datetime(["2020-01-02"])
    prices = pd.DataFrame({"A": [100.0]}, index=idx)
    volume = pd.DataFrame({"A": [1_000_000.0]}, index=idx)
    model = SquareRootImpactCost(capital=10_000_000)
    base_only = FixedBpsCost(model.commission_bps, model.spread_bps)

    def per_unit_impact(turnover: float) -> float:
        t = pd.DataFrame({"A": [turnover]}, index=idx)
        total = model.cost(t, prices, volume).iloc[0, 0]
        base = base_only.cost(t).iloc[0, 0]
        return (total - base) / turnover

    small = per_unit_impact(0.05)
    large = per_unit_impact(0.10)

    assert large > small, "Per-unit impact did not rise with size"
    assert large / small == pytest.approx(np.sqrt(2), rel=0.02), \
        f"Per-unit impact ratio {large/small:.3f}, expected sqrt(2)={np.sqrt(2):.3f}"


def test_sqrt_impact_total_cost_is_superlinear():
    """Total impact dollars scale ~size^1.5 -- the capacity constraint."""
    idx = pd.to_datetime(["2020-01-02"])
    prices = pd.DataFrame({"A": [100.0]}, index=idx)
    volume = pd.DataFrame({"A": [1_000_000.0]}, index=idx)
    model = SquareRootImpactCost(capital=10_000_000)

    small = model.cost(pd.DataFrame({"A": [0.05]}, index=idx), prices, volume).iloc[0, 0]
    large = model.cost(pd.DataFrame({"A": [0.10]}, index=idx), prices, volume).iloc[0, 0]
    assert large > 2 * small, "Total cost should more than double when size doubles"


def test_sqrt_impact_falls_back_without_volume():
    """With no volume data the model must degrade to the fixed-bps floor."""
    idx = pd.to_datetime(["2020-01-02"])
    t = pd.DataFrame({"A": [0.5]}, index=idx)
    model = SquareRootImpactCost()
    expected = FixedBpsCost(model.commission_bps, model.spread_bps).cost(t).iloc[0, 0]
    assert model.cost(t, None, None).iloc[0, 0] == pytest.approx(expected)


def test_composite_sums_components():
    idx = pd.to_datetime(["2020-01-02"])
    t = pd.DataFrame({"A": [1.0]}, index=idx)
    a, b = FixedBpsCost(5, 5), SlippageModel(2, 0)
    comp = CompositeCost([a, b])
    assert comp.cost(t).iloc[0, 0] == pytest.approx(
        a.cost(t).iloc[0, 0] + b.cost(t).iloc[0, 0])


def test_all_presets_load():
    for name in ("institutional_futures", "large_cap_equity", "retail_equity",
                 "small_cap_equity", "crypto", "zero"):
        assert get_cost_model(name) is not None
    with pytest.raises(KeyError):
        get_cost_model("nonexistent")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_sharpe_of_known_series():
    """Constructed series with mean 0.001 and sd 0.01 -> Sharpe = 0.1*sqrt(252)."""
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 100_000))
    assert M.sharpe_ratio(r) == pytest.approx(0.1 * np.sqrt(252), rel=0.05)


def test_max_drawdown_exact():
    # +100%, then -50% back to start, then flat.
    r = pd.Series([1.0, -0.5, 0.0, 0.0])
    assert M.max_drawdown(r) == pytest.approx(-0.5)


def test_drawdown_duration():
    r = pd.Series([0.1, -0.05, -0.05, -0.05, 0.30])  # 3 periods underwater
    assert M.max_drawdown_duration(r) == 3


def test_cagr_compounds_correctly():
    r = pd.Series([0.0] * 252)
    assert M.cagr(r) == pytest.approx(0.0, abs=1e-12)
    doubling = pd.Series([(2 ** (1 / 252)) - 1] * 252)
    assert M.cagr(doubling) == pytest.approx(1.0, rel=1e-6)


def test_deflated_sharpe_penalises_more_trials():
    """The core of the multiple-testing correction: more trials -> lower confidence."""
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0006, 0.01, 1500))
    dsr_1 = M.deflated_sharpe_ratio(r, n_trials=1)
    dsr_100 = M.deflated_sharpe_ratio(r, n_trials=100)
    dsr_10000 = M.deflated_sharpe_ratio(r, n_trials=10_000)
    assert dsr_1 > dsr_100 > dsr_10000, (dsr_1, dsr_100, dsr_10000)


def test_psr_bounded():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.0005, 0.01, 1000))
    assert 0.0 <= M.probabilistic_sharpe_ratio(r) <= 1.0


def test_zero_variance_returns_do_not_explode():
    """Regression: constant returns must give Sharpe 0, not 3.7e16.

    np.std of a constant array is ~1e-19, not exactly 0.0, so an `sd == 0`
    guard silently passes and the division produces an astronomical Sharpe.
    """
    flat = pd.Series([0.001] * 1000)
    assert M.sharpe_ratio(flat) == 0.0
    assert M.sortino_ratio(flat) == 0.0
    assert M.information_ratio(flat, flat) == 0.0

    zeros = pd.Series([0.0] * 500)
    assert M.sharpe_ratio(zeros) == 0.0
    assert abs(M.max_drawdown(zeros)) < 1e-12


def test_metrics_match_independent_implementation():
    """Cross-check against a from-scratch implementation over random inputs."""
    def indep_mdd(r):
        eq = np.cumprod(1 + np.asarray(r))
        return float((eq / np.maximum.accumulate(eq) - 1).min())

    def indep_cagr(r, ppy=252):
        r = np.asarray(r)
        return float(np.prod(1 + r) ** (ppy / len(r)) - 1)

    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(60, 2000))
        r = pd.Series(rng.normal(rng.uniform(-1e-3, 1e-3), rng.uniform(3e-3, 2e-2), n))
        assert M.max_drawdown(r) == pytest.approx(indep_mdd(r), abs=1e-12)
        assert M.cagr(r) == pytest.approx(indep_cagr(r), rel=1e-9)
        assert M.sharpe_ratio(r) == pytest.approx(
            r.mean() / r.std(ddof=1) * np.sqrt(252), rel=1e-9)


def test_drawdown_plausible_for_stated_volatility():
    """Regression: a -61% drawdown at ~9% annual vol signalled a broken generator.

    Guards the synthetic data source against reintroducing a deterministic
    cycle. Under IID normal returns with these parameters, 3000 Monte Carlo
    paths never produced a drawdown worse than about -50%.
    """
    from quantlab.data.loaders import SyntheticSource, to_wide

    px = to_wide(SyntheticSource(seed=7).fetch(
        [f"S{i}" for i in range(8)], "2012-01-01", "2024-12-31"), "close")
    rets = px.pct_change(fill_method=None).mean(axis=1).dropna()

    vol = M.volatility(rets)
    mdd = M.max_drawdown(rets)
    # A drawdown deeper than ~6x annual vol implies persistence far beyond
    # anything a random walk produces.
    assert mdd > -6 * vol, (
        f"Max drawdown {mdd:.1%} is implausible for {vol:.1%} annual vol -- "
        "the generator has reintroduced deterministic structure")


def test_summarize_has_required_keys(synthetic_prices):
    r = synthetic_prices.pct_change(fill_method=None).mean(axis=1).dropna()
    stats = M.summarize(r, n_trials=10)
    for k in ("cagr", "volatility", "sharpe", "max_drawdown", "max_dd_duration_days",
              "deflated_sharpe", "psr", "skew", "excess_kurtosis"):
        assert k in stats


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------

def test_equal_weight_sums_to_one(synthetic_prices):
    mask = pd.DataFrame(True, index=synthetic_prices.index, columns=synthetic_prices.columns)
    w = equal_weight(mask)
    assert w.sum(axis=1).round(9).eq(1.0).all()
    assert w.iloc[0].iloc[0] == pytest.approx(1 / 5)


def test_inverse_vol_favours_calm_assets(synthetic_prices):
    mask = pd.DataFrame(True, index=synthetic_prices.index, columns=synthetic_prices.columns)
    w = inverse_volatility(mask, synthetic_prices, window=63).iloc[-1]
    vol = synthetic_prices.pct_change(fill_method=None).rolling(63).std().iloc[-1]
    # The lowest-vol asset must get the largest weight.
    assert w.idxmax() == vol.idxmin()
    assert w.sum() == pytest.approx(1.0)


def test_risk_parity_equalises_risk_contribution(synthetic_prices):
    mask = pd.DataFrame(True, index=synthetic_prices.index, columns=synthetic_prices.columns)
    w = risk_parity(mask, synthetic_prices, window=126).iloc[-1]
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (w >= -1e-9).all(), "Risk parity produced a negative weight"

    cov = synthetic_prices.pct_change(fill_method=None).tail(126).cov().to_numpy() * 252
    wv = w.reindex(synthetic_prices.columns).to_numpy()
    port_vol = np.sqrt(wv @ cov @ wv)
    rc = wv * (cov @ wv) / port_vol
    # Contributions should be roughly equal -- allow generous tolerance for noise.
    assert rc.std() / rc.mean() < 0.25, f"Risk contributions not equalised: {rc}"


# ---------------------------------------------------------------------------
# Risk controls
# ---------------------------------------------------------------------------

def test_position_cap_binds(synthetic_prices):
    w = pd.DataFrame(0.0, index=synthetic_prices.index, columns=synthetic_prices.columns)
    w.iloc[:, 0] = 0.9
    w.iloc[:, 1] = 0.1
    capped = apply_position_limits(w, max_position=0.25)
    assert capped.abs().max().max() <= 0.25 + 1e-9


def test_gross_exposure_cap_binds(synthetic_prices):
    w = pd.DataFrame(0.5, index=synthetic_prices.index, columns=synthetic_prices.columns)
    out = apply_exposure_limits(w, max_gross=1.0, max_net=1.0)
    assert out.abs().sum(axis=1).max() <= 1.0 + 1e-9


def test_drawdown_control_is_causal(synthetic_prices):
    """Exposure on date t must depend only on equity through t-1.

    Verified by changing the LAST equity value and confirming no earlier
    exposure decision moves.
    """
    idx = synthetic_prices.index[:300]
    w = pd.DataFrame(1.0, index=idx, columns=["A"])
    eq = pd.Series(np.linspace(100, 60, 300), index=idx)  # steady 40% decline

    out_a = drawdown_control(w, eq, stop_level=0.25, derisk_to=0.5)

    eq_b = eq.copy()
    eq_b.iloc[-1] = 1000.0  # a huge future value
    out_b = drawdown_control(w, eq_b, stop_level=0.25, derisk_to=0.5)

    pd.testing.assert_frame_equal(out_a.iloc[:-1], out_b.iloc[:-1])
    assert (out_a["A"] < 1.0).any(), "Drawdown control never engaged on a 40% decline"


def test_risk_manager_audit_clean(synthetic_prices):
    limits = RiskLimits(max_position=0.30, max_gross_exposure=1.0, max_drawdown_stop=0.0)
    rm = RiskManager(limits)
    raw = pd.DataFrame(0.5, index=synthetic_prices.index, columns=synthetic_prices.columns)
    adj = rm.apply(raw, synthetic_prices)
    audit = rm.audit(adj)
    assert audit["breaches"].sum() == 0, audit
