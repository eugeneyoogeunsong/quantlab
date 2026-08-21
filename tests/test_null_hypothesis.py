# quantlab: Eugene (Yoogeun) Song (https://www.linkedin.com/in/yoogeunsong)
# Independent side project. MIT licensed; see LICENSE.

"""Null-hypothesis tests: strategies must find NOTHING in pure noise.

This is the strongest single check in the suite. Look-ahead bias can hide from
code review and from truncation tests, but it cannot hide here. If a strategy
earns a reliably positive Sharpe on data generated with no exploitable
structure, the profit is coming from the framework, not the market.

Analogy: a metal detector that beeps in an empty field. You do not need to know
which component is faulty to know it cannot be trusted on a real beach.

These are slower than the rest of the suite (many simulated paths), so they are
marked `slow` and can be skipped with `-m "not slow"`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest import metrics as M
from quantlab.backtest.costs import FixedBpsCost, ZeroCost
from quantlab.backtest.engine import BacktestConfig, BacktestEngine
from quantlab.research.strategies import STRATEGY_REGISTRY, get_strategy

N_TRIALS = 40
N_BARS = 1800
N_ASSETS = 8
ANNUAL_VOL = 0.18


def _pure_noise_prices(seed: int) -> pd.DataFrame:
    """Zero-drift geometric random walk. No trend, no cross-sectional signal.

    Every asset has identical parameters, so there is nothing for a ranking
    strategy to rank on beyond chance.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=N_BARS)
    return pd.DataFrame(
        {f"S{i}": 100 * np.exp(np.cumsum(rng.normal(0.0, ANNUAL_VOL / np.sqrt(252), N_BARS)))
         for i in range(N_ASSETS)},
        index=idx,
    )


def _sharpes_on_noise(strategy_name: str, n_trials: int = N_TRIALS) -> np.ndarray:
    engine = BacktestEngine(BacktestConfig(warmup_bars=300, rebalance="M"), ZeroCost())
    out = []
    for t in range(n_trials):
        px = _pure_noise_prices(2000 + t)
        res = engine.run_strategy(get_strategy(strategy_name), px)
        out.append(M.sharpe_ratio(res.returns))
    return np.array(out)


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(STRATEGY_REGISTRY))
def test_no_strategy_beats_noise(name):
    """Mean Sharpe on signal-free data must be statistically indistinguishable from ~0.

    The tolerance is not exactly zero. Log-drift of zero implies a small
    POSITIVE arithmetic drift (Jensen's inequality: E[exp(X)] > exp(E[X])),
    worth roughly sigma^2/2 = 1.6%/yr here. Diversified across 8 assets the
    portfolio vol is ~6.4%, so a long-only book should show a Sharpe near 0.25
    from that alone. The bar is set at 0.5 -- comfortably above the Jensen
    effect, far below anything leakage would produce (perfect foresight scores
    above 5).
    """
    sharpes = _sharpes_on_noise(name)
    mean = sharpes.mean()
    stderr = sharpes.std(ddof=1) / np.sqrt(len(sharpes))

    assert mean < 0.5, (
        f"{name} scored mean Sharpe {mean:.3f} (SE {stderr:.3f}) on pure noise. "
        "There is no signal in this data, so this is the framework leaking, "
        "not the strategy working."
    )


@pytest.mark.slow
def test_active_strategies_do_not_beat_passive_on_noise():
    """On noise, active trading can only dilute and add costs -- never add alpha."""
    passive = _sharpes_on_noise("buy_and_hold", n_trials=25).mean()
    for name in ("xs_momentum", "ts_momentum", "mean_reversion", "dual_momentum"):
        active = _sharpes_on_noise(name, n_trials=25).mean()
        assert active <= passive + 0.35, (
            f"{name} ({active:.3f}) beat passive buy-and-hold ({passive:.3f}) on "
            "signal-free data by more than sampling error explains."
        )


@pytest.mark.slow
def test_costs_make_noise_strategies_lose():
    """With realistic costs, trading noise must be strictly value-destroying.

    This is QA Section G in miniature: turnover with no edge is a guaranteed
    transfer to the broker.
    """
    free = BacktestEngine(BacktestConfig(warmup_bars=300, rebalance="M"), ZeroCost())
    paid = BacktestEngine(BacktestConfig(warmup_bars=300, rebalance="M"), FixedBpsCost(10, 10))

    deltas = []
    for t in range(15):
        px = _pure_noise_prices(3000 + t)
        strat = get_strategy("mean_reversion")  # highest turnover
        deltas.append(M.cagr(paid.run_strategy(strat, px).returns)
                      - M.cagr(free.run_strategy(strat, px).returns))
    assert np.mean(deltas) < 0, "Costs did not reduce returns on average"


@pytest.mark.slow
def test_deflated_sharpe_rejects_best_of_many_noise_runs():
    """The multiple-testing correction must catch a cherry-picked noise result.

    Procedure: run 40 strategies on pure noise, keep the single best, and ask
    the deflated Sharpe whether it is impressed. It should not be -- that is the
    entire purpose of the statistic, and the exact scenario Bailey & Lopez de
    Prado designed it for.
    """
    sharpes = _sharpes_on_noise("xs_momentum", n_trials=40)
    best_idx = int(np.argmax(sharpes))

    engine = BacktestEngine(BacktestConfig(warmup_bars=300, rebalance="M"), ZeroCost())
    best_returns = engine.run_strategy(
        get_strategy("xs_momentum"), _pure_noise_prices(2000 + best_idx)).returns

    naive_psr = M.probabilistic_sharpe_ratio(best_returns)
    dsr = M.deflated_sharpe_ratio(best_returns, n_trials=40)

    assert dsr < naive_psr, "Deflation did not penalise the selection at all"
    assert dsr < 0.95, (
        f"Deflated Sharpe {dsr:.3f} still endorses the best of 40 pure-noise runs "
        f"(naive PSR was {naive_psr:.3f}). The correction is too weak."
    )
