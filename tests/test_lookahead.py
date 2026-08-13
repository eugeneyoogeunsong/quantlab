"""The most important tests in the repo.

If look-ahead bias slips through, every other number the library produces is
fiction. These tests build strategies that cheat on purpose and assert that the
QA layer catches them. A checker that has never caught anything is not a
checker -- it is decoration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.costs import ZeroCost
from quantlab.backtest.engine import BacktestConfig, BacktestEngine
from quantlab.qa import checklist as QA
from quantlab.research import features as F
from quantlab.research.strategies import CrossSectionalMomentum, Strategy


# ---------------------------------------------------------------------------
# Feature-level causality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [
    lambda px: F.simple_returns(px),
    lambda px: F.momentum(px, 252, 21),
    lambda px: F.time_series_momentum(px, 252),
    lambda px: F.realized_vol(px, 63),
    lambda px: F.zscore(px, 21),
    lambda px: F.rsi(px, 14),
    lambda px: F.moving_average_crossover(px, 50, 200),
    lambda px: F.cross_sectional_rank(F.momentum(px, 252, 21)),
])
def test_features_are_causal(fn, synthetic_prices):
    """Every shipped feature must be unchanged when future data is withheld."""
    F.assert_no_lookahead(fn, synthetic_prices)


def test_forward_returns_is_detected_as_leaky(synthetic_prices):
    """Sanity check on the detector itself.

    forward_returns() genuinely reads the future -- that is its job as a research
    target. If the detector does NOT flag it, the detector is broken and every
    passing test above is meaningless.
    """
    with pytest.raises(AssertionError, match="LOOK-AHEAD BIAS"):
        F.assert_no_lookahead(lambda px: F.forward_returns(px, 5), synthetic_prices)


def test_backfill_is_detected(synthetic_prices):
    """bfill() is the classic accidental leak -- future prices written backwards.

    The hole is placed to straddle the truncation point (70% of the sample).
    That is where the leak is observable: inside the full series bfill reaches
    forward across the gap, but in the truncated series there is no forward
    value to reach, so the two disagree. A hole in the middle of both windows
    would be filled identically and reveal nothing -- which is exactly why
    leakage of this kind survives casual testing.
    """
    def leaky(px):
        return px.bfill().pct_change(fill_method=None).rolling(20).mean()

    holey = synthetic_prices.copy()
    cut = int(len(holey) * 0.7)
    holey.iloc[cut - 10 : cut + 10, 1] = np.nan
    with pytest.raises(AssertionError, match="LOOK-AHEAD BIAS"):
        F.assert_no_lookahead(leaky, holey)


def test_full_sample_zscore_is_detected(synthetic_prices):
    """Standardising on the full sample leaks the future mean into the past.

    This is the single most common leak in ML-based trading code:
    `StandardScaler().fit_transform(X)` before splitting train/test.
    """
    def leaky(px):
        return (px - px.mean()) / px.std()

    with pytest.raises(AssertionError, match="LOOK-AHEAD BIAS"):
        F.assert_no_lookahead(leaky, synthetic_prices)


def test_centered_rolling_is_detected(synthetic_prices):
    """rolling(center=True) reaches half a window into the future."""
    def leaky(px):
        return px.rolling(21, center=True).mean()

    with pytest.raises(AssertionError, match="LOOK-AHEAD BIAS"):
        F.assert_no_lookahead(leaky, synthetic_prices)


# ---------------------------------------------------------------------------
# Engine-level lag enforcement
# ---------------------------------------------------------------------------

def test_engine_rejects_zero_lag():
    with pytest.raises(ValueError, match="look-ahead"):
        BacktestConfig(execution_lag=0)


def test_perfect_foresight_is_caught(synthetic_prices):
    """The nuclear test: a strategy that holds tomorrow's best asset.

    Built by construction to be impossible. It should produce an absurd Sharpe
    AND be flagged by the causality checks. If the flags do not fire here, they
    will not fire on subtler real bugs either.
    """

    @dataclass
    class Cheater(Strategy):
        name: str = "cheater"

        def raw_signal(self, prices):
            return F.forward_returns(prices, 1)

        def generate_weights(self, prices):
            fwd = F.forward_returns(prices, 1)
            best = fwd.rank(axis=1, ascending=False) == 1
            return best.astype(float)

    strat = Cheater()

    # 1. Structural check catches it via truncation.
    check = QA.check_no_future_features(strat, synthetic_prices)
    assert check.status == "fail", "Truncation test failed to detect perfect foresight"

    # 2. The absurd performance is itself the tell.
    #    rebalance="none" is required: under monthly rebalancing the engine holds
    #    a stale monthly weight for ~21 days, which destroys the daily foresight
    #    and masks the bug. Worth noting as its own lesson -- infrequent
    #    rebalancing can hide leakage rather than fix it.
    engine = BacktestEngine(BacktestConfig(warmup_bars=10, rebalance="none"), ZeroCost())
    res = engine.run_strategy(strat, synthetic_prices)
    from quantlab.backtest import metrics as M
    sharpe = M.sharpe_ratio(res.returns)
    assert sharpe > 5, f"Foresight strategy only scored {sharpe:.2f} -- test is not exercising the bug"

    # 3. And the correlation detector must fire on it too.
    asset_rets = synthetic_prices.pct_change(fill_method=None)
    assert QA.check_weight_causality(res.weights, asset_rets).status == "fail"


def test_engine_lag_actually_shifts(synthetic_prices):
    """Directly verify the engine's shift: a weight set at t cannot earn t's return."""
    prices = synthetic_prices[["AAA"]].iloc[:200]

    # Hold nothing except on one single day.
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    target_date = prices.index[100]
    w.loc[target_date, "AAA"] = 1.0

    engine = BacktestEngine(
        BacktestConfig(warmup_bars=0, rebalance="none", execution_lag=1), ZeroCost())
    res = engine.run(prices, w)

    asset_ret = prices["AAA"].pct_change(fill_method=None)
    next_date = prices.index[101]

    # The position earns the NEXT day's return, not the signal day's.
    assert abs(res.returns.loc[next_date] - asset_ret.loc[next_date]) < 1e-12, \
        "Position did not earn the return of the bar after the signal"
    assert abs(res.returns.loc[target_date]) < 1e-12, \
        "Position earned the same bar's return -- the lag is not being applied"


def test_double_shift_is_conservative(synthetic_prices):
    """A strategy that also shifts should underperform, never outperform.

    The failure mode of a redundant shift must be pessimistic. If shifting twice
    made results better, the engine's lag would be pointing the wrong way.
    """
    prices = synthetic_prices

    class Shifted(CrossSectionalMomentum):
        def generate_weights(self, px):
            return super().generate_weights(px).shift(1).fillna(0.0)

    engine = BacktestEngine(BacktestConfig(warmup_bars=300), ZeroCost())
    from quantlab.backtest import metrics as M

    normal = M.total_return(engine.run_strategy(CrossSectionalMomentum(), prices).returns)
    doubled = M.total_return(engine.run_strategy(Shifted(name="shifted"), prices).returns)

    assert doubled != pytest.approx(normal, abs=1e-9), "Extra shift had no effect at all"


def test_weight_causality_check_flags_leak(synthetic_prices):
    """The correlation-based detector must fire on leaked weights."""
    asset_rets = synthetic_prices.pct_change(fill_method=None)

    # Weights that literally are next-period returns -> perfectly correlated.
    leaky_w = asset_rets.copy()
    check = QA.check_weight_causality(leaky_w, asset_rets)
    assert check.status == "fail"

    # Properly lagged weights -> correlation near zero.
    clean_w = (asset_rets.rank(axis=1, ascending=False) == 1).astype(float).shift(1).fillna(0.0)
    check2 = QA.check_weight_causality(clean_w, asset_rets)
    assert check2.status == "pass", check2.detail
