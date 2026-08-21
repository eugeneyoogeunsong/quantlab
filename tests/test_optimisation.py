# quantlab: Eugene (Yoogeun) Song (https://www.linkedin.com/in/yoogeunsong)
# Independent side project. MIT licensed; see LICENSE.

"""Mean-variance optimisation tests.

The optimiser is checked against analytically-known answers wherever one
exists, and against structural properties (constraints honoured, causality
preserved) everywhere else. A portfolio optimiser that silently violates its
own weight bounds, or that peeks at future returns, is worse than no optimiser
at all -- it produces a confident, backtested, wrong answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.portfolio.optimisation import (
    MeanVarianceOptimiser,
    capm_expected_returns,
    capm_regression,
    efficient_frontier,
    ewma_covariance,
    ewma_drift,
    optimise_portfolio,
)
from quantlab.portfolio.sizing import apply_sizing


@pytest.fixture
def three_assets():
    """Three assets with known, distinct risk/return characteristics."""
    mu = pd.Series({"LOW": 0.05, "MID": 0.08, "HIGH": 0.12})
    # Uncorrelated, ascending volatility: 10%, 15%, 25%
    cov = pd.DataFrame(np.diag([0.10**2, 0.15**2, 0.25**2]),
                       index=mu.index, columns=mu.index)
    return mu, cov


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

def test_weights_sum_to_one(three_assets):
    mu, cov = three_assets
    for objective in ("sharpe", "variance", "return"):
        w = optimise_portfolio(mu, cov, objective=objective, max_weight=1.0)
        assert w.sum() == pytest.approx(1.0, abs=1e-8), objective


def test_weight_bounds_respected(three_assets):
    mu, cov = three_assets
    w = optimise_portfolio(mu, cov, min_weight=0.10, max_weight=0.50)
    assert (w >= 0.10 - 1e-8).all(), w
    assert (w <= 0.50 + 1e-8).all(), w


def test_long_only_by_default(three_assets):
    mu, cov = three_assets
    w = optimise_portfolio(mu, cov, max_weight=1.0)
    assert (w >= -1e-9).all()


def test_infeasible_bounds_rejected(three_assets):
    mu, cov = three_assets
    with pytest.raises(ValueError, match="Infeasible"):
        optimise_portfolio(mu, cov, min_weight=0.5)   # 3 x 0.5 > 1
    with pytest.raises(ValueError, match="Infeasible"):
        optimise_portfolio(mu, cov, max_weight=0.2)   # 3 x 0.2 < 1


def test_unknown_objective_rejected(three_assets):
    mu, cov = three_assets
    with pytest.raises(ValueError, match="objective"):
        optimise_portfolio(mu, cov, objective="maximise_profit")


def test_mismatched_covariance_rejected(three_assets):
    mu, _ = three_assets
    with pytest.raises(ValueError, match="cov_matrix"):
        optimise_portfolio(mu, np.eye(2))


# ---------------------------------------------------------------------------
# Known analytic answers
# ---------------------------------------------------------------------------

def test_min_variance_of_uncorrelated_assets_is_inverse_variance():
    """For uncorrelated assets, min-variance weights are proportional to 1/variance.

    An exact closed form, so this is a real check rather than a plausibility one.
    """
    vols = np.array([0.10, 0.20, 0.40])
    mu = pd.Series([0.05, 0.05, 0.05], index=["A", "B", "C"])
    cov = pd.DataFrame(np.diag(vols**2), index=mu.index, columns=mu.index)

    w = optimise_portfolio(mu, cov, objective="variance", max_weight=1.0)
    expected = (1 / vols**2) / (1 / vols**2).sum()
    assert np.allclose(w.to_numpy(), expected, atol=1e-4), f"{w.to_numpy()} vs {expected}"


def test_identical_assets_get_equal_weight():
    """With identical inputs there is no reason to prefer any asset."""
    mu = pd.Series([0.08] * 4, index=list("ABCD"))
    cov = pd.DataFrame(np.eye(4) * 0.04, index=mu.index, columns=mu.index)
    w = optimise_portfolio(mu, cov, objective="variance", max_weight=1.0)
    assert np.allclose(w.to_numpy(), 0.25, atol=1e-4)


def test_max_return_objective_concentrates(three_assets):
    """Maximising return alone should load the highest-return asset to its cap."""
    mu, cov = three_assets
    w = optimise_portfolio(mu, cov, objective="return", max_weight=0.60)
    assert w["HIGH"] == pytest.approx(0.60, abs=1e-6)


def test_min_variance_prefers_the_calm_asset(three_assets):
    mu, cov = three_assets
    w = optimise_portfolio(mu, cov, objective="variance", max_weight=1.0)
    assert w["LOW"] > w["MID"] > w["HIGH"]


def test_sharpe_objective_beats_equal_weight_in_sample():
    """The optimiser must at least achieve its own objective on its own inputs.

    This is only an in-sample statement -- it says the solver works, NOT that
    the portfolio will do well out of sample. Those are very different claims,
    and conflating them is the central failure mode of mean-variance investing.
    """
    mu = pd.Series({"A": 0.10, "B": 0.06, "C": 0.08})
    cov = pd.DataFrame(np.diag([0.15**2, 0.10**2, 0.12**2]),
                       index=mu.index, columns=mu.index)
    rf = 0.02

    w = optimise_portfolio(mu, cov, risk_free=rf, objective="sharpe", max_weight=1.0)
    eq = np.full(3, 1 / 3)

    def sharpe(weights):
        weights = np.asarray(weights)
        return (weights @ mu - rf) / np.sqrt(weights @ cov.to_numpy() @ weights)

    assert sharpe(w.to_numpy()) >= sharpe(eq) - 1e-9


def test_efficient_frontier_is_upward_sloping(three_assets):
    """Higher target return must cost more volatility -- that is the frontier."""
    mu, cov = three_assets
    frontier = efficient_frontier(mu, cov, n_points=12, max_weight=1.0)
    assert len(frontier) >= 5
    vols = frontier["volatility"].to_numpy()
    assert vols[-1] > vols[0], "frontier not upward sloping"
    assert vols.min() == pytest.approx(vols[0], rel=0.05), "leftmost point is not min-variance"


# ---------------------------------------------------------------------------
# Input estimation
# ---------------------------------------------------------------------------

def test_ewma_covariance_shape_and_symmetry(synthetic_prices):
    rets = synthetic_prices.pct_change(fill_method=None).dropna()
    cov = ewma_covariance(rets, alpha=0.06)
    assert cov.shape == (rets.shape[1], rets.shape[1])
    assert np.allclose(cov.to_numpy(), cov.to_numpy().T), "covariance must be symmetric"
    assert (np.diag(cov.to_numpy()) > 0).all(), "variances must be positive"


def test_ewma_covariance_is_positive_semidefinite(synthetic_prices):
    rets = synthetic_prices.pct_change(fill_method=None).dropna()
    cov = ewma_covariance(rets, alpha=0.06).to_numpy()
    assert np.linalg.eigvalsh(cov).min() > -1e-10


def test_ewma_weights_recent_data_more_heavily():
    """A regime shift at the end should move an EWMA estimate far more than a mean."""
    n = 500
    idx = pd.bdate_range("2020-01-01", periods=n)
    rets = pd.DataFrame({"A": np.concatenate([np.full(n - 50, 0.0001),
                                              np.full(50, 0.01)])}, index=idx)
    ewma = float(ewma_drift(rets, alpha=0.10, annualise=False).iloc[0])
    simple = float(rets["A"].mean())
    assert ewma > simple * 3, f"EWMA {ewma:.5f} not tracking the recent regime vs {simple:.5f}"


def test_ewma_alpha_validated(synthetic_prices):
    rets = synthetic_prices.pct_change(fill_method=None).dropna()
    with pytest.raises(ValueError, match="alpha"):
        ewma_covariance(rets, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        ewma_covariance(rets, alpha=1.5)


def test_capm_regression_recovers_known_beta():
    """Construct an asset with beta exactly 1.5 and check the regression finds it."""
    rng = np.random.default_rng(0)
    n = 2000
    idx = pd.bdate_range("2015-01-01", periods=n)
    market = pd.Series(rng.normal(0.0003, 0.01, n), index=idx)
    asset = pd.Series(1.5 * market.to_numpy() + rng.normal(0, 0.002, n), index=idx)

    reg = capm_regression(asset, market, alpha_ewma=None)  # OLS
    assert reg["beta"] == pytest.approx(1.5, abs=0.05)
    assert abs(reg["alpha"]) < 0.05
    assert reg["n_obs"] == n


def test_capm_regression_handles_short_series():
    idx = pd.bdate_range("2020-01-01", periods=10)
    s = pd.Series(np.random.default_rng(1).normal(0, 0.01, 10), index=idx)
    reg = capm_regression(s, s)
    assert reg["beta"] == 1.0  # falls back rather than fitting 10 points


def test_capm_expected_returns_increase_with_beta():
    """Higher beta must imply higher expected return when the premium is positive."""
    rng = np.random.default_rng(2)
    n = 1500
    idx = pd.bdate_range("2015-01-01", periods=n)
    market = pd.Series(rng.normal(0.0004, 0.01, n), index=idx)
    prices = pd.DataFrame({
        "LOWBETA": 0.5 * market.to_numpy() + rng.normal(0, 0.001, n),
        "HIGHBETA": 2.0 * market.to_numpy() + rng.normal(0, 0.001, n),
    }, index=idx)

    er = capm_expected_returns(prices, market, risk_free=0.02, market_premium=0.06)
    assert er["HIGHBETA"] > er["LOWBETA"]


# ---------------------------------------------------------------------------
# Backtest integration -- the causality checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("frac", [0.55, 0.70, 0.83])
def test_optimiser_weights_are_causal(synthetic_prices, frac):
    """Weights must not change when future prices are withheld.

    The same truncation test used on every feature in the library. A rolling
    optimiser is an easy place to leak the future -- one `.tail()` on the wrong
    side of a slice does it.

    THE ONE PERMITTED EXCEPTION, and why it is not leakage
    ------------------------------------------------------
    The rebalance calendar is "last available session in each period". Truncate
    the data mid-quarter and that final day becomes the last available session
    of its (now partial) quarter, so the truncated run rebalances there while
    the full run does not.

    That is a boundary artifact, not a look-ahead: the truncated run's extra
    rebalance uses only data at or before that date. It is also harmless in a
    backtest, since no returns follow the final bar, and desirable live, where
    you do want to act on the most recent close.

    The test asserts the difference is confined to exactly that final row. Any
    divergence earlier in the series would be real leakage, and is what this
    test exists to catch -- so it deliberately does not just drop the last row
    and look away.
    """
    opt = MeanVarianceOptimiser(lookback=126, rebalance="QE", objective="variance")
    full = opt.generate_weights(synthetic_prices)
    cut = int(len(synthetic_prices) * frac)
    part = opt.generate_weights(synthetic_prices.iloc[:cut])

    common = full.index[:cut]
    divergence = (full.loc[common] - part.loc[common]).abs().sum(axis=1)
    offending = divergence[divergence > 1e-9]

    if len(offending):
        positions = [common.get_loc(d) for d in offending.index]
        assert positions == [cut - 1], (
            f"Weights diverged at positions {positions}; only the final row "
            f"({cut - 1}) is explainable as a calendar boundary. Anything "
            "earlier means the optimiser is reading the future."
        )

    # And everything strictly before the boundary must match exactly.
    pd.testing.assert_frame_equal(
        full.loc[common].iloc[:-1], part.loc[common].iloc[:-1],
        check_exact=False, atol=1e-9)


def test_optimiser_weights_valid(synthetic_prices):
    opt = MeanVarianceOptimiser(lookback=126, rebalance="QE", max_weight=0.5)
    w = opt.generate_weights(synthetic_prices)

    assert w.shape[1] == synthetic_prices.shape[1]
    assert not w.isna().any().any()
    assert (w >= -1e-9).all().all()
    active = w[w.sum(axis=1) > 1e-9]
    assert np.allclose(active.sum(axis=1), 1.0, atol=1e-6)
    assert w.max().max() <= 0.5 + 1e-6


def test_optimiser_holds_weights_between_rebalances(synthetic_prices):
    """Quarterly rebalancing must not produce daily weight churn."""
    opt = MeanVarianceOptimiser(lookback=126, rebalance="QE")
    w = opt.generate_weights(synthetic_prices)
    changes = (w.diff().abs().sum(axis=1) > 1e-9).sum()
    # ~5.5 years of synthetic data -> roughly 20-25 quarterly rebalances.
    assert changes < 40, f"{changes} weight changes is far more than quarterly"


def test_apply_sizing_dispatches_to_optimiser(synthetic_prices):
    mask = pd.DataFrame(True, index=synthetic_prices.index,
                        columns=synthetic_prices.columns)
    for method in ("mean_variance", "min_variance"):
        w = apply_sizing(mask, synthetic_prices, method=method, lookback=126)
        assert w.shape == synthetic_prices.shape
        assert not w.isna().any().any()


def test_apply_sizing_rejects_unknown_method(synthetic_prices):
    mask = pd.DataFrame(True, index=synthetic_prices.index,
                        columns=synthetic_prices.columns)
    with pytest.raises(KeyError, match="Unknown sizing method"):
        apply_sizing(mask, synthetic_prices, method="crystal_ball")


def test_optimiser_respects_signal_mask(synthetic_prices):
    """Names the strategy did not select must get zero weight."""
    mask = pd.DataFrame(False, index=synthetic_prices.index,
                        columns=synthetic_prices.columns)
    mask.iloc[:, :2] = True   # only the first two assets are eligible
    w = apply_sizing(mask, synthetic_prices, method="min_variance", lookback=126)
    assert (w.iloc[:, 2:].to_numpy() == 0).all(), "weight assigned to an unselected name"


def test_singular_covariance_does_not_crash():
    """Perfectly collinear assets make the covariance singular. Must not blow up."""
    mu = pd.Series({"A": 0.08, "B": 0.08})
    cov = pd.DataFrame([[0.04, 0.04], [0.04, 0.04]], index=mu.index, columns=mu.index)
    w = optimise_portfolio(mu, cov, objective="variance", max_weight=1.0)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert not w.isna().any()
