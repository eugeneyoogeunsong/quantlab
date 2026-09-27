# quantlab: Eugene (Yoogeun) Song. MIT licensed; see LICENSE.
"""Mean-variance and shrinkage-estimator tests.

Checked against closed-form solutions where they exist (uncorrelated
min-variance, identical-asset symmetry), against the statistical properties
the estimators are defined by (shrinkage intensity falls with sample size;
shrunk matrices are better conditioned), and against causality under
truncation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.portfolio.meanvariance import (
    CAPMImplied,
    EWMACovariance,
    EWMAMean,
    JamesStein,
    LedoitWolf,
    MaxSharpe,
    MaxUtility,
    MeanVariance,
    MinVariance,
    RollingMeanVariance,
    SampleCovariance,
    SampleMean,
    TargetReturn,
    efficient_frontier,
)
from quantlab.portfolio.sizing import apply_sizing


@pytest.fixture
def three_assets():
    mu = pd.Series({"LOW": 0.05, "MID": 0.08, "HIGH": 0.12})
    cov = pd.DataFrame(np.diag([0.10**2, 0.15**2, 0.25**2]), index=mu.index, columns=mu.index)
    return mu, cov


def _block_returns(n: int, p: int = 8, seed: int = 0) -> pd.DataFrame:
    """Two correlated blocks -- a population the constant-correlation target
    does NOT describe, so shrinkage should back off as n grows."""
    rng = np.random.default_rng(seed)
    C = np.full((p, p), -0.1)
    h = p // 2
    C[:h, :h] = 0.7
    C[h:, h:] = 0.7
    np.fill_diagonal(C, 1.0)
    sd = np.linspace(0.10, 0.30, p) / np.sqrt(252)
    X = rng.multivariate_normal(np.zeros(p), np.outer(sd, sd) * C, n)
    return pd.DataFrame(X, columns=[f"A{i}" for i in range(p)])


# ===========================================================================
# Covariance estimators
# ===========================================================================

def test_sample_covariance_is_symmetric_psd(synthetic_prices):
    r = synthetic_prices.pct_change(fill_method=None).dropna()
    S = SampleCovariance().fit(r).to_numpy()
    assert np.allclose(S, S.T)
    assert np.linalg.eigvalsh(S).min() > -1e-12


def test_ewma_halflife_semantics():
    """A shock halflife days ago should carry exactly half the weight of today's."""
    n = 400
    r = pd.DataFrame({"A": np.zeros(n)})
    r.iloc[-1, 0] = 0.01           # today
    today = EWMACovariance(halflife=20, annualise=False).fit(r).iloc[0, 0]
    r2 = pd.DataFrame({"A": np.zeros(n)})
    r2.iloc[-21, 0] = 0.01         # 20 days ago
    ago = EWMACovariance(halflife=20, annualise=False).fit(r2).iloc[0, 0]
    assert ago / today == pytest.approx(0.5, rel=0.05)


def test_ewma_rejects_bad_halflife(synthetic_prices):
    r = synthetic_prices.pct_change(fill_method=None).dropna()
    with pytest.raises(ValueError):
        EWMACovariance(halflife=0).fit(r)


def test_ledoit_wolf_intensity_falls_with_sample_size():
    """On a misspecified target, more data should mean less shrinkage."""
    deltas = [LedoitWolf().fit(_block_returns(n)).attrs["shrinkage"] for n in (40, 120, 500, 4000)]
    assert all(0.0 <= d <= 1.0 for d in deltas)
    assert deltas[0] > deltas[-1] * 5, deltas
    assert deltas[-1] < 0.02


def test_ledoit_wolf_shrinks_fully_when_target_is_exactly_right():
    """If the population IS constant-correlation, the target is the truth."""
    rng = np.random.default_rng(3)
    p = 6
    C = np.full((p, p), 0.4)
    np.fill_diagonal(C, 1.0)
    sd = np.full(p, 0.2 / np.sqrt(252))
    X = rng.multivariate_normal(np.zeros(p), np.outer(sd, sd) * C, 300)
    delta = LedoitWolf().fit(pd.DataFrame(X)).attrs["shrinkage"]
    assert delta > 0.8


def test_ledoit_wolf_improves_conditioning_when_p_near_n():
    """The regime shrinkage exists for: as many assets as observations."""
    r = _block_returns(n=30, p=25, seed=7)
    S = SampleCovariance().fit(r).to_numpy()
    LW = LedoitWolf().fit(r).to_numpy()
    cond = lambda M: np.linalg.cond(M)
    assert cond(LW) < cond(S) / 3
    assert np.linalg.eigvalsh(LW).min() > 0


def test_ledoit_wolf_preserves_variances():
    """The constant-correlation target keeps the diagonal; so must the blend."""
    r = _block_returns(200)
    S, LW = SampleCovariance().fit(r), LedoitWolf().fit(r)
    # Sample uses ddof=1, LW's internal S uses MLE; compare up to that factor.
    assert np.allclose(np.diag(LW), np.diag(S) * (len(r) - 1) / len(r), rtol=1e-10)


# ===========================================================================
# Return estimators
# ===========================================================================

def test_capm_recovers_beta_structure():
    rng = np.random.default_rng(2)
    n = 1500
    mkt = pd.Series(rng.normal(0.0004, 0.01, n))
    r = pd.DataFrame({"LO": 0.5 * mkt + rng.normal(0, 0.002, n),
                      "HI": 2.0 * mkt + rng.normal(0, 0.002, n)})
    er = CAPMImplied(risk_free=0.02, market_premium=0.06).fit(r, market=mkt)
    assert er["LO"] == pytest.approx(0.02 + 0.5 * 0.06, abs=0.01)
    assert er["HI"] == pytest.approx(0.02 + 2.0 * 0.06, abs=0.02)


def test_capm_requires_market():
    with pytest.raises(ValueError, match="market"):
        CAPMImplied().fit(pd.DataFrame({"A": [0.0] * 50}))


def test_james_stein_shrinks_toward_grand_mean():
    rng = np.random.default_rng(5)
    r = pd.DataFrame(rng.normal([0.0002, 0.0004, 0.0006, 0.0008], 0.01, (60, 4)), columns=list("ABCD"))
    raw, js = SampleMean().fit(r), JamesStein().fit(r)
    assert 0.0 < js.attrs["shrinkage"] <= 1.0
    assert js.std() <= raw.std()
    assert js.mean() == pytest.approx(raw.mean(), rel=1e-9)   # shrinking preserves the grand mean


def test_james_stein_intensity_falls_with_evidence():
    """Strongly different means with lots of data should barely be shrunk."""
    rng = np.random.default_rng(6)
    weak = pd.DataFrame(rng.normal([0.0001, 0.0002, 0.0003, 0.0004], 0.02, (40, 4)))
    strong = pd.DataFrame(rng.normal([0.0, 0.002, 0.004, 0.006], 0.005, (2000, 4)))
    assert JamesStein().fit(weak).attrs["shrinkage"] > JamesStein().fit(strong).attrs["shrinkage"]


def test_ewma_mean_tracks_regime_shift():
    n = 500
    r = pd.DataFrame({"A": np.r_[np.full(n - 50, 0.0001), np.full(50, 0.01)]})
    assert EWMAMean(halflife=10, annualise=False).fit(r).iloc[0] > 3 * SampleMean(annualise=False).fit(r).iloc[0]


# ===========================================================================
# Optimiser -- closed forms and constraints
# ===========================================================================

def test_min_variance_uncorrelated_is_inverse_variance():
    vols = np.array([0.10, 0.20, 0.40])
    mu = pd.Series(0.05, index=list("ABC"))
    cov = pd.DataFrame(np.diag(vols**2), index=mu.index, columns=mu.index)
    w = MeanVariance(MinVariance(), lower=0, upper=1).solve(mu, cov)
    expected = (1 / vols**2) / (1 / vols**2).sum()
    assert np.allclose(w, expected, atol=1e-5)


def test_identical_assets_get_equal_weight():
    mu = pd.Series(0.08, index=list("ABCD"))
    cov = pd.DataFrame(np.eye(4) * 0.04, index=mu.index, columns=mu.index)
    for obj in (MinVariance(), MaxSharpe(), MaxUtility(3.0)):
        w = MeanVariance(obj, lower=0, upper=1).solve(mu, cov)
        assert np.allclose(w, 0.25, atol=1e-4), obj.name


def test_weights_sum_to_one_and_respect_box(three_assets):
    mu, cov = three_assets
    for obj in (MinVariance(), MaxSharpe(), MaxUtility(2.0), TargetReturn(0.08)):
        w = MeanVariance(obj, lower=0.1, upper=0.5).solve(mu, cov)
        assert w.sum() == pytest.approx(1.0, abs=1e-8)
        assert (w >= 0.1 - 1e-8).all() and (w <= 0.5 + 1e-8).all(), obj.name


def test_infeasible_box_rejected(three_assets):
    mu, cov = three_assets
    with pytest.raises(ValueError):
        MeanVariance(MinVariance(), lower=0.5).solve(mu, cov)
    with pytest.raises(ValueError):
        MeanVariance(MinVariance(), upper=0.2).solve(mu, cov)


def test_max_utility_traces_the_frontier(three_assets):
    """Higher risk aversion -> lower volatility, monotonically."""
    mu, cov = three_assets
    vols = []
    for gamma in (0.5, 2.0, 8.0, 32.0):
        w = MeanVariance(MaxUtility(gamma), lower=0, upper=1).solve(mu, cov)
        vols.append(float(np.sqrt(w @ cov @ w)))
    assert all(b <= a + 1e-9 for a, b in zip(vols, vols[1:])), vols


def test_target_return_hits_its_target(three_assets):
    mu, cov = three_assets
    w = MeanVariance(TargetReturn(0.09), lower=0, upper=1).solve(mu, cov)
    assert float(w @ mu) == pytest.approx(0.09, abs=1e-6)


def test_max_sharpe_achieves_objective_in_sample(three_assets):
    """The solver must at least beat 1/N on its own inputs. In-sample only --
    this says nothing about out-of-sample performance, and conflating the two
    is the central error of mean-variance investing."""
    mu, cov = three_assets
    rf = 0.02
    w = MeanVariance(MaxSharpe(rf), lower=0, upper=1).solve(mu, cov)
    sharpe = lambda x: (x @ mu - rf) / np.sqrt(x @ cov @ x)
    assert sharpe(w.to_numpy()) >= sharpe(np.full(3, 1 / 3)) - 1e-9


def test_efficient_frontier_is_upward_sloping(three_assets):
    mu, cov = three_assets
    f = efficient_frontier(mu, cov, n_points=12, upper=1.0)
    assert len(f) >= 6
    assert f["volatility"].iloc[-1] > f["volatility"].iloc[0]
    assert f["volatility"].is_monotonic_increasing


def test_singular_covariance_does_not_crash():
    mu = pd.Series({"A": 0.08, "B": 0.08})
    cov = pd.DataFrame([[0.04, 0.04], [0.04, 0.04]], index=mu.index, columns=mu.index)
    w = MeanVariance(MinVariance(), lower=0, upper=1).solve(mu, cov)
    assert w.sum() == pytest.approx(1.0) and not w.isna().any()


# ===========================================================================
# Rolling adapter -- causality
# ===========================================================================

@pytest.mark.parametrize("frac", [0.55, 0.70, 0.83])
def test_rolling_optimiser_is_causal(synthetic_prices, frac):
    """Truncating the future may change only the final row (a calendar
    boundary, see the class docstring). Anything earlier is a leak."""
    roll = RollingMeanVariance(MeanVariance(MinVariance()), lookback=126, rebalance="QE")
    full = roll.generate_weights(synthetic_prices)
    cut = int(len(synthetic_prices) * frac)
    part = roll.generate_weights(synthetic_prices.iloc[:cut])
    common = full.index[:cut]
    diff = (full.loc[common] - part.loc[common]).abs().sum(axis=1)
    bad = [common.get_loc(d) for d in diff[diff > 1e-9].index]
    assert bad in ([], [cut - 1]), f"divergence at {bad}; only {cut-1} is a boundary"


def test_rolling_weights_are_valid(synthetic_prices):
    roll = RollingMeanVariance(MeanVariance(MaxSharpe(), upper=0.5), lookback=126)
    w = roll.generate_weights(synthetic_prices)
    assert not w.isna().any().any()
    assert (w >= -1e-9).all().all() and w.max().max() <= 0.5 + 1e-6
    active = w[w.sum(axis=1) > 1e-9]
    assert np.allclose(active.sum(axis=1), 1.0, atol=1e-6)


def test_rolling_holds_between_rebalances(synthetic_prices):
    w = RollingMeanVariance(lookback=126, rebalance="QE").generate_weights(synthetic_prices)
    assert (w.diff().abs().sum(axis=1) > 1e-9).sum() < 40


def test_sizing_dispatch(synthetic_prices):
    mask = pd.DataFrame(True, index=synthetic_prices.index, columns=synthetic_prices.columns)
    for method in ("mean_variance", "min_variance", "max_utility"):
        w = apply_sizing(mask, synthetic_prices, method=method, lookback=126)
        assert w.shape == synthetic_prices.shape and not w.isna().any().any()


def test_sizing_respects_selection_mask(synthetic_prices):
    mask = pd.DataFrame(False, index=synthetic_prices.index, columns=synthetic_prices.columns)
    mask.iloc[:, :2] = True
    w = apply_sizing(mask, synthetic_prices, method="min_variance", lookback=126)
    assert (w.iloc[:, 2:].to_numpy() == 0).all()
