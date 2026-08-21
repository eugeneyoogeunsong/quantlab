# quantlab: Eugene (Yoogeun) Song (https://www.linkedin.com/in/yoogeunsong)
# Independent side project. MIT licensed; see LICENSE.

"""Shared fixtures. All synthetic -- tests must not depend on a network call."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def synthetic_prices():
    """5 assets, ~8 years of daily GBM prices with differing drift and vol."""
    rng = np.random.default_rng(42)
    n, syms = 2000, ["AAA", "BBB", "CCC", "DDD", "EEE"]
    idx = pd.bdate_range("2015-01-01", periods=n)
    drifts = [0.12, 0.08, 0.05, 0.00, -0.03]
    vols = [0.18, 0.22, 0.15, 0.30, 0.25]
    data = {}
    for s, mu, sd in zip(syms, drifts, vols):
        rets = rng.normal(mu / 252, sd / np.sqrt(252), n)
        data[s] = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame(data, index=idx)


@pytest.fixture(scope="session")
def trending_prices():
    """One clear uptrend, one clear downtrend. Momentum must get the sign right."""
    n = 1500
    idx = pd.bdate_range("2016-01-01", periods=n)
    t = np.arange(n)
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "UP": 100 * np.exp(0.0006 * t + rng.normal(0, 0.004, n).cumsum()),
        "DOWN": 100 * np.exp(-0.0004 * t + rng.normal(0, 0.004, n).cumsum()),
        "FLAT": 100 * np.exp(rng.normal(0, 0.004, n).cumsum()),
    }, index=idx)


@pytest.fixture
def gappy_prices(synthetic_prices):
    """Prices with an injected month-long hole and one bad tick."""
    px = synthetic_prices.copy()
    px = px.drop(px.index[400:420])
    px.iloc[600, 0] = px.iloc[600, 0] * 3.0   # unadjusted-split style artefact
    return px
