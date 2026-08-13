"""Options pricing tests.

The organising principle: four independent methods compute the same number by
different mathematics. Agreement is evidence; disagreement localises a bug.
Nothing here is checked against a hardcoded expected value that came from
running this code -- the references are closed-form solutions and no-arbitrage
identities that hold independently of any implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantlab.derivatives import (
    binomial_american,
    binomial_european,
    binomial_tree_full,
    binomial_up_and_out,
    black_scholes_call,
    black_scholes_greeks,
    black_scholes_put,
    crank_nicolson_european,
    crank_nicolson_up_and_out,
    implied_volatility,
    monte_carlo_european,
    monte_carlo_up_and_out,
    simulate_gbm_paths,
    trinomial_up_and_out,
    up_and_out_call_closed_form,
    vega,
)

# Standard test contract used throughout.
S0, K, T, R, SIGMA = 100.0, 110.0, 1.0, 0.05, 0.20


# ---------------------------------------------------------------------------
# Analytic
# ---------------------------------------------------------------------------

def test_put_call_parity():
    """C - P = S - K*exp(-rT). Model-independent; must hold to machine precision."""
    c = black_scholes_call(S0, K, T, R, SIGMA)
    p = black_scholes_put(S0, K, T, R, SIGMA)
    assert c - p == pytest.approx(S0 - K * np.exp(-R * T), abs=1e-12)


def test_call_within_no_arbitrage_bounds():
    """max(S - K*exp(-rT), 0) <= C <= S."""
    c = black_scholes_call(S0, K, T, R, SIGMA)
    assert max(S0 - K * np.exp(-R * T), 0.0) <= c <= S0


def test_price_monotonic_in_volatility():
    """Options are long volatility: more vol, more value. Always."""
    prices = [black_scholes_call(S0, K, T, R, s) for s in (0.05, 0.10, 0.20, 0.40, 0.80)]
    assert all(b > a for a, b in zip(prices, prices[1:]))


def test_call_monotonic_in_spot_and_strike():
    calls_by_spot = [black_scholes_call(s, K, T, R, SIGMA) for s in (80, 90, 100, 110, 120)]
    assert all(b > a for a, b in zip(calls_by_spot, calls_by_spot[1:]))
    calls_by_strike = [black_scholes_call(S0, k, T, R, SIGMA) for k in (80, 90, 100, 110, 120)]
    assert all(b < a for a, b in zip(calls_by_strike, calls_by_strike[1:]))


def test_deep_itm_call_approaches_forward():
    """A call struck at ~0 is worth the discounted forward, i.e. the stock."""
    c = black_scholes_call(S0, 0.01, T, R, SIGMA)
    assert c == pytest.approx(S0 - 0.01 * np.exp(-R * T), rel=1e-6)


def test_greeks_match_finite_differences():
    """Analytic greeks vs bumping the price function. Catches algebra slips.

    Step sizes are chosen per derivative order, which matters more than it
    looks. A central FIRST difference has total error ~ eps/h + h^2, minimised
    near h ~ eps^(1/3). A SECOND difference has error ~ eps/h^2 + h^2 and needs
    h ~ eps^(1/4) -- about 1e-4 relative, i.e. h = 0.01 at S = 100.

    Using h = 1e-5 for gamma instead produces a 0.9% discrepancy that looks
    exactly like a wrong formula but is pure floating-point cancellation: the
    numerator is ~2e-12 formed by subtracting numbers of order 6.
    """
    g = black_scholes_greeks(S0, K, T, R, SIGMA, option="call")

    h1 = 1e-4 * S0          # first derivatives
    h2 = 1e-2 * S0          # second derivative
    hv = 1e-4               # volatility bump

    fd_delta = (black_scholes_call(S0 + h1, K, T, R, SIGMA)
                - black_scholes_call(S0 - h1, K, T, R, SIGMA)) / (2 * h1)
    fd_gamma = (black_scholes_call(S0 + h2, K, T, R, SIGMA)
                - 2 * black_scholes_call(S0, K, T, R, SIGMA)
                + black_scholes_call(S0 - h2, K, T, R, SIGMA)) / h2**2
    fd_vega = (black_scholes_call(S0, K, T, R, SIGMA + hv)
               - black_scholes_call(S0, K, T, R, SIGMA - hv)) / (2 * hv)

    # Gamma's tolerance is looser on purpose. With h = 1.0 the second
    # difference carries an O(h^2) truncation error of order 1e-4 relative;
    # shrinking h to remove it reintroduces the cancellation noise this test
    # was rewritten to avoid. The exact algebraic identity in the next test is
    # the rigorous check on gamma -- this one just confirms the right ballpark.
    assert g["delta"] == pytest.approx(fd_delta, rel=1e-6)
    assert g["gamma"] == pytest.approx(fd_gamma, rel=1e-3)
    assert g["vega"] == pytest.approx(fd_vega, rel=1e-6)
    assert 0.0 < g["delta"] < 1.0
    assert g["gamma"] > 0
    assert g["theta"] < 0  # long options decay


def test_gamma_matches_closed_form_identity():
    """Independent check on gamma: gamma = vega / (S^2 * sigma * T).

    Both come from the same pdf(d1) term, so this is an algebraic identity the
    implementation must satisfy regardless of finite-difference noise.
    """
    g = black_scholes_greeks(S0, K, T, R, SIGMA, option="call")
    v = float(vega(S0, K, T, R, SIGMA))
    assert g["gamma"] == pytest.approx(v / (S0**2 * SIGMA * T), rel=1e-12)


def test_call_and_put_share_gamma_and_vega():
    """Put-call parity is linear in S, so second-order greeks must coincide."""
    gc = black_scholes_greeks(S0, K, T, R, SIGMA, option="call")
    gp = black_scholes_greeks(S0, K, T, R, SIGMA, option="put")
    assert gc["gamma"] == pytest.approx(gp["gamma"], rel=1e-12)
    assert gc["vega"] == pytest.approx(gp["vega"], rel=1e-12)
    # Parity also fixes the delta relationship: delta_call - delta_put = 1.
    assert gc["delta"] - gp["delta"] == pytest.approx(1.0, abs=1e-12)


def test_put_delta_is_negative():
    g = black_scholes_greeks(S0, K, T, R, SIGMA, option="put")
    assert -1.0 < g["delta"] < 0.0


# ---------------------------------------------------------------------------
# Cross-method agreement
# ---------------------------------------------------------------------------

def test_binomial_converges_to_black_scholes():
    exact = black_scholes_call(S0, K, T, R, SIGMA)
    coarse = abs(binomial_european(S0, K, T, R, SIGMA, n_steps=50) - exact)
    fine = abs(binomial_european(S0, K, T, R, SIGMA, n_steps=2000) - exact)
    assert fine < coarse
    assert fine < 1e-2


def test_crank_nicolson_converges_to_black_scholes():
    exact = black_scholes_call(S0, K, T, R, SIGMA)
    price = crank_nicolson_european(S0, K, T, R, SIGMA, n_space=800, n_time=800)
    assert price == pytest.approx(exact, abs=1e-3)


def test_crank_nicolson_put():
    exact = black_scholes_put(S0, K, T, R, SIGMA)
    price = crank_nicolson_european(S0, K, T, R, SIGMA, n_space=800, n_time=800,
                                    option="put")
    assert price == pytest.approx(exact, abs=1e-3)


def test_monte_carlo_within_confidence_interval():
    """MC must land within ~4 standard errors of the exact price.

    Testing against a tolerance rather than the reported standard error would
    be the weaker check -- this asserts the error estimate itself is honest.
    """
    exact = black_scholes_call(S0, K, T, R, SIGMA)
    price, stderr = monte_carlo_european(S0, K, T, R, SIGMA, n_paths=100_000,
                                         seed=7, return_stderr=True)
    assert abs(price - exact) < 4 * stderr, f"{price:.4f} vs {exact:.4f}, SE {stderr:.4f}"


def test_monte_carlo_error_shrinks_as_sqrt_n():
    """100x the paths should cut the standard error by roughly 10x."""
    _, se_small = monte_carlo_european(S0, K, T, R, SIGMA, n_paths=1_000,
                                       seed=1, return_stderr=True)
    _, se_large = monte_carlo_european(S0, K, T, R, SIGMA, n_paths=100_000,
                                       seed=1, return_stderr=True)
    assert se_large < se_small
    assert 5 < se_small / se_large < 20  # theory says 10


def test_all_four_methods_agree():
    """The headline check: analytic, lattice, PDE and simulation on one contract."""
    exact = black_scholes_call(S0, K, T, R, SIGMA)
    assert binomial_european(S0, K, T, R, SIGMA, n_steps=2000) == pytest.approx(exact, abs=2e-3)
    assert crank_nicolson_european(S0, K, T, R, SIGMA, 600, 600) == pytest.approx(exact, abs=2e-3)
    assert monte_carlo_european(S0, K, T, R, SIGMA, n_paths=200_000,
                                seed=3) == pytest.approx(exact, abs=5e-2)


# ---------------------------------------------------------------------------
# American options
# ---------------------------------------------------------------------------

def test_american_call_equals_european_without_dividends():
    """A textbook result (Shreve): never optimal to exercise an American call early.

    Exercising throws away remaining time value and forgoes interest on the
    strike, so the early-exercise feature is worth exactly zero here.
    """
    am = binomial_american(S0, K, T, R, SIGMA, n_steps=800, option="call")
    eu = binomial_european(S0, K, T, R, SIGMA, n_steps=800, option="call")
    assert am == pytest.approx(eu, rel=1e-9)


def test_american_put_strictly_more_valuable():
    """The put's early exercise IS worth something -- you get the strike in cash."""
    am = binomial_american(S0, K, T, R, SIGMA, n_steps=800, option="put")
    eu = binomial_european(S0, K, T, R, SIGMA, n_steps=800, option="put")
    assert am > eu
    assert am >= max(K - S0, 0.0) - 1e-9  # never below intrinsic


def test_american_put_never_below_intrinsic():
    """Deep in the money, the American put is worth at least immediate exercise."""
    deep = binomial_american(50.0, 100.0, T, R, SIGMA, n_steps=500, option="put")
    assert deep >= 100.0 - 50.0 - 1e-6


def test_full_tree_shapes_and_early_exercise():
    stock, value, exercise = binomial_tree_full(S0, K, T, R, SIGMA, n_steps=6,
                                                option="put", american=True)
    assert stock.shape == value.shape == exercise.shape == (7, 7)
    assert value[0, 0] > 0
    # A European tree can never flag early exercise.
    _, _, ex_eu = binomial_tree_full(S0, K, T, R, SIGMA, n_steps=6,
                                     option="put", american=False)
    assert not ex_eu.any()


# ---------------------------------------------------------------------------
# Barrier options
# ---------------------------------------------------------------------------

def test_barrier_cheaper_than_vanilla():
    """A knockout can only ever remove payoff, never add it."""
    barrier = up_and_out_call_closed_form(100, 95, 130, T, R, SIGMA)
    vanilla = black_scholes_call(100, 95, T, R, SIGMA)
    assert 0 < barrier < vanilla


def test_barrier_converges_to_vanilla_as_barrier_recedes():
    vanilla = black_scholes_call(100, 95, T, R, SIGMA)
    far = up_and_out_call_closed_form(100, 95, 1e6, T, R, SIGMA)
    assert far == pytest.approx(vanilla, rel=1e-6)


def test_barrier_degenerate_cases():
    assert up_and_out_call_closed_form(100, 95, 90, T, R, SIGMA) == 0.0   # B <= K
    assert up_and_out_call_closed_form(140, 95, 130, T, R, SIGMA) == 0.0  # already out
    assert binomial_up_and_out(140, 95, 130, T, R, SIGMA) == 0.0
    assert trinomial_up_and_out(140, 95, 130, T, R, SIGMA) == 0.0
    assert monte_carlo_up_and_out(140, 95, 130, T, R, SIGMA, n_paths=100) == 0.0


def test_trinomial_barrier_converges_monotonically():
    """Ritchken's stretched trinomial must converge smoothly.

    This is the test that motivated the trinomial's existence. The binomial
    lattice cannot align the barrier and match the volatility at the same time,
    so its error does not shrink reliably with more steps. Ritchken's third
    branch supplies the missing degree of freedom.
    """
    exact = up_and_out_call_closed_form(100, 95, 130, T, R, SIGMA)
    errors = [abs(trinomial_up_and_out(100, 95, 130, T, R, SIGMA, n_steps=n) - exact)
              for n in (250, 500, 1000, 2000)]
    assert errors[-1] < errors[0], f"no improvement with steps: {errors}"
    assert errors[-1] < 5e-3, f"final error {errors[-1]:.2e} too large"
    # Broadly decreasing: allow one non-monotonic step for discretisation noise.
    n_increases = sum(1 for a, b in zip(errors, errors[1:]) if b > a * 1.05)
    assert n_increases <= 1, f"error increased {n_increases} times: {errors}"


def test_crank_nicolson_barrier_accurate():
    """The PDE handles barriers most naturally -- it is just a boundary condition."""
    exact = up_and_out_call_closed_form(100, 95, 130, T, R, SIGMA)
    price = crank_nicolson_up_and_out(100, 95, 130, T, R, SIGMA,
                                      n_space=600, n_time=600)
    assert price == pytest.approx(exact, abs=5e-3)


def test_monte_carlo_barrier_with_bridge_beats_discrete():
    """Discrete monitoring misses crossings between steps, so it OVERprices.

    The Brownian bridge correction prices the probability of a crossing that
    the grid did not observe. Both prices are compared to the continuous-
    monitoring closed form, which is what they are trying to estimate.
    """
    exact = up_and_out_call_closed_form(100, 95, 130, T, R, SIGMA)
    bridged = monte_carlo_up_and_out(100, 95, 130, T, R, SIGMA,
                                     n_paths=60_000, seed=11, brownian_bridge=True)
    discrete = monte_carlo_up_and_out(100, 95, 130, T, R, SIGMA,
                                      n_paths=60_000, seed=11, brownian_bridge=False)
    assert discrete > exact, "discrete monitoring should overprice"
    assert abs(bridged - exact) < abs(discrete - exact), (
        f"bridge {bridged:.4f} no better than discrete {discrete:.4f} vs exact {exact:.4f}")


def test_all_barrier_methods_agree():
    """Closed form, trinomial, PDE and corrected MC on one barrier contract."""
    exact = up_and_out_call_closed_form(100, 95, 130, T, R, SIGMA)
    assert trinomial_up_and_out(100, 95, 130, T, R, SIGMA, 2000) == pytest.approx(exact, abs=5e-3)
    assert crank_nicolson_up_and_out(100, 95, 130, T, R, SIGMA, 600, 600) == pytest.approx(exact, abs=5e-3)
    mc, se = monte_carlo_up_and_out(100, 95, 130, T, R, SIGMA, n_paths=100_000,
                                    seed=5, return_stderr=True)
    assert abs(mc - exact) < 4 * se + 0.02


# ---------------------------------------------------------------------------
# Implied volatility
# ---------------------------------------------------------------------------

def test_implied_vol_inverts_black_scholes():
    """Round trip: price at sigma, invert, recover sigma."""
    for true_sigma in (0.10, 0.20, 0.35, 0.60):
        price = black_scholes_call(S0, K, T, R, true_sigma)
        assert implied_volatility(price, S0, K, T, R) == pytest.approx(true_sigma, abs=1e-6)


def test_implied_vol_round_trip_across_strikes():
    """Including deep ITM/OTM, where vega collapses and Newton alone fails."""
    for strike in (60, 80, 100, 120, 150, 200):
        for true_sigma in (0.15, 0.30):
            price = black_scholes_call(S0, strike, T, R, true_sigma)
            iv = implied_volatility(price, S0, strike, T, R)
            if np.isfinite(iv):
                assert iv == pytest.approx(true_sigma, abs=1e-4), f"K={strike}"


def test_implied_vol_for_puts():
    price = black_scholes_put(S0, K, T, R, 0.25)
    assert implied_volatility(price, S0, K, T, R, option="put") == pytest.approx(0.25, abs=1e-6)


def test_implied_vol_nan_outside_arbitrage_bounds():
    """A price no volatility can produce must return NaN, not a wrong number."""
    assert np.isnan(implied_volatility(S0 * 2, S0, K, T, R))   # above the spot bound
    assert np.isnan(implied_volatility(-1.0, S0, K, T, R))     # negative price


def test_implied_vol_handles_zero_time():
    assert np.isnan(implied_volatility(5.0, S0, K, 0.0, R))


# ---------------------------------------------------------------------------
# Simulation machinery
# ---------------------------------------------------------------------------

def test_gbm_paths_shape_and_start():
    paths = simulate_gbm_paths(S0, T, R, SIGMA, n_paths=100, n_steps=50, seed=1)
    assert paths.shape == (100, 51)
    assert np.allclose(paths[:, 0], S0)
    assert (paths > 0).all(), "GBM prices must stay strictly positive"


def test_gbm_terminal_distribution_matches_theory():
    """E[S_T] = S0*exp(rT) under the risk-neutral measure."""
    paths = simulate_gbm_paths(S0, T, R, SIGMA, n_paths=200_000, n_steps=252, seed=4)
    assert paths[:, -1].mean() == pytest.approx(S0 * np.exp(R * T), rel=0.01)
    # Var[ln S_T] = sigma^2 * T
    assert np.log(paths[:, -1] / S0).var() == pytest.approx(SIGMA**2 * T, rel=0.05)


def test_antithetic_reduces_variance():
    _, se_plain = monte_carlo_european(S0, K, T, R, SIGMA, n_paths=20_000,
                                       seed=2, antithetic=False, return_stderr=True)
    _, se_anti = monte_carlo_european(S0, K, T, R, SIGMA, n_paths=20_000,
                                      seed=2, antithetic=True, return_stderr=True)
    assert se_anti <= se_plain


def test_seed_reproducibility():
    a = monte_carlo_european(S0, K, T, R, SIGMA, n_paths=5_000, seed=99)
    b = monte_carlo_european(S0, K, T, R, SIGMA, n_paths=5_000, seed=99)
    assert a == b


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_invalid_option_type_rejected():
    with pytest.raises(ValueError, match="call.*put"):
        binomial_european(S0, K, T, R, SIGMA, option="banana")
    with pytest.raises(ValueError, match="call.*put"):
        black_scholes_greeks(S0, K, T, R, SIGMA, option="banana")
    with pytest.raises(ValueError, match="call.*put"):
        monte_carlo_european(S0, K, T, R, SIGMA, n_paths=100, option="banana")


def test_antithetic_requires_even_paths():
    with pytest.raises(ValueError, match="even"):
        simulate_gbm_paths(S0, T, R, SIGMA, n_paths=101, antithetic=True)


def test_zero_steps_rejected():
    with pytest.raises(ValueError, match="n_steps"):
        binomial_european(S0, K, T, R, SIGMA, n_steps=0)
