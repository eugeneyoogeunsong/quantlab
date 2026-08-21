# quantlab: Eugene (Yoogeun) Song (https://www.linkedin.com/in/yoogeunsong)
# Independent side project. MIT licensed; see LICENSE.

"""Monte Carlo tests: dividends, variance reduction, LSM, and simulated Greeks.

The organising principle is the one used throughout the package: every simulated
number is checked against an independent route, never against a figure produced
by this code. Specifically, the European prices are checked against
Black-Scholes, the barrier against the reflection-principle closed form, the
American put against a 5,000-step binomial lattice, and the Greeks against
`black_scholes_greeks`.

Two conventions worth stating once. First, tolerances are quoted in standard
errors wherever a standard error exists, since asserting on a fixed absolute
band would silently pass a run whose error estimate had gone wrong. Second, the
seeds are fixed, so a failure here is a real regression rather than an unlucky
draw; the tolerances were nevertheless set from sweeps over eight seeds, and the
comment beside each one records the worst deviation actually observed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantlab.derivatives.analytic import (
    black_scholes_call,
    black_scholes_greeks,
    black_scholes_put,
    up_and_out_call_closed_form,
)
from quantlab.derivatives.binomial import binomial_american
from quantlab.derivatives.monte_carlo import (
    longstaff_schwartz_american,
    monte_carlo_european,
    monte_carlo_greeks,
    monte_carlo_up_and_out,
    simulate_gbm_paths,
)

# The standard contract used by tests/test_derivatives.py, kept identical so the
# two files can be read side by side.
S0, K, T, R, SIGMA = 100.0, 110.0, 1.0, 0.05, 0.20
Q = 0.04

# Barrier contract: in the money, with the knockout well above spot.
B_S0, B_K = 100.0, 95.0


# ---------------------------------------------------------------------------
# The dividend yield must be a pure extension: q=0 is the old behaviour
# ---------------------------------------------------------------------------

def _legacy_paths(S0, T, r, sigma, n_paths, n_steps, seed, antithetic):
    """The path generator exactly as it stood before `q` was added.

    Transcribed here rather than imported, so that the regression check does not
    depend on the very module it is policing.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    if antithetic:
        half = rng.standard_normal((n_paths // 2, n_steps))
        Z = np.concatenate([half, -half], axis=0)
    else:
        Z = rng.standard_normal((n_paths, n_steps))
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1)
    return S0 * np.exp(log_paths)


@pytest.mark.parametrize("antithetic", [False, True])
def test_zero_dividend_reproduces_legacy_paths_bitwise(antithetic):
    """Not merely close: `q=0.0` must give the identical float in every cell.

    `r - 0.0` is exact in IEEE 754, so any difference here would mean the
    refactor changed the order of the arithmetic or the consumption of the
    random stream.
    """
    legacy = _legacy_paths(S0, T, R, SIGMA, 2_000, 50, 11, antithetic)
    current = simulate_gbm_paths(S0, T, R, SIGMA, 2_000, 50, 11, antithetic)
    assert np.array_equal(current, legacy)
    assert np.array_equal(
        simulate_gbm_paths(S0, T, R, SIGMA, 2_000, 50, 11, antithetic, False, 0.0), legacy)


def test_zero_dividend_prices_are_unchanged():
    """The three pricers must give the same float with `q` defaulted and passed."""
    kwargs = dict(n_paths=5_000, n_steps=50, seed=5)
    assert (monte_carlo_european(S0, K, T, R, SIGMA, **kwargs)
            == monte_carlo_european(S0, K, T, R, SIGMA, **kwargs, q=0.0))
    assert (monte_carlo_up_and_out(B_S0, B_K, 130.0, T, R, SIGMA, **kwargs)
            == monte_carlo_up_and_out(B_S0, B_K, 130.0, T, R, SIGMA, **kwargs, q=0.0))
    assert (longstaff_schwartz_american(S0, K, T, R, SIGMA, n_paths=5_000, n_steps=25, seed=5)
            == longstaff_schwartz_american(S0, K, T, R, SIGMA, n_paths=5_000, n_steps=25,
                                           seed=5, q=0.0))


def test_dividend_yield_matches_black_scholes():
    """A dividend yield lowers the call and raises the put, by the amounts BSM says."""
    for option, reference in (("call", black_scholes_call), ("put", black_scholes_put)):
        exact = float(reference(S0, K, T, R, SIGMA, Q))
        price, se = monte_carlo_european(S0, K, T, R, SIGMA, n_paths=100_000, n_steps=50,
                                         option=option, seed=17, return_stderr=True, q=Q)
        assert abs(price - exact) < 3 * se, f"{option}: {price:.4f} vs {exact:.4f}, SE {se:.4f}"

    with_yield = monte_carlo_european(S0, K, T, R, SIGMA, n_paths=50_000, n_steps=50, seed=2, q=Q)
    without = monte_carlo_european(S0, K, T, R, SIGMA, n_paths=50_000, n_steps=50, seed=2)
    assert with_yield < without


def test_dividend_yield_lowers_the_barrier_price():
    """A yield drags the drift down, so fewer paths knock out but fewer finish ITM."""
    exact_no_q = up_and_out_call_closed_form(B_S0, B_K, 150.0, T, R, SIGMA)
    mc_no_q, se = monte_carlo_up_and_out(B_S0, B_K, 150.0, T, R, SIGMA, n_paths=50_000,
                                         n_steps=100, seed=8, return_stderr=True)
    assert abs(mc_no_q - exact_no_q) < 4 * se
    mc_q = monte_carlo_up_and_out(B_S0, B_K, 150.0, T, R, SIGMA, n_paths=50_000,
                                  n_steps=100, seed=8, q=Q)
    assert mc_q < mc_no_q


# ---------------------------------------------------------------------------
# Convergence to the closed form
# ---------------------------------------------------------------------------

def test_monte_carlo_within_three_standard_errors():
    """The headline convergence check, on both option types."""
    for option, reference in (("call", black_scholes_call), ("put", black_scholes_put)):
        exact = float(reference(S0, K, T, R, SIGMA))
        price, se = monte_carlo_european(S0, K, T, R, SIGMA, n_paths=100_000, n_steps=50,
                                         option=option, seed=3, return_stderr=True)
        assert abs(price - exact) < 3 * se, f"{option}: {price:.4f} vs {exact:.4f}, SE {se:.4f}"


def test_control_variate_price_stays_unbiased():
    """Variance reduction must move the noise, not the answer."""
    exact = float(black_scholes_call(S0, K, T, R, SIGMA))
    price, se = monte_carlo_european(S0, K, T, R, SIGMA, n_paths=100_000, n_steps=50,
                                     seed=3, return_stderr=True, control_variate=True)
    assert abs(price - exact) < 3 * se, f"{price:.4f} vs {exact:.4f}, SE {se:.4f}"


# ---------------------------------------------------------------------------
# Control variates
# ---------------------------------------------------------------------------

def test_control_variate_cuts_european_standard_error():
    """The underlying is a good control for a near-the-money call: ~1.9x measured.

    We assert 1.7x rather than the measured 1.90x, since the ratio itself is a
    random variable; over eight seeds it varied only between 1.896 and 1.901,
    so the margin is generous.
    """
    _, se_plain = monte_carlo_european(S0, K, T, R, SIGMA, n_paths=50_000, n_steps=50,
                                       seed=4, return_stderr=True)
    _, se_cv = monte_carlo_european(S0, K, T, R, SIGMA, n_paths=50_000, n_steps=50,
                                    seed=4, return_stderr=True, control_variate=True)
    assert se_plain / se_cv > 1.7, f"only {se_plain / se_cv:.2f}x"


def test_control_variate_is_strongest_deep_in_the_money():
    """corr(payoff, S_T) rises as the call moves ITM, and the reduction with it."""
    ratios = []
    for strike in (80.0, 110.0, 130.0):
        _, se_plain = monte_carlo_european(S0, strike, T, R, SIGMA, n_paths=50_000,
                                           n_steps=50, seed=6, return_stderr=True)
        _, se_cv = monte_carlo_european(S0, strike, T, R, SIGMA, n_paths=50_000, n_steps=50,
                                        seed=6, return_stderr=True, control_variate=True)
        ratios.append(se_plain / se_cv)
    assert ratios[0] > ratios[1] > ratios[2], ratios
    assert ratios[0] > 5.0, f"deep ITM should give ~8x, got {ratios[0]:.2f}x"


def test_control_variate_cuts_barrier_standard_error():
    """A distant barrier is nearly a vanilla, so the vanilla control works: ~2.7x."""
    plain, se_plain = monte_carlo_up_and_out(B_S0, B_K, 180.0, T, R, SIGMA, n_paths=40_000,
                                             n_steps=100, seed=9, return_stderr=True)
    price, se_cv = monte_carlo_up_and_out(B_S0, B_K, 180.0, T, R, SIGMA, n_paths=40_000,
                                          n_steps=100, seed=9, return_stderr=True,
                                          control_variate=True)
    assert se_plain / se_cv > 2.2, f"only {se_plain / se_cv:.2f}x"
    exact = up_and_out_call_closed_form(B_S0, B_K, 180.0, T, R, SIGMA)
    assert abs(price - exact) < 4 * se_cv, f"{price:.4f} vs {exact:.4f}, SE {se_cv:.4f}"
    assert abs(plain - exact) < 4 * se_plain


def test_control_variate_gains_nothing_on_a_near_barrier():
    """The documented failure mode, pinned so it cannot quietly change.

    With the barrier at 130 the knockout removes exactly the paths on which the
    vanilla pays most, corr(Y, X) falls to 0.15, and the linear projection has
    almost nothing left to remove: 1.01x. Fitting b* on the sample minimises the
    sample variance by construction, so the ratio can never fall below 1.
    """
    _, se_plain = monte_carlo_up_and_out(B_S0, B_K, 130.0, T, R, SIGMA, n_paths=40_000,
                                         n_steps=100, seed=9, return_stderr=True)
    _, se_cv = monte_carlo_up_and_out(B_S0, B_K, 130.0, T, R, SIGMA, n_paths=40_000,
                                      n_steps=100, seed=9, return_stderr=True,
                                      control_variate=True)
    assert 1.0 <= se_plain / se_cv < 1.15, f"{se_plain / se_cv:.3f}x"


# ---------------------------------------------------------------------------
# Least-squares Monte Carlo
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def american_put_lattice():
    """Reference American put from an independent method (0.25 s at 5,000 steps).

    Called with `binomial_american`'s current signature only: the dividend
    argument is being added to that module in parallel, and nothing here should
    depend on whether it has landed.
    """
    return binomial_american(S0, K, T, R, SIGMA, n_steps=5_000, option="put")


def test_lsm_american_put_matches_the_lattice(american_put_lattice):
    """LSM within 0.15 (1.3% of price, ~4 SE) of a 5,000-step binomial.

    The tolerance is deliberately loose relative to the standard error of 0.038,
    because two systematic gaps sit underneath the noise: 50 exercise dates
    instead of continuous exercise, and a degree-3 regression instead of the
    true continuation value. Both push the price down, and over eight seeds the
    worst error observed was -0.061.
    """
    price, se = longstaff_schwartz_american(S0, K, T, R, SIGMA, n_paths=50_000, n_steps=50,
                                            seed=3, return_stderr=True)
    assert price == pytest.approx(american_put_lattice, abs=0.15), f"SE {se:.4f}"
    assert se < 0.05


def test_lsm_is_low_biased_against_the_lattice(american_put_lattice):
    """Averaged over four seeds the sign of the bias is legible: LSM prices low.

    A single run cannot show this (the standard error of 0.038 swamps a bias of
    0.024), which is exactly why the docstring reports the averaged figure.
    """
    prices = [longstaff_schwartz_american(S0, K, T, R, SIGMA, n_paths=50_000, n_steps=50,
                                          seed=s) for s in range(4)]
    mean_error = float(np.mean(prices)) - american_put_lattice
    assert -0.10 < mean_error < 0.0, f"mean error {mean_error:+.4f}"


def test_lsm_american_put_beats_its_european_twin():
    """The early-exercise premium is real and large here: ~1.28 on a 10.68 put."""
    european = float(black_scholes_put(S0, K, T, R, SIGMA))
    american, se = longstaff_schwartz_american(S0, K, T, R, SIGMA, n_paths=50_000, n_steps=50,
                                               seed=1, return_stderr=True)
    assert american > european + 10 * se


def test_lsm_american_call_equals_european_without_dividends():
    """With q=0 early exercise of a call is never optimal, so LSM must reproduce BSM.

    Any spurious exercise the fitted rule commits shows up as a shortfall, so
    this is a sharper test of the regression than the put: over six seeds the
    worst deviation at 25 exercise dates was -0.072, against a standard error
    of 0.052.
    """
    exact = float(black_scholes_call(S0, K, T, R, SIGMA))
    price, se = longstaff_schwartz_american(S0, K, T, R, SIGMA, n_paths=50_000, n_steps=25,
                                            option="call", seed=2, return_stderr=True)
    assert abs(price - exact) < 4 * se, f"{price:.4f} vs {exact:.4f}, SE {se:.4f}"
    assert price < exact + 3 * se, "an American call on a non-payer cannot beat the European"


def test_lsm_dividend_call_matches_the_symmetric_lattice_put():
    """The dividend path, cross-checked against a lattice that has no dividends.

    American put-call symmetry (McDonald & Schroder 1998) gives

        C_A(S, K, T, r, q) = P_A(K, S, T, q, r),

    so an American call on a 6% yielder at a zero rate is the same contract as an
    American put with spot and strike swapped, priced at a 6% rate on a
    non-payer. That right-hand side is exactly what `binomial_american` prices
    with its current signature, which turns the identity into an independent
    check of `q` inside the LSM routine rather than a self-consistency check.

    The early-exercise premium is large and unambiguous here: 0.63 on a 5.17
    European call, or 17 standard errors, so this also confirms the routine is
    finding the exercise boundary and not merely reproducing the European price.
    """
    q_yield = 0.06
    symmetric = binomial_american(S0, S0, T, q_yield, SIGMA, n_steps=5_000, option="put")
    american, se = longstaff_schwartz_american(S0, S0, T, 0.0, SIGMA, n_paths=50_000,
                                               n_steps=50, option="call", seed=1,
                                               return_stderr=True, q=q_yield)
    assert american == pytest.approx(symmetric, abs=0.10), f"SE {se:.4f}"
    european = float(black_scholes_call(S0, S0, T, 0.0, SIGMA, q_yield))
    assert american > european + 10 * se, f"{american:.4f} vs {european:.4f}, SE {se:.4f}"


def test_lsm_out_of_sample_pass_also_matches_the_lattice(american_put_lattice):
    """Freezing the rule and resimulating removes the in-sample foresight bias.

    It cannot remove the low bias from a suboptimal rule, and should not: the
    resimulated figure is a valid lower bound on the American value, which is
    the number worth quoting.
    """
    price, se = longstaff_schwartz_american(S0, K, T, R, SIGMA, n_paths=50_000, n_steps=50,
                                            seed=3, return_stderr=True, out_of_sample=True)
    assert price == pytest.approx(american_put_lattice, abs=0.15), f"SE {se:.4f}"
    assert price < american_put_lattice + 3 * se


def test_lsm_bases_agree():
    """Power and weighted-Laguerre bases span the same continuation surface."""
    power, se = longstaff_schwartz_american(S0, K, T, R, SIGMA, n_paths=50_000, n_steps=50,
                                            seed=3, basis="power", return_stderr=True)
    lag = longstaff_schwartz_american(S0, K, T, R, SIGMA, n_paths=50_000, n_steps=50,
                                      seed=3, basis="laguerre")
    assert abs(power - lag) < 0.5 * se, f"{power:.4f} vs {lag:.4f}, SE {se:.4f}"


def test_lsm_rejects_bad_arguments():
    with pytest.raises(ValueError, match="call.*put"):
        longstaff_schwartz_american(S0, K, T, R, SIGMA, n_paths=100, option="banana")
    with pytest.raises(ValueError, match="basis"):
        longstaff_schwartz_american(S0, K, T, R, SIGMA, n_paths=100, basis="chebyshev")
    with pytest.raises(ValueError, match="degree"):
        longstaff_schwartz_american(S0, K, T, R, SIGMA, n_paths=100, degree=0)
    with pytest.raises(ValueError, match="n_steps"):
        longstaff_schwartz_american(S0, K, T, R, SIGMA, n_paths=100, n_steps=0)


def test_lsm_deep_in_the_money_put_exercises_immediately():
    """At S0 far below the strike the value is the intrinsic value, exactly."""
    price = longstaff_schwartz_american(10.0, K, T, R, SIGMA, n_paths=5_000, n_steps=25, seed=1)
    assert price == pytest.approx(K - 10.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Simulated Greeks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("q", [0.0, Q])
def test_pathwise_delta_and_vega_match_black_scholes(option, q):
    """Pathwise estimators, valid because the payoff is Lipschitz in S0 and sigma."""
    exact = black_scholes_greeks(S0, K, T, R, SIGMA, q, option)
    mc = monte_carlo_greeks(S0, K, T, R, SIGMA, n_paths=200_000, option=option,
                            seed=13, return_stderr=True, q=q)
    assert mc["delta"] == pytest.approx(exact["delta"], rel=1e-2)
    assert mc["vega"] == pytest.approx(exact["vega"], rel=1e-2)
    assert abs(mc["delta"] - exact["delta"]) < 4 * mc["delta_stderr"]
    assert abs(mc["vega"] - exact["vega"]) < 4 * mc["vega_stderr"]


@pytest.mark.parametrize("option", ["call", "put"])
def test_likelihood_ratio_gamma_matches_black_scholes(option):
    """Gamma needs the wider band: the LR weight multiplies the raw payoff.

    Its standard error is ~1.9% of gamma at 200,000 paths against 0.29% for the
    pathwise delta, so 5% relative is the honest tolerance here, and the 4-SE
    check is what actually keeps it sharp.
    """
    exact = black_scholes_greeks(S0, K, T, R, SIGMA, 0.0, option)
    mc = monte_carlo_greeks(S0, K, T, R, SIGMA, n_paths=200_000, option=option,
                            seed=13, return_stderr=True)
    assert mc["gamma"] == pytest.approx(exact["gamma"], rel=5e-2)
    assert abs(mc["gamma"] - exact["gamma"]) < 4 * mc["gamma_stderr"]


def test_gamma_is_identical_for_call_and_put():
    """Put-call parity is linear in S, so the two gammas coincide."""
    call = monte_carlo_greeks(S0, K, T, R, SIGMA, n_paths=100_000, seed=21)
    put = monte_carlo_greeks(S0, K, T, R, SIGMA, n_paths=100_000, option="put", seed=21)
    exact = black_scholes_greeks(S0, K, T, R, SIGMA)["gamma"]
    assert call["gamma"] == pytest.approx(exact, rel=6e-2)
    assert put["gamma"] == pytest.approx(exact, rel=6e-2)


def test_pathwise_delta_is_less_noisy_than_likelihood_ratio():
    """Where both estimators are valid, pathwise wins: ~2.1x lower standard error.

    This is the practical content of the pathwise/LR distinction. LR is not a
    better estimator, it is the estimator that survives a payoff the pathwise
    method cannot differentiate.
    """
    mc = monte_carlo_greeks(S0, K, T, R, SIGMA, n_paths=200_000, seed=13, return_stderr=True)
    exact = black_scholes_greeks(S0, K, T, R, SIGMA)
    assert mc["delta_lr"] == pytest.approx(exact["delta"], rel=3e-2)
    assert mc["delta_lr_stderr"] / mc["delta_stderr"] > 1.8


def test_greeks_reject_a_bad_option_string():
    with pytest.raises(ValueError, match="call.*put"):
        monte_carlo_greeks(S0, K, T, R, SIGMA, n_paths=100, option="banana")


# ---------------------------------------------------------------------------
# Sobol' quasi-Monte Carlo
# ---------------------------------------------------------------------------

def test_sobol_beats_pseudo_random_on_an_equal_budget():
    """RMSE over eight replications, at 2^14 points and 8 time steps.

    Scrambling is what makes the replications independent, and therefore what
    makes this comparison meaningful at all.
    """
    exact = float(black_scholes_call(S0, K, T, R, SIGMA))
    sobol = [monte_carlo_european(S0, K, T, R, SIGMA, n_paths=2**14, n_steps=8,
                                  seed=s, sobol=True) for s in range(8)]
    pseudo = [monte_carlo_european(S0, K, T, R, SIGMA, n_paths=2**14, n_steps=8,
                                   seed=100 + s, antithetic=False) for s in range(8)]
    rmse_sobol = float(np.sqrt(np.mean((np.array(sobol) - exact) ** 2)))
    rmse_pseudo = float(np.sqrt(np.mean((np.array(pseudo) - exact) ** 2)))
    assert rmse_sobol < rmse_pseudo / 4, f"sobol {rmse_sobol:.5f} vs pseudo {rmse_pseudo:.5f}"
    assert max(abs(v - exact) for v in sobol) < 0.05


def test_sobol_requires_a_power_of_two_sample():
    with pytest.raises(ValueError, match="power of two"):
        monte_carlo_european(S0, K, T, R, SIGMA, n_paths=10_000, n_steps=8, sobol=True)


def test_sobol_and_antithetic_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        simulate_gbm_paths(S0, T, R, SIGMA, 2**10, 8, 1, antithetic=True, sobol=True)


def test_sobol_refuses_to_report_a_standard_error():
    """A dependent sample has no sample standard error; we return NaN, loudly."""
    price, se = monte_carlo_european(S0, K, T, R, SIGMA, n_paths=2**12, n_steps=8,
                                     seed=1, sobol=True, return_stderr=True)
    assert math.isnan(se)
    assert price > 0.0


def test_sobol_supersedes_antithetic_at_the_pricer_level():
    """`monte_carlo_european` defaults antithetic=True, so this must not raise."""
    price = monte_carlo_european(S0, K, T, R, SIGMA, n_paths=2**12, n_steps=8,
                                 seed=1, sobol=True)
    assert price == pytest.approx(float(black_scholes_call(S0, K, T, R, SIGMA)), abs=0.15)


# ---------------------------------------------------------------------------
# Slower cross-checks, kept out of the default fast loop
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_all_variance_reduction_routes_agree_at_high_path_counts():
    """Plain, antithetic, control-variate and Sobol' must price the same contract.

    Four samplers, one number: agreement is evidence that the variance reduction
    is doing what it claims, since a mistake in any of them (a wrong control
    mean, an unbalanced net) would move the price, not merely the noise.
    """
    exact = float(black_scholes_call(S0, K, T, R, SIGMA))
    plain, se = monte_carlo_european(S0, K, T, R, SIGMA, n_paths=200_000, seed=31,
                                     antithetic=False, return_stderr=True)
    anti = monte_carlo_european(S0, K, T, R, SIGMA, n_paths=200_000, seed=31)
    cv = monte_carlo_european(S0, K, T, R, SIGMA, n_paths=200_000, seed=31, control_variate=True)
    qmc_price = monte_carlo_european(S0, K, T, R, SIGMA, n_paths=2**17, n_steps=16,
                                     seed=31, sobol=True)
    for label, value in (("plain", plain), ("antithetic", anti),
                         ("control", cv), ("sobol", qmc_price)):
        assert value == pytest.approx(exact, abs=4 * se), f"{label}: {value:.4f}"


@pytest.mark.slow
def test_lsm_converges_as_exercise_dates_increase(american_put_lattice):
    """The Bermudan gap closes from below as the exercise set is refined.

    We average four seeds per step count, since a single run's standard error of
    0.038 is larger than the effect being measured (0.036 down to 0.014).
    """
    errors = []
    for n_steps in (25, 100):
        prices = [longstaff_schwartz_american(S0, K, T, R, SIGMA, n_paths=50_000,
                                              n_steps=n_steps, seed=s) for s in range(4)]
        errors.append(float(np.mean(prices)) - american_put_lattice)
    assert errors[0] < 0.0 and errors[1] < 0.0
    assert errors[1] > errors[0], f"25 dates {errors[0]:+.4f}, 100 dates {errors[1]:+.4f}"
