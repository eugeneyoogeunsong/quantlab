# quantlab: Eugene (Yoogeun) Song (https://www.linkedin.com/in/yoogeunsong)
# Independent side project. MIT licensed; see LICENSE.

"""Tests for the closed-form additions and for the implied-vol inversion.

Same organising principle as `test_derivatives.py`: every new number is checked
against an independent route, never against a value produced by this code.
Specifically, the barrier formula with a dividend yield is checked against a
high-step trinomial lattice, the digitals against a tight call spread, the
second-order Greeks against central differences of the first-order ones, and
put-call parity against the static replication identity, which holds
model-independently. Where a `q = 0.0` regression is claimed we assert bitwise
equality, not approximate equality: appending a parameter must reproduce the old
number exactly, or it is not a backwards-compatible change.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantlab.derivatives.analytic import (
    black_scholes_call,
    black_scholes_greeks,
    black_scholes_put,
    black_scholes_put_call_parity_residual,
    digital_call,
    digital_put,
    up_and_out_call_closed_form,
    vega,
)

# Imported from the module rather than the package so this test is insensitive to
# what `quantlab/derivatives/__init__.py` happens to re-export.
from quantlab.derivatives.binomial import trinomial_up_and_out
from quantlab.derivatives.implied_vol import (
    _manaster_koehler_sigma,
    implied_volatility,
)

# Standard test contract, matching test_derivatives.py.
S0, K, T, R, SIGMA = 100.0, 110.0, 1.0, 0.05, 0.20

# Contracts spanning moneyness, maturity, rate and dividend yield.
CONTRACTS = [
    (100.0, 110.0, 1.00, 0.05, 0.20, 0.00),
    (100.0, 110.0, 1.00, 0.05, 0.20, 0.03),
    (100.0, 90.0, 0.50, 0.03, 0.35, 0.05),
    (100.0, 100.0, 2.00, 0.06, 0.15, 0.02),
    (120.0, 80.0, 0.25, 0.01, 0.45, 0.00),
    (80.0, 120.0, 3.00, 0.04, 0.25, 0.06),
]

BARRIER_CONTRACTS = [
    # S0,     K,     B,     T,    r,     sigma, q
    (100.0, 95.0, 130.0, 1.00, 0.05, 0.20, 0.03),
    (100.0, 90.0, 140.0, 0.50, 0.03, 0.30, 0.06),
    (100.0, 100.0, 120.0, 2.00, 0.06, 0.25, 0.02),
    (100.0, 80.0, 115.0, 1.50, 0.04, 0.18, 0.08),
]


# ---------------------------------------------------------------------------
# q = 0.0 must reproduce the previous behaviour exactly
# ---------------------------------------------------------------------------

def test_q_zero_is_bitwise_identical_to_the_default():
    """Appending `q` may not perturb a single bit of the q-free result.

    Approximate equality would hide a reordered expression that costs a few ulps;
    the contract we are making is stronger than that, so we assert `==`.
    """
    for (S, k, t, r, s, _q) in CONTRACTS:
        assert black_scholes_call(S, k, t, r, s) == black_scholes_call(S, k, t, r, s, 0.0)
        assert black_scholes_put(S, k, t, r, s) == black_scholes_put(S, k, t, r, s, 0.0)
        assert vega(S, k, t, r, s) == vega(S, k, t, r, s, 0.0)
        assert digital_call(S, k, t, r, s) == digital_call(S, k, t, r, s, 0.0)
        assert digital_put(S, k, t, r, s) == digital_put(S, k, t, r, s, 0.0)
        for option in ("call", "put"):
            assert (black_scholes_greeks(S, k, t, r, s, option=option)
                    == black_scholes_greeks(S, k, t, r, s, 0.0, option))

    for (S, k, b, t, r, s, _q) in BARRIER_CONTRACTS:
        assert (up_and_out_call_closed_form(S, k, b, t, r, s)
                == up_and_out_call_closed_form(S, k, b, t, r, s, 0.0))


def test_first_order_greek_dict_is_unchanged_by_default():
    """`second_order` is opt-in: the default keys and values must not move."""
    plain = black_scholes_greeks(S0, K, T, R, SIGMA, option="call")
    assert set(plain) == {"delta", "gamma", "vega", "theta", "rho"}

    extended = black_scholes_greeks(S0, K, T, R, SIGMA, option="call", second_order=True)
    assert set(extended) == {"delta", "gamma", "vega", "theta", "rho",
                             "vanna", "volga", "charm"}
    for key, value in plain.items():
        assert extended[key] == value, f"{key} changed when second_order was switched on"


def test_barrier_at_zero_dividend_still_matches_the_lattice():
    """The q = 0.0 barrier price must keep agreeing with an independent lattice."""
    for (S, k, b, t, r, s, _q) in BARRIER_CONTRACTS:
        exact = up_and_out_call_closed_form(S, k, b, t, r, s, 0.0)
        lattice = trinomial_up_and_out(S, k, b, t, r, s, n_steps=3000)
        assert lattice == pytest.approx(exact, abs=3e-3), f"K={k} B={b}"


# ---------------------------------------------------------------------------
# Barrier with a dividend yield
# ---------------------------------------------------------------------------

def test_barrier_with_dividend_matches_high_step_trinomial():
    """Cross-check the Merton/Reiner-Rubinstein barrier formula against a lattice.

    The lattice used here carries no dividend yield in the signature this test
    was written against, so we exploit the fact that `q` only enters through the
    drift: pricing with drift `(r - q)` and discounting at `r` is the same
    computation as pricing with drift `(r - q)` and discounting at `(r - q)`,
    rescaled by the ratio of the two discount factors. Therefore

        price(S0, K, B, T, r, sigma, q) = exp(-q*T) * lattice(S0, K, B, T, r-q, sigma).

    That gives us an independent route with no shared code beyond the payoff.
    """
    worst = 0.0
    for (S, k, b, t, r, s, q) in BARRIER_CONTRACTS:
        closed = up_and_out_call_closed_form(S, k, b, t, r, s, q)
        lattice = np.exp(-q * t) * trinomial_up_and_out(S, k, b, t, r - q, s, n_steps=3000)
        worst = max(worst, abs(closed - lattice))
        assert closed == pytest.approx(lattice, abs=3e-3), f"K={k} B={b} q={q}"
    assert worst < 3e-3, f"worst barrier vs trinomial error {worst:.2e}"


def test_barrier_with_dividend_converges_to_the_vanilla_with_dividend():
    """As B recedes the knockout is unreachable, so the price is the vanilla one.

    This is the independent check that the `exp(-q*T)` factors sit on the right
    two of the four terms: `black_scholes_call` shares no code with the barrier
    formula beyond `_d1_d2`.
    """
    for (S, k, _b, t, r, s, q) in BARRIER_CONTRACTS:
        far = up_and_out_call_closed_form(S, k, 1e7, t, r, s, q)
        assert far == pytest.approx(float(black_scholes_call(S, k, t, r, s, q)), rel=1e-8)


def test_barrier_with_dividend_stays_below_the_vanilla():
    """A knockout can only remove payoff, never add it, whatever `q` is."""
    for (S, k, b, t, r, s, q) in BARRIER_CONTRACTS:
        priced = up_and_out_call_closed_form(S, k, b, t, r, s, q)
        assert 0.0 < priced < float(black_scholes_call(S, k, t, r, s, q)), f"K={k} q={q}"


def test_barrier_response_to_the_dividend_yield_is_not_monotone():
    """Raising `q` cuts the drift, and that pulls the price two ways at once.

    A lower drift makes the call less likely to finish in the money (value down)
    and the barrier less likely to be touched (value up). Which effect wins is a
    property of the contract, not a general rule, and the closed form has to
    reproduce both. On (K=95, B=130) the first effect dominates and the price
    falls monotonically in `q`; on (K=80, B=115), struck deep in the money with
    the barrier close by, the second effect wins over the first part of the
    ladder and the price rises before it falls.
    """
    ladder = (0.0, 0.02, 0.04, 0.06, 0.08)
    falling = [up_and_out_call_closed_form(100.0, 95.0, 130.0, 1.0, 0.05, 0.20, q)
               for q in ladder]
    assert all(b < a for a, b in zip(falling, falling[1:], strict=False)), falling

    humped = [up_and_out_call_closed_form(100.0, 80.0, 115.0, 1.5, 0.04, 0.18, q)
              for q in ladder]
    assert humped[1] > humped[0] and humped[-1] < humped[-2], humped


# ---------------------------------------------------------------------------
# Digitals
# ---------------------------------------------------------------------------

def test_digital_call_matches_a_tight_call_spread():
    """A digital is the limit of a call spread: `digital = -dC/dK`.

    We use a central difference in the STRIKE with h = 1e-4*K, which sits well
    inside the regime where O(h^2) truncation dominates and cancellation noise
    is still ~1e-13.
    """
    worst = 0.0
    for (S, k, t, r, s, q) in CONTRACTS:
        h = 1e-4 * k
        spread = float(black_scholes_call(S, k - h, t, r, s, q)
                       - black_scholes_call(S, k + h, t, r, s, q)) / (2 * h)
        digital = float(digital_call(S, k, t, r, s, q))
        worst = max(worst, abs(digital - spread))
        assert digital == pytest.approx(spread, abs=1e-7), f"K={k} q={q}"
    assert worst < 1e-7, f"worst digital vs call-spread error {worst:.2e}"


def test_digital_put_matches_a_tight_put_spread():
    for (S, k, t, r, s, q) in CONTRACTS:
        h = 1e-4 * k
        spread = float(black_scholes_put(S, k + h, t, r, s, q)
                       - black_scholes_put(S, k - h, t, r, s, q)) / (2 * h)
        assert float(digital_put(S, k, t, r, s, q)) == pytest.approx(spread, abs=1e-7)


def test_digitals_are_complementary():
    """Buying both pays `cash` whatever happens, so they must sum to its PV."""
    for (S, k, t, r, s, q) in CONTRACTS:
        total = float(digital_call(S, k, t, r, s, q, 250.0)
                      + digital_put(S, k, t, r, s, q, 250.0))
        assert total == pytest.approx(250.0 * np.exp(-r * t), rel=1e-14)


def test_digital_is_bounded_and_monotone_in_strike():
    """Price is a discounted probability, so it lives in [0, cash*exp(-rT)]."""
    disc = np.exp(-R * T)
    prices = [float(digital_call(S0, k, T, R, SIGMA)) for k in (60, 80, 100, 120, 150)]
    assert all(0.0 < p < disc for p in prices)
    assert all(b < a for a, b in zip(prices, prices[1:], strict=False))  # higher strike, less likely


# ---------------------------------------------------------------------------
# Put-call parity as an explicit invariant
# ---------------------------------------------------------------------------

def test_parity_residual_is_zero_to_machine_precision():
    """Parity is a replication identity: it holds for any sigma, any q, any smile."""
    worst = 0.0
    for (S, k, t, r, s, q) in CONTRACTS:
        residual = float(black_scholes_put_call_parity_residual(S, k, t, r, s, q))
        worst = max(worst, abs(residual))
        assert residual == pytest.approx(0.0, abs=1e-12), f"K={k} q={q}"
    assert worst < 1e-12, f"worst parity residual {worst:.2e}"


def test_parity_residual_is_zero_across_extreme_volatilities():
    """Including the wings, where the two CDFs are evaluated far into the tails."""
    for sigma in (0.01, 0.05, 0.2, 1.0, 3.0):
        for strike in (10.0, 100.0, 400.0):
            residual = float(
                black_scholes_put_call_parity_residual(S0, strike, T, R, sigma, 0.04))
            assert abs(residual) < 1e-11, f"sigma={sigma} K={strike}: {residual:.3e}"


# ---------------------------------------------------------------------------
# Second-order Greeks
# ---------------------------------------------------------------------------

def _fd_second_order(S, k, t, r, s, q, option):
    """Central differences of the FIRST-order Greeks, per second derivative.

    Step sizes follow the same reasoning as `test_greeks_match_finite_differences`:
    each of these is a central FIRST difference of an analytic first-order Greek,
    so h ~ eps^(1/3) in the differentiated variable is about right, and we use
    1e-4 in sigma, 1e-4 in T, and 1e-4 relative in S. The spot step matters more
    than it looks: at 1e-2 relative (h = 1 at S = 100) the O(h^2) truncation
    error in d(vega)/dS is 1.9e-3 relative, which would read exactly like a wrong
    formula.
    """
    hv, ht, hs = 1e-4, 1e-4, 1e-4 * S
    sig_up = black_scholes_greeks(S, k, t, r, s + hv, q, option)
    sig_dn = black_scholes_greeks(S, k, t, r, s - hv, q, option)
    tau_up = black_scholes_greeks(S, k, t + ht, r, s, q, option)
    tau_dn = black_scholes_greeks(S, k, t - ht, r, s, q, option)
    spot_up = black_scholes_greeks(S + hs, k, t, r, s, q, option)
    spot_dn = black_scholes_greeks(S - hs, k, t, r, s, q, option)
    return {
        "vanna_from_delta": (sig_up["delta"] - sig_dn["delta"]) / (2 * hv),
        "vanna_from_vega": (spot_up["vega"] - spot_dn["vega"]) / (2 * hs),
        "volga": (sig_up["vega"] - sig_dn["vega"]) / (2 * hv),
        # charm is d(delta)/dt and t runs the other way from T.
        "charm": -(tau_up["delta"] - tau_dn["delta"]) / (2 * ht),
    }


@pytest.mark.parametrize("option", ["call", "put"])
def test_second_order_greeks_match_finite_differences(option):
    """vanna, volga and charm against central differences of delta and vega."""
    for (S, k, t, r, s, q) in CONTRACTS:
        g = black_scholes_greeks(S, k, t, r, s, q, option, second_order=True)
        fd = _fd_second_order(S, k, t, r, s, q, option)
        tag = f"{option} S={S} K={k} T={t} q={q}"
        assert g["vanna"] == pytest.approx(fd["vanna_from_delta"], rel=1e-5, abs=1e-9), tag
        assert g["vanna"] == pytest.approx(fd["vanna_from_vega"], rel=1e-5, abs=1e-9), tag
        assert g["volga"] == pytest.approx(fd["volga"], rel=1e-5, abs=1e-9), tag
        assert g["charm"] == pytest.approx(fd["charm"], rel=1e-5, abs=1e-9), tag


def test_vanna_is_the_cross_derivative_both_ways():
    """d(vega)/dS and d(delta)/dsigma are the same mixed partial (Clairaut)."""
    for (S, k, t, r, s, q) in CONTRACTS:
        fd = _fd_second_order(S, k, t, r, s, q, "call")
        assert fd["vanna_from_delta"] == pytest.approx(fd["vanna_from_vega"], rel=1e-5)


def test_vanna_and_volga_coincide_for_calls_and_puts():
    """Parity is linear in S and free of sigma, so both cross-derivatives match."""
    for (S, k, t, r, s, q) in CONTRACTS:
        gc = black_scholes_greeks(S, k, t, r, s, q, "call", second_order=True)
        gp = black_scholes_greeks(S, k, t, r, s, q, "put", second_order=True)
        assert gc["vanna"] == pytest.approx(gp["vanna"], rel=1e-12)
        assert gc["volga"] == pytest.approx(gp["volga"], rel=1e-12)
        # Parity fixes charm too: delta_call - delta_put = exp(-q*T), whose
        # time derivative is q*exp(-q*T).
        assert gc["charm"] - gp["charm"] == pytest.approx(q * np.exp(-q * t), abs=1e-12)


def test_volga_vanishes_at_both_roots_and_is_negative_between_them():
    """volga = vega * d1*d2/sigma, so its sign is the sign of `d1*d2`.

    Both roots are available in closed form, which makes this an algebraic
    identity rather than a fitted number: `d1 = 0` at
    `K = S*exp((r - q + sigma^2/2)*T)` and `d2 = 0` at
    `K = S*exp((r - q - sigma^2/2)*T)`. Strictly between the two we have
    `d1 > 0 > d2`, so volga is negative there and positive outside; a position
    struck in that window is short volatility convexity.
    """
    k_d1 = S0 * np.exp((R + 0.5 * SIGMA**2) * T)
    k_d2 = S0 * np.exp((R - 0.5 * SIGMA**2) * T)
    assert k_d2 < k_d1

    def volga(strike):
        return black_scholes_greeks(S0, strike, T, R, SIGMA, option="call",
                                    second_order=True)["volga"]

    assert abs(volga(k_d1)) < 1e-10
    assert abs(volga(k_d2)) < 1e-10
    assert volga(0.5 * (k_d1 + k_d2)) < 0
    assert volga(k_d2 * 0.9) > 0
    assert volga(k_d1 * 1.1) > 0


def test_second_order_greeks_reject_a_bad_option_type():
    with pytest.raises(ValueError, match="call.*put"):
        black_scholes_greeks(S0, K, T, R, SIGMA, option="banana", second_order=True)


# ---------------------------------------------------------------------------
# Implied volatility
# ---------------------------------------------------------------------------

MONEYNESS = (0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0)
MATURITIES = (0.05, 0.25, 1.0, 2.0, 5.0)
TRUE_VOLS = (0.10, 0.20, 0.35, 0.60)


def _round_trip_grid(q: float):
    """Every (strike, maturity, volatility, option) combination on the test grid."""
    for t in MATURITIES:
        for m in MONEYNESS:
            for true_sigma in TRUE_VOLS:
                for option in ("call", "put"):
                    yield S0 * m, t, true_sigma, option, q


@pytest.mark.parametrize("q", [0.0, 0.04])
def test_implied_vol_price_round_trip(q):
    """price -> vol -> price to 1e-10, deep ITM and OTM included.

    We assert on the PRICE, not on the volatility, and that choice is the point.
    Deep in the wings the map from vol to price is nearly flat, so a price
    reproduced to 1e-10 can still correspond to a volatility that is wrong in the
    third decimal; claiming to recover sigma there would be dishonest. The
    inversion is only ever as well posed as vega makes it.

    `tol` is tightened from its 1e-8 default because the default is an absolute
    tolerance on the PRICE, so it would allow a 1e-8 residual by construction.
    """
    worst = 0.0
    n_solved = 0
    for strike, t, true_sigma, option, qq in _round_trip_grid(q):
        pricer = black_scholes_call if option == "call" else black_scholes_put
        price = float(pricer(S0, strike, t, R, true_sigma, qq))
        iv = implied_volatility(price, S0, strike, t, R, qq, option, tol=1e-12)
        assert np.isfinite(iv), f"no solution for K={strike} T={t} {option}"
        reprice = float(pricer(S0, strike, t, R, iv, qq))
        worst = max(worst, abs(reprice - price))
        n_solved += 1
        assert reprice == pytest.approx(price, abs=1e-10), (
            f"K={strike} T={t} sigma={true_sigma} {option} q={qq}")
    assert n_solved == len(MONEYNESS) * len(MATURITIES) * len(TRUE_VOLS) * 2
    assert worst < 1e-10, f"worst round-trip price error {worst:.2e}"


@pytest.mark.parametrize("q", [0.0, 0.04])
def test_implied_vol_recovers_sigma_where_vega_is_healthy(q):
    """Near the money the inversion is well conditioned, so sigma itself returns."""
    for t in MATURITIES:
        forward = S0 * np.exp((R - q) * t)
        for m in (0.9, 1.0, 1.1):
            for true_sigma in TRUE_VOLS:
                for option in ("call", "put"):
                    pricer = black_scholes_call if option == "call" else black_scholes_put
                    strike = forward * m
                    price = float(pricer(S0, strike, t, R, true_sigma, q))
                    iv = implied_volatility(price, S0, strike, t, R, q, option, tol=1e-12)
                    assert iv == pytest.approx(true_sigma, abs=1e-8), (
                        f"K/F={m} T={t} {option} q={q}")


def test_implied_vol_dividend_yield_actually_bites():
    """Ignoring `q` on a dividend-paying underlying biases the answer, visibly."""
    price = float(black_scholes_call(S0, S0, T, R, 0.25, 0.06))
    correct = implied_volatility(price, S0, S0, T, R, 0.06, tol=1e-12)
    ignored = implied_volatility(price, S0, S0, T, R, 0.0, tol=1e-12)
    assert correct == pytest.approx(0.25, abs=1e-8)
    assert abs(ignored - 0.25) > 0.01, "dropping q should shift the implied vol materially"


def test_diagnostics_report_the_route_and_the_iteration_count():
    price = float(black_scholes_call(S0, K, T, R, SIGMA))
    sigma, diag = implied_volatility(price, S0, K, T, R, return_diagnostics=True)
    assert sigma == pytest.approx(SIGMA, abs=1e-6)
    assert diag["route"] == "newton"
    assert diag["converged"] is True
    assert 0 < diag["iterations"] <= 10
    assert diag["iterations"] == diag["newton_iterations"] + diag["bisection_iterations"]
    assert diag["bisection_iterations"] == 0
    assert 0.0 < diag["sigma_initial"] < 5.0


def test_diagnostics_flag_the_bisection_route():
    """Force the fallback with a starting guess whose vega is degenerate."""
    price = float(black_scholes_call(S0, 200.0, T, R, 0.30))
    sigma, diag = implied_volatility(price, S0, 200.0, T, R, initial_guess=1e-4,
                                     return_diagnostics=True)
    assert diag["route"] == "bisection"
    assert diag["bisection_iterations"] > 0
    assert float(black_scholes_call(S0, 200.0, T, R, sigma)) == pytest.approx(price, abs=1e-8)


def test_diagnostics_on_a_price_with_no_solution():
    sigma, diag = implied_volatility(S0 * 2, S0, K, T, R, return_diagnostics=True)
    assert np.isnan(sigma)
    assert diag["route"] == "none"
    assert diag["converged"] is False
    assert diag["iterations"] == 0


def test_return_diagnostics_defaults_to_a_bare_float():
    """The flag is appended last and off by default, so old callers are unaffected."""
    price = float(black_scholes_call(S0, K, T, R, SIGMA))
    assert isinstance(implied_volatility(price, S0, K, T, R), float)


# ---------------------------------------------------------------------------
# The Brenner-Subrahmanyam starting guess
# ---------------------------------------------------------------------------

def _iteration_grid():
    """A moneyness x maturity x volatility grid, quoted relative to the FORWARD.

    Moneyness is measured against the forward rather than spot because that is
    where the Brenner-Subrahmanyam approximation is defined; laying the grid out
    in spot terms would put its own reference point at a different place for
    every maturity.
    """
    for q in (0.0, 0.03):
        for t in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0):
            forward = S0 * np.exp((R - q) * t)
            for m in (0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 1.7):
                for true_sigma in (0.10, 0.20, 0.35, 0.60):
                    for option in ("call", "put"):
                        yield forward * m, t, true_sigma, option, q, abs(np.log(1.0 / m))


def _mean_iterations(use_default_guess: bool, band: float | None = None):
    """Mean Newton iterations over the grid, optionally restricted to a band.

    The controlled comparison holds the wing fallback fixed and toggles only the
    Brenner-Subrahmanyam branch: passing the Manaster-Koehler point explicitly as
    `initial_guess` is exactly what the solver would have used had the branch not
    fired, so the two arms differ in one thing only.
    """
    newton, total, n_bisection = [], [], 0
    for strike, t, true_sigma, option, q, log_moneyness in _iteration_grid():
        if band is not None and log_moneyness > band:
            continue
        pricer = black_scholes_call if option == "call" else black_scholes_put
        price = float(pricer(S0, strike, t, R, true_sigma, q))
        guess = None if use_default_guess else _manaster_koehler_sigma(S0, strike, t, R, q)
        _, diag = implied_volatility(price, S0, strike, t, R, q, option,
                                     initial_guess=guess, return_diagnostics=True)
        newton.append(diag["newton_iterations"])
        total.append(diag["iterations"])
        n_bisection += diag["route"] == "bisection"
    return float(np.mean(newton)), float(np.mean(total)), n_bisection, len(newton)


def test_brenner_subrahmanyam_guess_reduces_mean_newton_iterations():
    """The measured claim in the module docstring, asserted rather than asserted-to.

    Two bands and the whole grid. The near-the-money reduction is the large one,
    because that is the only place the approximation is allowed to fire; the
    whole-grid figure is diluted by the wings and is reported for honesty rather
    than for effect.
    """
    for band, floor in ((0.05, 0.12), (0.10, 0.05), (None, 0.0)):
        without, _, _, n = _mean_iterations(False, band)
        with_bs, _, _, _ = _mean_iterations(True, band)
        assert n > 0
        assert with_bs < without, (
            f"band={band}: {with_bs:.3f} not below {without:.3f} over {n} contracts")
        assert (without - with_bs) / without >= floor, (
            f"band={band}: reduction {100 * (1 - with_bs / without):.2f}% "
            f"below the {100 * floor:.0f}% floor ({without:.3f} -> {with_bs:.3f})")


def test_gating_the_guess_removes_the_bisection_fallbacks():
    """Applying the approximation everywhere, as the module used to, is the failure.

    Deep out of the money the price tends to zero, so an ungated
    Brenner-Subrahmanyam start returns a volatility near zero, vega is degenerate
    there, and Newton hands over to bisection. Bisection costs 40 or more further
    price evaluations, so the ungated guess wins on Newton iterations only by not
    performing them.
    """
    ungated_newton, ungated_total, ungated_fallbacks = [], [], 0
    for strike, t, true_sigma, option, q, _lm in _iteration_grid():
        pricer = black_scholes_call if option == "call" else black_scholes_put
        price = float(pricer(S0, strike, t, R, true_sigma, q))
        guess = max(np.sqrt(2 * np.pi / t) * price / S0, 1e-3)  # the ungated form
        _, diag = implied_volatility(price, S0, strike, t, R, q, option,
                                     initial_guess=guess, return_diagnostics=True)
        ungated_newton.append(diag["newton_iterations"])
        ungated_total.append(diag["iterations"])
        ungated_fallbacks += diag["route"] == "bisection"

    _, gated_total, gated_fallbacks, n = _mean_iterations(True, None)
    assert n == len(ungated_total)
    assert ungated_fallbacks > 100, "expected the ungated guess to fall back often"
    assert gated_fallbacks == 0, f"gated guess still fell back {gated_fallbacks} times"
    assert gated_total < 0.8 * float(np.mean(ungated_total)), (
        f"total iterations {gated_total:.3f} vs ungated {np.mean(ungated_total):.3f}")


def test_manaster_koehler_point_maximises_vega():
    """The fallback is the vega-maximising volatility, which is why it is safe."""
    for (S, k, t, r, _s, q) in CONTRACTS:
        peak = _manaster_koehler_sigma(S, k, t, r, q)
        if not 0.05 < peak < 1.0:
            continue  # clipped, so it is no longer the stationary point
        v_peak = float(vega(S, k, t, r, peak, q))
        for delta in (-0.02, -0.005, 0.005, 0.02):
            assert float(vega(S, k, t, r, peak + delta, q)) <= v_peak * (1 + 1e-12)
