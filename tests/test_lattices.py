# quantlab: Eugene (Yoogeun) Song (https://www.linkedin.com/in/yoogeunsong)
# Independent side project. MIT licensed; see LICENSE.

"""Lattice tests: parameterisations, extrapolation, tree Greeks, dividend yield.

`test_derivatives.py` establishes that the CRR lattice agrees with Black-Scholes;
this file is about the choices made INSIDE the lattice, so the questions are
different: does a given parameterisation converge, at what order, and in which
direction; does extrapolating two trees help or hurt; do the Greeks read off the
nodes match the closed form; and does a dividend yield enter the drift without
leaking into the discounting.

The references are the same as everywhere else in the suite: closed-form prices,
moment conditions, and no-arbitrage identities, none of which depend on this
implementation. The one exception is `test_default_arguments_reproduce_pre_dividend_prices`,
which pins literal numbers captured from the pre-dividend code on purpose; it is
a regression lock, not a correctness claim, and it is the only place a hardcoded
figure is allowed to stand in for a derivation.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantlab.derivatives.analytic import (
    black_scholes_call,
    black_scholes_greeks,
    black_scholes_put,
)
from quantlab.derivatives.binomial import (
    TREE_METHODS,
    _tree_params,
    binomial_american,
    binomial_european,
    binomial_greeks,
    binomial_richardson,
    binomial_tree_full,
    binomial_up_and_out,
    trinomial_up_and_out,
)

# The reference contract, shared with test_derivatives.py so the two files
# describe the same option.
S0, K, T, R, SIGMA = 100.0, 110.0, 1.0, 0.05, 0.20
Q = 0.04                      # the dividend yield used throughout below

BS_CALL = float(black_scholes_call(S0, K, T, R, SIGMA))
BS_PUT = float(black_scholes_put(S0, K, T, R, SIGMA))


def _err(price: float, exact: float) -> float:
    return abs(price - exact)


# ---------------------------------------------------------------------------
# The dividend-yield convention: q=0.0 must change nothing at all
# ---------------------------------------------------------------------------

def test_default_arguments_reproduce_pre_dividend_prices():
    """Bit-for-bit regression lock on every pricer in the module.

    The literals below were captured from the implementation as it stood before
    `q`, `method`, and the shared backward-induction engine existed. Equality is
    exact rather than approximate on purpose: `r - 0.0` is exactly `r` in IEEE
    754, so appending a dividend yield with a zero default cannot perturb the
    arithmetic by even one unit in the last place, and if it has, the refactor
    changed something it was not meant to touch.
    """
    assert binomial_european(S0, K, T, R, SIGMA, n_steps=50) == 6.060695891870758
    assert binomial_european(S0, K, T, R, SIGMA, n_steps=501) == 6.03967386746722
    assert binomial_european(S0, K, T, R, SIGMA, n_steps=500,
                             option="put") == 10.677455962649905
    assert binomial_european(S0, K, T, R, SIGMA) == 6.042219267573525

    assert binomial_american(S0, K, T, R, SIGMA, n_steps=500,
                             option="put") == 11.974393469523198
    assert binomial_american(90.0, 100.0, 0.75, 0.03, 0.35, n_steps=37,
                             option="put") == 16.02316806100407
    assert binomial_american(S0, K, T, R, SIGMA, n_steps=800,
                             option="call") == 6.040890418014769

    assert binomial_up_and_out(100.0, 95.0, 130.0, T, R, SIGMA) == 5.035596342518989
    assert binomial_up_and_out(100.0, 95.0, 130.0, T, R, SIGMA,
                               bgk_adjust=False) == 5.376486213653802
    assert trinomial_up_and_out(100.0, 95.0, 130.0, T, R, SIGMA) == 5.147700941418595

    stock, value, exercise = binomial_tree_full(S0, K, T, R, SIGMA, n_steps=6, option="put")
    assert float(value[0, 0]) == 12.065701292622604
    assert float(stock[6, 6]) == 61.26889167823924
    assert int(exercise.sum()) == 9


def test_explicit_zero_yield_equals_the_default():
    """Passing q=0.0 by hand and omitting it must be the same computation."""
    for option in ("call", "put"):
        assert (binomial_european(S0, K, T, R, SIGMA, option=option, q=0.0)
                == binomial_european(S0, K, T, R, SIGMA, option=option))
        assert (binomial_american(S0, K, T, R, SIGMA, option=option, q=0.0)
                == binomial_american(S0, K, T, R, SIGMA, option=option))
    assert (binomial_european(S0, K, T, R, SIGMA, method="crr", q=0.0)
            == binomial_european(S0, K, T, R, SIGMA))
    assert (trinomial_up_and_out(100.0, 95.0, 130.0, T, R, SIGMA, q=0.0)
            == trinomial_up_and_out(100.0, 95.0, 130.0, T, R, SIGMA))


# ---------------------------------------------------------------------------
# The four parameterisations
# ---------------------------------------------------------------------------

def test_tian_matches_the_first_three_moments():
    """Tian's construction is defined by three moment conditions, so check all three.

    This is the independent check on the parameterisation: no price is involved,
    only whether (u, d, p) reproduce E[S/S0], E[(S/S0)^2] and E[(S/S0)^3] of the
    lognormal over one step. CRR and JR satisfy the first two and miss the third;
    Tian is the one that must hit all three to machine precision.
    """
    dt, u, d, p, _ = _tree_params("tian", T, R, 0.0, SIGMA, 101, S0, K)
    M, V = np.exp(R * dt), np.exp(SIGMA**2 * dt)
    assert p * u + (1 - p) * d == pytest.approx(M, abs=1e-15)
    assert p * u**2 + (1 - p) * d**2 == pytest.approx(M**2 * V, abs=1e-15)
    assert p * u**3 + (1 - p) * d**3 == pytest.approx(M**3 * V**3, abs=1e-15)


def test_every_tree_reproduces_the_forward_exactly():
    """p*u + (1-p)*d = exp((r-q)*dt) is the martingale condition the tree is built on.

    CRR, Tian and LR all define p by that equation, so it holds to the last bit.
    JR instead fixes p = 1/2 and absorbs the drift into u and d, which matches the
    forward only to O(dt^2) per step: the residual measured here is 1.3e-8 at
    n = 101 and 5.2e-11 at n = 1601, a factor of 252 for a 15.9x change in n,
    which is the quadratic rate. That defect is the whole reason JR's put-call
    parity residual is 1.7e-5 rather than floating-point noise.
    """
    for method in ("crr", "tian", "lr"):
        dt, u, d, p, _ = _tree_params(method, T, R, Q, SIGMA, 401, S0, K)
        assert p * u + (1 - p) * d == pytest.approx(np.exp((R - Q) * dt), abs=1e-15), method

    residuals = []
    for n in (101, 1601):
        dt, u, d, p, _ = _tree_params("jr", T, R, Q, SIGMA, n, S0, K)
        residuals.append(abs(p * u + (1 - p) * d - np.exp((R - Q) * dt)))
    assert residuals[0] < 1e-7
    assert residuals[1] < residuals[0] / 100, f"JR forward error not O(dt^2): {residuals}"


def test_per_step_variance_matches_to_first_order():
    """Var[log S] per step must approach sigma^2*dt, and does so at O(dt).

    JR is exact by construction, since its log-moves are nu +/- sigma*sqrt(dt)
    with p = 1/2. The other three carry a relative error that falls like dt:
    quadrupling n cuts it by ~4 (measured 2.2e-4 -> 5.6e-5 for CRR, 7.3e-4 ->
    1.8e-4 for Tian, 5.7e-3 -> 1.4e-3 for LR, going from n = 101 to n = 401).
    LR's constant is the largest of the three, which is worth knowing: it buys its
    accuracy by centring the terminal distribution, not by being a better
    one-step approximation.
    """
    def variance_error(method, n):
        _, u, d, p, _ = _tree_params(method, T, R, 0.0, SIGMA, n, S0, K)
        log_u, log_d = np.log(u), np.log(d)
        var = p * log_u**2 + (1 - p) * log_d**2 - (p * log_u + (1 - p) * log_d) ** 2
        return abs(var - SIGMA**2 * (T / n)) / (SIGMA**2 * (T / n))

    assert variance_error("jr", 401) < 1e-12
    for method in ("crr", "tian", "lr"):
        coarse, fine = variance_error(method, 101), variance_error(method, 401)
        assert coarse < 1e-2, f"{method}: {coarse:.2e}"
        assert fine < coarse / 3.0, f"{method} not O(dt): {coarse:.2e} -> {fine:.2e}"


def test_jarrow_rudd_is_the_equal_probability_tree():
    _, _, _, p, _ = _tree_params("jr", T, R, Q, SIGMA, 77, S0, K)
    assert p == 0.5


def test_leisen_reimer_rounds_an_even_step_count_up():
    """Even n is not a rounding nuisance for LR; it destroys the method.

    The Peizer-Pratt inversion centres the strike in the terminal distribution,
    and with an even number of steps a terminal node lands exactly on the strike,
    i.e. exactly on the payoff kink the construction exists to avoid. Measured on
    this contract, an even-step LR tree used as-is is out by 4.9e-3 at n = 802
    against 5.9e-7 at n = 801. We therefore round up rather than raise, so that
    `binomial_richardson` (which prices at n and 2n, and 2n is never odd) stays
    usable; the test is that the round-up actually happened.
    """
    _, _, _, _, n_used = _tree_params("lr", T, R, 0.0, SIGMA, 100, S0, K)
    assert n_used == 101
    assert (binomial_european(S0, K, T, R, SIGMA, n_steps=100, method="lr")
            == binomial_european(S0, K, T, R, SIGMA, n_steps=101, method="lr"))
    # Odd counts pass through untouched.
    assert _tree_params("lr", T, R, 0.0, SIGMA, 101, S0, K)[4] == 101


def test_every_method_converges_to_black_scholes():
    """All four parameterisations must reach the closed form, call and put.

    Coarse and fine are 21 and 2001 steps, far enough apart that the oscillation
    of the first-order trees cannot fake an improvement. Measured fine errors:
    CRR 1.5e-4, JR 1.9e-4, Tian 3.3e-4, LR 9.5e-8.
    """
    for method in TREE_METHODS:
        for option, exact in (("call", BS_CALL), ("put", BS_PUT)):
            coarse = _err(binomial_european(S0, K, T, R, SIGMA, n_steps=21, option=option,
                                            method=method), exact)
            fine = _err(binomial_european(S0, K, T, R, SIGMA, n_steps=2001, option=option,
                                          method=method), exact)
            assert fine < coarse, f"{method} {option}: {fine:.2e} not better than {coarse:.2e}"
            assert fine < 1e-3, f"{method} {option}: {fine:.2e} above tolerance 1e-3"


def test_alternative_trees_price_where_crr_refuses():
    """With sigma small relative to r, CRR has no admissible lattice and the others do.

    CRR ties d to 1/u, so the up/down range is symmetric in log-space and can fail
    to straddle the forward: at r = 0.25, sigma = 0.05, n = 10 the implied p is
    1.2965, which is not a probability, and the pricer correctly refuses rather
    than returning a number. JR, Tian and LR all put the drift into u and d
    instead, so they remain admissible and price the contract to 5.4e-6, 2.4e-7
    and 6.1e-8 respectively. This is a second, independent reason to keep the
    alternatives around.
    """
    with pytest.raises(ValueError, match="outside"):
        binomial_european(100.0, 100.0, 1.0, 0.25, 0.05, n_steps=10)
    exact = float(black_scholes_call(100.0, 100.0, 1.0, 0.25, 0.05))
    for method in ("jr", "tian", "lr"):
        price = binomial_european(100.0, 100.0, 1.0, 0.25, 0.05, n_steps=10, method=method)
        assert price == pytest.approx(exact, abs=1e-4), method


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="method"):
        binomial_european(S0, K, T, R, SIGMA, method="binomial-ish")


# ---------------------------------------------------------------------------
# The point of Leisen-Reimer: order, not just accuracy
# ---------------------------------------------------------------------------

def test_leisen_reimer_beats_crr_at_equal_steps():
    """Stated benchmark: European call, S0=100, K=110, T=1, r=0.05, sigma=0.20.

    Measured error ratio CRR/LR at n = 101, 201, 401, 801: 245, 378, 747, 1043.
    The assertions ask for a factor of 50 at every rung and 100 at the headline
    n = 801, which leaves an order of magnitude of margin whilst still failing
    loudly if the LR tree ever silently degrades to first order.
    """
    ratios = {}
    for n in (101, 201, 401, 801):
        crr = _err(binomial_european(S0, K, T, R, SIGMA, n_steps=n, method="crr"), BS_CALL)
        lr = _err(binomial_european(S0, K, T, R, SIGMA, n_steps=n, method="lr"), BS_CALL)
        ratios[n] = crr / lr
        assert ratios[n] > 50, f"n={n}: LR only {ratios[n]:.0f}x better than CRR"
    assert ratios[801] > 100, f"benchmark n=801 ratio {ratios[801]:.0f}, expected ~1043"


def test_leisen_reimer_error_is_monotone_and_crr_is_not():
    """The property worth paying for: an error that always shrinks.

    Over n = 51, 101, 201, 401, 801, 1601 the LR error falls at every doubling
    (1.4e-4 -> 3.7e-5 -> 9.3e-6 -> 2.4e-6 -> 5.9e-7 -> 1.5e-7, i.e. ratios
    converging on 4, which is O(1/n^2)), whilst the CRR error rises twice
    (2.0e-3 -> 9.0e-3 at the first doubling, and 6.2e-4 -> 1.2e-3 at the last).
    Doubling the work on a CRR tree therefore buys no guarantee at all.
    """
    ladder = [51, 101, 201, 401, 801, 1601]
    lr = [_err(binomial_european(S0, K, T, R, SIGMA, n_steps=n, method="lr"), BS_CALL)
          for n in ladder]
    crr = [_err(binomial_european(S0, K, T, R, SIGMA, n_steps=n, method="crr"), BS_CALL)
           for n in ladder]

    assert all(b < a for a, b in zip(lr, lr[1:], strict=False)), f"LR not monotone: {lr}"
    ratios = [a / b for a, b in zip(lr, lr[1:], strict=False)]
    assert all(3.5 < ratio < 4.5 for ratio in ratios), f"LR order is not 2: {ratios}"

    assert any(b > a for a, b in zip(crr, crr[1:], strict=False)), f"CRR unexpectedly monotone: {crr}"


# ---------------------------------------------------------------------------
# Richardson extrapolation
# ---------------------------------------------------------------------------

def test_richardson_improves_leisen_reimer():
    """2*P(2n) - P(n) beats P(n) at every step count, by the expected factor.

    LR's leading error is O(1/n^2) whilst the classical two-point form is built
    for O(1/n), so it halves the term rather than annihilating it: the measured
    gain is 1.99, 2.00, 2.00, 2.00 at n = 51, 101, 201, 401. We assert a gain
    above 1.5, which passes only if the underlying error really is smooth and
    single-signed.
    """
    for n in (51, 101, 201, 401):
        base = _err(binomial_european(S0, K, T, R, SIGMA, n_steps=n, method="lr"), BS_CALL)
        extrap = _err(binomial_richardson(S0, K, T, R, SIGMA, n_steps=n, method="lr"), BS_CALL)
        assert extrap < base / 1.5, f"n={n}: gain only {base / extrap:.2f}x"


def test_richardson_is_unreliable_on_crr():
    """The counter-demonstration, and the reason the docstring warns about it.

    CRR's error alternates in sign as the strike drifts across the terminal
    nodes, so differencing two step counts amplifies the oscillation instead of
    cancelling a trend. Extrapolation makes the answer WORSE in five of the eight
    step counts below, and at n = 51 it is 12.8x worse than the tree it was built
    from. Contrast `test_richardson_improves_leisen_reimer`, where the same
    formula on the same contract never once loses.
    """
    worse = 0
    for n in (25, 51, 75, 101, 151, 201, 251, 401):
        base = _err(binomial_european(S0, K, T, R, SIGMA, n_steps=n, method="crr"), BS_CALL)
        extrap = _err(binomial_richardson(S0, K, T, R, SIGMA, n_steps=n, method="crr"),
                      BS_CALL)
        worse += extrap > base
    assert worse >= 3, f"CRR extrapolation degraded only {worse}/8 step counts"

    base_51 = _err(binomial_european(S0, K, T, R, SIGMA, n_steps=51, method="crr"), BS_CALL)
    extrap_51 = _err(binomial_richardson(S0, K, T, R, SIGMA, n_steps=51, method="crr"),
                     BS_CALL)
    assert extrap_51 > 5 * base_51, f"expected ~12.8x worse, got {extrap_51 / base_51:.1f}x"


def test_richardson_converges_to_the_closed_form():
    """Whatever it does to the error constant, the extrapolant must still be a price."""
    assert binomial_richardson(S0, K, T, R, SIGMA, n_steps=201) == pytest.approx(BS_CALL,
                                                                                 abs=1e-5)
    assert binomial_richardson(S0, K, T, R, SIGMA, n_steps=201,
                               option="put") == pytest.approx(BS_PUT, abs=1e-5)


# ---------------------------------------------------------------------------
# No-arbitrage identities, which hold whatever the parameterisation
# ---------------------------------------------------------------------------

def test_american_put_at_least_european_put():
    """Early exercise is an option, never an obligation, so it cannot subtract value.

    Checked for every parameterisation and with and without a dividend yield. The
    early-exercise premium measured here is ~1.30 at q = 0 and ~0.39 at q = 0.04:
    a yield lifts the European put towards the American one, because the drift it
    removes from the forward is precisely what made waiting expensive.
    """
    for q in (0.0, Q):
        for method in TREE_METHODS:
            american = binomial_american(S0, K, T, R, SIGMA, n_steps=501, option="put",
                                         method=method, q=q)
            european = binomial_european(S0, K, T, R, SIGMA, n_steps=501, option="put",
                                         method=method, q=q)
            assert american >= european, f"{method}, q={q}"
            assert american >= max(K - S0, 0.0) - 1e-9


def test_european_put_call_parity_holds_on_every_tree():
    """C - P = S*exp(-qT) - K*exp(-rT), a model-free identity the lattice must respect.

    CRR, Tian and LR are exact discrete martingales (p is defined by the forward
    condition), so parity holds at the lattice level and the residual is pure
    floating point, ~3e-12. JR fixes p = 1/2 instead and matches the forward only
    to O(dt^2) per step, leaving a residual of 1.7e-5 at n = 801: real, small, and
    a property of the parameterisation rather than a bug. Testing both at the same
    loose tolerance would hide exactly that distinction, so they get separate ones.
    """
    for q in (0.0, Q):
        parity = S0 * np.exp(-q * T) - K * np.exp(-R * T)
        for method in TREE_METHODS:
            call = binomial_european(S0, K, T, R, SIGMA, n_steps=801, option="call",
                                     method=method, q=q)
            put = binomial_european(S0, K, T, R, SIGMA, n_steps=801, option="put",
                                    method=method, q=q)
            tol = 1e-4 if method == "jr" else 1e-9
            assert call - put == pytest.approx(parity, abs=tol), f"{method}, q={q}"


def test_american_call_early_exercise_appears_only_with_a_yield():
    """Without dividends the American call is worth its European twin; with them it is not.

    The classical result (Shreve I, Ch. 4) is that exercising a call early throws
    away time value and the interest on the strike, so it is never optimal. A
    dividend yield changes the trade: at q = 0.08 the holder of a call struck at
    90 gives up 8% a year on the underlying by waiting, and the American price
    (11.92) rises above the European one (10.93).
    """
    american = binomial_american(S0, K, T, R, SIGMA, n_steps=501, option="call")
    european = binomial_european(S0, K, T, R, SIGMA, n_steps=501, option="call")
    assert american == pytest.approx(european, rel=1e-9)

    american_q = binomial_american(S0, 90.0, T, R, SIGMA, n_steps=501, option="call", q=0.08)
    european_q = binomial_european(S0, 90.0, T, R, SIGMA, n_steps=501, option="call", q=0.08)
    assert american_q > european_q + 0.5


# ---------------------------------------------------------------------------
# Tree Greeks
# ---------------------------------------------------------------------------

def test_binomial_greeks_match_black_scholes():
    """All five Greeks off an LR tree at n = 501, against the closed form.

    Stated tolerances (absolute): delta 1e-3, gamma 1e-4, vega 1e-3, theta 2e-2,
    rho 1e-3. Measured on the call: 3.5e-4, 1.2e-5, 9.1e-6, 4.5e-3, 5.5e-6; on
    the put: 3.5e-4, 1.2e-5, 9.1e-6, 4.0e-3, 5.6e-6. Theta gets the loosest bound
    because its node construction is a one-sided difference over 2*dt and so
    carries an O(dt) truncation the others do not.
    """
    tolerances = {"delta": 1e-3, "gamma": 1e-4, "vega": 1e-3, "theta": 2e-2, "rho": 1e-3}
    for option in ("call", "put"):
        exact = black_scholes_greeks(S0, K, T, R, SIGMA, 0.0, option=option)
        tree = binomial_greeks(S0, K, T, R, SIGMA, n_steps=501, option=option, method="lr")
        assert set(tree) == set(exact)
        for name, tol in tolerances.items():
            assert tree[name] == pytest.approx(exact[name], abs=tol), f"{option} {name}"


def test_binomial_greeks_with_a_dividend_yield():
    """The same agreement once the drift is (r - q); measured worst error 5.9e-3 on theta."""
    exact = black_scholes_greeks(S0, K, T, R, SIGMA, Q, option="call")
    tree = binomial_greeks(S0, K, T, R, SIGMA, n_steps=501, option="call", method="lr", q=Q)
    for name, tol in {"delta": 1e-3, "gamma": 1e-4, "vega": 1e-3,
                      "theta": 2e-2, "rho": 1e-3}.items():
        assert tree[name] == pytest.approx(exact[name], abs=tol), name
    # A yield must push the call's delta down relative to the same contract without one.
    assert tree["delta"] < binomial_greeks(S0, K, T, R, SIGMA, n_steps=501,
                                           option="call", method="lr")["delta"]


def test_theta_survives_a_drifting_middle_node():
    """The step-2 middle node sits at S0*u*d, which equals S0 only for CRR.

    JR and LR both drift it (by 1.2e-2 and 3.8e-2 in spot here), and a
    displacement of order dt divided by 2*dt contaminates theta at O(1): the
    uncorrected figures are -4.56 and -1.62 against a true -5.90. The pricer
    subtracts delta*(S_ud - S0) before dividing, so all four trees must land on
    the same theta; this test fails outright if that correction is ever removed.
    """
    exact = black_scholes_greeks(S0, K, T, R, SIGMA, 0.0, option="call")["theta"]
    for method in TREE_METHODS:
        theta = binomial_greeks(S0, K, T, R, SIGMA, n_steps=501, option="call",
                                method=method)["theta"]
        assert theta == pytest.approx(exact, abs=2e-2), method


def test_crr_greeks_are_weak_on_vega():
    """The documented failure mode, pinned so it cannot be forgotten.

    A vega bump differences the lattice's OWN error as well as the price, and
    CRR's error swings with sigma: at n = 501 its vega is out by 0.57 (1.4%)
    where LR's is out by 9.1e-6. Delta and gamma are unaffected, since they are
    read off one tree rather than differenced across two.
    """
    exact = black_scholes_greeks(S0, K, T, R, SIGMA, 0.0, option="call")
    crr = binomial_greeks(S0, K, T, R, SIGMA, n_steps=501, option="call", method="crr")
    lr = binomial_greeks(S0, K, T, R, SIGMA, n_steps=501, option="call", method="lr")

    assert _err(crr["vega"], exact["vega"]) > 0.1
    assert _err(crr["vega"], exact["vega"]) > 100 * _err(lr["vega"], exact["vega"])
    # Delta and gamma cost nothing extra and are equally good on both trees.
    assert crr["delta"] == pytest.approx(exact["delta"], abs=1e-3)
    assert crr["gamma"] == pytest.approx(exact["gamma"], abs=1e-4)


def test_binomial_greeks_reject_a_tree_with_no_step_two():
    with pytest.raises(ValueError, match="n_steps"):
        binomial_greeks(S0, K, T, R, SIGMA, n_steps=1)


def test_american_put_greeks_are_signed_correctly():
    """No closed form to check against, so check the signs and the delta by bumping."""
    greeks = binomial_greeks(S0, K, T, R, SIGMA, n_steps=401, option="put", american=True)
    assert -1.0 < greeks["delta"] < 0.0
    assert greeks["gamma"] > 0.0
    assert greeks["vega"] > 0.0
    bump = 0.5
    fd_delta = (binomial_american(S0 + bump, K, T, R, SIGMA, n_steps=401, option="put")
                - binomial_american(S0 - bump, K, T, R, SIGMA, n_steps=401,
                                    option="put")) / (2 * bump)
    assert greeks["delta"] == pytest.approx(fd_delta, abs=5e-3)


# ---------------------------------------------------------------------------
# Dividend yield
# ---------------------------------------------------------------------------

def test_dividend_yield_lowers_a_call_and_raises_a_put():
    """A yield drags the forward down, so calls lose and puts gain. Monotonically.

    Measured at n = 501 on the reference contract: the call falls 6.040 -> 4.429
    and the put rises 10.675 -> 12.986 as q goes 0 -> 0.04.
    """
    for method in TREE_METHODS:
        calls = [binomial_european(S0, K, T, R, SIGMA, n_steps=501, option="call",
                                   method=method, q=q) for q in (0.0, 0.02, 0.04, 0.08)]
        puts = [binomial_european(S0, K, T, R, SIGMA, n_steps=501, option="put",
                                  method=method, q=q) for q in (0.0, 0.02, 0.04, 0.08)]
        assert all(b < a for a, b in zip(calls, calls[1:], strict=False)), f"{method}: {calls}"
        assert all(b > a for a, b in zip(puts, puts[1:], strict=False)), f"{method}: {puts}"


def test_binomial_with_yield_converges_to_black_scholes_with_yield():
    """The cross-check that the yield went into the drift and NOT the discounting.

    Getting the convention wrong (discounting at r - q) would shift the price by
    roughly q*T*V, i.e. ~0.2 here, which no step count would ever remove. Measured
    LR errors against the closed form at n = 51, 201, 801: 1.1e-4, 7.1e-6, 4.5e-7.
    """
    exact_call = float(black_scholes_call(S0, K, T, R, SIGMA, Q))
    exact_put = float(black_scholes_put(S0, K, T, R, SIGMA, Q))

    errors = [_err(binomial_european(S0, K, T, R, SIGMA, n_steps=n, method="lr", q=Q),
                   exact_call) for n in (51, 201, 801)]
    assert all(b < a for a, b in zip(errors, errors[1:], strict=False)), f"not converging: {errors}"
    assert errors[-1] < 1e-5, f"final error {errors[-1]:.2e}"

    for method in TREE_METHODS:
        assert binomial_european(S0, K, T, R, SIGMA, n_steps=2001, method=method,
                                 q=Q) == pytest.approx(exact_call, abs=5e-3), method
        assert binomial_european(S0, K, T, R, SIGMA, n_steps=2001, option="put",
                                 method=method, q=Q) == pytest.approx(exact_put,
                                                                      abs=5e-3), method


def test_barrier_lattices_accept_a_dividend_yield():
    """Both barrier routes take q, and it can only make an up-and-out CALL cheaper here.

    A yield lowers the forward, which cuts the call payoff; it also makes the
    upper barrier less likely to be touched, which works the other way. On this
    contract the payoff effect wins on both lattices (5.149 -> 4.622 and 5.030 ->
    4.523 at n = 1000).

    The two lattices do NOT agree closely, and are not asked to: the ~0.11 gap is
    the known binomial barrier bias documented in `binomial_up_and_out`, not
    something the yield introduced. The check that matters is that the gap does
    not WIDEN once q > 0 (measured 0.119 -> 0.100), i.e. the yield enters both
    lattices the same way.
    """
    plain_tri = trinomial_up_and_out(100.0, 95.0, 130.0, T, R, SIGMA, n_steps=1000)
    yield_tri = trinomial_up_and_out(100.0, 95.0, 130.0, T, R, SIGMA, n_steps=1000, q=Q)
    plain_bin = binomial_up_and_out(100.0, 95.0, 130.0, T, R, SIGMA, n_steps=1000)
    yield_bin = binomial_up_and_out(100.0, 95.0, 130.0, T, R, SIGMA, n_steps=1000, q=Q)
    assert yield_tri < plain_tri
    assert yield_bin < plain_bin
    assert abs(yield_tri - yield_bin) <= abs(plain_tri - plain_bin) + 1e-9


def test_far_barrier_with_yield_recovers_the_vanilla_call():
    """As B recedes the knockout becomes unreachable, so the price must be BS with q.

    This is the only fully independent check available on the barrier lattices
    once q > 0 (the barrier closed form in `analytic.py` carries no yield), and it
    exercises the drift and the discounting separately: get either wrong and the
    limit misses.
    """
    exact = float(black_scholes_call(100.0, 95.0, T, R, SIGMA, Q))
    far = trinomial_up_and_out(100.0, 95.0, 1e5, T, R, SIGMA, n_steps=800, q=Q)
    assert far == pytest.approx(exact, rel=2e-3), f"{far:.4f} vs {exact:.4f}"
