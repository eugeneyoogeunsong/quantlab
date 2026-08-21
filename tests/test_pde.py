# quantlab: Eugene (Yoogeun) Song (https://www.linkedin.com/in/yoogeunsong)
# Independent side project. MIT licensed; see LICENSE.

"""Finite-difference (PDE) pricing tests.

Same organising principle as `test_derivatives.py`: a numerical method earns
trust by reproducing something computed by different mathematics. Here the
independent routes are (i) the closed-form Black-Scholes price for the European
contracts, (ii) a high-step binomial lattice for the American put, whose free
boundary has no closed form at all, and (iii) no-arbitrage theorems that hold
whatever the implementation does, such as an American call on a non-dividend
payer being worth exactly its European twin.

One exception to the house rule against hardcoded expectations is deliberate and
confined to `test_zero_dividend_reproduces_frozen_prices`: those numbers were
captured from the module BEFORE the dividend yield was threaded through it, and
their job is to detect any change in the arithmetic, not to certify that the
arithmetic is right. Correctness is certified by the cross-checks below; the
frozen values are a regression lock, and the two roles are complementary.

We import from `quantlab.derivatives.finite_difference` rather than the package
root so these tests do not depend on the package `__init__` re-exporting the new
functions.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantlab.derivatives.analytic import (
    black_scholes_call,
    black_scholes_greeks,
    black_scholes_put,
)
from quantlab.derivatives.binomial import binomial_american
from quantlab.derivatives.finite_difference import (
    crank_nicolson_american,
    crank_nicolson_european,
    crank_nicolson_greeks,
    crank_nicolson_up_and_out,
)

# The standard contract, matching `test_derivatives.py` so figures are comparable.
S0, K, T, R, SIGMA = 100.0, 110.0, 1.0, 0.05, 0.20

# With S_max = 4*S0 = 400 and n_space a multiple of 100, both S0 and K sit exactly
# on grid nodes. That matters more than it looks: the price is read off by linear
# interpolation, which on a convex value function biases upwards by O(dS^2), and
# that bias has the opposite sign to the free-boundary discretisation error. On a
# grid where S0 falls between nodes the two partially cancel and the total error
# oscillates in sign under refinement. Aligning removes the interpolation term, so
# convergence-rate assertions become meaningful rather than lucky.
S_MAX_ALIGNED = 400.0


@pytest.fixture(scope="module")
def american_put_reference() -> float:
    """American put by a 10,000-step CRR lattice: the independent cross-check.

    The lattice value itself wobbles at the 1e-5 level with step count (11.973196,
    11.972840, 11.972845, 11.972861 at 2k, 5k, 10k, 20k steps), which sets the
    floor on any tolerance stated against it. Every tolerance below is at least
    thirty times that, so the reference noise is not what the tests are measuring.

    Called with keyword arguments only, so appending further parameters to
    `binomial_american` cannot silently change which contract we price.
    """
    return binomial_american(S0, K, T, R, SIGMA, n_steps=10_000, option="put")


# ---------------------------------------------------------------------------
# The dividend yield must be a pure extension: q = 0 changes nothing
# ---------------------------------------------------------------------------

# Captured from this module before `q` existed. Exact equality is the assertion:
# `q` enters as (r - q) in the convection coefficients, and r - 0.0 is r to the
# bit, so anything other than an identical float means the refactor moved
# arithmetic it had no business moving.
FROZEN = {
    "call_default": 6.038623331659073,
    "call_600": 6.040228115849091,
    "call_800": 6.039919820398451,
    "call_smax": 6.03444888724946,
    "put_default": 10.673860019925383,
    "put_800": 10.67515651377447,
    "put_odd": 19.00402802421083,
    "barrier_default": 5.151737464204198,
    "barrier_600": 5.151880251976694,
    "barrier_odd": 5.99752837511224,
}


def test_zero_dividend_reproduces_frozen_prices():
    """Bit-for-bit agreement with the pre-dividend implementation."""
    current = {
        "call_default": crank_nicolson_european(S0, K, T, R, SIGMA),
        "call_600": crank_nicolson_european(S0, K, T, R, SIGMA, 600, 600),
        "call_800": crank_nicolson_european(S0, K, T, R, SIGMA, 800, 800),
        "call_smax": crank_nicolson_european(S0, K, T, R, SIGMA, 300, 250, "call", 500.0),
        "put_default": crank_nicolson_european(S0, K, T, R, SIGMA, option="put"),
        "put_800": crank_nicolson_european(S0, K, T, R, SIGMA, 800, 800, option="put"),
        "put_odd": crank_nicolson_european(90.0, 105.0, 0.75, 0.03, 0.35, 257, 193, "put"),
        "barrier_default": crank_nicolson_up_and_out(100, 95, 130, T, R, SIGMA),
        "barrier_600": crank_nicolson_up_and_out(100, 95, 130, T, R, SIGMA, 600, 600),
        "barrier_odd": crank_nicolson_up_and_out(95.0, 90.0, 125.0, 0.5, 0.04, 0.25, 211, 167),
    }
    for name, expected in FROZEN.items():
        assert current[name] == expected, (
            f"{name}: {current[name]!r} != {expected!r}; the q=0 path is no longer "
            "bit-identical to the implementation these values were frozen from"
        )


def test_passing_q_zero_explicitly_is_identical_to_omitting_it():
    """The default and an explicit q=0.0 must be the same float, not merely close."""
    assert (crank_nicolson_european(S0, K, T, R, SIGMA, q=0.0)
            == crank_nicolson_european(S0, K, T, R, SIGMA))
    assert (crank_nicolson_european(S0, K, T, R, SIGMA, option="put", q=0.0)
            == crank_nicolson_european(S0, K, T, R, SIGMA, option="put"))
    assert (crank_nicolson_up_and_out(100, 95, 130, T, R, SIGMA, q=0.0)
            == crank_nicolson_up_and_out(100, 95, 130, T, R, SIGMA))
    assert (crank_nicolson_american(S0, K, T, R, SIGMA, 200, 200, q=0.0)
            == crank_nicolson_american(S0, K, T, R, SIGMA, 200, 200))
    assert (crank_nicolson_greeks(S0, K, T, R, SIGMA, 200, 200, q=0.0)
            == crank_nicolson_greeks(S0, K, T, R, SIGMA, 200, 200))


# ---------------------------------------------------------------------------
# European: the PDE against the closed form
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", [0.0, 0.03, 0.07])
@pytest.mark.parametrize("option", ["call", "put"])
def test_european_matches_black_scholes_with_dividends(option, q):
    """Crank-Nicolson vs Black-Scholes-Merton, to 5e-4 on an 800x800 grid.

    The worst case measured across these six combinations is 1.8e-4 on prices of
    order 3 to 15, i.e., about 2e-5 relative. The dividend convention is under
    test as much as the scheme is: q enters the drift as (r - q) but the
    discounting stays at r, and getting that pairing wrong shows up here as an
    error of order q*T*S, thousands of times the tolerance.
    """
    exact = float(black_scholes_call(S0, K, T, R, SIGMA, q) if option == "call"
                  else black_scholes_put(S0, K, T, R, SIGMA, q))
    price = crank_nicolson_european(S0, K, T, R, SIGMA, 800, 800, option=option, q=q)
    assert price == pytest.approx(exact, abs=5e-4)


def test_european_convergence_is_second_order_in_space():
    """Halving dS should quarter the error, and on an aligned grid it does.

    Measured errors against the closed form are 8.1e-3, 2.0e-3 and 5.1e-4 at
    n_space = 200, 400 and 800, i.e., ratios of 4.00 and 4.00. This is the
    strongest available statement of "refinement helps": not merely that the
    error falls, but that it falls at the rate the scheme claims.
    """
    exact = float(black_scholes_call(S0, K, T, R, SIGMA))
    errors = [abs(crank_nicolson_european(S0, K, T, R, SIGMA, n, n,
                                          S_max=S_MAX_ALIGNED) - exact)
              for n in (200, 400, 800)]
    assert errors[0] > errors[1] > errors[2]
    for coarse, fine in zip(errors, errors[1:], strict=False):
        assert 3.0 < coarse / fine < 5.0, f"rate {coarse / fine:.2f} is not O(dS^2)"


@pytest.mark.parametrize("option,sign", [("call", -1.0), ("put", 1.0)])
def test_dividend_yield_moves_european_prices_the_right_way(option, sign):
    """A dividend depresses the forward: calls fall, puts rise, monotonically."""
    prices = [crank_nicolson_european(S0, K, T, R, SIGMA, 300, 300, option=option, q=q)
              for q in (0.0, 0.02, 0.05, 0.09)]
    diffs = np.diff(prices) * sign
    assert np.all(diffs > 0.0), prices


def test_barrier_with_dividend_converges_to_the_vanilla_price():
    """Push the knockout out of reach and the barrier call must become a call.

    This is the only independent check available on the barrier pricer's dividend
    handling, since `up_and_out_call_closed_form` carries no q. With B = 600 the
    knockout is effectively unreachable and the gap to Black-Scholes-Merton is
    4.6e-4 to 5.1e-4 across q, all of it ordinary discretisation error on a
    domain stretched to six times spot.
    """
    for q in (0.0, 0.03, 0.06):
        vanilla = float(black_scholes_call(S0, K, T, R, SIGMA, q))
        barrier = crank_nicolson_up_and_out(S0, K, 600.0, T, R, SIGMA, 1200, 800, q=q)
        assert barrier == pytest.approx(vanilla, abs=2e-3)
        # And a reachable barrier must be worth strictly less than the vanilla.
        near = crank_nicolson_up_and_out(S0, K, 130.0, T, R, SIGMA, 400, 400, q=q)
        assert 0.0 < near < vanilla


# ---------------------------------------------------------------------------
# American: PSOR against the lattice, and against theory
# ---------------------------------------------------------------------------

def test_american_put_matches_high_step_binomial(american_put_reference):
    """THE cross-check: two unrelated methods on a problem with no closed form.

    The lattice enumerates early exercise node by node; PSOR solves a linear
    complementarity problem on a grid. Nothing is shared between them but the
    contract, so agreement is real evidence.

    Measured against the 10,000-step lattice: 1.8e-3 on the default 400x400 grid
    and 7.6e-4 on an aligned 800x200 grid, on a price of 11.97 (1.5e-4 and 6.3e-5
    relative). The tolerances below are roughly 1.5x the measured error, tight
    enough that a sign slip in the dividend term or a mis-set boundary would
    breach them by orders of magnitude.
    """
    default_grid = crank_nicolson_american(S0, K, T, R, SIGMA, 400, 400, option="put")
    assert default_grid == pytest.approx(american_put_reference, abs=3e-3)

    refined = crank_nicolson_american(S0, K, T, R, SIGMA, 800, 200, option="put",
                                      S_max=S_MAX_ALIGNED)
    assert refined == pytest.approx(american_put_reference, abs=1.2e-3)


def test_american_put_error_falls_under_grid_refinement(american_put_reference):
    """Errors of 3.0e-2, 9.8e-3 and 2.7e-3, i.e., ratios of 3.1 and 3.7.

    The rate climbs towards 4 rather than sitting on it, which is what a
    free-boundary problem should do: the exercise boundary is only ever located
    to within one grid spacing, so it contributes a term that decays a little
    slower than the second-order interior.
    """
    errors = [abs(crank_nicolson_american(S0, K, T, R, SIGMA, n, n, option="put",
                                          S_max=S_MAX_ALIGNED) - american_put_reference)
              for n in (100, 200, 400)]
    assert errors[0] > errors[1] > errors[2]
    for coarse, fine in zip(errors, errors[1:], strict=False):
        assert coarse / fine > 2.5, f"refinement gained only {coarse / fine:.2f}x"


def test_american_call_without_dividends_equals_european_call():
    """A theorem, not an approximation: with q = 0 early exercise is never optimal.

    Exercising a call early throws away the remaining time value AND the interest
    on the strike, so the continuation value strictly dominates and the constraint
    V >= S - K never binds. PSOR must therefore converge to the solution of the
    very same linear system the European solver factorises, and it does: the two
    prices agree to 2.7e-15, which is round-off, not agreement to a tolerance.
    """
    american = crank_nicolson_american(S0, K, T, R, SIGMA, 400, 400, option="call")
    european = crank_nicolson_european(S0, K, T, R, SIGMA, 400, 400, option="call")
    assert american == pytest.approx(european, abs=1e-10)


@pytest.mark.parametrize("q", [0.05, 0.08])
def test_american_call_with_dividends_exceeds_european_call(q):
    """Once q > 0 the early-exercise right is worth something, and it shows.

    A dividend yield transfers value out of the option and into the stock, so at
    some spot it becomes optimal to exercise and collect it. The premium measured
    here is 3.3e-2 at q = 5% and 1.6e-1 at q = 8%: small, but hundreds of times
    the discretisation error, and strictly increasing in q as it must be.
    """
    american = crank_nicolson_american(S0, K, T, R, SIGMA, 300, 300, option="call", q=q)
    european = crank_nicolson_european(S0, K, T, R, SIGMA, 300, 300, option="call", q=q)
    assert american > european + 1e-3


def test_american_put_carries_an_early_exercise_premium():
    """The American put is worth strictly more even with no dividends.

    Exercising frees the strike in cash, which then earns r; that is a real
    benefit no amount of time value compensates for once the option is deep
    enough in the money. The premium here is 1.30 on a European value of 10.67.
    """
    american = crank_nicolson_american(S0, K, T, R, SIGMA, 400, 400, option="put")
    european = crank_nicolson_european(S0, K, T, R, SIGMA, 400, 400, option="put")
    assert american > european + 1.0


@pytest.mark.parametrize("option,q", [("put", 0.0), ("put", 0.05), ("call", 0.08)])
def test_american_value_never_falls_below_intrinsic(option, q):
    """The constraint PSOR projects onto, checked on the returned surface.

    If this fails the projection is not being applied where it should be, which
    is the one failure mode that makes an American pricer silently wrong rather
    than merely inaccurate.
    """
    _, S, V = crank_nicolson_american(S0, K, T, R, SIGMA, 300, 300, option=option,
                                      q=q, return_grid=True)
    intrinsic = np.maximum(S - K, 0.0) if option == "call" else np.maximum(K - S, 0.0)
    assert np.min(V - intrinsic) >= -1e-12


# ---------------------------------------------------------------------------
# The free boundary
# ---------------------------------------------------------------------------

def test_put_free_boundary_is_monotone_in_time():
    """S*(tau) falls as time to expiry grows, at every one of 400 timesteps.

    The economics are unambiguous: more time remaining means more optionality
    given up by exercising, so the spot must fall further before exercising wins.
    At expiry the boundary is the strike itself, and it decays from there (107.8
    at the first step to 89.1 at tau = 1). A boundary that wandered back up would
    mean the projection was leaking value between timesteps.
    """
    _, tau, S_free = crank_nicolson_american(S0, K, T, R, SIGMA, 400, 400,
                                             option="put", return_boundary=True)
    assert tau[0] == pytest.approx(T / 400) and tau[-1] == pytest.approx(T)
    assert not np.isnan(S_free).any()
    assert S_free[0] < K              # the boundary has already left the strike
    assert np.all(np.diff(S_free) <= 1e-9), "put exercise boundary is not decreasing"
    assert S_free[-1] < S_free[0]     # and it genuinely moves, rather than sticking


def test_call_free_boundary_rises_with_time_and_vanishes_without_dividends():
    """Mirror image for the call, plus the degenerate case done honestly.

    For a dividend-paying call the boundary moves the other way (112.9 at the
    first step to 139.3 at tau = 1): more time remaining means the spot must be
    higher before surrendering the option is worthwhile. With q = 0 there is no
    exercise region at all, and the right answer is to report NaN rather than to
    invent a boundary at the edge of the grid.
    """
    _, _, S_free = crank_nicolson_american(S0, K, T, R, SIGMA, 300, 300,
                                           option="call", q=0.08, return_boundary=True)
    assert not np.isnan(S_free).any()
    assert S_free[0] > K
    assert np.all(np.diff(S_free) >= -1e-9), "call exercise boundary is not increasing"
    assert S_free[-1] > S_free[0]

    _, _, none_at_all = crank_nicolson_american(S0, K, T, R, SIGMA, 300, 300,
                                                option="call", return_boundary=True)
    assert np.all(np.isnan(none_at_all))


def test_psor_answer_does_not_depend_on_the_relaxation_parameter():
    """omega changes how fast we converge, never what we converge to.

    Across omega in {0.9, 1.2, 1.6} the price spread is 7.6e-8, of the order of
    `tol` itself. This is the evidence that PSOR is solving the linear
    complementarity problem rather than reporting wherever the relaxation
    happened to stop, and it is why the free boundary can be read off the answer.
    """
    prices = [crank_nicolson_american(S0, K, T, R, SIGMA, 200, 200, option="put",
                                      omega=w) for w in (0.9, 1.2, 1.6)]
    assert max(prices) - min(prices) < 1e-6


def test_tightening_the_psor_tolerance_does_not_move_the_answer():
    """A hundred-fold tighter tolerance buys sweeps, not accuracy."""
    loose = crank_nicolson_american(S0, K, T, R, SIGMA, 200, 200, option="put", tol=1e-6)
    tight = crank_nicolson_american(S0, K, T, R, SIGMA, 200, 200, option="put", tol=1e-10)
    assert loose == pytest.approx(tight, abs=1e-5)


# ---------------------------------------------------------------------------
# Greeks off the grid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", [0.0, 0.04])
@pytest.mark.parametrize("option", ["call", "put"])
def test_pde_greeks_match_black_scholes(option, q):
    """All five Greeks from one PDE solve, against the closed form.

    Tolerance is 1e-3 relative OR 6e-3 absolute, whichever is looser, because the
    five are not equally accurate and pretending otherwise would hide the
    interesting part. Measured worst cases on a 400x400 grid: delta 9.6e-5,
    gamma 3.0e-6, vega 4.8e-3 (1.2e-4 relative), rho 3.4e-3 (8.8e-5 relative),
    theta 2.5e-3. Theta is the loose one because a two-slice difference is only
    O(dt); the other four are at 1e-4 relative or better.
    """
    fd = crank_nicolson_greeks(S0, K, T, R, SIGMA, 400, 400, option=option, q=q)
    bs = black_scholes_greeks(S0, K, T, R, SIGMA, q, option=option)
    assert set(fd) == set(bs), "the PDE Greeks must be a drop-in for the analytic ones"
    for name in bs:
        assert fd[name] == pytest.approx(bs[name], rel=1e-3, abs=6e-3), name


@pytest.mark.parametrize("q", [0.0, 0.04])
def test_pde_greeks_satisfy_the_parity_identities(q):
    """Put-call parity is linear in S, so the second-order Greeks must coincide.

    C - P = S*exp(-qT) - K*exp(-rT), so differentiating twice in S kills both
    terms: gamma and vega are shared, and the deltas differ by exactly exp(-qT).
    Because both sides come off grids built by the same code, these hold far
    tighter than the comparison with Black-Scholes does (4e-15 on gamma, 2e-11 on
    vega, 1e-10 on the delta identity), which makes them a sharp check on the
    differencing itself rather than on the scheme's accuracy.
    """
    call = crank_nicolson_greeks(S0, K, T, R, SIGMA, 400, 400, option="call", q=q)
    put = crank_nicolson_greeks(S0, K, T, R, SIGMA, 400, 400, option="put", q=q)
    assert call["delta"] - put["delta"] == pytest.approx(float(np.exp(-q * T)), abs=1e-8)
    assert call["gamma"] == pytest.approx(put["gamma"], rel=1e-10)
    assert call["vega"] == pytest.approx(put["vega"], rel=1e-8)


def test_pde_greeks_have_the_right_signs():
    """Cheap, but it catches a transposed dictionary key instantly."""
    call = crank_nicolson_greeks(S0, K, T, R, SIGMA, 300, 300, option="call")
    put = crank_nicolson_greeks(S0, K, T, R, SIGMA, 300, 300, option="put")
    assert 0.0 < call["delta"] < 1.0
    assert -1.0 < put["delta"] < 0.0
    assert call["gamma"] > 0.0 and put["gamma"] > 0.0
    assert call["vega"] > 0.0 and put["vega"] > 0.0
    assert call["theta"] < 0.0 and put["theta"] < 0.0      # long options decay
    assert call["rho"] > 0.0 and put["rho"] < 0.0


def test_theta_improves_when_the_time_grid_is_refined():
    """Theta is the O(dt) Greek, so more timesteps must visibly help.

    Errors against the closed form of 1.9e-3, 7.0e-4 and 4.1e-4 at n_time = 400,
    1600 and 6400: falling, but only about as fast as sqrt(n_time) suggests,
    which is the honest advertisement for the two-slice difference.
    """
    exact = black_scholes_greeks(S0, K, T, R, SIGMA, option="call")["theta"]
    errors = [abs(crank_nicolson_greeks(S0, K, T, R, SIGMA, 400, nt,
                                        option="call")["theta"] - exact)
              for nt in (400, 1600)]
    assert errors[1] < errors[0]


# ---------------------------------------------------------------------------
# The theta family, and refusing to run outside it
# ---------------------------------------------------------------------------

def test_fully_implicit_scheme_still_converges():
    """theta_scheme = 0 is first order but unconditionally stable, so it must work.

    Error against Black-Scholes is 3.0e-3 for the European call and 5.4e-3 for
    the American put against the lattice, both roughly twice the Crank-Nicolson
    figure on the same grid: the expected cost of dropping from O(dt^2) to O(dt).
    """
    exact = float(black_scholes_call(S0, K, T, R, SIGMA))
    implicit = crank_nicolson_european(S0, K, T, R, SIGMA, 400, 400, theta_scheme=0.0)
    assert implicit == pytest.approx(exact, abs=1e-2)

    american = crank_nicolson_american(S0, K, T, R, SIGMA, 200, 200, option="put",
                                       theta_scheme=0.0)
    assert american == pytest.approx(11.9728, abs=2e-2)


def test_explicit_scheme_is_rejected_outside_its_stability_limit():
    """The condition is dt <= 1/(sigma^2 * n_space^2 * (2*theta_scheme - 1)).

    At theta_scheme = 1, sigma = 0.2 and n_space = 400 that caps dt at 1.6e-4,
    i.e., more than 6400 steps for a one-year contract. Asking for 400 must fail
    loudly and say so, because an unstable run does not degrade gracefully: it
    overflows to NaN inside a few dozen steps and surfaces as an opaque error
    from the banded solver several layers down.
    """
    with pytest.raises(ValueError, match="unstable"):
        crank_nicolson_european(S0, K, T, R, SIGMA, 400, 400, theta_scheme=1.0)
    with pytest.raises(ValueError, match="unstable"):
        crank_nicolson_american(S0, K, T, R, SIGMA, 400, 400, theta_scheme=1.0)

    # Satisfy the condition and the same scheme is fine, if wasteful.
    stable = crank_nicolson_european(S0, K, T, R, SIGMA, 100, 3000, theta_scheme=1.0)
    assert stable == pytest.approx(float(black_scholes_call(S0, K, T, R, SIGMA)), abs=5e-2)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

def test_bad_arguments_raise_with_a_useful_message():
    with pytest.raises(ValueError, match="option must be"):
        crank_nicolson_american(S0, K, T, R, SIGMA, option="banana")
    with pytest.raises(ValueError, match="option must be"):
        crank_nicolson_greeks(S0, K, T, R, SIGMA, option="banana")
    with pytest.raises(ValueError, match="omega"):
        crank_nicolson_american(S0, K, T, R, SIGMA, 50, 50, omega=2.0)
    with pytest.raises(ValueError, match="omega"):
        crank_nicolson_american(S0, K, T, R, SIGMA, 50, 50, omega=0.0)
    with pytest.raises(ValueError, match="theta_scheme"):
        crank_nicolson_european(S0, K, T, R, SIGMA, 50, 50, theta_scheme=1.5)
    with pytest.raises(ValueError, match="n_space"):
        crank_nicolson_american(S0, K, T, R, SIGMA, 1, 50)


def test_psor_iteration_cap_is_enforced_rather_than_silently_ignored():
    """Starving the iteration must raise, not return a half-relaxed surface.

    A cap that is quietly hit is the worst outcome available: the answer looks
    like a price and is not one. At tol = 1e-8 and omega = 1.2 the worst timestep
    on a 400x400 grid needs 12 sweeps, so 3 is comfortably too few.
    """
    with pytest.raises(RuntimeError, match="failed to converge"):
        crank_nicolson_american(S0, K, T, R, SIGMA, 400, 400, option="put", max_iter=3)


def test_return_flags_hand_back_the_right_shapes():
    """The four return signatures, since callers unpack them positionally."""
    n_space, n_time = 100, 60
    price = crank_nicolson_american(S0, K, T, R, SIGMA, n_space, n_time)
    assert isinstance(price, float)

    p, S, V = crank_nicolson_american(S0, K, T, R, SIGMA, n_space, n_time, return_grid=True)
    assert p == price and S.shape == V.shape == (n_space + 1,)

    p, tau, S_free = crank_nicolson_american(S0, K, T, R, SIGMA, n_space, n_time,
                                             return_boundary=True)
    assert p == price and tau.shape == S_free.shape == (n_time,)

    p, S, V, tau, S_free = crank_nicolson_american(S0, K, T, R, SIGMA, n_space, n_time,
                                                   return_grid=True, return_boundary=True)
    assert p == price and V.shape == (n_space + 1,) and S_free.shape == (n_time,)
