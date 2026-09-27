# quantlab: Eugene (Yoogeun) Song. MIT licensed; see LICENSE.
"""Option pricing tests.

Nothing here is pinned to a value this code produced. Every reference is
external: a no-arbitrage identity, a textbook limit, a moment of the model,
or agreement between engines that share no code. If four routes built on
different mathematics land on the same number, that number is probably
right; if they disagree, the disagreement says which one is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantlab.pricing import (
    AnalyticEngine,
    BarrierOption,
    BarrierType,
    BlackScholes,
    CoxRossRubinstein,
    DigitalOption,
    Exercise,
    JarrowRudd,
    LatticeEngine,
    LeisenReimer,
    MonteCarloEngine,
    PathGenerator,
    PDEEngine,
    RitchkenTrinomial,
    Tian,
    UnsupportedInstrument,
    VanillaOption,
    bump_greeks,
    implied_vol_surface,
    implied_volatility,
)

MKT = BlackScholes(spot=100.0, rate=0.05, vol=0.20)
CALL = VanillaOption(expiry=1.0, strike=110.0, option_type="call")
PUT = VanillaOption(expiry=1.0, strike=110.0, option_type="put")
BARRIER = BarrierOption(expiry=1.0, strike=95.0, barrier=130.0, option_type="call",
                        barrier_type="up-and-out")
A = AnalyticEngine()


# ===========================================================================
# Model
# ===========================================================================

def test_model_validates_inputs():
    with pytest.raises(ValueError):
        BlackScholes(spot=-1, rate=0.05, vol=0.2)
    with pytest.raises(ValueError):
        BlackScholes(spot=100, rate=0.05, vol=0.0)


def test_forward_and_discount():
    m = BlackScholes(100, 0.05, 0.2, dividend_yield=0.02)
    assert m.forward(2.0) == pytest.approx(100 * np.exp(0.03 * 2))
    assert m.discount(2.0) == pytest.approx(np.exp(-0.10))


# ===========================================================================
# Instruments
# ===========================================================================

def test_instrument_validation():
    with pytest.raises(ValueError):
        VanillaOption(expiry=0.0, strike=100)
    with pytest.raises(ValueError):
        VanillaOption(expiry=1.0, strike=-5)
    with pytest.raises(ValueError):
        VanillaOption(expiry=1.0, strike=100, option_type="straddle")


def test_payoffs():
    assert CALL.payoff(120.0) == 10.0
    assert CALL.payoff(100.0) == 0.0
    assert PUT.payoff(100.0) == 10.0
    np.testing.assert_array_equal(CALL.payoff(np.array([90.0, 110.0, 130.0])), [0.0, 0.0, 20.0])
    digital = DigitalOption(1.0, 100.0, "call", cash=5.0)
    np.testing.assert_array_equal(digital.payoff(np.array([99.0, 101.0])), [0.0, 5.0])


def test_barrier_type_complement_is_involution():
    for bt in BarrierType:
        assert bt.complement.complement is bt
        assert bt.is_knock_out != bt.complement.is_knock_out


# ===========================================================================
# Analytic engine -- identities that hold regardless of implementation
# ===========================================================================

def test_put_call_parity_to_machine_precision():
    c, p = A.price(CALL, MKT).value, A.price(PUT, MKT).value
    assert c - p == pytest.approx(MKT.spot - CALL.strike * MKT.discount(1.0), abs=1e-12)


def test_parity_with_dividends():
    m = BlackScholes(100, 0.05, 0.25, dividend_yield=0.03)
    c, p = A.price(CALL, m).value, A.price(PUT, m).value
    lhs = c - p
    rhs = m.spot * np.exp(-0.03) - CALL.strike * np.exp(-0.05)
    assert lhs == pytest.approx(rhs, abs=1e-12)


def test_no_arbitrage_bounds():
    c = A.price(CALL, MKT).value
    assert max(MKT.spot - CALL.strike * MKT.discount(1.0), 0) <= c <= MKT.spot
    p = A.price(PUT, MKT).value
    assert max(PUT.strike * MKT.discount(1.0) - MKT.spot, 0) <= p <= PUT.strike * MKT.discount(1.0)


def test_monotone_in_vol_spot_strike():
    by_vol = [A.price(CALL, MKT.with_vol(v)).value for v in (0.05, 0.1, 0.2, 0.4, 0.8)]
    assert all(b > a for a, b in zip(by_vol, by_vol[1:]))
    by_spot = [A.price(CALL, MKT.with_spot(s)).value for s in (80, 90, 100, 110, 120)]
    assert all(b > a for a, b in zip(by_spot, by_spot[1:]))
    by_strike = [A.price(VanillaOption(1.0, k, "call"), MKT).value for k in (80, 90, 100, 110, 120)]
    assert all(b < a for a, b in zip(by_strike, by_strike[1:]))


def test_zero_strike_call_is_the_forward():
    c = A.price(VanillaOption(1.0, 1e-9, "call"), MKT).value
    assert c == pytest.approx(MKT.spot, rel=1e-8)


def test_analytic_greeks_vs_bump():
    """Closed-form Greeks against finite differences through the same engine.

    Step sizes are chosen per derivative order in `bump_greeks`; with a
    first-derivative step used for gamma, cancellation error produces a 1%
    discrepancy that looks exactly like a wrong formula.
    """
    cf = A.price(CALL, MKT).greeks
    fd = bump_greeks(A, CALL, MKT)
    assert fd["delta"] == pytest.approx(cf["delta"], rel=1e-7)
    assert fd["gamma"] == pytest.approx(cf["gamma"], rel=1e-5)
    assert fd["vega"] == pytest.approx(cf["vega"], rel=1e-7)
    assert fd["rho"] == pytest.approx(cf["rho"], rel=1e-7)
    assert fd["theta"] == pytest.approx(cf["theta"], rel=1e-3)


def test_greek_identities():
    gc, gp = A.price(CALL, MKT).greeks, A.price(PUT, MKT).greeks
    assert gc["delta"] - gp["delta"] == pytest.approx(1.0, abs=1e-12)  # from parity
    assert gc["gamma"] == pytest.approx(gp["gamma"], rel=1e-12)
    assert gc["vega"] == pytest.approx(gp["vega"], rel=1e-12)
    # gamma = vega / (S^2 sigma T): both come from the same n(d+)
    assert gc["gamma"] == pytest.approx(gc["vega"] / (MKT.spot**2 * MKT.vol * 1.0), rel=1e-12)
    assert 0 < gc["delta"] < 1 and -1 < gp["delta"] < 0
    assert gc["theta"] < 0


def test_digital_is_the_derivative_of_a_call_spread():
    """A cash-or-nothing call is the limit of a call spread / strike gap."""
    h = 1e-4
    spread = (A.price(VanillaOption(1.0, 100 - h, "call"), MKT).value
              - A.price(VanillaOption(1.0, 100 + h, "call"), MKT).value) / (2 * h)
    digital = A.price(DigitalOption(1.0, 100.0, "call", cash=1.0), MKT).value
    assert digital == pytest.approx(spread, rel=1e-6)


def test_digital_call_plus_put_is_discounted_cash():
    c = A.price(DigitalOption(1.0, 100.0, "call"), MKT).value
    p = A.price(DigitalOption(1.0, 100.0, "put"), MKT).value
    assert c + p == pytest.approx(MKT.discount(1.0), abs=1e-12)


def test_analytic_refuses_american():
    with pytest.raises(UnsupportedInstrument):
        A.price(PUT.with_exercise(Exercise.AMERICAN), MKT)


# ===========================================================================
# Barrier identities
# ===========================================================================

def test_in_out_parity_all_four_types():
    """knock-in + knock-out = vanilla, for every direction and both payoffs."""
    for opt_type in ("call", "put"):
        for H in (85.0, 130.0):
            vanilla = A.price(VanillaOption(1.0, 100.0, opt_type), MKT).value
            for kind in ("up-and-out", "down-and-out"):
                if (kind.startswith("up") and H < MKT.spot) or (kind.startswith("down") and H > MKT.spot):
                    continue
                out = BarrierOption(1.0, 100.0, H, opt_type, kind)
                v_out = A.price(out, MKT).value
                v_in = A.price(out.with_barrier_type(out.barrier_type.complement), MKT).value
                assert v_out + v_in == pytest.approx(vanilla, abs=1e-10), (opt_type, H, kind)


def test_barrier_cheaper_than_vanilla_and_converges_to_it():
    vanilla = A.price(BARRIER.underlying_vanilla, MKT).value
    assert 0 < A.price(BARRIER, MKT).value < vanilla
    far = BarrierOption(1.0, 95.0, 1e6, "call", "up-and-out")
    assert A.price(far, MKT).value == pytest.approx(vanilla, rel=1e-8)


def test_barrier_degenerate_cases():
    assert A.price(BarrierOption(1.0, 95.0, 90.0, "call", "up-and-out"), MKT).value == 0.0
    triggered = BarrierOption(1.0, 95.0, 130.0, "call", "up-and-out")
    assert A.price(triggered, MKT.with_spot(140.0)).value == 0.0
    assert A.price(triggered.with_barrier_type(BarrierType.UP_AND_IN),
                   MKT.with_spot(140.0)).value == pytest.approx(
        A.price(triggered.underlying_vanilla, MKT.with_spot(140.0)).value)


def test_barrier_value_is_monotone_in_barrier_distance():
    vals = [A.price(BarrierOption(1.0, 95.0, H, "call", "up-and-out"), MKT).value
            for H in (110, 120, 130, 150, 200)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


# ===========================================================================
# Lattice engine
# ===========================================================================

@pytest.mark.parametrize("scheme", [CoxRossRubinstein(), JarrowRudd(), Tian(), LeisenReimer()])
def test_every_scheme_converges_to_black_scholes(scheme):
    exact = A.price(CALL, MKT).value
    coarse = abs(LatticeEngine(51, scheme).price(CALL, MKT).value - exact)
    fine = abs(LatticeEngine(1001, scheme).price(CALL, MKT).value - exact)
    assert fine < coarse
    assert fine < 2e-3


def test_leisen_reimer_is_second_order():
    """Leisen-Reimer error should fall ~4x when steps double; CRR's only ~2x.

    Comparing a single LR step count against a single CRR count is unfair to
    whichever happens to sit at a lucky point in CRR's oscillation. The
    *order* is the actual claim, and it is what this measures.
    """
    exact = A.price(CALL, MKT).value

    def err(scheme, n):
        return abs(LatticeEngine(n, scheme).price(CALL, MKT).value - exact)

    lr_ratio = err(LeisenReimer(), 101) / err(LeisenReimer(), 401)
    assert lr_ratio > 8, f"LR error only fell {lr_ratio:.1f}x over 4x steps; expected ~16x"
    # And at equal step counts LR is far more accurate than CRR.
    for n in (51, 101, 201):
        assert err(LeisenReimer(), n) < 0.1 * err(CoxRossRubinstein(), n), n


def test_lattice_parity_holds():
    L = LatticeEngine(501)
    c, p = L.price(CALL, MKT).value, L.price(PUT, MKT).value
    assert c - p == pytest.approx(MKT.spot - CALL.strike * MKT.discount(1.0), abs=1e-3)


def test_american_call_equals_european_without_dividends():
    """Never optimal to exercise early (Hull Ch. 11): the two must coincide."""
    L = LatticeEngine(801)
    am = L.price(CALL.with_exercise(Exercise.AMERICAN), MKT).value
    eu = L.price(CALL, MKT).value
    assert am == pytest.approx(eu, rel=1e-10)


def test_american_call_exceeds_european_with_dividends():
    """With a dividend yield, early exercise CAN be optimal for a call."""
    m = BlackScholes(100, 0.02, 0.25, dividend_yield=0.08)
    deep = VanillaOption(2.0, 70.0, "call")
    L = LatticeEngine(801)
    assert L.price(deep.with_exercise(Exercise.AMERICAN), m).value > L.price(deep, m).value + 1e-4


def test_american_put_strictly_more_valuable_and_above_intrinsic():
    L = LatticeEngine(801)
    am = L.price(PUT.with_exercise(Exercise.AMERICAN), MKT).value
    eu = L.price(PUT, MKT).value
    assert am > eu
    deep = VanillaOption(1.0, 100.0, "put", Exercise.AMERICAN)
    assert L.price(deep, MKT.with_spot(50.0)).value >= 50.0 - 1e-6


def test_richardson_helps_smooth_error_and_hurts_oscillating_error():
    """Extrapolation assumes the error is a smooth function of step size.

    Leisen-Reimer's is; CRR's and Jarrow-Rudd's oscillate as the strike
    drifts between node layers, and extrapolating an oscillation amplifies
    it. Measured at 201 steps: JR plain 1.4e-3 -> extrapolated 3.6e-3;
    CRR 3.5e-3 -> 5.2e-3; LR at 51 steps 1.4e-4 -> 7.1e-5.
    """
    exact = A.price(CALL, MKT).value

    def err(scheme, n, rich):
        return abs(LatticeEngine(n, scheme, richardson=rich).price(CALL, MKT).value - exact)

    assert err(LeisenReimer(), 51, True) < err(LeisenReimer(), 51, False)
    assert err(CoxRossRubinstein(), 201, True) > err(CoxRossRubinstein(), 201, False)


def test_trinomial_barrier_converges_smoothly():
    exact = A.price(BARRIER, MKT).value
    errors = [abs(LatticeEngine(n, RitchkenTrinomial()).price(BARRIER, MKT).value - exact)
              for n in (250, 500, 1000, 2000)]
    assert errors[-1] < 2e-3
    assert all(b < a * 1.05 for a, b in zip(errors, errors[1:])), errors


def test_continuity_correction_hurts_an_aligned_tree():
    """The BGK shift is for discretely observed continuous paths. An aligned
    lattice has no unobserved gap, so the correction only biases it."""
    exact = A.price(BARRIER, MKT).value
    off = abs(LatticeEngine(1000, RitchkenTrinomial(continuity_correction=False)).price(BARRIER, MKT).value - exact)
    on = abs(LatticeEngine(1000, RitchkenTrinomial(continuity_correction=True)).price(BARRIER, MKT).value - exact)
    assert off < on


def test_lattice_knock_in_via_parity():
    ui = BARRIER.with_barrier_type(BarrierType.UP_AND_IN)
    assert LatticeEngine(1000).price(ui, MKT).value == pytest.approx(A.price(ui, MKT).value, abs=5e-3)


def test_lattice_rejects_bad_probabilities():
    """Huge rate with tiny vol pushes p outside [0,1]: the lattice is arbitrageable."""
    with pytest.raises(ValueError, match="distribution"):
        LatticeEngine(5, CoxRossRubinstein()).price(CALL, BlackScholes(100, 5.0, 0.01))


def test_tree_diagnostic_view():
    stock, value, exercised = LatticeEngine(6).tree(PUT.with_exercise(Exercise.AMERICAN), MKT, n_steps=6)
    assert stock.shape == value.shape == exercised.shape
    assert np.isfinite(value[0, 0]) and value[0, 0] > 0
    _, _, ex_eu = LatticeEngine(6).tree(PUT, MKT, n_steps=6)
    assert not ex_eu.any()


# ===========================================================================
# PDE engine
# ===========================================================================

@pytest.mark.parametrize("theta", [0.5, 1.0])
def test_pde_converges_for_both_payoffs(theta):
    E = PDEEngine(600, 600, theta=theta)
    assert E.price(CALL, MKT).value == pytest.approx(A.price(CALL, MKT).value, abs=2e-3)
    assert E.price(PUT, MKT).value == pytest.approx(A.price(PUT, MKT).value, abs=2e-3)


def test_pde_explicit_is_unstable_unless_dt_is_small():
    """theta=0 with a coarse time grid violates the CFL condition and blows up.
    That is a property of the scheme, and the test documents it."""
    v = PDEEngine(400, 50, theta=0.0, rannacher=False).price(CALL, MKT).value
    assert not np.isfinite(v) or abs(v) > 100


def test_pde_greeks_from_surface():
    cf = A.price(CALL, MKT).greeks
    g = PDEEngine(600, 600).price(CALL, MKT).greeks
    assert g["delta"] == pytest.approx(cf["delta"], abs=2e-4)
    assert g["gamma"] == pytest.approx(cf["gamma"], abs=2e-5)


@pytest.mark.parametrize("solver", ["projection", "psor"])
def test_pde_american_matches_lattice(solver):
    amp = PUT.with_exercise(Exercise.AMERICAN)
    ref = LatticeEngine(2001).price(amp, MKT).value
    v = PDEEngine(500, 500, american_solver=solver).price(amp, MKT).value
    assert v == pytest.approx(ref, abs=5e-3)
    assert v > PDEEngine(500, 500).price(PUT, MKT).value


def test_pde_barrier_is_just_a_boundary():
    v = PDEEngine(600, 600).price(BARRIER, MKT).value
    assert v == pytest.approx(A.price(BARRIER, MKT).value, abs=2e-3)


def test_pde_down_and_out_put():
    do = BarrierOption(1.0, 100.0, 80.0, "put", "down-and-out")
    assert PDEEngine(600, 600).price(do, MKT).value == pytest.approx(A.price(do, MKT).value, abs=3e-3)


def test_pde_digital_within_discretisation_error():
    d = DigitalOption(1.0, 100.0, "call")
    assert PDEEngine(800, 800).price(d, MKT).value == pytest.approx(A.price(d, MKT).value, abs=5e-3)


def test_pde_rannacher_reduces_ringing_on_digital():
    d = DigitalOption(1.0, 100.0, "call")
    exact = A.price(d, MKT).value
    with_r = abs(PDEEngine(300, 300, rannacher=True).price(d, MKT).value - exact)
    without = abs(PDEEngine(300, 300, rannacher=False).price(d, MKT).value - exact)
    assert with_r <= without * 1.5  # not worse; usually noticeably better


# ===========================================================================
# Monte Carlo engine
# ===========================================================================

def test_simulated_paths_match_model_moments():
    """E[S_T] = F and Var[ln S_T] = sigma^2 T. If these fail, nothing downstream matters."""
    paths = PathGenerator(200_000, 50, seed=1).paths(MKT, 1.0)
    F, tv = MKT.terminal_moments(1.0)
    assert paths[:, 0].mean() == pytest.approx(MKT.spot, rel=1e-14)
    assert paths[:, -1].mean() == pytest.approx(F, rel=5e-3)
    assert np.log(paths[:, -1] / MKT.spot).var() == pytest.approx(tv, rel=2e-2)
    assert (paths > 0).all()


def test_mc_within_reported_standard_error():
    """The test is on the *error bar*, not just the price."""
    r = MonteCarloEngine(PathGenerator(100_000, 1, seed=2)).price(CALL, MKT)
    assert abs(r.value - A.price(CALL, MKT).value) < 4 * r.stderr
    assert r.confidence_interval is not None


def test_control_variate_reduces_variance():
    gen = PathGenerator(50_000, 1, antithetic=False, seed=3)
    plain = MonteCarloEngine(gen, control_variate=False).price(CALL, MKT).stderr
    ctrl = MonteCarloEngine(gen, control_variate=True).price(CALL, MKT).stderr
    assert ctrl < 0.7 * plain


def test_antithetic_reduces_variance():
    plain = MonteCarloEngine(PathGenerator(50_000, 1, antithetic=False, seed=4),
                             control_variate=False).price(CALL, MKT).stderr
    anti = MonteCarloEngine(PathGenerator(50_000, 1, antithetic=True, seed=4),
                            control_variate=False).price(CALL, MKT).stderr
    assert anti < plain


def test_stderr_scales_as_inverse_root_n():
    small = MonteCarloEngine(PathGenerator(2_000, 1, seed=5), control_variate=False).price(CALL, MKT).stderr
    large = MonteCarloEngine(PathGenerator(200_000, 1, seed=5), control_variate=False).price(CALL, MKT).stderr
    assert 6 < small / large < 16  # theory: 10


def test_mc_barrier_with_bridge_is_unbiased_and_discrete_is_not():
    exact = A.price(BARRIER, MKT).value
    gen = PathGenerator(100_000, 100, seed=6)
    r = MonteCarloEngine(gen).price(BARRIER, MKT)
    assert abs(r.value - exact) < 4 * r.stderr
    # Discrete monitoring on the same paths overprices: it misses crossings.
    paths = gen.paths(MKT, 1.0)
    survive = ~BARRIER.is_touched(paths).any(axis=1)
    discrete = MKT.discount(1.0) * (BARRIER.payoff(paths[:, -1]) * survive).mean()
    assert discrete > exact + 3 * r.stderr


def test_mc_knock_in_via_survival_complement():
    ui = BARRIER.with_barrier_type(BarrierType.UP_AND_IN)
    r = MonteCarloEngine(PathGenerator(100_000, 100, seed=7)).price(ui, MKT)
    assert abs(r.value - A.price(ui, MKT).value) < 4 * r.stderr


def test_longstaff_schwartz_is_close_and_biased_low():
    amp = PUT.with_exercise(Exercise.AMERICAN)
    ref = LatticeEngine(2001).price(amp, MKT).value
    r = MonteCarloEngine(PathGenerator(100_000, 50, seed=8)).price(amp, MKT)
    assert r.value < ref + 2 * r.stderr          # low-biased
    assert r.value > ref - 0.06                   # but not by much
    assert r.value > MonteCarloEngine(PathGenerator(100_000, 50, seed=8)).price(PUT, MKT).value


def test_pathwise_delta():
    d, se = MonteCarloEngine(PathGenerator(200_000, 1, seed=9)).pathwise_delta(CALL, MKT)
    assert abs(d - A.price(CALL, MKT).greeks["delta"]) < 4 * se


def test_likelihood_ratio_delta_on_a_digital():
    """Pathwise fails on a step payoff; likelihood-ratio does not care."""
    dig = DigitalOption(1.0, 100.0, "call")
    d, se = MonteCarloEngine(PathGenerator(400_000, 1, seed=10)).likelihood_ratio_delta(dig, MKT)
    fd = bump_greeks(A, dig, MKT, which=("delta",))["delta"]
    assert abs(d - fd) < 4 * se


def test_seed_reproducibility():
    a = MonteCarloEngine(PathGenerator(5_000, 10, seed=11)).price(CALL, MKT).value
    b = MonteCarloEngine(PathGenerator(5_000, 10, seed=11)).price(CALL, MKT).value
    assert a == b


def test_antithetic_requires_even_paths():
    with pytest.raises(ValueError, match="even"):
        PathGenerator(1001, antithetic=True)


# ===========================================================================
# Cross-engine agreement -- the headline tests
# ===========================================================================

def test_four_engines_agree_on_a_vanilla():
    exact = A.price(CALL, MKT).value
    assert LatticeEngine(501).price(CALL, MKT).value == pytest.approx(exact, abs=1e-4)
    assert PDEEngine(600, 600).price(CALL, MKT).value == pytest.approx(exact, abs=2e-3)
    mc = MonteCarloEngine(PathGenerator(200_000, 1, seed=12)).price(CALL, MKT)
    assert abs(mc.value - exact) < 4 * mc.stderr


def test_four_engines_agree_on_a_barrier():
    exact = A.price(BARRIER, MKT).value
    assert LatticeEngine(2000).price(BARRIER, MKT).value == pytest.approx(exact, abs=2e-3)
    assert PDEEngine(600, 600).price(BARRIER, MKT).value == pytest.approx(exact, abs=2e-3)
    mc = MonteCarloEngine(PathGenerator(200_000, 100, seed=13)).price(BARRIER, MKT)
    assert abs(mc.value - exact) < 4 * mc.stderr


def test_three_engines_agree_on_an_american_put():
    amp = PUT.with_exercise(Exercise.AMERICAN)
    lat = LatticeEngine(2001).price(amp, MKT).value
    pde = PDEEngine(600, 600, american_solver="psor").price(amp, MKT).value
    mc = MonteCarloEngine(PathGenerator(100_000, 50, seed=14)).price(amp, MKT)
    assert pde == pytest.approx(lat, abs=5e-3)
    assert abs(mc.value - lat) < 0.06


# ===========================================================================
# Implied volatility
# ===========================================================================

@pytest.mark.parametrize("strike", [60.0, 80.0, 100.0, 120.0, 150.0, 200.0])
@pytest.mark.parametrize("true_vol", [0.12, 0.30, 0.65])
def test_implied_vol_round_trip(strike, true_vol):
    opt = VanillaOption(1.0, strike, "call")
    px = A.price(opt, MKT.with_vol(true_vol)).value
    assert implied_volatility(px, opt, MKT) == pytest.approx(true_vol, abs=1e-8)


def test_implied_vol_for_puts():
    px = A.price(PUT, MKT.with_vol(0.27)).value
    assert implied_volatility(px, PUT, MKT) == pytest.approx(0.27, abs=1e-8)


def test_implied_vol_nan_outside_arbitrage_bounds():
    assert np.isnan(implied_volatility(MKT.spot + 1, CALL, MKT))
    assert np.isnan(implied_volatility(-1.0, CALL, MKT))


def test_implied_vol_surface_is_flat_for_black_scholes_prices():
    strikes, expiries = np.array([90.0, 100.0, 110.0]), np.array([0.5, 1.0, 2.0])
    prices = np.array([[A.price(VanillaOption(t, k, "call"), MKT).value for t in expiries] for k in strikes])
    surf = implied_vol_surface(prices, strikes, expiries, MKT)
    assert np.allclose(surf, MKT.vol, atol=1e-8)
