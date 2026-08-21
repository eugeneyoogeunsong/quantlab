# quantlab: Eugene (Yoogeun) Song (https://www.linkedin.com/in/yoogeunsong)
# Independent side project. MIT licensed; see LICENSE.

"""Monte Carlo pricing under geometric Brownian motion.

Vectorised across paths rather than looped path by path: the same arithmetic,
roughly two orders of magnitude faster.

Monte Carlo is the slowest and least accurate method here for a vanilla European
call, where a closed form exists. It earns its place on payoffs that depend on
the whole path, or on many underlyings at once, where the lattice and the PDE
both become intractable.

Its error shrinks as 1/sqrt(n_paths): 100x the compute buys 10x the accuracy.
That is a poor exchange rate, and the reason variance-reduction techniques
(antithetic variates, control variates) exist at all.

What the module covers
----------------------
(i) `simulate_gbm_paths`, the exact lognormal stepper shared by everything else;
(ii) European and up-and-out barrier pricers, each with optional antithetic and
control variates; (iii) `longstaff_schwartz_american`, least-squares Monte Carlo
for early exercise (Longstaff & Schwartz 2001); and (iv) `monte_carlo_greeks`,
which contrasts the pathwise and likelihood-ratio estimators.

Sources: Boyle (1977) for the original application of simulation to option
pricing; Glasserman, *Monte Carlo Methods in Financial Engineering* (2004) for
variance reduction (Ch. 4), quasi-Monte Carlo (Ch. 5), Greeks (Ch. 7) and
American pricing (Ch. 8). Shreve II Ch. 5 supplies the change of measure that
makes the simulated drift `r - q` rather than the real-world drift.

Dividend convention (shared across the package): the risk-neutral drift of `S`
is `(r - q)`, whilst discounting is always at `r`, never at `r - q`. Every
public function takes `q` last and defaults it to 0.0, so existing calls are
unchanged bit for bit.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial import laguerre
from scipy.stats import norm, qmc

from .analytic import black_scholes_call

__all__ = [
    "simulate_gbm_paths",
    "monte_carlo_european",
    "monte_carlo_up_and_out",
    "longstaff_schwartz_american",
    "monte_carlo_greeks",
]


def _payoff(S, K: float, option: str):
    """Terminal intrinsic value, with the option string validated in one place."""
    if option == "call":
        return np.maximum(S - K, 0.0)
    if option == "put":
        return np.maximum(K - S, 0.0)
    raise ValueError(f"option must be 'call' or 'put', got {option!r}")


def _standard_normals(n_paths: int, n_steps: int, seed, antithetic: bool,
                      sobol: bool) -> np.ndarray:
    """Draw the (n_paths, n_steps) matrix of normals that drives every path.

    Three sampling regimes share this helper: plain pseudo-random draws,
    antithetic pairs, and a scrambled Sobol' net inverted through the normal
    quantile function. Keeping them together is what guarantees the pricers
    consume randomness identically.
    """
    if sobol:
        if antithetic:
            raise ValueError(
                "sobol=True and antithetic=True are mutually exclusive: a Sobol' net is "
                "already balanced by construction, and mirroring it destroys that balance."
            )
        if n_paths < 1 or n_paths & (n_paths - 1):
            raise ValueError(
                f"sobol=True requires n_paths to be a power of two, got {n_paths}. "
                "The equidistribution properties of a Sobol' net only hold on 2^m points; "
                "truncating a net elsewhere leaves a biased, unbalanced sample."
            )
        engine = qmc.Sobol(d=n_steps, scramble=True, seed=seed)
        u = engine.random(n_paths)
        # Owen scrambling can land on the endpoints of the unit cube, where the
        # normal quantile is infinite; clipping at 1e-12 caps such a draw at 7 sigma.
        return norm.ppf(np.clip(u, 1e-12, 1.0 - 1e-12))

    rng = np.random.default_rng(seed)
    if antithetic:
        if n_paths % 2:
            raise ValueError("antithetic sampling requires an even n_paths")
        half = rng.standard_normal((n_paths // 2, n_steps))
        return np.concatenate([half, -half], axis=0)
    return rng.standard_normal((n_paths, n_steps))


def _apply_control(payoffs: np.ndarray, control: np.ndarray, control_mean: float):
    """Subtract the optimally-scaled control from a vector of discounted payoffs.

    Given a control `X` with known mean, the estimator is `Y - b*(X - E[X])`, and
    the variance-minimising coefficient is `b* = Cov(Y, X) / Var(X)`, which cuts
    the variance by a factor `1 / (1 - corr(Y, X)^2)` (Glasserman Ch. 4.1).

    The bias, stated plainly
    ------------------------
    We estimate `b*` from the same paths we price on, so the estimator is a ratio
    of correlated sample moments and is therefore no longer unbiased. In
    principle that matters; in practice it does not, because the bias is O(1/n)
    whilst the standard error it buys down is O(1/sqrt(n)) (Glasserman Ch. 4.1.3).
    We measured it rather than assumed it: pairing each control-variate run
    against the plain run on the same paths over 2,000 replications of the
    110-strike call, the mean difference is -2.0e-3 +/- 4.9e-3 at 2,000 paths and
    +1.9e-3 +/- 1.6e-3 at 20,000 paths, against per-run standard errors of 0.136
    and 0.044 respectively. No bias is detectable at either count, so we fit `b*`
    in sample and spend the paths on variance instead. The exact fix, should an
    unbiased estimator ever be required, is to estimate `b*` on a pilot run and
    hold it fixed on an independent pricing run.

    Returns the adjusted payoffs and the fitted `b*`.
    """
    variance = float(control.var(ddof=1))
    if not np.isfinite(variance) or variance <= 0.0:
        return payoffs, 0.0  # degenerate control (e.g., every path identical)
    b = float(np.cov(payoffs, control, ddof=1)[0, 1] / variance)
    return payoffs - b * (control - control_mean), b


def _spawn_seeds(seed, n: int) -> list[int]:
    """Split one seed into `n` statistically independent child seeds.

    `SeedSequence` hashes the entropy, so the children are independent streams
    rather than merely adjacent ones; `seed=None` draws fresh OS entropy.
    """
    return [int(child.generate_state(1)[0])
            for child in np.random.SeedSequence(seed).spawn(n)]


def simulate_gbm_paths(S0: float, T: float, r: float, sigma: float,
                       n_paths: int = 10_000, n_steps: int = 252,
                       seed: int | None = None,
                       antithetic: bool = False,
                       sobol: bool = False,
                       q: float = 0.0) -> np.ndarray:
    """Simulate GBM paths under the risk-neutral measure.

    Returns an array of shape (n_paths, n_steps + 1) including S0 at column 0.

    We step the exact lognormal solution rather than an Euler discretisation, so
    there is no time-discretisation bias in the terminal value, only sampling
    error. Under a continuous dividend yield `q` the drift becomes `(r - q)`;
    with the default `q=0.0` the arithmetic is bit-for-bit what it always was.

    `antithetic=True` pairs each path with its mirror image (Z and -Z). Because
    the pair's errors are negatively correlated, the average is less noisy for
    the same number of draws; it requires an even `n_paths`.

    `sobol=True` replaces the pseudo-random draws with a scrambled Sobol'
    sequence (Sobol' 1967; Owen 1997), inverted through the normal quantile
    function. Two caveats come with it, and both are enforced or documented
    rather than hidden: (i) `n_paths` must be a power of two, since a Sobol' net
    is only equidistributed on 2^m points, and we raise otherwise; and (ii) the
    points are deliberately dependent, so the usual sample standard error is not
    a valid error estimate (see `monte_carlo_european`, which returns NaN for it).
    Quasi-Monte Carlo also degrades as the dimension `n_steps` grows, and the
    measured degradation is large: pricing the 110-strike call over 2^10 to 2^16
    points, a scrambled net cuts the RMSE by 10x to 23x at `n_steps=8` but only
    2.4x to 6.3x at `n_steps=64`. Recovering the gain at 252 steps needs a
    Brownian-bridge construction, which concentrates the variance in the leading
    dimensions and leaves the net's good projections doing the work
    (Glasserman Ch. 5.5); that is not implemented here.
    """
    dt = T / n_steps
    Z = _standard_normals(n_paths, n_steps, seed, antithetic, sobol)

    increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1)
    return S0 * np.exp(log_paths)


def monte_carlo_european(S0, K, T, r, sigma, n_paths: int = 50_000,
                         n_steps: int = 252, option: str = "call",
                         seed: int | None = None, antithetic: bool = True,
                         return_stderr: bool = False,
                         control_variate: bool = False,
                         sobol: bool = False,
                         q: float = 0.0):
    """European option by simulation.

    Returns the price, or `(price, standard_error)` if `return_stderr=True`.

    Always look at that standard error: a Monte Carlo price quoted without one is
    a number of unknown precision, and the 95% interval is roughly
    price +/- 2*stderr, which is usually wider than people expect.

    Control variates
    ----------------
    `control_variate=True` uses the discounted terminal underlying,
    `X = exp(-r*T) * S_T`, whose expectation `S0 * exp(-q*T)` is the forward
    priced by no-arbitrage alone. This is Glasserman's canonical example
    (Ch. 4.1.1), and the choice deserves a word, because the obvious alternative
    is degenerate: taking the option's own analytic price as its control makes
    `Y` and `X` the same random variable, so `b*` collapses to 1, the estimator
    returns Black-Scholes exactly, and nothing has been estimated at all. The
    underlying is the informative control instead, and its usefulness tracks
    `corr(Y, X)`: measured on S0=100, T=1, r=0.05, sigma=0.2 with 200,000 paths,
    the standard error falls by 8.1x at K=80 (corr 0.99), 2.6x at K=100 (0.93),
    1.9x at K=110 (0.85) and 1.3x at K=130 (0.64). Deep in the money the payoff
    is `S_T - K` on nearly every path and the control removes nearly everything;
    deep out of the money it is zero on nearly every path, the linear projection
    has almost nothing to explain, and the technique earns almost nothing.

    Sobol' quasi-Monte Carlo
    ------------------------
    `sobol=True` supersedes `antithetic` (the two cannot be combined; see
    `simulate_gbm_paths`) and requires `n_paths` to be a power of two. Since a
    scrambled net is a dependent sample by construction, `return_stderr=True`
    returns NaN rather than a sample standard deviation that would look like a
    confidence measure and would not be one. To obtain an honest error bar under
    QMC, average several independent scrambles and take the standard deviation
    across those replications (Glasserman Ch. 5.4).
    """
    paths = simulate_gbm_paths(S0, T, r, sigma, n_paths, n_steps, seed,
                               antithetic and not sobol, sobol, q)
    S_T = paths[:, -1]
    payoffs = _payoff(S_T, K, option)

    discounted = np.exp(-r * T) * payoffs
    if control_variate:
        control = np.exp(-r * T) * S_T
        discounted, _ = _apply_control(discounted, control, S0 * np.exp(-q * T))

    price = float(discounted.mean())
    if return_stderr:
        if sobol:
            return price, float("nan")
        return price, float(discounted.std(ddof=1) / np.sqrt(n_paths))
    return price


def monte_carlo_up_and_out(S0, K, B, T, r, sigma, n_paths: int = 50_000,
                           n_steps: int = 252, seed: int | None = None,
                           brownian_bridge: bool = True,
                           return_stderr: bool = False,
                           control_variate: bool = False,
                           q: float = 0.0):
    """Up-and-out barrier call by simulation.

    The subtlety that makes this interesting
    ----------------------------------------
    Checking only the simulated grid points for a barrier breach systematically
    OVERPRICES the option. The path is continuous, so between two observations
    that both sit below the barrier the true path may still have crossed it and
    knocked the option out; discrete monitoring misses those crossings, and too
    many paths survive.

    The Brownian bridge correction fixes this analytically. Conditional on the
    endpoints of a step, the probability that the bridge between them touched
    the barrier has a closed form:

        p_cross = exp( -2 * ln(S_prev/B) * ln(S_next/B) / (sigma^2 * dt) )

    Each path is then weighted by its probability of *surviving* every step,
    rather than being counted as a binary survivor. We compute those products as
    exp(sum(log(.))) to avoid underflow when many steps are involved.

    Set `brownian_bridge=False` to see the bias for yourself: the test suite
    asserts the uncorrected price is the higher of the two.

    Control variates, and where they stop working
    ---------------------------------------------
    `control_variate=True` uses the vanilla European call on the same paths,
    priced by `black_scholes_call`, as the control. The optimal coefficient
    `b* = Cov(Y, X)/Var(X)` is fitted on those same paths, with the O(1/n) bias
    discussed in `_apply_control`.

    The honest caveat is that this control is only as good as its correlation
    with the barrier payoff, and that correlation collapses as the barrier
    approaches the money: measured on S0=100, K=95, T=1, r=0.05, sigma=0.2 with
    100,000 paths, corr(Y, X) runs -0.03 / 0.15 / 0.39 / 0.61 / 0.93 for
    B = 120 / 130 / 140 / 150 / 180, giving standard-error reductions of
    1.00x / 1.01x / 1.09x / 1.26x / 2.76x. A distant barrier is nearly a vanilla
    and the control works; a near barrier knocks out exactly the paths on which
    the vanilla pays most, and the linear projection has almost nothing to
    remove. The natural improvement (not implemented here, since it stops being
    a vanilla) is to knock the control out on the terminal value as well, which
    lifts the same five figures to 1.16x / 1.31x / 1.56x / 1.93x / 4.30x.
    """
    if B <= K or S0 >= B:
        return (0.0, 0.0) if return_stderr else 0.0

    paths = simulate_gbm_paths(S0, T, r, sigma, n_paths, n_steps, seed, False, False, q)
    dt = T / n_steps
    S_T = paths[:, -1]

    if brownian_bridge:
        prev, nxt = paths[:, :-1], paths[:, 1:]
        # A grid observation at or above the barrier is an outright knockout, so
        # it overrides whatever survival weight the bridge would otherwise assign.
        breached = (paths >= B).any(axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            log_prev = np.log(np.maximum(prev, 1e-300) / B)
            log_next = np.log(np.maximum(nxt, 1e-300) / B)
            p_cross = np.exp(-2.0 * log_prev * log_next / (sigma**2 * dt))
        p_cross = np.clip(np.nan_to_num(p_cross, nan=1.0), 0.0, 1.0)

        log_survive = np.log(np.maximum(1.0 - p_cross, 1e-300)).sum(axis=1)
        survival = np.exp(log_survive)
        survival[breached] = 0.0
    else:
        survival = (~(paths >= B).any(axis=1)).astype(float)

    vanilla = np.exp(-r * T) * np.maximum(S_T - K, 0.0)
    discounted = vanilla * survival
    if control_variate:
        discounted, _ = _apply_control(
            discounted, vanilla, float(black_scholes_call(S0, K, T, r, sigma, q)))

    price = float(discounted.mean())
    if return_stderr:
        return price, float(discounted.std(ddof=1) / np.sqrt(n_paths))
    return price


def _basis_matrix(x: np.ndarray, basis: str, degree: int) -> np.ndarray:
    """Regression design matrix for the least-squares Monte Carlo continuation fit.

    `x` is the moneyness `S/K` rather than `S` itself: the columns of a raw power
    basis in `S` span four orders of magnitude at degree 3 (1e0 to 1e6 for a spot
    near 100), which is a needlessly ill-conditioned normal-equations problem.

    Two bases are offered. `"power"` gives 1, x, ..., x^degree, and `"laguerre"`
    gives the weighted Laguerre functions exp(-x/2) * L_k(x) used by Longstaff and
    Schwartz (2001, p. 121). The two agree far inside Monte Carlo noise on a
    one-dimensional put (11.9114 vs 11.9101 on the contract in the tests, against
    a standard error of 0.0384), which is the expected outcome: what matters is
    that the basis spans the shape of the continuation value, not which
    polynomials are used to span it.
    """
    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree}")
    if basis == "power":
        return np.vander(x, degree + 1, increasing=True)
    if basis == "laguerre":
        weight = np.exp(-x / 2.0)
        return np.column_stack(
            [laguerre.lagval(x, c) * weight for c in np.eye(degree + 1)])
    raise ValueError(f"basis must be 'power' or 'laguerre', got {basis!r}")


def _lsm_replay(paths: np.ndarray, coeffs: dict[int, np.ndarray], K: float,
                option: str, basis: str, degree: int, r: float, T: float) -> np.ndarray:
    """Apply a frozen exercise rule forward on fresh paths.

    Each path stops at the first exercise date whose intrinsic value beats the
    fitted continuation value; unstopped paths take the terminal intrinsic value.
    Returns the discounted realised cashflow per path.
    """
    n_steps = paths.shape[1] - 1
    dt = T / n_steps
    alive = np.ones(paths.shape[0], dtype=bool)
    value = np.zeros(paths.shape[0])

    for t in range(1, n_steps):
        beta = coeffs.get(t)
        if beta is None:
            continue
        intrinsic = _payoff(paths[:, t], K, option)
        live_itm = alive & (intrinsic > 0.0)
        if not live_itm.any():
            continue
        design = _basis_matrix(paths[live_itm, t] / K, basis, degree)
        stopped = np.flatnonzero(live_itm)[intrinsic[live_itm] > design @ beta]
        value[stopped] = np.exp(-r * t * dt) * intrinsic[stopped]
        alive[stopped] = False

    terminal = _payoff(paths[:, -1], K, option)
    value[alive] = np.exp(-r * T) * terminal[alive]
    return value


def longstaff_schwartz_american(S0, K, T, r, sigma, n_paths: int = 50_000,
                                n_steps: int = 50, option: str = "put",
                                seed: int | None = None, antithetic: bool = True,
                                basis: str = "power", degree: int = 3,
                                return_stderr: bool = False,
                                out_of_sample: bool = False,
                                q: float = 0.0):
    """American option by least-squares Monte Carlo (Longstaff & Schwartz 2001).

    Returns the price, or `(price, standard_error)` if `return_stderr=True`.
    `n_steps` is the number of equally-spaced exercise dates, so the contract
    priced is strictly a Bermudan one converging to the American value from
    below as `n_steps` grows.

    The idea
    --------
    Early exercise is an optimal-stopping problem: at each date we must compare
    the intrinsic value with the continuation value, and the continuation value
    is a conditional expectation, which forward simulation cannot see. Longstaff
    and Schwartz estimate it by regression. Working backwards from expiry, we
    regress each path's realised discounted future cashflow on a basis of the
    current spot, restricted to the in-the-money paths (the only ones where the
    exercise decision is live, and restricting the fit keeps the regression
    focused on the region that matters), then exercise wherever intrinsic value
    exceeds the fitted continuation. The regression is a cross-sectional
    projection, so no lookahead enters the *decision*: the fitted value at each
    date depends on the whole sample, but the rule applied to a path uses only
    that path's current state.

    Two biases, in opposite directions
    ----------------------------------
    (i) The fitted rule is suboptimal, since a degree-3 basis only approximates
    the true continuation value; any suboptimal stopping rule undervalues the
    option, so this pushes the price DOWN and makes the estimator a lower bound
    in principle. (ii) The rule is fitted on the very paths it is then evaluated
    on, so it exercises with partial foresight of that sample's noise, which
    pushes the price UP. At realistic path counts the first dominates, and the
    in-sample estimator comes out low: on the contract in the test suite
    (S0=100, K=110, T=1, r=0.05, sigma=0.2, 50,000 paths, 50 dates) it averages
    11.949 over eight seeds against 11.9728 from a 5,000-step binomial, a gap of
    -0.024 (-0.20%) with a per-run standard error of 0.038. Most of that gap is
    the Bermudan restriction rather than the regression: raising the exercise
    dates from 25 to 100 moves the mean error from -0.036 to -0.014.

    The dividend path is checked independently rather than against itself: the
    American put-call symmetry `C_A(S, K, r, q) = P_A(K, S, q, r)` (McDonald &
    Schroder 1998) lets a lattice with no dividend argument price the symmetric
    contract, and the two agree to within one standard error in the tests.

    `out_of_sample=True` removes the second bias outright: the regression
    coefficients are frozen, an independent set of paths is drawn from a spawned
    seed, and the frozen rule is replayed forward on them (Glasserman Ch. 8.6).
    What remains is a genuine lower bound on the American value, which is the
    quantity one should quote; the cost is a second simulation, so the call is
    roughly twice as slow.

    When to use this, and when not to
    ---------------------------------
    LSM exists for high-dimensional American problems: a basket of five assets,
    a swaption in a multi-factor model, anything where a lattice would need a
    grid in five dimensions and the PDE is hopeless. For the one-dimensional
    American put priced here it is the wrong tool, and the timings say so
    plainly: `binomial_american` at 5,000 steps settles to 11.97285 (stable to
    1e-4 against 40,000 steps) in 0.25 s, whilst 50,000 LSM paths across 50
    dates cost 0.31 s and land 0.024 low. Same order of compute, two orders of
    magnitude more error. We implement LSM on the one-dimensional case because
    that is where it can be checked against an independent method, not because
    it is competitive there.

    Parameters
    ----------
    basis : str      'power' (1, x, ..., x^degree) or 'laguerre' (weighted
                     Laguerre functions), with x = S/K
    degree : int     polynomial degree of the continuation-value regression
    out_of_sample : bool  refit-free resimulation pass, as described above
    q : float        continuous dividend yield
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    _payoff(float(S0), K, option)  # reject a bad option string before any work
    _basis_matrix(np.ones(1), basis, degree)  # and a bad basis or degree

    fit_seed, eval_seed = _spawn_seeds(seed, 2)
    paths = simulate_gbm_paths(S0, T, r, sigma, n_paths, n_steps, fit_seed,
                               antithetic, False, q)
    step_disc = np.exp(-r * T / n_steps)

    # Backward induction. `cashflow` holds each path's realised cashflow, valued
    # at the date currently under consideration, so one discount factor per step.
    cashflow = _payoff(paths[:, -1], K, option)
    coeffs: dict[int, np.ndarray] = {}
    for t in range(n_steps - 1, 0, -1):
        cashflow *= step_disc
        intrinsic = _payoff(paths[:, t], K, option)
        itm = intrinsic > 0.0
        if itm.sum() <= degree + 1:
            continue  # too few live paths to identify the regression
        design = _basis_matrix(paths[itm, t] / K, basis, degree)
        beta, *_ = np.linalg.lstsq(design, cashflow[itm], rcond=None)
        coeffs[t] = beta
        stopped = np.flatnonzero(itm)[intrinsic[itm] > design @ beta]
        cashflow[stopped] = intrinsic[stopped]

    if out_of_sample:
        fresh = simulate_gbm_paths(S0, T, r, sigma, n_paths, n_steps, eval_seed,
                                   antithetic, False, q)
        values = _lsm_replay(fresh, coeffs, K, option, basis, degree, r, T)
    else:
        values = cashflow * step_disc

    # Exercising at t=0 is a deterministic alternative, so it is a max, not a
    # regression: it binds only when the option is so deep in the money that
    # waiting is never worthwhile.
    price = max(float(values.mean()), float(_payoff(float(S0), K, option)))
    if return_stderr:
        return price, float(values.std(ddof=1) / np.sqrt(n_paths))
    return price


def monte_carlo_greeks(S0, K, T, r, sigma, n_paths: int = 100_000,
                       n_steps: int = 1, option: str = "call",
                       seed: int | None = None, antithetic: bool = True,
                       return_stderr: bool = False, q: float = 0.0) -> dict:
    """Delta, vega and gamma by simulation, each with the estimator it deserves.

    Returns a dict with keys `delta`, `vega`, `gamma` and `delta_lr`; with
    `return_stderr=True` each gains a `_stderr` companion. All four estimators
    are computed from one set of paths, so their standard errors are directly
    comparable.

    Why delta and vega are pathwise, and gamma is not
    -------------------------------------------------
    This distinction is the most instructive thing in the module, so it is worth
    stating carefully (Glasserman Ch. 7.2 and 7.3).

    The pathwise method differentiates the payoff along the path and takes the
    expectation of the derivative: it swaps the order of differentiation and
    expectation, which is legitimate only when the payoff is Lipschitz in the
    parameter (a.s. differentiability with an integrable Lipschitz constant is
    the sufficient condition). The call payoff `(S_T - K)^+` qualifies: it is
    continuous, and non-differentiable only at the single point `S_T = K`, which
    the lognormal law assigns probability zero. Under GBM `dS_T/dS0 = S_T/S0`,
    so the pathwise delta is

        exp(-r*T) * 1{S_T > K} * S_T / S0,

    and differentiating `ln S_T = ln S0 + (r - q - sigma^2/2)*T + sigma*W_T` with
    respect to sigma gives `dS_T/dsigma = S_T * (W_T - sigma*T)`, hence the
    pathwise vega

        exp(-r*T) * 1{S_T > K} * S_T * (W_T - sigma*T).

    Gamma is where the method breaks. It is the second derivative of the payoff,
    and the second derivative of `(S_T - K)^+` in `S_T` is a Dirac delta at the
    strike: it is not a function, its pathwise "estimator" is zero on every path
    that does not land exactly on `K`, and no amount of sampling recovers the
    mass concentrated at that point. Differentiating the DENSITY instead of the
    payoff sidesteps this entirely, since the lognormal density is smooth in
    `S0` however rough the payoff is. With
    `Z = (ln(S_T/S0) - (r - q - sigma^2/2)*T) / (sigma*sqrt(T))`, the
    likelihood-ratio weights are the first two derivatives of the log-density,

        delta:  Z / (S0*sigma*sqrt(T))
        gamma:  (Z^2 - 1) / (S0^2*sigma^2*T) - Z / (S0^2*sigma*sqrt(T)),

    applied to the undifferentiated discounted payoff.

    The trade-off is variance, and it goes the way the theory predicts: on
    S0=100, K=110, T=1, r=0.05, sigma=0.2 with 200,000 antithetic paths, the
    pathwise delta has standard error 0.0013 against 0.0027 for the
    likelihood-ratio delta, a factor of 2.1. Therefore the rule is: use pathwise
    wherever the payoff is smooth enough to allow it, and keep likelihood-ratio
    for the cases (gamma, digitals, anything with a jump in the payoff) where it
    is the only estimator that works at all.

    `n_steps` defaults to 1 because every estimator here depends on the path only
    through `S_T`, and the lognormal step is exact: 252 steps would cost 252x the
    work for identical statistics.
    """
    paths = simulate_gbm_paths(S0, T, r, sigma, n_paths, n_steps, seed,
                               antithetic, False, q)
    S_T = paths[:, -1]
    payoffs = _payoff(S_T, K, option)

    sqrt_T = np.sqrt(T)
    disc = np.exp(-r * T)
    # Recover the driving Brownian increment from the terminal value: exact,
    # since the stepper integrates the SDE exactly.
    Z = (np.log(S_T / S0) - (r - q - 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    W_T = sqrt_T * Z

    # d(payoff)/dS_T: +1 above the strike for a call, -1 below it for a put.
    slope = (S_T > K).astype(float) if option == "call" else -(S_T < K).astype(float)
    estimators = {
        "delta": disc * slope * S_T / S0,
        "vega": disc * slope * S_T * (W_T - sigma * T),
        "gamma": disc * payoffs * ((Z**2 - 1.0) / (S0**2 * sigma**2 * T)
                                   - Z / (S0**2 * sigma * sqrt_T)),
        "delta_lr": disc * payoffs * Z / (S0 * sigma * sqrt_T),
    }

    out = {name: float(v.mean()) for name, v in estimators.items()}
    if return_stderr:
        out.update({f"{name}_stderr": float(v.std(ddof=1) / np.sqrt(n_paths))
                    for name, v in estimators.items()})
    return out
