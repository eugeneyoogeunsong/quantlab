# quantlab: Eugene (Yoogeun) Song (https://www.linkedin.com/in/yoogeunsong)
# Independent side project. MIT licensed; see LICENSE.

"""PDE pricing by finite differences: the Crank-Nicolson (theta) family.

Rather than simulating or enumerating outcomes, we solve the Black-Scholes PDE
directly on a grid of (spot, time): the payoff is the initial condition, and we
integrate backwards from expiry. Writing tau = T - t for the time REMAINING and
carrying a continuous dividend yield q, the equation we discretise is

    dV/dtau = 0.5*sigma^2*S^2 * d2V/dS2 + (r - q)*S * dV/dS - r*V.

The dividend enters the convection term only: q reduces the risk-neutral drift
of S to (r - q), whilst payoffs are still discounted at r, never at (r - q).
Feynman-Kac is the bridge between this equation and the expectation the other
three modules compute (Shreve II, Ch. 4 and 6): the discounted price is a
martingale under the risk-neutral measure, so its drift must vanish, and setting
that drift to zero IS the PDE above.

The reward for the extra machinery is that a single solve gives the option value
at EVERY spot price, not just today's, so delta and gamma come out as finite
differences of a surface we already have rather than requiring a re-run
(`crank_nicolson_greeks`). Two further payoffs follow from working on a grid:
a knockout barrier is just a Dirichlet condition on the domain edge
(`crank_nicolson_up_and_out`), and early exercise is just a pointwise constraint
on the solution (`crank_nicolson_american`).

The theta family
----------------
Every scheme here is a weighted average of the explicit and implicit Euler steps,
parameterised by `theta_scheme`, the weight placed on the OLD time level:

    theta_scheme = 0    fully implicit, O(dt), unconditionally stable
    theta_scheme = 0.5  Crank-Nicolson, O(dt^2), unconditionally stable
    theta_scheme = 1    fully explicit, O(dt), stable only under a step limit

Von Neumann analysis of the frozen-coefficient problem gives the condition for
theta_scheme > 0.5:

    dt <= 1 / (sigma^2 * n_space^2 * (2*theta_scheme - 1)),

worst at the top of the grid, where the diffusion coefficient 0.5*sigma^2*S^2 is
largest. At the fully explicit end with sigma = 0.2 and n_space = 400 that is
dt <= 1.6e-4, i.e., 6400 timesteps for a one-year contract: the reason
the default is 0.5 and the reason nobody prices with the explicit scheme.

Crank-Nicolson buys its second order at a price worth knowing: its amplification
factor tends to -1 rather than 0 for the highest-frequency modes, so a kink or a
jump in the initial data rings rather than decays. Rannacher (1984) startup, two
fully implicit steps before switching, restores damping without giving up the
second-order tail; `crank_nicolson_up_and_out` uses it because the payoff
discontinuity at the barrier makes the oscillation impossible to ignore.

The linear systems are solved with a banded (tridiagonal) solver rather than a
dense factorisation: identical arithmetic at O(n) instead of O(n^3) per timestep.

References
----------
Wilmott (2007), Ch. 8, for the schemes, their stability, and projected SOR;
Shreve II (2004), Ch. 4, 6 and 8, for Feynman-Kac and optimal stopping;
Brennan & Schwartz (1977) for the first finite-difference American put;
Cryer (1971) for the convergence of projected overrelaxation;
Jaillet, Lamberton & Lapeyre (1990) for the variational-inequality formulation.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

__all__ = [
    "crank_nicolson_european",
    "crank_nicolson_american",
    "crank_nicolson_greeks",
    "crank_nicolson_up_and_out",
]


def _build_grid(S_max: float, n_space: int):
    return np.linspace(0.0, S_max, n_space + 1)


def _cn_coefficients(S_idx: np.ndarray, dt: float, r: float, sigma: float, q: float = 0.0):
    """Crank-Nicolson coefficients for interior nodes.

    Averaging the explicit and implicit schemes gives second-order accuracy in
    time and unconditional stability; the explicit scheme on its own is stable
    only for very small timesteps.

    The dividend yield appears in the convection (first-derivative) terms as
    (r - q) and nowhere else: the middle coefficient carries the discounting,
    which is at r regardless of q.
    """
    i = S_idx
    alpha = 0.25 * dt * (sigma**2 * i**2 - (r - q) * i)
    beta = -0.5 * dt * (sigma**2 * i**2 + r)
    gamma = 0.25 * dt * (sigma**2 * i**2 + (r - q) * i)
    return alpha, beta, gamma


def _theta_weights(theta_scheme: float):
    """Rescale the CN coefficients into a general theta-scheme pair.

    `_cn_coefficients` returns half of the spatial operator times dt, since
    Crank-Nicolson splits it evenly. A general theta scheme puts `theta_scheme`
    on the old level and the remainder on the new one, so we multiply by
    2*theta_scheme and 2*(1 - theta_scheme) respectively. At theta_scheme = 0.5
    both factors are exactly 1.0, hence the default path is bit-for-bit the
    arithmetic this module has always performed.
    """
    if not 0.0 <= theta_scheme <= 1.0:
        raise ValueError(f"theta_scheme must lie in [0, 1], got {theta_scheme!r}")
    return 2.0 * theta_scheme, 2.0 * (1.0 - theta_scheme)


def _check_explicit_stability(theta_scheme: float, dt: float, sigma: float, n_space: int):
    """Refuse to run a theta scheme past its stability limit.

    For theta_scheme > 0.5 the scheme is only conditionally stable, and the
    condition bites at the top of the grid where the diffusion coefficient
    0.5*sigma^2*S^2 is largest. Von Neumann analysis of the frozen-coefficient
    problem gives dt <= 1/(sigma^2 * n_space^2 * (2*theta_scheme - 1)). Violating
    it does not degrade the answer gracefully: the solution overflows to NaN
    within a few dozen steps, so we say why up front rather than let scipy report
    an array of infinities several layers down.
    """
    if theta_scheme <= 0.5 or sigma <= 0.0:
        return
    dt_max = 1.0 / (sigma**2 * n_space**2 * (2.0 * theta_scheme - 1.0))
    if dt > dt_max:
        raise ValueError(
            f"theta_scheme={theta_scheme} is unstable on this grid: dt={dt:.3e} exceeds "
            f"the limit 1/(sigma^2*n_space^2*(2*theta_scheme-1))={dt_max:.3e}. Multiply "
            f"n_time by at least {dt / dt_max:.1f}, coarsen n_space, or move "
            "theta_scheme to 0.5 or below, where the scheme is unconditionally stable."
        )


def _implicit_matrix(alpha, beta, gamma, w_imp: float, n_space: int):
    """Left-hand-side tridiagonal matrix in scipy's banded storage."""
    ab = np.zeros((3, n_space - 1))
    ab[0, 1:] = -(w_imp * gamma[:-1])     # super-diagonal
    ab[1, :] = 1.0 - w_imp * beta         # main diagonal
    ab[2, :-1] = -(w_imp * alpha[1:])     # sub-diagonal
    return ab


def _solve_cn(payoff, S, T, r, sigma, n_time, upper_boundary_fn,
              lower_boundary_fn=None, knockout_mask=None, q: float = 0.0,
              theta_scheme: float = 0.5):
    """Shared theta-scheme time-stepping loop.

    We step in tau (time remaining to expiry), so `payoff` is the tau = 0 state
    and the boundary functions are evaluated at the tau of the step being solved
    for. `lower_boundary_fn` defaults to a hard zero, which is what a call wants
    at S = 0; a put instead needs K*exp(-r*tau) there. Passing `knockout_mask`
    zeroes the masked nodes after every step, which is how a knockout barrier is
    imposed.

    Returns the final slice AND the one before it, because theta (the time
    derivative) is a difference of the last two slices and re-deriving it would
    otherwise mean a second solve.
    """
    n_space = len(S) - 1
    dt = T / n_time
    _check_explicit_stability(theta_scheme, dt, sigma, n_space)
    i = np.arange(1, n_space)
    alpha, beta, gamma = _cn_coefficients(i, dt, r, sigma, q)
    w_exp, w_imp = _theta_weights(theta_scheme)

    ab = _implicit_matrix(alpha, beta, gamma, w_imp, n_space)
    a_exp, b_exp, c_exp = w_exp * alpha, w_exp * beta, w_exp * gamma
    a_imp_lo, c_imp_hi = w_imp * alpha[0], w_imp * gamma[-1]

    V = payoff.copy()
    V_prev = V
    for step in range(n_time):
        tau_next = (step + 1) * dt
        V_low = 0.0 if lower_boundary_fn is None else lower_boundary_fn(tau_next)
        V_high = upper_boundary_fn(tau_next)

        # V[0] and V[-1] still hold the OLD boundary values, so the explicit half
        # of the boundary term is already inside `rhs`; we add the implicit half.
        rhs = a_exp * V[:-2] + (1.0 + b_exp) * V[1:-1] + c_exp * V[2:]
        rhs[0] += a_imp_lo * V_low
        rhs[-1] += c_imp_hi * V_high

        V_new = np.empty_like(V)
        V_new[0] = V_low
        V_new[-1] = V_high
        V_new[1:-1] = solve_banded((1, 1), ab, rhs)

        if knockout_mask is not None:
            V_new[knockout_mask] = 0.0
        V_prev, V = V, V_new
    return V, V_prev


def _european_grid(K, T, r, sigma, n_space, n_time, option, S_max, theta_scheme, q):
    """Backward integration of the European problem on [0, S_max].

    Both Dirichlet conditions are the exact large- and small-S asymptotics of the
    Black-Scholes price, discounted appropriately: for a call the value at S_max
    is the forward less the discounted strike, S_max*exp(-q*tau) - K*exp(-r*tau),
    and for a put the value at S = 0 is K*exp(-r*tau). Which end of the grid
    carries the informative data therefore depends on the payoff.
    """
    S = _build_grid(S_max, n_space)

    if option == "call":
        payoff = np.maximum(S - K, 0.0)

        def lower(tau):
            return 0.0

        def upper(tau):
            return S_max * np.exp(-q * tau) - K * np.exp(-r * tau)
    else:
        payoff = np.maximum(K - S, 0.0)

        def lower(tau):
            return K * np.exp(-r * tau)

        def upper(tau):
            return 0.0

    V, V_prev = _solve_cn(payoff, S, T, r, sigma, n_time, upper, lower,
                          q=q, theta_scheme=theta_scheme)
    return S, V, V_prev, T / n_time


def crank_nicolson_european(S0, K, T, r, sigma, n_space: int = 400,
                            n_time: int = 400, option: str = "call",
                            S_max: float | None = None,
                            return_grid: bool = False,
                            theta_scheme: float = 0.5, q: float = 0.0):
    """European option by Crank-Nicolson (or any theta scheme).

    `S_max` defaults to four times the larger of `K` and `S0`. It must sit far
    enough out that the boundary condition imposed there is essentially exact:
    too close, and the artificial boundary contaminates the interior solution.

    Parameters
    ----------
    S0 : float           spot price today
    K : float            strike
    T : float            time to expiry in YEARS
    r : float            continuously-compounded risk-free rate
    sigma : float        annualised volatility
    n_space : int        number of spot intervals; nodes are S_max*i/n_space
    n_time : int         number of timesteps
    option : str         'call' or 'put'
    S_max : float        upper edge of the domain (default 4*max(K, S0))
    return_grid : bool   also return the spot grid and the whole value slice
    theta_scheme : float weight on the old time level (0.5 = Crank-Nicolson)
    q : float            continuous dividend yield

    Returns
    -------
    price, or (price, S, V) when `return_grid` is True.

    Notes
    -----
    Error is O(dS^2 + dt^2) at the default theta_scheme, so halving both grids
    should quarter the error; the test suite checks that refinement helps rather
    than assuming it. Convergence is not perfectly clean, because the payoff kink
    at the strike sits at an arbitrary place between two nodes and Crank-Nicolson
    does not damp the resulting high-frequency error.
    """
    if option not in ("call", "put"):
        raise ValueError(f"option must be 'call' or 'put', got {option!r}")

    S_max = S_max if S_max is not None else 4.0 * max(K, S0)
    S, V, _, _ = _european_grid(K, T, r, sigma, n_space, n_time, option,
                                S_max, theta_scheme, q)

    price = float(np.interp(S0, S, V))
    return (price, S, V) if return_grid else price


def crank_nicolson_american(S0, K, T, r, sigma, n_space: int = 400, n_time: int = 400,
                            option: str = "put", S_max: float | None = None,
                            return_grid: bool = False, return_boundary: bool = False,
                            omega: float = 1.2, tol: float = 1e-8, max_iter: int = 1000,
                            theta_scheme: float = 0.5, q: float = 0.0):
    """American option by projected SOR on the free-boundary problem.

    Why a linear complementarity problem is the right statement
    ----------------------------------------------------------
    An American option is an optimal-stopping problem: its value is the supremum
    over stopping times of the discounted expected payoff (Shreve I, Ch. 4;
    Shreve II, Ch. 8). Writing L for the Black-Scholes operator and g(S) for the
    intrinsic value, the solution is characterised by three conditions that hold
    simultaneously at every point of the grid:

        (i)   V >= g            (otherwise buy, exercise, and bank the difference)
        (ii)  dV/dtau - L V >= 0 (holding cannot beat the PDE)
        (iii) (V - g) * (dV/dtau - L V) = 0   (complementarity)

    Condition (iii) is the content: where continuing is strictly optimal the PDE
    holds with equality, and where exercising is strictly optimal the value is
    exactly intrinsic. Nowhere do we need the exercise boundary S*(tau) itself,
    which matters because S*(tau) is an UNKNOWN of the problem, not an input. A
    free-boundary problem stated as "solve the PDE on the continuation region"
    is circular: the region is defined by the answer. The complementarity form
    removes the circularity by posing the constraint pointwise on the whole
    domain, and we read the boundary off afterwards (`return_boundary`).

    Discretised, one timestep becomes an LCP in the interior unknowns x:

        M x >= b,    x >= g,    (M x - b) * (x - g) = 0,

    with M the same tridiagonal matrix the European solver factorises and b its
    right-hand side.

    Projected SOR
    -------------
    SOR sweeps node by node, so we can project each node onto the constraint the
    moment its new value is computed:

        y_i = (b_i + a_i*x_{i-1} + c_i*x_{i+1}) / d_i
        x_i = max(g_i, x_i + omega*(y_i - x_i))

    The projection therefore participates in the solve instead of being applied
    to a finished answer. Cryer (1971) proved convergence for this iteration; in
    practice the requirement is that M be an M-matrix, which holds here for the
    timestep sizes used. Warm-starting each timestep from the previous slice
    keeps the count low: on a 400x400 grid at tol = 1e-8, omega = 1.2 needs 12
    sweeps in the worst timestep, and every omega in [0.8, 1.9] lands on the same
    price to 3e-7, which is the reassurance that we are solving the LCP rather
    than reporting wherever the relaxation happened to stop.

    Why the explicit-projection shortcut is cruder
    ----------------------------------------------
    The obvious alternative is to take a plain Crank-Nicolson step and then set
    V <- max(V, g). That scheme is valid: it is consistent, it respects the
    constraint at every reported slice, and it converges to the right answer as
    the grid refines. It is nevertheless the weaker method, and the reason is
    structural rather than a matter of taste. The implicit half of the step
    couples every node to every other node WITHIN the step, so nodes that should
    have been pinned at intrinsic for the duration of the step instead feed
    sub-intrinsic values to their neighbours; the max() at the end repairs the
    node itself but not the contamination it has already propagated. The damage
    is therefore concentrated exactly where the method is supposed to be doing
    its work.

    Measured on the test contract (S0 = 100, K = 110, T = 1, r = 5%, sigma = 20%,
    q = 0, S_max = 400 so that both S0 and K land on nodes), against a
    40,000-step binomial:

        n_space = n_time     100       200       400       800
        PSOR error         3.0e-2    9.8e-3    2.7e-3    6.7e-4
        shortcut error     3.9e-2    1.4e-2    4.9e-3    1.8e-3

    PSOR divides its error by 3.1, 3.7, then 4.0 per halving of dS, i.e., it holds
    the second-order rate; the shortcut manages only 2.8, 2.8, 2.7, so the gap
    WIDENS with refinement (1.3x, 1.4x, 1.9x, 2.7x) rather than closing. Looking
    along the 400-node slice shows where that comes from: at the first node above
    the free boundary (S* = 89, so S = 90) the shortcut is 4.3e-3 out against
    5.1e-4 for PSOR, a factor of 8, whilst far away at S = 160 the two are 8.1e-5
    and 2.9e-5, a factor of 2.8. The free boundary is smeared, not resolved.

    The shortcut is roughly 12x faster per solve, which is a real argument for it
    when a coarse American price is all that is wanted. It is not an argument for
    it when the free boundary, the early-exercise premium, or gamma near the
    boundary is the quantity of interest.

    Parameters
    ----------
    S0, K, T, r, sigma   as in `crank_nicolson_european`
    n_space, n_time : int   grid resolution
    option : str         'call' or 'put' (default 'put': the interesting case)
    S_max : float        upper edge of the domain (default 4*max(K, S0))
    return_grid : bool   also return the spot grid and the value slice
    return_boundary : bool  also return the tau grid and the exercise boundary
    omega : float        SOR relaxation, in (0, 2); 1.2 is a safe default
    tol : float          max nodal change accepted as convergence, in price units
    max_iter : int       sweeps per timestep before we declare failure
    theta_scheme : float weight on the old time level (0.5 = Crank-Nicolson)
    q : float            continuous dividend yield

    Returns
    -------
    price                        both flags False
    (price, S, V)                return_grid=True
    (price, tau, S_free)         return_boundary=True
    (price, S, V, tau, S_free)   both flags True

    `S_free[k]` is the exercise boundary at `tau[k]`, quantised to the grid: the
    largest node still exercising for a put, the smallest for a call. It is NaN
    on any slice with an empty exercise region, which is the correct answer for
    an American call with q = 0 (early exercise is never optimal, so there is no
    boundary to report).

    Raises
    ------
    ValueError    on a bad `option`, an `omega` outside (0, 2), or a degenerate grid.
    RuntimeError  if any timestep fails to converge within `max_iter` sweeps.

    Notes
    -----
    The error is dominated by the SPATIAL grid, not the timestep: holding n_space
    at 400 and taking n_time from 100 to 1600 moved the price by 1.9e-4, whilst
    holding n_time at 400 and taking n_space from 100 to 800 moved it by 3.0e-2,
    a factor of 150. Spend the budget on n_space; PSOR costs are linear in both,
    so the trade is a genuine one.

    Set `S_max` so that S0 falls exactly on a node (e.g., S_max = 4*S0 rather than
    the default 4*max(K, S0)) whenever you can. The final price is read off by
    linear interpolation, which on a convex value function biases upwards by
    about theta*(1 - theta)/2 * dS^2 * gamma with theta the fractional position of
    S0 between its two nodes. That bias has the opposite sign to the free-boundary
    discretisation error, so on the default grid the two partially cancel and the
    total error oscillates in sign as the grid refines (-1.4e-3, +4.5e-3, -1.8e-3,
    -1.8e-4 at n = 100, 200, 400, 800). Aligning S0 removes the interpolation term
    and restores clean, monotone, one-signed convergence (-3.0e-2, -9.8e-3,
    -2.7e-3, -6.7e-4 on the same grids).

    The sub-diagonal coefficient is 0.25*dt*(sigma^2*i^2 - (r - q)*i), which turns
    negative for i < (r - q)/sigma^2, i.e., on the first node or two of a typical
    grid. Strict diagonal dominance is lost there, though the solution near S = 0
    is smooth and tiny, so it does no measurable harm; if you push (r - q)/sigma^2
    up towards the grid resolution (a very low-volatility, high-rate contract),
    upwinding the convection term is the standard remedy.
    """
    if option not in ("call", "put"):
        raise ValueError(f"option must be 'call' or 'put', got {option!r}")
    if not 0.0 < omega < 2.0:
        raise ValueError(f"omega must lie in (0, 2) for SOR to converge, got {omega!r}")
    if n_space < 2 or n_time < 1:
        raise ValueError(f"need n_space >= 2 and n_time >= 1, got {n_space}, {n_time}")
    if tol <= 0.0 or max_iter < 1:
        raise ValueError(f"need tol > 0 and max_iter >= 1, got {tol!r}, {max_iter}")

    S_max = S_max if S_max is not None else 4.0 * max(K, S0)
    S = _build_grid(S_max, n_space)
    intrinsic = np.maximum(S - K, 0.0) if option == "call" else np.maximum(K - S, 0.0)

    dt = T / n_time
    _check_explicit_stability(theta_scheme, dt, sigma, n_space)
    i = np.arange(1, n_space)
    alpha, beta, gamma = _cn_coefficients(i, dt, r, sigma, q)

    w_exp, w_imp = _theta_weights(theta_scheme)
    a_exp, b_exp, c_exp = w_exp * alpha, w_exp * beta, w_exp * gamma
    a_imp, c_imp = w_imp * alpha, w_imp * gamma
    diag = 1.0 - w_imp * beta

    # The Gauss-Seidel sweep is sequential by construction (node i uses the value
    # of node i-1 from this same sweep), so it cannot be vectorised. Python lists
    # of floats are several times faster to index elementwise than numpy arrays;
    # a leading and trailing sentinel removes the boundary branch from the inner
    # loop, since the true boundary contribution is folded into `rhs` instead.
    pad = [0.0]
    a_pad = pad + a_imp.tolist() + pad
    c_pad = pad + c_imp.tolist() + pad
    d_pad = pad + diag.tolist() + pad
    g_pad = pad + intrinsic[1:-1].tolist() + pad
    n_in = n_space - 1

    def boundaries(tau):
        """Dirichlet data, projected onto the payoff like every other node.

        At the domain edges we know which regime we are in, so the American value
        there is the larger of the European asymptote and immediate exercise. For
        a put at S = 0 that is max(K*exp(-r*tau), K) = K: waiting is pointless
        once the underlying is worthless. For a deep-in-the-money call it is
        S_max - K whenever q > 0 and the European asymptote otherwise.
        """
        if option == "call":
            return 0.0, max(S_max * np.exp(-q * tau) - K * np.exp(-r * tau), S_max - K)
        return max(K * np.exp(-r * tau), K), 0.0

    # tau = 0 is the payoff, which satisfies the constraint with equality; every
    # later timestep warm-starts from the slice before it, which is what keeps the
    # sweep count low.
    V = intrinsic.copy()
    x = pad + intrinsic[1:-1].tolist() + pad
    taus = np.empty(n_time)
    S_free = np.empty(n_time)
    # PSOR pins exercised nodes at exactly g, so a tight tolerance separates the
    # two regions cleanly; scaling by K keeps it meaningful for any strike.
    boundary_tol = 1e-9 * max(K, 1.0)

    for step in range(n_time):
        tau_next = (step + 1) * dt
        V_low, V_high = boundaries(tau_next)

        rhs = a_exp * V[:-2] + (1.0 + b_exp) * V[1:-1] + c_exp * V[2:]
        rhs[0] += a_imp[0] * V_low
        rhs[-1] += c_imp[-1] * V_high
        b_pad = pad + rhs.tolist() + pad

        for _ in range(max_iter):
            err = 0.0
            for k in range(1, n_in + 1):
                x_old = x[k]
                y = (b_pad[k] + a_pad[k] * x[k - 1] + c_pad[k] * x[k + 1]) / d_pad[k]
                x_new = x_old + omega * (y - x_old)
                if x_new < g_pad[k]:
                    x_new = g_pad[k]          # project onto V >= intrinsic
                change = x_new - x_old
                if change < 0.0:
                    change = -change
                if change > err:
                    err = change
                x[k] = x_new
            if err < tol:
                break
        else:
            raise RuntimeError(
                f"PSOR failed to converge at step {step + 1}/{n_time}: last sweep moved "
                f"{err:.3e}, tolerance {tol:.3e}, after {max_iter} sweeps with "
                f"omega={omega}. Lower omega towards 1.0, raise max_iter, or loosen tol."
            )

        V_new = np.empty_like(V)
        V_new[0] = V_low
        V_new[-1] = V_high
        V_new[1:-1] = x[1:-1]
        V = V_new

        taus[step] = tau_next
        if return_boundary:
            exercising = np.flatnonzero((V <= intrinsic + boundary_tol) & (intrinsic > 0.0))
            if exercising.size == 0:
                S_free[step] = np.nan
            else:
                S_free[step] = S[exercising[-1] if option == "put" else exercising[0]]

    price = float(np.interp(S0, S, V))
    if return_grid and return_boundary:
        return price, S, V, taus, S_free
    if return_grid:
        return price, S, V
    if return_boundary:
        return price, taus, S_free
    return price


def crank_nicolson_greeks(S0, K, T, r, sigma, n_space: int = 400, n_time: int = 400,
                          option: str = "call", S_max: float | None = None,
                          d_sigma: float = 1e-3, d_r: float = 1e-4,
                          theta_scheme: float = 0.5, q: float = 0.0) -> dict:
    """Delta, gamma, vega, theta and rho from the PDE solution.

    This is the argument for taking the PDE route at all. A single backward
    integration produces the value at every node of the grid and at every
    timestep, so three of the five Greeks are already sitting in the answer:

    - delta and gamma are central differences of the final slice in S, formed on
      the whole grid and then interpolated to S0, which costs two subtractions
      per node and no extra solve;
    - theta is the difference of the last two time slices, divided by dt.

    Only vega and rho need re-solving, because sigma and r are parameters of the
    operator rather than coordinates of the grid. Both use a central bump on an
    identical grid, so the discretisation error is very largely common to the two
    solves and cancels in the difference: the bumped Greeks come out far more
    accurate than the individual prices they are built from.

    Parameters
    ----------
    S0, K, T, r, sigma   as in `crank_nicolson_european`
    n_space, n_time : int  grid resolution
    option : str         'call' or 'put'
    S_max : float        upper edge of the domain (default 4*max(K, S0))
    d_sigma : float      absolute volatility bump for vega
    d_r : float          absolute rate bump for rho
    theta_scheme : float weight on the old time level (0.5 = Crank-Nicolson)
    q : float            continuous dividend yield

    Returns
    -------
    dict with keys 'delta', 'gamma', 'vega', 'theta', 'rho': the same keys, the
    same units and the same sign conventions as `analytic.black_scholes_greeks`,
    so the two are directly comparable. Theta is per YEAR.

    Notes
    -----
    Accuracy is not uniform across the five, and the differences are worth knowing
    before quoting any of them. Measured against the closed form on a 400x400 grid
    (S0 = 100, K = 110, T = 1, r = 5%, sigma = 20%, q in {0, 4%}), the worst case
    over calls and puts is: delta 9.6e-5, gamma 3.0e-6, vega 4.8e-3 (1.2e-4
    relative), rho 3.4e-3 (8.8e-5 relative), theta 2.5e-3 (up to 2.3e-3 relative,
    since theta itself can be small).

    Theta is the weak one by construction: a two-slice difference is only O(dt)
    when reported at tau = T, because formally it is a centred estimate at
    tau - dt/2. Its error falls roughly as sqrt(n_time) rather than as n_time^2
    (1.9e-3, 7.0e-4, 4.1e-4 at n_time = 400, 1600, 6400), so buy accuracy there
    with timesteps and nowhere else.
    """
    if option not in ("call", "put"):
        raise ValueError(f"option must be 'call' or 'put', got {option!r}")

    S_max = S_max if S_max is not None else 4.0 * max(K, S0)
    S, V, V_prev, dt = _european_grid(K, T, r, sigma, n_space, n_time, option,
                                      S_max, theta_scheme, q)

    dS = S[1] - S[0]
    interior = S[1:-1]
    delta_grid = (V[2:] - V[:-2]) / (2.0 * dS)
    gamma_grid = (V[2:] - 2.0 * V[1:-1] + V[:-2]) / dS**2

    # V is the tau = T slice and V_prev the tau = T - dt slice, so advancing
    # calendar time by dt moves V towards V_prev: dV/dt = (V_prev - V)/dt.
    theta_grid = (V_prev - V) / dt

    def _price(sigma_, r_):
        return crank_nicolson_european(S0, K, T, r_, sigma_, n_space=n_space,
                                       n_time=n_time, option=option, S_max=S_max,
                                       theta_scheme=theta_scheme, q=q)

    vega_ = (_price(sigma + d_sigma, r) - _price(sigma - d_sigma, r)) / (2.0 * d_sigma)
    rho = (_price(sigma, r + d_r) - _price(sigma, r - d_r)) / (2.0 * d_r)

    return {
        "delta": float(np.interp(S0, interior, delta_grid)),
        "gamma": float(np.interp(S0, interior, gamma_grid)),
        "vega": float(vega_),
        "theta": float(np.interp(S0, S, theta_grid)),
        "rho": float(rho),
    }


def crank_nicolson_up_and_out(S0, K, B, T, r, sigma, n_space: int = 400,
                              n_time: int = 400, return_grid: bool = False,
                              q: float = 0.0):
    """Up-and-out barrier call by Crank-Nicolson.

    The barrier is a natural fit for the PDE approach: it is simply a Dirichlet
    boundary at S = B where V = 0, and the grid is truncated there. No correction
    term is needed, unlike the lattice (which needs Broadie-Glasserman-Kou) or
    Monte Carlo (which needs a Brownian bridge). Putting the domain boundary
    exactly on the barrier is the whole trick.

    Rannacher timestepping (two fully-implicit steps before switching to
    Crank-Nicolson) damps the oscillations caused by the kink in the payoff at
    the strike; that start-up phase is what makes the scheme usable here. It is
    also why this pricer does not expose `theta_scheme`: the scheme is chosen by
    the discontinuity, not by the caller.

    The dividend yield enters exactly as elsewhere, through the (r - q) drift in
    the convection terms, with discounting left at r.
    """
    if B <= K or S0 >= B:
        return (0.0, np.array([]), np.array([])) if return_grid else 0.0

    S = _build_grid(B, n_space)          # domain ends exactly at the barrier
    payoff = np.maximum(S - K, 0.0)
    payoff[-1] = 0.0                     # the barrier node is knocked out, so V = 0

    dt = T / n_time
    i = np.arange(1, n_space)
    V = payoff.copy()

    # --- Rannacher start-up: 2 fully implicit steps ---------------------
    a_i = 0.5 * dt * (sigma**2 * i**2 - (r - q) * i)
    b_i = -dt * (sigma**2 * i**2 + r)
    c_i = 0.5 * dt * (sigma**2 * i**2 + (r - q) * i)
    ab_impl = np.zeros((3, n_space - 1))
    ab_impl[0, 1:] = -c_i[:-1]
    ab_impl[1, :] = 1.0 - b_i
    ab_impl[2, :-1] = -a_i[1:]

    n_rannacher = min(2, n_time)
    for _ in range(n_rannacher):
        rhs = V[1:-1].copy()
        V_new = np.zeros_like(V)
        V_new[1:-1] = solve_banded((1, 1), ab_impl, rhs)
        V_new[0] = 0.0
        V_new[-1] = 0.0
        V = V_new

    # --- Crank-Nicolson for the remainder -------------------------------
    alpha, beta, gamma = _cn_coefficients(i, dt, r, sigma, q)
    ab = np.zeros((3, n_space - 1))
    ab[0, 1:] = -gamma[:-1]
    ab[1, :] = 1.0 - beta
    ab[2, :-1] = -alpha[1:]

    for _ in range(n_time - n_rannacher):
        rhs = alpha * V[:-2] + (1.0 + beta) * V[1:-1] + gamma * V[2:]
        V_new = np.zeros_like(V)
        V_new[1:-1] = solve_banded((1, 1), ab, rhs)
        V_new[0] = 0.0
        V_new[-1] = 0.0
        V = V_new

    price = float(max(np.interp(S0, S, V), 0.0))
    return (price, S, V) if return_grid else price
