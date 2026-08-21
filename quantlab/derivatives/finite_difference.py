# quantlab: Eugene (Yoogeun) Song (https://www.linkedin.com/in/yoogeunsong)
# Ported from Adrian's (Adrian.ph689) independent work, used with permission;
# see CREDITS.md. Independent side project. MIT licensed; see LICENSE.

"""PDE pricing by Crank-Nicolson finite differences.

Original implementations by Adrian (Adrian.ph689), 2025. Refactored into
library form; the dense matrix solve of the original has been replaced with a
banded (tridiagonal) solver, which is the same arithmetic at O(n) instead of
O(n^3) per timestep.

Rather than simulating or enumerating outcomes, we solve the Black-Scholes PDE
directly on a grid of (spot, time): the payoff is the initial condition, and we
integrate backwards from expiry.

The reward for the extra machinery is that a single solve gives the option value
at EVERY spot price, not just today's, so delta and gamma come out as finite
differences of a surface we already have rather than requiring a re-run.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

__all__ = ["crank_nicolson_european", "crank_nicolson_up_and_out"]


def _build_grid(S_max: float, n_space: int):
    return np.linspace(0.0, S_max, n_space + 1)


def _cn_coefficients(S_idx: np.ndarray, dt: float, r: float, sigma: float):
    """Crank-Nicolson coefficients for interior nodes.

    Averaging the explicit and implicit schemes gives second-order accuracy in
    time and unconditional stability; the explicit scheme on its own is stable
    only for very small timesteps.
    """
    i = S_idx
    alpha = 0.25 * dt * (sigma**2 * i**2 - r * i)
    beta = -0.5 * dt * (sigma**2 * i**2 + r)
    gamma = 0.25 * dt * (sigma**2 * i**2 + r * i)
    return alpha, beta, gamma


def _solve_cn(payoff, S, T, r, sigma, n_time, upper_boundary_fn, knockout_mask=None):
    """Shared Crank-Nicolson time-stepping loop.

    We step in tau (time remaining to expiry), so `payoff` is the tau = 0 state
    and `upper_boundary_fn` is evaluated at the tau of the step being solved for.
    Passing `knockout_mask` zeroes the masked nodes after every step, which is
    how a knockout barrier is imposed.
    """
    n_space = len(S) - 1
    dt = T / n_time
    i = np.arange(1, n_space)
    alpha, beta, gamma = _cn_coefficients(i, dt, r, sigma)

    # Left-hand (implicit) tridiagonal matrix in banded storage.
    ab = np.zeros((3, n_space - 1))
    ab[0, 1:] = -gamma[:-1]      # super-diagonal
    ab[1, :] = 1.0 - beta        # main diagonal
    ab[2, :-1] = -alpha[1:]      # sub-diagonal

    V = payoff.copy()
    for step in range(n_time):
        tau_next = (step + 1) * dt
        V_low = 0.0
        V_high = upper_boundary_fn(tau_next)

        rhs = alpha * V[:-2] + (1.0 + beta) * V[1:-1] + gamma * V[2:]
        rhs[0] += alpha[0] * V_low
        rhs[-1] += gamma[-1] * V_high

        V_new = np.empty_like(V)
        V_new[0] = V_low
        V_new[-1] = V_high
        V_new[1:-1] = solve_banded((1, 1), ab, rhs)

        if knockout_mask is not None:
            V_new[knockout_mask] = 0.0
        V = V_new
    return V


def crank_nicolson_european(S0, K, T, r, sigma, n_space: int = 400,
                            n_time: int = 400, option: str = "call",
                            S_max: float | None = None,
                            return_grid: bool = False):
    """European option by Crank-Nicolson.

    `S_max` defaults to four times the larger of `K` and `S0`. It must sit far
    enough out that the boundary condition imposed there is essentially exact:
    too close, and the artificial boundary contaminates the interior solution.
    """
    if option not in ("call", "put"):
        raise ValueError(f"option must be 'call' or 'put', got {option!r}")

    S_max = S_max if S_max is not None else 4.0 * max(K, S0)
    S = _build_grid(S_max, n_space)

    if option == "call":
        payoff = np.maximum(S - K, 0.0)
        def upper(tau):
            return S_max - K * np.exp(-r * tau)
    else:
        payoff = np.maximum(K - S, 0.0)
        def upper(tau):
            return 0.0

    if option == "put":
        # For a put the informative boundary sits at S=0, where V = K*exp(-r*tau),
        # and the value decays to zero at large S; we therefore swap which end of
        # the grid carries the Dirichlet data and step the same scheme by hand.
        V = payoff.copy()
        dt = T / n_time
        i = np.arange(1, n_space)
        alpha, beta, gamma = _cn_coefficients(i, dt, r, sigma)
        ab = np.zeros((3, n_space - 1))
        ab[0, 1:] = -gamma[:-1]
        ab[1, :] = 1.0 - beta
        ab[2, :-1] = -alpha[1:]
        for step in range(n_time):
            tau_next = (step + 1) * dt
            V_low = K * np.exp(-r * tau_next)
            V_high = 0.0
            rhs = alpha * V[:-2] + (1.0 + beta) * V[1:-1] + gamma * V[2:]
            rhs[0] += alpha[0] * V_low
            rhs[-1] += gamma[-1] * V_high
            V_new = np.empty_like(V)
            V_new[0], V_new[-1] = V_low, V_high
            V_new[1:-1] = solve_banded((1, 1), ab, rhs)
            V = V_new
    else:
        V = _solve_cn(payoff, S, T, r, sigma, n_time, upper)

    price = float(np.interp(S0, S, V))
    return (price, S, V) if return_grid else price


def crank_nicolson_up_and_out(S0, K, B, T, r, sigma, n_space: int = 400,
                              n_time: int = 400, return_grid: bool = False):
    """Up-and-out barrier call by Crank-Nicolson.

    The barrier is a natural fit for the PDE approach: it is simply a Dirichlet
    boundary at S = B where V = 0, and the grid is truncated there. No correction
    term is needed, unlike the lattice (which needs Broadie-Glasserman-Kou) or
    Monte Carlo (which needs a Brownian bridge). Putting the domain boundary
    exactly on the barrier is the whole trick.

    Original implementation by Adrian (Adrian.ph689), 2025, which used Rannacher
    timestepping (two fully-implicit steps before switching to Crank-Nicolson)
    to damp the oscillations caused by the kink in the payoff at the strike.
    That start-up phase is preserved here.
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
    a_i = 0.5 * dt * (sigma**2 * i**2 - r * i)
    b_i = -dt * (sigma**2 * i**2 + r)
    c_i = 0.5 * dt * (sigma**2 * i**2 + r * i)
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
    alpha, beta, gamma = _cn_coefficients(i, dt, r, sigma)
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
