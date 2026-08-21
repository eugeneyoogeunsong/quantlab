# quantlab: Eugene (Yoogeun) Song (https://www.linkedin.com/in/yoogeunsong)
# Ported from Adrian's (Adrian.ph689) independent work, used with permission;
# see CREDITS.md. Independent side project. MIT licensed; see LICENSE.

"""Binomial lattice pricing (Cox-Ross-Rubinstein).

Original implementations by Adrian (Adrian.ph689), 2025, drawing on Shreve,
*Stochastic Calculus for Finance I* (2004). Refactored into library form.

The lattice discretises the underlying into up/down moves and works backwards
from expiry, discounting under the risk-neutral measure. Its advantage over the
closed form is that it can price American options: at every node we can ask
whether exercising now is worth more than holding, a question the Black-Scholes
PDE has no room for.

Analogy: Black-Scholes computes the value of a journey knowing only the
destination, whilst the lattice walks every fork in the road, which is the only
way to notice that stopping early is sometimes better.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "binomial_european",
    "binomial_american",
    "binomial_tree_full",
    "binomial_up_and_out",
]


def _crr_params(T: float, r: float, sigma: float, n_steps: int):
    """Cox-Ross-Rubinstein up/down factors and the risk-neutral probability.

    We set u = exp(sigma*sqrt(dt)) and d = 1/u, so the lattice recombines; p is
    then fixed by requiring the discounted spot to be a martingale. A p outside
    [0, 1] means the parameters do not admit a no-arbitrage lattice at all.
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    if not 0.0 <= p <= 1.0:
        raise ValueError(
            f"Risk-neutral probability p={p:.4f} outside [0,1]. The lattice is "
            f"arbitrageable: with dt={dt:.5f}, u={u:.5f}, d={d:.5f}, the rate r={r} "
            "moves the forward outside the up/down range. Use more steps or check "
            "that sigma is not far too small for r."
        )
    return dt, u, d, p


def _payoff(S, K: float, option: str):
    if option == "call":
        return np.maximum(S - K, 0.0)
    if option == "put":
        return np.maximum(K - S, 0.0)
    raise ValueError(f"option must be 'call' or 'put', got {option!r}")


def binomial_european(S0, K, T, r, sigma, n_steps: int = 500, option: str = "call") -> float:
    """European option by backward induction on a CRR lattice.

    Converges to Black-Scholes as `n_steps` grows, but not monotonically: the
    error oscillates depending on whether the strike sits near a lattice node.
    Averaging the n-step and (n+1)-step prices is the standard trick to damp it.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    dt, u, d, p = _crr_params(T, r, sigma, n_steps)
    disc = np.exp(-r * dt)

    j = np.arange(n_steps + 1)
    S_T = S0 * u ** (n_steps - j) * d**j
    V = _payoff(S_T, K, option)

    for _ in range(n_steps):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
    return float(V[0])


def binomial_american(S0, K, T, r, sigma, n_steps: int = 500, option: str = "put") -> float:
    """American option: the same backward induction, plus an early-exercise test.

    At each node the holder takes the better of continuing or exercising:

        V = max(discounted expected value, intrinsic value)

    For a non-dividend-paying underlying an American CALL is worth exactly what
    its European twin is worth, since exercising early throws away both time
    value and the interest earned on the strike, so it is never optimal. The
    American PUT is genuinely worth more, because exercising frees up the strike
    in cash early. Both facts are asserted in the tests, which makes this a real
    check on the implementation rather than a comment.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    dt, u, d, p = _crr_params(T, r, sigma, n_steps)
    disc = np.exp(-r * dt)

    j = np.arange(n_steps + 1)
    S_T = S0 * u ** (n_steps - j) * d**j
    V = _payoff(S_T, K, option)

    for step in range(n_steps - 1, -1, -1):
        j = np.arange(step + 1)
        S_node = S0 * u ** (step - j) * d**j
        continuation = disc * (p * V[:-1] + (1 - p) * V[1:])
        V = np.maximum(continuation, _payoff(S_node, K, option))
    return float(V[0])


def binomial_tree_full(S0, K, T, r, sigma, n_steps: int = 6,
                       option: str = "put", american: bool = True):
    """Return the full lattice, for plotting or inspection.

    Returns
    -------
    stock    : (n+1, n+1) upper-triangular stock price lattice
    value    : (n+1, n+1) option value lattice
    exercise : (n+1, n+1) bool, True where early exercise beats holding
               (all False when american=False)

    Intended for small `n_steps`: this is the teaching and diagnostic view. Use
    `binomial_european` / `binomial_american` for actual pricing.
    """
    dt, u, d, p = _crr_params(T, r, sigma, n_steps)
    disc = np.exp(-r * dt)
    n = n_steps

    stock = np.zeros((n + 1, n + 1))
    for step in range(n + 1):
        j = np.arange(step + 1)
        stock[: step + 1, step] = S0 * u ** (step - j) * d**j

    value = np.zeros_like(stock)
    exercise = np.zeros_like(stock, dtype=bool)
    value[: n + 1, n] = _payoff(stock[: n + 1, n], K, option)

    for step in range(n - 1, -1, -1):
        cont = disc * (p * value[: step + 1, step + 1] + (1 - p) * value[1 : step + 2, step + 1])
        if american:
            intrinsic = _payoff(stock[: step + 1, step], K, option)
            value[: step + 1, step] = np.maximum(cont, intrinsic)
            exercise[: step + 1, step] = intrinsic > cont
        else:
            value[: step + 1, step] = cont
    return stock, value, exercise


# Broadie-Glasserman-Kou continuity-correction constant, equal to
# -zeta(1/2)/sqrt(2*pi), where zeta is the Riemann zeta function.
BGK_BETA = 0.5825971579390106


def trinomial_up_and_out(S0, K, B, T, r, sigma, n_steps: int = 500) -> float:
    """Up-and-out barrier call on a Ritchken (1995) stretched trinomial tree.

    THIS is the lattice method to use for barriers; `binomial_up_and_out` is
    kept for comparison, but it converges badly (see its docstring).

    The problem with a binomial tree
    --------------------------------
    A CRR lattice can only represent prices S0*u^k for integer k. The barrier
    almost never lands on one, so the barrier the lattice actually enforces is
    displaced by up to half a level. Barrier prices are acutely sensitive to
    that displacement: a 0.3% error in barrier location moved the price by ~2%
    in testing.

    The alignment can be fixed by choosing u = (B/S0)^(1/m), but then u no
    longer equals exp(sigma*sqrt(dt)) and the lattice's effective volatility
    drifts by ~1%, which is just as damaging. With only two branches there are
    not enough free parameters to match the mean, the variance, AND the barrier
    position at once.

    Ritchken's solution
    -------------------
    A third branch supplies the missing degree of freedom. We introduce a
    stretch parameter lambda >= 1 with up-move exp(lambda*sigma*sqrt(dt)), and
    choose lambda so that exactly m up-moves land on the barrier:

        m = floor( ln(B/S0) / (sigma*sqrt(dt)) )
        lambda = ln(B/S0) / (m*sigma*sqrt(dt))

    The branch probabilities then still match the mean and variance of the
    log-price exactly:

        pu = 1/(2*lambda^2) + (r - sigma^2/2)*sqrt(dt) / (2*lambda*sigma)
        pd = 1/(2*lambda^2) - (r - sigma^2/2)*sqrt(dt) / (2*lambda*sigma)
        pm = 1 - 1/lambda^2

    Barrier aligned exactly, volatility matched exactly; convergence becomes
    smooth and roughly O(1/n).

    Analogy: a binomial tree is a ruler with fixed markings, so to measure to a
    line falling between them you must either move the line or stretch the
    ruler, and both distort something. The trinomial adds an adjustable marking.
    """
    if B <= K or S0 >= B:
        return 0.0
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    log_ratio = np.log(B / S0)

    m = int(np.floor(log_ratio / (sigma * sqrt_dt)))
    if m < 1:
        # Barrier sits within a single up-move of spot: the grid is too coarse
        # to resolve it, so we decline to quote rather than return noise.
        return 0.0
    lam = log_ratio / (m * sigma * sqrt_dt)

    drift_term = (r - 0.5 * sigma**2) * sqrt_dt / (2.0 * lam * sigma)
    pu = 1.0 / (2.0 * lam**2) + drift_term
    pd = 1.0 / (2.0 * lam**2) - drift_term
    pm = 1.0 - 1.0 / lam**2
    if min(pu, pd, pm) < -1e-12:
        raise ValueError(
            f"Negative trinomial probability (pu={pu:.4f}, pm={pm:.4f}, pd={pd:.4f}). "
            "Increase n_steps."
        )

    disc = np.exp(-r * dt)
    dx = lam * sigma * sqrt_dt

    # Levels reachable at the final step, capped at the barrier level m: anything
    # at or above level m is knocked out, so we never need to carry it.
    max_level = m                     # knocked out at exactly this level
    min_level = -n_steps
    levels = np.arange(min_level, max_level + 1)
    S_nodes = S0 * np.exp(levels * dx)

    V = np.maximum(S_nodes - K, 0.0)
    V[levels >= m] = 0.0              # knocked out

    for _ in range(n_steps):
        # A node at level k branches to k+1, k, and k-1; the rolls line those
        # three destinations up so the step is one vectorised combination.
        up = np.roll(V, -1)
        up[-1] = 0.0
        down = np.roll(V, 1)
        down[0] = 0.0
        V = disc * (pu * up + pm * V + pd * down)
        V[levels >= m] = 0.0          # barrier enforced at every step

    return float(V[np.searchsorted(levels, 0)])


def binomial_up_and_out(S0, K, B, T, r, sigma, n_steps: int = 500,
                        bgk_adjust: bool = True) -> float:
    """Up-and-out barrier call on a CRR binomial lattice.

    Prefer `trinomial_up_and_out` for real use. This function is retained
    because comparing the two is instructive, and because the original study
    this code came from benchmarked exactly these methods against each other.

    KNOWN LIMITATION (this is not a bug)
    ------------------------------------
    The binomial price for a barrier option converges slowly and
    NON-MONOTONICALLY. Measured against the closed form, the error moved
    1.2e-1 -> 1.4e-2 -> 2.7e-2 as steps went 500 -> 2000 -> 5000: more work,
    worse answer.

    The cause is structural, not a coding error. A CRR lattice can only take the
    values S0*u^k, and the barrier generally falls between two of them, so the
    barrier actually enforced is displaced by up to half a level; changing
    n_steps changes u, which moves that displacement erratically. Fixing it by
    choosing u = (B/S0)^(1/m) aligns the barrier but throws the lattice's
    effective volatility off by ~1%, which does comparable damage. Two branches
    simply do not provide enough freedom to satisfy both constraints; Ritchken's
    trinomial adds a third and does.

    Two refinements ARE applied here, and both help:

    1. Node prices are built from integer levels (``S0 * u**k``) rather than
       ``u**(step-j) * d**j``. The two are mathematically identical but not
       bitwise identical, so without this a node sitting exactly on the barrier
       is knocked out or not depending on floating-point rounding.

    2. `bgk_adjust` applies the Broadie-Glasserman-Kou continuity correction,
       shifting the lattice barrier to ``B*exp(-beta*sigma*sqrt(dt))`` with
       beta = -zeta(1/2)/sqrt(2*pi) ~ 0.5826. A lattice monitors the barrier
       only at its own timesteps, which makes knockout harder than under
       continuous monitoring and biases the price upward; the shift compensates
       for that, and it vanishes as dt -> 0, so the limit is unchanged.

    Original implementation by Adrian (Adrian.ph689), 2025.
    """
    if B <= K or S0 >= B:
        return 0.0

    dt_tmp = T / n_steps
    B_eff = B * np.exp(-BGK_BETA * sigma * np.sqrt(dt_tmp)) if bgk_adjust else B
    if B_eff <= S0:
        return 0.0

    dt, u, d, p = _crr_params(T, r, sigma, n_steps)
    disc = np.exp(-r * dt)

    def node_prices(step: int) -> np.ndarray:
        """Prices at `step`, built from integer levels so comparisons are exact."""
        levels = step - 2 * np.arange(step + 1)
        return S0 * u**levels

    # A relative tolerance, so a node mathematically ON the barrier is knocked
    # out consistently rather than flickering with floating-point rounding.
    def alive(S_node: np.ndarray) -> np.ndarray:
        return S_node < B_eff * (1.0 - 1e-9)

    S_T = node_prices(n_steps)
    V = np.where(alive(S_T), np.maximum(S_T - K, 0.0), 0.0)

    for step in range(n_steps - 1, -1, -1):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        V = np.where(alive(node_prices(step)), V, 0.0)
    return float(V[0])
