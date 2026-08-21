# quantlab: Eugene (Yoogeun) Song (https://www.linkedin.com/in/yoogeunsong)
# Ported from Adrian's (Adrian.ph689) independent work, used with permission;
# see CREDITS.md. Independent side project. MIT licensed; see LICENSE.

"""Monte Carlo pricing under geometric Brownian motion.

Original implementations by Adrian (Adrian.ph689), 2025. Refactored into
library form and vectorised across paths; the original looped path by path,
which is clearer to read but roughly two orders of magnitude slower.

Monte Carlo is the slowest and least accurate method here for a vanilla European
call, where a closed form exists. It earns its place on payoffs that depend on
the whole path, or on many underlyings at once, where the lattice and the PDE
both become intractable.

Its error shrinks as 1/sqrt(n_paths): 100x the compute buys 10x the accuracy.
That is a poor exchange rate, and the reason variance-reduction techniques
(antithetic variates, control variates) exist at all.
"""

from __future__ import annotations

import numpy as np

__all__ = ["simulate_gbm_paths", "monte_carlo_european", "monte_carlo_up_and_out"]


def simulate_gbm_paths(S0: float, T: float, r: float, sigma: float,
                       n_paths: int = 10_000, n_steps: int = 252,
                       seed: int | None = None,
                       antithetic: bool = False) -> np.ndarray:
    """Simulate GBM paths under the risk-neutral measure.

    Returns an array of shape (n_paths, n_steps + 1) including S0 at column 0.

    We step the exact lognormal solution rather than an Euler discretisation, so
    there is no time-discretisation bias in the terminal value, only sampling
    error.

    `antithetic=True` pairs each path with its mirror image (Z and -Z). Because
    the pair's errors are negatively correlated, the average is less noisy for
    the same number of draws; it requires an even `n_paths`.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    if antithetic:
        if n_paths % 2:
            raise ValueError("antithetic sampling requires an even n_paths")
        half = rng.standard_normal((n_paths // 2, n_steps))
        Z = np.concatenate([half, -half], axis=0)
    else:
        Z = rng.standard_normal((n_paths, n_steps))

    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1)
    return S0 * np.exp(log_paths)


def monte_carlo_european(S0, K, T, r, sigma, n_paths: int = 50_000,
                         n_steps: int = 252, option: str = "call",
                         seed: int | None = None, antithetic: bool = True,
                         return_stderr: bool = False):
    """European option by simulation.

    Returns the price, or `(price, standard_error)` if `return_stderr=True`.

    Always look at that standard error: a Monte Carlo price quoted without one is
    a number of unknown precision, and the 95% interval is roughly
    price +/- 2*stderr, which is usually wider than people expect.
    """
    paths = simulate_gbm_paths(S0, T, r, sigma, n_paths, n_steps, seed, antithetic)
    S_T = paths[:, -1]

    if option == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    elif option == "put":
        payoffs = np.maximum(K - S_T, 0.0)
    else:
        raise ValueError(f"option must be 'call' or 'put', got {option!r}")

    discounted = np.exp(-r * T) * payoffs
    price = float(discounted.mean())
    if return_stderr:
        return price, float(discounted.std(ddof=1) / np.sqrt(n_paths))
    return price


def monte_carlo_up_and_out(S0, K, B, T, r, sigma, n_paths: int = 50_000,
                           n_steps: int = 252, seed: int | None = None,
                           brownian_bridge: bool = True,
                           return_stderr: bool = False):
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

    Original implementation by Adrian (Adrian.ph689), 2025.
    """
    if B <= K or S0 >= B:
        return (0.0, 0.0) if return_stderr else 0.0

    paths = simulate_gbm_paths(S0, T, r, sigma, n_paths, n_steps, seed)
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

    discounted = np.exp(-r * T) * np.maximum(S_T - K, 0.0) * survival
    price = float(discounted.mean())
    if return_stderr:
        return price, float(discounted.std(ddof=1) / np.sqrt(n_paths))
    return price
