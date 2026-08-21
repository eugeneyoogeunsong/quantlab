# quantlab: Eugene (Yoogeun) Song (https://www.linkedin.com/in/yoogeunsong)
# Independent side project. MIT licensed; see LICENSE.

"""Derivatives pricing.

Options priced by four independent routes (analytic, lattice, PDE, and
simulation), plus implied volatility inversion.

Each method is benchmarked against the closed-form solution, so the package
doubles as a convergence study: agreement to machine precision on a vanilla
European call is the entry requirement for trusting a method on anything harder.

Why four methods for the same number
------------------------------------
Because agreement between independent methods is evidence, and disagreement is
a bug report. The analytic Black-Scholes price is exact, but it exists only for
simple payoffs; binomial trees handle early exercise; finite differences give
the whole price surface; Monte Carlo handles path dependence and high
dimensions. Wherever two of them overlap they must agree, and the test suite
requires it.

What each route is actually for
-------------------------------
The four are not interchangeable, and the differences are the point:

- **Analytic** (``analytic``): exact, instant, and available only for the
  handful of payoffs whose expectation integrates in closed form. It is the
  reference every other method is scored against, following the risk-neutral
  expectation under the measure change (Shreve II, Ch. 5) and its Feynman-Kac
  equivalence with the Black-Scholes PDE (Ch. 6).
- **Lattice** (``binomial``): discrete replication in the sense of Shreve I,
  Ch. 1 to 2, which makes early exercise natural (Ch. 4) because the optimal
  stopping decision is taken node by node. Four parameterisations are provided;
  Leisen-Reimer converges at O(1/n^2) and monotonically, whilst
  Cox-Ross-Rubinstein oscillates at O(1/n), which is why extrapolation helps one
  and misleads on the other.
- **PDE** (``finite_difference``): solves for the whole price surface rather
  than a single number, so delta and gamma cost nothing extra. American
  exercise becomes a linear complementarity problem, solved here by projected
  SOR (Wilmott, Ch. 8).
- **Simulation** (``monte_carlo``): the only route that scales to path
  dependence and high dimension. Slowest and least accurate on a vanilla
  European, which is exactly why it should never be used for one. Includes
  Longstaff-Schwartz for American exercise, control variates, and pathwise and
  likelihood-ratio Greeks (Glasserman, Ch. 4 and 7).

A continuous dividend yield ``q`` is supported throughout: the risk-neutral
drift becomes ``r - q`` whilst discounting stays at ``r``. Setting ``q = 0.0``
reproduces the no-dividend case exactly.

    from quantlab.derivatives import black_scholes_call, binomial_european
    black_scholes_call(S=100, K=110, T=1.0, r=0.05, sigma=0.2)
"""

from .analytic import (
    black_scholes_call,
    black_scholes_greeks,
    black_scholes_put,
    black_scholes_put_call_parity_residual,
    digital_call,
    digital_put,
    up_and_out_call_closed_form,
    vega,
)
from .binomial import (
    binomial_american,
    binomial_european,
    binomial_greeks,
    binomial_richardson,
    binomial_tree_full,
    binomial_up_and_out,
    trinomial_up_and_out,
)
from .finite_difference import (
    crank_nicolson_american,
    crank_nicolson_european,
    crank_nicolson_greeks,
    crank_nicolson_up_and_out,
)
from .implied_vol import implied_vol_surface, implied_volatility
from .monte_carlo import (
    longstaff_schwartz_american,
    monte_carlo_european,
    monte_carlo_greeks,
    monte_carlo_up_and_out,
    simulate_gbm_paths,
)

__all__ = [
    # analytic
    "black_scholes_call",
    "black_scholes_put",
    "black_scholes_greeks",
    "black_scholes_put_call_parity_residual",
    "digital_call",
    "digital_put",
    "vega",
    "up_and_out_call_closed_form",
    # lattice
    "binomial_european",
    "binomial_american",
    "binomial_greeks",
    "binomial_richardson",
    "binomial_tree_full",
    "binomial_up_and_out",
    "trinomial_up_and_out",
    # PDE
    "crank_nicolson_european",
    "crank_nicolson_american",
    "crank_nicolson_greeks",
    "crank_nicolson_up_and_out",
    # simulation
    "monte_carlo_european",
    "monte_carlo_up_and_out",
    "monte_carlo_greeks",
    "longstaff_schwartz_american",
    "simulate_gbm_paths",
    # inversion
    "implied_volatility",
    "implied_vol_surface",
]
