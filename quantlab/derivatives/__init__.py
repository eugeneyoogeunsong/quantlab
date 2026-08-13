"""Derivatives pricing.

Options pricing by four independent routes -- analytic, lattice, PDE and
simulation -- plus implied volatility inversion.

Original implementations by Adrian (Adrian.ph689), 2025, from a study
benchmarking numerical methods against closed-form solutions. Refactored here
from standalone scripts into importable, tested modules; the algorithms are
unchanged.

Why four methods for the same number
------------------------------------
Because agreement between independent methods is evidence, and disagreement is
a bug report. The analytic Black-Scholes price is exact but exists only for
simple payoffs. Binomial trees handle early exercise. Finite differences give
the whole price surface. Monte Carlo handles path dependence and high
dimensions. Where two of them overlap, they must agree -- and the test suite
requires it.

    from quantlab.derivatives import black_scholes_call, binomial_european
    black_scholes_call(S=100, K=110, T=1.0, r=0.05, sigma=0.2)
"""

from .analytic import (
    black_scholes_call,
    black_scholes_put,
    black_scholes_greeks,
    up_and_out_call_closed_form,
    vega,
)
from .binomial import (
    binomial_american,
    binomial_european,
    binomial_tree_full,
    binomial_up_and_out,
    trinomial_up_and_out,
)
from .finite_difference import crank_nicolson_european, crank_nicolson_up_and_out
from .implied_vol import implied_volatility, implied_vol_surface
from .monte_carlo import monte_carlo_european, monte_carlo_up_and_out, simulate_gbm_paths

__all__ = [
    # analytic
    "black_scholes_call",
    "black_scholes_put",
    "black_scholes_greeks",
    "vega",
    "up_and_out_call_closed_form",
    # lattice
    "binomial_european",
    "binomial_american",
    "binomial_tree_full",
    "binomial_up_and_out",
    "trinomial_up_and_out",
    # PDE
    "crank_nicolson_european",
    "crank_nicolson_up_and_out",
    # simulation
    "monte_carlo_european",
    "monte_carlo_up_and_out",
    "simulate_gbm_paths",
    # inversion
    "implied_volatility",
    "implied_vol_surface",
]
