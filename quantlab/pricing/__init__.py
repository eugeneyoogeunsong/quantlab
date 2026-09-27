# quantlab: Eugene (Yoogeun) Song. MIT licensed; see LICENSE.
"""Option pricing: instruments, models, and four independent engines.

    from quantlab.pricing import VanillaOption, BlackScholes, AnalyticEngine

    contract = VanillaOption(expiry=1.0, strike=110, option_type="call")
    market   = BlackScholes(spot=100, rate=0.05, vol=0.20)
    AnalyticEngine().price(contract, market)          # PricingResult(6.040088, engine='analytic')

The three nouns are kept apart on purpose. An *instrument* is a payoff and an
exercise right and knows nothing about markets. A *model* is market data and
dynamics and knows nothing about contracts. An *engine* is a numerical
procedure that takes one of each. Because of that split, the same contract
can go through the closed form, a lattice, a PDE solve, and a simulation --
and the test suite makes them agree.

Engines
-------
AnalyticEngine     closed forms (Hull Ch. 15, 19, 26); the reference
LatticeEngine      trees with pluggable schemes: CRR, Jarrow-Rudd, Tian,
                   Leisen-Reimer, Ritchken trinomial (Hull Ch. 21)
PDEEngine          theta-scheme finite differences with PSOR for American
                   exercise (Wilmott Ch. 77-78)
MonteCarloEngine   simulation with antithetic / control variates, Brownian-
                   bridge barrier correction, Longstaff-Schwartz, pathwise
                   and likelihood-ratio Greeks (Glasserman Ch. 3-8)

Provenance
----------
This package replaces an earlier one that had ported Adrian Phillips-Hernaez's
public option-pricing scripts into quantlab. That code was removed. What is
here was written afresh from the textbooks cited above with a different
structure; see the README for the full account.
"""

from .calibration import implied_vol_smile, implied_vol_surface, implied_volatility
from .engines import AnalyticEngine, Engine, UnsupportedInstrument
from .engines.lattice import (
    CoxRossRubinstein,
    JarrowRudd,
    LatticeEngine,
    LeisenReimer,
    RitchkenTrinomial,
    Tian,
)
from .engines.montecarlo import MonteCarloEngine, PathGenerator
from .engines.pde import PDEEngine
from .greeks import bump_greeks
from .instruments import (
    BarrierOption,
    BarrierType,
    DigitalOption,
    Exercise,
    Instrument,
    OptionType,
    VanillaOption,
)
from .models import BlackScholes
from .results import PricingResult

__all__ = [
    # contracts
    "Instrument", "VanillaOption", "DigitalOption", "BarrierOption",
    "OptionType", "Exercise", "BarrierType",
    # model
    "BlackScholes",
    # engines
    "Engine", "UnsupportedInstrument", "AnalyticEngine", "LatticeEngine",
    "PDEEngine", "MonteCarloEngine", "PathGenerator",
    # lattice schemes
    "CoxRossRubinstein", "JarrowRudd", "Tian", "LeisenReimer", "RitchkenTrinomial",
    # results and tools
    "PricingResult", "implied_volatility", "implied_vol_smile", "implied_vol_surface",
    "bump_greeks",
]
