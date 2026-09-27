# quantlab: Eugene (Yoogeun) Song. MIT licensed; see LICENSE.
"""The engine contract.

    engine.price(instrument, model) -> PricingResult

An engine is a pure function of a contract and a model. It holds numerical
settings (grid size, path count) but no market data -- that is the model's --
and no payoff rule -- that is the instrument's. This is what lets the test
suite line four of them up against one contract and demand agreement.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..instruments import Instrument
from ..models import BlackScholes
from ..results import PricingResult

__all__ = ["Engine", "UnsupportedInstrument"]


class UnsupportedInstrument(TypeError):
    """Raised when an engine has no procedure for a given contract type."""


@runtime_checkable
class Engine(Protocol):
    name: str

    def price(self, instrument: Instrument, model: BlackScholes) -> PricingResult: ...

    def supports(self, instrument: Instrument) -> bool: ...
