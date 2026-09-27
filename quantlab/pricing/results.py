# quantlab: Eugene (Yoogeun) Song. MIT licensed; see LICENSE.
"""What an engine hands back.

A bare float hides the one thing a numerical price must carry with it: how
much to trust it. Every engine returns a `PricingResult`, and for stochastic
engines `stderr` is mandatory rather than optional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["PricingResult"]


@dataclass(frozen=True)
class PricingResult:
    value: float
    engine: str
    stderr: float | None = None
    greeks: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __float__(self) -> float:
        return float(self.value)

    @property
    def confidence_interval(self) -> tuple[float, float] | None:
        """95% interval, when the engine reports a standard error."""
        if self.stderr is None:
            return None
        return self.value - 1.96 * self.stderr, self.value + 1.96 * self.stderr

    def __repr__(self) -> str:
        se = f" ± {self.stderr:.4g}" if self.stderr is not None else ""
        return f"PricingResult({self.value:.6f}{se}, engine={self.engine!r})"
