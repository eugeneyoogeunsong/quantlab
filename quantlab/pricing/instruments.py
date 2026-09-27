# quantlab: Eugene (Yoogeun) Song. MIT licensed; see LICENSE.
"""Instruments: what a contract pays, and when its holder may act.

The separation this package is built on comes straight from how Hull organises
the subject. A *contract* is a payoff rule and an exercise right. A *model* is a
statement about how the underlying moves. An *engine* is a numerical procedure
that combines the two into a number. Keeping them apart means the same
`VanillaOption` can be priced by the closed form, a lattice, a PDE solve, or
simulation without the contract knowing which -- and the test suite can insist
they agree.

Nothing in this module knows about volatility, interest rates, or numerics.
Instruments answer exactly two questions:

    payoff(spot)          -> what do I receive if exercised at this spot?
    is_alive(path)        -> has anything along this path extinguished me?

Everything else lives elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

__all__ = [
    "OptionType",
    "Exercise",
    "BarrierType",
    "Instrument",
    "VanillaOption",
    "DigitalOption",
    "BarrierOption",
]


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"

    @property
    def sign(self) -> int:
        """+1 for a call, -1 for a put. Lets one formula serve both."""
        return 1 if self is OptionType.CALL else -1


class Exercise(str, Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


class BarrierType(str, Enum):
    """Direction the barrier sits relative to spot, and what touching it does."""

    UP_AND_OUT = "up-and-out"
    UP_AND_IN = "up-and-in"
    DOWN_AND_OUT = "down-and-out"
    DOWN_AND_IN = "down-and-in"

    @property
    def is_up(self) -> bool:
        return self in (BarrierType.UP_AND_OUT, BarrierType.UP_AND_IN)

    @property
    def is_knock_out(self) -> bool:
        return self in (BarrierType.UP_AND_OUT, BarrierType.DOWN_AND_OUT)

    @property
    def complement(self) -> "BarrierType":
        """The knock-in twin of a knock-out, and vice versa.

        In-out parity: an in and an out with the same barrier sum to the
        vanilla. Engines that can only price one kind use this to get the other.
        """
        return {
            BarrierType.UP_AND_OUT: BarrierType.UP_AND_IN,
            BarrierType.UP_AND_IN: BarrierType.UP_AND_OUT,
            BarrierType.DOWN_AND_OUT: BarrierType.DOWN_AND_IN,
            BarrierType.DOWN_AND_IN: BarrierType.DOWN_AND_OUT,
        }[self]


@dataclass(frozen=True)
class Instrument:
    """Base contract. `expiry` is in years."""

    expiry: float

    def __post_init__(self) -> None:
        if self.expiry <= 0:
            raise ValueError(f"expiry must be positive, got {self.expiry}")

    def payoff(self, spot: np.ndarray | float) -> np.ndarray | float:
        raise NotImplementedError

    @property
    def is_path_dependent(self) -> bool:
        return False

    @property
    def exercise(self) -> Exercise:
        return Exercise.EUROPEAN


@dataclass(frozen=True)
class VanillaOption(Instrument):
    """Plain call or put, European or American."""

    strike: float = 100.0
    option_type: OptionType = OptionType.CALL
    exercise_style: Exercise = Exercise.EUROPEAN

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.strike <= 0:
            raise ValueError(f"strike must be positive, got {self.strike}")
        # Accept plain strings for convenience at the call site.
        object.__setattr__(self, "option_type", OptionType(self.option_type))
        object.__setattr__(self, "exercise_style", Exercise(self.exercise_style))

    def payoff(self, spot):
        return np.maximum(self.option_type.sign * (np.asarray(spot, dtype=float) - self.strike), 0.0)

    @property
    def exercise(self) -> Exercise:
        return self.exercise_style

    @property
    def is_call(self) -> bool:
        return self.option_type is OptionType.CALL

    def with_exercise(self, style: Exercise) -> "VanillaOption":
        return VanillaOption(self.expiry, self.strike, self.option_type, style)


@dataclass(frozen=True)
class DigitalOption(Instrument):
    """Cash-or-nothing: pays `cash` if it finishes in the money, else nothing.

    The payoff is a step function, which is exactly what makes digitals a
    stress test for numerical methods -- a lattice or grid with a node sitting
    near the strike sees a discontinuity and converges erratically. Hull, Ch. 26.
    """

    strike: float = 100.0
    option_type: OptionType = OptionType.CALL
    cash: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.strike <= 0:
            raise ValueError(f"strike must be positive, got {self.strike}")
        object.__setattr__(self, "option_type", OptionType(self.option_type))

    def payoff(self, spot):
        s = np.asarray(spot, dtype=float)
        in_the_money = (s > self.strike) if self.option_type is OptionType.CALL else (s < self.strike)
        return np.where(in_the_money, self.cash, 0.0)

    @property
    def is_call(self) -> bool:
        return self.option_type is OptionType.CALL


@dataclass(frozen=True)
class BarrierOption(Instrument):
    """A vanilla whose existence depends on whether spot touches `barrier`.

    Continuous monitoring is assumed by the closed form and approximated by the
    numerical engines, each in its own way -- a lattice by aligning nodes to the
    barrier, a PDE by placing a grid boundary on it, simulation by weighting
    paths with the probability of an unobserved crossing.

    `rebate` is paid at expiry if a knock-out is triggered (or a knock-in never
    is). Zero by default.
    """

    strike: float = 100.0
    barrier: float = 120.0
    option_type: OptionType = OptionType.CALL
    barrier_type: BarrierType = BarrierType.UP_AND_OUT
    rebate: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.strike <= 0 or self.barrier <= 0:
            raise ValueError("strike and barrier must be positive")
        object.__setattr__(self, "option_type", OptionType(self.option_type))
        object.__setattr__(self, "barrier_type", BarrierType(self.barrier_type))

    @property
    def underlying_vanilla(self) -> VanillaOption:
        return VanillaOption(self.expiry, self.strike, self.option_type, Exercise.EUROPEAN)

    def payoff(self, spot):
        """Terminal payoff *conditional on surviving*. Engines apply survival."""
        return self.underlying_vanilla.payoff(spot)

    def is_touched(self, spot: np.ndarray | float) -> np.ndarray:
        """Whether a given spot level is at or beyond the barrier."""
        s = np.asarray(spot, dtype=float)
        return s >= self.barrier if self.barrier_type.is_up else s <= self.barrier

    def already_triggered(self, spot: float) -> bool:
        return bool(self.is_touched(spot))

    @property
    def is_path_dependent(self) -> bool:
        return True

    def with_barrier_type(self, kind: BarrierType) -> "BarrierOption":
        return BarrierOption(self.expiry, self.strike, self.barrier,
                             self.option_type, kind, self.rebate)
