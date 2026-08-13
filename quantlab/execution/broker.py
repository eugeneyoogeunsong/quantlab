"""Layer 5 - Execution.

A paper broker plus an order generator that turns target weights into concrete,
reviewable orders. The `Broker` protocol is what a live adapter (Alpaca, IBKR)
would implement -- deliberately small, because the smaller the live surface, the
less can go wrong at 09:30.

Nothing here places real orders. That is a decision for the account holder, not
for a backtesting library.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

Side = Literal["buy", "sell"]


@dataclass
class Order:
    symbol: str
    side: Side
    quantity: float
    order_type: str = "market"
    limit_price: float | None = None
    notional: float = 0.0
    reason: str = ""
    created: str = field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Fill:
    symbol: str
    side: Side
    quantity: float
    price: float
    commission: float = 0.0
    slippage: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds"))


class Broker(Protocol):
    def get_positions(self) -> dict[str, float]: ...
    def get_cash(self) -> float: ...
    def submit(self, order: Order) -> Fill | None: ...


@dataclass
class PaperBroker:
    """In-memory broker for dry runs and integration tests."""

    cash: float = 1_000_000.0
    positions: dict[str, float] = field(default_factory=dict)
    prices: dict[str, float] = field(default_factory=dict)
    commission_bps: float = 5.0
    slippage_bps: float = 5.0
    fills: list[Fill] = field(default_factory=list)

    def get_positions(self) -> dict[str, float]:
        return dict(self.positions)

    def get_cash(self) -> float:
        return self.cash

    def mark(self, prices: dict[str, float]) -> None:
        self.prices.update(prices)

    def equity(self) -> float:
        holdings = sum(q * self.prices.get(s, 0.0) for s, q in self.positions.items())
        return self.cash + holdings

    def submit(self, order: Order) -> Fill | None:
        px = self.prices.get(order.symbol)
        if px is None or px <= 0:
            log.warning("no price for %s; order rejected", order.symbol)
            return None

        # Slippage always works against you -- buys fill higher, sells lower.
        direction = 1 if order.side == "buy" else -1
        fill_px = px * (1 + direction * self.slippage_bps / 1e4)
        notional = abs(order.quantity) * fill_px
        commission = notional * self.commission_bps / 1e4

        if order.side == "buy" and notional + commission > self.cash:
            log.warning("insufficient cash for %s: need %.2f have %.2f",
                        order.symbol, notional + commission, self.cash)
            return None

        self.cash -= direction * notional + commission
        self.positions[order.symbol] = self.positions.get(order.symbol, 0.0) + direction * abs(order.quantity)
        if abs(self.positions[order.symbol]) < 1e-9:
            self.positions.pop(order.symbol, None)

        fill = Fill(order.symbol, order.side, abs(order.quantity), fill_px,
                    commission, abs(fill_px - px) * abs(order.quantity))
        self.fills.append(fill)
        return fill


def generate_orders(
    target_weights: pd.Series,
    current_positions: dict[str, float],
    prices: pd.Series,
    equity: float,
    min_trade_notional: float = 100.0,
    rebalance_threshold: float = 0.005,
) -> list[Order]:
    """Diff target weights against current holdings into orders.

    Two filters that matter in production:

    `rebalance_threshold` -- ignore drift below 0.5% of the book. Without it you
    will trade every day to correct rounding, paying full costs for weight
    changes that are noise.

    `min_trade_notional` -- skip trades too small to be worth the commission.

    Together these typically cut live turnover well below what the backtest
    assumed, which is the rare case of reality being kinder than the model.
    """
    orders: list[Order] = []
    symbols = set(target_weights.index) | set(current_positions)

    for sym in sorted(symbols):
        px = prices.get(sym, np.nan)
        if not np.isfinite(px) or px <= 0:
            continue

        target_w = float(target_weights.get(sym, 0.0))
        current_qty = float(current_positions.get(sym, 0.0))
        current_w = current_qty * px / equity if equity > 0 else 0.0

        drift = target_w - current_w
        if abs(drift) < rebalance_threshold:
            continue

        delta_notional = drift * equity
        if abs(delta_notional) < min_trade_notional:
            continue

        qty = delta_notional / px
        orders.append(Order(
            symbol=sym,
            side="buy" if qty > 0 else "sell",
            quantity=abs(qty),
            notional=abs(delta_notional),
            reason=f"target {target_w:.3f} vs current {current_w:.3f}",
        ))
    return orders


def write_order_blotter(orders: list[Order], path: str | Path) -> Path:
    """Persist orders as JSON. QA Section: live trading + logs.

    An order file that predates execution is the only way to tell later whether
    a bad day was a bad signal or a bad fill.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": datetime.utcnow().isoformat(timespec="seconds"),
        "n_orders": len(orders),
        "gross_notional": round(sum(o.notional for o in orders), 2),
        "orders": [o.to_dict() for o in orders],
    }
    path.write_text(json.dumps(payload, indent=2))
    log.info("wrote %d orders to %s", len(orders), path)
    return path
