#!/usr/bin/env python3
"""Kalshi order execution, with risk controls sized for a small account.

Separate from `kalshi.py` on purpose: that module is read-only and provably
cannot trade. Everything that can spend money lives here, behind an explicit
`--live` flag that defaults off.

The single most dangerous detail in this file is the side mapping. Kalshi's V2
API models each market as ONE book, the YES book:

    buy  YES n @ p   ->  side="bid",  price = p
    buy  NO  n @ q   ->  side="ask",  price = 1 - q      (selling YES is buying NO)

That inversion is verified empirically against the live book (`yes_ask` equals
`1 - best_no_bid`), and `verify_side_mapping()` can confirm it with a single
one-contract order before any real size is risked. Getting it backwards would
take the exact opposite of every intended position, so it is checked rather
than trusted.

Orders are fill-or-kill by default: the edge is a resting quote that is about
to move, so an order must take it now or not at all. A resting remainder turns
a short-dated arb into an unhedged directional bet.
"""

from __future__ import annotations

import asyncio
import math
import uuid
from dataclasses import dataclass

import aiohttp

from kalshi import KalshiAuthError, KalshiClient, trading_fee
from btc_polymarket_arb import RETRYABLE_STATUS, RetryableError, json_loads, log

ORDERS_PATH = "/portfolio/events/orders"

#: Kalshi trades whole contracts.
MIN_CONTRACTS = 1


@dataclass(slots=True)
class RiskLimits:
    """Limits expressed as fractions of the starting balance.

    Defaults are deliberately tight for a small account. On $75 they give a
    $6 maximum stake per trade and a $15 daily stop, which means roughly 12
    losing trades before trading halts - enough runway to learn something,
    little enough to survive being wrong.
    """

    max_stake_pct: float = 0.08  # per trade
    max_exposure_pct: float = 0.25  # total open at once
    daily_loss_pct: float = 0.20  # halt for the session
    max_consecutive_losses: int = 3
    max_trades: int = 40  # hard cap on a session
    min_contracts: int = MIN_CONTRACTS


@dataclass(slots=True)
class OrderResult:
    ok: bool
    dry_run: bool
    ticker: str
    outcome: str  # "YES" | "NO" - what we intended to own
    api_side: str  # "bid" | "ask" - what was actually sent
    price: float  # price of the outcome we wanted
    api_price: float  # price actually sent on the YES book
    count: int
    order_id: str | None = None
    status: str = ""
    error: str | None = None

    @property
    def stake(self) -> float:
        return self.count * self.price

    def summary(self) -> str:
        tag = "SIMULATED" if self.dry_run else "LIVE"
        if not self.ok:
            return f"{tag} {self.outcome} x{self.count} REJECTED: {self.error}"
        return (
            f"{tag} BUY {self.outcome} x{self.count} @ {self.price:.4f} "
            f"(${self.stake:.2f}, sent as {self.api_side} {self.api_price:.4f})"
            + (f" id={self.order_id}" if self.order_id else "")
        )


class KalshiTrader:
    """Places orders and enforces the account limits.

    Dry run is the default. `arm()` must succeed before anything is sent, and
    it fails closed if the balance cannot be read - without a balance there is
    no way to size a position or know a stop has been hit.
    """

    def __init__(
        self,
        client: KalshiClient,
        limits: RiskLimits | None = None,
        dry_run: bool = True,
    ) -> None:
        self._client = client
        self.limits = limits or RiskLimits()
        self.dry_run = dry_run
        self.starting_balance = 0.0
        self.realized = 0.0
        self.open_stake = 0.0
        self.trades = 0
        self.consecutive_losses = 0
        self.halted = False
        self.halt_reason = ""
        self.side_mapping_verified = False
        self._lock = asyncio.Lock()
        self._orders: list[OrderResult] = []

    # -- lifecycle ---------------------------------------------------------- #

    async def arm(self) -> bool:
        if self.dry_run:
            log.info("DRY RUN: no orders will be sent")
            self.starting_balance = 75.0
            return True

        try:
            self._client.authenticate()
        except KalshiAuthError as exc:
            log.error("Cannot arm: %s", exc)
            return False

        try:
            payload = await self._client.balance()
        except Exception as exc:  # noqa: BLE001
            log.error("Cannot read balance, refusing to trade live: %s", exc)
            return False

        cents = (payload or {}).get("balance")
        if not isinstance(cents, (int, float)) or cents <= 0:
            log.error("Balance unreadable or zero: %r", payload)
            return False

        self.starting_balance = float(cents) / 100.0
        log.warning(
            "LIVE TRADING ARMED | balance $%.2f | max stake $%.2f/trade | "
            "stop at -$%.2f | max %d trades",
            self.starting_balance,
            self.max_stake(),
            self.starting_balance * self.limits.daily_loss_pct,
            self.limits.max_trades,
        )
        return True

    # -- sizing and gates --------------------------------------------------- #

    def max_stake(self) -> float:
        return self.starting_balance * self.limits.max_stake_pct

    def equity(self) -> float:
        return self.starting_balance + self.realized

    def size_for(self, price: float) -> int:
        """Whole contracts affordable at `price`, inside every limit.

        Returns 0 when no legal size exists, which the caller must treat as
        "do not trade" rather than "trade smaller".
        """
        if not (0.0 < price < 1.0):
            return 0
        budget = min(
            self.max_stake(),
            max(0.0, self.starting_balance * self.limits.max_exposure_pct - self.open_stake),
            max(0.0, self.equity()),
        )
        if budget <= 0:
            return 0
        # Leave room for the fee, which peaks near 1.75c per contract.
        per_contract = price + 0.0175
        count = int(math.floor(budget / per_contract))
        return count if count >= self.limits.min_contracts else 0

    def check_halt(self) -> str | None:
        if self.halted:
            return self.halt_reason
        loss = -self.realized
        limit = self.starting_balance * self.limits.daily_loss_pct
        if loss >= limit > 0:
            self.halted, self.halt_reason = True, (
                f"session loss ${loss:.2f} hit the ${limit:.2f} stop"
            )
        elif self.consecutive_losses >= self.limits.max_consecutive_losses:
            self.halted, self.halt_reason = True, (
                f"{self.consecutive_losses} losing trades in a row"
            )
        elif self.trades >= self.limits.max_trades:
            self.halted, self.halt_reason = True, (
                f"hit the {self.limits.max_trades}-trade session cap"
            )
        if self.halted:
            log.error("TRADING HALTED: %s", self.halt_reason)
        return self.halt_reason if self.halted else None

    # -- the side mapping --------------------------------------------------- #

    @staticmethod
    def to_api_side(outcome: str, price: float) -> tuple[str, float]:
        """Map an intended outcome and its price onto the YES book.

        Buying NO at q is selling YES at 1-q. This is the inversion that must
        never be wrong, so it lives in one place and is unit tested both ways.
        """
        if outcome.upper() == "YES":
            return "bid", price
        return "ask", round(1.0 - price, 4)

    async def verify_side_mapping(self, ticker: str) -> bool:
        """Prove the mapping with one 1-contract order before risking size.

        Buys a single NO contract, then reads the position back. If the account
        ends up holding NO, the inversion is right. Costs at most a dollar and
        removes the one error that would reverse every trade.
        """
        if self.dry_run:
            self.side_mapping_verified = True
            return True

        log.warning("Verifying side mapping with a single 1-contract NO order on %s", ticker)
        result = await self.place(ticker, "NO", 0.99, 1, tif="immediate_or_cancel")
        if not result.ok:
            log.error("Verification order rejected: %s", result.error)
            return False

        await asyncio.sleep(2.0)
        try:
            positions = await self._client.positions()
        except Exception as exc:  # noqa: BLE001
            log.error("Could not read positions to verify: %s", exc)
            return False

        rows = (positions or {}).get("market_positions") or []
        row = next((r for r in rows if r.get("ticker") == ticker), None)
        if row is None:
            log.error("No position found after the verification order; not proceeding")
            return False

        # Kalshi reports a signed position: negative means short YES = long NO.
        qty = row.get("position") or row.get("market_exposure") or 0
        try:
            qty = float(qty)
        except (TypeError, ValueError):
            qty = 0.0
        ok = qty < 0
        log.warning(
            "Side mapping %s: a NO buy produced position %s (%s)",
            "VERIFIED" if ok else "WRONG - ABORTING",
            qty,
            "short YES = long NO, as intended" if ok else "expected a short YES position",
        )
        self.side_mapping_verified = ok
        return ok

    # -- order placement ---------------------------------------------------- #

    async def place(
        self,
        ticker: str,
        outcome: str,
        price: float,
        count: int,
        tif: str = "fill_or_kill",
    ) -> OrderResult:
        """Buy `count` contracts of `outcome`. Never raises."""
        api_side, api_price = self.to_api_side(outcome, price)
        result = OrderResult(
            ok=False,
            dry_run=self.dry_run,
            ticker=ticker,
            outcome=outcome.upper(),
            api_side=api_side,
            price=price,
            api_price=api_price,
            count=count,
        )

        if self.halted:
            result.error = f"halted: {self.halt_reason}"
            return result
        if count < self.limits.min_contracts:
            result.error = f"size {count} below the {self.limits.min_contracts}-contract minimum"
            return result
        stake = count * price
        if stake > self.max_stake() + 1e-9:
            result.error = f"stake ${stake:.2f} exceeds the ${self.max_stake():.2f} per-trade cap"
            log.error("Order blocked: %s", result.error)
            return result
        if not (0.0 < api_price < 1.0):
            result.error = f"api price {api_price} outside (0,1)"
            return result

        async with self._lock:
            if self.dry_run:
                result.ok = True
                result.status = "simulated"
                self._book_trade(result)
                return result

            body = {
                "ticker": ticker,
                "side": api_side,
                "count": f"{count}",
                "price": f"{api_price:.4f}",
                "time_in_force": tif,
                "self_trade_prevention_type": "taker_at_cross",
                "client_order_id": str(uuid.uuid4()),
            }
            try:
                response = await self._post(ORDERS_PATH, body)
            except Exception as exc:  # noqa: BLE001
                result.error = f"{type(exc).__name__}: {exc}"
                log.error("Order failed (%s %s x%d): %s", outcome, ticker, count, exc)
                self.consecutive_losses += 1
                self.check_halt()
                return result

            order = (response or {}).get("order") or response or {}
            result.order_id = str(order.get("order_id") or "") or None
            result.status = str(order.get("status") or "")
            result.ok = bool(result.order_id) or result.status.lower() in ("resting", "executed")
            if not result.ok:
                result.error = str(response)[:200]
            else:
                self._book_trade(result)
            return result

    def _book_trade(self, result: OrderResult) -> None:
        self.trades += 1
        self.open_stake += result.stake + trading_fee(result.price, result.count)
        self._orders.append(result)

    async def _post(self, path: str, body: dict) -> dict:
        """Authenticated POST. Deliberately NOT retried.

        A POST that times out may already have executed; resending risks a
        duplicate position, which is worse than a missed trade.
        """
        if self._client._signer is None:
            raise KalshiAuthError("not authenticated")
        full = "/trade-api/v2" + path
        headers = self._client._signer.headers("POST", full)
        url = self._client._host + full
        async with self._client._session.post(
            url, headers=headers, json=body, timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            raw = await resp.read()
            if resp.status in RETRYABLE_STATUS:
                raise RetryableError(f"order -> HTTP {resp.status}", status=resp.status)
            if resp.status in (401, 403):
                raise KalshiAuthError(f"order rejected: HTTP {resp.status} {raw[:200]!r}")
            if resp.status not in (200, 201):
                raise RuntimeError(f"order -> HTTP {resp.status}: {raw[:250].decode('utf-8','replace')}")
            return json_loads(raw) if raw else {}

    # -- settlement accounting ---------------------------------------------- #

    def settle(self, ticker: str, result: str) -> float:
        """Book the outcome of every order on `ticker`. Returns realized PnL."""
        total = 0.0
        for order in [o for o in self._orders if o.ticker == ticker and o.ok]:
            won = (order.outcome == "YES" and result == "yes") or (
                order.outcome == "NO" and result == "no"
            )
            payout = order.count * (1.0 if won else 0.0)
            fees = trading_fee(order.price, order.count)
            pnl = payout - order.stake - fees
            total += pnl
            self.realized += pnl
            self.open_stake = max(0.0, self.open_stake - order.stake - fees)
            self.consecutive_losses = 0 if pnl > 0 else self.consecutive_losses + 1
            log.warning(
                "SETTLED %s %s x%d @ %.4f -> %s : %+.2f  (session %+.2f)",
                ticker, order.outcome, order.count, order.price,
                result.upper(), pnl, self.realized,
            )
        if total:
            self.check_halt()
        return total

    def stats(self) -> str:
        return (
            f"bal ${self.equity():.2f} (start ${self.starting_balance:.2f}) "
            f"realized {self.realized:+.2f} open ${self.open_stake:.2f} "
            f"trades {self.trades}/{self.limits.max_trades}"
            + (f" HALTED: {self.halt_reason}" if self.halted else "")
        )
