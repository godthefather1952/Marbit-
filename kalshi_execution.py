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

from kalshi import KalshiAuthError, KalshiClient, fee_per_contract, trading_fee
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
    #: How many ~$1 side-mapping probes a session will buy before giving up on
    #: proving the mapping and refusing to trade. Each costs about a dollar,
    #: so an unbounded retry quietly bleeds the account: a live session bought
    #: three and would have kept going.
    max_verification_attempts: int = 3
    #: Sell once the position is worth this multiple of what it cost, net of
    #: the fees on both sides. 0 disables early exits.
    #:
    #: This is not a bolt-on: STALE's thesis is that the book has not yet
    #: repriced a spot move. When it does reprice, the edge is CAPTURED, and
    #: holding to settlement is a different bet that was never intended - a
    #: contract bought at 0.32 because the model said 0.45 has no thesis left
    #: at 0.64. Taking the gain there converts a binary lottery into a realized
    #: profit and removes the reversal risk that has produced every loss.
    #:
    #: 1.5x, not 2.0x, on evidence. Across every position observable in the
    #: logs so far the peak marks were 4.34x, 1.75x, 1.25x, 1.23x, 1.23x,
    #: 0.93x, 0.91x and 0.63x - so a 2.0x rule caught exactly one of eight and
    #: sat out a 1.75x mover that then settled worthless. 1.5x catches both
    #: without cutting the one winner short (it peaked at 1.25x and settled
    #: full). Small sample; revisit as the peak reports accumulate.
    take_profit_multiple: float = 1.5
    #: Sell if the position falls to this fraction of its cost. 0 disables.
    #: Deliberately off by default: on a cheap contract the mark is noisy and a
    #: stop mostly pays the spread to exit trades that would have recovered.
    stop_loss_fraction: float = 0.0
    #: Also exit when the book reprices to the fair value the signal was based
    #: on, even if that is short of the multiple. This is the most faithful
    #: exit of all: STALE bought because the book had not caught up to our fair
    #: value, so the moment it does, the thesis is complete by definition.
    exit_at_fair_value: bool = True
    #: Never try to exit inside the final seconds - the book thins to nothing
    #: there (the winning side stops being offered at all), so an exit would
    #: cross a huge spread to escape a position about to settle anyway.
    min_seconds_to_exit: float = 45.0


def _fixed_point(value) -> float | None:
    """Kalshi returns counts and prices as fixed-point STRINGS ("1", "0.9900").

    Returns None for absent/unparseable rather than 0.0, because "the venue did
    not tell us" and "the venue told us zero" must not be confused: the first
    means fall back to another signal, the second means the order did not fill.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    #: Contracts the venue did NOT fill, when it reports them.
    remaining: float | None = None
    #: Fee the venue actually charged per contract, when reported. Preferred
    #: over our own fee model at settlement, since it is the real number.
    avg_fee_paid: float | None = None
    #: True for the 1-contract side-mapping probe. Its cost is a known,
    #: bounded cost of doing business - not a strategy's opinion being wrong -
    #: so it must not feed the consecutive-loss breaker.
    verification: bool = False
    #: Set once the position has been exited early, so settlement does not
    #: also pay out a position we already sold.
    closed: bool = False
    #: True when this order was itself a closing trade.
    closing: bool = False
    #: Which strategy opened it, for the exit rules and the scorecard.
    strategy: str = ""
    #: Our fair value for THIS outcome at entry. The book reaching it means the
    #: mispricing we bought has closed.
    entry_fair: float = 0.0
    #: Best mark seen while the position was open. Reported at settlement so
    #: the take-profit threshold can be set from evidence rather than taste:
    #: "peaked at 4.3x then settled worthless" is the number that tells you
    #: whether the exit rule is set too high.
    peak_mark: float = 0.0

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
        self.verification_attempts = 0
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

    async def go_live(self) -> bool:
        """Promote a dry trader to live, after the warm-up gates have passed.

        The paper phase's numbers are simulations: carrying them into the live
        risk state would either eat the daily-loss budget with fake losses or
        pad it with fake wins. So everything resets - counters, halt state,
        simulated orders - and the live session starts clean from the real
        balance, re-proving the side mapping with a real order before size.

        On any arming failure the trader falls back to dry rather than being
        left half-armed with dry_run=False and no balance.
        """
        if not self.dry_run:
            return True
        self.dry_run = False
        self.realized = 0.0
        self.open_stake = 0.0
        self.trades = 0
        self.consecutive_losses = 0
        self.halted = False
        self.halt_reason = ""
        self.side_mapping_verified = False
        self._orders.clear()
        ok = await self.arm()
        if not ok:
            self.dry_run = True
            log.error("Promotion to live failed; staying on paper")
        return ok

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

    async def verify_side_mapping(self, ticker: str) -> bool | None:
        """Prove the mapping with one 1-contract order before risking size.

        Buys a single NO contract, then reads the position back. If the account
        ends up holding NO, the inversion is right. Costs at most a dollar and
        removes the one error that would reverse every trade.

        Three outcomes, and the difference matters:

            True   position is short YES = long NO -> mapping proven
            False  position is LONG YES after a NO buy -> mapping reversed,
                   the caller must halt
            None   inconclusive - the probe did not fill, or the position
                   could not be read. Not evidence of anything; retry on a
                   later signal instead of halting.

        A live session taught the inconclusive case the hard way: on a market
        already at 0.009/0.010 the NO ask was 0.991, the old 0.99 probe limit
        could not cross it, the IOC cancelled unfilled, and the empty position
        was reported as "mapping WRONG" - halting a healthy session. The probe
        now bids 0.999, the top of the venue's price ladder, so it crosses any
        book that exists.
        """
        if self.dry_run:
            self.side_mapping_verified = True
            return True

        if self.verification_attempts >= self.limits.max_verification_attempts:
            # Out of attempts is NOT the same as reversed. Saying "REVERSED"
            # here mislabelled a whole session's halt reason.
            if not self.halted:
                self.halted = True
                self.halt_reason = (
                    f"side mapping unproven after {self.verification_attempts} "
                    f"probes - refusing to trade rather than keep buying probes"
                )
                log.error("TRADING HALTED: %s", self.halt_reason)
            return False

        self.verification_attempts += 1
        log.warning(
            "Verifying side mapping with a single 1-contract NO order on %s "
            "(attempt %d/%d)",
            ticker, self.verification_attempts, self.limits.max_verification_attempts,
        )
        before = await self._balance_or_none()
        result = await self.place(
            ticker, "NO", 0.999, 1, tif="immediate_or_cancel", verification=True
        )
        if not result.ok:
            log.warning(
                "Verification probe did not fill (%s); inconclusive - will retry "
                "on a later signal", result.error,
            )
            return None

        # The FILL PRICE settles this on its own, with no second API call.
        #
        # We sent side="ask" at 0.001 - an order to sell YES at 0.1c or better.
        # A limit order can only ever fill on the favourable side of its own
        # limit, so:
        #
        #   sell semantics: fills at >= 0.001, i.e. wherever the YES BID is.
        #                   Filling at 0.945 means it crossed a 0.945 bid, and
        #                   we are now short YES = long NO at 0.055. Correct.
        #   buy semantics:  could only fill at <= 0.001, so a 0.945 fill is
        #                   arithmetically impossible.
        #
        # This matters because the position endpoint has now failed to confirm a
        # fill nine times across four sessions - reading empty 8s after an order
        # that demonstrably filled and later settled. Asking it at all was the
        # mistake: the venue already told us the price, and the price is proof.
        limit = 1.0 - 0.999  # the YES-book price we sent
        fill = result.api_price
        if fill > limit + 0.005:
            log.warning(
                "Side mapping VERIFIED from the fill: sold YES at %.4f against a "
                "%.4f limit, so we are short YES = long NO at %.4f. A buy could "
                "not have filled above its limit.",
                fill, limit, result.price,
            )
            self.side_mapping_verified = True
            return True
        # Filled at (not above) the limit, so the price says nothing: the YES
        # bid really was ~0.001. Ask the BALANCE instead, which distinguishes
        # both directions decisively and, unlike the position endpoint, has
        # answered correctly in every session. Buying NO at 0.055 debits 5.5c;
        # buying YES at 0.945 debits 94.5c. Those cannot be confused.
        log.info("Probe filled at its limit; checking the balance delta instead")
        after = await self._balance_or_none()
        if before is not None and after is not None:
            spent = before - after
            no_cost = result.price * result.count
            yes_cost = (1.0 - result.price) * result.count
            if abs(spent - no_cost) < abs(spent - yes_cost):
                log.warning(
                    "Side mapping VERIFIED from the balance: spent $%.4f, which "
                    "matches NO at %.4f (a YES fill would have cost $%.4f)",
                    spent, result.price, yes_cost,
                )
                self.side_mapping_verified = True
                return True
            log.error(
                "Side mapping REVERSED - ABORTING: spent $%.4f, which matches "
                "YES at %.4f rather than the NO at %.4f we intended. Every "
                "position would be the opposite of the one asked for.",
                spent, 1.0 - result.price, result.price,
            )
            self.side_mapping_verified = False
            return False

        qty = await self._probe_position(ticker)
        if qty is None:
            log.warning(
                "Could not confirm the probe's position or fill; inconclusive - "
                "will retry on a later signal"
            )
            return None
        if qty > 0:
            log.error(
                "Side mapping WRONG - ABORTING: a NO buy produced a LONG YES "
                "position (%s). Every order would be reversed.", qty,
            )
            self.side_mapping_verified = False
            return False
        log.warning(
            "Side mapping VERIFIED: a NO buy produced position %s "
            "(short YES = long NO, as intended)", qty,
        )
        self.side_mapping_verified = True
        return True

    async def _balance_or_none(self) -> float | None:
        """Account balance in dollars, or None if it cannot be read."""
        try:
            payload = await self._client.balance()
        except Exception as exc:  # noqa: BLE001
            log.warning("Balance read failed (%s)", exc)
            return None
        cents = (payload or {}).get("balance")
        return float(cents) / 100.0 if isinstance(cents, (int, float)) else None

    async def _probe_position(self, ticker: str) -> float | None:
        """Signed position on `ticker` after the probe, or None if unreadable.

        Two sources, because the first one failed live. A position snapshot is
        eventually consistent: a live session read it 2s after a fill and saw
        nothing three separate times, then watched the same contract settle
        from that very position 15 minutes later. So the read is retried with
        backoff and scoped to the one ticker, and if it still comes back empty
        the fills endpoint - the record of the trade itself, which carries the
        side the venue booked - is asked directly.
        """
        for delay in (1.5, 2.5, 4.0):
            await asyncio.sleep(delay)
            try:
                payload = await self._client.positions(ticker=ticker)
            except Exception as exc:  # noqa: BLE001
                log.warning("Position read failed (%s); retrying", exc)
                continue
            for row in (payload or {}).get("market_positions") or []:
                if row.get("ticker") != ticker:
                    continue
                raw = row.get("position")
                if raw in (None, ""):
                    raw = row.get("market_exposure")
                try:
                    qty = float(raw)
                except (TypeError, ValueError):
                    continue
                if qty != 0.0:
                    return qty

        # The position view never showed it. Ask what actually filled.
        try:
            payload = await self._client.fills(ticker=ticker, limit=10)
        except Exception as exc:  # noqa: BLE001
            log.warning("Fill read failed (%s)", exc)
            return None

        net = 0.0
        for fill in (payload or {}).get("fills") or []:
            if fill.get("ticker") != ticker:
                continue
            try:
                count = float(fill.get("count") or 0)
            except (TypeError, ValueError):
                continue
            # A fill's `side` is the YES-book side the venue booked. Our NO buy
            # was sent as an ask (sell YES), so it must come back short.
            side = str(fill.get("side") or "").lower()
            action = str(fill.get("action") or "buy").lower()
            signed = count if side in ("yes", "bid") else -count
            net += signed if action == "buy" else -signed
        if net != 0.0:
            log.warning("Confirmed from fills instead of positions: net %s", net)
            return net
        return None

    # -- order placement ---------------------------------------------------- #

    async def place(
        self,
        ticker: str,
        outcome: str,
        price: float,
        count: int,
        tif: str = "fill_or_kill",
        verification: bool = False,
        strategy: str = "",
        entry_fair: float = 0.0,
    ) -> OrderResult:
        """Buy `count` contracts of `outcome`. Never raises.

        `verification` marks the single 1-contract side-mapping probe. It is
        exempt from the per-trade stake cap - and only from that cap - because
        its worst case is bounded at a dollar and it IS the safety check: a
        graded live session on a $9.80 balance had an 8% cap of $0.78 reject
        the $0.99 probe, which halted trading before the first real order. The
        halt, minimum-size and price checks still apply.
        """
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
            verification=verification,
            strategy=strategy,
            entry_fair=entry_fair,
        )

        if self.halted:
            result.error = f"halted: {self.halt_reason}"
            return result
        if count < self.limits.min_contracts:
            result.error = f"size {count} below the {self.limits.min_contracts}-contract minimum"
            return result
        stake = count * price
        if not (verification and count == 1) and stake > self.max_stake() + 1e-9:
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

            # An accepted order is not a filled order, and CreateOrder V2 does
            # not return a status at all - it returns fill_count and
            # remaining_count. Treating "came back with an order_id" as a fill
            # booked phantom positions across two live sessions: orders that
            # never filled were counted as trades, "settled" for money that was
            # never at risk, and left the side-mapping probe hunting a position
            # that had never existed. The venue's own fill count is the only
            # honest answer.
            filled = _fixed_point(order.get("fill_count"))
            if filled is None:  # older payloads used the taker_* naming
                filled = _fixed_point(order.get("taker_fill_count"))
            result.remaining = _fixed_point(order.get("remaining_count"))
            avg_price = _fixed_point(order.get("average_fill_price"))
            avg_fee = _fixed_point(order.get("average_fee_paid"))

            if filled is not None:
                result.ok = filled > 0
                # Book what actually filled, at what it actually cost. An IOC
                # can fill part of the size, and settlement must not credit us
                # contracts we never owned.
                if result.ok:
                    result.count = int(filled)
                    if avg_price and 0.0 < avg_price < 1.0:
                        # average_fill_price is on the YES book; convert back to
                        # the outcome we asked for.
                        result.api_price = avg_price
                        result.price = (
                            avg_price if result.outcome == "YES"
                            else round(1.0 - avg_price, 4)
                        )
                    if avg_fee is not None:
                        result.avg_fee_paid = avg_fee
            else:
                # No fill information at all: fall back to the status wording
                # rather than assuming success.
                status_l = result.status.lower()
                result.ok = status_l in ("executed", "filled")

            if not result.ok:
                result.error = result.error or (
                    f"no fill (filled {filled if filled is not None else '?'} of "
                    f"{result.count}, status {result.status or 'none reported'})"
                )
                log.warning(
                    "Order not filled (%s %s x%d @ %.4f sent as %s %.4f): %s",
                    result.outcome, ticker, count, price,
                    result.api_side, result.api_price, result.error,
                )
            else:
                log.info(
                    "Filled %d/%d %s on %s at %.4f",
                    result.count, count, result.outcome, ticker, result.price,
                )
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

    # -- exits -------------------------------------------------------------- #

    def open_positions(self, ticker: str | None = None) -> list[OrderResult]:
        """Filled orders still exposed to settlement."""
        return [
            o for o in self._orders
            if o.ok and not o.closed and not o.closing
            and (ticker is None or o.ticker == ticker)
        ]

    @staticmethod
    def mark(order: OrderResult, book) -> float | None:
        """What the position could be sold for right now, per contract.

        A long YES is worth the YES bid; a long NO is worth the NO bid. Both
        are BIDS on purpose - the mark has to be what someone will actually pay
        us, not the mid or the ask, or every exit rule fires on a price we
        could never get.
        """
        value = book.yes_bid if order.outcome == "YES" else book.no_bid
        return value if value is not None and 0.0 < value < 1.0 else None

    def exit_reason(
        self, order: OrderResult, book, seconds_left: float
    ) -> tuple[str, float] | None:
        """Whether to close now, and at what mark. None means hold.

        Both the entry and the exit pay Kalshi's taker fee, so the test is on
        the NET multiple: a contract bought at 0.32 and sold at 0.64 grosses
        +0.32 but nets about +0.29 after both fees. Testing the raw price ratio
        would exit a "double" that is really 1.9x.
        """
        # The side-mapping probe IS managed, deliberately. Its PnL is kept out
        # of the consecutive-loss breaker because a 1-contract safety cost says
        # nothing about the model - but it is still a real contract bought with
        # real money, and excluding it from exits threw away the best trade in
        # the record: a probe bought at 0.12 reached a 0.57 bid (4.3x net) and
        # was then held to a worthless settlement.
        if seconds_left < self.limits.min_seconds_to_exit:
            return None
        value = self.mark(order, book)
        if value is None:
            return None
        order.peak_mark = max(order.peak_mark, value)

        cost = order.price + fee_per_contract(order.price)
        proceeds = value - fee_per_contract(value)
        if cost <= 0:
            return None
        ratio = proceeds / cost

        tp = self.limits.take_profit_multiple
        if tp > 0 and ratio >= tp:
            return ("take-profit", value)
        # The thesis target: the book has caught up to what we thought the
        # contract was worth when we bought it. Nothing is left to be right
        # about, so holding on is a fresh directional bet.
        if (
            self.limits.exit_at_fair_value
            and order.entry_fair > 0.0
            and proceeds >= order.entry_fair
            and ratio > 1.0
        ):
            return ("thesis-complete", value)
        sl = self.limits.stop_loss_fraction
        if sl > 0 and ratio <= sl:
            return ("stop-loss", value)
        return None

    async def close_position(
        self, order: OrderResult, price: float, reason: str = "exit"
    ) -> OrderResult:
        """Sell a position back to the book.

        Closing inverts the side: a long YES is closed by SELLING yes
        (side="ask"), and a long NO - which on this venue is a short YES - is
        closed by BUYING yes back (side="bid"). Getting this backwards would
        double the position instead of flattening it, so it is derived from the
        same to_api_side() the entry used rather than written out again.
        """
        # Selling outcome X is buying the opposite outcome, in side terms.
        opposite = "NO" if order.outcome == "YES" else "YES"
        api_side, api_price = self.to_api_side(opposite, round(1.0 - price, 4))
        result = OrderResult(
            ok=False, dry_run=self.dry_run, ticker=order.ticker,
            outcome=order.outcome, api_side=api_side, price=price,
            api_price=api_price, count=order.count, closing=True,
            strategy=order.strategy,
        )

        gross = order.count * (price - order.price)
        fees = trading_fee(price, order.count)
        async with self._lock:
            if self.dry_run:
                result.ok = True
                result.status = "simulated"
            else:
                body = {
                    "ticker": order.ticker,
                    "side": api_side,
                    "count": f"{order.count}",
                    "price": f"{api_price:.4f}",
                    "time_in_force": "immediate_or_cancel",
                    "self_trade_prevention_type": "taker_at_cross",
                    "client_order_id": str(uuid.uuid4()),
                }
                try:
                    response = await self._post(ORDERS_PATH, body)
                except Exception as exc:  # noqa: BLE001
                    result.error = f"{type(exc).__name__}: {exc}"
                    log.error("Exit order failed on %s: %s", order.ticker, exc)
                    return result
                payload = (response or {}).get("order") or response or {}
                filled = _fixed_point(payload.get("fill_count"))
                avg = _fixed_point(payload.get("average_fill_price"))
                result.order_id = str(payload.get("order_id") or "") or None
                result.ok = bool(filled and filled > 0)
                if result.ok:
                    result.count = int(filled)
                    if avg and 0.0 < avg < 1.0:
                        result.price = (
                            avg if order.outcome == "YES" else round(1.0 - avg, 4)
                        )
                    gross = result.count * (result.price - order.price)
                    fees = trading_fee(result.price, result.count)
                else:
                    result.error = f"exit did not fill (filled {filled})"
                    log.warning("Exit on %s did not fill; holding", order.ticker)
                    return result

        pnl = gross - fees
        order.closed = True
        self.realized += pnl
        self.open_stake = max(0.0, self.open_stake - order.stake)
        self.consecutive_losses = 0 if pnl > 0 else self.consecutive_losses + 1
        self._orders.append(result)
        log.warning(
            "%s %s %s x%d: %.4f -> %.4f = %+.2f  (%.2fx net, session %+.2f)",
            "EXITED" if not self.dry_run else "EXITED [sim]",
            reason, order.outcome, result.count, order.price, result.price,
            pnl, (result.price - fee_per_contract(result.price))
            / max(order.price + fee_per_contract(order.price), 1e-9),
            self.realized,
        )
        self.check_halt()
        return result

    def settle(self, ticker: str, result: str) -> float:
        """Book the outcome of every order on `ticker` still open at expiry."""
        total = 0.0
        for order in [
            o for o in self._orders
            if o.ticker == ticker and o.ok and not o.closed and not o.closing
        ]:
            won = (order.outcome == "YES" and result == "yes") or (
                order.outcome == "NO" and result == "no"
            )
            payout = order.count * (1.0 if won else 0.0)
            # Prefer the fee the venue actually charged over our model of it.
            fees = (
                order.avg_fee_paid * order.count
                if order.avg_fee_paid is not None
                else trading_fee(order.price, order.count)
            )
            pnl = payout - order.stake - fees
            total += pnl
            self.realized += pnl
            self.open_stake = max(0.0, self.open_stake - order.stake - fees)
            # The consecutive-loss breaker exists to stop a STRATEGY that has
            # started being wrong. The verification probe is not a strategy: it
            # is a fixed ~$1 safety cost paid to prove the venue's side
            # semantics, and a market moving against a 1-contract probe says
            # nothing about the model. Letting it count halted a live session
            # after three probes - see L_081626_085656.
            if order.verification:
                peak = (
                    f"  [peaked at {order.peak_mark:.3f}]"
                    if order.peak_mark > order.price else ""
                )
                log.warning(
                    "SETTLED [probe] %s %s x%d @ %.4f -> %s : %+.2f  "
                    "(safety cost, not counted against the loss breaker)%s",
                    ticker, order.outcome, order.count, order.price,
                    result.upper(), pnl, peak,
                )
                continue
            self.consecutive_losses = 0 if pnl > 0 else self.consecutive_losses + 1
            missed = ""
            if pnl < 0 and order.peak_mark > order.price:
                ratio = (
                    (order.peak_mark - fee_per_contract(order.peak_mark))
                    / (order.price + fee_per_contract(order.price))
                )
                missed = (
                    f"  [peaked at {order.peak_mark:.3f} = {ratio:.2f}x net - "
                    f"a --take-profit of {ratio:.2f} would have exited here]"
                )
            log.warning(
                "SETTLED %s %s x%d @ %.4f -> %s : %+.2f  (session %+.2f)%s",
                ticker, order.outcome, order.count, order.price,
                result.upper(), pnl, self.realized, missed,
            )
        if total:
            self.check_halt()
        return total

    def stats(self) -> str:
        """One line, leading with whether any of this was real.

        The distinction is not cosmetic: a session reported five winning trades
        worth +$3.85 while the account moved ten cents, because every one was
        simulated or skipped.
        """
        real = [o for o in self._orders if o.ok and not o.dry_run]
        probes = sum(1 for o in real if o.verification)
        tag = "SIMULATED" if self.dry_run else "LIVE"
        return (
            f"[{tag}] bal ${self.equity():.2f} (start ${self.starting_balance:.2f}) "
            f"realized {self.realized:+.2f} open ${self.open_stake:.2f} "
            f"trades {self.trades}/{self.limits.max_trades} | "
            f"REAL fills {len(real)}"
            + (f" ({probes} side-mapping probe)" if probes else "")
            + (f" HALTED: {self.halt_reason}" if self.halted else "")
        )
