#!/usr/bin/env python3
"""End-to-end exercise of the signal and execution paths without live venues.

Spins up a local websocket server that speaks the Binance `btcusdt@trade`
payload format and replays a scripted spike, pairs it with a stubbed CLOB book
whose asks are held deliberately stale, and asserts each gate independently.

Covers signal generation, dry-run execution, the position gate, the notional
cap, the circuit breaker, and the live-mode credential refusal.

Run:  python test_signal_path.py
"""

from __future__ import annotations

import asyncio
import functools
import json as _stdlib_json
import logging
import math
import time

from websockets.asyncio.server import serve

from btc_polymarket_arb import (
    JSON_BACKEND,
    LatencyProfiler,
    BinanceTradeStream,
    BookFeed,
    Config,
    Quote,
    Credentials,
    ExecutionSettings,
    Executor,
    PriceBuffer,
    RetryableError,
    RiskManager,
    RiskSettings,
    SignalEngine,
    TrackedMarket,
    _scale_usdc,
    _tick_str,
    _usdc,
    json_loads,
    is_retryable,
    quantize_price,
    retry_async,
)
from btc_polymarket_arb import POLYMARKET_SDK

# The Polymarket CLOB SDK is optional (see requirements-polymarket.txt). Without
# it the venue-specific checks are skipped and everything else - model, risk,
# resilience, Kalshi - still runs, so a light install can still verify itself.
if POLYMARKET_SDK:
    from py_clob_client.exceptions import PolyApiException
    from py_clob_client.order_builder.constants import BUY, SELL
else:  # pragma: no cover
    PolyApiException = None
    BUY, SELL = "BUY", "SELL"

BASE = 100_000.0
FLAT_TICKS = 40  # ~2s of flat tape, so a pre-spike book anchor exists
TICK_INTERVAL = 0.05

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    (PASSED if ok else FAILED).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


class StubBookFeed(BookFeed):
    """CLOB stand-in whose asks never reprice - the staleness we hunt for."""

    def __init__(self, up_ask: float, down_ask: float) -> None:  # noqa: super-init-not-called
        self.up_ask = up_ask
        self.down_ask = down_ask
        self.calls = 0

    async def fetch(self, token_ids):  # type: ignore[override]
        self.calls += 1
        return {
            "UP_TOKEN": Quote(
                bid=round(self.up_ask - 0.01, 4), ask=self.up_ask, tick_size=0.01
            ),
            "DOWN_TOKEN": Quote(
                bid=round(self.down_ask - 0.01, 4), ask=self.down_ask, tick_size=0.01
            ),
        }


async def mock_binance(ws, spike_bps: float) -> None:
    """Flat tape, then a sharp dislocation, then hold at the new level."""
    for _ in range(FLAT_TICKS):
        await ws.send(_trade(BASE))
        await asyncio.sleep(TICK_INTERVAL)
    spiked = BASE * math.exp(spike_bps / 10_000.0)
    while True:
        await ws.send(_trade(spiked))
        await asyncio.sleep(TICK_INTERVAL)


def _trade(price: float) -> str:
    return _stdlib_json.dumps(
        {
            "e": "trade",
            "s": "BTCUSDT",
            "p": f"{price:.2f}",
            "q": "0.01",
            "T": int(time.time() * 1000),
        }
    )


def make_market(slug: str = "btc-updown-5m-TEST") -> TrackedMarket:
    now = time.time()
    return TrackedMarket(
        slug=slug,
        title="Bitcoin Up or Down - TEST",
        horizon="5m",
        condition_id="0xtest",
        up_token="UP_TOKEN",
        down_token="DOWN_TOKEN",
        window_open=now - 60.0,
        window_close=now + 240.0,
        twap_lookback=60.0,
        tick_size=0.01,
        accepting_orders=True,
    )


class LogCapture:
    """Capture engine logs, bypassing the root logger's level."""

    def __init__(self) -> None:
        self.records: list[logging.LogRecord] = []
        self._logger = logging.getLogger("btc-arb")
        outer = self

        class Handler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                outer.records.append(record)

        self._handler = Handler()

    def __enter__(self) -> "LogCapture":
        self._prior = (self._logger.level, self._logger.propagate)
        # The engine logs signals at WARNING/ERROR. The logger must be permissive
        # or isEnabledFor() drops them before any handler is consulted.
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False
        self._logger.addHandler(self._handler)
        return self

    def __exit__(self, *exc) -> None:
        self._logger.removeHandler(self._handler)
        self._logger.setLevel(self._prior[0])
        self._logger.propagate = self._prior[1]

    @property
    def signals(self) -> list[str]:
        return [r.getMessage() for r in self.records if r.levelno >= logging.WARNING]

    def count(self, needle: str) -> int:
        return sum(1 for m in self.signals if needle in m)

    def count_all(self, needle: str) -> int:
        """Match across every level, not just the WARNING+ signal banners."""
        return sum(1 for r in self.records if needle in r.getMessage())


def fixed_size_risk(**kw) -> RiskSettings:
    """Deterministic sizing so signal-path assertions stay focused.

    Dynamic sizing is exercised separately in test_dynamic_sizing.
    """
    kw.setdefault("dynamic_sizing", False)
    kw.setdefault("paper_bankroll", 1_000.0)
    return RiskSettings(**kw)


async def drive(
    *,
    up_ask: float,
    down_ask: float,
    spike_bps: float = 25.0,
    enforce_ceiling: bool = True,
    cooldown: float = 5.0,
    settings: ExecutionSettings | None = None,
    risk_settings: RiskSettings | None = None,
    executor: Executor | None = None,
    connect: bool = True,
    run_for: float | None = None,
    on_tick=None,
) -> tuple[LogCapture, Executor, int]:
    """Run the full pipeline against a mock tape. Returns (logs, executor, ticks).

    Stops at the first signal unless `run_for` is given, in which case it runs
    for that many seconds so repeat-suppression can be counted.
    """
    exec_settings = settings or ExecutionSettings(dry_run=True, order_size=10.0)
    if executor is None:
        executor = Executor(
            exec_settings, Credentials(), RiskManager(risk_settings or fixed_size_risk())
        )
    if connect:
        await executor.connect()  # seeds the paper bankroll
    ticks = 0

    with LogCapture() as logs:
        handler_fn = functools.partial(mock_binance, spike_bps=spike_bps)
        async with serve(handler_fn, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]

            buffer = PriceBuffer()
            books = StubBookFeed(up_ask, down_ask)
            cfg = Config(
                spike_bps=12.0,
                min_edge=0.03,
                max_yes_ask=0.52,
                enforce_ask_ceiling=enforce_ceiling,
                signal_cooldown=cooldown,
            )
            engine = SignalEngine(buffer, books, cfg, executor)
            engine.sync_markets([make_market()])

            stream = BinanceTradeStream(buffer, [f"ws://127.0.0.1:{port}"])
            tape = asyncio.create_task(stream.run())

            deadline = time.monotonic() + (run_for if run_for else 10.0)
            while time.monotonic() < deadline:
                await engine.refresh_books()
                await engine.evaluate()
                ticks += 1
                if on_tick is not None:
                    on_tick(ticks, executor)
                if run_for is None and logs.signals:
                    break
                await asyncio.sleep(0.1)

            tape.cancel()
            await asyncio.gather(tape, return_exceptions=True)

    return logs, executor, ticks


# --------------------------------------------------------------------------- #
# Signal generation
# --------------------------------------------------------------------------- #


async def test_signals() -> None:
    print("\n--- signal generation ---")

    logs, ex, _ = await drive(up_ask=0.45, down_ask=0.56)
    check(
        "UP spike, stale ask under the 0.52 gate -> SIGNAL",
        logs.count("ARBITRAGE WINDOW OPEN") == 1,
    )
    if logs.signals:
        print(logs.signals[0])

    logs, _, _ = await drive(up_ask=0.56, down_ask=0.45, spike_bps=-25.0)
    check(
        "DOWN spike, stale NO ask -> SIGNAL on the NO leg",
        logs.count("ARBITRAGE WINDOW OPEN") == 1 and "BUY NO" in "".join(logs.signals),
    )

    logs, _, _ = await drive(up_ask=0.97, down_ask=0.04, enforce_ceiling=False)
    check("book already repriced -> edge gate suppresses", not logs.signals)

    logs, _, _ = await drive(up_ask=0.60, down_ask=0.41)
    check("ask above 0.52 ceiling -> ceiling gate suppresses", not logs.signals)

    logs, _, _ = await drive(up_ask=0.45, down_ask=0.50)
    check(
        "asks sum to 0.95 -> RISK-FREE CROSS-BOOK ARB",
        logs.count("RISK-FREE CROSS-BOOK ARB") == 1,
    )

    logs, _, _ = await drive(up_ask=0.45, down_ask=0.56, spike_bps=0.0)
    check("flat tape, no spike -> silent", not logs.signals)


# --------------------------------------------------------------------------- #
# Dry-run execution
# --------------------------------------------------------------------------- #


async def test_dry_run_execution() -> None:
    print("\n--- dry-run execution ---")

    logs, ex, _ = await drive(up_ask=0.45, down_ask=0.56)
    check(
        "model signal submits one simulated order",
        ex.orders_sent == 1 and ex.orders_filled == 1,
        f"sent={ex.orders_sent} filled={ex.orders_filled}",
    )
    check(
        "dry run never builds a signing client",
        ex._client is None,
    )
    check(
        "expected PnL is booked and positive",
        ex.expected_pnl > 0,
        f"exp_pnl={ex.expected_pnl:+.2f}",
    )
    check(
        "execution is logged as SIMULATED",
        any("SIMULATED" in m for m in logs.signals),
    )
    if logs.signals:
        print("       " + next(m for m in logs.signals if "execution" in m).strip())

    # Cross-book fires BOTH legs, concurrently.
    logs, ex, _ = await drive(up_ask=0.45, down_ask=0.50)
    check(
        "cross-book signal submits two legs",
        ex.orders_sent == 2 and ex.orders_filled == 2,
        f"sent={ex.orders_sent}",
    )
    locked = 10.0 * (1.0 - 0.95)
    check(
        "cross-book books the locked spread as expected PnL",
        abs(ex.expected_pnl - locked) < 1e-6,
        f"exp_pnl={ex.expected_pnl:.4f} want {locked:.4f}",
    )
    check(
        "cross-book logs a PAIRED confirmation",
        any("PAIRED" in m for m in logs.signals),
    )


# --------------------------------------------------------------------------- #
# Position gate
# --------------------------------------------------------------------------- #


async def test_position_gate() -> None:
    print("\n--- position gate ---")

    # Cooldown disabled so the ONLY thing that can suppress a repeat order is
    # the position gate. The spike is sustained, so the signal re-fires on
    # every one of the many evaluation ticks.
    logs, ex, ticks = await drive(
        up_ask=0.45, down_ask=0.56, cooldown=0.0, run_for=3.0
    )
    banners = logs.count("ARBITRAGE WINDOW OPEN")
    check(
        "sustained spike re-signals on many ticks",
        banners > 3,
        f"{banners} banners over {ticks} ticks",
    )
    check(
        "position gate allows exactly ONE order despite repeat signals",
        ex.orders_sent == 1,
        f"orders_sent={ex.orders_sent} vs {banners} signals",
    )
    suppressed = logs.count_all("Position gate")
    check(
        "every duplicate attempt is logged as gate-suppressed",
        suppressed == banners - 1,
        f"{suppressed} suppression logs for {banners - 1} duplicate signals",
    )
    check(
        "market slug is held in active_positions",
        ex.active_positions == {"btc-updown-5m-TEST"},
        f"{ex.active_positions}",
    )

    # Releasing the slot mid-run must permit exactly one more entry. Trigger on
    # observed state, not a tick index: the spike does not begin until the flat
    # priming period is over, so a fixed tick can fire before anything is held.
    released = {"done": False}

    def release_once(tick: int, executor: Executor) -> None:
        if executor.orders_sent == 1 and not released["done"]:
            executor.release_market("btc-updown-5m-TEST")
            released["done"] = True

    logs, ex, _ = await drive(
        up_ask=0.45,
        down_ask=0.56,
        cooldown=0.0,
        run_for=4.0,
        on_tick=release_once,
    )
    check(
        "releasing the gate permits exactly one further entry",
        ex.orders_sent == 2,
        f"orders_sent={ex.orders_sent}",
    )

    # The gate is per-market, and must be cleared when the window retires.
    ex = Executor(ExecutionSettings(dry_run=True), Credentials(), RiskManager(fixed_size_risk()))
    got_first = await ex.acquire_market("mkt-a")
    got_again = await ex.acquire_market("mkt-a")
    got_other = await ex.acquire_market("mkt-b")
    check(
        "acquire is exclusive per market, independent across markets",
        got_first and not got_again and got_other,
    )
    ex.release_market("mkt-a")
    check("release frees the slot", await ex.acquire_market("mkt-a"))

    engine = SignalEngine(PriceBuffer(), StubBookFeed(0.5, 0.5), Config(), ex)
    stale = make_market("mkt-a")
    stale.window_close = time.time() - 1.0  # already expired
    engine.sync_markets([])
    engine._markets["mkt-a"] = stale
    engine.sync_markets([])
    check(
        "retiring a market releases its gate slot",
        "mkt-a" not in ex.active_positions,
        f"{ex.active_positions}",
    )


# --------------------------------------------------------------------------- #
# Safety gates
# --------------------------------------------------------------------------- #


async def test_safety_gates() -> None:
    print("\n--- safety gates ---")

    # Notional cap.
    ex = Executor(
        ExecutionSettings(dry_run=True, order_size=1000.0, max_notional=50.0),
        Credentials(),
        RiskManager(fixed_size_risk()),
    )
    await ex.connect()
    res = await ex.execute_arb_order("TOK", BUY, 0.45, 1000.0)
    check(
        "notional cap blocks an oversized order",
        not res.ok and "max-notional" in (res.error or ""),
        res.error or "",
    )
    check("blocked order is not counted as sent", ex.orders_sent == 0)

    # Live mode must refuse to arm without credentials.
    ex = Executor(ExecutionSettings(dry_run=False), Credentials())
    check("live mode refuses to arm with no private key", not await ex.connect())

    ex = Executor(
        ExecutionSettings(dry_run=False),
        Credentials(private_key="0x" + "11" * 32, signature_type=1),
    )
    check(
        "live mode refuses a proxy sig_type with no funder address",
        not await ex.connect(),
    )

    # Dry run arms regardless of missing credentials.
    ex = Executor(ExecutionSettings(dry_run=True), Credentials(), RiskManager(fixed_size_risk()))
    check("dry run arms without any credentials", await ex.connect())

    # Circuit breaker.
    class ExplodingExecutor(Executor):
        def _submit(self, *a, **kw):
            raise RuntimeError("venue on fire")

    ex = ExplodingExecutor(
        ExecutionSettings(dry_run=False, max_consecutive_errors=3),
        Credentials(),
        RiskManager(fixed_size_risk(max_consecutive_losses=99)),
    )
    ex.risk.bankroll = 1_000.0  # bypass connect(); we only exercise the error path
    ex._client = object()
    for _ in range(3):
        await ex.execute_arb_order("TOK", BUY, 0.45, 10.0)
    check("circuit breaker trips after repeated failures", ex.halted)
    res = await ex.execute_arb_order("TOK", BUY, 0.45, 10.0)
    check(
        "halted executor rejects further orders",
        not res.ok and "halted" in (res.error or ""),
        res.error or "",
    )

    # Tick quantization: buys round up, sells round down, always on-grid.
    cases = [
        (0.45, 0.01, BUY, 0.45),
        (0.451, 0.01, BUY, 0.46),
        (0.4501, 0.01, SELL, 0.45),
        (1.5, 0.01, BUY, 0.99),
        (0.0, 0.01, BUY, 0.01),
        (0.4567, 0.001, BUY, 0.457),
    ]
    ok = all(abs(quantize_price(p, t, s) - want) < 1e-9 for p, t, s, want in cases)
    check("prices snap onto the venue tick grid", ok)


# --------------------------------------------------------------------------- #
# Balance, allowance and dynamic sizing
# --------------------------------------------------------------------------- #


async def test_balance_and_sizing() -> None:
    print("\n--- balance & dynamic sizing ---")

    # The CLOB has returned raw 6-decimal integers, plain decimals, and a
    # nested allowances map. All three must parse, or sizing silently zeroes.
    shapes = [
        ({"balance": "1234560000"}, 1234.56),
        ({"balance": "1234.56"}, 1234.56),
        ({"balance": 1234560000}, 1234.56),
        ({"allowances": {"exchange": "50000000"}}, 50.0),
        ({"nothing": "1"}, None),
    ]
    ok = all(
        (_usdc(p, "balance", "allowance", "allowances") is None and want is None)
        or abs((_usdc(p, "balance", "allowance", "allowances") or -1) - (want or -1)) < 1e-6
        for p, want in shapes
    )
    check("USDC parses across every response shape the CLOB returns", ok)
    check("a malformed amount yields None, not a silent zero", _scale_usdc("abc") is None)

    risk = RiskManager(RiskSettings(max_risk_pct=0.02))
    risk.bankroll = 1_000.0
    risk.peak_equity = 1_000.0
    size = risk.size_for(0.45, max_notional=50.0, min_size=5.0)
    check(
        "2% of a $1000 bankroll sizes to $20 of premium",
        abs(size * 0.45 - 20.0) < 0.01,
        f"size={size} notional=${size * 0.45:.2f}",
    )

    capped = risk.size_for(0.45, max_notional=10.0, min_size=5.0)
    check(
        "max-notional overrides the risk percentage when tighter",
        abs(capped * 0.45 - 10.0) < 0.01,
        f"notional=${capped * 0.45:.2f}",
    )

    check(
        "a size below the venue minimum returns 0 (do not trade)",
        risk.size_for(0.45, max_notional=1.0, min_size=5.0) == 0.0,
    )

    # Realized losses must shrink both buying power and the risk budget.
    risk.realized_pnl = -300.0
    check(
        "realized losses reduce buying power",
        abs(risk.available() - 700.0) < 1e-6,
        f"available=${risk.available():.2f}",
    )
    shrunk = risk.size_for(0.45, max_notional=50.0, min_size=5.0)
    check(
        "risk budget scales down with equity",
        abs(shrunk * 0.45 - 14.0) < 0.01,
        f"notional=${shrunk * 0.45:.2f} (2% of $700)",
    )

    risk.allowance = 100.0
    check(
        "allowance caps available funds below cash",
        abs(risk.available() - 100.0) < 1e-6,
        f"available=${risk.available():.2f}",
    )

    # Pre-flight: an order larger than free funds is refused before signing.
    ex = Executor(
        ExecutionSettings(dry_run=True, max_notional=500.0),
        Credentials(),
        RiskManager(RiskSettings()),
    )
    ex.risk.bankroll = 10.0
    res = await ex.execute_arb_order("TOK", BUY, 0.50, 100.0)  # $50 of a $10 wallet
    check(
        "pre-flight balance check blocks an unfundable order",
        not res.ok and "insufficient unlocked funds" in (res.error or ""),
        res.error or "",
    )
    check("blocked order never reaches the venue", ex.orders_sent == 0)
    check("failed reservation leaves no committed capital", ex.risk.committed == 0.0)

    # Concurrent legs must not both claim the same funds.
    ex2 = Executor(ExecutionSettings(dry_run=True), Credentials(), RiskManager(RiskSettings()))
    ex2.risk.bankroll = 30.0
    r1, r2 = await asyncio.gather(
        ex2.execute_arb_order("A", BUY, 0.50, 40.0),  # $20
        ex2.execute_arb_order("B", BUY, 0.50, 40.0),  # $20 - only one fits
    )
    check(
        "capital reservation stops two concurrent legs over-committing",
        [r1.ok, r2.ok].count(True) == 1,
        f"ok={[r1.ok, r2.ok]}",
    )


# --------------------------------------------------------------------------- #
# Session risk breakers
# --------------------------------------------------------------------------- #


async def test_risk_breakers() -> None:
    print("\n--- session risk breakers ---")

    # Drawdown: tighter of (pct of peak, absolute USD) binds.
    risk = RiskManager(
        RiskSettings(
            max_drawdown_pct=0.05,
            max_drawdown_usd=100.0,
            risk_cooldown=0.4,
            max_drawdown_trips=2,
            max_consecutive_losses=99,
        )
    )
    risk.bankroll = 1_000.0
    risk.peak_equity = 1_000.0

    risk.realized_pnl = -40.0
    check("drawdown inside the limit does not trip", risk.check_breakers() is None)
    check("trading still allowed at -$40", risk.trading_allowed()[0])

    risk.realized_pnl = -55.0  # 5% of $1000 = $50 limit
    tripped = risk.check_breakers()
    check("drawdown past the limit trips the breaker", tripped is not None)
    check("tripped breaker blocks trading", not risk.trading_allowed()[0])
    check("trip enters a timed cooldown, not a permanent halt", not risk.halted_permanently)

    await asyncio.sleep(0.5)
    check("trading resumes after the cooldown expires", risk.trading_allowed()[0])
    check(
        "peak is re-baselined so the same loss does not instantly re-trip",
        abs(risk.peak_equity - 945.0) < 1e-6,
        f"peak=${risk.peak_equity:.2f}",
    )

    risk.realized_pnl = -110.0
    risk.check_breakers()
    check(
        "exhausting the trip budget halts the session permanently",
        risk.halted_permanently and risk.drawdown_trips == 2,
        f"trips={risk.drawdown_trips}",
    )

    # Consecutive losses, from failed executions.
    risk2 = RiskManager(
        RiskSettings(max_consecutive_losses=3, max_drawdown_usd=1e9, max_drawdown_pct=1.0)
    )
    risk2.bankroll = 1_000.0
    risk2.peak_equity = 1_000.0
    for _ in range(2):
        risk2.note_execution_failure()
        risk2.check_breakers()
    check("two failures do not yet halt", not risk2.halted_permanently)
    risk2.note_execution_failure()
    risk2.check_breakers()
    check(
        "three consecutive failures halt the session",
        risk2.halted_permanently and not risk2.trading_allowed()[0],
        risk2.halt_reason,
    )

    # Losing closes also count, and a win resets the streak.
    risk3 = RiskManager(
        RiskSettings(max_consecutive_losses=3, max_drawdown_usd=1e9, max_drawdown_pct=1.0)
    )
    risk3.bankroll = 1_000.0
    risk3.peak_equity = 1_000.0
    risk3.open_position("m1", "t1", "YES", 10, 0.60)
    risk3.mark_to_market("t1", 0.10)
    risk3.close_market("m1")
    check(
        "a losing close realizes PnL and increments the streak",
        risk3.consecutive_losses == 1 and abs(risk3.realized_pnl + 5.0) < 1e-6,
        f"streak={risk3.consecutive_losses} realized={risk3.realized_pnl:+.2f}",
    )
    risk3.open_position("m2", "t2", "YES", 10, 0.40)
    risk3.mark_to_market("t2", 0.90)
    risk3.close_market("m2")
    check(
        "a winning close resets the consecutive-loss streak",
        risk3.consecutive_losses == 0,
        f"streak={risk3.consecutive_losses} realized={risk3.realized_pnl:+.2f}",
    )

    # End to end: a halted breaker suppresses signals, not just orders.
    halted = Executor(
        ExecutionSettings(dry_run=True, order_size=10.0),
        Credentials(),
        RiskManager(fixed_size_risk(max_consecutive_losses=1)),
    )
    halted.risk.bankroll = 1_000.0
    halted.risk.peak_equity = 1_000.0
    halted.risk.note_execution_failure()
    halted.risk.check_breakers()
    logs, ex, _ = await drive(
        up_ask=0.45, down_ask=0.56, executor=halted, connect=False, run_for=3.0
    )
    check(
        "a halted session emits no signal banners at all",
        logs.count("ARBITRAGE WINDOW OPEN") == 0,
        f"{logs.count('ARBITRAGE WINDOW OPEN')} banners",
    )
    check("a halted session submits no orders", ex.orders_sent == 0)
    check(
        "suppression is announced once, not per tick",
        logs.count_all("Signals suppressed") == 1,
        f"{logs.count_all('Signals suppressed')} notices",
    )


# --------------------------------------------------------------------------- #
# Retry / rate-limit handling
# --------------------------------------------------------------------------- #


async def test_retry_backoff() -> None:
    print("\n--- retry & rate-limit handling ---")

    check("429 is classified retryable", is_retryable(RetryableError("rate", status=429)))
    check("503 is classified retryable", is_retryable(RetryableError("busy", status=503)))
    check("400 is NOT retryable", not is_retryable(RetryableError("bad", status=400)))
    check("timeouts are retryable", is_retryable(asyncio.TimeoutError()))

    if POLYMARKET_SDK:
        poly = PolyApiException(error_msg="rate limited")
        poly.status_code = 429
        check("PolyApiException 429 is classified retryable", is_retryable(poly))
        poly_bad = PolyApiException(error_msg="bad request")
        poly_bad.status_code = 400
        check("PolyApiException 400 is not retried", not is_retryable(poly_bad))

    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RetryableError("429 Too Many Requests", status=429)
        return "recovered"

    result = await retry_async(flaky, attempts=4, base_delay=0.01, label="test")
    check(
        "backoff retries through 429s and returns the eventual success",
        result == "recovered" and calls["n"] == 3,
        f"attempts={calls['n']}",
    )

    hard = {"n": 0}

    async def always_400():
        hard["n"] += 1
        raise RetryableError("400 Bad Request", status=400)

    try:
        await retry_async(always_400, attempts=4, base_delay=0.01, label="test")
        raised = False
    except RetryableError:
        raised = True
    check(
        "a non-retryable status fails immediately without burning attempts",
        raised and hard["n"] == 1,
        f"attempts={hard['n']}",
    )

    exhaust = {"n": 0}

    async def always_429():
        exhaust["n"] += 1
        raise RetryableError("429", status=429)

    try:
        await retry_async(always_429, attempts=3, base_delay=0.01, label="test")
        raised = False
    except RetryableError:
        raised = True
    check(
        "retries are bounded and the final failure propagates",
        raised and exhaust["n"] == 3,
        f"attempts={exhaust['n']}",
    )

    # A server-sent Retry-After must win over the computed backoff.
    slow = {"n": 0}

    async def retry_after():
        slow["n"] += 1
        if slow["n"] == 1:
            raise RetryableError("429", status=429, retry_after=0.35)
        return "ok"

    started = time.monotonic()
    await retry_async(retry_after, attempts=3, base_delay=0.001, label="test")
    waited = time.monotonic() - started
    check(
        "Retry-After overrides the computed backoff delay",
        waited >= 0.3,
        f"waited {waited:.2f}s for a 0.35s hint",
    )


# --------------------------------------------------------------------------- #
# Reconnect resilience
# --------------------------------------------------------------------------- #


async def test_reconnect_resilience() -> None:
    print("\n--- reconnect resilience ---")

    buf = PriceBuffer()
    for _ in range(30):
        buf.add(BASE, 0)
    check("buffer is ready with clean history", buf.ready)

    buf.reset("test disconnect", warmup=0.4)
    check("reset drops the stale tick window", buf.last() is None)
    check("reset enters warm-up", buf.warming_up)
    buf.add(BASE * 1.01, 0)
    buf.add(BASE * 1.01, 0)
    check(
        "no signal is possible during warm-up even with ticks present",
        not buf.ready,
        "prevents a phantom spike across the gap",
    )
    check(
        "largest_move cannot see across the gap",
        (buf.largest_move(3.0) is None) or abs(buf.largest_move(3.0).bps) < 1e-6,
    )
    await asyncio.sleep(0.5)
    check("buffer becomes ready again after warm-up", buf.ready)

    # The stream must reset the buffer when the tape drops.
    async def slam_shut(ws):
        await ws.close()

    buf2 = PriceBuffer()
    for _ in range(5):
        buf2.add(BASE, 0)
    ex = Executor(ExecutionSettings(dry_run=True), Credentials(), RiskManager(fixed_size_risk()))
    await ex.acquire_market("btc-updown-5m-LOCKED")
    ex.risk.open_position("btc-updown-5m-LOCKED", "TOK", "YES", 10, 0.45)

    async with serve(slam_shut, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        stream = BinanceTradeStream(buf2, [f"ws://127.0.0.1:{port}"])
        tape = asyncio.create_task(stream.run())
        await asyncio.sleep(1.2)  # allow at least one connect/drop cycle
        tape.cancel()
        await asyncio.gather(tape, return_exceptions=True)

    check(
        "a dropped tape resets the price buffer",
        buf2.resets >= 1,
        f"{buf2.resets} resets",
    )
    check(
        "reconnect does NOT clear active_positions",
        ex.active_positions == {"btc-updown-5m-LOCKED"},
        f"{ex.active_positions}",
    )
    check(
        "reconnect does NOT disturb open positions or PnL",
        len(ex.risk.positions) == 1,
        f"{len(ex.risk.positions)} positions held",
    )

    # A worthless leg with no bid must still mark down, or a real loss stays
    # invisible to the drawdown breaker until the moment it settles.
    class OneSidedBook(BookFeed):
        def __init__(self):  # noqa: super-init-not-called
            self.calls = 0

        async def fetch(self, token_ids):  # type: ignore[override]
            self.calls += 1
            return {
                "UP_TOKEN": Quote(bid=None, ask=None, tick_size=0.001),
                "DOWN_TOKEN": Quote(bid=0.98, ask=0.99, tick_size=0.001),
            }

    ex2 = Executor(ExecutionSettings(dry_run=True), Credentials(), RiskManager(fixed_size_risk()))
    ex2.risk.bankroll = 1_000.0
    ex2.risk.peak_equity = 1_000.0
    ex2.risk.open_position("btc-updown-5m-TEST", "UP_TOKEN", "YES", 100, 0.40)
    engine = SignalEngine(PriceBuffer(), OneSidedBook(), Config(), ex2)
    engine.sync_markets([make_market()])
    await engine.refresh_books()
    marked = ex2.risk.positions["btc-updown-5m-TEST:UP_TOKEN"].mark
    check(
        "a bidless leg marks off the opposite ask, not its entry price",
        abs(marked - 0.01) < 1e-9,
        f"mark={marked} (1 - 0.99), unrealized {ex2.risk.unrealized_pnl():+.2f}",
    )
    check(
        "that loss is visible to the drawdown breaker",
        ex2.risk.drawdown() > 38.0,
        f"drawdown=${ex2.risk.drawdown():.2f}",
    )

    # The venue's live tick size must override the Gamma snapshot: Polymarket
    # tightens the tick for extreme prices, and quantizing onto a stale 0.01
    # grid would price an order 10x away from the resting liquidity.
    market = engine._markets["btc-updown-5m-TEST"]
    check(
        "live book tick size overrides the Gamma snapshot",
        abs(market.tick_size - 0.001) < 1e-9,
        f"tick={market.tick_size} (Gamma said 0.01)",
    )
    check(
        "a 0.001 ask quantizes onto the tightened grid",
        abs(quantize_price(0.001, market.tick_size, BUY) - 0.001) < 1e-9,
        f"got {quantize_price(0.001, market.tick_size, BUY)}",
    )

    # A stalled tape must not poison the volatility estimate.
    buf3 = PriceBuffer()
    now = time.monotonic()
    buf3._bars.extend([(now - 400 + i, BASE) for i in range(60)])
    buf3._bars.append((now, BASE * 1.05))  # 5% jump across a long gap
    check(
        "vol estimator ignores returns spanning a tape gap",
        buf3.sigma_per_sqrt_second() < 1e-4,
        f"sigma={buf3.sigma_per_sqrt_second():.2e}",
    )


# --------------------------------------------------------------------------- #
# Latency optimizations
# --------------------------------------------------------------------------- #


async def test_fast_json() -> None:
    print("\n--- fast serialization ---")

    check(
        "a JSON backend is selected at import",
        JSON_BACKEND in {"orjson", "ujson", "json"},
        f"backend={JSON_BACKEND}",
    )

    payload = {"e": "trade", "s": "BTCUSDT", "p": "63630.89", "q": "0.01", "T": 1786517666170}
    encoded = _stdlib_json.dumps(payload)
    check("fast backend parses a str frame", json_loads(encoded)["p"] == "63630.89")
    check("fast backend parses a bytes frame", json_loads(encoded.encode())["p"] == "63630.89")

    # Every backend must raise a ValueError subclass so callers can catch one type.
    for bad in ("{not json", "", "[1,"):
        try:
            json_loads(bad)
            raised = False
        except ValueError:
            raised = True
        except Exception:
            raised = False
        if not raised:
            break
    check("malformed input raises ValueError on every backend", raised)

    # The tape must survive a corrupt frame rather than dying on it.
    buf = PriceBuffer()
    stream = BinanceTradeStream(buf, ["ws://unused"])
    for junk in (b"{bad", b"{}", b'{"p":"not-a-number"}', b'{"p":"0"}', b"[]"):
        stream._on_message(junk)
    check("corrupt frames are dropped, not fatal", buf.tick_count == 0)
    stream._on_message(encoded.encode())
    check("a valid frame after corruption is still ingested", buf.tick_count == 1)

    tick = buf.last()
    check(
        "each tick carries a perf_counter_ns T1 stamp",
        tick is not None and tick.perf_ns > 0,
        f"perf_ns={tick.perf_ns if tick else 0}",
    )

    # Deserialization must be fast enough to be irrelevant next to the network.
    n = 20_000
    t0 = time.perf_counter_ns()
    for _ in range(n):
        json_loads(encoded)
    per_op_us = (time.perf_counter_ns() - t0) / n / 1e3
    check(
        "tick deserialization stays well under 10us",
        per_op_us < 10.0,
        f"{per_op_us:.2f} us/op with {JSON_BACKEND}",
    )


async def test_order_caching() -> None:
    print("\n--- order payload pre-caching ---")

    ex = Executor(ExecutionSettings(dry_run=True), Credentials(), RiskManager(fixed_size_risk()))
    m1 = make_market("btc-updown-5m-AAA")
    m2 = make_market("btc-updown-5m-BBB")
    m2.up_token, m2.down_token = "UP_B", "DOWN_B"
    m2.tick_size, m2.neg_risk = 0.001, True

    t1 = ex.template_for(m1.up_token, m1)
    t2 = ex.template_for(m2.up_token, m2)

    check(
        "each market gets its own template",
        t1 is not t2 and t1.token_id != t2.token_id,
        f"{t1.token_id} vs {t2.token_id}",
    )
    check(
        "a second market does not pollute the first's tick size",
        t1.tick_size == 0.01 and t2.tick_size == 0.001,
        f"{t1.tick_size} / {t2.tick_size}",
    )
    check(
        "neg_risk stays per-market",
        t1.neg_risk is False and t2.neg_risk is True,
    )
    check(
        "options carry the venue-formatted tick string",
        t1.options.tick_size == "0.01" and t2.options.tick_size == "0.001",
        f"{t1.options.tick_size!r} / {t2.options.tick_size!r}",
    )
    check(
        "templates are cached, not rebuilt per call",
        ex.template_for(m1.up_token, m1) is t1,
    )

    # A tightened tick from the live book must rebuild the options object,
    # or the signed order would be quantized onto a grid the venue rejects.
    m1.tick_size = 0.001
    t1b = ex.template_for(m1.up_token, m1)
    check(
        "a tightened live tick rebuilds the cached options",
        t1b is t1 and t1.options.tick_size == "0.001",
        f"tick_size now {t1.options.tick_size!r}",
    )

    check("templates are held for both markets", ex.templates_cached == 2)
    ex.forget_market(m1.slug, (m1.up_token, m1.down_token))
    check(
        "retiring a market evicts only its own templates",
        ex.templates_cached == 1 and ex.template_for(m2.up_token, m2) is t2,
        f"{ex.templates_cached} cached",
    )

    # Templates key off token id, so a stale one can never be served to a market
    # that minted fresh tokens.
    m3 = make_market("btc-updown-5m-CCC")
    m3.up_token = "UP_C"
    check(
        "a new market's token gets a fresh template",
        ex.template_for(m3.up_token, m3).token_id == "UP_C",
    )

    check("tick strings render for the venue", _tick_str(0.01) == "0.01" and _tick_str(0.1) == "0.1")


async def test_latency_profiler() -> None:
    print("\n--- latency profiling ---")

    prof = LatencyProfiler(capacity=8)
    check("an empty profiler reports no data", prof.percentiles() is None)
    check("empty summary is safe to log", prof.summary() == "lat=n/a")

    base = 1_000_000_000
    for i in range(5):
        prof.record(base, base + 2_000_000, base + 52_000_000)  # 2ms signal, 50ms order
    pct = prof.percentiles()
    check(
        "T1->T2 and T2->T3 are reported in milliseconds",
        pct is not None and abs(pct[0] - 2.0) < 1e-6 and abs(pct[1] - 50.0) < 1e-6,
        f"signal {pct[0]:.2f}ms order {pct[1]:.2f}ms p95 {pct[2]:.2f}ms",
    )
    last = prof.last()
    check(
        "last() returns signal, order and total",
        last is not None and abs(last[2] - 52.0) < 1e-6,
        f"total {last[2]:.2f}ms",
    )
    check("incomplete samples are ignored", (prof.record(0, 1, 2), prof.count)[1] == 5)

    for _ in range(20):
        prof.record(base, base + 1_000_000, base + 10_000_000)
    check(
        "the sample window is bounded but the total keeps counting",
        len(prof._samples) == 8 and prof.count == 25,
        f"window={len(prof._samples)} total={prof.count}",
    )

    # Recording must be cheap enough to sit on the hot path.
    n = 50_000
    t0 = time.perf_counter_ns()
    for _ in range(n):
        prof.record(base, base + 1, base + 2)
    per_op_ns = (time.perf_counter_ns() - t0) / n
    check(
        "recording a sample costs well under 1us",
        per_op_ns < 1000,
        f"{per_op_ns:.0f} ns/op",
    )

    # End to end: a dry-run order populates the profiler through the real path.
    logs, ex, _ = await drive(up_ask=0.45, down_ask=0.56)
    check(
        "a completed order records a latency sample",
        ex.latency.count == 1,
        ex.latency.summary(),
    )
    sample = ex.latency.last()
    check(
        "T1 comes from the tick, so T1->T2 spans real signal work",
        sample is not None and sample[0] > 0.0,
        f"T1->T2 {sample[0]:.2f} ms",
    )


async def test_eval_loop_benchmark() -> None:
    print("\n--- evaluation loop benchmark ---")

    buf = PriceBuffer()
    now_price = 100_000.0
    for i in range(2_000):  # ~10s of a busy tape
        buf.add(now_price * (1.0 + (i % 7) * 1e-5), 0)

    n = 2_000
    t0 = time.perf_counter_ns()
    for _ in range(n):
        buf.largest_move(3.0)
    move_us = (time.perf_counter_ns() - t0) / n / 1e3
    # Compare against the naive per-tick logarithm the scan replaced.
    ticks = list(buf._ticks)
    latest = ticks[-1]
    t0 = time.perf_counter_ns()
    for _ in range(n):
        best = None
        for tk in ticks:
            if tk is latest:
                continue
            d = math.log(latest.price / tk.price)
            if best is None or abs(d) > abs(best):
                best = d
    naive_us = (time.perf_counter_ns() - t0) / n / 1e3
    check(
        "spike detection over a full buffer stays under 1ms",
        move_us < 1000.0,
        f"{move_us:.1f} us/pass over {len(buf._ticks)} ticks "
        f"({naive_us:.1f} us for the per-tick log() it replaced, "
        f"{naive_us / move_us:.1f}x)",
    )

    # Bars are sampled on a 1s grid, so a burst of ticks in one instant yields a
    # single bar. Populate a realistic 300-bar history directly.
    mono = time.monotonic()
    buf._bars.clear()
    buf._bars.extend((mono - 300 + i, now_price * (1.0 + (i % 11) * 2e-5)) for i in range(300))
    t0 = time.perf_counter_ns()
    for _ in range(2_000):
        buf.sigma_per_sqrt_second()
    sigma_us = (time.perf_counter_ns() - t0) / 2_000 / 1e3
    check(
        "vol estimation allocates no intermediate list and stays under 1ms",
        sigma_us < 1000.0,
        f"{sigma_us:.1f} us/pass over {len(buf._bars)} bars",
    )

    check(
        "tick buffer is bounded by maxlen",
        buf._ticks.maxlen is not None and buf._bars.maxlen is not None,
        f"ticks maxlen={buf._ticks.maxlen} bars maxlen={buf._bars.maxlen}",
    )

    # The optimized min/max scan must agree with the naive per-tick search.
    naive = PriceBuffer()
    for price in (100.0, 101.0, 99.0, 100.5):
        naive.add(price, 0)
    mv = naive.largest_move(3.0)
    expected = math.log(100.5 / 99.0)  # largest excursion is from the 99.0 low
    check(
        "the min/max scan finds the same move as a per-tick search",
        mv is not None and abs(mv.delta_log - expected) < 1e-12,
        f"got {mv.bps:+.2f} bps, expected {expected * 1e4:+.2f} bps",
    )

    # The O(1) tripwire: the tape must wake the evaluator on a qualifying tick
    # rather than the evaluator waiting out its polling period.
    trip = PriceBuffer()
    fired = {"n": 0}
    trip.set_wake(lambda: fired.__setitem__("n", fired["n"] + 1))
    for _ in range(20):
        trip.add(100_000.0, 0)
    trip.arm_trigger(12.0)  # 12 bps
    trip.add(100_000.0 * 1.0005, 0)  # +5 bps, inside the threshold
    check("a sub-threshold tick does not wake the evaluator", fired["n"] == 0)
    trip.add(100_000.0 * 1.0020, 0)  # +20 bps, past it
    check("a qualifying tick wakes the evaluator immediately", fired["n"] == 1)
    trip.add(100_000.0 * 1.0030, 0)
    check(
        "the tripwire debounces until re-armed",
        fired["n"] == 1,
        "one wake per arm, not one per tick",
    )
    trip.arm_trigger(12.0)
    trip.add(100_000.0 * 0.9950, 0)  # large move down
    check("re-arming re-enables the tripwire in both directions", fired["n"] == 2)

    # It must agree with the authoritative scan about what qualifies.
    agree = PriceBuffer()
    agree.set_wake(lambda: None)
    for _ in range(10):
        agree.add(100_000.0, 0)
    agree.arm_trigger(12.0)
    agree.add(100_000.0 * 1.0015, 0)
    mv2 = agree.largest_move(3.0)
    check(
        "tripwire and largest_move agree on a qualifying move",
        mv2 is not None and abs(mv2.bps) >= 12.0 and agree.wakes == 1,
        f"{mv2.bps:+.1f} bps, wakes={agree.wakes}",
    )

    # An absurd threshold must not raise on the tick path.
    safe = PriceBuffer()
    for _ in range(5):
        safe.add(100_000.0, 0)
    safe.arm_trigger(1e9)
    safe.add(200_000.0, 0)
    check("an unreachable threshold arms without overflowing", safe.wakes == 0)

    n2 = 100_000
    armed = PriceBuffer()
    armed.add(100_000.0, 0)
    armed.arm_trigger(12.0)
    t0 = time.perf_counter_ns()
    for _ in range(n2):
        armed.add(100_000.0, 0)
    add_ns = (time.perf_counter_ns() - t0) / n2
    # At Binance's ~200 trades/s peak, 5us/tick is 0.1% of a core - the tape
    # must never be the bottleneck, but chasing below that buys nothing.
    check(
        "the per-tick ingest path stays cheap with the tripwire armed",
        add_ns < 5000,
        f"{add_ns:.0f} ns/tick = {add_ns * 200 / 1e7:.3f}% of a core at 200 ticks/s",
    )

    # Whole-evaluation timing against a stubbed book.
    ex = Executor(ExecutionSettings(dry_run=True), Credentials(), RiskManager(fixed_size_risk()))
    await ex.connect()
    engine = SignalEngine(buf, StubBookFeed(0.45, 0.56), Config(spike_bps=1e9), ex)
    engine.sync_markets([make_market()])
    await engine.refresh_books()

    t0 = time.perf_counter_ns()
    for _ in range(500):
        await engine.evaluate()
    eval_us = (time.perf_counter_ns() - t0) / 500 / 1e3
    check(
        "a full evaluate() pass stays well inside the 10 Hz budget",
        eval_us < 5_000.0,
        f"{eval_us:.1f} us/pass (budget 100,000 us at 10 Hz)",
    )

    # The headline number: T1 -> T2 against a zero-latency book, i.e. the whole
    # controllable pipeline (parse, buffer, spike scan, fair value, sizing, risk
    # gates, payload construction) with the network removed. Anything above this
    # in a live run is round-trip time, not our code.
    samples: list[float] = []
    for _ in range(10):
        buf2 = PriceBuffer()
        ex2 = Executor(
            ExecutionSettings(dry_run=True), Credentials(), RiskManager(fixed_size_risk())
        )
        await ex2.connect()
        eng2 = SignalEngine(
            buf2,
            StubBookFeed(0.45, 0.56),
            Config(
                spike_bps=12.0, min_edge=0.03, enforce_ask_ceiling=False, signal_cooldown=0.0
            ),
            ex2,
        )
        eng2.sync_markets([make_market()])
        for _ in range(60):
            buf2.add(BASE, 0)
        await eng2.refresh_books()  # snapshot that will serve as the anchor
        await asyncio.sleep(0.02)
        await eng2.refresh_books()
        buf2.add(BASE, 0)  # flat tick after the anchor
        await eng2.evaluate()  # arms the tripwire
        buf2.add(BASE * 1.0025, 0, time.perf_counter_ns())  # the spike: T1
        await eng2.evaluate()  # event-driven, no polling delay
        last2 = ex2.latency.last()
        if last2:
            samples.append(last2[0])

    median_ms = sorted(samples)[len(samples) // 2] if samples else 999.0
    check(
        "signal-to-payload (T1->T2) stays under 5ms with the network removed",
        samples and median_ms < 5.0,
        f"{median_ms:.3f} ms median over {len(samples)} spikes "
        f"-- the rest of any live figure is round-trip time",
    )


# --------------------------------------------------------------------------- #
# Polymarket US adapter (read-only)
# --------------------------------------------------------------------------- #


async def test_polymarket_us() -> None:
    print("\n--- polymarket US adapter (read-only) ---")

    import base64
    import inspect

    import polymarket_us as pu
    from polymarket_us import (
        AuthError,
        Ed25519Signer,
        PolymarketUSClient,
        USCredentials,
        summarize_balances,
    )

    # The safety property this module exists for: it must not be able to trade.
    methods = [
        n for n, _ in inspect.getmembers(PolymarketUSClient, inspect.isfunction)
        if not n.startswith("_")
    ]
    banned = [
        m for m in methods
        if any(w in m.lower() for w in
               ("order", "cancel", "modify", "create", "place", "submit", "trade", "buy", "sell"))
    ]
    check("the US client exposes no order-capable method", not banned, f"methods={methods}")
    src = inspect.getsource(pu)
    writes = sum(src.count(f".{verb}(") for verb in ("post", "put", "delete"))
    check("the US module issues no POST/PUT/DELETE at all", writes == 0, f"{writes} found")

    # Credential validation must name the actual problem.
    check("missing credentials are reported", len(USCredentials().problems()) == 2)
    check(
        "a non-base64 secret is rejected with a reason",
        "base64" in (USCredentials(key_id="k", secret_key="!!!not-base64!!!").seed_error() or ""),
    )
    short = USCredentials(key_id="k", secret_key=base64.b64encode(b"tooshort").decode())
    check(
        "an under-length secret is rejected",
        "at least 32" in (short.seed_error() or ""),
        short.seed_error() or "",
    )
    good = USCredentials(key_id="k", secret_key=base64.b64encode(bytes(64)).decode())
    check(
        "a 64-byte secret yields a 32-byte Ed25519 seed",
        good.seed_error() is None and len(good.seed()) == 32,
    )
    check(
        "describe() never leaks the secret",
        "set" in good.describe() and base64.b64encode(bytes(64)).decode() not in good.describe(),
        good.describe(),
    )

    # Signature construction must match the documented scheme exactly.
    signer = Ed25519Signer(good)
    h1 = signer.headers("GET", "/v1/account/balances")
    check(
        "signing produces the three documented headers",
        {"X-PM-Access-Key", "X-PM-Timestamp", "X-PM-Signature"} <= set(h1),
        sorted(h1),
    )
    sig = base64.b64decode(h1["X-PM-Signature"])
    check("the signature is a 64-byte Ed25519 signature", len(sig) == 64, f"{len(sig)} bytes")
    check("the timestamp is in milliseconds", len(h1["X-PM-Timestamp"]) == 13)

    # Verify against the message the docs specify: timestamp + METHOD + path.
    from cryptography.hazmat.primitives.asymmetric import ed25519 as _ed

    pub = _ed.Ed25519PrivateKey.from_private_bytes(good.seed()).public_key()
    message = f"{h1['X-PM-Timestamp']}GET/v1/account/balances".encode()
    try:
        pub.verify(sig, message)
        verified = True
    except Exception:
        verified = False
    check("signature verifies over `{timestamp}{METHOD}{path}`", verified)

    h2 = signer.headers("GET", "/v1/account/balances")
    check(
        "a different path or time produces a different signature",
        signer.headers("GET", "/v1/portfolio/positions")["X-PM-Signature"]
        != h2["X-PM-Signature"],
    )

    # An unauthenticated client must refuse to sign rather than send unsigned.
    class _NoSession:
        pass

    client = PolymarketUSClient(_NoSession(), USCredentials())  # type: ignore[arg-type]
    check("an unauthenticated client is flagged", not client.authenticated)
    try:
        await client.account_balances()
        raised = False
    except AuthError:
        raised = True
    except Exception:
        raised = False
    check("an authenticated read without a signer raises AuthError", raised)
    try:
        PolymarketUSClient(_NoSession(), USCredentials()).authenticate()  # type: ignore[arg-type]
        built = True
    except AuthError:
        built = False
    check("authenticate() refuses to build a signer with no credentials", not built)

    # Balance parsing must tolerate the shapes the venue may return.
    shapes = [
        ({"balances": [{"currency": "USD", "cashBalance": 3.21}]}, 3.21),
        ({"accountBalances": [{"currency": "USD", "cash": 3.21}]}, 3.21),
        ([{"currency": "USD", "balance": "3.21"}], 3.21),
        ({"currency": "USD", "cashBalance": 3.21}, 3.21),
    ]
    ok = all(
        summarize_balances(p) and abs(summarize_balances(p)[0][1] - want) < 1e-9
        for p, want in shapes
    )
    check("balances parse across every plausible response shape", ok)
    check("an unrecognized payload yields no rows, not a false zero", summarize_balances({}) == [])


# --------------------------------------------------------------------------- #
# Kalshi adapter (read-only)
# --------------------------------------------------------------------------- #


async def test_kalshi() -> None:
    print("\n--- kalshi adapter (read-only) ---")

    import inspect

    import kalshi as kx
    from kalshi import (
        KalshiBook,
        KalshiClient,
        KalshiCredentials,
        breakeven_fair_value,
        fee_per_contract,
        net_edge,
        parse_market,
        quantize_kalshi_price,
        trading_fee,
    )

    # Read-only, same property as the Polymarket US adapter.
    methods = [
        n for n, _ in inspect.getmembers(KalshiClient, inspect.isfunction)
        if not n.startswith("_")
    ]
    banned = [m for m in methods if any(w in m.lower() for w in
              ("order", "cancel", "modify", "create", "place", "submit", "buy", "sell"))]
    banned = [m for m in banned if m != "orderbook"]  # orderbook is a read
    check("the Kalshi client exposes no order-placing method", not banned, f"{methods}")
    src = inspect.getsource(kx)
    check(
        "the Kalshi module issues no POST/PUT/DELETE",
        sum(src.count(f".{v}(") for v in ("post", "put", "delete")) == 0,
    )

    # Fees. This is what decides whether the strategy is viable at all.
    check("fee is zero outside (0,1)", trading_fee(0.0, 10) == 0 and trading_fee(1.0, 10) == 0)
    check(
        "fee peaks at 50c, which is where these markets open",
        abs(fee_per_contract(0.50) - 0.0175) < 1e-9,
        f"{fee_per_contract(0.50) * 100:.2f}c/contract",
    )
    check(
        "fee is smaller at the extremes",
        fee_per_contract(0.05) < fee_per_contract(0.20) < fee_per_contract(0.50),
    )
    check(
        "total fee rounds UP to the cent",
        abs(trading_fee(0.50, 20) - 0.35) < 1e-9,
        f"20 contracts @ 0.50 = ${trading_fee(0.50, 20):.2f}",
    )
    check(
        "net edge subtracts the fee from the gross",
        abs(net_edge(0.55, 0.50) - (0.05 - 0.0175)) < 1e-9,
        f"gross 0.05 -> net {net_edge(0.55, 0.50):.4f}",
    )
    check(
        "breakeven fair value sits above the ask by exactly the fee",
        abs(breakeven_fair_value(0.50) - 0.5175) < 1e-9,
    )
    check(
        "a 3c gross edge at 50c is more than half consumed by fees",
        net_edge(0.53, 0.50) < 0.015,
        f"net {net_edge(0.53, 0.50) * 100:.2f}c of a 3c gross edge",
    )

    # Tapered tick ladder.
    ladder = [
        {"start": "0.0000", "end": "0.1000", "step": "0.0010"},
        {"start": "0.1000", "end": "0.9000", "step": "0.0100"},
        {"start": "0.9000", "end": "1.0000", "step": "0.0010"},
    ]
    check(
        "sub-cent ticks are honoured below $0.10",
        abs(quantize_kalshi_price(0.0990, ladder, True) - 0.099) < 1e-9,
        f"0.0990 -> {quantize_kalshi_price(0.0990, ladder, True)}",
    )
    check(
        "cent ticks apply in the middle band",
        abs(quantize_kalshi_price(0.1234, ladder, True) - 0.13) < 1e-9,
        f"0.1234 -> {quantize_kalshi_price(0.1234, ladder, True)}",
    )
    check(
        "sub-cent ticks return above $0.90",
        abs(quantize_kalshi_price(0.9123, ladder, True) - 0.913) < 1e-9,
        f"0.9123 -> {quantize_kalshi_price(0.9123, ladder, True)}",
    )
    check(
        "buys round up and sells round down",
        quantize_kalshi_price(0.1234, ladder, True) > quantize_kalshi_price(0.1234, ladder, False),
    )

    # Book convention: both ladders are BIDS; the YES ask is the NO complement.
    book = KalshiBook.from_payload({"orderbook_fp": {
        "yes_dollars": [["0.0010", "5"], ["0.0850", "12"]],
        "no_dollars": [["0.0010", "7"], ["0.9100", "9"]],
    }})
    check(
        "YES ask is derived as 1 - best NO bid, not read off the yes ladder",
        abs(book.yes_ask - 0.09) < 1e-9 and abs(book.yes_bid - 0.085) < 1e-9,
        f"bid {book.yes_bid} / ask {book.yes_ask}",
    )
    check("mid sits between the two", abs(book.yes_mid - 0.0875) < 1e-9)
    empty = KalshiBook.from_payload({"orderbook_fp": {"yes_dollars": [], "no_dollars": []}})
    check("an empty book yields None, not a false zero", empty.yes_ask is None)

    # Market parsing and fair value from the PUBLISHED strike.
    raw = {
        "ticker": "KXBTC15M-TEST-15", "event_ticker": "KXBTC15M-TEST",
        "title": "BTC price up in next 15 mins?", "floor_strike": "63777.35",
        "open_time": "2026-08-13T06:15:00Z", "close_time": "2026-08-13T06:30:00Z",
        "status": "active", "yes_bid_dollars": "0.47", "yes_ask_dollars": "0.48",
        "volume_fp": "779773", "open_interest_fp": "354832", "price_ranges": ladder,
    }
    m = parse_market(raw)
    check("market parses with its published strike", m is not None and m.strike == 63777.35)

    now = m.close_ts - 600.0  # 10 minutes left
    check(
        "effective_tau matches the Polymarket derivation (tau - 2L/3)",
        abs(m.effective_tau(now) - (600.0 - 40.0)) < 1e-6,
        f"{m.effective_tau(now):.0f}s",
    )
    near = m.close_ts - 30.0  # inside the averaging window
    check(
        "the tau < L branch switches to tau/3",
        abs(m.effective_tau(near) - 10.0) < 1e-6,
        f"{m.effective_tau(near):.1f}s",
    )

    sigma = 0.8e-4
    at_strike = m.fair_value(63777.35, sigma, now)
    check("spot exactly at the strike prices near 50/50", abs(at_strike - 0.5) < 1e-6)
    above = m.fair_value(63777.35 * 1.001, sigma, now)
    below = m.fair_value(63777.35 * 0.999, sigma, now)
    check(
        "above the strike is more likely than below, symmetrically",
        above > 0.5 > below and abs((above - 0.5) - (0.5 - below)) < 1e-3,
        f"+10bps -> {above:.3f}, -10bps -> {below:.3f}",
    )

    # The USDT-vs-USD trap that produced a phantom 30-point edge live.
    usdt_spot = 63_841.94          # Binance BTCUSDT
    usd_spot = 63_766.69           # USD composite at the same instant
    fair_wrong = m.fair_value(usdt_spot, sigma, now)
    fair_right = m.fair_value(usd_spot, sigma, now)
    check(
        "a USDT-quoted feed misprices this USD-settled contract badly",
        abs(fair_wrong - fair_right) > 0.20,
        f"USDT feed {fair_wrong:.3f} vs USD feed {fair_right:.3f} "
        f"= {abs(fair_wrong - fair_right) * 100:.0f} points of error",
    )
    check(
        "the USD feed lands near the market's own quote (0.47/0.48)",
        abs(fair_right - 0.475) < 0.10,
        f"fair {fair_right:.3f} vs market 0.475",
    )

    # Feed basis correction. This was the dominant source of phantom edge in
    # the first hour-long run: a single venue sits ~1.7 bps off the composite,
    # which on a 15-minute contract is ~5 points of probability.
    from kalshi import CompositeBasis

    basis = CompositeBasis.__new__(CompositeBasis)
    basis.offset, basis.samples, basis.last_composite = 0.0, 0, None
    basis._alpha = 0.5
    check("an unmeasured basis is a no-op", basis.correct(63_000.0) == 63_000.0)
    basis.offset = 10.41
    check(
        "a measured basis lifts the tape toward the composite",
        abs(basis.correct(63_000.0) - 63_010.41) < 1e-9,
    )

    m_off = parse_market(raw)
    at_raw = m_off.fair_value(63_777.35, sigma, now)
    at_corrected = m_off.fair_value(63_777.35 + 10.41, sigma, now)
    check(
        "a 1.7 bps feed error is worth several points of probability",
        (at_corrected - at_raw) > 0.03,
        f"{at_raw:.3f} -> {at_corrected:.3f} = {(at_corrected - at_raw) * 100:.1f} points",
    )

    # Credentials.
    check("missing Kalshi credentials are reported", len(KalshiCredentials().problems()) == 2)
    check(
        "a non-PEM private key is rejected",
        any("PEM" in p for p in KalshiCredentials(key_id="k", private_key_pem="nope").problems()),
    )
    check(
        "describe() never leaks the key material",
        "PRIVATE" not in KalshiCredentials(key_id="abcdefgh1234", private_key_pem="x").describe(),
    )

    # RSA-PSS signing, against a generated key.
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    signer = kx.RsaPssSigner(KalshiCredentials(key_id="test-key", private_key_pem=pem))
    headers = signer.headers("GET", "/trade-api/v2/portfolio/balance")
    check(
        "signing produces the three KALSHI-ACCESS headers",
        {"KALSHI-ACCESS-KEY", "KALSHI-ACCESS-TIMESTAMP", "KALSHI-ACCESS-SIGNATURE"} <= set(headers),
    )
    import base64 as _b64

    message = f"{headers['KALSHI-ACCESS-TIMESTAMP']}GET/trade-api/v2/portfolio/balance".encode()
    try:
        key.public_key().verify(
            _b64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"]),
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        verified = True
    except Exception:
        verified = False
    check("signature verifies as RSA-PSS over `{timestamp}{METHOD}{path}`", verified)


# --------------------------------------------------------------------------- #
# Run logging
# --------------------------------------------------------------------------- #


async def test_run_log() -> None:
    print("\n--- per-run log files ---")

    import datetime as dt
    import tempfile
    from pathlib import Path

    from run_log import run_log_name, start_run_log

    cases = [
        ("2026-08-13T06:32:45+00:00", "L_081326_063245.log"),
        ("2026-01-05T00:00:00+00:00", "L_010526_000000.log"),
        ("2026-12-31T23:59:59+00:00", "L_123126_235959.log"),
    ]
    ok = all(run_log_name(dt.datetime.fromisoformat(iso)) == want for iso, want in cases)
    check("filename is L_MMDDYY_HHMMSS.log", ok, cases[0][1])

    # A non-UTC instant must be converted, not formatted as-is.
    local = dt.datetime(2026, 8, 13, 6, 32, 45, tzinfo=dt.timezone(dt.timedelta(hours=-5)))
    check(
        "the timestamp is UTC even when given a local-time instant",
        run_log_name(local) == "L_081326_113245.log",
        f"06:32:45 UTC-5 -> {run_log_name(local)}",
    )

    logger = logging.getLogger("run-log-test")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # keep the suite's own output clean
    with tempfile.TemporaryDirectory() as tmp:
        run = start_run_log(logger, directory=tmp, title="TEST RUN", context=["ctx : value"])
        check("a log file is created on start", run is not None and run.path.is_file())
        logger.info("a recorded line")
        logger.warning("a warning line")
        run.close(["summary : 42"])

        text = run.path.read_text(encoding="utf-8")
        check("the header names the run", "TEST RUN" in text and "ctx : value" in text)
        check("the command line is recorded", "command :" in text)
        check("records are captured", "a recorded line" in text and "a warning line" in text)
        check("the footer holds the summary", "SESSION SUMMARY" in text and "summary : 42" in text)
        check("elapsed time is reported", "elapsed :" in text)
        check(
            "timestamps are full dates, readable without context",
            any(line.startswith("2") and "-" in line[:10] for line in text.splitlines()),
        )
        check("close() is idempotent", (run.close(), True)[1])
        check(
            "the handler is detached after close",
            run.handler not in logger.handlers,
        )

        # A second run in the same second must not clobber the first.
        again = start_run_log(logger, directory=tmp, title="SECOND")
        check("a second run opens its own handler", again is not None)
        again.close()
        check(
            "log files land in the requested directory",
            len(list(Path(tmp).glob("L_*.log"))) >= 1,
        )

    # Logging must never be the reason a run fails.
    broken = start_run_log(logger, directory="/proc/definitely-not-writable")
    check("an unwritable directory degrades to console-only", broken is None)


# --------------------------------------------------------------------------- #
# Strategies: the strike guard and the volatility cross-check
# --------------------------------------------------------------------------- #


async def test_strategies() -> None:
    print("\n--- strategies: strike guard and volatility cross-check ---")

    import math as _math
    from statistics import NormalDist

    from kalshi import KalshiBook, parse_market
    from strategies import (
        implied_sigma,
        scan_cross,
        scan_endgame,
        scan_stale,
        vol_agreement,
    )

    def market(strike: str, close: str = "2026-08-13T06:30:00Z"):
        return parse_market({
            "ticker": "KXBTC15M-TEST-15", "event_ticker": "KXBTC15M-TEST",
            "title": "BTC price up in next 15 mins?", "floor_strike": strike,
            "open_time": "2026-08-13T06:15:00Z", "close_time": close,
            "status": "active", "volume_fp": "1000", "open_interest_fp": "500",
        })

    def book(yes_bid: float, yes_ask: float) -> KalshiBook:
        # Kalshi publishes two bid ladders; the YES ask is 1 - best NO bid.
        return KalshiBook.from_payload({"orderbook_fp": {
            "yes_dollars": [[f"{yes_bid:.4f}", "500"]],
            "no_dollars": [[f"{1.0 - yes_ask:.4f}", "500"]],
        }})

    # -- the strike guard ---------------------------------------------------- #
    # Kalshi lists the contract before floor_strike posts. The log from a live
    # run showed `Tracking KXBTC15M-26AUG132015-15 | strike $0.00` followed by
    # 30-45s of heartbeats reading fair=0.500 against a book at 0.15/0.16 -
    # a fabricated 34c edge on every one of those passes.
    pending = market("")
    check("a market with no published strike reports strike_known False",
          pending is not None and not pending.strike_known, f"strike={pending.strike}")
    check(
        "fair_value returns None rather than a 0.5 that reads as a coin flip",
        pending.fair_value(63_000.0, 0.8e-4) is None,
    )
    live = market("63777.35")
    check("a published strike prices normally", live.strike_known
          and live.fair_value(63_777.35, 0.8e-4, live.close_ts - 600.0) is not None)

    # The scanners read the wall clock, so these fixtures need a close time in
    # the real future rather than the fixed one used for the pricing checks.
    import time as _time

    def closing_in(seconds: float) -> str:
        return _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime(_time.time() + seconds))

    b = book(0.15, 0.16)
    live_now = market("63777.35", close=closing_in(600.0))
    pending_now = market("", close=closing_in(600.0))

    check(
        "STALE refuses to fire while the strike is unpublished",
        scan_stale(pending_now, b, 63_000.0, 0.50, 63_400.0, 20.0, 0.01, 0.8e-4) is None,
    )
    check(
        "ENDGAME refuses to fire while the strike is unpublished",
        scan_endgame(pending_now, b, 63_400.0, 20.0, 0.8e-4,
                     max_seconds_left=1e9, min_z=0.0) is None,
    )
    # CROSS is exempt: it never forms an opinion about the strike, so a missing
    # one is no reason to sit out a locked profit.
    crossed = book(0.60, 0.35)  # yes bid 0.60 > yes ask 0.35: the bids cross
    check(
        "CROSS still fires with no strike, because it never uses one",
        scan_cross(pending_now, crossed, 20.0, 0.01) is not None,
    )

    # -- volatility agreement ------------------------------------------------ #
    check("agreement is symmetric and >= 1",
          abs(vol_agreement(2e-4, 1e-4) - 2.0) < 1e-9
          and abs(vol_agreement(1e-4, 2e-4) - 2.0) < 1e-9)
    check("perfect agreement is exactly 1.0", abs(vol_agreement(1e-4, 1e-4) - 1.0) < 1e-12)
    check("an uninvertible quote counts as unvalidated, not as agreement",
          vol_agreement(1e-4, None) is None)

    # An implied sigma, recovered from a quote we construct at a known value.
    tau = live_now.effective_tau()
    true_sigma = 1.2e-4
    spot = 63_777.35 * _math.exp(true_sigma * _math.sqrt(tau) * 0.8)  # z = +0.8
    target_mid = NormalDist().cdf(0.8)
    quote = book(target_mid - 0.005, target_mid + 0.005)
    recovered = implied_sigma(live_now, quote, spot)
    check(
        "implied sigma inverts the quote back to the volatility that made it",
        recovered is not None and abs(recovered / true_sigma - 1.0) < 0.02,
        f"{recovered * 1e4:.3f} vs {true_sigma * 1e4:.3f} bps/s",
    )

    # The degeneracy that forces ENDGAME to use MEASURED sigma: feeding it the
    # market's own implied value reproduces the market's own z, so fair value
    # equals the mid and the edge is half the spread minus the fee - negative
    # by construction. Worth a test, because the alternative looks reasonable.
    em = market("63777.35", close=closing_in(90.0))
    etau = em.effective_tau()
    esigma = 1.0e-4
    espot = 63_777.35 * _math.exp(esigma * _math.sqrt(etau) * 4.0)  # 4 sigma above
    # A realistic late-window quote on a near-decided contract: 97/98, not the
    # 0.9999 the model would like. That gap is exactly what ENDGAME buys.
    ebook = book(0.97, 0.98)
    eimplied = implied_sigma(em, ebook, espot)
    check(
        "ENDGAME on implied sigma is degenerate - no edge exists by construction",
        eimplied is not None
        and scan_endgame(em, ebook, espot, 20.0, eimplied, min_z=0.5, min_edge=0.0) is None,
    )
    with_measured = scan_endgame(em, ebook, espot, 20.0, esigma * 0.5, min_z=0.5, min_edge=0.0)
    check(
        "ENDGAME only finds an edge when our sigma differs from the market's",
        with_measured is not None,
        f"implied {eimplied * 1e4:.2f} vs measured {esigma * 0.5 * 1e4:.2f} bps/s",
    )

    # -- the graded failure mode: no real spot move, no trade ---------------- #
    # Session L_081426_195232 bought four STALE trades on spot moves of 0.1 to
    # 2.7 bps - pure noise, with the whole "edge" being the book repricing away
    # from its own stale anchor mid - and won one of four. These assert that
    # class of trade is now structurally impossible.
    rich_book = book(0.25, 0.26)  # book dropped from an anchor mid of 0.285
    anchor_spot = 63_400.0
    still_spot = anchor_spot * math.exp(0.2 / 10_000.0)  # spot moved 0.2 bps
    check(
        "STALE refuses to fade a book move when spot has not really moved",
        scan_stale(live_now, rich_book, anchor_spot, 0.285, still_spot,
                   20.0, 0.01, 0.8e-4) is None,
    )
    moved_spot = anchor_spot * math.exp(12.0 / 10_000.0)  # a real 12 bps jump
    check(
        "a real spot move past the threshold can still signal",
        scan_stale(live_now, rich_book, anchor_spot, 0.285, moved_spot,
                   20.0, 0.01, 0.8e-4) is not None,
    )
    check(
        "the threshold is tunable and respected",
        scan_stale(live_now, rich_book, anchor_spot, 0.285, moved_spot,
                   20.0, 0.01, 0.8e-4, min_move_bps=15.0) is None,
    )

    # Trade 4's degenerate inversion: mid 0.535 (z = 0.09) produced sigma =
    # 0.06 bps/s from two near-zero inputs and hypersensitised the model.
    near_money = book(0.53, 0.54)
    check(
        "a near-the-money quote no longer yields a garbage implied sigma",
        implied_sigma(live_now, near_money, 63_778.9) is None,
    )

    # -- STALE without an invertible quote ----------------------------------- #
    atm = book(0.495, 0.505)  # z ~ 0, cannot be inverted
    check(
        "STALE sits out when the quote cannot supply a volatility",
        implied_sigma(live_now, atm, 63_777.40) is None
        and scan_stale(live_now, atm, 63_000.0, 0.50, 63_600.0, 20.0, 0.01, 0.8e-4) is None,
    )
    check(
        "the fallback is reachable only by opting in explicitly",
        scan_stale(live_now, atm, 63_000.0, 0.50, 63_600.0, 20.0, 0.01, 0.8e-4,
                   require_implied=False) is not None,
    )


# --------------------------------------------------------------------------- #
# Credential setup and the confirm-over-time gate
# --------------------------------------------------------------------------- #


async def test_setup_and_confirmation() -> None:
    print("\n--- credential setup and edge confirmation ---")

    import argparse as _ap
    import os
    import stat as _stat
    import tempfile
    import time as _time
    from pathlib import Path

    from kalshi_setup import PEM_FILE, save_credentials
    from strategies import Leg, PaperLedger, Signal

    # -- .env persistence ---------------------------------------------------- #
    with tempfile.TemporaryDirectory() as tmp:
        env = Path(tmp) / ".env"
        env.write_text(
            "# comment survives\nOTHER_KEY=untouched\nKALSHI_API_KEY_ID=old-id\n",
            encoding="utf-8",
        )
        save_credentials(env, "new-id", pem_text="-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\n")
        text = env.read_text(encoding="utf-8")
        check("an existing key line is replaced, not duplicated",
              text.count("KALSHI_API_KEY_ID") == 1 and "new-id" in text and "old-id" not in text)
        check("unrelated lines and comments survive the rewrite",
              "# comment survives" in text and "OTHER_KEY=untouched" in text)
        key_file = Path(tmp) / PEM_FILE
        check("a pasted PEM lands in its own file, referenced by path",
              key_file.is_file() and str(key_file) in text)
        mode = _stat.S_IMODE(key_file.stat().st_mode)
        check("the key file is owner-only (600)", mode == 0o600, oct(mode))
        check("the process environment is updated without a restart",
              os.environ.get("KALSHI_API_KEY_ID") == "new-id")

        # A path-based key must not create a copy of the PEM.
        save_credentials(env, "path-id", pem_path="/somewhere/key.pem")
        text = env.read_text(encoding="utf-8")
        check("a path-based key is stored as the path alone",
              "KALSHI_PRIVATE_KEY_PATH=/somewhere/key.pem" in text
              and text.count("KALSHI_PRIVATE_KEY_PATH") == 1)
    for var in ("KALSHI_API_KEY_ID", "KALSHI_PRIVATE_KEY_PATH"):
        os.environ.pop(var, None)

    # -- the confirm-over-time gate ------------------------------------------ #
    from kalshi_monitor import Monitor

    def fresh_monitor(confirm_seconds: float, confirm_passes: int, tmp: str) -> Monitor:
        args = _ap.Namespace(
            confirm_seconds=confirm_seconds, confirm_passes=confirm_passes,
            series="KXBTC15M", min_edge=0.02, spike_bps=12.0, size=20.0,
            min_seconds_left=20.0, book_interval=1.0, discovery_interval=10.0,
            cooldown=5.0, heartbeat=15.0, binance=False, live=False,
            max_stake_pct=8.0, max_exposure_pct=25.0, daily_loss_pct=20.0,
            max_trades=40, min_profit=0.01, anchor_age=20.0, stale_min_move=8.0,
            endgame_window=120.0, endgame_z=3.0, vol_ratio_max=1.5,
            allow_unvalidated_vol=False, no_cross=False, no_stale=False,
            no_endgame=False, no_basis=True, log_dir=tmp, no_log=True,
            env_file=".env", verbose=False,
        )
        mon = Monitor(args)
        mon.ledger = PaperLedger(Path(tmp) / "paper_test.jsonl")
        return mon

    def sig(side: str = "YES") -> Signal:
        return Signal(
            strategy="STALE", ticker="KXBTC15M-TEST-15",
            legs=[Leg(side, 0.40, 20.0)], fair_yes=0.48, expected_net=0.60,
            max_loss=8.0, spot=63_400.0, strike=63_380.0, seconds_left=500.0,
            sigma_used=1.2e-4,
        )

    with tempfile.TemporaryDirectory() as tmp:
        mon = fresh_monitor(0.05, 3, tmp)
        mon._emit(sig())
        check("a first sighting is held, not recorded", not mon._pending_orders)
        mon._emit(sig())
        check("a second sighting inside the window is still held",
              not mon._pending_orders)
        _time.sleep(0.06)
        mon._emit(sig())
        check("the third sighting past the window confirms and queues",
              len(mon._pending_orders) == 1)
        check("the confirmation is stamped into the signal's note",
              "confirmed over" in mon._pending_orders[0].note)
        mon._emit(sig())
        check("re-confirmation does not double-queue (ledger dedupes)",
              len(mon._pending_orders) == 1)

    with tempfile.TemporaryDirectory() as tmp:
        mon = fresh_monitor(0.05, 3, tmp)
        mon._emit(sig("YES"))
        mon._emit(sig("NO"))
        check("YES and NO edges confirm independently",
              len(mon._candidates) == 2 and not mon._pending_orders)

        # A gap longer than the window restarts the clock: the edge closed and
        # reopened, which is a new event, not persistence.
        key = ("STALE", "KXBTC15M-TEST-15", ("YES",))
        mon._candidates[key]["last"] -= 10.0
        mon._candidates[key]["first"] -= 10.0
        mon._candidates[key]["passes"] = 99
        mon._emit(sig("YES"))
        check("a lapsed candidate restarts instead of confirming stale history",
              mon._candidates[key]["passes"] == 1 and not mon._pending_orders)

    with tempfile.TemporaryDirectory() as tmp:
        mon = fresh_monitor(0.0, 1, tmp)
        mon._emit(sig())
        check("confirm-seconds 0 restores immediate recording",
              len(mon._pending_orders) == 1)


# --------------------------------------------------------------------------- #
# Autopilot: promotion and quality gates
# --------------------------------------------------------------------------- #


async def test_autopilot() -> None:
    print("\n--- autopilot: promotion and quality gates ---")

    import tempfile
    import time as _time
    from types import SimpleNamespace

    from kalshi import KalshiClient
    from kalshi_execution import KalshiTrader, RiskLimits
    from kalshi_main import evaluate_gates
    from kalshi_monitor import Monitor

    # -- go_live: the paper phase must not leak into the live risk state ----- #
    class FakeClient(KalshiClient):
        def __init__(self, balance_cents=7500, fail=False):  # noqa: super-init-not-called
            self._cents, self._fail = balance_cents, fail

        def authenticate(self):
            if self._fail:
                from kalshi import KalshiAuthError

                raise KalshiAuthError("bad key")

        async def balance(self):
            return {"balance": self._cents}

    trader = KalshiTrader(FakeClient(), RiskLimits(), dry_run=True)
    await trader.arm()
    trader.realized = -10.0
    trader.trades = 7
    trader.consecutive_losses = 3
    trader.halted = True
    trader.halt_reason = "paper losses"
    trader.side_mapping_verified = True
    ok = await trader.go_live()
    check("go_live arms against the real balance", ok and not trader.dry_run
          and abs(trader.starting_balance - 75.0) < 1e-9)
    check("paper-phase PnL, halts and counters do not leak into live",
          trader.realized == 0.0 and trader.trades == 0
          and trader.consecutive_losses == 0 and not trader.halted)
    check("the side mapping must be re-proven with a real order",
          not trader.side_mapping_verified)
    check("go_live on an already-live trader is a no-op", await trader.go_live())

    failing = KalshiTrader(FakeClient(fail=True), RiskLimits(), dry_run=True)
    await failing.arm()
    check("a failed promotion falls back to paper, not half-armed",
          not await failing.go_live() and failing.dry_run)

    # -- the graded live halt: the stake cap must not block its own safety ---- #
    # Session L_081526_012650 went live on a $9.80 balance; the 8% per-trade
    # cap ($0.78) rejected the $0.99 side-mapping probe, halting the session
    # before the first real order. The probe is a bounded ~$1 safety cost and
    # is exempt from that one check - and only that one.
    small = KalshiTrader(FakeClient(), RiskLimits(), dry_run=True)
    await small.arm()
    small.starting_balance = 9.80
    blocked = await small.place("KXBTC15M-T", "NO", 0.99, 1)
    check("a normal 1-lot above the cap is still blocked",
          not blocked.ok and "per-trade cap" in (blocked.error or ""))
    probe = await small.place("KXBTC15M-T", "NO", 0.99, 1, verification=True)
    check("the verification probe is exempt from the per-trade cap", probe.ok)
    sized = await small.place("KXBTC15M-T", "NO", 0.99, 2, verification=True)
    check("the exemption is strictly 1 contract - size cannot ride on it",
          not sized.ok)
    small.halted, small.halt_reason = True, "test"
    halted_probe = await small.place("KXBTC15M-T", "NO", 0.99, 1, verification=True)
    check("a halt still stops the verification probe", not halted_probe.ok)

    # -- evaluate_gates ------------------------------------------------------ #
    def ready_monitor(tmp: str) -> Monitor:
        import argparse as _ap

        mon = Monitor(_ap.Namespace(
            confirm_seconds=3.0, confirm_passes=3, series="KXBTC15M",
            min_edge=0.02, size=20.0, no_basis=True, binance=False,
            vol_ratio_max=1.5, allow_unvalidated_vol=False, live=False,
            min_seconds_left=20.0, spike_bps=12.0, book_interval=1.0,
            discovery_interval=10.0, cooldown=5.0, heartbeat=15.0,
            max_stake_pct=8.0, max_exposure_pct=25.0, daily_loss_pct=20.0,
            max_trades=40, min_profit=0.01, anchor_age=20.0, stale_min_move=8.0,
            endgame_window=120.0, endgame_z=3.0, no_cross=False,
            no_stale=False, no_endgame=False, log_dir=tmp, no_log=True,
            env_file=".env", verbose=False,
        ))
        mon._stream = SimpleNamespace(connected=True)
        mon._basis = None
        mon._market = SimpleNamespace(ticker="KXBTC15M-T", strike_known=True)
        now = _time.monotonic()
        for i in range(35):  # past the 31-bar vol_is_measured threshold
            mon._buffer._bars.append((now - 35 + i, 63_000.0))
        mon._observations = 500
        mon._vol_ratios = [1.1] * 60
        return mon

    with tempfile.TemporaryDirectory() as tmp:
        mon = ready_monitor(tmp)
        passed, lines = evaluate_gates(mon, 1.5)
        check("a healthy warm-up passes every gate", passed, lines[-1].strip())

        mon = ready_monitor(tmp)
        mon._vol_ratios = [2.4] * 60
        passed, lines = evaluate_gates(mon, 1.5)
        check("a sigma disagreement fails promotion",
              not passed and any("sigma agrees" in l and "FAIL" in l for l in lines))

        mon = ready_monitor(tmp)
        mon._vol_ratios = [1.1] * 5  # too few to validate
        passed, _ = evaluate_gates(mon, 1.5)
        check("an unvalidatable sigma fails closed, not open", not passed)

        mon = ready_monitor(tmp)
        mon._market = SimpleNamespace(ticker="X", strike_known=False)
        passed, _ = evaluate_gates(mon, 1.5)
        check("a market without a published strike fails promotion", not passed)

        mon = ready_monitor(tmp)
        mon._stream = SimpleNamespace(connected=False)
        passed, _ = evaluate_gates(mon, 1.5)
        check("a dead spot feed fails promotion", not passed)

    # -- reset_for_live ------------------------------------------------------ #
    from strategies import Leg, PaperLedger, Signal
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        mon = ready_monitor(tmp)
        mon.ledger = PaperLedger(Path(tmp) / "p.jsonl")
        sig = Signal(strategy="STALE", ticker="T", legs=[Leg("YES", 0.4, 20.0)],
                     fair_yes=0.5, expected_net=1.0, max_loss=8.0, spot=1.0,
                     strike=1.0, seconds_left=100.0, sigma_used=1e-4)
        mon.ledger.record(sig)
        mon._pending_orders.append(sig)
        mon._candidates[("STALE", "T", ("YES",))] = {"first": 0, "last": 0, "passes": 1}
        mon.reset_for_live()
        check("promotion clears stale paper-phase queues",
              not mon._pending_orders and not mon._candidates)
        check("a market that signalled on paper can trade live",
              mon.ledger.record(sig))


# --------------------------------------------------------------------------- #
# Kalshi execution
# --------------------------------------------------------------------------- #


async def test_kalshi_execution() -> None:
    print("\n--- kalshi execution ($75 account) ---")

    from kalshi_execution import KalshiTrader, RiskLimits

    # The side mapping. This is the error that would reverse every position.
    bid = KalshiTrader.to_api_side("YES", 0.45)
    ask = KalshiTrader.to_api_side("NO", 0.30)
    check("buying YES bids the YES book at its own price", bid == ("bid", 0.45), f"{bid}")
    check(
        "buying NO sells the YES book at 1 - price",
        ask == ("ask", 0.70),
        f"buy NO @0.30 -> {ask}",
    )
    roundtrip = all(
        abs(KalshiTrader.to_api_side("NO", q)[1] - (1 - q)) < 1e-9
        for q in (0.01, 0.25, 0.5, 0.75, 0.99)
    )
    check("the NO inversion holds across the price range", roundtrip)

    t = KalshiTrader(None, RiskLimits(), dry_run=True)
    t.starting_balance = 75.0

    check("per-trade stake is capped at 8% of balance", abs(t.max_stake() - 6.0) < 1e-9)
    check(
        "sizing respects the stake cap at every price",
        all(t.size_for(p) * p <= t.max_stake() + 1e-9 for p in (0.05, 0.3, 0.5, 0.9, 0.99)),
    )
    check("an impossible price yields no size", t.size_for(0.0) == 0 and t.size_for(1.0) == 0)
    check(
        "sizing leaves room for the fee",
        t.size_for(0.50) * (0.50 + 0.0175) <= t.max_stake() + 1e-9,
        f"{t.size_for(0.50)} contracts at 0.50",
    )

    # Exposure cap.
    t.open_stake = 18.0
    check(
        "the exposure cap shrinks size as positions accumulate",
        t.size_for(0.50) * 0.5 <= 75 * 0.25 - 18.0 + 1e-9,
        f"{t.size_for(0.50)} contracts with $18 already open",
    )
    t.open_stake = 75 * 0.25
    check("at full exposure no new size is allowed", t.size_for(0.50) == 0)
    t.open_stake = 0.0

    # Halts.
    t.realized = -15.0
    check("the daily stop halts at -20%", t.check_halt() is not None and t.halted)
    t2 = KalshiTrader(None, RiskLimits(), dry_run=True)
    t2.starting_balance = 75.0
    t2.consecutive_losses = 3
    check("three losses in a row halts", t2.check_halt() is not None)
    t3 = KalshiTrader(None, RiskLimits(max_trades=2), dry_run=True)
    t3.starting_balance = 75.0
    t3.trades = 2
    check("the session trade cap halts", t3.check_halt() is not None)

    # A halted trader must refuse to send.
    res = await t.place("TEST", "YES", 0.5, 5)
    check("a halted trader rejects orders", not res.ok and "halted" in (res.error or ""))

    # Oversized orders are blocked before they can be sent.
    t4 = KalshiTrader(None, RiskLimits(), dry_run=True)
    t4.starting_balance = 75.0
    big = await t4.place("TEST", "YES", 0.5, 100)  # $50 stake vs a $6 cap
    check(
        "an oversized stake is blocked",
        not big.ok and "cap" in (big.error or ""),
        big.error or "",
    )

    # Settlement accounting, both directions.
    t5 = KalshiTrader(None, RiskLimits(), dry_run=True)
    t5.starting_balance = 75.0
    win = await t5.place("MKT-A", "YES", 0.40, 10)
    check("a dry-run order is booked", win.ok and t5.trades == 1)
    pnl = t5.settle("MKT-A", "yes")
    check(
        "a winning YES settles to payout minus stake and fee",
        abs(pnl - (10 - 4.0 - 0.17)) < 0.02,
        f"{pnl:+.2f} on 10 @ 0.40",
    )
    t6 = KalshiTrader(None, RiskLimits(), dry_run=True)
    t6.starting_balance = 75.0
    await t6.place("MKT-B", "NO", 0.40, 10)
    loss = t6.settle("MKT-B", "yes")  # bought NO, YES won
    check(
        "a losing trade costs the stake plus fee",
        loss < -4.0,
        f"{loss:+.2f}",
    )
    check("a loss increments the streak", t6.consecutive_losses == 1)

    check(
        "dry run never marks the side mapping as verified against the venue",
        (await KalshiTrader(None, RiskLimits(), dry_run=True).verify_side_mapping("X")) is True,
        "dry run short-circuits; live requires a real 1-contract probe",
    )


# --------------------------------------------------------------------------- #


async def main() -> None:
    logging.basicConfig(level=logging.CRITICAL)
    print("=" * 68)
    # Venue-agnostic: always run.
    await test_retry_backoff()
    await test_fast_json()
    await test_latency_profiler()
    await test_reconnect_resilience()
    await test_risk_breakers()
    await test_kalshi()
    await test_strategies()
    await test_setup_and_confirmation()
    await test_autopilot()
    await test_kalshi_execution()
    await test_polymarket_us()
    await test_run_log()

    if POLYMARKET_SDK:
        await test_signals()
        await test_dry_run_execution()
        await test_position_gate()
        await test_safety_gates()
        await test_balance_and_sizing()
        await test_order_caching()
        await test_eval_loop_benchmark()
    else:
        print(
            "\n(skipping the Polymarket CLOB suites - py-clob-client is not installed;"
            "\n install it with: pip install -r requirements-polymarket.txt)"
        )
    print("=" * 68)
    total = len(PASSED) + len(FAILED)
    if FAILED:
        print(f"{len(FAILED)}/{total} FAILED:")
        for name in FAILED:
            print(f"  - {name}")
    else:
        print(f"ALL {total} CHECKS PASS")
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    asyncio.run(main())
