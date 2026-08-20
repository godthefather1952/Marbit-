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
import json
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
    # The whole of ENDGAME's apparent edge was an UNDER-estimate of volatility
    # making it overconfident. Taking the larger of the two sigmas removes it.
    under = scan_endgame(em, ebook, espot, 20.0, esigma * 0.5, min_z=0.5, min_edge=0.0)
    check(
        "a too-low sigma can no longer manufacture certainty",
        under is None,
        f"implied {eimplied * 1e4:.2f} vs measured {esigma * 0.5 * 1e4:.2f} bps/s",
    )
    check(
        "ENDGAME sits out when the quote cannot supply a volatility to check against",
        scan_endgame(em, book(0.50, 0.51), espot, 20.0, esigma,
                     min_z=0.5, min_edge=0.0) is None,
    )

    # --- the trade that actually lost $8.79 -------------------------------- #
    # L_081726_031221: NO at 0.975 on a "2.8 sigma" reading, with our sigma
    # 1.45x BELOW the market's - inside the 1.50x gate. On the market's sigma
    # it was 1.95 sigma. BTC moved $38 in 37s and it settled the other way.
    loser = market("63337.39", close=closing_in(106.0))
    loser_book = book(0.0250, 0.0260)  # NO ask 0.975
    loser_spot = 63_300.83
    ours, mkt = 0.25e-4, implied_sigma(loser, loser_book, loser_spot)
    check(
        "the market's own sigma was well above ours on the losing setup",
        mkt is not None and 1.3 < mkt / ours < 1.6,
        f"ours {ours * 1e4:.2f} vs market {mkt * 1e4:.2f} bps/s = {mkt / ours:.2f}x",
    )
    check(
        "that trade is now refused outright",
        scan_endgame(loser, loser_book, loser_spot, 20.0, ours,
                     max_seconds_left=120.0, min_z=2.5) is None,
    )
    check(
        "and it only fired before because our sigma was the smaller one",
        scan_endgame(loser, loser_book, loser_spot, 20.0, ours,
                     max_seconds_left=120.0, min_z=2.5,
                     require_implied=False) is None,
        "conservative sigma applies even without require_implied",
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

    # -- the 93-contract trade from L_081726_202145 -------------------------- #
    # Spot sat almost exactly AT the strike (ln(S/K) = 2.6e-5) while the market
    # quoted 0.9935. Inverting that gave sigma = 0.03 bps/s against our measured
    # 0.32 - a 9.53x disagreement - and STALE turned it into a claimed $19.83
    # edge on $0.15 of risk, bought 93 contracts, and lost.
    #
    # The divergence is structural, not noise: near expiry the market knows how
    # much of the settlement TWAP is already realized and we only know spot, so
    # the quote can be confident while ln(S/K) is ~0.
    near_expiry = market("64285.01", close=closing_in(37.0))
    tight = book(0.993, 0.994)
    at_strike = 64_286.67
    check(
        "an implausibly low implied sigma is rejected outright",
        implied_sigma(near_expiry, tight, at_strike) is None,
        "0.03 bps/s is ~1.7% annualized; crypto does not trade there",
    )
    check(
        "so STALE cannot trade on it",
        scan_stale(near_expiry, tight, 64_329.0, 0.998, at_strike, 20.0,
                   0.01, 0.32e-4, max_vol_ratio=1.5) is None,
    )

    # Even a plausible implied sigma is refused when it disagrees with ours.
    # Taking sigma from the quote only protects us while the quote is sane.
    plausible = market("63777.35", close=closing_in(600.0))
    q = book(0.10, 0.11)
    imp = implied_sigma(plausible, q, 63_400.0)
    check("this quote does invert to a usable sigma", imp is not None,
          f"{imp * 1e4:.2f} bps/s" if imp else "n/a")
    if imp:
        far = imp * 12.0  # our measurement wildly different
        check(
            "STALE sits out a large sigma disagreement rather than trusting the quote",
            scan_stale(plausible, q, 63_800.0, 0.30, 63_400.0, 20.0,
                       0.01, far, max_vol_ratio=1.5) is None,
            f"{far / imp:.1f}x apart - one of them is broken and we cannot tell which",
        )
        check(
            "max_vol_ratio 0 restores the old behaviour",
            scan_stale(plausible, q, 63_800.0, 0.30, 63_400.0, 20.0,
                       0.01, far, max_vol_ratio=0.0) is not None,
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
                   require_implied=False, max_edge=0.0) is not None,
        "max_edge=0 here so the test isolates require_implied",
    )

    # -- the edge sanity ceiling --------------------------------------------- #
    # A 95 bps move against a 0.8 bps/s sigma drives fair value to the rail and
    # claims ~50c of edge per contract. Real books do not offer that.
    huge = scan_stale(live_now, atm, 63_000.0, 0.50, 63_600.0, 20.0, 0.01,
                      0.8e-4, require_implied=False, max_edge=0.0)
    check("without the cap, a runaway model claims an enormous edge",
          huge is not None and huge.expected_net / 20.0 > 0.35,
          f"{huge.expected_net / 20.0:+.3f}/contract" if huge else "no signal")
    check(
        "with the cap, that signal is refused as a model error",
        scan_stale(live_now, atm, 63_000.0, 0.50, 63_600.0, 20.0, 0.01,
                   0.8e-4, require_implied=False, max_edge=0.35) is None,
        "every edge this large the project has produced was a bug",
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
            max_trades=40, min_profit=0.01, anchor_age=20.0, stale_min_move=8.0, aggressive=False,
            assets=None, eval_interval=0.2, take_profit=1.5,
            stop_loss=0.0, min_exit_seconds=45.0, no_fair_exit=False,
            max_edge=0.35, max_slippage=0.03, min_fill_edge=0.005,
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

    # -- a confirmation must be NEW market data, not another loop pass ------ #
    # The evaluator runs every 0.1s while the book refreshes every 0.4s, so
    # counting passes counted the same snapshot ~4 times. A live log read
    # "confirmed over 1.0s / 11 passes" on roughly two distinct books.
    def versioned(v):
        s = sig()
        s.book_version = v
        return s

    with tempfile.TemporaryDirectory() as tmp:
        mon = fresh_monitor(0.05, 3, tmp)
        for _ in range(30):
            mon._emit(versioned(1021))       # the SAME book, thirty times
            _time.sleep(0.003)
        check("an unchanged book cannot confirm itself, however many passes",
              not mon._pending_orders, "book 1021 x30")
        mon._emit(versioned(1024))
        mon._emit(versioned(1027))
        check("three DISTINCT books do confirm it",
              len(mon._pending_orders) == 1, "1021 -> 1024 -> 1027")

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

    # -- two strategies must not take opposite sides of one market ---------- #
    # L_081726_031221 bought ENDGAME NO @ 0.975 and, 32s later, STALE YES @
    # 0.820 on the SAME contract. Settlement pays exactly one, so the pair is a
    # guaranteed loss of both fees plus the gap - and it meant two of our own
    # strategies flatly disagreed while we funded both opinions.
    def named(strategy: str, side: str, price: float) -> Signal:
        return Signal(
            strategy=strategy, ticker="KXBTC15M-SAME", legs=[Leg(side, price, 20.0)],
            fair_yes=0.5, expected_net=0.4, max_loss=8.0, spot=63_300.0,
            strike=63_337.0, seconds_left=100.0, sigma_used=2.5e-5,
        )

    with tempfile.TemporaryDirectory() as tmp:
        mon = fresh_monitor(0.0, 1, tmp)
        mon._emit(named("ENDGAME", "NO", 0.975))
        check("the first side is taken", len(mon._pending_orders) == 1)
        mon._emit(named("STALE", "YES", 0.820))
        check("a second strategy cannot buy the opposing side of the same market",
              len(mon._pending_orders) == 1,
              "settlement pays one of them; holding both is a guaranteed loss")
        mon._emit(named("STALE", "NO", 0.970))
        check("the SAME side from another strategy is still allowed",
              len(mon._pending_orders) == 2)

        rows = [json.loads(l) for l in
                (mon.ledger.path).read_text().splitlines() if l.strip()]
        blocked = [r for r in rows if r.get("kind") == "execution"
                   and r.get("outcome") == "skipped"]
        check("the block is recorded in the ledger, not silent",
              len(blocked) == 1 and "opposing" in blocked[0]["detail"],
              blocked[0]["detail"] if blocked else "nothing recorded")

    with tempfile.TemporaryDirectory() as tmp:
        # CROSS is exempt: taking both sides IS its thesis, and it only fires
        # when the pair costs less than the dollar it pays.
        mon = fresh_monitor(0.0, 1, tmp)
        both = Signal(
            strategy="CROSS", ticker="KXBTC15M-SAME",
            legs=[Leg("YES", 0.48, 20.0), Leg("NO", 0.49, 20.0)],
            fair_yes=0.5, expected_net=0.4, max_loss=0.0, spot=0.0,
            strike=63_337.0, seconds_left=100.0, sigma_used=0.0,
        )
        mon._emit(both)
        check("CROSS may still hold both sides at once", len(mon._pending_orders) == 1)
        mon._emit(both)
        check("and a repeat CROSS is deduped, not conflict-blocked",
              len(mon._pending_orders) == 1)

    # -- marketable limits: cross far enough to actually fill ---------------- #
    # L_081726_224233 filled ZERO strategy orders. It bid exactly the ask it had
    # seen (`sent as bid 0.8400`, `filled 0.0 of 3`) while the probe, which
    # crosses hard at 0.999, filled every single time. We are takers by
    # construction, so the limit should be the highest price that still leaves
    # the edge worth having.
    from types import SimpleNamespace as _NS

    from kalshi import fee_per_contract

    with tempfile.TemporaryDirectory() as tmp:
        mon = fresh_monitor(0.0, 1, tmp)
        mon._args.min_edge = 0.01
        mon._args.min_fill_edge = 0.005
        mon._args.max_slippage = 0.03
        mon.instruments[0].market = _NS(ticker="T", price_ranges=())
        leg = Leg("YES", 0.84, 20.0)
        s = named("STALE", "YES", 0.84)
        s.ticker = "T"
        limit = mon._marketable_limit(s, leg, 0.8683)   # the real BTC signal
        check("the limit crosses above the quoted ask",
              limit is not None and limit > 0.84, f"0.840 -> {limit}")
        check("but never past the price that still leaves min-fill-edge",
              limit <= 0.8683 - fee_per_contract(0.84) - 0.005 + 1e-9,
              f"ceiling {0.8683 - fee_per_contract(0.84) - 0.005:.4f}")
        check("the fill still carries real edge after crossing",
              0.8683 - limit - fee_per_contract(limit) >= 0.005 - 1e-9,
              f"{0.8683 - limit - fee_per_contract(limit):+.4f}/contract left")

        mon._args.max_slippage = 0.50   # absurd allowance
        wide = mon._marketable_limit(s, leg, 0.8683)
        check("the edge, not the slippage allowance, is what binds",
              abs(wide - limit) < 1e-9, f"{wide} vs {limit}")

        thin = mon._marketable_limit(s, Leg("YES", 0.865, 20.0), 0.8683)
        check("a signal whose ask already eats the edge is refused",
              thin is None, "no price leaves min-fill-edge")



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

    # -- fills decide, not order acceptance ----------------------------------- #
    # Session L_081526_052800: on a book at 0.009/0.010 the NO ask was 0.991,
    # the 0.99 probe limit could not cross, the IOC cancelled with zero fills -
    # and came back WITH an order_id. That was booked as a phantom position and
    # reported as "side mapping WRONG", halting a healthy session.
    class VenueTrader(KalshiTrader):
        def __init__(self, order_response, positions_response=None):
            class _Client:
                async def positions(self_inner, **kw):
                    return positions_response

            super().__init__(_Client(), RiskLimits(), dry_run=False)
            self.starting_balance = 20.58
            self._order_response = order_response

        async def _post(self, path, body):
            self._last_body = body
            return self._order_response

    killed = VenueTrader({"order": {"order_id": "abc", "status": "canceled"}})
    res = await killed.place("T", "NO", 0.999, 1, tif="immediate_or_cancel")
    check("a cancelled zero-fill is not a trade, despite its order_id",
          not res.ok and killed.trades == 0 and "no fill" in (res.error or ""))

    partial = VenueTrader({"order": {"order_id": "abc", "status": "canceled",
                                     "taker_fill_count": 1}})
    res = await partial.place("T", "NO", 0.999, 1, tif="immediate_or_cancel")
    check("a cancelled order WITH taker fills is a trade", res.ok)

    filled = VenueTrader({"order": {"order_id": "abc", "status": "executed"}})
    res = await filled.place("T", "NO", 0.999, 1)
    check("an executed order books normally", res.ok and filled.trades == 1)

    # -- entries are IOC, not FOK --------------------------------------------- #
    # L_081726_161637: 4 of 11 live entry attempts died with
    # `fill_or_kill_insufficient_resting_volume` - a 36% miss on otherwise good
    # signals, because FOK needs the WHOLE size resting at the price. IOC takes
    # the 2 contracts that exist rather than refusing 3.
    tif_check = VenueTrader({"order": {"order_id": "a", "fill_count": "2",
                                       "remaining_count": "1"}})
    res = await tif_check.place("T", "YES", 0.40, 3)  # $1.20, inside the cap
    check("entries default to immediate-or-cancel",
          tif_check._last_body["time_in_force"] == "immediate_or_cancel",
          tif_check._last_body["time_in_force"])
    check("a partial fill is taken rather than refused",
          res.ok and res.count == 2, f"asked 3, got {res.count}")


    # -- the real CreateOrder V2 shape -------------------------------------- #
    # V2 returns NO status field at all - only fill_count / remaining_count.
    # Session L_081626_233108 fell through to "has an order_id, so it filled",
    # which booked six phantom probe positions across two sessions and left the
    # side-mapping check hunting a position that never existed.
    v2_nofill = VenueTrader({"order": {
        "order_id": "abc", "fill_count": "0", "remaining_count": "1"}})
    res = await v2_nofill.place("T", "NO", 0.999, 1)
    check("a V2 zero-fill is not a trade, despite carrying an order_id",
          not res.ok and v2_nofill.trades == 0 and "no fill" in (res.error or ""),
          res.error or "")

    v2_fill = VenueTrader({"order": {
        "order_id": "abc", "fill_count": "1", "remaining_count": "0",
        "average_fill_price": "0.0100", "average_fee_paid": "0.0007"}})
    res = await v2_fill.place("T", "NO", 0.999, 1)
    check("a V2 fill books, priced at what the venue actually charged",
          res.ok and abs(res.price - 0.99) < 1e-9,
          f"limit 0.999 -> filled {res.price:.4f}")
    check("the venue's own fee is captured, not just modelled",
          abs((res.avg_fee_paid or 0) - 0.0007) < 1e-12)

    # A partial fill must book what filled, not what was asked for: settlement
    # cannot credit contracts we never owned.
    from kalshi import trading_fee as _fee

    v2_part = VenueTrader({"order": {
        "order_id": "abc", "fill_count": "3", "remaining_count": "7",
        "average_fill_price": "0.1000"}})
    res = await v2_part.place("T", "YES", 0.10, 10)  # $1.00, inside the cap
    check("a partial fill books only the filled contracts",
          res.ok and res.count == 3 and res.remaining == 7.0,
          f"asked 10, filled {res.count}")
    v2_part.settle("T", "yes")
    check("settlement pays the filled size, not the requested size",
          abs(v2_part.realized - (3 * 1.0 - 3 * 0.10 - _fee(0.10, 3))) < 1e-6,
          f"${v2_part.realized:+.4f}")

    # -- paper must never be reported as money ------------------------------- #
    # Session L_081626_233108 graded five winning signals worth +$3.85 while the
    # account moved ten cents: two were simulated during warm-up, three were
    # skipped when the side-mapping probe came back inconclusive, and nothing
    # downstream could tell any of that apart from a real fill.
    import io as _io
    from contextlib import redirect_stdout

    from kalshi_score import _execution_index, score as score_trades

    def _sig(strategy, ticker, asset, price, size):
        return {"strategy": strategy, "ticker": ticker, "note": f"[{asset}] x",
                "fair_yes": 0.99, "expected_net": 0.3,
                "legs": [{"side": "YES", "price": price, "size": size}]}

    def _exe(strategy, ticker, outcome):
        return {"kind": "execution", "strategy": strategy, "ticker": ticker,
                "outcome": outcome}

    signals = [_sig("STALE", "M1", "BTC", 0.89, 9),
               _sig("ENDGAME", "M2", "ETH", 0.99, 8),
               _sig("ENDGAME", "M3", "BTC", 0.97, 20)]
    settled_all = {t: {"result": "yes"} for t in ("M1", "M2", "M3")}

    none_real = [_exe("STALE", "M1", "simulated"), _exe("ENDGAME", "M2", "simulated"),
                 _exe("ENDGAME", "M3", "skipped")]
    buf = _io.StringIO()
    with redirect_stdout(buf):
        score_trades(signals, settled_all, none_real)
    out = buf.getvalue()
    check("a session with no fills says so, in the headline",
          "REAL MONEY : nothing filled" in out and "balance did not move" in out)
    check("hypothetical winnings are labelled hypothetical",
          "HYPOTHETICAL" in out and "PAPER ONLY - no money moved" in out)
    check("no strategy block claims real money when none filled",
          "[REAL MONEY]" not in out)

    one_real = [_exe("STALE", "M1", "filled"), _exe("ENDGAME", "M2", "simulated"),
                _exe("ENDGAME", "M3", "skipped")]
    buf = _io.StringIO()
    with redirect_stdout(buf):
        score_trades(signals, settled_all, one_real)
    out = buf.getvalue()
    check("a real fill is reported separately from the paper ones",
          "REAL MONEY : 1 filled order(s)" in out and "HYPOTHETICAL: 2 signal(s)" in out)
    check("only the filled strategy is tagged REAL MONEY", out.count("[REAL MONEY]") == 1)

    # -- the scorecard must grade what FILLED, not what was proposed --------- #
    # Pooled scoring reported +$52.72 of "REAL MONEY" on $184.42 staked, for a
    # session whose log said `realized +1.75` on a $23 account. Two causes:
    # the signal names a nominal --size of 20 while 3 contracts actually
    # filled, and a position closed early does not settle at all.
    sized = [_sig("STALE", "M1", "BTC", 0.80, 20.0)]
    got_3 = [_exe("STALE", "M1", "filled")]
    got_3[0].update(count=3, price=0.80)
    buf = _io.StringIO()
    with redirect_stdout(buf):
        score_trades(sized, {"M1": {"result": "no"}}, got_3)
    out = buf.getvalue()
    check("stake is the 3 contracts filled, not the 20 proposed",
          "$2.40 staked" in out, "20 x 0.80 would have read $16.00")

    # An early exit inverts this one: held to expiry it is a total loss, but it
    # was sold at 0.919 and made money.
    exited = [_sig("STALE", "M2", "ETH", 0.89, 20.0)]
    exit_rows = [_exe("STALE", "M2", "filled"), _exe("STALE", "M2", "closed")]
    exit_rows[0].update(count=2, price=0.89)
    exit_rows[1].update(count=2, price=0.919)
    buf = _io.StringIO()
    with redirect_stdout(buf):
        score_trades(exited, {"M2": {"result": "yes"}}, exit_rows)
    out = buf.getvalue()
    net_line = next((l for l in out.splitlines() if "ACTUAL net" in l), "")
    check("a position sold before expiry is graded on its exit, not settlement",
          "$+" in net_line and "$-" not in net_line,
          f"bought NO at 0.89, sold at 0.919, market then settled YES ->{net_line}")

    idx = _execution_index(exit_rows)
    check("the exit price is carried alongside the entry",
          abs(idx[("STALE", "M2")]["exit_price"] - 0.919) < 1e-9
          and idx[("STALE", "M2")]["count"] == 2)

    # "filled" must win over an earlier "skipped" on the same market.
    idx = _execution_index([_exe("STALE", "M1", "skipped"), _exe("STALE", "M1", "filled")])
    check("a later fill outranks an earlier skip",
          idx[("STALE", "M1")]["outcome"] == "filled")
    idx = _execution_index([_exe("STALE", "M1", "filled"), _exe("STALE", "M1", "skipped")])
    check("and order does not matter", idx[("STALE", "M1")]["outcome"] == "filled")

    # The ledger must actually write these rows.
    with tempfile.TemporaryDirectory() as tmp:
        from pathlib import Path as _Path

        from strategies import Leg as _Leg, PaperLedger as _PL, Signal as _Sig

        led = _PL(_Path(tmp) / "p.jsonl")
        s = _Sig(strategy="STALE", ticker="T", legs=[_Leg("YES", 0.4, 5.0)],
                 fair_yes=0.5, expected_net=1.0, max_loss=2.0, spot=1.0,
                 strike=1.0, seconds_left=100.0, sigma_used=1e-4)
        led.record(s)
        led.record_execution(s, "filled", count=5, price=0.39)
        lines = [json.loads(l) for l in
                 _Path(led.path).read_text().splitlines() if l.strip()]
        check("the ledger carries a signal row and an execution row",
              len(lines) == 2 and lines[1]["kind"] == "execution"
              and lines[1]["outcome"] == "filled")
        check("the execution row records what actually filled",
              lines[1]["count"] == 5 and abs(lines[1]["price"] - 0.39) < 1e-9)

        led2 = _PL(_Path(tmp) / "q.jsonl")
        led2.record(s)
        led2.record_execution(s, "filled", count=5, price=0.42,
                              signal_price=0.40, elapsed_ms=850.0)
        row = [json.loads(l) for l in
               _Path(led2.path).read_text().splitlines() if l.strip()][1]
        check("and the quote that justified the trade, alongside the fill",
              abs(row["signal_price"] - 0.40) < 1e-9 and row["elapsed_ms"] == 850.0,
              "2c of slippage is not recoverable later by joining rows")

    # -- per-strategy execution quality -------------------------------------- #
    # "Did the model call it right" and "did we get a price worth having" are
    # different questions, and a strategy can pass the first while failing the
    # second on every single trade.
    slipped = [_sig("STALE", "M9", "BTC", 0.40, 10.0)]
    slip_rows = [_exe("STALE", "M9", "filled")]
    slip_rows[0].update(count=10, price=0.46, signal_price=0.40,
                        elapsed_ms=1500.0, ts=1000.0)
    buf = _io.StringIO()
    with redirect_stdout(buf):
        score_trades(slipped, {"M9": {"result": "yes"}}, slip_rows)
    out = buf.getvalue()
    check("slippage is reported against the quote that produced the signal",
          "+6.00c/contract" in out, "paid 0.46 on a signal built at 0.40")
    check("time to fill is reported", "1.5s median" in out)
    check("fill rate is reported per strategy", "1/1 attempts" in out)
    check("predicted and realized edge are compared per contract",
          "per contract" in out and "predicted" in out)

    missed = [_sig("STALE", "M8", "BTC", 0.40, 10.0)]
    miss_rows = [_exe("STALE", "M8", "rejected"), _exe("STALE", "M8", "rejected"),
                 _exe("STALE", "M8", "filled")]
    miss_rows[2].update(count=10, price=0.40, signal_price=0.40, ts=1000.0)
    idx = _execution_index(miss_rows)
    check("attempts count every order sent, not just the one that filled",
          idx[("STALE", "M8")]["attempts"] == 3,
          "a strategy that fills one order in three is not a 100% fill rate")

    # -- the probe's three outcomes ------------------------------------------- #
    async def _noop(_s):  # verification sleeps 2s between order and read
        return None

    import asyncio as _aio
    real_sleep = _aio.sleep
    _aio.sleep = _noop
    try:
        nofill = VenueTrader({"order": {"order_id": "abc", "status": "canceled"}})
        check("an unfilled probe is inconclusive, not a reversed mapping",
              await nofill.verify_side_mapping("T") is None
              and not nofill.side_mapping_verified)
        check("the probe bids the top of the ladder so any book crosses",
              float(nofill._last_body["price"]) < 0.0011)  # api ask = 1 - 0.999

        empty = VenueTrader({"order": {"order_id": "a", "status": "executed"}},
                            {"market_positions": [{"ticker": "T", "position": 0}]})
        check("a filled claim with no visible position stays inconclusive",
              await empty.verify_side_mapping("T") is None)

        good = VenueTrader({"order": {"order_id": "a", "status": "executed"}},
                           {"market_positions": [{"ticker": "T", "position": -1}]})
        check("a short-YES position verifies the mapping",
              await good.verify_side_mapping("T") is True
              and good.side_mapping_verified)

        reversed_ = VenueTrader({"order": {"order_id": "a", "status": "executed"}},
                                {"market_positions": [{"ticker": "T", "position": 1}]})
        check("a LONG-YES position after a NO buy is the definitive halt",
              await reversed_.verify_side_mapping("T") is False)

        # -- the fills fallback ---------------------------------------------- #
        # Session L_081626_085656 read an empty position 2s after a fill three
        # separate times, then watched that same contract settle from the very
        # position it could not see. Fills are the trade itself, not a derived
        # snapshot, so they are asked when positions come back empty.
        class FillsTrader(VenueTrader):
            def __init__(self, fills_response):
                super().__init__({"order": {"order_id": "a", "status": "executed"}},
                                 {"market_positions": []})
                self._fills_response = fills_response
                trader_self = self

                class _Client:
                    async def positions(self, **kw):
                        return {"market_positions": []}

                    async def fills(self, **kw):
                        return trader_self._fills_response

                self._client = _Client()

        from_fills = FillsTrader({"fills": [
            {"ticker": "T", "count": 1, "side": "no", "action": "buy"}]})
        check("an invisible position falls back to the fills record",
              await from_fills.verify_side_mapping("T") is True)

        empty_fills = FillsTrader({"fills": []})
        check("no position and no fill stays inconclusive",
              await empty_fills.verify_side_mapping("T") is None)

        # -- the probe attempt cap ------------------------------------------- #
        capped = VenueTrader({"order": {"order_id": "a", "status": "canceled"}})
        capped.limits.max_verification_attempts = 2
        r1 = await capped.verify_side_mapping("T")
        r2 = await capped.verify_side_mapping("T")
        r3 = await capped.verify_side_mapping("T")
        check("probes stop after the attempt cap instead of bleeding dollars",
              r1 is None and r2 is None and r3 is False
              and capped.verification_attempts == 2)
    finally:
        _aio.sleep = real_sleep

    # -- the probe is a safety cost, not a strategy loss ---------------------- #
    # This is what actually ended session L_081626_085656: three 1-contract
    # probes settled at -0.01, -0.01 and -1.01, the consecutive-loss breaker
    # counted all three, and trading halted at 13:00 - so the two good signals
    # later that afternoon never executed.
    probes = VenueTrader({"order": {"order_id": "a", "status": "executed"}})
    for i in range(3):
        res = await probes.place(f"M{i}", "NO", 0.999, 1, verification=True)
        assert res.ok
    for i in range(3):
        probes.settle(f"M{i}", "yes")  # every probe loses
    check("settled probes never touch the consecutive-loss breaker",
          probes.consecutive_losses == 0 and not probes.halted,
          f"{probes.consecutive_losses} losses counted, halted={probes.halted}")
    check("probe losses still show up in realized PnL",
          probes.realized < -2.9, f"${probes.realized:.2f}")

    real_losses = VenueTrader({"order": {"order_id": "a", "status": "executed"}})
    for i in range(3):
        await real_losses.place(f"M{i}", "NO", 0.50, 1)
    for i in range(3):
        real_losses.settle(f"M{i}", "yes")
    check("three real losing trades still halt, as designed",
          real_losses.halted and real_losses.consecutive_losses == 3)

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
            max_trades=40, min_profit=0.01, anchor_age=20.0, stale_min_move=8.0, aggressive=False,
            assets=None, eval_interval=0.2, take_profit=1.5,
            stop_loss=0.0, min_exit_seconds=45.0, no_fair_exit=False,
            max_edge=0.35, max_slippage=0.03, min_fill_edge=0.005,
            endgame_window=120.0, endgame_z=3.0, no_cross=False,
            no_stale=False, no_endgame=False, log_dir=tmp, no_log=True,
            env_file=".env", verbose=False,
        ))
        inst = mon.instruments[0]
        inst.stream = SimpleNamespace(connected=True)
        inst.basis = None
        inst.market = SimpleNamespace(ticker="KXBTC15M-T", strike_known=True)
        now = _time.monotonic()
        for i in range(35):  # past the 31-bar vol_is_measured threshold
            inst.buffer._bars.append((now - 35 + i, 63_000.0))
        inst.observations = 500
        inst.vol_ratios = [1.1] * 60
        return mon

    with tempfile.TemporaryDirectory() as tmp:
        mon = ready_monitor(tmp)
        passed, lines = evaluate_gates(mon, 1.5)
        check("a healthy warm-up passes every gate", passed, lines[-1].strip())

        mon = ready_monitor(tmp)
        mon.instruments[0].vol_ratios = [2.4] * 60
        passed, lines = evaluate_gates(mon, 1.5)
        check("a sigma disagreement fails promotion",
              not passed and any("sigma agrees" in l and "FAIL" in l for l in lines))

        mon = ready_monitor(tmp)
        mon.instruments[0].vol_ratios = [1.1] * 5  # too few to validate
        passed, _ = evaluate_gates(mon, 1.5)
        check("an unvalidatable sigma fails closed, not open", not passed)

        mon = ready_monitor(tmp)
        mon.instruments[0].market = SimpleNamespace(ticker="X", strike_known=False)
        passed, _ = evaluate_gates(mon, 1.5)
        check("a market without a published strike fails promotion", not passed)

        mon = ready_monitor(tmp)
        mon.instruments[0].stream = SimpleNamespace(connected=False)
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
# Taking profit before settlement
# --------------------------------------------------------------------------- #


async def test_take_profit() -> None:
    print("\n--- exits: taking profit once the thesis has played out ---")

    from kalshi import KalshiBook, KalshiClient, fee_per_contract, trading_fee
    from kalshi_execution import KalshiTrader, OrderResult, RiskLimits

    def book(yes_bid: float, yes_ask: float) -> KalshiBook:
        return KalshiBook.from_payload({"orderbook_fp": {
            "yes_dollars": [[f"{yes_bid:.4f}", "500"]],
            "no_dollars": [[f"{1.0 - yes_ask:.4f}", "500"]],
        }})

    class FakeClient(KalshiClient):
        def __init__(self):  # noqa: super-init-not-called
            pass

        async def balance(self):
            return {"balance": 7500}

    def trader(**kw) -> KalshiTrader:
        t = KalshiTrader(FakeClient(), RiskLimits(**kw), dry_run=True)
        t.starting_balance = 75.0
        return t

    def held(outcome: str, price: float, count: int = 20) -> OrderResult:
        return OrderResult(ok=True, dry_run=True, ticker="T", outcome=outcome,
                           api_side="bid", price=price, api_price=price,
                           count=count, strategy="STALE")

    # -- the mark must be a bid, never a mid or an ask ----------------------- #
    t = trader()
    b = book(0.60, 0.66)  # yes bid 0.60, yes ask 0.66 -> no bid 0.34
    check("a YES position marks at the YES bid - what someone will actually pay",
          abs(t.mark(held("YES", 0.30), b) - 0.60) < 1e-9)
    check("a NO position marks at the NO bid",
          abs(t.mark(held("NO", 0.30), b) - 0.34) < 1e-9,
          "1 - yes_ask, not 1 - yes_bid")

    # -- the doubling test is on the NET multiple ---------------------------- #
    # Bought at 0.32, quoted at 0.64: gross is exactly 2x, but both sides pay
    # Kalshi's fee, so the real multiple is under 2. Testing the raw ratio
    # would exit a "double" that is really 1.9x.
    t = trader(take_profit_multiple=2.0)
    pos = held("YES", 0.32)
    gross_double = t.exit_reason(pos, book(0.64, 0.66), 300.0)
    net_ratio = (0.64 - fee_per_contract(0.64)) / (0.32 + fee_per_contract(0.32))
    check("a gross double that is not a NET double does not trigger",
          gross_double is None and net_ratio < 2.0, f"net {net_ratio:.3f}x")
    decision = t.exit_reason(pos, book(0.72, 0.74), 300.0)
    check("a genuine net double does trigger",
          decision is not None and decision[0] == "take-profit",
          f"sell at {decision[1]:.2f}" if decision else "no exit")

    check("a position below target is held",
          t.exit_reason(pos, book(0.40, 0.42), 300.0) is None)
    # The probe IS managed. Its PnL stays out of the loss breaker, but it is a
    # real contract bought with real money - and excluding it threw away the
    # best trade in the record (see the 0.12 -> 0.57 replay below).
    probe = OrderResult(ok=True, dry_run=True, ticker="T", outcome="NO",
                        api_side="ask", price=0.12, api_price=0.88, count=1,
                        verification=True)
    check("the side-mapping probe is managed like any other position",
          t.exit_reason(probe, book(0.42, 0.43), 300.0) is not None,
          "it is real money, even if its PnL is excluded from the breaker")
    check("no exit is attempted in the final seconds, where the book empties",
          t.exit_reason(pos, book(0.90, 0.92), 20.0) is None,
          "min_seconds_to_exit")
    check("an empty book cannot be marked, so nothing is sold into it",
          t.exit_reason(pos, KalshiBook.from_payload(
              {"orderbook_fp": {"yes_dollars": [], "no_dollars": []}}), 300.0) is None)

    # -- closing actually books the gain and flattens ------------------------ #
    t = trader(take_profit_multiple=2.0)
    pos = held("YES", 0.32, count=20)
    t._orders.append(pos)
    t.open_stake = pos.stake
    res = await t.close_position(pos, 0.72, "take-profit")
    expected = 20 * (0.72 - 0.32) - trading_fee(0.72, 20)
    check("the exit books the realized gain", res.ok
          and abs(t.realized - expected) < 1e-6, f"${t.realized:+.2f}")
    check("the position is flat afterwards", pos.closed
          and not t.open_positions("T"))
    # Settlement must not ALSO pay out a position we already sold.
    t.settle("T", "no")  # would have been a total loss if still held
    check("settlement skips a position that was already exited",
          abs(t.realized - expected) < 1e-6, f"${t.realized:+.2f}")

    # -- the side inversion on the way out ----------------------------------- #
    t = trader()
    yes_pos = held("YES", 0.30)
    t._orders.append(yes_pos)
    res = await t.close_position(yes_pos, 0.70)
    check("closing a YES long SELLS yes (side=ask)", res.api_side == "ask",
          f"sent as {res.api_side} {res.api_price:.4f}")
    t = trader()
    no_pos = held("NO", 0.30)
    t._orders.append(no_pos)
    res = await t.close_position(no_pos, 0.70)
    check("closing a NO long BUYS yes back (side=bid)", res.api_side == "bid",
          "a NO long is a short YES on this venue")

    # -- replay: the ETH position from L_081726_040520 ----------------------- #
    # Bought NO at 0.087; the book then ran the other way (no bid 0.091, 0.069,
    # 0.051, 0.042, 0.033 ...) and it settled worthless. A take-profit never
    # fires on a position that only ever falls - the rule cannot rescue a
    # losing trade, it can only stop a winner from becoming one.
    t = trader(take_profit_multiple=2.0)
    losing = held("NO", 0.087, count=86)
    fired = [t.exit_reason(losing, book(1.0 - nb, 1.0 - nb + 0.001), 250.0)
             for nb in (0.091, 0.069, 0.051, 0.042, 0.033, 0.014, 0.006)]
    check("the real losing ETH position never triggers an exit",
          not any(fired), "it fell from 0.087 to 0.006 without ever doubling")

    # And the trade it WOULD have saved: same entry, book doubling instead.
    t = trader(take_profit_multiple=2.0)
    winner = held("NO", 0.087, count=86)
    t._orders.append(winner)
    # no_bid = 1 - yes_ask, so a 0.25 NO bid is book(0.74, 0.75).
    decision = t.exit_reason(winner, book(0.74, 0.75), 250.0)
    check("the same entry exits when the book DOES double",
          decision is not None, f"sell at {decision[1]:.3f}" if decision else "no")
    if decision:
        await t.close_position(winner, decision[1], decision[0])
        check("locking that gain beats the settlement it actually got",
              t.realized > 0 and winner.closed,
              f"${t.realized:+.2f} realized vs -$7.96 at settlement")

    # -- a position closed by hand must not be re-opened backwards ----------- #
    # The user closed one position manually in the Kalshi app. A close is a
    # SELL, and selling what you do not own OPENS the opposite position here -
    # so an exit firing against a position that is already gone would silently
    # start a new trade in the other direction. reduce_only lets the venue
    # refuse that outright.
    class ExitTrader(KalshiTrader):
        def __init__(self, fill_count="3"):
            class _C:
                pass

            super().__init__(_C(), RiskLimits(), dry_run=False)
            self.starting_balance = 20.0
            self._fill_count = fill_count
            self.sent = None

        async def _post(self, path, body):
            self.sent = body
            return {"order": {"order_id": "x", "fill_count": self._fill_count,
                              "average_fill_price": "0.7000"}}

    t = ExitTrader()
    pos = held("YES", 0.30, count=3)
    t._orders.append(pos)
    await t.close_position(pos, 0.70, "take-profit")
    check("every closing order is sent reduce_only",
          t.sent.get("reduce_only") is True,
          "so it can shrink a position but never create or flip one")
    check("closes are immediate-or-cancel too",
          t.sent["time_in_force"] == "immediate_or_cancel")

    gone = ExitTrader(fill_count="0")   # nothing left to reduce
    pos2 = held("NO", 0.40, count=3)
    gone._orders.append(pos2)
    gone.open_stake = pos2.stake
    res = await gone.close_position(pos2, 0.80, "take-profit")
    check("an exit that cannot fill stops tracking the position",
          not res.ok and pos2.closed,
          "reduce_only + no fill means it is already gone")
    check("and its stake is released rather than pinned open",
          abs(gone.open_stake) < 1e-9, f"${gone.open_stake:.2f}")
    check("a vanished position books no phantom PnL",
          abs(gone.realized) < 1e-9, f"${gone.realized:+.2f}")

    # -- replay: the two positions in the user's Kalshi screenshots ---------- #
    # Both were 1-contract side-mapping probes that settled worthless:
    #   BTC NO @ 0.19 (target $63,441.03) - peaked at a 0.26 bid = 1.23x net
    #   ETH NO @ 0.12 (target $1,901.25)  - peaked at a 0.57 bid = 4.34x net
    # The ETH one is the trade worth having: a 2x rule exits it for a profit
    # instead of losing the whole stake.
    t = trader(take_profit_multiple=2.0)
    eth_probe = OrderResult(ok=True, dry_run=True, ticker="KXETH15M-26AUG170100-00",
                            outcome="NO", api_side="ask", price=0.12,
                            api_price=0.88, count=1, verification=True)
    t._orders.append(eth_probe)
    path = [0.14, 0.22, 0.31, 0.44, 0.57]  # the NO bid as it actually moved
    exit_at = None
    for nb in path:
        d = t.exit_reason(eth_probe, book(1.0 - nb - 0.01, 1.0 - nb), 300.0)
        if d and exit_at is None:
            exit_at = d[1]
    check("the ETH screenshot position would now be sold on the way up",
          exit_at is not None and exit_at <= 0.57,
          f"exits at {exit_at:.2f} instead of riding 0.12 -> 0.57 -> $0.00"
          if exit_at else "never exits")
    check("its peak is remembered for the settlement report",
          abs(eth_probe.peak_mark - 0.57) < 1e-9, f"{eth_probe.peak_mark:.3f}")

    # The BTC one only reached 1.23x net, so a 2x rule correctly leaves it -
    # the rule is not a promise to catch every mover.
    t2 = trader(take_profit_multiple=2.0)
    btc_probe = OrderResult(ok=True, dry_run=True, ticker="KXBTC15M-26AUG170100-00",
                            outcome="NO", api_side="ask", price=0.19,
                            api_price=0.81, count=1, verification=True)
    fired = [t2.exit_reason(btc_probe, book(1.0 - nb - 0.01, 1.0 - nb), 300.0)
             for nb in (0.20, 0.25, 0.26)]
    check("the BTC screenshot position never reached 2x, so it is not exited",
          not any(fired), "peaked at 1.23x net - a lower threshold would be needed")
    check("but its peak is still recorded, so the threshold can be tuned",
          abs(btc_probe.peak_mark - 0.26) < 1e-9, f"{btc_probe.peak_mark:.3f}")

    # -- verification from the fill price, not a position read --------------- #
    # The position endpoint failed to confirm a fill NINE times across four
    # sessions - reading empty 8s after orders that demonstrably filled and
    # later settled. That blocked every strategy trade in a whole overnight run
    # (L_081726_053906: 3 probes, 0 strategy orders, -$0.30). The fill price
    # answers it with no second call: an order sent as ask 0.001 can only fill
    # ABOVE its limit if it sold into a bid.
    class ProbeTrader(KalshiTrader):
        def __init__(self, avg_fill, fill_count="1", balances=None):
            seq = list(balances or [])

            class _C:
                async def positions(self, **kw):
                    return {"market_positions": []}   # the endpoint that fails

                async def fills(self, **kw):
                    return {"fills": []}

                async def balance(self):
                    return {"balance": seq.pop(0)} if seq else {"balance": 2000}

            super().__init__(_C(), RiskLimits(), dry_run=False)
            self.starting_balance = 20.0
            self._resp = {"order": {"order_id": "x", "fill_count": fill_count,
                                    "average_fill_price": avg_fill}}

        def authenticate(self):
            pass

        async def _post(self, path, body):
            return self._resp

    real = ProbeTrader("0.9450")   # sold YES at 0.945 against a 0.001 limit
    check("the mapping is proven from the fill price alone",
          await real.verify_side_mapping("T") is True and real.side_mapping_verified,
          "a sell can only fill above its limit; a buy cannot")
    check("the position endpoint is not consulted when the fill is decisive",
          real.verification_attempts == 1)

    # A fill AT the limit says nothing about direction (the YES bid really was
    # ~0.001), so the balance decides. Intending NO at 0.999: a correct fill
    # debits ~99.9c, a reversed one would debit ~0.1c.
    right = ProbeTrader("0.0010", balances=[2000, 1900])  # spent $1.00
    check("a limit fill is resolved by the balance delta",
          await right.verify_side_mapping("T") is True,
          "$1.00 spent matches NO at 0.999")
    wrong = ProbeTrader("0.0010", balances=[2000, 1999])  # spent $0.01
    check("a balance matching the OTHER side is the reversed mapping, and halts",
          await wrong.verify_side_mapping("T") is False,
          "$0.01 spent matches YES at 0.001, not the NO we asked for")

    nofill = ProbeTrader("0", fill_count="0")
    check("a probe that does not fill stays inconclusive",
          await nofill.verify_side_mapping("T") is None)

    capped = ProbeTrader("0", fill_count="0")
    capped.limits.max_verification_attempts = 1
    await capped.verify_side_mapping("T")
    check("running out of attempts halts with 'unproven', not 'REVERSED'",
          await capped.verify_side_mapping("T") is False
          and "unproven" in capped.halt_reason and "REVERSED" not in capped.halt_reason,
          capped.halt_reason)

    # -- the thesis-complete exit -------------------------------------------- #
    # Bought NO at 0.14 because the model said NO was worth 0.24. When the book
    # reprices to 0.24 the mispricing is closed - that is the whole trade.
    t = trader(take_profit_multiple=99.0)  # multiple far out of reach
    pos = held("NO", 0.14)
    pos.entry_fair = 0.2411
    check("no exit before the book reaches our fair value",
          t.exit_reason(pos, book(1.0 - 0.19 - 0.01, 1.0 - 0.19), 300.0) is None)
    d = t.exit_reason(pos, book(1.0 - 0.28 - 0.01, 1.0 - 0.28), 300.0)
    check("exits once the book reprices to what we thought it was worth",
          d is not None and d[0] == "thesis-complete",
          f"fair 0.241, sold at {d[1]:.2f}" if d else "no exit")
    t2 = trader(take_profit_multiple=99.0, exit_at_fair_value=False)
    check("--no-fair-exit turns that off",
          t2.exit_reason(pos, book(1.0 - 0.28 - 0.01, 1.0 - 0.28), 300.0) is None)

    # -- the 1.75x mover that a 2.0x rule sat out ---------------------------- #
    # L_081726_053906: probe NO @ 0.15 peaked at a 0.31 bid = 1.75x net, then
    # settled worthless. The default is 1.5x precisely so this one is taken.
    t = trader(take_profit_multiple=1.5)
    p2 = held("NO", 0.15, count=1)
    d = t.exit_reason(p2, book(1.0 - 0.31 - 0.01, 1.0 - 0.31), 300.0)
    check("the 1.75x mover from the overnight run is now exited",
          d is not None, "2.0x sat this out and it settled worthless")
    old = trader(take_profit_multiple=2.0)
    check("and a 2.0x threshold demonstrably would not have",
          old.exit_reason(p2, book(1.0 - 0.31 - 0.01, 1.0 - 0.31), 300.0) is None)

    # -- the stop, off by default -------------------------------------------- #
    check("no stop-loss unless asked for",
          trader().limits.stop_loss_fraction == 0.0)
    t = trader(stop_loss_fraction=0.5)
    d = t.exit_reason(held("YES", 0.40), book(0.15, 0.17), 300.0)
    check("a stop fires when enabled", d is not None and d[0] == "stop-loss")


# --------------------------------------------------------------------------- #
# Multi-asset and the aggressive preset
# --------------------------------------------------------------------------- #


async def test_multi_asset_and_preset() -> None:
    print("\n--- multi-asset instruments and the aggressive preset ---")

    from kalshi import ASSETS, asset_for
    from kalshi_monitor import (
        AGGRESSIVE_PRESET,
        Instrument,
        Monitor,
        _requested_assets,
        parse_args,
    )

    # -- the series/feed pairing ------------------------------------------- #
    # Pricing an ETH contract off the BTC tape compares ~$63,000 against a
    # ~$1,880 strike and reports certainty on every observation. The pairing is
    # one object so a caller cannot express the mismatch.
    btc, eth = asset_for("BTC"), asset_for("ETH")
    check("each asset carries its own series and spot product",
          btc.series != eth.series and btc.coinbase_product != eth.coinbase_product,
          f"{btc.series}/{btc.coinbase_product} vs {eth.series}/{eth.coinbase_product}")
    check("a series ticker resolves to its own asset",
          asset_for("KXETH15M").name == "ETH" and asset_for("KXBTC15M").name == "BTC")
    try:
        asset_for("KXSOL15M")
        refused = False
    except ValueError:
        refused = True
    check("an underlying with no configured feed is refused, not guessed", refused)
    check("no two assets share a composite source table",
          len({a.composite for a in ASSETS.values()}) == len(ASSETS))

    # -- instruments are independent --------------------------------------- #
    a, b = Instrument(btc), Instrument(eth)
    a.buffer.add(63_000.0, 0, 1)
    b.buffer.add(1_880.0, 0, 1)
    check("each instrument keeps its own tape",
          abs(a.buffer.last().price - 63_000.0) < 1e-9
          and abs(b.buffer.last().price - 1_880.0) < 1e-9)
    a.observations = 10
    check("counters do not leak between instruments", b.observations == 0)

    # -- asset selection ---------------------------------------------------- #
    import argparse as _ap
    check("--assets selects several",
          _requested_assets(_ap.Namespace(assets="BTC,ETH", series="KXBTC15M"))
          == ["BTC", "ETH"])
    check("duplicates collapse",
          _requested_assets(_ap.Namespace(assets="BTC,btc,BTC", series=None)) == ["BTC"])
    check("a bare --series still works, single-asset",
          _requested_assets(_ap.Namespace(assets=None, series="KXETH15M")) == ["ETH"])

    args = parse_args(["--assets", "BTC,ETH", "--no-log"])
    mon = Monitor(args)
    check("a monitor builds one instrument per asset",
          [i.name for i in mon.instruments] == ["BTC", "ETH"])
    check("instruments are wired to their own feed",
          mon.instruments[0].asset.coinbase_product == "BTC-USD"
          and mon.instruments[1].asset.coinbase_product == "ETH-USD")

    # -- the preset touches opportunity, never correctness ------------------ #
    plain = parse_args(["--no-log"])
    hot = parse_args(["--aggressive", "--no-log"])
    check("the preset speeds up confirmation and book polling",
          hot.confirm_seconds < plain.confirm_seconds
          and hot.book_interval < plain.book_interval,
          f"{hot.confirm_seconds}s / book {hot.book_interval}s")
    check("the preset loosens the opportunity thresholds",
          hot.endgame_z < plain.endgame_z and hot.min_edge < plain.min_edge
          and hot.max_stake_pct > plain.max_stake_pct)

    # The whole point of the preset is what it does NOT do. Every gate here
    # was added after a graded session lost money.
    for guard in ("vol_ratio_max", "allow_unvalidated_vol", "max_exposure_pct",
                  "daily_loss_pct", "max_trades", "min_seconds_left"):
        check(f"the preset leaves {guard} alone",
              getattr(hot, guard) == getattr(plain, guard),
              f"{getattr(hot, guard)}")
    check("no correctness gate is even named in the preset",
          not ({"vol_ratio_max", "allow_unvalidated_vol", "daily_loss_pct",
                "max_exposure_pct", "max_trades"} & set(AGGRESSIVE_PRESET)))

    explicit = parse_args(["--aggressive", "--endgame-z", "3.0",
                           "--max-stake-pct", "5", "--no-log"])
    check("a value typed on the command line outranks the preset",
          explicit.endgame_z == 3.0 and explicit.max_stake_pct == 5.0,
          "even when it equals the default")
    check("unspecified flags still take the preset",
          explicit.confirm_seconds == AGGRESSIVE_PRESET["confirm_seconds"])

    # -- gates require EVERY instrument ------------------------------------- #
    from types import SimpleNamespace

    from kalshi_main import evaluate_gates

    def ready(inst):
        inst.stream = SimpleNamespace(connected=True)
        inst.basis = None
        inst.market = SimpleNamespace(ticker=f"{inst.series}-T", strike_known=True)
        now = time.monotonic()
        for i in range(35):
            inst.buffer._bars.append((now - 35 + i, 1000.0))
        inst.observations = 500
        inst.vol_ratios = [1.1] * 60
        return inst

    both = Monitor(parse_args(["--assets", "BTC,ETH", "--no-log"]))
    for inst in both.instruments:
        ready(inst)
    passed, lines = evaluate_gates(both, 1.5)
    check("both instruments healthy promotes", passed)
    check("the report names each instrument",
          any("BTC" in line for line in lines) and any("ETH" in line for line in lines))

    both.instruments[1].vol_ratios = [3.0] * 60  # ETH sigma disagrees
    passed, lines = evaluate_gates(both, 1.5)
    check("one bad instrument blocks promotion for the whole account",
          not passed, "shared balance, so there is no half-live")


# --------------------------------------------------------------------------- #
# Data quality, depth, reconciliation, and the STALE time-decay correction
# --------------------------------------------------------------------------- #


async def test_book_stream() -> None:
    print("\n--- kalshi websocket order book ---")

    import contextlib
    import inspect as _insp

    import kalshi as kx
    from kalshi import KalshiBookStream

    sent: list[dict] = []
    pushed: list[tuple[str, object]] = []
    resets: list[int] = []

    class FakeWS:
        """Enough of a websocket to drive the stream deterministically."""

        def __init__(self, script):
            self._script = list(script)
            self.closed = False

        async def send(self, raw):
            sent.append(json.loads(raw))

        async def recv(self):
            if not self._script:
                raise asyncio.CancelledError
            item = self._script.pop(0)
            if item is None:
                raise asyncio.TimeoutError
            return json.dumps(item)

        async def close(self):
            self.closed = True

    def snap(seq, ticker="T", yes=(("0.40", "100"),), no=(("0.55", "200"),)):
        return {"type": "orderbook_snapshot", "sid": 1, "seq": seq,
                "msg": {"market_ticker": ticker, "market_id": "u",
                        "yes_dollars_fp": [list(x) for x in yes],
                        "no_dollars_fp": [list(x) for x in no]}}

    def delta(seq, price, amount, side, ticker="T"):
        return {"type": "orderbook_delta", "sid": 1, "seq": seq,
                "msg": {"market_ticker": ticker, "market_id": "u",
                        "price_dollars": price, "delta_fp": amount, "side": side}}

    def stream(script, tickers=("T",)):
        s = KalshiBookStream(
            signer=None, tickers=tickers,
            on_book=lambda t, b: pushed.append((t, b)),
            on_reset=lambda: resets.append(1),
        )
        return s, FakeWS(script)

    # -- a snapshot builds the book, resolving Kalshi's two-bid convention --- #
    s, ws = stream([snap(1)])
    with contextlib.suppress(asyncio.CancelledError):
        await s._read_until_closed(ws)
    book = s.book("T")
    check("a snapshot builds a book",
          book is not None and book.yes_bid == 0.40 and book.no_bid == 0.55,
          f"yes_bid {book.yes_bid} no_bid {book.no_bid}")
    check("and the YES ask is the complement of the best NO bid, not a yes level",
          abs(book.yes_ask - 0.45) < 1e-9,
          f"yes_ask {book.yes_ask} = 1 - {book.no_bid}")
    check("the subscription names the channel and the markets",
          sent and sent[0]["cmd"] == "subscribe"
          and sent[0]["params"]["channels"] == ["orderbook_delta"]
          and sent[0]["params"]["market_tickers"] == ["T"],
          f"{sent[0] if sent else None}")

    # -- deltas patch levels, including removal ------------------------------ #
    s, ws = stream([snap(1), delta(2, "0.41", "50", "yes"),
                    delta(3, "0.40", "-100", "yes")])
    with contextlib.suppress(asyncio.CancelledError):
        await s._read_until_closed(ws)
    book = s.book("T")
    check("a delta adds a level and moves the top of book",
          book.yes_bid == 0.41, f"yes_bid {book.yes_bid}")
    check("a delta that empties a level removes it",
          all(p != 0.40 for p, _ in book.yes_levels),
          f"levels {book.yes_levels}")
    check("every applied change publishes a book",
          len(pushed) >= 3, f"{len(pushed)} pushes")

    # -- a missed sequence number is the dangerous case ---------------------- #
    sent.clear()
    resets.clear()
    s, ws = stream([snap(1), delta(2, "0.41", "50", "yes"),
                    delta(9, "0.99", "500", "yes")])   # 3..8 missed
    with contextlib.suppress(asyncio.CancelledError):
        await s._read_until_closed(ws)
    check("a sequence gap is counted, not absorbed", s.gaps == 1, f"gaps {s.gaps}")
    check("the local book is DISCARDED rather than patched from a gap",
          s.book("T") is None,
          "a book built on a missed delta is wrong in a way no age check sees")
    check("and a fresh snapshot is requested without resubscribing",
          any(m["cmd"] == "update_subscription"
              and m["params"]["action"] == "get_snapshot" for m in sent),
          f"{[m['cmd'] for m in sent]}")
    check("holders of the old book are told it is no longer maintained",
          resets, "on_reset fired")

    # The delta that arrived after the gap must not be applied: it is exactly
    # the message we cannot place relative to the snapshot on its way.
    check("the post-gap delta is dropped, not applied to an empty book",
          s.book("T") is None and s.deltas == 1,
          f"{s.deltas} deltas applied (only the pre-gap one)")

    # -- a gapping stream must be rebuilt, not endlessly resnapshotted ------- #
    # Note the shape: every gap is followed by an in-sequence snapshot, so a
    # CONSECUTIVE-gap counter would reset each time and never fire. The rate is
    # the thing that says the socket is broken.
    s, ws = stream([snap(1), delta(5, "0.41", "1", "yes"), snap(6),
                    delta(20, "0.41", "1", "yes"), snap(21),
                    delta(40, "0.41", "1", "yes")])
    raised = None
    try:
        await s._read_until_closed(ws)
    except ConnectionError as exc:
        raised = exc
    except asyncio.CancelledError:
        pass
    check("gaps arriving faster than the window force a reconnect",
          raised is not None and s.gaps == 3, f"{raised} after {s.gaps} gaps")

    # -- a delta with no snapshot cannot be applied -------------------------- #
    s, ws = stream([delta(1, "0.41", "50", "yes")])
    with contextlib.suppress(asyncio.CancelledError):
        await s._read_until_closed(ws)
    check("a delta for a market we have no snapshot for is ignored",
          s.book("T") is None and s.deltas == 0,
          "guessing a base book would invent depth nobody quoted")

    # -- subscription changes when the tracked contract rolls ---------------- #
    # A 15-minute contract expires roughly every 15 minutes, so this path runs
    # on every roll for the whole session.
    sent.clear()
    s, ws = stream([snap(1)], tickers=("T",))
    with contextlib.suppress(asyncio.CancelledError):
        await s._read_until_closed(ws)
    check("the first sync is a plain subscribe", sent[0]["cmd"] == "subscribe")

    s.track("T", "U")
    await s._sync_subscription(ws)
    check("adding a market updates the subscription in place",
          sent[-1]["cmd"] == "update_subscription"
          and sent[-1]["params"]["action"] == "add_markets"
          and sent[-1]["params"]["market_tickers"] == ["U"],
          f"{sent[-1]}")
    check("and the existing book survives the change",
          s.book("T") is not None, "no needless resubscribe")

    s.track("U")
    check("dropping a market drops its book immediately",
          s.book("T") is None,
          "an untracked contract must not be readable back out of the stream")
    await s._sync_subscription(ws)
    check("and tells the venue to stop sending it",
          sent[-1]["params"]["action"] == "delete_markets"
          and sent[-1]["params"]["market_tickers"] == ["T"],
          f"{sent[-1]}")

    # -- identity: version is content, seq is continuity --------------------- #
    s, ws = stream([snap(1), snap(2)])
    versions = []
    s._on_book = lambda t, b: versions.append((b.version, b.seq))
    with contextlib.suppress(asyncio.CancelledError):
        await s._read_until_closed(ws)
    check("an identical book keeps its version even as seq advances",
          len(versions) == 2 and versions[0][0] == versions[1][0]
          and versions[0][1] != versions[1][1],
          f"{versions}")

    # -- still no order surface --------------------------------------------- #
    src = _insp.getsource(kx)
    check("the module still issues no POST/PUT/DELETE",
          sum(src.count(f".{v}(") for v in ("post", "put", "delete")) == 0)
    cmds = {m["cmd"] for m in sent}
    check("the stream sends market-data commands only",
          cmds <= {"subscribe", "update_subscription", "unsubscribe"},
          f"{sorted(cmds)}")


async def test_accuracy_upgrades() -> None:
    print("\n--- accuracy: fees, freshness, depth, decay, reconciliation ---")

    import math as _m
    from statistics import NormalDist as _ND
    from types import SimpleNamespace as _NS

    from kalshi import KalshiBook, fee_at_size, fee_per_contract, trading_fee
    from kalshi_execution import KalshiTrader, RiskLimits
    from kalshi_monitor import (
        BOOK_SEQUENCE_GAP, EXCESSIVE_TIME_SKEW, STALE_BOOK, STALE_REFERENCE,
        STALE_SPOT, freshness_problem,
    )
    from strategies import scan_stale

    def book(levels_yes, levels_no):
        return KalshiBook.from_payload({"orderbook_fp": {
            "yes_dollars": [[f"{p:.4f}", f"{q}"] for p, q in levels_yes],
            "no_dollars": [[f"{p:.4f}", f"{q}"] for p, q in levels_no],
        }})

    # -- fees: rounding is charged on the ORDER, not per contract ------------ #
    check("fee_at_size reflects the rounding a small order actually pays",
          abs(fee_at_size(0.99, 3) - trading_fee(0.99, 3) / 3) < 1e-12
          and fee_at_size(0.99, 3) > 4 * fee_per_contract(0.99),
          f"{fee_at_size(0.99, 3):.5f}/c vs {fee_per_contract(0.99):.5f} modelled")
    check("it converges to the marginal rate as size grows",
          abs(fee_at_size(0.50, 200) - fee_per_contract(0.50)) < 5e-4)
    check("a 1-lot at 0.99 is charged a whole cent",
          abs(fee_at_size(0.99, 1) - 0.01) < 1e-9)

    # -- book identity: an unchanged snapshot is the SAME book --------------- #
    b1 = book([(0.60, 2)], [(0.36, 5)])
    b2 = book([(0.60, 2)], [(0.36, 5)])
    b3 = book([(0.61, 2)], [(0.36, 5)])
    check("identical books share a version", b1.version == b2.version)
    check("any change gives a new version", b1.version != b3.version)
    check("books carry their arrival time", b1.received_mono > 0 and b1.age >= 0.0)

    # -- depth: what a real order actually pays ------------------------------ #
    # 2 @ 0.63, 1 @ 0.64, 20 @ 0.67 - a 20-lot does NOT trade at 0.63.
    deep = book([], [(0.33, 20), (0.36, 1), (0.37, 2)])
    check("top of book is what a 1-lot pays",
          abs(deep.cost_for("YES", 1)[0] - 0.63) < 1e-9)
    vwap, filled = deep.cost_for("YES", 20)
    check("a larger order walks the ladder to a worse average",
          vwap > 0.63 and abs(filled - 20) < 1e-9, f"20 lots average {vwap:.4f}")
    check("it reports short fill rather than inventing depth",
          deep.cost_for("YES", 500)[1] == 23.0)
    check("an empty book cannot be priced at all",
          book([], []).cost_for("YES", 1) is None)

    # -- freshness: connected is not the same as current --------------------- #
    now = time.monotonic()

    def inst(spot_age=0.1, book_age=0.1, basis_age=1.0):
        buf = PriceBuffer()
        buf.add(63_000.0, 0, 1)
        buf._ticks[-1] = buf._ticks[-1].__class__(
            now - spot_age, 63_000.0, 0, 1)
        bk = book([(0.60, 5)], [(0.36, 5)])
        object.__setattr__(bk, "received_mono", now - book_age)
        return _NS(buffer=buf, book=bk,
                   basis=_NS(age=basis_age) if basis_age is not None else None)

    check("fresh data is allowed through",
          freshness_problem(inst(), now, 5.0, 3.0, 4.0, 180.0) is None)
    check("an old spot tick is refused by name",
          freshness_problem(inst(spot_age=30), now, 5.0, 3.0, 4.0, 180.0) == STALE_SPOT)
    check("an old book is refused by name",
          freshness_problem(inst(book_age=30), now, 5.0, 3.0, 4.0, 180.0) == STALE_BOOK)
    check("two individually fresh feeds that disagree about 'now' are refused",
          freshness_problem(inst(spot_age=0.1, book_age=2.9), now,
                            5.0, 3.0, 1.0, 180.0) == EXCESSIVE_TIME_SKEW,
          "a fresh book against a fresh spot is still useless if they are "
          "seconds apart from each other")
    check("a stale USD reference is refused by name",
          freshness_problem(inst(basis_age=9999), now, 5.0, 3.0, 4.0, 180.0)
          == STALE_REFERENCE)
    check("reference checking can be switched off",
          freshness_problem(inst(basis_age=9999), now, 5.0, 3.0, 4.0, 0.0) is None)

    # An orphaned book is the failure an age check cannot see: the stream
    # stopped maintaining it, but it still carries a recent timestamp.
    orphan = inst(book_age=0.1)
    orphan.book_gap_mono = now      # the gap happened after that book arrived
    check("a book the stream stopped maintaining is refused by name",
          freshness_problem(orphan, now, 5.0, 3.0, 4.0, 180.0) == BOOK_SEQUENCE_GAP,
          "recent timestamp, no longer true")
    replaced = inst(book_age=0.1)
    replaced.book_gap_mono = now - 60.0   # a REST poll has since replaced it
    check("and a book fetched after the gap is fine again",
          freshness_problem(replaced, now, 5.0, 3.0, 4.0, 180.0) is None)

    # -- the USD composite must describe ONE instant ------------------------- #
    from kalshi import CompositeBasis

    class FakeResp:
        def __init__(self, payload):
            self.status = 200
            self._payload = payload

        async def read(self):
            return json.dumps(self._payload).encode()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class FakeSession:
        """Each venue answers after its own delay, so lag is expressible."""

        def __init__(self, book):
            self._book = book      # url-substring -> (price, delay)
            self.calls = 0

        def get(self, url, **_):
            self.calls += 1
            for key, (price, delay) in self._book.items():
                if key in url:
                    return _DelayedResp(price, delay)
            raise AssertionError(f"unexpected url {url}")

    class _DelayedResp:
        def __init__(self, price, delay):
            self._price, self._delay = price, delay

        async def __aenter__(self):
            await asyncio.sleep(self._delay)
            return FakeResp({"last": f"{self._price}"})

        async def __aexit__(self, *a):
            return False

    sources = (("bitstamp", "//b/"), ("gemini", "//g/"),
               ("kraken2", "//k/"), ("stamp2", "//s/"))

    def basis_for(book, srcs=sources):
        # All four extract via the "last" key, which bitstamp/gemini share.
        named = tuple((("bitstamp" if i % 2 == 0 else "gemini"), url)
                      for i, (_, url) in enumerate(srcs))
        return CompositeBasis(FakeSession(book), sources=named), named

    slow = {"//b/": (63_000.0, 0.30), "//g/": (63_010.0, 0.30),
            "//k/": (63_005.0, 0.30), "//s/": (63_002.0, 0.30)}
    b, _ = basis_for(slow)
    t0 = time.monotonic()
    composite = await b.poll_once()
    elapsed = time.monotonic() - t0
    check("the venues are polled concurrently, not one after another",
          elapsed < 0.30 * 2 and composite is not None,
          f"{elapsed:.2f}s for 4 venues at 0.30s each")
    check("a concurrent poll leaves almost no self-inflicted skew",
          b.venue_skew < 0.15, f"skew {b.venue_skew:.3f}s")
    check("dispersion measures venue disagreement in dollars",
          abs(b.dispersion - 10.0) < 1e-6, f"${b.dispersion:.2f}")

    late = {"//b/": (63_000.0, 0.0), "//g/": (63_010.0, 0.0),
            "//k/": (63_005.0, 0.0), "//s/": (63_900.0, 2.5)}
    b, _ = basis_for(late)
    composite = await b.poll_once()
    check("a venue that answers far later is dropped, not averaged in",
          b.rejected_late == 1 and b.venues == 3,
          f"{b.rejected_late} late, {b.venues} kept - it describes another moment")

    wrong = {"//b/": (63_000.0, 0.0), "//g/": (63_010.0, 0.0),
             "//k/": (63_005.0, 0.0), "//s/": (70_000.0, 0.0)}
    b, _ = basis_for(wrong)
    await b.poll_once()
    check("a venue far from the median is dropped as an outlier",
          b.rejected_outlier == 1 and b.venues == 3,
          f"{b.rejected_outlier} outlier, {b.venues} kept")

    split = {"//b/": (63_000.0, 0.0), "//g/": (70_000.0, 0.0),
             "//k/": (55_000.0, 0.0), "//s/": (80_000.0, 0.0)}
    b, _ = basis_for(split)
    b.offset = -3.0
    check("venues that cannot agree produce NO composite",
          await b.poll_once() is None,
          "one venue and a disagreement is not a reference price")
    check("and the last good correction is left standing",
          b.offset == -3.0, "refusing to update beats publishing a guess")

    # -- STALE time decay ---------------------------------------------------- #
    # z = ln(S/K)/(sigma*sqrt(tau)), so the same moneyness is a LARGER z as tau
    # shrinks. Carrying the anchor's z forward unscaled assumes no time passed.
    def closing_in(seconds):
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + seconds))

    from kalshi import parse_market

    def mkt(close_s):
        return parse_market({
            "ticker": "T", "event_ticker": "E", "title": "t",
            "floor_strike": "63000.00", "open_time": "2026-08-19T00:00:00Z",
            "close_time": closing_in(close_s), "status": "active",
            "volume_fp": "1", "open_interest_fp": "1"})

    quote = book([(0.10, 500)], [(0.88, 500)])   # yes 0.10/0.12
    for left, label in ((900.0, "long dated"), (200.0, "mid"), (75.0, "near expiry")):
        m = mkt(left)
        tau = m.effective_tau()
        spot = 62_600.0
        old = scan_stale(m, quote, 62_540.0, 0.30, spot, 20.0, -9.0, 0.8e-4,
                         require_implied=False, max_edge=0.0, anchor_tau=0.0)
        new = scan_stale(m, quote, 62_540.0, 0.30, spot, 20.0, -9.0, 0.8e-4,
                         require_implied=False, max_edge=0.0, anchor_tau=tau * 2.0)
        if old and new:
            drift = abs(new.fair_yes - old.fair_yes)
            check(f"time decay shifts fair value at {label}", drift > 0.0,
                  f"tau {tau * 2:.0f}->{tau:.0f}s moves it {drift:+.4f}")

    z0 = _ND().inv_cdf(0.70)
    check("the correction matches the closed form z0*sqrt(tau0/tau1)",
          abs(_ND().cdf(z0 * _m.sqrt(90.0 / 30.0)) - 0.8181) < 1e-3,
          "a 0.70 anchor at tau 90 is really 0.818 by tau 30")

    # -- ENDGAME prices the settlement average, not the original strike ------ #
    # Inside the final 60s the contract settles on a mean that is PART PRINTED.
    # The terminal model treats all of it as still random, which is how it can
    # read 0.99 on a contract whose average is already lost.
    from kalshi import MIN_TWAP_COVERAGE
    from strategies import scan_endgame

    SIG = 0.4e-4          # bps/s, measured
    STRIKE = 63_000.0

    def realized_fv(left, mean, covered=None, spot=63_020.0, ref=0.0):
        m = mkt(left)
        cov = (m.twap_lookback - min(left, m.twap_lookback)) if covered is None else covered
        return m.fair_value_realized(spot, SIG, (mean, cov), reference_error=ref)

    check("before the averaging window opens there is nothing realized to use",
          realized_fv(120.0, 62_990.0, covered=60.0) is None,
          "falls back to the terminal model")
    check("a tape that did not watch the window refuses to price it",
          realized_fv(25.0, 62_980.0, covered=35.0 * MIN_TWAP_COVERAGE - 1.0) is None,
          "no silent extrapolation across the unwatched part")
    check("adequate coverage does price it",
          realized_fv(25.0, 62_980.0, covered=35.0) is not None)

    lost = realized_fv(25.0, 62_980.0)
    terminal = mkt(25.0).fair_value(63_020.0, SIG)
    check("an average already running against us is NOT near-certain",
          lost is not None and lost < 0.25 and terminal > 0.95,
          f"realized {lost:.3f} vs terminal {terminal:.3f} - the whole point")
    check("the observed mean actually moves the answer",
          realized_fv(25.0, 63_060.0) > realized_fv(25.0, 62_980.0) + 0.5,
          "a moving effective strike, not a constant one")

    # The moving strike amplifies any error in our reading of the realized part
    # by e/r, so the last seconds are the LEAST trustworthy, not the most.
    tight = realized_fv(25.0, 62_999.0, ref=0.0)
    wide = realized_fv(25.0, 62_999.0, ref=6.0)
    check("a disputed reference makes the realized model less certain, not more",
          wide < tight - 0.02,
          f"{tight:.3f} -> {wide:.3f} with $6 of venue disagreement")
    check("and it dominates as the remaining window vanishes",
          abs(realized_fv(3.0, 62_999.0, ref=6.0) - 0.5) < 0.15,
          "3s out, a 19x-amplified reference error is most of what we know")

    # z is reported outside clamp_prob's [0.01, 0.99] band, whose z is only
    # 2.33: inverting the clamped probability would cap every reading below a
    # 2.5 or 3.0 gate and disable the strategy on this path entirely.
    z_settled = mkt(25.0).realized_z(63_060.0, SIG, (63_060.0, 35.0))
    check("certainty is reported in sigma, not through the probability clamp",
          z_settled is not None and z_settled > 3.0,
          f"z {z_settled:.1f} would have been capped at 2.33 via inv_cdf(0.99)")

    # Coverage must be a DENSITY. A span would read a feed that died 30s ago as
    # full coverage (its endpoints are still far apart), which is exactly the
    # reading that lets a dead tape pose as a known settlement average.
    from btc_polymarket_arb import PriceBuffer as _PB

    def seeded(bars):
        buf = _PB()
        buf._bars.clear()
        for mono, price in bars:
            buf._bars.append((mono, price))
        return buf

    full = seeded([(1000.0 + i, 63_000.0) for i in range(40)])
    gapped = seeded([(1000.0 + i, 63_000.0) for i in range(10)])
    check("coverage counts the seconds actually observed",
          abs(full.mean_since(1000.0, 1039.0)[1] - 39.0) < 1.5,
          f"{full.mean_since(1000.0, 1039.0)[1]:.0f}s of a 39s window")
    check("a tape that stopped 30s ago reports the gap, not the span",
          gapped.mean_since(1000.0, 1039.0)[1] <= 10.0,
          f"{gapped.mean_since(1000.0, 1039.0)[1]:.0f}s - below the "
          f"{MIN_TWAP_COVERAGE:.0%} bar, so the model declines")

    # End to end: the setup where the two models disagree.
    near = book([(0.10, 500)], [(0.03, 500)])    # YES ask 0.97
    fires = scan_endgame(mkt(25.0), near, 63_030.0, 5.0, SIG,
                         min_z=3.0, require_implied=False)
    refuses = scan_endgame(mkt(25.0), near, 63_030.0, 5.0, SIG,
                           min_z=3.0, require_implied=False,
                           realized=(62_980.0, 35.0))
    check("ENDGAME still fires when only the strike is known",
          fires is not None and fires.legs[0].side == "YES",
          "terminal model sees 4.1 sigma")
    check("and refuses the same trade once the printed average contradicts it",
          refuses is None,
          "35s of the settlement mean came in below the strike")

    # -- reconciliation: a timed-out POST must never be guessed at ----------- #
    class Recon(KalshiTrader):
        def __init__(self, history, fail_history=False):
            super().__init__(_NS(), RiskLimits(), dry_run=False)
            self.starting_balance = 50.0
            self._history = history
            self._fail = fail_history
            self.slept = 0.0

        async def _order_history(self, ticker):
            if self._fail:
                raise RuntimeError("gateway down")
            return self._history

        async def _post(self, path, body):
            self._sent = body
            raise TimeoutError("no response")

    real_sleep = asyncio.sleep

    async def _fast(_s):
        return None

    asyncio.sleep = _fast
    try:
        landed = Recon({"orders": [{"client_order_id": "WILL-BE-SET",
                                    "order_id": "srv-1", "fill_count": "2",
                                    "average_fill_price": "0.4000"}]})
        # The id is generated inside place(); patch the history to match it.
        orig_post = landed._post

        async def post_and_record(path, body):
            landed._history["orders"][0]["client_order_id"] = body["client_order_id"]
            raise TimeoutError("no response")

        landed._post = post_and_record
        res = await landed.place("T", "YES", 0.40, 2)
        check("an order the venue HAS is reconciled, not resent",
              res.ok and res.count == 2,
              "recovered from the venue's own record after a timeout")

        absent = Recon({"orders": []})
        res = await absent.place("T", "YES", 0.40, 2)
        check("an order the venue does NOT have is a clean failure",
              not res.ok and not absent.halted,
              "safe to treat as never sent")

        unknown = Recon({}, fail_history=True)
        res = await unknown.place("T", "YES", 0.40, 2)
        check("an order whose state cannot be established HALTS the session",
              not res.ok and unknown.halted and "unknown state" in unknown.halt_reason,
              "an untracked position is worse than a missed trade")
        check("the halt names the order so a human can reconcile it",
              "T" in unknown.halt_reason)
    finally:
        asyncio.sleep = real_sleep


# --------------------------------------------------------------------------- #
# Telegram control
# --------------------------------------------------------------------------- #


async def test_telegram() -> None:
    print("\n--- telegram control ---")

    import argparse as _ap
    import logging as _lg

    from kalshi_telegram import NOTIFY, Notifier, TelegramBot, _split

    def bot(chat_id):
        b = TelegramBot("token", chat_id, _ap.Namespace(
            env_file=".env", session_min=0.0, settle_wait_min=25.0,
            monitor_flags=[]))
        b.sent = []

        async def _send(text, chat_id=None):
            b.sent.append((chat_id or b.chat_id, text))

        b.send = _send
        return b

    # -- the allowlist. This bot spends real money; a leaked token must not be
    # -- enough for a stranger to send /start.
    b = bot("111")
    fired = []
    b._command = lambda text: fired.append(text) or _noop()

    async def _noop():
        return None

    await b._on_update({"message": {"text": "/start live", "chat": {"id": 999}}})
    check("a command from an unknown chat is ignored entirely",
          not fired and not b.sent, "not even an error reply, which would confirm the bot exists")
    await b._on_update({"message": {"text": "/start live", "chat": {"id": 111}}})
    check("a command from the allowlisted chat is handled", fired == ["/start live"])

    # Before registration it helps you onboard, but still refuses to act.
    fresh = bot(None)
    acted = []
    fresh._command = lambda text: acted.append(text) or _noop()
    await fresh._on_update({"message": {"text": "/start live", "chat": {"id": 42}}})
    check("an unregistered bot replies with the chat id but runs nothing",
          not acted and fresh.sent and "42" in fresh.sent[0][1],
          "TELEGRAM_CHAT_ID onboarding")

    # -- notifications are lifted off the log, not wired into the trade path -- #
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    handler = Notifier(q, loop)
    logger = _lg.getLogger("tg-test")
    logger.addHandler(handler)
    logger.setLevel(_lg.INFO)
    logger.warning("EXITED take-profit NO x3: 0.20 -> 0.40")
    logger.info("hb | [BTC] spot=$64,000 ...")          # noise
    logger.warning("SETTLED KXBTC15M-X NO x3 -> NO : +0.57")
    await asyncio.sleep(0)
    got = []
    while not q.empty():
        got.append(q.get_nowait())
    logger.removeHandler(handler)
    check("fills, exits and settlements are forwarded", len(got) == 2, str(len(got)))
    check("heartbeats and routine chatter are not", all("hb |" not in g for g in got))
    check("every notify marker is a real log string the bot emits",
          all(isinstance(m, str) and m for m in NOTIFY))

    # -- long reports are split on line boundaries, never truncated ---------- #
    report = "\n".join(f"line {i} of the session summary" for i in range(400))
    chunks = _split(report, 500)
    check("a long summary is split rather than cut off",
          len(chunks) > 1 and "".join(chunks) == report)
    check("each chunk fits inside Telegram's message limit",
          all(len(c) <= 500 for c in chunks), f"max {max(len(c) for c in chunks)}")
    check("a short message is left alone", _split("hi", 500) == ["hi"])


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
    await test_book_stream()
    await test_accuracy_upgrades()
    await test_telegram()
    await test_take_profit()
    await test_multi_asset_and_preset()
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
