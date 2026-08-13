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
from py_clob_client.exceptions import PolyApiException
from py_clob_client.order_builder.constants import BUY, SELL

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


async def main() -> None:
    logging.basicConfig(level=logging.CRITICAL)
    print("=" * 68)
    await test_signals()
    await test_dry_run_execution()
    await test_position_gate()
    await test_safety_gates()
    await test_balance_and_sizing()
    await test_risk_breakers()
    await test_retry_backoff()
    await test_reconnect_resilience()
    await test_fast_json()
    await test_order_caching()
    await test_latency_profiler()
    await test_eval_loop_benchmark()
    await test_polymarket_us()
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
