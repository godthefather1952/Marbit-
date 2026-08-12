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
import logging
import math
import time

from websockets.asyncio.server import serve

from btc_polymarket_arb import (
    BinanceTradeStream,
    BookFeed,
    Config,
    Credentials,
    ExecutionSettings,
    Executor,
    PriceBuffer,
    SignalEngine,
    TrackedMarket,
    quantize_price,
)
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
            "UP_TOKEN": (round(self.up_ask - 0.01, 4), self.up_ask),
            "DOWN_TOKEN": (round(self.down_ask - 0.01, 4), self.down_ask),
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
    return json.dumps(
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


async def drive(
    *,
    up_ask: float,
    down_ask: float,
    spike_bps: float = 25.0,
    enforce_ceiling: bool = True,
    cooldown: float = 5.0,
    settings: ExecutionSettings | None = None,
    executor: Executor | None = None,
    run_for: float | None = None,
    on_tick=None,
) -> tuple[LogCapture, Executor, int]:
    """Run the full pipeline against a mock tape. Returns (logs, executor, ticks).

    Stops at the first signal unless `run_for` is given, in which case it runs
    for that many seconds so repeat-suppression can be counted.
    """
    exec_settings = settings or ExecutionSettings(dry_run=True, order_size=10.0)
    executor = executor or Executor(exec_settings)
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
    ex = Executor(ExecutionSettings(dry_run=True))
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
    ex = Executor(ExecutionSettings(dry_run=True, order_size=1000.0, max_notional=50.0))
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
    ex = Executor(ExecutionSettings(dry_run=True), Credentials())
    check("dry run arms without any credentials", await ex.connect())

    # Circuit breaker.
    class ExplodingExecutor(Executor):
        def _submit(self, *a, **kw):
            raise RuntimeError("venue on fire")

    ex = ExplodingExecutor(ExecutionSettings(dry_run=False, max_consecutive_errors=3))
    ex._client = object()  # bypass connect(); we only exercise the error path
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


async def main() -> None:
    logging.basicConfig(level=logging.CRITICAL)
    print("=" * 68)
    await test_signals()
    await test_dry_run_execution()
    await test_position_gate()
    await test_safety_gates()
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
