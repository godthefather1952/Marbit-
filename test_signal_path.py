#!/usr/bin/env python3
"""End-to-end exercise of the signal path without touching live venues.

Spins up a local websocket server that speaks the Binance `btcusdt@trade`
payload format and replays a scripted spike, pairs it with a stubbed CLOB book
whose asks are held deliberately stale, and asserts the engine fires (or stays
silent) for each gate independently.

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
    PriceBuffer,
    SignalEngine,
    TrackedMarket,
)

BASE = 100_000.0
FLAT_TICKS = 40  # ~2s of flat tape, so a pre-spike book anchor exists
TICK_INTERVAL = 0.05


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


def make_market() -> TrackedMarket:
    now = time.time()
    return TrackedMarket(
        slug="btc-updown-5m-TEST",
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


async def run_case(
    name: str,
    *,
    up_ask: float,
    down_ask: float,
    expect: str | None,
    spike_bps: float = 25.0,
    enforce_ceiling: bool = True,
) -> bool:
    """Drive one scenario. `expect` is a substring of the expected banner, or None."""
    captured: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    logger = logging.getLogger("btc-arb")
    handler = Capture()
    prior_level, prior_propagate = logger.level, logger.propagate
    # The engine logs signals at WARNING/ERROR. The logger must be permissive
    # or isEnabledFor() drops them before any handler is consulted.
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(handler)

    try:
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
            )
            engine = SignalEngine(buffer, books, cfg)
            engine.sync_markets([make_market()])

            stream = BinanceTradeStream(buffer, [f"ws://127.0.0.1:{port}"])
            tape = asyncio.create_task(stream.run())

            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                await engine.refresh_books()
                await engine.evaluate()
                if any(r.levelno >= logging.WARNING for r in captured):
                    break
                await asyncio.sleep(0.1)

            tape.cancel()
            await asyncio.gather(tape, return_exceptions=True)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prior_level)
        logger.propagate = prior_propagate

    fired = [r.getMessage() for r in captured if r.levelno >= logging.WARNING]
    if expect is None:
        ok = not fired
    else:
        ok = any(expect in msg for msg in fired)

    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if fired:
        print("\n".join(fired[:1]))
    elif expect is not None:
        print("       (no signal emitted)")
    return ok


async def main() -> None:
    logging.basicConfig(level=logging.CRITICAL)
    print("=" * 60)
    results = [
        # Spot rips +25bps; the Up ask still rests at 0.45 -> fire.
        await run_case(
            "UP spike, stale ask under the 0.52 gate -> SIGNAL",
            up_ask=0.45,
            down_ask=0.56,
            expect="ARBITRAGE WINDOW OPEN",
        ),
        # Spot dumps -25bps; the Down ask is stale -> fire on the NO leg.
        await run_case(
            "DOWN spike, stale NO ask -> SIGNAL",
            up_ask=0.56,
            down_ask=0.45,
            spike_bps=-25.0,
            expect="ARBITRAGE WINDOW OPEN",
        ),
        # Book already repriced to 0.97; ceiling disabled so the edge gate is
        # the only thing that can suppress it. Edge 0.01 < min_edge 0.03.
        await run_case(
            "book already repriced -> edge gate suppresses",
            up_ask=0.97,
            down_ask=0.04,
            enforce_ceiling=False,
            expect=None,
        ),
        # Real model edge, but the ask sits above the 0.52 staleness gate.
        await run_case(
            "ask above 0.52 ceiling -> ceiling gate suppresses",
            up_ask=0.60,
            down_ask=0.41,
            expect=None,
        ),
        # Model-free: both asks sum below $1.00.
        await run_case(
            "asks sum to 0.95 -> RISK-FREE CROSS-BOOK ARB",
            up_ask=0.45,
            down_ask=0.50,
            expect="RISK-FREE CROSS-BOOK ARB",
        ),
        # No spike at all -> nothing should ever fire.
        await run_case(
            "flat tape, no spike -> silent",
            up_ask=0.45,
            down_ask=0.56,
            spike_bps=0.0,
            expect=None,
        ),
    ]
    print("=" * 60)
    print("ALL PASS" if all(results) else f"{results.count(False)} FAILURE(S)")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
