#!/usr/bin/env python3
"""Live dry-run monitor for Kalshi's 15-minute BTC up/down market.

Streams spot, tracks the currently-open `KXBTC15M` contract, prices it against
the published strike, and reports the edge **after Kalshi's taker fee**. It
places no orders and cannot: it uses the read-only client.

    python kalshi_monitor.py
    python kalshi_monitor.py --min-edge 0.02 --spike-bps 12

The question it exists to answer is not "is there a mispricing" but "is there a
mispricing big enough to survive a fee that peaks at 1.75c per contract". Every
edge printed is net of that fee.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
import time

import aiohttp

from btc_polymarket_arb import (
    BINANCE_WS_FALLBACKS,
    BinanceTradeStream,
    PriceBuffer,
    load_dotenv,
    log,
)
from kalshi import (
    BTC_15M_SERIES,
    CoinbaseSpotStream,
    usd_spot_rest,
    KalshiBook,
    KalshiClient,
    KalshiCredentials,
    KalshiMarket,
    breakeven_fair_value,
    fee_per_contract,
    net_edge,
    trading_fee,
)


class Monitor:
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._buffer = PriceBuffer()
        # Kalshi settles on CF Benchmarks BRTI, a USD index. Binance quotes in
        # USDT, and that basis is as large as the signal, so the reference tape
        # must be USD-quoted. --binance exists only to demonstrate the error.
        self._stream = (
            BinanceTradeStream(self._buffer, BINANCE_WS_FALLBACKS)
            if args.binance
            else CoinbaseSpotStream(self._buffer)
        )
        self._market: KalshiMarket | None = None
        self._book: KalshiBook | None = None
        self._signals = 0
        self._observations = 0
        self._best_net = -1.0
        self._last_signal_mono = 0.0

    async def run(self) -> None:
        async with aiohttp.ClientSession(
            headers={"User-Agent": "kalshi-monitor/1.0"}
        ) as session:
            client = KalshiClient(session, KalshiCredentials.from_env())

            try:
                status = await client.exchange_status()
                log.info(
                    "Kalshi trading_active=%s | reference tape: %s",
                    (status or {}).get("trading_active"),
                    "Binance BTCUSDT (USDT - MISPRICED)" if self._args.binance
                    else "Coinbase BTC-USD (USD, BRTI constituent)",
                )
                rest = await usd_spot_rest(session)
                if rest:
                    log.info("USD spot cross-check (REST): $%s", f"{rest:,.2f}")
            except Exception as exc:  # noqa: BLE001
                log.error("Cannot reach Kalshi: %s", exc)
                return

            tasks = [
                asyncio.create_task(self._stream.run(), name="spot"),
                asyncio.create_task(self._discovery_loop(client), name="discovery"),
                asyncio.create_task(self._book_loop(client), name="book"),
                asyncio.create_task(self._eval_loop(), name="eval"),
                asyncio.create_task(self._heartbeat_loop(), name="heartbeat"),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    # -- loops -------------------------------------------------------------- #

    async def _discovery_loop(self, client: KalshiClient) -> None:
        while True:
            try:
                live = await client.live_markets(self._args.series)
                now = time.time()
                current = next(
                    (m for m in live if m.is_live(now) and m.seconds_remaining(now) > 5), None
                )
                if current and (self._market is None or current.ticker != self._market.ticker):
                    log.info(
                        "Tracking %s | strike %s | closes in %.0fs | vol %s",
                        current.ticker,
                        f"${current.strike:,.2f}",
                        current.seconds_remaining(now),
                        f"{current.volume:,.0f}",
                    )
                    self._market = current
                    self._book = None
                elif current:
                    # Keep quotes and status fresh on the tracked contract.
                    current_book = self._book
                    self._market = current
                    self._book = current_book
                elif self._market is not None and not self._market.is_live(now):
                    log.info("%s closed; waiting for the next window", self._market.ticker)
                    self._market = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("Discovery failed: %s", exc)
            await asyncio.sleep(self._args.discovery_interval)

    async def _book_loop(self, client: KalshiClient) -> None:
        while True:
            market = self._market
            if market is not None:
                try:
                    self._book = await client.book(market.ticker)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    log.warning("Book fetch failed: %s", exc)
            await asyncio.sleep(self._args.book_interval)

    async def _eval_loop(self) -> None:
        while True:
            with contextlib.suppress(Exception):
                self._evaluate()
            await asyncio.sleep(0.2)

    def _evaluate(self) -> None:
        market, book = self._market, self._book
        tick = self._buffer.last()
        if market is None or book is None or tick is None or not self._buffer.ready:
            return
        remaining = market.seconds_remaining()
        if remaining < self._args.min_seconds_left:
            return

        sigma = self._buffer.sigma_per_sqrt_second()
        fair = market.fair_value(tick.price, sigma)
        self._observations += 1

        for side, ask, fair_side in (
            ("YES", book.yes_ask, fair),
            ("NO", book.no_ask, 1.0 - fair),
        ):
            if ask is None or not (0.0 < ask < 1.0):
                continue
            edge = net_edge(fair_side, ask)
            self._best_net = max(self._best_net, edge)
            if edge < self._args.min_edge:
                continue
            if time.monotonic() - self._last_signal_mono < self._args.cooldown:
                continue

            move = self._buffer.largest_move(3.0)
            self._signals += 1
            self._last_signal_mono = time.monotonic()
            size = self._args.size
            gross = fair_side - ask
            fee = trading_fee(ask, size)
            log.warning(
                "\n"
                "================ EDGE (net of fees) ================\n"
                " market     : %s\n"
                " spot       : %s   strike %s   (%+.2f%% vs strike)\n"
                " expiry     : %.0fs remaining   tau_eff %.0fs   sigma %.2f bps/s\n"
                " fair value : %.4f  (%s)\n"
                " %-3s ask    : %.4f   size %s\n"
                " gross edge : %+.4f/contract\n"
                " taker fee  : -%.4f/contract  (breakeven fair %.4f)\n"
                " NET EDGE   : %+.4f/contract\n"
                " at %g contracts: gross $%+.2f, fee $%.2f, net $%+.2f\n"
                " 3s move    : %+.1f bps\n"
                "====================================================",
                market.ticker,
                f"${tick.price:,.2f}",
                f"${market.strike:,.2f}",
                (tick.price / market.strike - 1.0) * 100.0,
                remaining,
                market.effective_tau(),
                sigma * 10_000.0,
                fair_side,
                side,
                side,
                ask,
                f"{(book.yes_ask_size if side == 'YES' else book.yes_bid_size):,.0f}",
                gross,
                fee_per_contract(ask),
                breakeven_fair_value(ask),
                edge,
                size,
                gross * size,
                fee,
                gross * size - fee,
                move.bps if move else 0.0,
            )

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self._args.heartbeat)
            market, book, tick = self._market, self._book, self._buffer.last()
            if market is None:
                log.info("hb | no live %s market", self._args.series)
                continue
            sigma = self._buffer.sigma_per_sqrt_second()
            fair = market.fair_value(tick.price, sigma) if tick else float("nan")
            log.info(
                "hb | %s | spot=%s strike=%s | fair=%.3f | yes %s/%s | "
                "%.0fs left | obs=%d signals=%d best_net=%+.4f",
                market.ticker.split("-")[-2] if "-" in market.ticker else market.ticker,
                f"${tick.price:,.0f}" if tick else "n/a",
                f"${market.strike:,.0f}",
                fair,
                f"{book.yes_bid:.3f}" if book and book.yes_bid else "-",
                f"{book.yes_ask:.3f}" if book and book.yes_ask else "-",
                market.seconds_remaining(),
                self._observations,
                self._signals,
                self._best_net,
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--series", default=BTC_15M_SERIES, help="Kalshi series ticker")
    p.add_argument("--min-edge", type=float, default=0.02,
                   help="minimum NET edge per contract (after fees) to report")
    p.add_argument("--spike-bps", type=float, default=12.0, help="spot move of interest, bps")
    p.add_argument("--size", type=float, default=20.0, help="contracts, for the dollar figures")
    p.add_argument("--min-seconds-left", type=float, default=20.0,
                   help="ignore a market closer than this to expiry")
    p.add_argument("--book-interval", type=float, default=1.0, help="book poll seconds")
    p.add_argument("--discovery-interval", type=float, default=10.0, help="market poll seconds")
    p.add_argument("--cooldown", type=float, default=5.0, help="seconds between reports")
    p.add_argument("--heartbeat", type=float, default=15.0, help="heartbeat seconds")
    p.add_argument("--binance", action="store_true",
                   help="use Binance BTCUSDT instead of Coinbase BTC-USD. WRONG for this "
                        "venue - Kalshi settles in USD and Binance quotes USDT, a basis "
                        "as large as the signal. Provided to demonstrate the error.")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("websockets").setLevel(logging.WARNING)
    load_dotenv(args.env_file)
    return args


async def amain(args: argparse.Namespace) -> None:
    monitor = Monitor(args)
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    runner = asyncio.create_task(monitor.run())
    stopper = asyncio.create_task(stop.wait())
    done, pending = await asyncio.wait({runner, stopper}, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    for task in done:
        if task is runner:
            task.result()


def main() -> None:
    args = parse_args()
    log.info(
        "Kalshi monitor | READ-ONLY, no orders | series=%s | min NET edge %.3f",
        args.series,
        args.min_edge,
    )
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(amain(args))
    log.info("Monitor stopped")


if __name__ == "__main__":
    main()
