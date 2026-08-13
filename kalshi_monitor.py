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

from run_log import start_run_log
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
        # Every observation's best net edge, so the summary can report the
        # distribution rather than just the maximum. This is the number that
        # answers whether the strategy is worth funding.
        self._net_edges: list[float] = []
        self._signal_by_market: dict[str, int] = {}
        self._markets_seen: set[str] = set()
        self._edge_seconds = 0.0
        self._last_eval_mono = 0.0
        self._start_mono = 0.0
        self._fallback_vol_obs = 0
        self.run_log = None

    async def run(self) -> None:
        self._start_mono = time.monotonic()
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
        self._markets_seen.add(market.ticker)
        if not self._buffer.vol_is_measured:
            self._fallback_vol_obs += 1
        best_this_pass = -1.0

        for side, ask, fair_side in (
            ("YES", book.yes_ask, fair),
            ("NO", book.no_ask, 1.0 - fair),
        ):
            if ask is None or not (0.0 < ask < 1.0):
                continue
            edge = net_edge(fair_side, ask)
            self._best_net = max(self._best_net, edge)
            best_this_pass = max(best_this_pass, edge)
            if edge < self._args.min_edge:
                continue
            if time.monotonic() - self._last_signal_mono < self._args.cooldown:
                continue

            move = self._buffer.largest_move(3.0)
            self._signals += 1
            self._signal_by_market[market.ticker] = (
                self._signal_by_market.get(market.ticker, 0) + 1
            )
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

        # Post-pass bookkeeping for the session summary.
        if best_this_pass > -1.0:
            self._net_edges.append(best_this_pass)
            now_mono = time.monotonic()
            if self._last_eval_mono and best_this_pass >= self._args.min_edge:
                # Wall time spent with a tradeable edge on the screen - a far
                # more useful figure than a raw signal count, because it says
                # how long the window actually stays open.
                self._edge_seconds += min(now_mono - self._last_eval_mono, 5.0)
            self._last_eval_mono = now_mono

    def build_summary(self) -> list[str]:
        """The numbers worth reading after a long run."""
        edges = self._net_edges
        elapsed = max(time.monotonic() - (self._start_mono or time.monotonic()), 1e-9)
        lines = [
            f"series        : {self._args.series}",
            f"reference     : {'Binance BTCUSDT (USDT - MISPRICED)' if self._args.binance else 'Coinbase BTC-USD (USD, BRTI constituent)'}",
            f"min net edge  : {self._args.min_edge:+.4f}/contract",
            f"markets seen  : {len(self._markets_seen)}",
            f"observations  : {self._observations:,}",
            f"signals       : {self._signals}",
        ]

        if self._fallback_vol_obs:
            share = self._fallback_vol_obs / max(self._observations, 1) * 100.0
            lines += [
                "",
                f"WARNING: {self._fallback_vol_obs:,} of {self._observations:,} observations "
                f"({share:.0f}%) priced with",
                "         ASSUMED volatility (45% annualized), not measured. The",
                "         estimator needs ~5 minutes of tape. Edges below are only",
                "         as trustworthy as that assumption - run longer before",
                "         drawing any conclusion.",
            ]
        if edges:
            lines += [
                "",
                "net edge per observation (after fees), best of YES/NO:",
                f"   p50  {_fmt_edge(_percentile(edges, 50))}",
                f"   p90  {_fmt_edge(_percentile(edges, 90))}",
                f"   p99  {_fmt_edge(_percentile(edges, 99))}",
                f"   max  {_fmt_edge(max(edges))}",
                "",
                f"time with a tradeable edge : {self._edge_seconds:.0f}s of {elapsed:.0f}s "
                f"({self._edge_seconds / elapsed * 100.0:.2f}%)",
            ]
            positive = sum(1 for e in edges if e >= self._args.min_edge)
            lines.append(
                f"observations at or above the threshold : {positive:,} of "
                f"{len(edges):,} ({positive / len(edges) * 100.0:.2f}%)"
            )
        else:
            lines += ["", "no observations recorded - the market or tape never came up"]

        if self._signal_by_market:
            lines += ["", "signals by market:"]
            for ticker, count in sorted(
                self._signal_by_market.items(), key=lambda kv: -kv[1]
            ):
                lines.append(f"   {count:4d}  {ticker}")

        if edges and max(edges) >= self._args.min_edge:
            lines += [
                "",
                "NOTE: a large, persistent edge on a liquid market usually means a",
                "      model or feed error, not free money. Sanity-check the fair",
                "      value against the quote before believing it.",
            ]
        return lines

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
            if not self._buffer.vol_is_measured:
                log.info("     (sigma is still the assumed prior, not measured from tape)")


def _fmt_edge(value: float) -> str:
    return "n/a" if value != value else f"{value:+.4f}"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(pct / 100.0 * (len(ordered) - 1)))))
    return ordered[idx]


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
    p.add_argument("--log-dir", default="logs",
                   help="directory for per-run log files (L_MMDDYY_HHMMSS.log)")
    p.add_argument("--no-log", action="store_true", help="console only, write no log file")
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


async def amain(args: argparse.Namespace, monitor: "Monitor") -> None:
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
    monitor = Monitor(args)

    run_log = None
    if not args.no_log:
        run_log = start_run_log(
            log,
            directory=args.log_dir,
            title="KALSHI MONITOR (READ-ONLY, NO ORDERS)",
            context=[
                f"series  : {args.series}",
                f"feed    : {'Binance BTCUSDT (USDT - MISPRICED)' if args.binance else 'Coinbase BTC-USD (USD, BRTI constituent)'}",
                f"min edge: {args.min_edge:+.4f}/contract, net of Kalshi taker fees",
            ],
        )
    monitor.run_log = run_log

    try:
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(amain(args, monitor))
    finally:
        # Written even on Ctrl+C, which is how a long monitoring run ends.
        if run_log is not None:
            run_log.close(monitor.build_summary())
        log.info("Monitor stopped")


if __name__ == "__main__":
    main()
