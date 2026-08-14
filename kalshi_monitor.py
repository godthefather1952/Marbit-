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
from pathlib import Path
import logging
import signal
import time

import aiohttp
from collections import deque

from run_log import run_log_name, start_run_log
from kalshi_execution import KalshiTrader, RiskLimits
from strategies import (
    PaperLedger,
    implied_sigma,
    scan_cross,
    scan_endgame,
    scan_stale,
    vol_agreement,
)
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
    CompositeBasis,
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
        self._basis: CompositeBasis | None = None
        self._stream = None  # built in run(), once a session exists
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
        #: The subset where our volatility agreed with the market's. Only these
        #: are candidates for being real; the rest are a sigma argument.
        self._net_edges_validated: list[float] = []
        self._signal_by_market: dict[str, int] = {}
        self._markets_seen: set[str] = set()
        self._edge_seconds = 0.0
        self._last_eval_mono = 0.0
        self._start_mono = 0.0
        self._fallback_vol_obs = 0
        # Volatility cross-check bookkeeping. The absolute model has no input
        # the book lacks except sigma, so a persistent disagreement here is the
        # whole of any "edge" it reports - tracked so the summary can say so.
        self._vol_ratios: list[float] = []
        self._sigma_pairs: list[tuple[float, float]] = []
        self._no_strike_obs = 0
        self._unvalidated_vol_obs = 0
        self._vol_gated_obs = 0
        self.run_log = None
        self.ledger: PaperLedger | None = None
        # Rolling anchor for STALE: the market's own mid and our spot at the
        # same instant, from `--anchor-age` seconds ago.
        self._anchors: deque[tuple[float, float, float]] = deque(maxlen=600)
        self._strategy_hits: dict[str, int] = {}
        self.trader: KalshiTrader | None = None
        self._settled: set[str] = set()
        self._pending_orders: list = []
        self._client_ref = None
        # Confirmation state: an edge must be re-seen across passes spanning
        # `--confirm-seconds` before it is recorded or traded. Key is
        # (strategy, ticker, leg sides); value tracks first/last sighting.
        self._candidates: dict[tuple, dict] = {}

    async def run(self) -> None:
        self._start_mono = time.monotonic()
        async with aiohttp.ClientSession(
            headers={"User-Agent": "kalshi-monitor/1.0"}
        ) as session:
            client = KalshiClient(session, KalshiCredentials.from_env())
            self._client_ref = client
            self.trader = KalshiTrader(
                client,
                RiskLimits(
                    max_stake_pct=self._args.max_stake_pct / 100.0,
                    max_exposure_pct=self._args.max_exposure_pct / 100.0,
                    daily_loss_pct=self._args.daily_loss_pct / 100.0,
                    max_trades=self._args.max_trades,
                ),
                dry_run=not self._args.live,
            )
            if not await self.trader.arm():
                log.error("Execution layer failed to arm; stopping before any market data")
                return
            if self._args.binance:
                self._stream = BinanceTradeStream(self._buffer, BINANCE_WS_FALLBACKS)
            else:
                if not self._args.no_basis:
                    self._basis = CompositeBasis(session)
                self._stream = CoinbaseSpotStream(self._buffer, basis=self._basis)

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
                *([asyncio.create_task(self._basis.run(self._buffer), name="basis")]
                  if self._basis is not None else []),
                asyncio.create_task(self._discovery_loop(client), name="discovery"),
                asyncio.create_task(self._book_loop(client), name="book"),
                asyncio.create_task(self._eval_loop(), name="eval"),
                asyncio.create_task(self._execution_loop(), name="execution"),
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

        now_mono = time.monotonic()
        sigma = self._buffer.sigma_per_sqrt_second()
        self._markets_seen.add(market.ticker)

        # Kalshi lists the contract before `floor_strike` posts, so the first
        # ~30-45s of every window has no strike. Nothing model-based can be
        # priced there; CROSS can, because it never looks at one.
        if not market.strike_known:
            self._no_strike_obs += 1
            if not self._args.no_cross:
                self._emit(scan_cross(market, book, self._args.size, self._args.min_profit))
            return

        self._observations += 1
        if not self._buffer.vol_is_measured:
            self._fallback_vol_obs += 1

        # The one number that decides whether any of this is real. See
        # strategies.vol_agreement.
        sigma_implied = implied_sigma(market, book, tick.price)
        ratio = vol_agreement(sigma, sigma_implied)
        if sigma_implied is not None:
            self._sigma_pairs.append((sigma, sigma_implied))
        if ratio is None:
            self._unvalidated_vol_obs += 1
        else:
            self._vol_ratios.append(ratio)
        vol_ok = ratio is not None and ratio <= self._args.vol_ratio_max
        if not vol_ok and not self._args.allow_unvalidated_vol:
            self._vol_gated_obs += 1

        fair = market.fair_value(tick.price, sigma)
        best_this_pass = -1.0

        # Keep a rolling (time, spot, market mid) anchor for the STALE model.
        mid = book.yes_mid
        if mid is not None:
            self._anchors.append((now_mono, tick.price, mid))
        self._run_strategies(market, book, tick.price, sigma, now_mono, vol_ok)

        if fair is None:
            return

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
            if not vol_ok and not self._args.allow_unvalidated_vol:
                # Suppressed on purpose. With the strike published and the
                # clock public, a disagreement about sigma is the only thing
                # that can produce this number, and ours is the side more
                # likely to be wrong.
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
                " expiry     : %.0fs remaining   tau_eff %.0fs\n"
                " sigma      : ours %.2f bps/s vs market %s bps/s  (ratio %s)\n"
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
                f"{sigma_implied * 10_000.0:.2f}" if sigma_implied else "n/a",
                f"{ratio:.2f}x" if ratio else "unvalidated",
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
            if vol_ok:
                self._net_edges_validated.append(best_this_pass)
            now_mono = time.monotonic()
            if self._last_eval_mono and best_this_pass >= self._args.min_edge and vol_ok:
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
            f"feed basis    : {('%+.2f USD (n=%d polls)' % (self._basis.offset, self._basis.samples)) if self._basis else 'not corrected'}",
            f"markets seen  : {len(self._markets_seen)}",
            f"paper trades  : {self.ledger.count if self.ledger else 0} "
            f"(score with: python kalshi_score.py)",
            f"strategy hits : {self._strategy_hits or 'none'}",
            f"execution     : {self.trader.stats() if self.trader else 'none'}",
            f"observations  : {self._observations:,}",
            f"signals       : {self._signals}",
        ]

        if self._no_strike_obs:
            lines.append(
                f"no-strike     : {self._no_strike_obs:,} passes skipped while "
                f"floor_strike had not posted"
            )

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

        lines += self._vol_lines()

        if edges:
            validated = self._net_edges_validated
            lines += [
                "",
                "net edge per observation (after fees), best of YES/NO.",
                "  RAW is every observation. VALIDATED is the subset where our",
                "  volatility agreed with the market's; only those can be real.",
                f"                    {'RAW':>10}  {'VALIDATED':>10}",
                f"   observations  {len(edges):>10,}  {len(validated):>10,}",
                f"   p50           {_fmt_edge(_percentile(edges, 50)):>10}  "
                f"{_fmt_edge(_percentile(validated, 50)):>10}",
                f"   p90           {_fmt_edge(_percentile(edges, 90)):>10}  "
                f"{_fmt_edge(_percentile(validated, 90)):>10}",
                f"   p99           {_fmt_edge(_percentile(edges, 99)):>10}  "
                f"{_fmt_edge(_percentile(validated, 99)):>10}",
                f"   max           {_fmt_edge(max(edges)):>10}  "
                f"{_fmt_edge(max(validated)) if validated else 'n/a':>10}",
                "",
                f"time with a validated tradeable edge : {self._edge_seconds:.0f}s of "
                f"{elapsed:.0f}s ({self._edge_seconds / elapsed * 100.0:.2f}%)",
            ]
            raw_pos = sum(1 for e in edges if e >= self._args.min_edge)
            val_pos = sum(1 for e in validated if e >= self._args.min_edge)
            lines += [
                f"observations at or above the threshold : {raw_pos:,} raw "
                f"({raw_pos / len(edges) * 100.0:.2f}%), {val_pos:,} validated "
                f"({val_pos / max(len(edges), 1) * 100.0:.2f}% of all)",
            ]
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
                "      model or feed error, not free money. Read the VALIDATED",
                "      column and the volatility block above before believing it.",
            ]
        return lines

    def _vol_lines(self) -> list[str]:
        """Report our volatility against the market's - the decisive comparison.

        With the strike published and the clock public, our only conceivable
        advantage over the book is the spot price, which the book sees at least
        as fast. So on the absolute model, an "edge" and a sigma disagreement
        are the same event described two ways. Printing the two side by side is
        what makes that visible instead of flattering.
        """
        pairs = self._sigma_pairs
        ratios = self._vol_ratios
        total = max(self._observations, 1)
        lines = ["", "volatility cross-check (ours vs the market's own quote):"]
        if not pairs or not ratios:
            lines += [
                "   never invertible - the book sat too close to the money all run,",
                "   so nothing model-based was validated and nothing was traded on it.",
            ]
            return lines

        ours = sorted(s for s, _ in pairs)
        theirs = sorted(i for _, i in pairs)
        lines += [
            f"   ours   (measured)  p50 {_percentile(ours, 50) * 1e4:.2f} bps/s"
            f"   p10 {_percentile(ours, 10) * 1e4:.2f}   p90 {_percentile(ours, 90) * 1e4:.2f}",
            f"   market (implied)   p50 {_percentile(theirs, 50) * 1e4:.2f} bps/s"
            f"   p10 {_percentile(theirs, 10) * 1e4:.2f}   p90 {_percentile(theirs, 90) * 1e4:.2f}",
            f"   disagreement       p50 {_percentile(ratios, 50):.2f}x"
            f"   p90 {_percentile(ratios, 90):.2f}x   max {max(ratios):.2f}x",
            f"   unvalidated passes {self._unvalidated_vol_obs:,} of {total:,} "
            f"({self._unvalidated_vol_obs / total * 100.0:.0f}%) - quote too close to the money",
            f"   suppressed passes  {self._vol_gated_obs:,} of {total:,} "
            f"({self._vol_gated_obs / total * 100.0:.0f}%) - above the "
            f"{self._args.vol_ratio_max:.2f}x limit",
        ]
        median_ratio = _percentile(ratios, 50)
        if median_ratio > self._args.vol_ratio_max:
            lines += [
                "",
                f"   Our volatility disagrees with the market's by {median_ratio:.1f}x at the",
                "   median. That is not an opportunity, it is a broken estimator: the",
                "   market quotes this book with a one-cent spread and real size, and",
                "   we have five minutes of one-second bars from one venue. Any edge",
                "   the absolute model reports here is that disagreement wearing a",
                "   dollar sign. Fix the estimator before reading anything else.",
            ]
        return lines

    def _run_strategies(self, market, book, spot, sigma, now_mono, vol_ok: bool) -> None:
        """Run every enabled strategy and record what each would have traded.

        `vol_ok` gates the two model-dependent strategies. CROSS is exempt: it
        reads only the two bids and locks its profit at settlement, so it is
        the one strategy that cannot be wrong about volatility because it never
        forms an opinion about it.
        """
        args = self._args
        found = []

        if not args.no_cross:
            sig = scan_cross(market, book, args.size, args.min_profit)
            if sig:
                found.append(sig)

        model_ok = vol_ok or args.allow_unvalidated_vol

        if not args.no_stale:
            # An anchor from `--anchor-age` seconds ago: the market's mid and
            # our spot at the same moment, so only the CHANGE is used.
            cutoff = now_mono - args.anchor_age
            anchor = None
            for ts, price, mid in self._anchors:
                if ts <= cutoff:
                    anchor = (price, mid)
                else:
                    break
            if anchor:
                # STALE takes sigma from the quote itself, so it needs no
                # agreement check - it is already using the market's number.
                sig = scan_stale(
                    market, book, anchor[0], anchor[1], spot, args.size,
                    args.min_edge, sigma,
                    require_implied=not args.allow_unvalidated_vol,
                )
                if sig:
                    found.append(sig)

        if not args.no_endgame and model_ok:
            # Measured sigma, deliberately: see scan_endgame. Feeding it the
            # implied value would make its edge identically negative.
            sig = scan_endgame(
                market, book, spot, args.size, sigma,
                max_seconds_left=args.endgame_window,
                min_z=args.endgame_z,
            )
            if sig:
                found.append(sig)

        for sig in found:
            self._emit(sig)

    def reset_for_live(self) -> None:
        """Drop paper-phase signal state at the moment of promotion.

        Signals queued during warm-up are stale by the time the gates pass, and
        the ledger's once-per-market dedupe would otherwise block any market
        that signalled on paper from ever trading live.
        """
        self._pending_orders.clear()
        self._candidates.clear()
        if self.ledger is not None:
            self.ledger.reset_dedupe()

    def _emit(self, sig) -> None:
        """Hold a signal until it has been confirmed over time, then queue it.

        One evaluation pass is one glance at one book snapshot: a stale poll, a
        fleeting quote, or a single bad tick all look identical to a real edge
        for 200ms. Requiring the same signal to recur across passes spanning
        `--confirm-seconds` filters those out. The trade-off is honest and
        deliberate: a real edge that vanishes inside the window was never
        capturable at our polling cadence anyway - the book updates at
        `--book-interval`, so an edge we cannot see twice is an edge we would
        have been filled on late or not at all.

        The queued signal is the LATEST sighting, so execution prices at the
        current ask rather than the one from the start of the window.
        """
        if sig is None:
            return
        self._strategy_hits[sig.strategy] = self._strategy_hits.get(sig.strategy, 0) + 1
        now = time.monotonic()
        need_s = self._args.confirm_seconds
        need_n = max(self._args.confirm_passes, 1)

        if need_s > 0.0 or need_n > 1:
            key = (sig.strategy, sig.ticker, tuple(leg.side for leg in sig.legs))
            cand = self._candidates.get(key)
            # A gap longer than the window means the edge closed and reopened;
            # that is a new candidate, not a continuation of the old one.
            if cand is None or now - cand["last"] > max(need_s, 2.0):
                self._candidates[key] = {"first": now, "last": now, "passes": 1}
                log.info(
                    "candidate [%s] %s: confirming over %.1fs (%d passes)...",
                    sig.strategy, sig.ticker, need_s, need_n,
                )
                return
            cand["last"] = now
            cand["passes"] += 1
            if now - cand["first"] < need_s or cand["passes"] < need_n:
                return
            held = now - cand["first"]
            sig.note = f"{sig.note} | confirmed over {held:.1f}s / {cand['passes']} passes"

        if self.ledger is not None and self.ledger.record(sig):
            self._pending_orders.append(sig)
            log.warning(
                "\n---- PAPER TRADE ----\n %s\n %s\n"
                " fair(YES) %.4f | risk $%.2f | recorded for settlement scoring\n"
                "---------------------",
                sig.describe(), sig.note, sig.fair_yes, sig.max_loss,
            )

    async def _execution_loop(self) -> None:
        """Turn recorded signals into orders, and settle finished markets."""
        while True:
            try:
                trader = self.trader
                while self._pending_orders and trader is not None:
                    sig = self._pending_orders.pop(0)
                    if trader.check_halt():
                        continue
                    if not trader.side_mapping_verified:
                        # One 1-contract NO order, read back as a position,
                        # before any real size: getting the YES-book inversion
                        # backwards would take the opposite of every trade.
                        if not await trader.verify_side_mapping(sig.ticker):
                            trader.halted = True
                            trader.halt_reason = "side mapping verification failed"
                            log.error("TRADING HALTED: %s", trader.halt_reason)
                            continue
                    for leg in sig.legs:
                        count = trader.size_for(leg.price)
                        if count < 1:
                            log.warning(
                                " execution: SKIPPED %s %s - no legal size at %.4f "
                                "(max stake $%.2f)",
                                sig.strategy, leg.side, leg.price, trader.max_stake(),
                            )
                            continue
                        result = await trader.place(sig.ticker, leg.side, leg.price, count)
                        log.warning(" execution: [%s] %s", sig.strategy, result.summary())
                await self._settle_finished()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.exception("Execution loop error: %s", exc)
            await asyncio.sleep(1.0)

    async def _settle_finished(self) -> None:
        """Book the real outcome of any market we traded that has now settled."""
        trader = self.trader
        if trader is None:
            return
        traded = {o.ticker for o in trader._orders if o.ok} - self._settled
        for ticker in traded:
            try:
                payload = await self._client_ref._get(f"/markets/{ticker}")
            except Exception:  # noqa: BLE001 - not settled yet is normal
                continue
            result = str(((payload or {}).get("market") or {}).get("result", "")).lower()
            if result in ("yes", "no"):
                trader.settle(ticker, result)
                self._settled.add(ticker)

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self._args.heartbeat)
            market, book, tick = self._market, self._book, self._buffer.last()
            if market is None:
                log.info("hb | no live %s market", self._args.series)
                continue
            sigma = self._buffer.sigma_per_sqrt_second()
            fair = market.fair_value(tick.price, sigma) if tick else None
            implied = (
                implied_sigma(market, book, tick.price) if tick and book else None
            )
            ratio = vol_agreement(sigma, implied)
            log.info(
                "hb | %s | spot=%s strike=%s | fair=%s | yes %s/%s | "
                "sigma %.2f vs mkt %s bps/s (%s) | %.0fs left | "
                "obs=%d signals=%d best_net=%+.4f",
                market.ticker.split("-")[-2] if "-" in market.ticker else market.ticker,
                f"${tick.price:,.0f}" if tick else "n/a",
                f"${market.strike:,.0f}" if market.strike_known else "PENDING",
                f"{fair:.3f}" if fair is not None else "n/a",
                f"{book.yes_bid:.3f}" if book and book.yes_bid else "-",
                f"{book.yes_ask:.3f}" if book and book.yes_ask else "-",
                sigma * 10_000.0,
                f"{implied * 10_000.0:.2f}" if implied else "n/a",
                f"{ratio:.2f}x" if ratio else "unvalidated",
                market.seconds_remaining(),
                self._observations,
                self._signals,
                self._best_net,
            )
            if not market.strike_known:
                log.info("     (floor_strike has not posted yet - nothing model-based is priced)")
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    p.add_argument("--live", action="store_true",
                   help="PLACE REAL ORDERS WITH REAL MONEY. Off by default. Prompts "
                        "for and verifies Kalshi credentials on startup if .env "
                        "does not already hold a working pair.")
    p.add_argument("--confirm-seconds", type=float, default=3.0,
                   help="an edge must persist this long, re-seen across passes, "
                        "before it is recorded or traded. 0 disables. One glance at "
                        "one book snapshot is not a verified edge.")
    p.add_argument("--confirm-passes", type=int, default=3,
                   help="minimum number of sightings inside --confirm-seconds")
    p.add_argument("--max-stake-pct", type=float, default=8.0,
                   help="percent of starting balance staked per trade")
    p.add_argument("--max-exposure-pct", type=float, default=25.0,
                   help="percent of balance open across all positions at once")
    p.add_argument("--daily-loss-pct", type=float, default=20.0,
                   help="session loss that halts trading, percent of balance")
    p.add_argument("--max-trades", type=int, default=40,
                   help="hard cap on orders in one session")
    p.add_argument("--min-profit", type=float, default=0.01,
                   help="CROSS: minimum locked dollar profit per pair")
    p.add_argument("--anchor-age", type=float, default=20.0,
                   help="STALE: how far back the market anchor is taken, seconds")
    p.add_argument("--endgame-window", type=float, default=120.0,
                   help="ENDGAME: only consider markets closing within this many seconds")
    p.add_argument("--endgame-z", type=float, default=3.0,
                   help="ENDGAME: sigmas from the strike required to call it decided")
    p.add_argument("--vol-ratio-max", type=float, default=1.50,
                   help="how far our measured volatility may sit from the market's "
                        "implied volatility before model-based signals are suppressed. "
                        "The strike is published and the clock is public, so a sigma "
                        "disagreement is the only thing the absolute model can turn "
                        "into an 'edge' - and ours is the side more likely wrong.")
    p.add_argument("--allow-unvalidated-vol", action="store_true",
                   help="report and trade model-based signals even when our volatility "
                        "cannot be checked against the market's. Every phantom edge this "
                        "project has found came from exactly that. Diagnostics only.")
    p.add_argument("--no-cross", action="store_true", help="disable the CROSS strategy")
    p.add_argument("--no-stale", action="store_true", help="disable the STALE strategy")
    p.add_argument("--no-endgame", action="store_true", help="disable the ENDGAME strategy")
    p.add_argument("--no-basis", action="store_true",
                   help="do not correct the tape toward the USD composite. The raw "
                        "venue carries a persistent premium/discount worth several "
                        "points of probability on a 15-minute contract.")
    p.add_argument("--log-dir", default="logs",
                   help="directory for per-run log files (L_MMDDYY_HHMMSS.log)")
    p.add_argument("--no-log", action="store_true", help="console only, write no log file")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

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

    if args.live:
        # Interactive, synchronous, and BEFORE any market task exists: prompt
        # for keys if needed, prove them with a signed balance call, have the
        # human recognise the balance, and only then persist to .env. A live
        # session must never start on unverified credentials.
        from kalshi_setup import ensure_credentials

        if ensure_credentials(args.env_file) is None:
            log.error("Credential setup failed; not starting a live session.")
            return

    monitor = Monitor(args)

    run_log = None
    if not args.no_log:
        run_log = start_run_log(
            log,
            directory=args.log_dir,
            title=(
                "KALSHI MONITOR (LIVE - REAL ORDERS ENABLED)"
                if args.live
                else "KALSHI MONITOR (DRY RUN, NO ORDERS)"
            ),
            context=[
                f"series  : {args.series}",
                f"feed    : {'Binance BTCUSDT (USDT - MISPRICED)' if args.binance else 'Coinbase BTC-USD (USD, BRTI constituent)'}",
                f"min edge: {args.min_edge:+.4f}/contract, net of Kalshi taker fees",
            ],
        )
    monitor.run_log = run_log
    if not args.no_log:
        stem = (run_log.path.stem if run_log else run_log_name().replace(".log", ""))
        monitor.ledger = PaperLedger(Path(args.log_dir) / f"paper_{stem}.jsonl")
        log.info("Recording paper trades to %s", monitor.ledger.path)

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
