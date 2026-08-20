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
    ReplayLog,
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
    ASSETS,
    BTC_15M_SERIES,
    Asset,
    CoinbaseSpotStream,
    CompositeBasis,
    asset_for,
    usd_spot_rest,
    KalshiAuthError,
    KalshiBook,
    KalshiBookStream,
    KalshiClient,
    KalshiCredentials,
    KalshiMarket,
    breakeven_fair_value,
    fee_at_size,
    fee_per_contract,
    quantize_kalshi_price,
    net_edge,
    trading_fee,
)


#: Why a trade was refused on data quality alone. Named rather than boolean so
#: a session log can be asked which failure is actually costing opportunities.
STALE_SPOT = "STALE_SPOT"
STALE_BOOK = "STALE_BOOK"
BOOK_SEQUENCE_GAP = "BOOK_SEQUENCE_GAP"
EXCESSIVE_TIME_SKEW = "EXCESSIVE_TIME_SKEW"
STALE_REFERENCE = "STALE_REFERENCE"


def freshness_problem(
    inst, now_mono: float, max_spot_age: float, max_book_age: float,
    max_skew: float, max_reference_age: float,
) -> str | None:
    """Is the information this decision rests on new enough to act on?

    Connectivity is not freshness. A websocket can be open while its last
    message is ten seconds old, and a REST book can be "current" in the sense
    that it arrived without being current in the sense that matters. For a
    strategy whose entire claim is that it sees a move before the book does,
    acting on old data is worse than not acting: the move it is reacting to has
    already been priced.

    Returns the reason, or None when the data is good enough to trade on.
    """
    tick = inst.buffer.last()
    if tick is None or now_mono - tick.mono > max_spot_age:
        return STALE_SPOT

    book = inst.book
    if book is None or (book.received_mono and book.age > max_book_age):
        return STALE_BOOK

    # A book the stream has stopped maintaining - because a sequence number was
    # missed, or the socket dropped - is not stale, it is ORPHANED. It carries a
    # recent timestamp and will pass every age check while describing a market
    # that has moved on, which is the one failure mode a freshness test cannot
    # otherwise see.
    gap_mono = getattr(inst, "book_gap_mono", 0.0)
    if gap_mono and book.received_mono and book.received_mono <= gap_mono:
        return BOOK_SEQUENCE_GAP

    # The two feeds must describe the same moment. A fresh book compared against
    # a fresh spot is still useless if they are seconds apart from each other.
    if book.received_mono and abs(tick.mono - book.received_mono) > max_skew:
        return EXCESSIVE_TIME_SKEW

    basis = inst.basis
    if basis is not None and max_reference_age > 0.0:
        age = basis.age
        if age is not None and age > max_reference_age:
            return STALE_REFERENCE
    return None


def _reference_line(basis) -> str:
    """How much the USD venues disagreed, and how much of that is our own lag.

    This is the number the settlement model uses as the size of our reference
    error, so it is worth saying out loud rather than leaving inside an object.
    """
    if basis is None:
        return "none (uncorrected venue tape)"
    if not basis.venues:
        return "no composite formed - fewer than two venues agreed"
    detail = ", ".join(
        f"{name} ${price:,.0f}" for name, (price, _) in sorted(basis.venue_prices.items())
    )
    return (
        f"{basis.venues} venues spread ${basis.dispersion:,.2f} "
        f"(+/-${basis.dispersion / 2.0:,.2f} assumed vs BRTI), "
        f"replies within {basis.venue_skew:.1f}s | {detail}"
    )


class StrategyStats:
    """Live performance of ONE strategy on ONE underlying.

    The session summary reported strategy hits and a single global funnel, which
    cannot answer the question that actually decides what to keep: is CROSS on
    ETH paying for itself while STALE on BTC is not? Three strategies times two
    instruments blended into one number can show a profit while five of the six
    combinations lose money.

    Slippage is tracked separately from edge on purpose. A strategy can be right
    about direction on every trade and still lose, if the price it pays is
    consistently worse than the quote that justified the trade - and that shows
    up here as a healthy predicted edge against a negative realized one.
    """

    __slots__ = (
        "attempted", "filled", "rejected", "skipped", "contracts",
        "slippage_cents", "slippage_n", "fill_ms", "exit_ms",
        "predicted", "realized", "closed", "settled", "wins",
    )

    def __init__(self) -> None:
        self.attempted = 0
        self.filled = 0
        self.rejected = 0
        self.skipped = 0
        self.contracts = 0.0
        self.slippage_cents = 0.0
        self.slippage_n = 0
        self.fill_ms: list[float] = []
        self.exit_ms: list[float] = []
        self.predicted = 0.0   # dollars of edge the model claimed
        self.realized = 0.0    # dollars actually booked (exits + settlements)
        self.closed = 0
        self.settled = 0
        self.wins = 0

    @property
    def fill_rate(self) -> float:
        return self.filled / self.attempted if self.attempted else 0.0

    @property
    def avg_slippage(self) -> float:
        """Cents per contract paid above the quote that produced the signal."""
        return self.slippage_cents / self.slippage_n if self.slippage_n else 0.0

    def line(self) -> str:
        def med(values):
            if not values:
                return "n/a"
            ordered = sorted(values)
            return f"{ordered[len(ordered) // 2] / 1000.0:.1f}s"

        parts = [
            f"attempted {self.attempted}",
            f"filled {self.filled} ({self.fill_rate * 100:.0f}%)",
            f"{self.contracts:g} contracts",
            f"slip {self.avg_slippage:+.2f}c/contract",
            f"fill {med(self.fill_ms)}",
        ]
        if self.closed:
            parts.append(f"exit {med(self.exit_ms)} ({self.closed} closed early)")
        if self.settled or self.closed:
            graded = self.settled + self.closed
            parts.append(f"won {self.wins}/{graded}")
        parts.append(f"predicted ${self.predicted:+.2f} vs realized ${self.realized:+.2f}")
        return " | ".join(parts)


class _ExitStub:
    """Minimal shape `PaperLedger.record_execution` needs for an exit row."""

    __slots__ = ("strategy", "ticker")

    def __init__(self, strategy: str, ticker: str) -> None:
        self.strategy = strategy
        self.ticker = ticker


class Instrument:
    """One underlying, its own tape, and the contract currently open on it.

    Everything an instrument needs to be priced lives here, so adding a second
    one cannot accidentally share the first one's spot feed. Risk deliberately
    does NOT live here: the account is shared, so stake caps, exposure and the
    loss breakers stay on the single trader. Two instruments each believing
    they owned the whole balance would silently double the intended risk.
    """

    def __init__(self, asset: Asset) -> None:
        self.asset = asset
        self.buffer = PriceBuffer()
        self.basis: CompositeBasis | None = None
        self.stream = None
        self.market: KalshiMarket | None = None
        self.book: KalshiBook | None = None
        #: (monotonic, spot, market mid, effective tau) - tau is needed because
        #: the anchor's probability was measured with more time on the clock.
        self.anchors: deque[tuple[float, float, float, float]] = deque(maxlen=600)
        # Per-instrument diagnostics, so the summary can say which underlying
        # the numbers came from rather than blending two different markets.
        self.observations = 0
        self.signals = 0
        self.markets_seen: set[str] = set()
        self.net_edges: list[float] = []
        self.net_edges_validated: list[float] = []
        self.vol_ratios: list[float] = []
        self.sigma_pairs: list[tuple[float, float]] = []
        self.no_strike_obs = 0
        self.unvalidated_vol_obs = 0
        self.vol_gated_obs = 0
        self.fallback_vol_obs = 0
        self.signal_by_market: dict[str, int] = {}
        self.last_eval_mono = 0.0
        self.edge_seconds = 0.0
        #: Data-quality refusals by reason, so the summary can say which feed
        #: problem is actually costing trades.
        self.stale_rejections: dict[str, int] = {}
        #: Which path last supplied the book, and how often each did. A run that
        #: silently fell back to REST polling all session looks identical in the
        #: log to one on a healthy stream unless this is counted.
        self.book_source = "none"
        self.ws_books = 0
        self.rest_books = 0
        #: Times the REST resync disagreed with the streamed top of book. A few
        #: are normal (the two reads are seconds apart); a mismatch on nearly
        #: every resync means our delta application is wrong.
        self.book_mismatches = 0
        self.book_resyncs = 0
        #: When the stream last discarded its books. Anything we are still
        #: holding from before this moment is orphaned, not merely old.
        self.book_gap_mono = 0.0

    @property
    def name(self) -> str:
        return self.asset.name

    @property
    def series(self) -> str:
        return self.asset.series


class Monitor:
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        # Kalshi settles on CF Benchmarks BRTI, a USD index. Binance quotes in
        # USDT, and that basis is as large as the signal, so the reference tape
        # must be USD-quoted. --binance exists only to demonstrate the error.
        self.instruments = [
            Instrument(asset_for(name)) for name in _requested_assets(args)
        ]
        self._best_net = -1.0
        self._last_signal_mono = 0.0
        self._start_mono = 0.0
        self.run_log = None
        self.ledger: PaperLedger | None = None
        self._strategy_hits: dict[str, int] = {}
        self.trader: KalshiTrader | None = None
        self._settled: set[str] = set()
        self._pending_orders: list = []
        self._client_ref = None
        # Confirmation state: an edge must be re-seen across passes spanning
        # `--confirm-seconds` before it is recorded or traded. Key is
        # (strategy, ticker, leg sides); value tracks first/last sighting.
        self._candidates: dict[tuple, dict] = {}
        #: Sides already committed per market, so two strategies cannot take
        #: opposite ends of the same contract. See _conflicts().
        self._committed: dict[str, set[str]] = {}
        #: Conflicts already reported, so a persisting signal is logged once.
        self._conflicts_seen: set[tuple] = set()
        #: Set when a signal is queued, so execution runs immediately instead of
        #: waiting out its poll. A confirmed signal used to sit up to a full
        #: second before its order was sent - on a book that had just moved 6
        #: bps, which is what created the signal in the first place.
        self._work = asyncio.Event()
        #: Where signals die. Optimising anything requires knowing which stage
        #: is losing them: a session with 47 sightings and 0 fills was one
        #: execution bug, but nothing in the summary said which stage failed.
        self._funnel: dict[str, int] = {
            "sighted": 0, "confirmed": 0, "conflicted": 0,
            "attempted": 0, "filled": 0, "unpriceable": 0,
        }
        #: Websocket order book, when the session can sign for one. None means
        #: every book in this run came from REST polling.
        self._book_stream: KalshiBookStream | None = None
        #: "ASSET:STRATEGY" -> StrategyStats. Which combination is actually
        #: paying, rather than whether the blend of all of them is.
        self._strategy_stats: dict[str, StrategyStats] = {}
        #: ticker -> monotonic time of the fill, so time-to-exit is measurable.
        self._entry_mono: dict[tuple[str, str], float] = {}
        #: Structured replay record. None when recording is off.
        self._replay: ReplayLog | None = None

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
                    take_profit_multiple=self._args.take_profit,
                    exit_at_fair_value=not self._args.no_fair_exit,
                    stop_loss_fraction=self._args.stop_loss,
                    min_seconds_to_exit=self._args.min_exit_seconds,
                ),
                dry_run=not self._args.live,
            )
            if not await self.trader.arm():
                log.error("Execution layer failed to arm; stopping before any market data")
                return
            for inst in self.instruments:
                if self._args.binance:
                    # BTC only, and wrong on purpose - see --binance.
                    inst.stream = BinanceTradeStream(inst.buffer, BINANCE_WS_FALLBACKS)
                    continue
                if not self._args.no_basis:
                    inst.basis = CompositeBasis(
                        session, sources=inst.asset.composite, label=inst.name
                    )
                inst.stream = CoinbaseSpotStream(
                    inst.buffer,
                    product=inst.asset.coinbase_product,
                    basis=inst.basis,
                )

            self._book_stream = self._make_book_stream(client)

            try:
                status = await client.exchange_status()
                log.info(
                    "Kalshi trading_active=%s | instruments: %s",
                    (status or {}).get("trading_active"),
                    ", ".join(
                        f"{i.name} ({i.series} <- "
                        + ("Binance BTCUSDT - MISPRICED"
                           if self._args.binance else i.asset.coinbase_product)
                        + ")"
                        for i in self.instruments
                    ),
                )
                for inst in self.instruments:
                    rest = await usd_spot_rest(session, inst.asset.rest)
                    if rest:
                        log.info(
                            "%s USD spot cross-check (REST): $%s",
                            inst.name, f"{rest:,.2f}",
                        )
            except Exception as exc:  # noqa: BLE001
                log.error("Cannot reach Kalshi: %s", exc)
                return

            tasks = [
                asyncio.create_task(self._execution_loop(), name="execution"),
                asyncio.create_task(self._heartbeat_loop(), name="heartbeat"),
            ]
            if self._book_stream is not None:
                tasks.append(asyncio.create_task(
                    self._book_stream.run(), name="book-stream"
                ))
            for inst in self.instruments:
                tasks += [
                    asyncio.create_task(inst.stream.run(), name=f"spot-{inst.name}"),
                    asyncio.create_task(
                        self._discovery_loop(client, inst), name=f"disc-{inst.name}"
                    ),
                    asyncio.create_task(
                        self._book_loop(client, inst), name=f"book-{inst.name}"
                    ),
                    asyncio.create_task(self._eval_loop(inst), name=f"eval-{inst.name}"),
                ]
                if inst.basis is not None:
                    tasks.append(asyncio.create_task(
                        inst.basis.run(inst.buffer), name=f"basis-{inst.name}"
                    ))
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    # -- loops -------------------------------------------------------------- #

    async def _discovery_loop(self, client: KalshiClient, inst: Instrument) -> None:
        while True:
            try:
                live = await client.live_markets(inst.series)
                now = time.time()
                current = next(
                    (m for m in live if m.is_live(now) and m.seconds_remaining(now) > 5), None
                )
                if current and (inst.market is None or current.ticker != inst.market.ticker):
                    log.info(
                        "[%s] Tracking %s | strike %s | closes in %.0fs | vol %s",
                        inst.name,
                        current.ticker,
                        f"${current.strike:,.2f}",
                        current.seconds_remaining(now),
                        f"{current.volume:,.0f}",
                    )
                    inst.market = current
                    inst.book = None
                elif current:
                    # Keep quotes and status fresh on the tracked contract.
                    current_book = inst.book
                    inst.market = current
                    inst.book = current_book
                elif inst.market is not None and not inst.market.is_live(now):
                    log.info(
                        "[%s] %s closed; waiting for the next window",
                        inst.name, inst.market.ticker,
                    )
                    inst.market = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("[%s] Discovery failed: %s", inst.name, exc)
            await asyncio.sleep(self._args.discovery_interval)

    def _make_book_stream(self, client: KalshiClient) -> KalshiBookStream | None:
        """The websocket book, when we can sign for it.

        Kalshi authenticates the socket itself even for market data, so a paper
        run with no keys configured simply does not get one and keeps polling.
        Signing costs nothing and sends nothing on its own, so a DRY run WITH
        keys gets the same book quality as a live one - otherwise every measured
        result on paper would come from a slower book than the one that will be
        traded against, which is the wrong direction for a dry run to be wrong.
        """
        if self._args.no_book_stream:
            return None
        try:
            if not client.authenticated:
                client.authenticate()
            signer = client.signer
            if signer is None:
                raise KalshiAuthError("no signer available")
            return KalshiBookStream(
                signer,
                on_book=self._on_stream_book,
                on_reset=self._on_stream_reset,
            )
        except Exception as exc:  # noqa: BLE001 - REST still works
            log.info(
                "Book stream unavailable (%s); polling the book over REST", exc
            )
            return None

    def _on_stream_book(self, ticker: str, book: KalshiBook) -> None:
        """Push a streamed book straight onto its instrument.

        Deliberately not routed through the poll loop: the whole point of the
        stream is that the book reaches the evaluator when the venue changes
        it, not when our timer next fires.
        """
        for inst in self.instruments:
            if inst.market is not None and inst.market.ticker == ticker:
                inst.book = book
                inst.book_source = "ws"
                inst.ws_books += 1
                return

    def _on_stream_reset(self) -> None:
        """The stream dropped its books; ours are orphaned from this moment on.

        We do not clear `inst.book` here. The REST loop will replace it within a
        poll, and a None book and an orphaned book are different problems worth
        telling apart in the refusal counters.
        """
        now = time.monotonic()
        for inst in self.instruments:
            if inst.book_source == "ws":
                inst.book_gap_mono = now

    async def _book_loop(self, client: KalshiClient, inst: Instrument) -> None:
        """REST book: bootstrap, fallback, and periodic resync.

        The stream owns the fast path. This loop exists so that (a) there is a
        book before the first snapshot arrives, (b) a dropped stream degrades to
        the old behaviour instead of trading on nothing, and (c) the streamed
        book gets checked against the venue's own view often enough that a bug
        in delta application shows up as a logged mismatch rather than as a
        series of confusing fills.
        """
        last_resync = 0.0
        while True:
            market = inst.market
            if market is not None:
                stream = self._book_stream
                if stream is not None:
                    stream.track(*[
                        i.market.ticker for i in self.instruments
                        if i.market is not None
                    ])
                streamed = stream.book(market.ticker) if stream else None
                fresh = (
                    streamed is not None
                    and (streamed.age or 0.0) <= self._args.max_book_age
                )
                now = time.monotonic()
                due = now - last_resync >= self._args.book_resync
                if not fresh or due:
                    try:
                        rest = await client.book(market.ticker)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        log.warning("[%s] Book fetch failed: %s", inst.name, exc)
                    else:
                        if fresh and due:
                            last_resync = now
                            inst.book_resyncs += 1
                            self._check_resync(inst, streamed, rest)
                        else:
                            inst.book = rest
                            inst.book_source = "rest"
                            inst.rest_books += 1
            await asyncio.sleep(self._args.book_interval)

    def _check_resync(
        self, inst: Instrument, streamed: KalshiBook, rest: KalshiBook
    ) -> None:
        """Compare the streamed top of book against the venue's own REST view.

        A mismatch is not corrected here on purpose. The two reads are seconds
        apart and the book genuinely moves between them, so overwriting on every
        difference would just reintroduce the stale REST quote. What matters is
        that a systematic error becomes visible - a delta applied to the wrong
        side or the wrong sign shows up as a mismatch on essentially every
        resync, which no amount of normal book movement produces.
        """
        def close(a, b):
            if a is None or b is None:
                return a is None and b is None
            return abs(a - b) <= 0.01

        if close(streamed.yes_bid, rest.yes_bid) and close(streamed.no_bid, rest.no_bid):
            return
        inst.book_mismatches += 1
        log.warning(
            "[%s] streamed book disagrees with REST: yes_bid %s vs %s, "
            "no_bid %s vs %s (%d mismatches in %d resyncs)",
            inst.name, streamed.yes_bid, rest.yes_bid,
            streamed.no_bid, rest.no_bid,
            inst.book_mismatches, inst.book_resyncs,
        )

    async def _eval_loop(self, inst: Instrument) -> None:
        while True:
            with contextlib.suppress(Exception):
                self._evaluate(inst)
            await asyncio.sleep(self._args.eval_interval)

    def _evaluate(self, inst: Instrument) -> None:
        market, book = inst.market, inst.book
        tick = inst.buffer.last()
        if market is None or book is None or tick is None or not inst.buffer.ready:
            return
        remaining = market.seconds_remaining()
        if remaining < self._args.min_seconds_left:
            return

        now_mono = time.monotonic()
        sigma = inst.buffer.sigma_per_sqrt_second()
        inst.markets_seen.add(market.ticker)

        # Kalshi lists the contract before `floor_strike` posts, so the first
        # ~30-45s of every window has no strike. Nothing model-based can be
        # priced there; CROSS can, because it never looks at one.
        if not market.strike_known:
            inst.no_strike_obs += 1
            if not self._args.no_cross:
                self._emit(
                    scan_cross(market, book, self._args.size, self._args.min_profit), inst
                )
            return

        inst.observations += 1
        if not inst.buffer.vol_is_measured:
            inst.fallback_vol_obs += 1

        stale = freshness_problem(
            inst, now_mono,
            self._args.max_spot_age, self._args.max_book_age,
            self._args.max_feed_skew, self._args.max_reference_age,
        )
        if stale is not None:
            inst.stale_rejections[stale] = inst.stale_rejections.get(stale, 0) + 1
            # Recorded too. A replay that only holds the passes we acted on
            # cannot answer "why did nothing happen for twenty minutes?", which
            # is the question a refusal counter raises and cannot settle.
            self._replay_observe(inst, market, book, tick, sigma, refused=stale)
            return

        # The one number that decides whether any of this is real. See
        # strategies.vol_agreement.
        sigma_implied = implied_sigma(market, book, tick.price)
        ratio = vol_agreement(sigma, sigma_implied)
        if sigma_implied is not None:
            inst.sigma_pairs.append((sigma, sigma_implied))
        if ratio is None:
            inst.unvalidated_vol_obs += 1
        else:
            inst.vol_ratios.append(ratio)
        vol_ok = ratio is not None and ratio <= self._args.vol_ratio_max
        if not vol_ok and not self._args.allow_unvalidated_vol:
            inst.vol_gated_obs += 1

        realized = self._realized_twap(inst, market, now_mono)
        fair = market.fair_value(
            tick.price, sigma,
            realized=realized,
            reference_error=self._reference_error(inst),
        )
        best_this_pass = -1.0
        self._replay_observe(
            inst, market, book, tick, sigma,
            sigma_implied=sigma_implied, vol_ratio=ratio, vol_ok=vol_ok,
            fair=fair, realized=realized,
        )

        # Keep a rolling (time, spot, market mid) anchor for the STALE model.
        mid = book.yes_mid
        if mid is not None:
            inst.anchors.append(
                (now_mono, tick.price, mid, market.effective_tau())
            )
        self._run_strategies(inst, market, book, tick.price, sigma, now_mono, vol_ok)

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

            move = inst.buffer.largest_move(3.0)
            inst.signals += 1
            inst.signal_by_market[market.ticker] = (
                inst.signal_by_market.get(market.ticker, 0) + 1
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
            inst.net_edges.append(best_this_pass)
            if vol_ok:
                inst.net_edges_validated.append(best_this_pass)
            now_mono = time.monotonic()
            if inst.last_eval_mono and best_this_pass >= self._args.min_edge and vol_ok:
                # Wall time spent with a tradeable edge on the screen - a far
                # more useful figure than a raw signal count, because it says
                # how long the window actually stays open.
                inst.edge_seconds += min(now_mono - inst.last_eval_mono, 5.0)
            inst.last_eval_mono = now_mono

    def build_summary(self) -> list[str]:
        """The numbers worth reading after a long run, per instrument."""
        elapsed = max(time.monotonic() - (self._start_mono or time.monotonic()), 1e-9)
        lines = [
            f"instruments   : {', '.join(f'{i.name} ({i.series})' for i in self.instruments)}",
            f"reference     : {'Binance BTCUSDT (USDT - MISPRICED)' if self._args.binance else 'Coinbase USD spot (BRTI constituent)'}",
            f"min net edge  : {self._args.min_edge:+.4f}/contract",
            f"confirmation  : {self._args.confirm_seconds:.1f}s / "
            f"{self._args.confirm_passes} passes, book polled every "
            f"{self._args.book_interval:.2f}s",
            f"order book    : {self._book_stream_line()}",
            f"preset        : {'AGGRESSIVE' if getattr(self._args, 'aggressive', False) else 'standard'}",
            f"paper trades  : {self.ledger.count if self.ledger else 0} "
            f"(score with: python kalshi_score.py)",
            f"strategy hits : {self._strategy_hits or 'none'}",
            "signal funnel : "
            + " -> ".join(
                f"{k} {v}" for k, v in self._funnel.items() if k != "unpriceable"
            )
            + (f"  (+{self._funnel['unpriceable']} unpriceable)"
               if self._funnel["unpriceable"] else ""),
            f"execution     : {self.trader.stats() if self.trader else 'none'}",
        ]
        if self._replay is not None:
            lines.append(
                f"replay record : {self._replay.rows:,} rows "
                f"({self._replay.dropped:,} thinned) -> {self._replay.path.name}"
            )
        lines += self._strategy_lines()
        for inst in self.instruments:
            lines += [""] + self._instrument_lines(inst, elapsed)
        return lines

    def _strategy_lines(self) -> list[str]:
        """Per-strategy, per-instrument record - six experiments, not one.

        A blended number can show a profit while most of the combinations that
        produced it lose money, and the decision this feeds is which strategy to
        keep running.
        """
        if not self._strategy_stats:
            return []
        out = ["", " per-strategy record (asset:strategy):"]
        for key, stats in sorted(self._strategy_stats.items()):
            out.append(f"   {key:<16} {stats.line()}")
        return out

    def _book_stream_line(self) -> str:
        """Where the books actually came from - not where they were meant to.

        A stream that failed to connect, or dropped an hour in, leaves a run
        that looks entirely normal in the log while pricing off a book up to a
        second old. Saying so in the summary is the difference between reading
        a result and guessing at one.
        """
        stream = self._book_stream
        ws = sum(i.ws_books for i in self.instruments)
        rest = sum(i.rest_books for i in self.instruments)
        if stream is None:
            reason = "disabled" if self._args.no_book_stream else "unavailable"
            return f"REST polling only ({reason}); {rest:,} fetches"
        share = 100.0 * ws / max(ws + rest, 1)
        bad = sum(i.book_mismatches for i in self.instruments)
        resyncs = sum(i.book_resyncs for i in self.instruments)
        return (
            f"websocket {'connected' if stream.connected else 'DISCONNECTED'} | "
            f"{ws:,} pushed / {rest:,} polled ({share:.0f}% streamed) | "
            f"{stream.snapshots} snapshots, {stream.gaps} gaps, "
            f"{stream.reconnects} reconnects | "
            f"resync {bad}/{resyncs} mismatched"
        )

    def _instrument_lines(self, inst: Instrument, elapsed: float) -> list[str]:
        edges = inst.net_edges
        validated = inst.net_edges_validated
        bar = "-" * 62
        lines = [
            bar,
            f" {inst.name}  ({inst.series} priced off {inst.asset.coinbase_product})",
            bar,
            f"   feed basis    : {('%+.2f USD (n=%d polls)' % (inst.basis.offset, inst.basis.samples)) if inst.basis else 'not corrected'}",
            f"   reference     : {_reference_line(inst.basis)}",
            f"   markets seen  : {len(inst.markets_seen)}",
            f"   observations  : {inst.observations:,}",
            f"   signals       : {inst.signals}",
        ]
        if inst.stale_rejections:
            total = sum(inst.stale_rejections.values())
            lines.append(
                f"   data refusals : {total:,} ("
                + ", ".join(f"{k} {v:,}" for k, v in
                            sorted(inst.stale_rejections.items(), key=lambda kv: -kv[1]))
                + ")"
            )
        if inst.no_strike_obs:
            lines.append(
                f"   no-strike     : {inst.no_strike_obs:,} passes skipped while "
                f"floor_strike had not posted"
            )
        if inst.fallback_vol_obs:
            share = inst.fallback_vol_obs / max(inst.observations, 1) * 100.0
            lines += [
                f"   WARNING: {inst.fallback_vol_obs:,} of {inst.observations:,} "
                f"observations ({share:.0f}%) priced with ASSUMED",
                "            volatility (45% annualized), not measured from tape.",
            ]

        lines += self._vol_lines(inst)

        if edges:
            lines += [
                "",
                "   net edge per observation (after fees), best of YES/NO.",
                "   RAW is every observation; VALIDATED is the subset where our",
                "   volatility agreed with the market's - only those can be real.",
                f"                       {'RAW':>10}  {'VALIDATED':>10}",
                f"      observations  {len(edges):>10,}  {len(validated):>10,}",
                f"      p50           {_fmt_edge(_percentile(edges, 50)):>10}  "
                f"{_fmt_edge(_percentile(validated, 50)):>10}",
                f"      p90           {_fmt_edge(_percentile(edges, 90)):>10}  "
                f"{_fmt_edge(_percentile(validated, 90)):>10}",
                f"      p99           {_fmt_edge(_percentile(edges, 99)):>10}  "
                f"{_fmt_edge(_percentile(validated, 99)):>10}",
                f"      max           {_fmt_edge(max(edges)):>10}  "
                f"{_fmt_edge(max(validated)) if validated else 'n/a':>10}",
                "",
                f"   validated tradeable edge on screen : {inst.edge_seconds:.0f}s of "
                f"{elapsed:.0f}s ({inst.edge_seconds / elapsed * 100.0:.2f}%)",
            ]
            raw_pos = sum(1 for e in edges if e >= self._args.min_edge)
            val_pos = sum(1 for e in validated if e >= self._args.min_edge)
            lines.append(
                f"   at or above the threshold : {raw_pos:,} raw "
                f"({raw_pos / len(edges) * 100.0:.2f}%), {val_pos:,} validated "
                f"({val_pos / max(len(edges), 1) * 100.0:.2f}% of all)"
            )
        else:
            lines.append("   no observations recorded - the market or tape never came up")

        if inst.signal_by_market:
            lines.append("   signals by market:")
            for ticker, count in sorted(
                inst.signal_by_market.items(), key=lambda kv: -kv[1]
            ):
                lines.append(f"      {count:4d}  {ticker}")

        if edges and max(edges) >= self._args.min_edge:
            lines += [
                "   NOTE: a large, persistent edge on a liquid market usually means",
                "         a model or feed error, not free money. Read the VALIDATED",
                "         column and the volatility block above before believing it.",
            ]
        return lines

    def _vol_lines(self, inst: "Instrument") -> list[str]:
        """Report our volatility against the market's - the decisive comparison.

        With the strike published and the clock public, our only conceivable
        advantage over the book is the spot price, which the book sees at least
        as fast. So on the absolute model, an "edge" and a sigma disagreement
        are the same event described two ways. Printing the two side by side is
        what makes that visible instead of flattering.
        """
        pairs = inst.sigma_pairs
        ratios = inst.vol_ratios
        total = max(inst.observations, 1)
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
            f"   unvalidated passes {inst.unvalidated_vol_obs:,} of {total:,} "
            f"({inst.unvalidated_vol_obs / total * 100.0:.0f}%) - quote too close to the money",
            f"   suppressed passes  {inst.vol_gated_obs:,} of {total:,} "
            f"({inst.vol_gated_obs / total * 100.0:.0f}%) - above the "
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

    def _run_strategies(self, inst, market, book, spot, sigma, now_mono, vol_ok: bool) -> None:
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
            for ts, price, mid, atau in inst.anchors:
                if ts <= cutoff:
                    anchor = (price, mid, atau)
                else:
                    break
            if anchor:
                # STALE takes sigma from the quote itself, so it needs no
                # agreement check - it is already using the market's number.
                sig = scan_stale(
                    market, book, anchor[0], anchor[1], spot, args.size,
                    args.min_edge, sigma,
                    require_implied=not args.allow_unvalidated_vol,
                    min_move_bps=args.stale_min_move,
                    max_vol_ratio=(
                        0.0 if args.allow_unvalidated_vol else args.vol_ratio_max
                    ),
                    max_edge=args.max_edge,
                    anchor_tau=anchor[2],
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
                require_implied=not args.allow_unvalidated_vol,
                realized=self._realized_twap(inst, market, now_mono),
                reference_error=self._reference_error(inst),
            )
            if sig:
                found.append(sig)

        for sig in found:
            self._emit(sig, inst)

    @property
    def replay(self) -> ReplayLog | None:
        return self._replay

    def attach_replay(self, path, interval: float) -> None:
        """Start recording the structured replay stream to `path`."""
        self._replay = ReplayLog(path, interval)
        log.info(
            "Recording replay observations to %s (every %.1fs per market)",
            self._replay.path, interval,
        )

    def close_replay(self) -> None:
        if self._replay is not None:
            self._replay.close()

    def _replay_observe(
        self, inst, market, book, tick, sigma: float,
        refused: str | None = None,
        sigma_implied: float | None = None,
        vol_ratio: float | None = None,
        vol_ok: bool | None = None,
        fair: float | None = None,
        realized: tuple[float, float] | None = None,
    ) -> None:
        """One replay row: everything the decision rested on, flat and named.

        Wrapped in a blanket suppress on purpose. This is a recorder; if it
        throws - a odd field, a full disk, a renamed attribute - the run must
        carry on trading and simply lose the row.
        """
        replay = self._replay
        if replay is None:
            return
        with contextlib.suppress(Exception):
            basis = inst.basis
            replay.observe(
                market.ticker,
                asset=inst.name,
                refused=refused,
                spot=round(tick.price, 4),
                spot_age=round(time.monotonic() - tick.mono, 3),
                strike=market.strike,
                seconds_left=round(market.seconds_remaining(), 2),
                tau_eff=round(market.effective_tau(), 2),
                yes_bid=book.yes_bid, yes_ask=book.yes_ask,
                no_bid=book.no_bid, no_ask=book.no_ask,
                yes_bid_size=book.yes_bid_size, yes_ask_size=book.yes_ask_size,
                # Enough depth to replay a size decision, not the whole ladder.
                yes_levels=list(book.yes_levels[-5:]),
                no_levels=list(book.no_levels[-5:]),
                book_age=round(book.age, 3) if book.received_mono else None,
                book_version=book.version, book_seq=book.seq,
                book_source=inst.book_source,
                sigma=sigma, sigma_implied=sigma_implied,
                vol_ratio=vol_ratio, vol_ok=vol_ok,
                vol_measured=inst.buffer.vol_is_measured,
                fair=fair,
                realized_mean=realized[0] if realized else None,
                realized_covered=realized[1] if realized else None,
                basis_offset=getattr(basis, "offset", None),
                basis_dispersion=getattr(basis, "dispersion", None),
                basis_venues=getattr(basis, "venues", None),
                basis_age=getattr(basis, "age", None),
            )

    def _replay_event(self, kind: str, **fields) -> None:
        replay = self._replay
        if replay is None:
            return
        with contextlib.suppress(Exception):
            replay.event(kind, **fields)

    def _realized_twap(self, inst, market, now_mono: float):
        """What our tape has printed since this contract's averaging window opened.

        Returns None outside the window, or whenever the buffer cannot speak for
        it - the model then falls back to the terminal one on its own.
        """
        remaining = market.seconds_remaining()
        lookback = market.twap_lookback
        if lookback <= 0.0 or remaining >= lookback:
            return None  # window has not opened yet
        # The window opened `lookback - remaining` seconds ago in wall time,
        # which is the same number of seconds ago on the monotonic clock the
        # buffer is indexed by.
        opened_mono = now_mono - (lookback - max(remaining, 0.0))
        return inst.buffer.mean_since(opened_mono, now_mono)

    def _reference_error(self, inst) -> float:
        """1-sigma dollars our USD composite may sit away from the settlement index.

        Half the observed cross-venue range is a crude proxy, but it is a
        MEASURED one that widens exactly when the venues stop agreeing, which is
        when our reference deserves less trust. With no composite yet we pass 0
        rather than inventing a number - the caller is already gated on the
        reference being fresh.
        """
        basis = getattr(inst, "basis", None)
        dispersion = getattr(basis, "dispersion", 0.0) or 0.0
        return max(dispersion, 0.0) / 2.0

    def _conflicts(self, sig) -> str | None:
        """Reject a signal that opposes a position we have already taken here.

        A live session bought ENDGAME NO at 0.975 and, 32 seconds later, STALE
        YES at 0.820 on the SAME contract. That is not a hedge: settlement pays
        exactly one of them, so the pair is a guaranteed loss of both fees plus
        the gap between the two prices, and it means two of our own strategies
        flatly disagreed about the outcome while we funded both opinions.

        When they disagree the honest move is to hold the first view or none,
        not to buy both. CROSS is exempt because taking both sides IS its
        thesis - it only fires when the pair costs less than the dollar it pays.
        """
        if sig.strategy == "CROSS":
            return None
        held = self._committed.get(sig.ticker)
        if not held:
            return None
        for leg in sig.legs:
            opposite = "NO" if leg.side == "YES" else "YES"
            if opposite in held:
                return (
                    f"already committed {opposite} on {sig.ticker}; refusing the "
                    f"opposing {leg.side}"
                )
        return None

    def _marketable_limit(self, sig, leg, fair_leg: float) -> float | None:
        """The most we may pay and still clear `--min-edge`, snapped to a tick.

        Bidding exactly the ask we saw is why almost nothing filled: by the time
        an order lands the ask has moved, and an IOC at a stale price takes
        nothing. The probe fills every single time precisely because it crosses
        hard (it bids 0.999), and the strategy orders never did because they
        crossed by zero.

        We are a taker by construction - the whole thesis is that a quote is
        about to move - so the limit should be the highest price that still
        leaves the edge worth having, not the price that happened to be on the
        screen. Capped by `--max-slippage` so a thin book cannot walk us up, and
        it returns None rather than paying more than the edge is worth.
        """
        # Two different questions, and conflating them is why nothing filled.
        # `--min-edge` decides whether a signal is worth taking AT ALL, and it
        # was already applied against the quoted ask when the signal was made.
        # How far we may then cross is a separate question: a fill that keeps
        # only part of the edge still beats no fill. With 1c ticks these compete
        # directly - fair 0.8683 against a 0.84 ask leaves a 0.8483 ceiling
        # under the old rule, which floors straight back to 0.84 and crosses
        # nothing.
        ceiling = fair_leg - fee_per_contract(leg.price) - self._args.min_fill_edge
        limit = min(leg.price + self._args.max_slippage, ceiling)
        if limit < leg.price:
            # Even the quoted ask no longer clears the bar.
            return None
        inst = next(
            (i for i in self.instruments
             if i.market is not None and i.market.ticker == sig.ticker),
            None,
        )
        market = inst.market if inst is not None else None
        ranges = market.price_ranges if market is not None else ()
        # Round DOWN: rounding a buy up could push us past the ceiling.
        snapped = quantize_kalshi_price(limit, ranges, buy=False)
        return snapped if snapped >= leg.price else leg.price

    def executable_size(self, sig, leg, fair_leg: float, wanted: int) -> int:
        """Largest quantity whose VWAP still clears the edge, given real depth.

        Top-of-book is what a 1-lot pays. A book offering

            2 @ 0.63   1 @ 0.64   20 @ 0.67

        fills 20 contracts at a 0.667 average, not 0.63 - and an edge computed
        against 0.63 can be entirely spent walking the ladder. Shrinking the
        order until its own average price still clears `--min-fill-edge` keeps
        the trade honest instead of quietly turning a good signal into a bad
        fill. Never increases size beyond what risk already allowed.
        """
        inst = next(
            (i for i in self.instruments
             if i.market is not None and i.market.ticker == sig.ticker),
            None,
        )
        book = inst.book if inst is not None else None
        if book is None or not (book.yes_levels or book.no_levels):
            return wanted  # no depth visible; nothing to refine
        for count in range(wanted, 0, -1):
            quote = book.cost_for(leg.side, count)
            if quote is None:
                continue
            vwap, available = quote
            if available < count - 1e-9:
                continue
            if fair_leg - vwap - fee_at_size(vwap, count) >= self._args.min_fill_edge:
                if count < wanted:
                    log.info(
                        " execution: [%s] %s trimmed %d -> %d contracts; depth "
                        "puts the average at %.4f, not %.4f",
                        sig.strategy, sig.ticker, wanted, count, vwap, leg.price,
                    )
                return count
        return 0

    def _stats_key(self, item) -> str:
        """The "ASSET:STRATEGY" bucket a signal or a filled order belongs to.

        Keyed off the ticker so a signal and the order it produced always land
        in the same bucket - the two objects share no other field that survives
        the round trip through the venue.
        """
        ticker = getattr(item, "ticker", "") or ""
        asset = next(
            (i.name for i in self.instruments if ticker.startswith(i.series)), ""
        )
        strategy = getattr(item, "strategy", "") or "?"
        return f"{asset}:{strategy}" if asset else strategy

    def _stats_for(self, sig) -> StrategyStats:
        key = self._stats_key(sig)
        stats = self._strategy_stats.get(key)
        if stats is None:
            stats = self._strategy_stats[key] = StrategyStats()
        return stats

    def _track_strategy(self, sig, outcome: str, leg, result, elapsed_ms: float) -> None:
        """Fold one execution attempt into its strategy's running record."""
        stats = self._stats_for(sig)
        stats.attempted += 1
        if outcome in ("filled", "simulated"):
            stats.filled += 1
            stats.contracts += result.count
            stats.fill_ms.append(elapsed_ms)
            if result.price and leg.price:
                # Positive means we paid MORE than the quote that justified the
                # trade. Signed, because a limit that improves is real too.
                stats.slippage_cents += (result.price - leg.price) * 100.0 * result.count
                stats.slippage_n += result.count
            # The model's claim, scaled to what actually filled rather than to
            # the nominal size the signal was written against.
            nominal = sum(l.size for l in sig.legs) or 1.0
            stats.predicted += sig.expected_net * (result.count / nominal)
            self._entry_mono[(sig.ticker, leg.side)] = time.monotonic()
        elif outcome == "rejected":
            stats.rejected += 1
        else:
            stats.skipped += 1

    def _track_close(self, order, price: float, realized: float) -> None:
        """Book an early exit against its strategy."""
        stats = self._strategy_stats.get(self._stats_key(order))
        if stats is None:
            return
        stats.closed += 1
        stats.realized += realized
        if realized > 0:
            stats.wins += 1
        entered = self._entry_mono.pop((order.ticker, order.outcome), None)
        if entered is not None:
            stats.exit_ms.append((time.monotonic() - entered) * 1000.0)

    def _record_execution(self, sig, outcome: str, **kw) -> None:
        """Write the companion execution row, so nothing downstream can read a
        recorded signal as money that moved."""
        if self.ledger is not None:
            with contextlib.suppress(Exception):
                self.ledger.record_execution(sig, outcome, **kw)

    def reset_for_live(self) -> None:
        """Drop paper-phase signal state at the moment of promotion.

        Signals queued during warm-up are stale by the time the gates pass, and
        the ledger's once-per-market dedupe would otherwise block any market
        that signalled on paper from ever trading live.
        """
        self._pending_orders.clear()
        self._candidates.clear()
        self._committed.clear()
        self._conflicts_seen.clear()
        if self.ledger is not None:
            self.ledger.reset_dedupe()

    def _emit(self, sig, inst=None) -> None:
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
        # Tag with the underlying and the preset, so a graded ledger can answer
        # "did ETH pay?" and "did aggressive settings pay?" separately rather
        # than blending two experiments into one unreadable number.
        if inst is not None and not sig.note.startswith("["):
            sig.note = f"[{inst.name}] {sig.note}"
        key = f"{inst.name}:{sig.strategy}" if inst is not None else sig.strategy
        self._strategy_hits[key] = self._strategy_hits.get(key, 0) + 1
        self._funnel["sighted"] += 1
        now = time.monotonic()
        need_s = self._args.confirm_seconds
        need_n = max(self._args.confirm_passes, 1)

        if need_s > 0.0 or need_n > 1:
            ckey = (sig.strategy, sig.ticker, tuple(leg.side for leg in sig.legs))
            cand = self._candidates.get(ckey)
            # A gap longer than the window means the edge closed and reopened;
            # that is a new candidate, not a continuation of the old one.
            if cand is None or now - cand["last"] > max(need_s, 2.0):
                self._candidates[ckey] = {
                    "first": now, "last": now, "passes": 1,
                    "version": getattr(sig, "book_version", None),
                }
                log.info(
                    "candidate [%s] %s: confirming over %.1fs (%d passes)...",
                    sig.strategy, sig.ticker, need_s, need_n,
                )
                return
            cand["last"] = now
            # A confirmation must require NEW market information. The evaluator
            # runs every --eval-interval (0.1s) while the book refreshes every
            # --book-interval (0.4s), so counting passes counted the SAME
            # snapshot up to four times - a live log read "confirmed over 1.0s /
            # 11 passes" on roughly two distinct books. Versions make the
            # requirement honest: three confirmations means three books.
            version = getattr(sig, "book_version", None)
            if version is not None and version != cand.get("version"):
                cand["version"] = version
                cand["passes"] += 1
            elif version is None:
                cand["passes"] += 1  # no version available; fall back to passes
            if now - cand["first"] < need_s or cand["passes"] < need_n:
                return
            held = now - cand["first"]
            sig.note = (
                f"{sig.note} | confirmed over {held:.1f}s / "
                f"{cand['passes']} distinct books"
            )

        ckey_conflict = (sig.strategy, sig.ticker,
                         tuple(leg.side for leg in sig.legs))
        conflict = self._conflicts(sig)
        if conflict is not None:
            # Once per (strategy, market, side): a persisting signal is
            # re-evaluated several times a second, and logging each one wrote
            # 548 identical lines in a single session.
            if ckey_conflict not in self._conflicts_seen:
                self._conflicts_seen.add(ckey_conflict)
                self._funnel["conflicted"] += 1
                log.warning(" conflict: [%s] %s", sig.strategy, conflict)
                self._record_execution(sig, "skipped", detail=conflict)
            return

        if self.ledger is not None and self.ledger.record(sig):
            for leg in sig.legs:
                self._committed.setdefault(sig.ticker, set()).add(leg.side)
            self._pending_orders.append(sig)
            self._funnel["confirmed"] += 1
            self._work.set()
            self._replay_event(
                "signal", ticker=sig.ticker, strategy=sig.strategy,
                legs=[{"side": l.side, "price": l.price, "size": l.size}
                      for l in sig.legs],
                fair_yes=sig.fair_yes, expected_net=sig.expected_net,
                spot=sig.spot, strike=sig.strike,
                seconds_left=round(sig.seconds_left, 1),
                sigma_used=sig.sigma_used, book_version=sig.book_version,
                note=sig.note,
            )
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
                        # Halted is terminal for the session. Draining the queue
                        # one signal at a time logged 548 identical skip lines
                        # in one run, which buried everything else in the log.
                        if self._pending_orders:
                            log.warning(
                                " execution: halted (%s) - discarding %d queued "
                                "signal(s); no further orders this session",
                                trader.halt_reason, len(self._pending_orders) + 1,
                            )
                        self._record_execution(
                            sig, "skipped", detail=f"halted: {trader.halt_reason}"
                        )
                        self._pending_orders.clear()
                        break
                    if not trader.side_mapping_verified:
                        # One 1-contract NO order, read back as a position,
                        # before any real size: getting the YES-book inversion
                        # backwards would take the opposite of every trade.
                        verdict = await trader.verify_side_mapping(sig.ticker)
                        if verdict is False:
                            # Either genuinely reversed or out of attempts;
                            # verify_side_mapping has already set the reason.
                            if not trader.halted:
                                trader.halted = True
                                trader.halt_reason = "side mapping REVERSED on the venue"
                                log.error("TRADING HALTED: %s", trader.halt_reason)
                            self._record_execution(
                                sig, "skipped", detail=trader.halt_reason
                            )
                            continue
                        if verdict is not True:
                            # Inconclusive (no fill / unreadable position) is
                            # not evidence. Skip this signal, try on the next.
                            log.warning(
                                " execution: skipping %s - side mapping probe "
                                "inconclusive, will retry", sig.ticker,
                            )
                            self._record_execution(
                                sig, "skipped",
                                detail="side mapping probe inconclusive",
                            )
                            continue
                    for leg in sig.legs:
                        count = trader.size_for(leg.price)
                        if count < 1:
                            log.warning(
                                " execution: SKIPPED %s %s - no legal size at %.4f "
                                "(max stake $%.2f)",
                                sig.strategy, leg.side, leg.price, trader.max_stake(),
                            )
                            self._record_execution(
                                sig, "skipped",
                                detail=f"no legal size at {leg.price:.4f}",
                            )
                            continue
                        # fair_yes is the model's YES probability; the exit
                        # target for a NO leg is its complement.
                        fair_leg = (
                            sig.fair_yes if leg.side == "YES"
                            else 1.0 - sig.fair_yes
                        )
                        limit = self._marketable_limit(sig, leg, fair_leg)
                        if limit is None:
                            log.info(
                                " execution: [%s] %s no longer worth taking once "
                                "crossing costs are allowed for", sig.strategy,
                                sig.ticker,
                            )
                            self._record_execution(
                                sig, "skipped",
                                detail="no price leaves the required edge",
                            )
                            self._funnel["unpriceable"] += 1
                            continue
                        count = trader.size_for(limit)
                        if count < 1:
                            self._record_execution(
                                sig, "skipped",
                                detail=f"no legal size at {limit:.4f}",
                            )
                            continue
                        # Risk says how much we MAY buy; depth says how much is
                        # still worth buying. Take the smaller.
                        count = self.executable_size(sig, leg, fair_leg, count)
                        if count < 1:
                            self._record_execution(
                                sig, "skipped",
                                detail="no size clears the edge once depth is walked",
                            )
                            self._funnel["unpriceable"] += 1
                            continue
                        self._funnel["attempted"] += 1
                        result = await trader.place(
                            sig.ticker, leg.side, limit, count,
                            strategy=sig.strategy, entry_fair=fair_leg,
                        )
                        if result.ok:
                            self._funnel["filled"] += 1
                        log.warning(" execution: [%s] %s", sig.strategy, result.summary())
                        outcome = (
                            "simulated" if result.dry_run else
                            ("filled" if result.ok else "rejected")
                        )
                        elapsed_ms = max(time.time() - sig.ts, 0.0) * 1000.0
                        self._record_execution(
                            sig, outcome,
                            count=result.count if result.ok else 0.0,
                            price=result.price,
                            detail=result.error or "",
                            signal_price=leg.price,
                            elapsed_ms=elapsed_ms,
                        )
                        self._track_strategy(
                            sig, outcome, leg, result, elapsed_ms
                        )
                        self._replay_event(
                            "order", ticker=sig.ticker, strategy=sig.strategy,
                            side=leg.side, outcome=outcome,
                            signal_price=leg.price, limit=limit,
                            fill_price=result.price, count=result.count,
                            requested=count, elapsed_ms=round(elapsed_ms, 1),
                            error=result.error or "",
                        )
                await self._manage_exits()
                await self._settle_finished()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.exception("Execution loop error: %s", exc)
            # Wake instantly on a new signal; otherwise tick for exits/settles.
            self._work.clear()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._work.wait(), timeout=1.0)

    async def _manage_exits(self) -> None:
        """Mark open positions against the live book and take profits.

        The reason this belongs in the loop rather than at settlement: STALE
        buys because the book has NOT yet repriced a spot move. The moment it
        reprices, the thesis has played out and the edge is realized - whatever
        happens to BTC afterwards is a directional bet nobody chose to make. A
        position bought at 0.32 on a model value of 0.45 has no thesis left at
        0.64, and every loss this project has taken came from holding one of
        those into settlement.
        """
        trader = self.trader
        if trader is None or trader.limits.take_profit_multiple <= 0:
            return
        # Books are per-instrument, so a position can only be marked against
        # the instrument that owns its ticker.
        for inst in self.instruments:
            market, book = inst.market, inst.book
            if market is None or book is None:
                continue
            left = market.seconds_remaining()
            for order in trader.open_positions(market.ticker):
                decision = trader.exit_reason(order, book, left)
                if decision is None:
                    continue
                reason, price = decision
                # close_position is the only thing that moves `realized` in
                # this call, so the delta across it IS this exit's P&L - more
                # reliable than recomputing it from a partially filled result.
                before = trader.realized
                await trader.close_position(order, price, reason)
                booked = trader.realized - before
                self._track_close(order, price, booked)
                self._replay_event(
                    "exit", ticker=order.ticker, strategy=order.strategy,
                    side=order.outcome, reason=reason, entry=order.price,
                    exit=price, count=order.count, realized=round(booked, 4),
                    seconds_left=round(left, 1),
                )
                self._committed.get(order.ticker, set()).discard(order.outcome)
                if self.ledger is not None:
                    with contextlib.suppress(Exception):
                        self.ledger.record_execution(
                            _ExitStub(order.strategy, order.ticker),
                            "closed",
                            count=order.count,
                            price=price,
                            detail=f"{reason} at {price:.4f} from {order.price:.4f}",
                            signal_price=order.price,
                        )

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
                # Attribute each settlement to the strategy that opened it,
                # before settle() marks the orders closed and the link is gone.
                held = [
                    o for o in trader.open_positions(ticker)
                ]
                before = trader.realized
                trader.settle(ticker, result)
                booked = trader.realized - before
                self._attribute_settlement(held, result, booked)
                self._replay_event(
                    "settlement", ticker=ticker, result=result,
                    realized=round(booked, 4),
                    positions=[
                        {"strategy": o.strategy, "side": o.outcome,
                         "count": o.count, "price": o.price} for o in held
                    ],
                )
                self._settled.add(ticker)

    def _attribute_settlement(self, held: list, result: str, booked: float) -> None:
        """Split a settled market's P&L across the strategies that opened it.

        Two strategies can hold the same contract, so the total is apportioned
        by stake rather than credited whole to whichever one is found first.
        """
        if not held:
            return
        total_stake = sum(o.stake for o in held) or 1.0
        for order in held:
            stats = self._strategy_stats.get(self._stats_key(order))
            if stats is None:
                continue
            stats.settled += 1
            stats.realized += booked * (order.stake / total_stake)
            if (order.outcome == "YES") == (result == "yes"):
                stats.wins += 1
            self._entry_mono.pop((order.ticker, order.outcome), None)

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self._args.heartbeat)
            for inst in self.instruments:
                self._heartbeat(inst)

    def _heartbeat(self, inst: Instrument) -> None:
        market, book, tick = inst.market, inst.book, inst.buffer.last()
        if market is None:
            log.info("hb | [%s] no live %s market", inst.name, inst.series)
            return
        sigma = inst.buffer.sigma_per_sqrt_second()
        fair = market.fair_value(
            tick.price, sigma,
            realized=self._realized_twap(inst, market, time.monotonic()),
            reference_error=self._reference_error(inst),
        ) if tick else None
        implied = implied_sigma(market, book, tick.price) if tick and book else None
        ratio = vol_agreement(sigma, implied)
        log.info(
            "hb | [%s] %s | spot=%s strike=%s | fair=%s | yes %s/%s | "
            "sigma %.2f vs mkt %s bps/s (%s) | %.0fs left | "
            "obs=%d signals=%d best_net=%+.4f",
            inst.name,
            market.ticker.split("-")[-2] if "-" in market.ticker else market.ticker,
            f"${tick.price:,.2f}" if tick else "n/a",
            f"${market.strike:,.2f}" if market.strike_known else "PENDING",
            f"{fair:.3f}" if fair is not None else "n/a",
            f"{book.yes_bid:.3f}" if book and book.yes_bid else "-",
            f"{book.yes_ask:.3f}" if book and book.yes_ask else "-",
            sigma * 10_000.0,
            f"{implied * 10_000.0:.2f}" if implied else "n/a",
            f"{ratio:.2f}x" if ratio else "unvalidated",
            market.seconds_remaining(),
            inst.observations,
            inst.signals,
            self._best_net,
        )
        if not market.strike_known:
            log.info("     (floor_strike has not posted yet - nothing model-based is priced)")
        if not inst.buffer.vol_is_measured:
            log.info("     (sigma is still the assumed prior, not measured from tape)")


#: What --aggressive changes, and nothing else changes.
#:
#: Every entry here is a threshold on OPPORTUNITY - how sure, how cheap, how
#: long-lived a setup must be before it is taken. None of them is a threshold
#: on TRUTH. The correctness gates (--vol-ratio-max, the strike guard, the
#: side-mapping probe, the loss breakers, --stale-min-move's purpose) stay
#: exactly where they are, because each one was added after a graded session
#: lost money, and loosening them would not be aggression but amnesia.
#:
#: The confirmation and book-poll changes are the load-bearing ones. A logged
#: session (L_081626_181730) found a real, priced, edge-positive ENDGAME setup
#: that lasted about two seconds against a three-second confirmation window,
#: so it was never taken. Confirming in 1s only helps if the book underneath
#: is fresher than 1s, which is why both move together.
AGGRESSIVE_PRESET: dict[str, float] = {
    "confirm_seconds": 1.0,
    "book_interval": 0.4,
    "eval_interval": 0.1,
    "endgame_z": 2.5,
    "min_edge": 0.01,
    "stale_min_move": 6.0,
    "max_stake_pct": 12.0,
    "cooldown": 2.0,
}


def _explicitly_set(parser: argparse.ArgumentParser, argv: list[str] | None) -> set[str]:
    """Which options the caller actually typed, as dest names.

    Comparing a parsed value against the parser default cannot tell "unset"
    from "set to the default value", so `--aggressive --endgame-z 3.0` would
    silently become 2.5. Re-parsing with every default suppressed leaves only
    the options that were genuinely supplied.
    """
    saved = {id(a): a.default for a in parser._actions}
    try:
        for action in parser._actions:
            action.default = argparse.SUPPRESS
        seen, _ = parser.parse_known_args(argv)
        return set(vars(seen))
    except SystemExit:  # a malformed argv is the real parse's problem, not ours
        return set()
    finally:
        for action in parser._actions:
            action.default = saved[id(action)]


def apply_aggressive(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    argv: list[str] | None = None,
) -> None:
    """Apply the preset, but never over an explicit choice.

    A value the caller typed outranks the preset: `--aggressive --endgame-z 3.0`
    must mean 3.0 even though 3.0 is also the default.
    """
    if not getattr(args, "aggressive", False):
        return
    explicit = _explicitly_set(parser, argv)
    changed = []
    for name, value in AGGRESSIVE_PRESET.items():
        if not hasattr(args, name) or name in explicit:
            continue  # explicitly set on the command line - leave it alone
        setattr(args, name, value)
        changed.append(f"{name.replace('_', '-')}={value:g}")
    log.warning(
        "AGGRESSIVE preset: %s\n"
        "         (correctness gates unchanged: vol-ratio-max=%.2f, strike guard, "
        "side-mapping probe, loss breakers)",
        ", ".join(changed) or "nothing (all set explicitly)",
        args.vol_ratio_max,
    )


def _requested_assets(args: argparse.Namespace) -> list[str]:
    """Which underlyings this run should trade.

    `--assets` wins; otherwise a `--series` that names a known asset selects
    that one, so the old single-series invocation keeps working unchanged.
    """
    raw = getattr(args, "assets", None)
    if raw:
        names = [a.strip().upper() for a in str(raw).split(",") if a.strip()]
    else:
        names = [asset_for(getattr(args, "series", None) or BTC_15M_SERIES).name]
    seen: list[str] = []
    for name in names:
        asset_for(name)  # raises on an underlying with no configured feed
        if name not in seen:
            seen.append(name)
    return seen


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
    p.add_argument("--series", default=BTC_15M_SERIES,
                   help="Kalshi series ticker (single-asset shorthand; --assets wins)")
    p.add_argument("--assets", default=None,
                   help=f"comma-separated underlyings to trade concurrently, e.g. "
                        f"'BTC,ETH'. Known: {', '.join(sorted(ASSETS))}. Each gets its "
                        f"own spot feed and basis correction; they share one account, "
                        f"so the risk limits apply across all of them together.")
    p.add_argument("--aggressive", action="store_true",
                   help="loosen the OPPORTUNITY thresholds (faster confirmation, "
                        "faster book polling, lower endgame-z and min-edge, bigger "
                        "stake). Deliberately does NOT touch the correctness gates - "
                        "sigma agreement, the strike guard, the side-mapping check "
                        "or the loss breakers - because those are what caught every "
                        "phantom edge this project has produced.")
    p.add_argument("--eval-interval", type=float, default=0.2,
                   help="seconds between evaluation passes per instrument")
    p.add_argument("--min-edge", type=float, default=0.02,
                   help="minimum NET edge per contract (after fees) to report")
    p.add_argument("--spike-bps", type=float, default=12.0, help="spot move of interest, bps")
    p.add_argument("--size", type=float, default=20.0, help="contracts, for the dollar figures")
    p.add_argument("--min-seconds-left", type=float, default=20.0,
                   help="ignore a market closer than this to expiry")
    p.add_argument("--book-interval", type=float, default=1.0, help="book poll seconds")
    p.add_argument("--no-book-stream", action="store_true",
                   help="do not open the Kalshi websocket book; poll over REST only. "
                        "The stream is the current book by construction, so this is a "
                        "diagnostic switch, not a tuning knob.")
    p.add_argument("--no-replay", action="store_true",
                   help="do not write the structured replay log. The analyzer falls "
                        "back to scraping heartbeat lines, which are 15s apart and "
                        "carry top of book only.")
    p.add_argument("--replay-interval", type=float, default=1.0,
                   help="seconds between replay observations PER MARKET. 0 records "
                        "every evaluation pass.")
    p.add_argument("--book-resync", type=float, default=30.0,
                   help="seconds between REST cross-checks of the streamed book. The "
                        "check only logs: a delta applied to the wrong side would "
                        "otherwise surface as inexplicable fills rather than an error.")
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
    p.add_argument("--take-profit", type=float, default=1.5,
                   help="sell once a position is worth this multiple of what it "
                        "cost, net of the fees on both sides. 0 holds every "
                        "position to settlement. STALE buys because the book has "
                        "not repriced yet; when it does, the thesis has played out "
                        "and holding on is a directional bet nobody chose.")
    p.add_argument("--no-fair-exit", action="store_true",
                   help="do not exit when the book reprices to the fair value the "
                        "signal was based on. That target is the most faithful exit "
                        "there is - STALE bought because the book had not caught up, "
                        "so when it does the thesis is complete by definition.")
    p.add_argument("--stop-loss", type=float, default=0.0,
                   help="sell if a position falls to this fraction of its cost "
                        "(e.g. 0.4). 0 disables - on a cheap contract the mark is "
                        "noisy and a stop mostly pays the spread to exit trades "
                        "that would have recovered.")
    p.add_argument("--min-exit-seconds", type=float, default=45.0,
                   help="never try to exit inside this many seconds of expiry; the "
                        "book thins to nothing there")
    p.add_argument("--min-profit", type=float, default=0.01,
                   help="CROSS: minimum locked dollar profit per pair")
    p.add_argument("--anchor-age", type=float, default=20.0,
                   help="STALE: how far back the market anchor is taken, seconds")
    p.add_argument("--min-fill-edge", type=float, default=0.005,
                   help="edge per contract that must SURVIVE crossing the spread. "
                        "Distinct from --min-edge, which decides whether a signal "
                        "is worth taking at the quoted ask; this decides how much "
                        "of that edge we may spend to actually get filled. On a 1c "
                        "tick ladder the two compete, and setting them equal "
                        "crosses zero ticks - which filled zero orders.")
    p.add_argument("--max-slippage", type=float, default=0.03,
                   help="how far above the quoted ask an entry may reach, in "
                        "dollars. We are a taker by construction - the thesis is "
                        "that the quote is about to move - so bidding exactly the "
                        "ask we saw fills only if the book stood still. A whole "
                        "session filled zero strategy orders that way. The limit "
                        "never exceeds the price that still clears --min-edge.")
    p.add_argument("--max-spot-age", type=float, default=5.0,
                   help="refuse to trade on a spot tick older than this (seconds)")
    p.add_argument("--max-book-age", type=float, default=3.0,
                   help="refuse to trade on a Kalshi book older than this")
    p.add_argument("--max-feed-skew", type=float, default=4.0,
                   help="refuse when the spot tick and the book describe moments "
                        "further apart than this. Two individually fresh feeds "
                        "can still be useless if they disagree about when 'now' is.")
    p.add_argument("--max-reference-age", type=float, default=180.0,
                   help="refuse when the USD composite basis has not updated in "
                        "this long; 0 disables")
    p.add_argument("--max-edge", type=float, default=0.35,
                   help="refuse any signal claiming more than this net edge per "
                        "contract. An edge this large on a liquid book is a model "
                        "error, not an opportunity - every one this project has "
                        "produced has been. 0 disables the cap.")
    p.add_argument("--stale-min-move", type=float, default=8.0,
                   help="STALE: minimum spot move vs the anchor, bps. Below this "
                        "the 'edge' is the book repricing on its own information "
                        "(which we should not fade) or amplified noise - a graded "
                        "session bought four of those and won one.")
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
    apply_aggressive(args, p, argv)

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
        if not args.no_replay:
            monitor.attach_replay(
                Path(args.log_dir) / f"replay_{stem}.jsonl", args.replay_interval
            )

    try:
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(amain(args, monitor))
    finally:
        # Written even on Ctrl+C, which is how a long monitoring run ends.
        monitor.close_replay()
        if run_log is not None:
            run_log.close(monitor.build_summary())
        log.info("Monitor stopped")


if __name__ == "__main__":
    main()
