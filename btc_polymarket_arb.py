#!/usr/bin/env python3
"""
Binance <-> Polymarket short-term BTC "Up or Down" dislocation scanner.

Strategy thesis
---------------
Polymarket runs rolling 5-minute and 15-minute "Bitcoin Up or Down" binaries.
Each contract resolves "Up" if the Chainlink BTC/USD TWAP over the tail of the
window is >= the reference price at the window open, else "Down".

Because the resolution reference is a spot BTC price, the fair value of the
contract is a deterministic function of (spot, strike, vol, time remaining).
Binance spot is the fastest venue in the complex; the Polymarket CLOB is a
human/bot-quoted order book that repriced on a lag. When spot dislocates hard
over a sub-3-second horizon, the book's resting asks are briefly *stale*: they
still reflect the pre-spike distribution. That staleness window is the edge.

This scanner detects those windows and logs them. It does NOT place orders.

Model
-----
We deliberately avoid trying to pin down the absolute strike (the window-open
reference price), because we usually attach to a market mid-window and cannot
observe the exact Chainlink reference the resolver will use. Instead we run a
*relative* (delta-shift) model, which is what the staleness trade actually
needs:

  1. Anchor on the last order book seen strictly BEFORE the spike began. Its
     mid `p0` embeds the market's own consensus about moneyness, so:

         z0 = Phi^-1(p0)          (implied standardized log-distance to strike)

  2. The spike moves log-spot by `d = ln(S_now / S_pre)`. Under a driftless
     GBM the standardized distance shifts by d / (sigma * sqrt(tau_eff)):

         z1 = z0 + d / (sigma * sqrt(tau_eff))
         p_fair = Phi(z1)

  3. Compare `p_fair` against the CURRENT best ask on the corresponding token.
     If the ask has not caught up, the window is open.

`tau_eff` carries a TWAP variance haircut. The contract settles on the average
of the price over the final L seconds, not the terminal point. For a Brownian
path the variance of that average, seen from now, is:

     Var = sigma^2 * (tau - L + L/3) = sigma^2 * (tau - 2L/3)

so `tau_eff = tau - 2L/3`. Averaging damps terminal variance, which makes the
contract slightly *more* sensitive to a given spot move than a naive terminal
model would suggest.

Volatility is estimated as an EWMA of 1-second log returns sampled from the
live trade stream (the 10-second buffer mandated for spike detection is far too
short to estimate sigma from, so a longer, separate return series is kept).

A model-free check runs alongside: if best_ask(Up) + best_ask(Down) < 1.0, both
legs can be lifted for a locked profit at settlement regardless of any vol
assumption. That is reported at higher severity.

Usage
-----
    pip install -r requirements.txt
    python btc_polymarket_arb.py
    python btc_polymarket_arb.py --spike-bps 12 --max-yes-ask 0.52 --min-edge 0.03

No API credentials are required: Gamma and the CLOB read endpoints are public.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import math
import signal
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, Sequence

import aiohttp

try:  # websockets >= 13 exposes the modern asyncio client here
    from websockets.asyncio.client import connect as ws_connect
except ImportError:  # pragma: no cover - older websockets
    from websockets.client import connect as ws_connect  # type: ignore[no-redef]

from websockets.exceptions import ConnectionClosed

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import BookParams

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"

#: Failover chain for the spot tape. The primary is the canonical endpoint;
#: `data-stream.binance.vision` is Binance's public market-data mirror carrying
#: the byte-identical `btcusdt@trade` payload, and it stays reachable from
#: regions where the main endpoint answers HTTP 451 ("restricted location").
#: The futures tape is last: it is a different contract and will basis-drift
#: from spot, but it is a usable proxy for pure spike *detection*.
BINANCE_WS_FALLBACKS = (
    BINANCE_WS_URL,
    "wss://data-stream.binance.vision/ws/btcusdt@trade",
    "wss://fstream.binance.com/ws/btcusdt@trade",
)

#: Consecutive failures on one endpoint before rotating to the next.
WS_FAILOVER_THRESHOLD = 2

GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
CLOB_HOST = "https://clob.polymarket.com"
POLYGON_CHAIN_ID = 137

#: Rolling spot buffer horizon, per spec.
PRICE_BUFFER_SECONDS = 10.0

#: Horizon over which a qualifying move must occur, per spec.
SPIKE_LOOKBACK_SECONDS = 3.0

#: Separate, longer return history used purely for the volatility estimate.
VOL_HISTORY_SECONDS = 300.0
VOL_EWMA_HALFLIFE_SECONDS = 120.0

#: Fallback BTC vol (annualized) used until enough 1s bars have accumulated.
FALLBACK_ANNUAL_VOL = 0.45
SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0

#: Polymarket prices live on a 1c grid; probabilities are clamped inside this
#: band before any Phi^-1 call so the inverse CDF cannot blow up.
PROB_EPS = 0.01

#: Rolling short-dated BTC series on Gamma. Window-open epoch is embedded in
#: the slug, which lets us address the live market directly instead of paging.
SERIES = {
    "5m": {"series_slug": "btc-up-or-down-5m", "slug_prefix": "btc-updown-5m", "period": 300},
    "15m": {"series_slug": "btc-up-or-down-15m", "slug_prefix": "btc-updown-15m", "period": 900},
}

log = logging.getLogger("btc-arb")


# --------------------------------------------------------------------------- #
# Math helpers
# --------------------------------------------------------------------------- #

_NORM = statistics.NormalDist()


def norm_cdf(x: float) -> float:
    return _NORM.cdf(x)


def norm_ppf(p: float) -> float:
    return _NORM.inv_cdf(min(max(p, PROB_EPS), 1.0 - PROB_EPS))


def clamp_prob(p: float) -> float:
    return min(max(p, PROB_EPS), 1.0 - PROB_EPS)


# --------------------------------------------------------------------------- #
# Spot tape
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Tick:
    mono: float  # local monotonic receive time - used for all windowing
    price: float
    exch_ms: int  # Binance trade time, kept for latency diagnostics


@dataclass(frozen=True, slots=True)
class Move:
    """A spot excursion measured inside the spike lookback window."""

    delta_log: float
    from_price: float
    to_price: float
    from_mono: float
    to_mono: float

    @property
    def bps(self) -> float:
        return self.delta_log * 10_000.0

    @property
    def elapsed(self) -> float:
        return self.to_mono - self.from_mono

    @property
    def direction(self) -> str:
        return "UP" if self.delta_log > 0 else "DOWN"


class PriceBuffer:
    """Rolling 10-second trade buffer plus a longer-horizon vol estimator.

    The 10s buffer satisfies the spike-detection requirement. Volatility needs
    a much longer sample than 10s to be meaningful, so 1-second sampled log
    returns are accumulated separately and EWMA-weighted.
    """

    def __init__(
        self,
        window: float = PRICE_BUFFER_SECONDS,
        vol_window: float = VOL_HISTORY_SECONDS,
        vol_halflife: float = VOL_EWMA_HALFLIFE_SECONDS,
    ) -> None:
        self._window = window
        self._ticks: Deque[Tick] = deque()
        self._vol_window = vol_window
        self._vol_decay = 0.5 ** (1.0 / max(vol_halflife, 1.0))
        self._bars: Deque[tuple[float, float]] = deque()  # (mono, price) 1s grid
        self._last_bar_mono: float | None = None
        self._tick_count = 0
        self._latency_ms = 0.0

    # -- ingest ------------------------------------------------------------- #

    def add(self, price: float, exch_ms: int) -> None:
        now = time.monotonic()
        self._ticks.append(Tick(now, price, exch_ms))
        self._tick_count += 1
        self._latency_ms = max(0.0, time.time() * 1000.0 - exch_ms)

        cutoff = now - self._window
        while self._ticks and self._ticks[0].mono < cutoff:
            self._ticks.popleft()

        # Sample onto a 1-second grid for the vol estimator.
        if self._last_bar_mono is None or now - self._last_bar_mono >= 1.0:
            self._last_bar_mono = now
            self._bars.append((now, price))
            bar_cutoff = now - self._vol_window
            while self._bars and self._bars[0][0] < bar_cutoff:
                self._bars.popleft()

    # -- state -------------------------------------------------------------- #

    @property
    def ready(self) -> bool:
        return len(self._ticks) >= 2

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def latency_ms(self) -> float:
        return self._latency_ms

    def last(self) -> Tick | None:
        return self._ticks[-1] if self._ticks else None

    # -- signals ------------------------------------------------------------ #

    def largest_move(self, lookback: float = SPIKE_LOOKBACK_SECONDS) -> Move | None:
        """Largest absolute log move ending at the latest tick, within `lookback`.

        Scanning every earlier tick in the window (rather than only the tick at
        exactly ``now - lookback``) means a 2.4s burst inside a 3s window is
        caught with its true magnitude, which is what "moved >X% in under Y
        seconds" actually asks for.
        """
        if len(self._ticks) < 2:
            return None

        latest = self._ticks[-1]
        floor_mono = latest.mono - lookback
        best: Move | None = None

        for tick in self._ticks:
            if tick.mono < floor_mono or tick is latest:
                continue
            if tick.price <= 0.0:
                continue
            delta = math.log(latest.price / tick.price)
            if best is None or abs(delta) > abs(best.delta_log):
                best = Move(delta, tick.price, latest.price, tick.mono, latest.mono)

        return best

    def sigma_per_sqrt_second(self) -> float:
        """EWMA realized vol of 1-second log returns, in per-sqrt-second units."""
        fallback = FALLBACK_ANNUAL_VOL / math.sqrt(SECONDS_PER_YEAR)
        if len(self._bars) < 30:
            return fallback

        rets: list[tuple[float, float]] = []  # (dt, log return)
        prev_mono, prev_price = self._bars[0]
        for mono, price in list(self._bars)[1:]:
            dt = mono - prev_mono
            if dt > 0 and price > 0 and prev_price > 0:
                rets.append((dt, math.log(price / prev_price)))
            prev_mono, prev_price = mono, price

        if len(rets) < 30:
            return fallback

        # Newest observation carries weight 1, decaying backwards.
        num = 0.0
        den = 0.0
        weight = 1.0
        for dt, ret in reversed(rets):
            num += weight * (ret * ret) / dt  # normalize to per-second variance
            den += weight
            weight *= self._vol_decay

        if den <= 0.0:
            return fallback
        sigma = math.sqrt(num / den)
        # Guard against a dead tape collapsing sigma to ~0 and exploding the
        # z-shift; floor at 20% of the long-run prior.
        return max(sigma, 0.2 * fallback)


# --------------------------------------------------------------------------- #
# Binance stream
# --------------------------------------------------------------------------- #


class BinanceTradeStream:
    """Resilient consumer of the Binance raw trade stream.

    Reconnects with exponential backoff and rotates through `endpoints` after
    repeated failures on the current one, so a geo-blocked or degraded host
    does not silently kill the tape (and with it every signal).
    """

    def __init__(
        self,
        buffer: PriceBuffer,
        endpoints: Sequence[str] = BINANCE_WS_FALLBACKS,
    ) -> None:
        self._buffer = buffer
        self._endpoints = list(endpoints) or [BINANCE_WS_URL]
        self._idx = 0
        self._consecutive_failures = 0
        self._connected = asyncio.Event()

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    @property
    def url(self) -> str:
        return self._endpoints[self._idx]

    async def run(self) -> None:
        backoff = 1.0
        while True:
            url = self.url
            try:
                async with ws_connect(
                    url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=5,
                    max_queue=1024,
                ) as ws:
                    log.info("Spot tape connected: %s", url)
                    self._connected.set()
                    self._consecutive_failures = 0
                    backoff = 1.0
                    async for raw in ws:
                        self._on_message(raw)
            except asyncio.CancelledError:
                raise
            except (ConnectionClosed, OSError) as exc:
                log.warning("Spot tape dropped from %s (%s)", url, exc)
                self._note_failure()
            except Exception as exc:  # noqa: BLE001 - keep the tape alive
                log.exception("Unexpected spot tape error on %s: %s", url, exc)
                self._note_failure()
            finally:
                self._connected.clear()

            log.info("Reconnecting to %s in %.1fs", self.url, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)

    def _note_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures < WS_FAILOVER_THRESHOLD or len(self._endpoints) < 2:
            return
        self._consecutive_failures = 0
        self._idx = (self._idx + 1) % len(self._endpoints)
        log.warning(
            "Failing over spot tape to %s "
            "(the canonical endpoint answers HTTP 451 from restricted regions)",
            self.url,
        )

    def _on_message(self, raw: str | bytes) -> None:
        try:
            msg = json.loads(raw)
            price = float(msg["p"])
            exch_ms = int(msg.get("T") or msg.get("E") or 0)
        except (ValueError, KeyError, TypeError):
            return
        if price > 0.0:
            self._buffer.add(price, exch_ms)


# --------------------------------------------------------------------------- #
# Polymarket market discovery
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class TrackedMarket:
    slug: str
    title: str
    horizon: str  # "5m" | "15m"
    condition_id: str
    up_token: str
    down_token: str
    window_open: float  # unix seconds
    window_close: float  # unix seconds
    twap_lookback: float
    tick_size: float
    accepting_orders: bool
    books: Deque["BookSnapshot"] = field(default_factory=lambda: deque(maxlen=64))
    last_signal: dict[str, float] = field(default_factory=dict)

    def seconds_remaining(self, now: float | None = None) -> float:
        return self.window_close - (now if now is not None else time.time())

    def effective_tau(self, now: float | None = None) -> float:
        """Time-to-expiry with the TWAP variance haircut applied (seconds)."""
        tau = self.seconds_remaining(now)
        return max(tau - (2.0 * self.twap_lookback / 3.0), 1.0)

    def is_live(self, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        return self.accepting_orders and self.window_open <= now < self.window_close


@dataclass(frozen=True, slots=True)
class BookSnapshot:
    mono: float
    up_ask: float | None
    down_ask: float | None
    up_bid: float | None
    down_bid: float | None

    @property
    def up_mid(self) -> float | None:
        if self.up_ask is not None and self.up_bid is not None:
            return (self.up_ask + self.up_bid) / 2.0
        # Fall back to the complement of the Down side if Up is one-sided.
        if self.down_ask is not None and self.down_bid is not None:
            return 1.0 - (self.down_ask + self.down_bid) / 2.0
        return self.up_ask or self.up_bid


def _best_ask(levels: Sequence) -> float | None:
    """Lowest ask price across the book.

    The CLOB returns *both* sides sorted ascending by price, so the best ask is
    the last element, not the first. Taking an explicit min is ordering-agnostic
    and immune to that changing underneath us.
    """
    prices = [float(lvl.price) for lvl in levels if float(lvl.size) > 0.0]
    return min(prices) if prices else None


def _best_bid(levels: Sequence) -> float | None:
    prices = [float(lvl.price) for lvl in levels if float(lvl.size) > 0.0]
    return max(prices) if prices else None


class MarketDiscovery:
    """Finds the currently-trading BTC Up/Down markets on Gamma.

    Fast path: these series embed the window-open unix epoch in the slug
    (``btc-updown-5m-1786517100``), so the live market is addressable directly
    with one tiny query instead of paging the whole event list.

    Fallback: if the slug convention ever changes, scan the series by
    ``series_slug`` and filter on the window bounds client-side.

    Note: Polymarket pre-creates these markets roughly 24h ahead of their
    window, so "newest by startDate" is NOT the live market. Selection is
    always done on ``eventStartTime <= now < endDate``.
    """

    def __init__(self, session: aiohttp.ClientSession) -> None:
        self._session = session

    async def discover(self) -> list[TrackedMarket]:
        markets = await self._by_slug()
        if not markets:
            log.debug("Slug fast-path returned nothing; falling back to series scan")
            markets = await self._by_series()
        return markets

    # -- fast path ---------------------------------------------------------- #

    async def _by_slug(self) -> list[TrackedMarket]:
        now = int(time.time())
        slugs: list[str] = []
        for cfg in SERIES.values():
            period = cfg["period"]
            current = now // period * period
            # Current window plus the next one, so we are already warm when the
            # roll happens rather than discovering it a poll late.
            slugs.append(f"{cfg['slug_prefix']}-{current}")
            slugs.append(f"{cfg['slug_prefix']}-{current + period}")

        params = [("slug", s) for s in slugs]
        payload = await self._get(params)
        return self._parse(payload)

    # -- fallback ----------------------------------------------------------- #

    async def _by_series(self) -> list[TrackedMarket]:
        out: list[TrackedMarket] = []
        for cfg in SERIES.values():
            params = [
                ("series_slug", cfg["series_slug"]),
                ("closed", "false"),
                ("active", "true"),
                ("limit", "60"),
                ("order", "startDate"),
                ("ascending", "false"),
            ]
            payload = await self._get(params)
            out.extend(self._parse(payload))
        return out

    # -- plumbing ----------------------------------------------------------- #

    async def _get(self, params: list[tuple[str, str]]) -> list[dict]:
        try:
            async with self._session.get(
                GAMMA_EVENTS_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    log.warning("Gamma returned HTTP %s", resp.status)
                    return []
                data = await resp.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as exc:
            log.warning("Gamma discovery failed: %s", exc)
            return []

        if isinstance(data, dict):
            data = data.get("data", [])
        return data if isinstance(data, list) else []

    def _parse(self, events: Iterable[dict]) -> list[TrackedMarket]:
        now = time.time()
        out: list[TrackedMarket] = []

        for event in events:
            slug = event.get("slug", "")
            horizon = next(
                (h for h, cfg in SERIES.items() if slug.startswith(cfg["slug_prefix"] + "-")),
                None,
            )
            if horizon is None:
                continue

            for market in event.get("markets") or []:
                parsed = self._parse_market(event, market, horizon, now)
                if parsed is not None:
                    out.append(parsed)

        return out

    def _parse_market(
        self, event: dict, market: dict, horizon: str, now: float
    ) -> TrackedMarket | None:
        if market.get("closed") or not market.get("enableOrderBook", True):
            return None

        try:
            outcomes = json.loads(market.get("outcomes") or "[]")
            token_ids = json.loads(market.get("clobTokenIds") or "[]")
        except json.JSONDecodeError:
            return None
        if len(outcomes) != 2 or len(token_ids) != 2:
            return None

        # Outcomes are ["Up", "Down"] on these series; map to YES/NO by name
        # rather than by position so a reordering upstream cannot silently
        # invert every signal.
        lookup = {str(o).strip().lower(): tid for o, tid in zip(outcomes, token_ids)}
        up_token = lookup.get("up") or lookup.get("yes")
        down_token = lookup.get("down") or lookup.get("no")
        if not up_token or not down_token:
            return None

        open_ts = _parse_iso(market.get("eventStartTime") or event.get("startTime"))
        close_ts = _parse_iso(market.get("endDate") or event.get("endDate"))
        if open_ts is None or close_ts is None or close_ts <= open_ts:
            return None
        if not (open_ts <= now < close_ts):
            return None  # not the live window

        cfg = market.get("cryptoMarketConfig") or {}
        twap = float(cfg.get("twapLookbackSeconds") or 0.0) if cfg.get("twapEnabled") else 0.0

        return TrackedMarket(
            slug=market.get("slug") or event.get("slug", ""),
            title=market.get("question") or event.get("title", ""),
            horizon=horizon,
            condition_id=market.get("conditionId", ""),
            up_token=str(up_token),
            down_token=str(down_token),
            window_open=open_ts,
            window_close=close_ts,
            twap_lookback=twap,
            tick_size=float(market.get("orderPriceMinTickSize") or 0.01),
            accepting_orders=bool(market.get("acceptingOrders", True)),
        )


def _parse_iso(value: str | None) -> float | None:
    if not value:
        return None
    try:
        import datetime as _dt

        text = value.replace("Z", "+00:00")
        parsed = _dt.datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_dt.timezone.utc)
        return parsed.timestamp()
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Order book access
# --------------------------------------------------------------------------- #


class BookFeed:
    """Batched CLOB order book reader.

    py-clob-client is synchronous, so calls are dispatched to a worker thread to
    keep the event loop (and therefore the Binance tape) unblocked. Level-0
    access is unauthenticated, which is all that public book reads require.
    """

    def __init__(self, host: str = CLOB_HOST, chain_id: int = POLYGON_CHAIN_ID) -> None:
        self._client = ClobClient(host, chain_id=chain_id)
        self._lock = asyncio.Lock()

    async def fetch(self, token_ids: Sequence[str]) -> dict[str, tuple[float | None, float | None]]:
        """Return {token_id: (best_bid, best_ask)}."""
        if not token_ids:
            return {}

        params = [BookParams(token_id=tid) for tid in token_ids]
        try:
            async with self._lock:
                books = await asyncio.to_thread(self._client.get_order_books, params)
        except Exception as exc:  # noqa: BLE001 - network/venue errors are routine
            log.warning("CLOB book fetch failed: %s", exc)
            return {}

        out: dict[str, tuple[float | None, float | None]] = {}
        for book in books or []:
            token = str(getattr(book, "asset_id", "") or "")
            if not token:
                continue
            out[token] = (_best_bid(book.bids or []), _best_ask(book.asks or []))
        return out


# --------------------------------------------------------------------------- #
# Signal engine
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Config:
    spike_bps: float = 12.0  # 0.12%
    spike_lookback: float = SPIKE_LOOKBACK_SECONDS
    max_yes_ask: float = 0.52  # staleness gate from the spec
    min_edge: float = 0.03  # required fair-value edge over the ask
    signal_cooldown: float = 5.0
    book_poll_interval: float = 1.0
    discovery_interval: float = 20.0
    eval_hz: float = 10.0
    min_seconds_left: float = 8.0  # ignore markets about to settle
    enforce_ask_ceiling: bool = True
    ws_endpoints: tuple[str, ...] = BINANCE_WS_FALLBACKS


class SignalEngine:
    def __init__(self, buffer: PriceBuffer, books: BookFeed, cfg: Config) -> None:
        self._buffer = buffer
        self._books = books
        self._cfg = cfg
        self._markets: dict[str, TrackedMarket] = {}
        self._last_forced_fetch = 0.0

    # -- market registry ---------------------------------------------------- #

    def sync_markets(self, discovered: Sequence[TrackedMarket]) -> None:
        now = time.time()
        seen: set[str] = set()

        for market in discovered:
            seen.add(market.slug)
            existing = self._markets.get(market.slug)
            if existing is None:
                self._markets[market.slug] = market
                log.info(
                    "Tracking %s [%s] | window %s -> %s | %.0fs left",
                    market.slug,
                    market.horizon,
                    _fmt_clock(market.window_open),
                    _fmt_clock(market.window_close),
                    market.seconds_remaining(now),
                )
            else:
                # Preserve accumulated book history across refreshes.
                existing.accepting_orders = market.accepting_orders
                existing.window_close = market.window_close

        for slug in [s for s, m in self._markets.items() if not m.is_live(now)]:
            log.info("Retiring %s (window closed)", slug)
            self._markets.pop(slug, None)

    def active_markets(self) -> list[TrackedMarket]:
        now = time.time()
        return [
            m
            for m in self._markets.values()
            if m.is_live(now) and m.seconds_remaining(now) >= self._cfg.min_seconds_left
        ]

    def tracked_tokens(self) -> list[str]:
        tokens: list[str] = []
        for market in self.active_markets():
            tokens.extend((market.up_token, market.down_token))
        return tokens

    # -- book ingestion ----------------------------------------------------- #

    async def refresh_books(self, force: bool = False) -> None:
        markets = self.active_markets()
        if not markets:
            return
        if force:
            # Never let a burst of spikes hammer the venue.
            if time.monotonic() - self._last_forced_fetch < 0.25:
                return
            self._last_forced_fetch = time.monotonic()

        quotes = await self._books.fetch(self.tracked_tokens())
        if not quotes:
            return

        mono = time.monotonic()
        for market in markets:
            up_bid, up_ask = quotes.get(market.up_token, (None, None))
            down_bid, down_ask = quotes.get(market.down_token, (None, None))
            if up_ask is None and down_ask is None:
                continue
            market.books.append(
                BookSnapshot(
                    mono=mono,
                    up_ask=up_ask,
                    down_ask=down_ask,
                    up_bid=up_bid,
                    down_bid=down_bid,
                )
            )

    # -- evaluation --------------------------------------------------------- #

    async def evaluate(self) -> None:
        markets = self.active_markets()
        if not markets or not self._buffer.ready:
            return

        move = self._buffer.largest_move(self._cfg.spike_lookback)
        if move is None or abs(move.bps) < self._cfg.spike_bps:
            return

        # Refresh before judging: the whole question is whether the book has
        # ALREADY repriced. A stale local snapshot would manufacture fake edge.
        await self.refresh_books(force=True)

        sigma = self._buffer.sigma_per_sqrt_second()
        for market in self.active_markets():
            self._evaluate_market(market, move, sigma)

    def _evaluate_market(self, market: TrackedMarket, move: Move, sigma: float) -> None:
        if len(market.books) < 2:
            return

        # Anchor: the newest snapshot taken strictly BEFORE the move began.
        anchor = None
        for snap in market.books:
            if snap.mono <= move.from_mono:
                anchor = snap
            else:
                break
        if anchor is None:
            return

        current = market.books[-1]
        if current.mono <= anchor.mono:
            return

        p0 = anchor.up_mid
        if p0 is None:
            return

        tau_eff = market.effective_tau()
        denom = sigma * math.sqrt(tau_eff)
        if denom <= 0.0:
            return

        z0 = norm_ppf(clamp_prob(p0))
        z1 = z0 + move.delta_log / denom
        p_fair_up = clamp_prob(norm_cdf(z1))
        p_fair_down = 1.0 - p_fair_up

        self._check_cross_book(market, current, move)

        if move.delta_log > 0:
            self._check_leg(
                market, move, sigma, "UP", "YES", current.up_ask, p_fair_up, p0, tau_eff
            )
        else:
            self._check_leg(
                market, move, sigma, "DOWN", "NO", current.down_ask, p_fair_down, 1.0 - p0, tau_eff
            )

    def _check_leg(
        self,
        market: TrackedMarket,
        move: Move,
        sigma: float,
        side: str,
        token_label: str,
        ask: float | None,
        p_fair: float,
        p_anchor: float,
        tau_eff: float,
    ) -> None:
        if ask is None:
            return

        edge = p_fair - ask
        if edge < self._cfg.min_edge:
            return
        if self._cfg.enforce_ask_ceiling and ask >= self._cfg.max_yes_ask:
            return
        if not self._cooldown_ok(market, side):
            return

        log.warning(
            "\n"
            "=============== ARBITRAGE WINDOW OPEN ===============\n"
            " market      : %s  [%s]\n"
            " title       : %s\n"
            " spot        : %s spike %+.1f bps in %.2fs  (%.2f -> %.2f)\n"
            " book anchor : %s implied %.3f (pre-spike mid)\n"
            " fair value  : %.3f   (sigma %.1f bps/s, tau_eff %.0fs)\n"
            " %-11s : %.3f   <-- STALE, below %.2f gate\n"
            " edge        : %+.3f  (%.1f cents/contract, gross)\n"
            " expiry      : %.0fs remaining\n"
            " action      : BUY %s @ %.3f\n"
            "=====================================================",
            market.slug,
            market.horizon,
            market.title,
            move.direction,
            move.bps,
            move.elapsed,
            move.from_price,
            move.to_price,
            token_label,
            p_anchor,
            p_fair,
            sigma * 10_000.0,
            tau_eff,
            f"{token_label} ask",
            ask,
            self._cfg.max_yes_ask,
            edge,
            edge * 100.0,
            market.seconds_remaining(),
            token_label,
            ask,
        )
        market.last_signal[side] = time.monotonic()

    def _check_cross_book(self, market: TrackedMarket, book: BookSnapshot, move: Move) -> None:
        """Model-free leg: both asks summing below $1 is a locked profit."""
        if book.up_ask is None or book.down_ask is None:
            return
        total = book.up_ask + book.down_ask
        if total >= 1.0 or not self._cooldown_ok(market, "CROSS"):
            return

        log.error(
            "\n"
            "*********** RISK-FREE CROSS-BOOK ARB ***********\n"
            " market : %s [%s]\n"
            " YES ask %.3f + NO ask %.3f = %.3f  (< 1.000)\n"
            " locked : %.1f cents/pair, %.0fs to settlement\n"
            " context: spot %s %+.1f bps in %.2fs\n"
            "************************************************",
            market.slug,
            market.horizon,
            book.up_ask,
            book.down_ask,
            total,
            (1.0 - total) * 100.0,
            market.seconds_remaining(),
            move.direction,
            move.bps,
            move.elapsed,
        )
        market.last_signal["CROSS"] = time.monotonic()

    def _cooldown_ok(self, market: TrackedMarket, key: str) -> bool:
        last = market.last_signal.get(key, 0.0)
        return time.monotonic() - last >= self._cfg.signal_cooldown


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def _fmt_clock(unix_ts: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(unix_ts)) + "Z"


class Scanner:
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._buffer = PriceBuffer()
        self._stream = BinanceTradeStream(self._buffer, cfg.ws_endpoints)
        self._books = BookFeed()
        self._engine = SignalEngine(self._buffer, self._books, cfg)

    async def run(self) -> None:
        async with aiohttp.ClientSession(
            headers={"User-Agent": "btc-polymarket-arb/1.0"}
        ) as session:
            discovery = MarketDiscovery(session)

            tasks = [
                asyncio.create_task(self._stream.run(), name="binance"),
                asyncio.create_task(self._discovery_loop(discovery), name="discovery"),
                asyncio.create_task(self._book_loop(), name="books"),
                asyncio.create_task(self._eval_loop(), name="eval"),
                asyncio.create_task(self._heartbeat_loop(), name="heartbeat"),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    async def _discovery_loop(self, discovery: MarketDiscovery) -> None:
        while True:
            try:
                found = await discovery.discover()
                self._engine.sync_markets(found)
                if not found:
                    log.info("No live BTC Up/Down window found; will retry")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.exception("Discovery loop error: %s", exc)
            await asyncio.sleep(self._cfg.discovery_interval)

    async def _book_loop(self) -> None:
        while True:
            try:
                await self._engine.refresh_books()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.exception("Book loop error: %s", exc)
            await asyncio.sleep(self._cfg.book_poll_interval)

    async def _eval_loop(self) -> None:
        period = 1.0 / max(self._cfg.eval_hz, 1.0)
        while True:
            try:
                await self._engine.evaluate()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.exception("Eval loop error: %s", exc)
            await asyncio.sleep(period)

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(15.0)
            last = self._buffer.last()
            markets = self._engine.active_markets()
            move = self._buffer.largest_move(self._cfg.spike_lookback)
            log.info(
                "hb | spot=%s | sigma=%.2f bps/s | ws=%s lat=%.0fms ticks=%d | 3s move=%+.1f bps | markets=%s",
                f"{last.price:,.2f}" if last else "n/a",
                self._buffer.sigma_per_sqrt_second() * 10_000.0,
                "up" if self._stream.connected else "DOWN",
                self._buffer.latency_ms,
                self._buffer.tick_count,
                move.bps if move else 0.0,
                ", ".join(
                    f"{m.horizon}:{_quote(m)}@{m.seconds_remaining():.0f}s" for m in markets
                )
                or "none",
            )


def _quote(market: TrackedMarket) -> str:
    if not market.books:
        return "-"
    book = market.books[-1]
    up = f"{book.up_ask:.2f}" if book.up_ask is not None else "-"
    down = f"{book.down_ask:.2f}" if book.down_ask is not None else "-"
    return f"{up}/{down}"


def parse_args(argv: Sequence[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(
        description="Binance/Polymarket short-term BTC Up-or-Down dislocation scanner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--spike-bps", type=float, default=12.0, help="spot move threshold in basis points"
    )
    parser.add_argument(
        "--spike-window", type=float, default=SPIKE_LOOKBACK_SECONDS, help="spike horizon, seconds"
    )
    parser.add_argument(
        "--max-yes-ask", type=float, default=0.52, help="staleness gate: only flag asks below this"
    )
    parser.add_argument(
        "--min-edge", type=float, default=0.03, help="minimum fair-value edge over the ask"
    )
    parser.add_argument(
        "--no-ask-ceiling",
        action="store_true",
        help="ignore --max-yes-ask and flag on model edge alone",
    )
    parser.add_argument("--book-interval", type=float, default=1.0, help="book poll seconds")
    parser.add_argument("--cooldown", type=float, default=5.0, help="per-signal cooldown seconds")
    parser.add_argument(
        "--ws-url",
        action="append",
        default=None,
        metavar="URL",
        help=(
            "spot trade stream endpoint; repeat to build a failover chain. "
            "Defaults to the canonical Binance endpoint, then the "
            "data-stream.binance.vision mirror (use the mirror directly if "
            "Binance returns HTTP 451 in your region)."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("websockets").setLevel(logging.WARNING)

    return Config(
        spike_bps=args.spike_bps,
        spike_lookback=args.spike_window,
        max_yes_ask=args.max_yes_ask,
        min_edge=args.min_edge,
        signal_cooldown=args.cooldown,
        book_poll_interval=args.book_interval,
        enforce_ask_ceiling=not args.no_ask_ceiling,
        ws_endpoints=tuple(args.ws_url) if args.ws_url else BINANCE_WS_FALLBACKS,
    )


async def amain(cfg: Config) -> None:
    scanner = Scanner(cfg)
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    runner = asyncio.create_task(scanner.run())
    stopper = asyncio.create_task(stop.wait())
    done, pending = await asyncio.wait({runner, stopper}, return_when=asyncio.FIRST_COMPLETED)

    for task in pending:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    for task in done:
        if task is runner:
            task.result()  # surface a genuine crash


def main() -> None:
    cfg = parse_args()
    log.info(
        "Starting scanner | spike>%.0fbps/%.0fs | ask gate %.2f | min edge %.3f",
        cfg.spike_bps,
        cfg.spike_lookback,
        cfg.max_yes_ask,
        cfg.min_edge,
    )
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(amain(cfg))
    log.info("Scanner stopped")


if __name__ == "__main__":
    main()
