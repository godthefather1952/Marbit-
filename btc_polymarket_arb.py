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
import logging
import math
import os
import random
import signal
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Iterable, Sequence

import aiohttp

try:  # websockets >= 13 exposes the modern asyncio client here
    from websockets.asyncio.client import connect as ws_connect
except ImportError:  # pragma: no cover - older websockets
    from websockets.client import connect as ws_connect  # type: ignore[no-redef]

from websockets.exceptions import ConnectionClosed

# py-clob-client is only needed to trade Polymarket's global CLOB. The Kalshi
# adapter and the shared machinery (model, risk, resilience) import from this
# module, and dragging the whole eth-account/web3 stack in just to watch a
# Kalshi market is a slow and failure-prone install. So the SDK is optional:
# without it everything except Polymarket execution still runs, and the parts
# that genuinely need it fail with a clear instruction instead of an ImportError
# at startup.
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import (
        ApiCreds,
        AssetType,
        BalanceAllowanceParams,
        BookParams,
        OrderArgs,
        OrderType,
        PartialCreateOrderOptions,
    )
    from py_clob_client.order_builder.constants import BUY

    POLYMARKET_SDK = True
except ImportError:  # pragma: no cover - exercised by the fallback test
    POLYMARKET_SDK = False
    BUY = "BUY"  # the literal wire value; no SDK needed to know it
    ClobClient = None  # type: ignore[assignment]
    ApiCreds = AssetType = BalanceAllowanceParams = None  # type: ignore[assignment]
    OrderArgs = OrderType = None  # type: ignore[assignment]

    @dataclass(slots=True)
    class BookParams:  # type: ignore[no-redef]
        """Stand-in matching the SDK's shape, so cached objects still build."""

        token_id: str
        side: str | None = None

    @dataclass(slots=True)
    class PartialCreateOrderOptions:  # type: ignore[no-redef]
        tick_size: str | None = None
        neg_risk: bool | None = None


def require_polymarket_sdk(what: str = "this operation") -> None:
    """Fail with an actionable message rather than an ImportError traceback."""
    if not POLYMARKET_SDK:
        raise RuntimeError(
            f"{what} needs the Polymarket CLOB SDK. Install it with:\n"
            "    pip install py-clob-client\n"
            "It is not required for the Kalshi tools or the shared model."
        )

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

#: Hard ceilings on the rolling buffers. BTC trades peak around 200/s, so 10s of
#: tape is ~2k ticks; 8k leaves headroom for a burst without ever reallocating.
MAX_TICKS_BUFFERED = 8192
MAX_VOL_BARS = 512

#: Fallback BTC vol (annualized) used until enough 1s bars have accumulated.
FALLBACK_ANNUAL_VOL = 0.45
SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0

#: Polymarket prices live on a 1c grid; probabilities are clamped inside this
#: band before any Phi^-1 call so the inverse CDF cannot blow up.
PROB_EPS = 0.01

#: Signature schemes accepted by the CLOB. 0 = the private key itself holds the
#: funds; 1/2 = a Polymarket proxy wallet holds them and `funder` must name it.
SIG_TYPE_EOA = 0
SIG_TYPE_EMAIL_PROXY = 1
SIG_TYPE_BROWSER_PROXY = 2
PROXY_SIG_TYPES = (SIG_TYPE_EMAIL_PROXY, SIG_TYPE_BROWSER_PROXY)

#: Rolling short-dated BTC series on Gamma. Window-open epoch is embedded in
#: the slug, which lets us address the live market directly instead of paging.
SERIES = {
    "5m": {"series_slug": "btc-up-or-down-5m", "slug_prefix": "btc-updown-5m", "period": 300},
    "15m": {"series_slug": "btc-up-or-down-15m", "slug_prefix": "btc-updown-15m", "period": 900},
}

log = logging.getLogger("btc-arb")


# --------------------------------------------------------------------------- #
# Fast serialization
# --------------------------------------------------------------------------- #
#
# orjson is ~2.7x faster than the stdlib on a Binance trade payload (0.9us vs
# 2.5us) and accepts bytes directly, so the websocket frame never needs decoding
# to str first. ujson is the fallback, stdlib json the last resort - the bot must
# still start on a machine where neither wheel is installed.
#
# All three raise a ValueError subclass on malformed input, so callers catch
# ValueError rather than a backend-specific type.

try:  # pragma: no cover - import path depends on the host
    import orjson as _fastjson

    JSON_BACKEND = "orjson"
    json_loads = _fastjson.loads
except ImportError:  # pragma: no cover
    try:
        import ujson as _fastjson

        JSON_BACKEND = "ujson"
        json_loads = _fastjson.loads
    except ImportError:
        import json as _fastjson

        JSON_BACKEND = "json"
        json_loads = _fastjson.loads


# --------------------------------------------------------------------------- #
# Retry / rate-limit handling
# --------------------------------------------------------------------------- #

#: HTTP statuses worth retrying: rate limiting plus transient server faults.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class RetryableError(Exception):
    """A transient failure that backoff may clear."""

    def __init__(self, message: str, status: int | None = None, retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


def _status_of(exc: BaseException) -> int | None:
    """Pull an HTTP status off whichever client raised."""
    for attr in ("status", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def _retry_after_of(exc: BaseException) -> float | None:
    """Honour an explicit Retry-After hint when the venue sends one."""
    hint = getattr(exc, "retry_after", None)
    if isinstance(hint, (int, float)) and hint >= 0:
        return float(hint)

    headers = getattr(exc, "headers", None)
    if headers:
        with contextlib.suppress(TypeError, ValueError, AttributeError):
            raw = headers.get("Retry-After")
            if raw is not None:
                return max(0.0, float(raw))
    return None


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    # An explicit status always decides, even on a RetryableError: a 400 wrapped
    # in one is still a client error that no amount of backoff will fix.
    status = _status_of(exc)
    if status is not None:
        return status in RETRYABLE_STATUS
    if isinstance(exc, RetryableError):
        return True
    # Connection-level faults carry no status but are exactly what backoff is for.
    return isinstance(exc, (aiohttp.ClientError, ConnectionError, OSError))


async def retry_async(
    func,
    *args,
    attempts: int = 4,
    base_delay: float = 0.4,
    max_delay: float = 8.0,
    label: str = "request",
    **kwargs,
):
    """Call an async function with exponential backoff on transient failures.

    Backoff is `base_delay * 2**n` with full jitter, capped at `max_delay`, and
    a server-sent `Retry-After` always wins over the computed delay.

    Deliberately NOT applied to order submission. A POST that times out may have
    executed, so retrying it risks a double fill - far worse than a missed
    trade. Only idempotent reads (books, discovery, balances) retry.
    """
    delay = base_delay
    last: BaseException | None = None

    for attempt in range(1, attempts + 1):
        try:
            return await func(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - classified immediately below
            last = exc
            if not is_retryable(exc) or attempt == attempts:
                raise

            hint = _retry_after_of(exc)
            wait = hint if hint is not None else random.uniform(0.0, min(delay, max_delay))
            status = _status_of(exc)
            log.warning(
                "%s failed (attempt %d/%d%s): %s - retrying in %.2fs",
                label,
                attempt,
                attempts,
                f", HTTP {status}" if status else "",
                exc,
                wait,
            )
            await asyncio.sleep(wait)
            delay = min(delay * 2.0, max_delay)

    assert last is not None
    raise last


# --------------------------------------------------------------------------- #
# Latency profiling
# --------------------------------------------------------------------------- #


class LatencyProfiler:
    """Microsecond-resolution timing across three hot-path checkpoints.

        T1  Binance websocket frame received
        T2  signal evaluated and order payload constructed
        T3  order response received from the Polymarket CLOB

    Recording is a tuple append into a bounded deque - a few hundred nanoseconds,
    entirely off the critical measurement. `perf_counter_ns` is monotonic and
    immune to wall-clock adjustments, which `time.time()` is not.
    """

    __slots__ = ("_samples", "_total")

    def __init__(self, capacity: int = 256) -> None:
        self._samples: Deque[tuple[int, int, int]] = deque(maxlen=capacity)
        self._total = 0

    def record(self, t1_ns: int, t2_ns: int, t3_ns: int) -> None:
        if t1_ns and t2_ns and t3_ns:
            self._samples.append((t1_ns, t2_ns, t3_ns))
            self._total += 1

    @property
    def count(self) -> int:
        return self._total

    def last(self) -> tuple[float, float, float] | None:
        """(signal_ms, order_ms, total_ms) for the most recent order."""
        if not self._samples:
            return None
        t1, t2, t3 = self._samples[-1]
        return (t2 - t1) / 1e6, (t3 - t2) / 1e6, (t3 - t1) / 1e6

    def percentiles(self) -> tuple[float, float, float] | None:
        """(median signal_ms, median order_ms, p95 total_ms) over the window."""
        if not self._samples:
            return None
        signal = sorted((t2 - t1) / 1e6 for t1, t2, _ in self._samples)
        order = sorted((t3 - t2) / 1e6 for _, t2, t3 in self._samples)
        total = sorted((t3 - t1) / 1e6 for t1, _, t3 in self._samples)
        mid = len(signal) // 2
        p95 = total[min(len(total) - 1, int(len(total) * 0.95))]
        return signal[mid], order[mid], p95

    def summary(self) -> str:
        pct = self.percentiles()
        if pct is None:
            return "lat=n/a"
        signal_ms, order_ms, p95_ms = pct
        return (
            f"lat[n={self._total}] T1->T2 {signal_ms:.2f}ms "
            f"T2->T3 {order_ms:.1f}ms p95 {p95_ms:.1f}ms"
        )


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
    perf_ns: int = 0  # T1: perf_counter_ns() at frame receipt


@dataclass(frozen=True, slots=True)
class Move:
    """A spot excursion measured inside the spike lookback window."""

    delta_log: float
    from_price: float
    to_price: float
    from_mono: float
    to_mono: float
    t1_ns: int = 0  # perf_counter_ns of the tick that completed this move

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
        # Fixed maxlen gives O(1) eviction and a hard memory ceiling with no
        # reallocation as ticks arrive. Time-based pruning still runs for
        # correctness (the window is 10 *seconds*, not 10k ticks); maxlen is the
        # backstop that keeps a venue burst from growing the deque unboundedly.
        self._ticks: Deque[Tick] = deque(maxlen=MAX_TICKS_BUFFERED)
        self._vol_window = vol_window
        self._vol_decay = 0.5 ** (1.0 / max(vol_halflife, 1.0))
        self._bars: Deque[tuple[float, float]] = deque(
            maxlen=MAX_VOL_BARS
        )  # (mono, price) on a 1s grid
        self._last_bar_mono: float | None = None
        self._tick_count = 0
        self._latency_ms = 0.0
        self._warm_after = 0.0
        self.resets = 0
        # O(1) spike tripwire: absolute price levels that, if crossed, mean a
        # move past the threshold has occurred. Re-armed after each evaluation.
        self._trigger_low = 0.0
        self._trigger_high = math.inf
        self._trigger_armed = False
        self._on_trigger = None
        self.wakes = 0

    # -- ingest ------------------------------------------------------------- #

    def set_wake(self, callback) -> None:
        """Register a zero-argument callback fired when the tripwire trips."""
        self._on_trigger = callback

    def arm_trigger(self, threshold_bps: float) -> None:
        """Re-arm the tripwire from the current window's price extremes.

        A move of `threshold_bps` from the window low upward lands at
        `low * exp(+thr)`; from the window high downward at `high * exp(-thr)`.
        Comparing each incoming tick against those two levels is two float
        comparisons - no logarithm, no scan - so the tape can detect a candidate
        spike per tick and wake the evaluator immediately instead of waiting out
        the polling period.

        This is only a hint: evaluate() still recomputes the real move, so a
        false wake costs one wasted pass and a missed wake is bounded by the
        periodic timer.
        """
        # Clamped: a threshold large enough to overflow exp() is one that can
        # never trip anyway, and the tripwire must not raise on the tick path.
        thr = min(max(threshold_bps, 0.0) / 10_000.0, 5.0)
        lo = hi = 0.0
        floor_mono = time.monotonic() - SPIKE_LOOKBACK_SECONDS
        for tick in reversed(self._ticks):
            if tick.mono < floor_mono:
                break
            if tick.price <= 0.0:
                continue
            if lo == 0.0 or tick.price < lo:
                lo = tick.price
            if tick.price > hi:
                hi = tick.price

        if lo <= 0.0:
            self._trigger_low, self._trigger_high = 0.0, math.inf
        else:
            self._trigger_low = hi * math.exp(-thr)
            self._trigger_high = lo * math.exp(thr)
        self._trigger_armed = True

    def add(self, price: float, exch_ms: int, perf_ns: int = 0) -> None:
        now = time.monotonic()
        self._ticks.append(Tick(now, price, exch_ms, perf_ns or time.perf_counter_ns()))
        self._tick_count += 1
        # Wall-clock feed latency is a heartbeat diagnostic, not a trading
        # input, so it is sampled rather than measured on every tick - one
        # fewer clock syscall per tick on the hottest path in the process.
        if not self._tick_count & 0x3F:
            self._latency_ms = max(0.0, time.time() * 1000.0 - exch_ms)

        # Two comparisons on the per-tick path; everything else is amortized.
        if self._trigger_armed and (price >= self._trigger_high or price <= self._trigger_low):
            self._trigger_armed = False  # debounce until re-armed by evaluate()
            self.wakes += 1
            if self._on_trigger is not None:
                self._on_trigger()

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

    def reset(self, reason: str, warmup: float = SPIKE_LOOKBACK_SECONDS) -> None:
        """Drop the tick window after a tape gap and re-enter warm-up.

        Without this, the ticks captured before a disconnect stay in the 10s
        buffer and get compared against the first tick after reconnect. Any
        price drift that happened during the outage would then read as an
        instantaneous spike and fire a signal against a book that had every
        opportunity to reprice - a phantom edge, and a live order behind it.

        Deliberately scoped to market data only. `active_positions`, session
        PnL and the breakers live on the Executor and RiskManager, so a
        reconnect never disturbs the risk state.
        """
        dropped = len(self._ticks)
        self._ticks.clear()
        self._last_bar_mono = None
        self._warm_after = time.monotonic() + max(warmup, 0.0)
        self.resets += 1
        log.info(
            "Price buffer reset (%s): dropped %d ticks, warming up for %.1fs",
            reason,
            dropped,
            warmup,
        )

    # -- state -------------------------------------------------------------- #

    @property
    def ready(self) -> bool:
        """Enough clean, gap-free history to trust a spike measurement."""
        return len(self._ticks) >= 2 and time.monotonic() >= self._warm_after

    @property
    def warming_up(self) -> bool:
        return time.monotonic() < self._warm_after

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

        # The largest |log(latest/earlier)| over the window is attained at
        # either the cheapest or the dearest earlier tick, so scanning for those
        # two extremes and taking two logarithms is equivalent to taking a
        # logarithm per tick - and at ~600 ticks per 3s window, evaluated at
        # 10 Hz, that is the difference between 2 and 600 log() calls per pass.
        lo_price = hi_price = 0.0
        lo_tick = hi_tick = None
        for tick in reversed(self._ticks):
            if tick.mono < floor_mono:
                break  # deque is time-ordered; everything earlier is older
            if tick is latest or tick.price <= 0.0:
                continue
            if lo_tick is None or tick.price < lo_price:
                lo_price, lo_tick = tick.price, tick
            if hi_tick is None or tick.price > hi_price:
                hi_price, hi_tick = tick.price, tick

        best: Move | None = None
        for candidate in (lo_tick, hi_tick):
            if candidate is None:
                continue
            delta = math.log(latest.price / candidate.price)
            if best is None or abs(delta) > abs(best.delta_log):
                best = Move(
                    delta,
                    candidate.price,
                    latest.price,
                    candidate.mono,
                    latest.mono,
                    latest.perf_ns,
                )
        return best

    def sigma_per_sqrt_second(self) -> float:
        """EWMA realized vol of 1-second log returns, in per-sqrt-second units."""
        fallback = FALLBACK_ANNUAL_VOL / math.sqrt(SECONDS_PER_YEAR)
        if len(self._bars) < 30:
            return fallback

        # Walk the deque backwards, pairing each bar with its predecessor. The
        # EWMA weights the newest observation most, which is exactly the order
        # reverse iteration yields - so no intermediate list is materialised
        # (the previous `list(self._bars)[1:]` copied the whole history on every
        # call, and this runs on the evaluation path).
        num = 0.0
        den = 0.0
        weight = 1.0
        counted = 0
        newer_mono: float | None = None
        newer_price = 0.0

        for mono, price in reversed(self._bars):
            if newer_mono is not None:
                dt = newer_mono - mono
                # Bars are sampled at 1s. A much larger gap means the tape
                # stalled, so the bridging "return" spans unobserved time and
                # would bias the estimate upward; skip it rather than
                # annualize a hole.
                if 0 < dt <= 5.0 and price > 0 and newer_price > 0:
                    ret = math.log(newer_price / price)
                    num += weight * (ret * ret) / dt  # per-second variance
                    den += weight
                    weight *= self._vol_decay
                    counted += 1
            newer_mono, newer_price = mono, price

        if counted < 30 or den <= 0.0:
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
                # Market-data state only. Position locks and PnL are untouched.
                self._buffer.reset(f"tape disconnect from {url}")

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
        # T1: stamped before parsing, so the checkpoint measures our own
        # handling cost rather than excluding deserialization from it.
        t1 = time.perf_counter_ns()
        try:
            msg = json_loads(raw)
            price = float(msg["p"])
            exch_ms = int(msg.get("T") or msg.get("E") or 0)
        except (ValueError, KeyError, TypeError):
            return
        if price > 0.0:
            self._buffer.add(price, exch_ms, t1)


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
    neg_risk: bool = False
    min_order_size: float = 5.0
    books: Deque["BookSnapshot"] = field(default_factory=lambda: deque(maxlen=64))
    last_signal: dict[str, float] = field(default_factory=dict)

    def seconds_remaining(self, now: float | None = None) -> float:
        return self.window_close - (now if now is not None else time.time())

    def effective_tau(self, now: float | None = None) -> float:
        """Effective time-to-expiry, defined so that z_shift = d / (sigma*sqrt(tau_eff)).

        The contract settles on the mean price over the final L seconds, so the
        haircut is piecewise in how much of that averaging window is still in
        the future:

        * ``tau >= L`` - the whole window is ahead. Var[mean] = sigma^2*(tau - 2L/3),
          and a spot move of d shifts the expected mean by the full d.
          => tau_eff = tau - 2L/3

        * ``tau < L``  - part of the window has already been realized and is
          locked in. Only the remaining stub is random:
          Var[mean] = sigma^2 * tau^3/(3L^2)  (Monte-Carlo verified), while a
          spot move of d now shifts the expected mean by only d*(tau/L).
          The L cancels between numerator and denominator:
          => tau_eff = tau/3

        The two branches agree at tau == L (both give L/3), so the curve is
        continuous. Getting this wrong matters: naively extending the first
        branch below tau == L drives the haircut negative, and clamping that to
        a floor pins fair value at ~1.0 in the closing seconds of every window.
        """
        tau = max(self.seconds_remaining(now), 1e-3)
        lookback = self.twap_lookback
        if lookback <= 0.0 or tau >= lookback:
            return max(tau - (2.0 * lookback / 3.0), 1e-3)
        return tau / 3.0

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


@dataclass(frozen=True, slots=True)
class Quote:
    """Top of book plus the venue metadata needed to price an order."""

    bid: float | None
    ask: float | None
    tick_size: float | None = None
    min_order_size: float | None = None
    neg_risk: bool | None = None


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
            return await retry_async(self._get_once, params, label="Gamma discovery")
        except Exception as exc:  # noqa: BLE001 - discovery retries next cycle
            log.warning("Gamma discovery failed after retries: %s", exc)
            return []

    async def _get_once(self, params: list[tuple[str, str]]) -> list[dict]:
        async with self._session.get(
            GAMMA_EVENTS_URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status in RETRYABLE_STATUS:
                raise RetryableError(
                    f"Gamma returned HTTP {resp.status}",
                    status=resp.status,
                    retry_after=_header_seconds(resp.headers.get("Retry-After")),
                )
            if resp.status != 200:
                # A 4xx that is not rate limiting will not fix itself; do not
                # burn retries on a malformed query.
                log.warning("Gamma returned HTTP %s (not retrying)", resp.status)
                return []
            body = await resp.read()

        try:
            data = json_loads(body) if body else []
        except ValueError as exc:
            raise RetryableError(f"Gamma returned unparseable JSON: {exc}") from exc

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
            outcomes = json_loads(market.get("outcomes") or "[]")
            token_ids = json_loads(market.get("clobTokenIds") or "[]")
        except ValueError:
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
            # Carried so orders can be signed without an extra round-trip to
            # resolve neg_risk / tick size at execution time.
            neg_risk=bool(market.get("negRisk", False)),
            min_order_size=float(market.get("orderMinSize") or 5.0),
        )


def _header_seconds(raw: str | None) -> float | None:
    """Parse a Retry-After header expressed in seconds."""
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


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
        require_polymarket_sdk("Reading the Polymarket order book")
        self._client = ClobClient(host, chain_id=chain_id)
        self._lock = asyncio.Lock()

    async def fetch(self, token_ids: Sequence[str]) -> dict[str, "Quote"]:
        """Return {token_id: Quote} with live top-of-book and venue metadata."""
        if not token_ids:
            return {}

        params = [BookParams(token_id=tid) for tid in token_ids]
        try:
            async with self._lock:
                # Book reads are idempotent, so retrying a 429 or a 5xx is safe.
                books = await retry_async(
                    asyncio.to_thread,
                    self._client.get_order_books,
                    params,
                    attempts=3,
                    label="CLOB book fetch",
                )
        except Exception as exc:  # noqa: BLE001 - network/venue errors are routine
            log.warning("CLOB book fetch failed after retries: %s", exc)
            return {}

        out: dict[str, Quote] = {}
        for book in books or []:
            token = str(getattr(book, "asset_id", "") or "")
            if not token:
                continue
            out[token] = Quote(
                bid=_best_bid(book.bids or []),
                ask=_best_ask(book.asks or []),
                # Trust the book over the Gamma snapshot. Polymarket tightens the
                # tick for extreme prices near expiry, and quantizing a 0.001 ask
                # onto a stale 0.01 grid would price the order an order of
                # magnitude away from the liquidity we are trying to take.
                tick_size=_maybe_float(getattr(book, "tick_size", None)),
                min_order_size=_maybe_float(getattr(book, "min_order_size", None)),
                neg_risk=_maybe_bool(getattr(book, "neg_risk", None)),
            )
        return out


def _maybe_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _maybe_bool(value) -> bool | None:
    return bool(value) if isinstance(value, bool) else None


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


def tune_clob_http_client(
    connect_timeout: float = 3.0,
    read_timeout: float = 8.0,
    pool_size: int = 20,
) -> bool:
    """Retune py-clob-client's shared httpx client for low-latency trading.

    The library already keeps one module-level `httpx.Client(http2=True)`, so
    connections pool and TLS is negotiated once. What it does not set is an
    explicit timeout budget or pool sizing: the stock 5s blanket timeout is far
    too long for an order that is only valuable for a few hundred milliseconds.

    Best-effort - a library refactor that moves the client just means we keep
    the stock behaviour, which still works.
    """
    try:
        import httpx
        from py_clob_client.http_helpers import helpers

        old = getattr(helpers, "_http_client", None)
        helpers._http_client = httpx.Client(
            http2=True,
            timeout=httpx.Timeout(read_timeout, connect=connect_timeout),
            limits=httpx.Limits(
                max_connections=pool_size,
                max_keepalive_connections=pool_size,
                keepalive_expiry=90.0,
            ),
        )
        if old is not None:
            with contextlib.suppress(Exception):
                old.close()
        return True
    except Exception as exc:  # noqa: BLE001 - optimization only
        log.debug("Could not retune the CLOB http client: %s", exc)
        return False


def load_dotenv(path: str | Path = ".env") -> None:
    """Populate os.environ from a .env file without clobbering real env vars.

    python-dotenv arrives as a py-clob-client dependency, but this falls back to
    a minimal parser so a missing optional package can never be the reason a
    live trading bot silently starts up unauthenticated.
    """
    try:
        from dotenv import load_dotenv as _load  # type: ignore[import-not-found]

        _load(path, override=False)
        return
    except ImportError:
        pass

    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


@dataclass(slots=True)
class Credentials:
    """Polymarket auth material, sourced from the environment."""

    private_key: str | None = None
    api_key: str | None = None
    api_secret: str | None = None
    api_passphrase: str | None = None
    signature_type: int = SIG_TYPE_EOA
    funder: str | None = None
    #: Private Polygon RPC (Alchemy / QuickNode / Chainstack). When set, balance
    #: and allowance are read straight from chain instead of via the CLOB API,
    #: which is both faster and independent of Polymarket's own uptime.
    rpc_url: str | None = None
    #: Websocket RPC. Plumbed through and validated for on-chain subscriptions;
    #: the trading path is REST-only today and does not use it.
    ws_rpc_url: str | None = None

    @classmethod
    def from_env(cls) -> "Credentials":
        raw_sig = os.getenv("POLYMARKET_SIGNATURE_TYPE", "").strip()
        try:
            sig_type = int(raw_sig) if raw_sig else SIG_TYPE_EOA
        except ValueError:
            log.warning("POLYMARKET_SIGNATURE_TYPE=%r is not an int; using 0 (EOA)", raw_sig)
            sig_type = SIG_TYPE_EOA

        return cls(
            private_key=_env("POLYMARKET_PK"),
            api_key=_env("POLYMARKET_API_KEY"),
            api_secret=_env("POLYMARKET_SECRET"),
            api_passphrase=_env("POLYMARKET_PASSPHRASE"),
            signature_type=sig_type,
            funder=_env("POLYMARKET_FUNDER"),
            rpc_url=_env("POLYGON_RPC_URL"),
            ws_rpc_url=_env("POLYGON_WS_URL"),
        )

    @property
    def has_l1(self) -> bool:
        """Level 1 auth: a private key, enough to sign orders and derive creds."""
        return bool(self.private_key)

    @property
    def has_l2(self) -> bool:
        """Level 2 auth: the full API credential triplet for posting orders."""
        return bool(self.api_key and self.api_secret and self.api_passphrase)

    def problems(self) -> list[str]:
        """Blocking issues that make live trading impossible or unsafe."""
        issues: list[str] = []
        if not self.has_l1:
            issues.append("POLYMARKET_PK is not set (required to sign orders)")
        if self.signature_type in PROXY_SIG_TYPES and not self.funder:
            issues.append(
                f"POLYMARKET_SIGNATURE_TYPE={self.signature_type} is a proxy-wallet scheme, "
                "so POLYMARKET_FUNDER must name the address holding the USDC"
            )
        if self.signature_type not in (SIG_TYPE_EOA, *PROXY_SIG_TYPES):
            issues.append(f"unsupported POLYMARKET_SIGNATURE_TYPE={self.signature_type}")
        return issues

    def describe(self) -> str:
        return (
            f"sig_type={self.signature_type} "
            f"pk={'set' if self.has_l1 else 'MISSING'} "
            f"api_creds={'set' if self.has_l2 else 'will-derive'} "
            f"funder={self.funder or '-'}"
        )


def _env(name: str) -> str | None:
    value = os.getenv(name)
    value = value.strip() if value else ""
    return value or None


#: Bridged USDC (USDC.e) on Polygon - the collateral Polymarket settles in.
USDC_POLYGON = "0x2791Bca1f2de4661eD88A30C99A7a9449Aa84174"
#: ERC-20 selectors: balanceOf(address) and allowance(address,address).
ERC20_BALANCE_OF = "0x70a08231"
ERC20_ALLOWANCE = "0xdd62ed3e"
USDC_DECIMALS = 1_000_000


class PolygonRpc:
    """Minimal JSON-RPC client for reading USDC balance and allowance.

    Deliberately raw eth_calls over the shared aiohttp session rather than a
    web3 dependency: two ERC-20 selectors need no ABI machinery, and reusing the
    session keeps the pooled TLS connection to the provider warm. A private
    endpoint (Alchemy / QuickNode / Chainstack) answers these in well under the
    Polymarket API's own latency, and does not go down when Polymarket does.
    """

    def __init__(self, session: aiohttp.ClientSession, url: str) -> None:
        self._session = session
        self._url = url
        self._id = 0

    async def _call(self, to: str, data: str) -> int:
        self._id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._id,
            "method": "eth_call",
            "params": [{"to": to, "data": data}, "latest"],
        }
        async with self._session.post(
            self._url, json=payload, timeout=aiohttp.ClientTimeout(total=5)
        ) as resp:
            if resp.status in RETRYABLE_STATUS:
                raise RetryableError(f"RPC HTTP {resp.status}", status=resp.status)
            body = await resp.read()
        parsed = json_loads(body)
        if "error" in parsed:
            raise RuntimeError(f"RPC error: {parsed['error']}")
        result = parsed.get("result") or "0x"
        return int(result, 16) if result not in ("0x", "") else 0

    @staticmethod
    def _addr_arg(address: str) -> str:
        return address.lower().removeprefix("0x").rjust(64, "0")

    async def usdc_balance(self, owner: str) -> float:
        raw = await self._call(USDC_POLYGON, ERC20_BALANCE_OF + self._addr_arg(owner))
        return raw / USDC_DECIMALS

    async def usdc_allowance(self, owner: str, spender: str) -> float:
        data = ERC20_ALLOWANCE + self._addr_arg(owner) + self._addr_arg(spender)
        raw = await self._call(USDC_POLYGON, data)
        return raw / USDC_DECIMALS


@dataclass(slots=True)
class ExecutionSettings:
    dry_run: bool = True
    order_size: float = 20.0  # fallback contracts/leg when sizing is disabled
    max_notional: float = 50.0  # hard USDC cap per leg
    slippage_ticks: int = 0  # ticks of price improvement offered to cross
    max_consecutive_errors: int = 3  # trip the breaker after this many


@dataclass(slots=True)
class RiskSettings:
    """Account-level risk limits. All are session-scoped."""

    #: Fraction of bankroll put at risk per leg. For a binary the entire
    #: premium is at risk, so risk == size * price.
    max_risk_pct: float = 0.02
    dynamic_sizing: bool = True
    #: Session drawdown limits, measured peak-to-trough on equity.
    max_drawdown_pct: float = 0.05
    max_drawdown_usd: float = 100.0
    #: Losing closes (or failed executions) in a row before halting.
    max_consecutive_losses: int = 3
    #: Protective pause after a drawdown trip, then trading resumes.
    risk_cooldown: float = 300.0
    #: Drawdown trips tolerated before the session halts for good.
    max_drawdown_trips: int = 3
    #: Simulated bankroll for dry runs, where there is no wallet to read.
    paper_bankroll: float = 1_000.0
    #: Venue floor; overridden per-market by `orderMinSize` when available.
    min_order_size: float = 5.0


@dataclass(slots=True)
class Position:
    market: str
    token_id: str
    label: str  # "YES" / "NO"
    size: float
    entry: float
    opened: float
    mark: float  # latest observable exit price

    @property
    def cost(self) -> float:
        return self.size * self.entry

    @property
    def value(self) -> float:
        return self.size * self.mark

    @property
    def unrealized(self) -> float:
        return self.value - self.cost


class RiskManager:
    """Bankroll, position sizing, and the session-level circuit breakers.

    Equity is tracked as `bankroll + realized + unrealized`, where unrealized
    marks open positions to the best bid (the price we could actually exit at,
    not the mid). Drawdown is measured peak-to-trough on that equity.
    """

    def __init__(self, settings: RiskSettings) -> None:
        self._s = settings
        self.settings = settings
        self.balance: float | None = None
        self.allowance: float | None = None
        self.bankroll: float = 0.0
        self.committed: float = 0.0  # capital reserved for in-flight orders
        self.positions: dict[str, Position] = {}
        self.realized_pnl: float = 0.0
        self.peak_equity: float = 0.0
        self.consecutive_losses: int = 0
        self.wins: int = 0
        self.losses: int = 0
        self.drawdown_trips: int = 0
        self.halted_until: float = 0.0
        self.halted_permanently: bool = False
        self.halt_reason: str = ""
        self._lock = asyncio.Lock()

    # -- bankroll ----------------------------------------------------------- #

    def seed_paper_bankroll(self) -> None:
        self.bankroll = self._s.paper_bankroll
        self.peak_equity = self.equity()
        log.info("Paper bankroll seeded at $%.2f", self.bankroll)

    async def refresh_balance_via_rpc(
        self, rpc: "PolygonRpc", owner: str, neg_risk: bool = False
    ) -> bool:
        """Read balance and allowance straight from Polygon. Faster path."""
        from py_clob_client.config import get_contract_config

        exchange = get_contract_config(POLYGON_CHAIN_ID, neg_risk).exchange
        try:
            balance, allowance = await asyncio.gather(
                retry_async(rpc.usdc_balance, owner, attempts=3, label="RPC balanceOf"),
                retry_async(
                    rpc.usdc_allowance, owner, exchange, attempts=3, label="RPC allowance"
                ),
            )
        except Exception as exc:  # noqa: BLE001 - caller falls back to the CLOB
            log.warning("Direct RPC balance read failed (%s); falling back to CLOB", exc)
            return False

        self.balance = balance
        self.allowance = allowance
        self.bankroll = balance
        self.peak_equity = self.equity()
        log.warning(
            "Wallet via private RPC: USDC balance $%.2f | CTF Exchange allowance $%.2f",
            balance,
            allowance,
        )
        return self._validate_allowance()

    def _validate_allowance(self) -> bool:
        if self.allowance is not None and self.allowance <= 0.0:
            log.error(
                "USDC allowance to the CTF Exchange is zero - every order will be "
                "rejected on-chain. Approve the exchange before trading."
            )
            return False
        if self.allowance is not None and self.balance is not None and self.allowance < self.balance:
            log.warning(
                "Allowance ($%.2f) is below balance ($%.2f); only the approved "
                "portion is actually usable.",
                self.allowance,
                self.balance,
            )
        return True

    async def refresh_balance(self, client: ClobClient, signature_type: int) -> bool:
        """Read on-chain USDC balance and CTF-Exchange allowance via the CLOB.

        The CLOB reports the wallet's actual Polygon USDC balance and its
        allowance to the exchange contract, which is the number that decides
        whether an order is accepted. Reading it here rather than through a
        direct RPC keeps the check aligned with the venue's own view and avoids
        a web3 dependency plus RPC configuration.
        """
        require_polymarket_sdk("Reading balance through the Polymarket CLOB")
        params = BalanceAllowanceParams(
            asset_type=AssetType.COLLATERAL,
            signature_type=signature_type,
        )
        try:
            raw = await retry_async(
                asyncio.to_thread,
                client.get_balance_allowance,
                params,
                attempts=3,
                label="balance/allowance",
            )
        except Exception as exc:  # noqa: BLE001 - caller decides whether to trade
            log.error("Could not read USDC balance/allowance: %s", exc)
            return False

        self.balance = _usdc(raw, "balance")
        self.allowance = _usdc(raw, "allowance", "allowances")
        if self.balance is None:
            log.error("Balance response had no parseable balance field: %r", raw)
            return False

        self.bankroll = self.balance
        self.peak_equity = self.equity()
        log.warning(
            "Wallet: USDC balance $%.2f | CTF Exchange allowance %s",
            self.balance,
            f"${self.allowance:.2f}" if self.allowance is not None else "unknown",
        )

        return self._validate_allowance()

    # -- sizing ------------------------------------------------------------- #

    def cash(self) -> float:
        """Spendable cash: starting balance adjusted by realized PnL.

        Realized losses must reduce buying power, otherwise a losing session
        keeps sizing off the balance it started with.
        """
        return self.bankroll + self.realized_pnl

    def available(self) -> float:
        """Cash not already spent on open positions or reserved in flight."""
        spent = sum(p.cost for p in self.positions.values())
        usable = self.cash()
        if self.allowance is not None:
            usable = min(usable, self.allowance)
        return max(0.0, usable - spent - self.committed)

    def size_for(self, price: float, max_notional: float, min_size: float) -> float:
        """Contracts to buy at `price`, respecting risk %, caps and free funds.

        Returns 0.0 when the position cannot be opened at a legal size, which
        the caller must treat as "do not trade".
        """
        if price <= 0.0:
            return 0.0

        # Size off current equity, not the opening balance, so risk per trade
        # scales down as the session loses ground.
        base = max(0.0, self.equity())
        budget = base * self._s.max_risk_pct if self._s.dynamic_sizing else max_notional
        budget = min(budget, max_notional, self.available())
        if budget <= 0.0:
            return 0.0

        size = math.floor((budget / price) * 100.0) / 100.0  # venue accepts 2dp
        return size if size >= min_size else 0.0

    async def reserve(self, amount: float) -> bool:
        """Ring-fence capital for an in-flight order.

        The cross-book path submits both legs concurrently; without a
        reservation each leg would independently see the full free balance and
        together they could commit more than the wallet holds.
        """
        async with self._lock:
            if amount > self.available():
                return False
            self.committed += amount
            return True

    def release(self, amount: float) -> None:
        self.committed = max(0.0, self.committed - amount)

    # -- position lifecycle ------------------------------------------------- #

    @staticmethod
    def key(market: str, token_id: str) -> str:
        return f"{market}:{token_id}"

    def open_position(
        self,
        market: str,
        token_id: str,
        label: str,
        size: float,
        price: float,
        reserved: float = 0.0,
    ) -> Position:
        """Record a fill, converting its reservation into position cost.

        `reserved` is released here rather than by the executor so the capital
        moves from committed to spent atomically. Releasing earlier would leave
        a window in which the funds look free to a concurrently sizing leg.
        """
        if reserved:
            self.release(reserved)
        pos = Position(
            market=market,
            token_id=token_id,
            label=label,
            size=size,
            entry=price,
            opened=time.monotonic(),
            mark=price,
        )
        self.positions[self.key(market, token_id)] = pos
        self._update_peak()
        return pos

    def mark_to_market(self, token_id: str, exit_price: float | None) -> None:
        if exit_price is None:
            return
        for pos in self.positions.values():
            if pos.token_id == token_id:
                pos.mark = exit_price
        self._update_peak()

    def close_market(self, market: str) -> float:
        """Realize every position in a settled market at its last mark.

        Marks are used rather than confirmed settlement because the breakers
        need a fast risk signal; by expiry the book has converged to ~0 or ~1,
        so the mark is a close proxy. It is not accounting-grade PnL.
        """
        total = 0.0
        for key in [k for k, p in self.positions.items() if p.market == market]:
            pos = self.positions.pop(key)
            pnl = pos.unrealized
            total += pnl
            self.realized_pnl += pnl
            if pnl < 0:
                self.losses += 1
                self.consecutive_losses += 1
            else:
                self.wins += 1
                self.consecutive_losses = 0
            log.info(
                "Closed %s %s: %g @ %.3f -> %.3f = %+.2f USDC (session %+.2f)",
                pos.market,
                pos.label,
                pos.size,
                pos.entry,
                pos.mark,
                pnl,
                self.realized_pnl,
            )
        if total:
            self._update_peak()
        return total

    def note_execution_failure(self) -> None:
        """A failed execution counts toward the consecutive-loss limit."""
        self.consecutive_losses += 1

    # -- equity and breakers ------------------------------------------------ #

    def unrealized_pnl(self) -> float:
        return sum(p.unrealized for p in self.positions.values())

    def equity(self) -> float:
        return self.bankroll + self.realized_pnl + self.unrealized_pnl()

    def drawdown(self) -> float:
        """Peak-to-trough equity loss, always >= 0."""
        return max(0.0, self.peak_equity - self.equity())

    def _update_peak(self) -> None:
        self.peak_equity = max(self.peak_equity, self.equity())

    def check_breakers(self) -> str | None:
        """Evaluate the session breakers. Returns a reason if one just tripped."""
        if self.halted_permanently:
            return None

        if self.consecutive_losses >= self._s.max_consecutive_losses:
            self.halted_permanently = True
            self.halt_reason = (
                f"{self.consecutive_losses} consecutive losing or failed trades"
            )
            log.error(
                "RISK HALT (permanent): %s. No further orders this session.", self.halt_reason
            )
            return self.halt_reason

        dd = self.drawdown()
        pct_limit = self.peak_equity * self._s.max_drawdown_pct
        limit = min(pct_limit, self._s.max_drawdown_usd) if self.peak_equity > 0 else self._s.max_drawdown_usd
        if dd >= limit > 0 and time.monotonic() >= self.halted_until:
            self.drawdown_trips += 1
            reason = (
                f"session drawdown ${dd:.2f} breached the ${limit:.2f} limit "
                f"({self._s.max_drawdown_pct:.1%} of peak equity ${self.peak_equity:.2f} "
                f"vs ${self._s.max_drawdown_usd:.2f} cap)"
            )
            self.halt_reason = reason
            if self.drawdown_trips >= self._s.max_drawdown_trips:
                self.halted_permanently = True
                log.error(
                    "RISK HALT (permanent): %s. Trip %d of %d - session over.",
                    reason,
                    self.drawdown_trips,
                    self._s.max_drawdown_trips,
                )
            else:
                self.halted_until = time.monotonic() + self._s.risk_cooldown
                # Re-baseline the peak, or the same loss re-trips instantly on
                # resume. This is why trips are counted and capped: total
                # session loss can reach roughly trips x limit.
                self.peak_equity = self.equity()
                log.error(
                    "RISK COOLDOWN: %s. Pausing new entries for %gs (trip %d of %d).",
                    reason,
                    self._s.risk_cooldown,
                    self.drawdown_trips,
                    self._s.max_drawdown_trips,
                )
            return reason
        return None

    def trading_allowed(self) -> tuple[bool, str]:
        if self.halted_permanently:
            return False, f"halted: {self.halt_reason}"
        if time.monotonic() < self.halted_until:
            remaining = self.halted_until - time.monotonic()
            return False, f"risk cooldown, {remaining:.0f}s remaining"
        return True, ""

    def stats(self) -> str:
        allowed, why = self.trading_allowed()
        return (
            f"eq=${self.equity():.2f} dd=${self.drawdown():.2f} "
            f"real=${self.realized_pnl:+.2f} unreal=${self.unrealized_pnl():+.2f} "
            f"w/l={self.wins}/{self.losses} streak={self.consecutive_losses}"
            + ("" if allowed else f" [{why}]")
        )


def _usdc(payload, *keys: str) -> float | None:
    """Pull a USDC amount out of a balance/allowance response.

    The CLOB reports raw 6-decimal integer strings, but has historically also
    returned plain decimals and a nested allowances map, so all three shapes are
    accepted rather than assuming one and silently sizing to zero.
    """
    if not isinstance(payload, dict):
        return None

    for key in keys:
        if key not in payload:
            continue
        raw = payload[key]
        if isinstance(raw, dict):  # nested allowances map: take the largest
            candidates = [_scale_usdc(v) for v in raw.values()]
            candidates = [c for c in candidates if c is not None]
            if candidates:
                return max(candidates)
            continue
        scaled = _scale_usdc(raw)
        if scaled is not None:
            return scaled
    return None


def _scale_usdc(raw) -> float | None:
    if isinstance(raw, (int, float)):
        return float(raw) / 1e6 if float(raw) > 1e5 else float(raw)
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    try:
        value = float(text)
    except ValueError:
        return None
    # An integer string is raw 6-decimal units; a decimal string is already USDC.
    return value / 1e6 if "." not in text else value


@dataclass(slots=True)
class ExecutionResult:
    ok: bool
    dry_run: bool
    token_id: str
    side: str
    price: float
    size: float
    filled_size: float = 0.0
    order_id: str | None = None
    status: str = ""
    error: str | None = None
    #: Capital still committed for this fill, released by risk.open_position().
    reserved: float = 0.0

    @property
    def notional(self) -> float:
        return self.filled_size * self.price

    def summary(self) -> str:
        tag = "SIMULATED" if self.dry_run else "LIVE"
        if not self.ok:
            return f"{tag} {self.side} REJECTED: {self.error}"
        return (
            f"{tag} {self.side} {self.filled_size:g} @ {self.price:.3f} "
            f"(${self.notional:.2f}) status={self.status or 'filled'}"
            + (f" id={self.order_id}" if self.order_id else "")
        )


def _tick_str(tick: float) -> str:
    """Render a tick size the way the CLOB expects it ("0.01", "0.001")."""
    text = f"{tick:.6f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def quantize_price(price: float, tick: float, side: str) -> float:
    """Snap a price onto the venue's tick grid.

    The CLOB rejects off-grid prices outright. Buys round *up* and sells round
    *down* so float error can never push a marketable order to the passive side
    of the spread, where it would rest instead of crossing.
    """
    tick = tick if tick > 0 else 0.01
    steps = price / tick
    # Nudge before rounding so a price already on-grid is not bumped a full
    # tick by binary representation error (0.45/0.01 == 44.99999...).
    rounded = math.ceil(steps - 1e-9) if side == BUY else math.floor(steps + 1e-9)
    snapped = rounded * tick
    decimals = max(0, round(-math.log10(tick)))
    return min(max(round(snapped, decimals), tick), round(1.0 - tick, decimals))


@dataclass(slots=True)
class OrderTemplate:
    """Everything about an order that is knowable before the signal fires.

    Built once per token at market discovery so the hot path only fills in
    price and size. Crucially this also warms py-clob-client's three internal
    caches (tick size, neg-risk flag, fee rate), each of which is otherwise a
    *blocking HTTP GET issued from inside create_order* on the first order for
    a token. These markets roll every five minutes with brand-new token ids, so
    without pre-warming every new window's first order - the one the whole
    system exists to send quickly - pays all three round-trips before signing
    even begins.
    """

    token_id: str
    tick_size: float
    neg_risk: bool
    options: PartialCreateOrderOptions
    fee_rate_bps: int = 0
    warmed: bool = False


class Executor:
    """Order submission with a dry-run mode, a position gate and a breaker.

    Every venue call is dispatched through `asyncio.to_thread`: py-clob-client
    is synchronous, and a blocking HTTP round-trip on the event loop would stall
    the Binance tape - which is the one thing this whole system cannot afford.
    """

    def __init__(
        self,
        settings: ExecutionSettings,
        creds: Credentials | None = None,
        risk: RiskManager | None = None,
    ) -> None:
        self._settings = settings
        self._creds = creds or Credentials()
        self._client: ClobClient | None = None
        self.risk = risk if risk is not None else RiskManager(RiskSettings())
        #: Market slugs currently holding an open trade. The position gate.
        #: Owned here, not by the tape, so a websocket reconnect cannot drop it.
        self.active_positions: set[str] = set()
        self._gate_lock = asyncio.Lock()
        self._consecutive_errors = 0
        self._halted = False
        self.orders_sent = 0
        self.orders_filled = 0
        self.expected_pnl = 0.0
        self.latency = LatencyProfiler()
        #: token_id -> pre-built order template, populated off the hot path.
        self._templates: dict[str, OrderTemplate] = {}
        self._warm_failures = 0
        self._rpc: PolygonRpc | None = None

    def attach_session(self, session: aiohttp.ClientSession) -> None:
        """Share the pooled HTTP session, enabling the private-RPC fast path."""
        if self._creds.rpc_url:
            self._rpc = PolygonRpc(session, self._creds.rpc_url)
            log.info("Private Polygon RPC configured for balance reads")
        if self._creds.ws_rpc_url:
            log.info(
                "POLYGON_WS_URL is set but unused: the trading path is REST-only. "
                "It is plumbed through for on-chain subscriptions."
            )

    def _signer_address(self) -> str | None:
        try:
            return self._client.get_address() if self._client else None
        except Exception:  # noqa: BLE001 - address is best-effort
            return None

    # -- lifecycle ---------------------------------------------------------- #

    @property
    def dry_run(self) -> bool:
        return self._settings.dry_run

    @property
    def halted(self) -> bool:
        return self._halted

    @property
    def order_size(self) -> float:
        return self._settings.order_size

    @property
    def slippage_ticks(self) -> int:
        return self._settings.slippage_ticks

    def size_for(
        self, price: float, market: "TrackedMarket", pair_price: float | None = None
    ) -> float:
        """Contracts to buy, from the risk budget and the venue's minimum.

        `pair_price` is the combined cost of both legs of a cross-book trade:
        the risk budget must cover the pair, not each leg independently, or the
        position ends up at twice the intended size.
        """
        if not self.risk.settings.dynamic_sizing:
            return self._settings.order_size
        # Dividing the budget by the pair cost yields the number of PAIRS, which
        # is the per-leg size. It also caps the pair (not each leg) at
        # --max-notional, which is the conservative reading.
        return self.risk.size_for(
            pair_price if pair_price is not None else price,
            self._settings.max_notional,
            market.min_order_size,
        )

    async def connect(self) -> bool:
        """Build the signing client. Returns False if live trading can't start."""
        if self._settings.dry_run:
            log.info("DRY RUN: simulating fills, no orders will be submitted")
            # Still surface credential problems so the first live run is not the
            # first time anyone discovers the wallet is misconfigured.
            for issue in self._creds.problems():
                log.info("  (live mode would fail: %s)", issue)
            self.risk.seed_paper_bankroll()
            return True

        problems = self._creds.problems()
        if problems:
            for issue in problems:
                log.error("Cannot trade live: %s", issue)
            return False

        if not POLYMARKET_SDK:
            log.error(
                "Live Polymarket trading needs the CLOB SDK: pip install py-clob-client"
            )
            return False

        try:
            self._client = await asyncio.to_thread(self._build_client)
        except Exception as exc:  # noqa: BLE001 - auth failures must be loud
            log.error("Failed to initialize signing CLOB client: %s", exc)
            return False

        # Fail closed: without a confirmed balance and allowance we cannot size
        # a position or know an order will be accepted, so we do not trade.
        # Prefer the private RPC when configured, fall back to the CLOB.
        ok = False
        owner = self._creds.funder or self._signer_address()
        if self._rpc is not None and owner:
            ok = await self.risk.refresh_balance_via_rpc(self._rpc, owner)
        if not ok:
            ok = await self.risk.refresh_balance(self._client, self._creds.signature_type)
        if not ok:
            log.error("Balance/allowance preflight failed; refusing to trade live")
            return False

        log.warning(
            "LIVE TRADING ARMED | %s | bankroll=$%.2f risk/trade=%.1f%% max_notional=$%.2f",
            self._creds.describe(),
            self.risk.bankroll,
            self.risk.settings.max_risk_pct * 100.0,
            self._settings.max_notional,
        )
        return True

    def _build_client(self) -> ClobClient:
        """Synchronous client construction; runs in a worker thread."""
        kwargs: dict = {
            "chain_id": POLYGON_CHAIN_ID,
            "key": self._creds.private_key,
            "signature_type": self._creds.signature_type,
        }
        if self._creds.funder:
            kwargs["funder"] = self._creds.funder

        client = ClobClient(CLOB_HOST, **kwargs)

        if self._creds.has_l2:
            creds = ApiCreds(
                api_key=self._creds.api_key,
                api_secret=self._creds.api_secret,
                api_passphrase=self._creds.api_passphrase,
            )
            log.info("Using API credentials from the environment")
        else:
            # L1 (the private key) is enough to mint or recover the L2 triplet.
            creds = client.create_or_derive_api_creds()
            log.info("Derived API credentials from POLYMARKET_PK (api_key=%s)", creds.api_key)

        client.set_api_creds(creds)
        return client

    # -- order pre-caching -------------------------------------------------- #

    def template_for(self, token_id: str, market: "TrackedMarket") -> OrderTemplate:
        """Cached template for a token, created on demand if discovery missed it."""
        tmpl = self._templates.get(token_id)
        if tmpl is None:
            tmpl = OrderTemplate(
                token_id=token_id,
                tick_size=market.tick_size,
                neg_risk=market.neg_risk,
                options=PartialCreateOrderOptions(
                    tick_size=_tick_str(market.tick_size), neg_risk=market.neg_risk
                ),
            )
            self._templates[token_id] = tmpl
        elif tmpl.tick_size != market.tick_size or tmpl.neg_risk != market.neg_risk:
            # The live book tightened the tick (or neg-risk flipped); rebuild the
            # options object so the signed order matches the venue's grid.
            tmpl.tick_size = market.tick_size
            tmpl.neg_risk = market.neg_risk
            tmpl.options = PartialCreateOrderOptions(
                tick_size=_tick_str(market.tick_size), neg_risk=market.neg_risk
            )
        return tmpl

    async def prepare_market(self, market: "TrackedMarket") -> None:
        """Warm order metadata for a newly discovered market, off the hot path.

        Runs in a worker thread: the three getters below each issue a blocking
        HTTP GET on a cold token, and paying that here costs nothing, whereas
        paying it inside create_order costs the trade.
        """
        for token_id in (market.up_token, market.down_token):
            tmpl = self.template_for(token_id, market)
            if tmpl.warmed or self._client is None:
                continue
            try:
                fee = await asyncio.to_thread(self._warm_token, token_id)
            except Exception as exc:  # noqa: BLE001 - warming is best-effort
                self._warm_failures += 1
                log.warning(
                    "Order metadata pre-warm failed for %s (the first order in "
                    "this market will pay the round-trips): %s",
                    market.slug,
                    exc,
                )
                continue
            tmpl.fee_rate_bps = fee
            tmpl.warmed = True
            log.debug("Pre-warmed order metadata for %s / %s", market.slug, token_id[:12])

    def _warm_token(self, token_id: str) -> int:
        """Populate py-clob-client's tick/neg-risk/fee caches. Blocking."""
        assert self._client is not None
        self._client.get_tick_size(token_id)
        self._client.get_neg_risk(token_id)
        return int(self._client.get_fee_rate_bps(token_id) or 0)

    @property
    def templates_cached(self) -> int:
        return len(self._templates)

    def forget_market(self, market_slug: str, tokens: Sequence[str]) -> None:
        """Drop cached templates for a retired market.

        Templates are keyed by token id, and every window mints new ones, so
        without eviction the cache grows for the life of the process. Keying by
        token also means one market's template can never be served to another.
        """
        for token_id in tokens:
            self._templates.pop(token_id, None)

    # -- position gate ------------------------------------------------------ #

    async def acquire_market(self, market_key: str) -> bool:
        """Claim the one trade slot for a market window. False if already held.

        This is what stops a single spike - which stays above threshold for many
        consecutive 10 Hz ticks - from being bet on over and over.
        """
        async with self._gate_lock:
            if market_key in self.active_positions:
                return False
            self.active_positions.add(market_key)
            return True

    def release_market(self, market_key: str) -> None:
        """Free the slot: the order never filled, so no position was taken.

        Unconditional and idempotent, so unlike `acquire_market` it needs no
        mutual exclusion - which lets the synchronous market-retirement path
        call it directly.
        """
        self.active_positions.discard(market_key)

    def holds(self, market_key: str) -> bool:
        return market_key in self.active_positions

    # -- execution ---------------------------------------------------------- #

    async def execute_arb_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
        order_type: str = "FOK",
        *,
        tick_size: float = 0.01,
        neg_risk: bool = False,
        template: OrderTemplate | None = None,
        t1_ns: int = 0,
    ) -> ExecutionResult:
        """Submit one order leg. Never raises; always returns a result.

        FOK is the right default here: the edge is a stale resting ask, so the
        order must either take that ask in full right now or die. A partial or
        resting remainder converts a latency arb into an unhedged directional
        bet at exactly the moment the book is repricing against us.
        """
        price = quantize_price(price, tick_size, side)
        notional = price * size

        result = ExecutionResult(
            ok=False,
            dry_run=self._settings.dry_run,
            token_id=token_id,
            side=side,
            price=price,
            size=size,
        )

        if self._halted:
            result.error = "execution halted by circuit breaker"
            return result

        allowed, why = self.risk.trading_allowed()
        if not allowed:
            result.error = f"risk manager blocked the order ({why})"
            log.warning("Order blocked: %s", result.error)
            return result

        if notional > self._settings.max_notional:
            result.error = (
                f"notional ${notional:.2f} exceeds --max-notional "
                f"${self._settings.max_notional:.2f}"
            )
            log.error("Order blocked: %s", result.error)
            return result

        # Pre-flight funding check. Reserving up front is what stops the two
        # concurrent cross-book legs from each seeing the full free balance and
        # together committing more than the wallet actually holds.
        if not await self.risk.reserve(notional):
            result.error = (
                f"insufficient unlocked funds: need ${notional:.2f}, "
                f"${self.risk.available():.2f} available "
                f"(bankroll ${self.risk.bankroll:.2f})"
            )
            log.warning("Order blocked: %s", result.error)
            return result

        # T2: the payload is fully determined here - price quantized, size
        # resolved, funds reserved. Everything after this is signing and wire.
        t2_ns = time.perf_counter_ns()
        try:
            outcome = await self._dispatch(
                result, token_id, side, price, size, order_type, tick_size, neg_risk, template
            )
        except BaseException:
            self.risk.release(notional)
            raise
        finally:
            # T3: response in hand (or failure known).
            self.latency.record(t1_ns, t2_ns, time.perf_counter_ns())

        if outcome.ok:
            # Hand the reservation to the caller: it stays committed until
            # risk.open_position() converts it into position cost. Releasing it
            # here would briefly make spent capital look available again.
            outcome.reserved = notional
        else:
            self.risk.release(notional)
        return outcome

    async def _dispatch(
        self,
        result: ExecutionResult,
        token_id: str,
        side: str,
        price: float,
        size: float,
        order_type: str,
        tick_size: float,
        neg_risk: bool,
        template: OrderTemplate | None = None,
    ) -> ExecutionResult:
        if self._settings.dry_run:
            # Assume the resting liquidity we just measured is still there. The
            # caller only reaches this path when our limit crosses the ask.
            result.ok = True
            result.filled_size = size
            result.status = "simulated-fill"
            self.orders_sent += 1
            self.orders_filled += 1
            return result

        if self._client is None:
            result.error = "signing client not initialized"
            return result

        try:
            self.orders_sent += 1
            # No retry wrapper here, by design: a POST that times out may still
            # have executed, and a duplicate fill is far worse than a miss.
            response = await asyncio.to_thread(
                self._submit,
                token_id,
                side,
                price,
                size,
                order_type,
                tick_size,
                neg_risk,
                template,
            )
        except Exception as exc:  # noqa: BLE001 - venue errors are routine
            self._note_error()
            self.risk.note_execution_failure()
            self.risk.check_breakers()
            result.error = f"{type(exc).__name__}: {exc}"
            log.error("Order submission failed (%s %s @ %.3f): %s", side, token_id[:12], price, exc)
            return result

        self._consecutive_errors = 0
        result.order_id = str(response.get("orderID") or response.get("orderId") or "") or None
        result.status = str(response.get("status") or "")
        # `success` is the transport-level ack; `status` carries the fill state.
        accepted = bool(response.get("success", True)) and result.status.lower() not in {
            "unmatched",
            "delayed",
            "cancelled",
            "canceled",
        }

        if accepted:
            result.ok = True
            result.filled_size = _filled_size(response, size)
            self.orders_filled += 1
        else:
            result.error = str(response.get("errorMsg") or response.get("error") or result.status)
            # An unmatched FOK is a missed trade, not a venue fault - but a run
            # of them still counts toward the consecutive-loss limit.
            self.risk.note_execution_failure()
            self.risk.check_breakers()

        return result

    def _submit(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
        order_type: str,
        tick_size: float,
        neg_risk: bool,
        template: OrderTemplate | None = None,
    ) -> dict:
        """Blocking sign-and-post. Runs in a worker thread."""
        require_polymarket_sdk("Submitting a Polymarket order")
        assert self._client is not None
        # Reuse the pre-built options object rather than allocating one per
        # order; more importantly, a warmed template means create_order's
        # internal tick/neg-risk/fee lookups all hit cache instead of the wire.
        options = (
            template.options
            if template is not None
            else PartialCreateOrderOptions(tick_size=_tick_str(tick_size), neg_risk=neg_risk)
        )
        signed = self._client.create_order(
            OrderArgs(token_id=token_id, price=price, size=size, side=side),
            options,
        )
        resolved = getattr(OrderType, order_type.upper(), OrderType.FOK)
        response = self._client.post_order(signed, resolved)
        return response if isinstance(response, dict) else {"success": True, "raw": str(response)}

    def _note_error(self) -> None:
        self._consecutive_errors += 1
        if self._consecutive_errors >= self._settings.max_consecutive_errors and not self._halted:
            self._halted = True
            log.error(
                "CIRCUIT BREAKER TRIPPED after %d consecutive execution errors; "
                "no further orders will be submitted this session",
                self._consecutive_errors,
            )

    # -- accounting --------------------------------------------------------- #

    def record_expected_pnl(self, amount: float) -> None:
        self.expected_pnl += amount

    def stats(self) -> str:
        return (
            f"orders={self.orders_filled}/{self.orders_sent} "
            f"exp_pnl=${self.expected_pnl:+.2f} "
            f"open={len(self.active_positions)}"
            + (" HALTED" if self._halted else "")
        )


def _filled_size(response: dict, requested: float) -> float:
    """Best-effort fill size across the shapes the CLOB returns."""
    for key in ("sizeMatched", "size_matched", "matchedAmount", "makingAmount"):
        if key in response:
            try:
                return float(response[key])
            except (TypeError, ValueError):
                continue
    return requested


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
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)
    risk: RiskSettings = field(default_factory=RiskSettings)


class SignalEngine:
    def __init__(
        self,
        buffer: PriceBuffer,
        books: BookFeed,
        cfg: Config,
        executor: Executor | None = None,
    ) -> None:
        self._buffer = buffer
        self._books = books
        self._cfg = cfg
        self._executor = executor
        self._markets: dict[str, TrackedMarket] = {}
        self._last_forced_fetch = 0.0
        self._risk_block_logged = ""

    @property
    def executor(self) -> Executor | None:
        return self._executor

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
            retired = self._markets.pop(slug)
            # The window is over, so any position in it has settled. Realize the
            # PnL and free the gate slot, or the set would grow without bound.
            if self._executor is not None:
                self._executor.risk.close_market(slug)
                self._executor.risk.check_breakers()
                self._executor.release_market(slug)
                self._executor.forget_market(slug, (retired.up_token, retired.down_token))

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
        empty = Quote(bid=None, ask=None)
        for market in markets:
            up_q = quotes.get(market.up_token, empty)
            down_q = quotes.get(market.down_token, empty)
            up_bid, up_ask = up_q.bid, up_q.ask
            down_bid, down_ask = down_q.bid, down_q.ask

            # Adopt the venue's live tick/min-size over the Gamma snapshot.
            for quote in (up_q, down_q):
                if quote.tick_size:
                    market.tick_size = quote.tick_size
                if quote.min_order_size:
                    market.min_order_size = quote.min_order_size
                if quote.neg_risk is not None:
                    market.neg_risk = quote.neg_risk

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
            if self._executor is not None:
                # Mark to the best BID - the price we could actually exit at.
                # Marking to the mid would flatter the book and let a real
                # drawdown hide from the breaker.
                #
                # When a side has no bid at all, fall back to the complement of
                # the opposite ask: the pair must sum to $1, so an Up with no
                # bid and a Down offered at 0.99 is worth about 0.01. Without
                # this a position that has gone worthless keeps its entry mark
                # and stays invisible to the drawdown breaker right up until it
                # settles - exactly when the breaker most needs to see it.
                self._executor.risk.mark_to_market(
                    market.up_token, _exit_mark(up_bid, down_ask)
                )
                self._executor.risk.mark_to_market(
                    market.down_token, _exit_mark(down_bid, up_ask)
                )

        if self._executor is not None:
            self._executor.risk.check_breakers()

    # -- evaluation --------------------------------------------------------- #

    async def evaluate(self) -> None:
        markets = self.active_markets()
        if not markets or not self._buffer.ready:
            return

        # A tripped breaker suppresses signal generation entirely, not just
        # order submission: logging tradeable edges we have no intention of
        # taking trains the operator to ignore the banner.
        if self._executor is not None:
            allowed, why = self._executor.risk.trading_allowed()
            if not allowed:
                if self._risk_block_logged != why:
                    log.warning("Signals suppressed: %s", why)
                    self._risk_block_logged = why
                return
            self._risk_block_logged = ""

        move = self._buffer.largest_move(self._cfg.spike_lookback)
        # Re-arm the tripwire from the window we just measured, so the next
        # qualifying tick wakes this loop instead of waiting for the timer.
        self._buffer.arm_trigger(self._cfg.spike_bps)
        if move is None or abs(move.bps) < self._cfg.spike_bps:
            return

        # Refresh before judging: the whole question is whether the book has
        # ALREADY repriced. A stale local snapshot would manufacture fake edge.
        await self.refresh_books(force=True)

        sigma = self._buffer.sigma_per_sqrt_second()
        for market in self.active_markets():
            await self._evaluate_market(market, move, sigma)

    async def _evaluate_market(self, market: TrackedMarket, move: Move, sigma: float) -> None:
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

        await self._check_cross_book(market, current, move)

        if move.delta_log > 0:
            await self._check_leg(
                market, move, sigma, "UP", "YES", current.up_ask, p_fair_up, p0, tau_eff
            )
        else:
            await self._check_leg(
                market, move, sigma, "DOWN", "NO", current.down_ask, p_fair_down, 1.0 - p0, tau_eff
            )

    async def _check_leg(
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

        token_id = market.up_token if side == "UP" else market.down_token
        await self._execute_model_signal(
            market, token_id, token_label, ask, p_fair, move.t1_ns
        )

    async def _execute_model_signal(
        self,
        market: TrackedMarket,
        token_id: str,
        token_label: str,
        ask: float,
        p_fair: float,
        t1_ns: int = 0,
    ) -> None:
        """Single-leg take of a stale ask, behind the position gate."""
        executor = self._executor
        if executor is None:
            return

        if not await executor.acquire_market(market.slug):
            log.info(
                "Position gate: already holding %s, skipping duplicate %s entry",
                market.slug,
                token_label,
            )
            return

        limit = quantize_price(
            ask + executor.slippage_ticks * market.tick_size, market.tick_size, BUY
        )
        size = executor.size_for(limit, market)
        if size <= 0.0:
            log.warning(
                " execution   : %s | SKIPPED - no legal size at %.3f "
                "(available $%.2f, min size %g)",
                market.slug,
                limit,
                executor.risk.available(),
                market.min_order_size,
            )
            executor.release_market(market.slug)
            return

        result = await executor.execute_arb_order(
            token_id,
            BUY,
            limit,
            size,
            order_type="FOK",
            tick_size=market.tick_size,
            neg_risk=market.neg_risk,
            template=executor.template_for(token_id, market),
            t1_ns=t1_ns,
        )

        if result.ok:
            executor.risk.open_position(
                market.slug,
                token_id,
                token_label,
                result.filled_size,
                result.price,
                reserved=result.reserved,
            )
            expected = result.filled_size * (p_fair - result.price)
            executor.record_expected_pnl(expected)
            log.warning(
                " execution   : %s | %s | expected PnL %+.2f USDC (fair %.3f - paid %.3f)",
                market.slug,
                result.summary(),
                expected,
                p_fair,
                result.price,
            )
        else:
            # Nothing was taken, so the slot must go back or this market is
            # locked out for the rest of its window on a single failed attempt.
            executor.release_market(market.slug)
            log.warning(" execution   : %s | %s", market.slug, result.summary())

    async def _check_cross_book(
        self, market: TrackedMarket, book: BookSnapshot, move: Move
    ) -> None:
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
        await self._execute_cross_book(market, book.up_ask, book.down_ask, move.t1_ns)

    async def _execute_cross_book(
        self, market: TrackedMarket, up_ask: float, down_ask: float, t1_ns: int = 0
    ) -> None:
        """Lift both legs at once. The pair is the position, not either leg."""
        executor = self._executor
        if executor is None:
            return

        if not await executor.acquire_market(market.slug):
            log.info(
                "Position gate: already holding %s, skipping duplicate cross-book entry",
                market.slug,
            )
            return

        slip = executor.slippage_ticks * market.tick_size
        up_limit = quantize_price(up_ask + slip, market.tick_size, BUY)
        down_limit = quantize_price(down_ask + slip, market.tick_size, BUY)

        # Both legs must be the same size or the pair is not actually hedged, so
        # size the pair off their combined cost and use the smaller of the two.
        size = min(
            executor.size_for(up_limit, market, pair_price=up_limit + down_limit),
            executor.size_for(down_limit, market, pair_price=up_limit + down_limit),
        )
        if size <= 0.0:
            log.warning(
                " execution   : %s | SKIPPED - no legal paired size "
                "(available $%.2f, min size %g)",
                market.slug,
                executor.risk.available(),
                market.min_order_size,
            )
            executor.release_market(market.slug)
            return

        # Both legs are submitted concurrently. Sequencing them would leave the
        # second leg exposed to the book moving between the two round-trips,
        # which is precisely the risk this trade is supposed to avoid.
        up_result, down_result = await asyncio.gather(
            executor.execute_arb_order(
                market.up_token,
                BUY,
                up_limit,
                size,
                order_type="FOK",
                tick_size=market.tick_size,
                neg_risk=market.neg_risk,
                template=executor.template_for(market.up_token, market),
                t1_ns=t1_ns,
            ),
            executor.execute_arb_order(
                market.down_token,
                BUY,
                down_limit,
                size,
                order_type="FOK",
                tick_size=market.tick_size,
                neg_risk=market.neg_risk,
                template=executor.template_for(market.down_token, market),
                t1_ns=t1_ns,
            ),
        )

        log.warning(" execution   : %s | UP  leg | %s", market.slug, up_result.summary())
        log.warning(" execution   : %s | DOWN leg | %s", market.slug, down_result.summary())

        for leg, token, label in (
            (up_result, market.up_token, "YES"),
            (down_result, market.down_token, "NO"),
        ):
            if leg.ok:
                executor.risk.open_position(
                    market.slug, token, label, leg.filled_size, leg.price, reserved=leg.reserved
                )

        if up_result.ok and down_result.ok:
            paired = min(up_result.filled_size, down_result.filled_size)
            locked = paired * (1.0 - (up_result.price + down_result.price))
            executor.record_expected_pnl(locked)
            log.warning(
                " execution   : %s | PAIRED %g x (1.000 - %.3f) = %+.2f USDC locked",
                market.slug,
                paired,
                up_result.price + down_result.price,
                locked,
            )
        elif up_result.ok or down_result.ok:
            # One leg filled and the other did not. The hedge is gone and what
            # is left is a naked directional bet - the single worst outcome of
            # this trade, so it is surfaced at ERROR rather than buried.
            filled = up_result if up_result.ok else down_result
            missed = down_result if up_result.ok else up_result
            log.error(
                "LEG RISK on %s: %s leg filled (%g @ %.3f) but %s leg did not (%s). "
                "Position is now UNHEDGED and directional - manual intervention required.",
                market.slug,
                "UP" if up_result.ok else "DOWN",
                filled.filled_size,
                filled.price,
                "DOWN" if up_result.ok else "UP",
                missed.error,
            )
        else:
            executor.release_market(market.slug)

    def _cooldown_ok(self, market: TrackedMarket, key: str) -> bool:
        last = market.last_signal.get(key, 0.0)
        return time.monotonic() - last >= self._cfg.signal_cooldown


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def _exit_mark(own_bid: float | None, opposite_ask: float | None) -> float | None:
    """Best available estimate of what this leg could be sold for."""
    if own_bid is not None:
        return own_bid
    if opposite_ask is not None:
        return max(0.0, min(1.0, 1.0 - opposite_ask))
    return None


def _fmt_clock(unix_ts: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(unix_ts)) + "Z"


class Scanner:
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._buffer = PriceBuffer()
        self._stream = BinanceTradeStream(self._buffer, cfg.ws_endpoints)
        self._books = BookFeed()
        self._executor = Executor(
            cfg.execution, Credentials.from_env(), RiskManager(cfg.risk)
        )
        self._engine = SignalEngine(self._buffer, self._books, cfg, self._executor)
        #: Set by the tape when a tick crosses the spike tripwire.
        self._wake = asyncio.Event()
        self._buffer.set_wake(self._wake.set)

    async def run(self) -> None:
        # One pooled session for every REST call this process makes (Gamma
        # discovery and the private Polygon RPC). Keeping connections alive
        # removes a TCP handshake and a TLS negotiation - together the dominant
        # cost of a cold request - from every call after the first.
        connector = aiohttp.TCPConnector(
            limit=32,
            limit_per_host=16,
            ttl_dns_cache=300,
            keepalive_timeout=90.0,
            enable_cleanup_closed=True,
        )
        async with aiohttp.ClientSession(
            connector=connector,
            headers={"User-Agent": "btc-polymarket-arb/1.0", "Connection": "keep-alive"},
            timeout=aiohttp.ClientTimeout(total=10, connect=3, sock_read=8),
        ) as session:
            self._executor.attach_session(session)
            if not await self._executor.connect():
                log.error(
                    "Execution layer failed to arm; aborting before any market data "
                    "is consumed"
                )
                return

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
                # Warm order metadata here, well before any signal needs it.
                for market in found:
                    await self._executor.prepare_market(market)
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
        """Event-driven evaluation with a periodic floor.

        Waiting on the tripwire rather than sleeping a fixed period removes up
        to a full polling interval of dead time between the tick that creates
        the edge and the evaluation that acts on it. The timeout keeps the
        periodic pass for housekeeping (expiry, cooldowns, re-arming).
        """
        period = 1.0 / max(self._cfg.eval_hz, 1.0)
        while True:
            try:
                await self._engine.evaluate()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.exception("Eval loop error: %s", exc)

            self._wake.clear()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=period)

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(15.0)
            last = self._buffer.last()
            markets = self._engine.active_markets()
            move = self._buffer.largest_move(self._cfg.spike_lookback)
            log.info(
                "hb | %s | spot=%s | sigma=%.2f bps/s | ws=%s lat=%.0fms ticks=%d "
                "| 3s move=%+.1f bps | %s | %s | %s | markets=%s",
                "DRY" if self._executor.dry_run else "LIVE",
                f"{last.price:,.2f}" if last else "n/a",
                self._buffer.sigma_per_sqrt_second() * 10_000.0,
                "up" if self._stream.connected else "DOWN",
                self._buffer.latency_ms,
                self._buffer.tick_count,
                move.bps if move else 0.0,
                self._executor.stats(),
                self._executor.latency.summary(),
                self._executor.risk.stats(),
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
    execution = parser.add_argument_group("execution")
    execution.add_argument(
        "--live",
        action="store_true",
        help=(
            "ARM LIVE TRADING and submit real orders with real funds. "
            "Omitted by default: the bot runs dry, simulating fills and logging "
            "expected PnL without ever touching the order API."
        ),
    )
    execution.add_argument(
        "--size", type=float, default=20.0, help="contracts per order leg"
    )
    execution.add_argument(
        "--max-notional",
        type=float,
        default=50.0,
        help="hard USDC cap per leg; orders above this are blocked outright",
    )
    execution.add_argument(
        "--slippage-ticks",
        type=int,
        default=0,
        help="extra ticks above the ask offered to guarantee the cross",
    )
    execution.add_argument(
        "--env-file", default=".env", help="path to the dotenv file holding credentials"
    )

    risk = parser.add_argument_group("risk management")
    risk.add_argument(
        "--max-risk-pct",
        type=float,
        default=2.0,
        help="percent of bankroll put at risk per trade (binaries risk the full premium)",
    )
    risk.add_argument(
        "--fixed-size",
        action="store_true",
        help="disable dynamic sizing and always use --size contracts",
    )
    risk.add_argument(
        "--max-drawdown-pct",
        type=float,
        default=5.0,
        help="percent peak-to-trough session drawdown that trips the breaker",
    )
    risk.add_argument(
        "--max-drawdown-usd",
        type=float,
        default=100.0,
        help="absolute USDC drawdown that trips the breaker (whichever binds first)",
    )
    risk.add_argument(
        "--max-consecutive-losses",
        type=int,
        default=3,
        help="losing or failed trades in a row before halting the session",
    )
    risk.add_argument(
        "--risk-cooldown",
        type=float,
        default=300.0,
        help="protective pause after a drawdown trip, seconds",
    )
    risk.add_argument(
        "--max-drawdown-trips",
        type=int,
        default=3,
        help="drawdown trips tolerated before the session halts permanently",
    )
    risk.add_argument(
        "--paper-bankroll",
        type=float,
        default=1000.0,
        help="simulated bankroll for dry runs, where there is no wallet to read",
    )
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("websockets").setLevel(logging.WARNING)
    load_dotenv(args.env_file)

    return Config(
        spike_bps=args.spike_bps,
        spike_lookback=args.spike_window,
        max_yes_ask=args.max_yes_ask,
        min_edge=args.min_edge,
        signal_cooldown=args.cooldown,
        book_poll_interval=args.book_interval,
        enforce_ask_ceiling=not args.no_ask_ceiling,
        ws_endpoints=tuple(args.ws_url) if args.ws_url else BINANCE_WS_FALLBACKS,
        execution=ExecutionSettings(
            dry_run=not args.live,
            order_size=args.size,
            max_notional=args.max_notional,
            slippage_ticks=args.slippage_ticks,
        ),
        risk=RiskSettings(
            max_risk_pct=args.max_risk_pct / 100.0,
            dynamic_sizing=not args.fixed_size,
            max_drawdown_pct=args.max_drawdown_pct / 100.0,
            max_drawdown_usd=args.max_drawdown_usd,
            max_consecutive_losses=args.max_consecutive_losses,
            risk_cooldown=args.risk_cooldown,
            max_drawdown_trips=args.max_drawdown_trips,
            paper_bankroll=args.paper_bankroll,
        ),
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


def _ecc_backend() -> str:
    """Which elliptic-curve backend eth-keys picked.

    The pure-Python fallback signs an order in ~7.4ms; libsecp256k1 via
    coincurve does it in ~0.8ms. Worth surfacing at startup, because the only
    difference is whether the wheel is installed.
    """
    try:
        from eth_keys import backends

        name = type(backends.get_backend()).__name__
        return "coincurve" if "CoinCurve" in name else f"{name} (SLOW - pip install coincurve)"
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> None:
    cfg = parse_args()
    tuned = tune_clob_http_client()
    log.info(
        "Runtime | json=%s | ecc=%s | clob-http=%s",
        JSON_BACKEND,
        _ecc_backend(),
        "tuned" if tuned else "stock",
    )
    log.info(
        "Starting scanner | %s | spike>%gbps/%gs | ask gate %.2f | min edge %.3f",
        "DRY RUN" if cfg.execution.dry_run else "*** LIVE TRADING ***",
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
