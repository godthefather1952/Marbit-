#!/usr/bin/env python3
"""Read-only Kalshi venue adapter.

Kalshi is a CFTC-regulated US exchange that lists `KXBTC15M` - "BTC price up in
next 15 mins?" - which is structurally the same contract as Polymarket's
`btc-updown-15m`, and legally accessible to US persons.

Settlement, verbatim from the API:

    "If the simple average of the sixty seconds of CF Benchmarks' BRTI before
     HH:15 is at least the simple average of the sixty seconds of BRTI before
     HH:00, then the market resolves to Yes."

Two consequences, both good:

* Identical 60-second TWAP structure, so the piecewise `effective_tau` haircut
  derived for Polymarket transfers unchanged.
* The opening average is already fixed when the window starts and Kalshi
  publishes it as `floor_strike`. On Polymarket the strike was unobservable,
  which forced a strike-free delta-shift model; here fair value can be computed
  outright.

**This module cannot place, modify, or cancel an order.** Reads only. Order
support is a separate, deliberate decision - not something that should arrive
alongside a balance check.

Auth (docs.kalshi.com/getting_started/quick_start_authenticated_requests):

    message   = f"{timestamp_ms}{METHOD}{path}"      # query string EXCLUDED
    signature = base64(RSA-PSS-SHA256(message))      # salt = digest length
    headers   = KALSHI-ACCESS-KEY / -TIMESTAMP / -SIGNATURE

Credentials come from the environment:

    KALSHI_API_KEY_ID
    KALSHI_PRIVATE_KEY_PATH   (path to the RSA PEM)   - or -
    KALSHI_PRIVATE_KEY        (the PEM itself)
"""

from __future__ import annotations

import asyncio
import base64
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode

import aiohttp

from btc_polymarket_arb import (
    RETRYABLE_STATUS,
    RetryableError,
    _header_seconds,
    clamp_prob,
    json_loads,
    log,
    norm_cdf,
    retry_async,
)

API_HOST = "https://api.elections.kalshi.com"
API_BASE = "/trade-api/v2"

#: The 15-minute BTC up/down series - the direct analogue of btc-updown-15m.
BTC_15M_SERIES = "KXBTC15M"
#: Hourly above/below, and the other 15-minute crypto series.
BTC_HOURLY_SERIES = "KXBTCD"
CRYPTO_15M_SERIES = (
    "KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M", "KXDOGE15M",
    "KXADA15M", "KXBCH15M", "KXBNB15M", "KXHYPE15M", "KXNEAR15M",
    "KXTON15M", "KXZEC15M",
)

#: Kalshi's taker fee: 0.07 x contracts x P x (1-P), rounded UP to the cent.
FEE_COEFFICIENT = 0.07

#: Settlement averages the final 60 seconds.
TWAP_LOOKBACK_SECONDS = 60.0


#: Settlement is CF Benchmarks BRTI, which is a **USD** index built from USD
#: spot venues. Binance's BTCUSDT is quoted in USDT, and USDT/USD routinely
#: drifts 5-15 bps from parity - the same order of magnitude as an entire
#: 15-minute BTC move. Pricing a USD-settled contract off a USDT feed therefore
#: injects an error as large as the signal, and it is a *biased* error, not
#: noise that averages out.
#:
#: Measured on 2026-08-13: Binance $63,841.94 vs a USD composite of
#: $63,766.69 - a +11.8 bps premium.
#:
#: These are USD-quoted BRTI constituents. Use them, not Binance.
COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"
COINBASE_PRODUCT = "BTC-USD"
USD_SPOT_REST = (
    ("Coinbase", "https://api.coinbase.com/v2/prices/BTC-USD/spot"),
    ("Bitstamp", "https://www.bitstamp.net/api/v2/ticker/btcusd/"),
)


class KalshiAuthError(RuntimeError):
    """Credentials missing, malformed, or rejected."""


# --------------------------------------------------------------------------- #
# Fees
# --------------------------------------------------------------------------- #


def trading_fee(price: float, contracts: float) -> float:
    """Kalshi taker fee in dollars, rounded up to the next cent.

    Quadratic in price, so it peaks at 50c - which is exactly where these
    markets open. At $0.50 the fee is 1.75c per contract, which is more than
    half of a 3c edge. Any edge threshold that ignores this is fiction.
    """
    if contracts <= 0 or not (0.0 < price < 1.0):
        return 0.0
    raw = FEE_COEFFICIENT * contracts * price * (1.0 - price)
    return math.ceil(raw * 100.0) / 100.0


def fee_per_contract(price: float) -> float:
    """Marginal fee at this price, in dollars per contract (unrounded)."""
    if not (0.0 < price < 1.0):
        return 0.0
    return FEE_COEFFICIENT * price * (1.0 - price)


def net_edge(fair_value: float, ask: float) -> float:
    """Edge per contract after the taker fee. This is the number that matters."""
    return fair_value - ask - fee_per_contract(ask)


def breakeven_fair_value(ask: float) -> float:
    """Fair value a contract must reach at `ask` just to cover the fee."""
    return ask + fee_per_contract(ask)


# --------------------------------------------------------------------------- #
# Tick ladder
# --------------------------------------------------------------------------- #


def quantize_kalshi_price(price: float, price_ranges: Sequence[Mapping[str, Any]], buy: bool) -> float:
    """Snap a price onto Kalshi's *tapered* tick ladder.

    Unlike Polymarket's single tick size, Kalshi uses finer steps at the
    extremes - typically 0.1c below $0.10 and above $0.90, 1c in between. A
    single-tick assumption would round a legal 0.099 up to 0.10 and quote past
    the liquidity it was aiming at.

    Buys round up and sells round down, so float error can never push a
    marketable order to the passive side.
    """
    if not price_ranges:
        step = 0.01
        steps = price / step
        snapped = (math.ceil(steps - 1e-9) if buy else math.floor(steps + 1e-9)) * step
        return min(max(round(snapped, 4), step), 1.0 - step)

    bands = []
    for band in price_ranges:
        try:
            bands.append(
                (float(band["start"]), float(band["end"]), float(band["step"]))
            )
        except (KeyError, TypeError, ValueError):
            continue
    if not bands:
        return quantize_kalshi_price(price, (), buy)
    bands.sort()

    lo = bands[0][0]
    hi = bands[-1][1]
    price = min(max(price, lo), hi)

    start, step = bands[0][0], bands[0][2]
    for band_start, band_end, band_step in bands:
        if band_start <= price <= band_end:
            start, step = band_start, band_step
            break

    offset = (price - start) / step
    snapped = start + (math.ceil(offset - 1e-9) if buy else math.floor(offset + 1e-9)) * step
    decimals = max(0, round(-math.log10(step)) + 1)
    return round(min(max(snapped, lo), hi), decimals)


# --------------------------------------------------------------------------- #
# Market and book
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class KalshiMarket:
    ticker: str
    event_ticker: str
    title: str
    strike: float  # floor_strike: the 60s TWAP at window open, published
    open_ts: float
    close_ts: float
    status: str
    yes_bid: float | None
    yes_ask: float | None
    volume: float
    open_interest: float
    price_ranges: tuple[Mapping[str, Any], ...] = ()
    twap_lookback: float = TWAP_LOOKBACK_SECONDS

    def seconds_remaining(self, now: float | None = None) -> float:
        return self.close_ts - (now if now is not None else time.time())

    def effective_tau(self, now: float | None = None) -> float:
        """Same piecewise TWAP haircut derived for Polymarket. See its README.

        tau >= L : tau - 2L/3      (whole averaging window still ahead)
        tau <  L : tau / 3         (part already realized and locked in)

        Continuous at tau == L, where both give L/3.
        """
        tau = max(self.seconds_remaining(now), 1e-3)
        lookback = self.twap_lookback
        if lookback <= 0.0 or tau >= lookback:
            return max(tau - (2.0 * lookback / 3.0), 1e-3)
        return tau / 3.0

    def is_live(self, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        return self.status in ("active", "open") and self.open_ts <= now < self.close_ts

    @property
    def strike_known(self) -> bool:
        """False for the first ~30-45s of a window, before `floor_strike` posts.

        Kalshi opens the contract before the 60s TWAP that defines the strike
        has finished printing, so a freshly-discovered market arrives with
        `floor_strike` absent and parses to 0.0. Every model price computed in
        that gap is meaningless, and a monitor that quietly substitutes 0.5
        turns the gap into a large fake edge against whichever side the book is
        leaning. Callers must check this before pricing anything.
        """
        return self.strike > 0.0

    def fair_value(
        self, spot: float, sigma_per_sqrt_s: float, now: float | None = None
    ) -> float | None:
        """P(settlement TWAP >= strike), from the PUBLISHED strike.

        Returns None - not a neutral 0.5 - when the inputs cannot support a
        price. 0.5 is a real probability and reads as "coin flip", so returning
        it for "unknown" is indistinguishable downstream from a genuine
        at-the-money quote, and any book trading away from 0.5 then looks like
        free money.

        Kalshi publishing `floor_strike` is the material improvement over
        Polymarket: fair value is absolute rather than a shift relative to the
        market's own pre-spike mid, so it no longer inherits whatever the book
        happened to believe a moment ago.

        Approximation worth naming: E[closing TWAP] is taken as the current
        spot. Once tau < L a slice of that average is already realized and
        known, so the true expectation is a blend of that slice and spot. The
        error is small at 15-minute horizons and is bounded by staying out of
        the final seconds.
        """
        if spot <= 0 or not self.strike_known:
            return None
        denom = sigma_per_sqrt_s * math.sqrt(self.effective_tau(now))
        if denom <= 0.0:
            return None
        return clamp_prob(norm_cdf(math.log(spot / self.strike) / denom))


@dataclass(slots=True)
class KalshiBook:
    """Top of book, with Kalshi's two-sided-bid convention resolved.

    Both `yes` and `no` ladders are *bid* ladders sorted ascending. There is no
    ask ladder: the best offer to sell you YES is the complement of the best NO
    bid. Reading `yes[-1]` as an ask would systematically misprice every signal.
    """

    yes_bid: float | None
    yes_ask: float | None
    no_bid: float | None
    no_ask: float | None
    yes_bid_size: float = 0.0
    yes_ask_size: float = 0.0

    @property
    def yes_mid(self) -> float | None:
        if self.yes_bid is not None and self.yes_ask is not None:
            return (self.yes_bid + self.yes_ask) / 2.0
        return self.yes_ask if self.yes_ask is not None else self.yes_bid

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "KalshiBook":
        book = payload.get("orderbook_fp") or payload.get("orderbook") or {}
        yes = _levels(book.get("yes_dollars") or book.get("yes"))
        no = _levels(book.get("no_dollars") or book.get("no"))

        yes_bid = max((p for p, _ in yes), default=None)
        no_bid = max((p for p, _ in no), default=None)
        yes_ask = round(1.0 - no_bid, 4) if no_bid is not None else None
        no_ask = round(1.0 - yes_bid, 4) if yes_bid is not None else None

        return cls(
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            yes_bid_size=next((q for p, q in yes if p == yes_bid), 0.0),
            yes_ask_size=next((q for p, q in no if p == no_bid), 0.0),
        )


def _levels(raw: Any) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for entry in raw or []:
        try:
            out.append((float(entry[0]), float(entry[1])))
        except (TypeError, ValueError, IndexError):
            continue
    return out


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ts(value: Any) -> float:
    if not value:
        return 0.0
    try:
        import datetime as _dt

        return _dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def parse_market(raw: Mapping[str, Any]) -> KalshiMarket | None:
    ticker = raw.get("ticker")
    if not ticker:
        return None
    return KalshiMarket(
        ticker=str(ticker),
        event_ticker=str(raw.get("event_ticker") or ""),
        title=str(raw.get("title") or ""),
        strike=_f(raw.get("floor_strike")),
        open_ts=_ts(raw.get("open_time")),
        close_ts=_ts(raw.get("close_time")),
        status=str(raw.get("status") or ""),
        yes_bid=_f(raw.get("yes_bid_dollars")) or None,
        yes_ask=_f(raw.get("yes_ask_dollars")) or None,
        volume=_f(raw.get("volume_fp")),
        open_interest=_f(raw.get("open_interest_fp")),
        price_ranges=tuple(raw.get("price_ranges") or ()),
    )


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class KalshiCredentials:
    key_id: str | None = None
    private_key_pem: str | None = None

    @classmethod
    def from_env(cls) -> "KalshiCredentials":
        pem = (os.getenv("KALSHI_PRIVATE_KEY") or "").strip() or None
        path = (os.getenv("KALSHI_PRIVATE_KEY_PATH") or "").strip()
        if not pem and path:
            try:
                pem = open(path, "r", encoding="utf-8").read()
            except OSError as exc:
                log.warning("Could not read KALSHI_PRIVATE_KEY_PATH: %s", exc)
        return cls(key_id=(os.getenv("KALSHI_API_KEY_ID") or "").strip() or None, private_key_pem=pem)

    @property
    def complete(self) -> bool:
        return bool(self.key_id and self.private_key_pem)

    def problems(self) -> list[str]:
        issues: list[str] = []
        if not self.key_id:
            issues.append("KALSHI_API_KEY_ID is not set")
        if not self.private_key_pem:
            issues.append("set KALSHI_PRIVATE_KEY_PATH (or KALSHI_PRIVATE_KEY) to your RSA PEM")
        elif "PRIVATE KEY" not in self.private_key_pem:
            issues.append("the private key does not look like a PEM block")
        return issues

    def describe(self) -> str:
        key = f"{self.key_id[:8]}..." if self.key_id else "MISSING"
        return f"key_id={key} private_key={'loaded' if self.private_key_pem else 'MISSING'}"


class RsaPssSigner:
    """Signs requests. Holds the private key; never logs or exposes it."""

    __slots__ = ("_key_id", "_private", "_padding", "_hash")

    def __init__(self, creds: KalshiCredentials) -> None:
        problems = creds.problems()
        if problems:
            raise KalshiAuthError("; ".join(problems))
        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding
        except ImportError as exc:  # pragma: no cover
            raise KalshiAuthError("pip install cryptography") from exc

        try:
            self._private = serialization.load_pem_private_key(
                creds.private_key_pem.encode(), password=None
            )
        except Exception as exc:  # noqa: BLE001
            raise KalshiAuthError(f"could not load the RSA private key: {exc}") from exc

        self._key_id = creds.key_id
        self._padding = padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH
        )
        self._hash = hashes.SHA256()

    def headers(self, method: str, path: str) -> dict[str, str]:
        """Auth headers. `path` must EXCLUDE the query string, per Kalshi's spec."""
        timestamp = str(int(time.time() * 1000))
        message = f"{timestamp}{method.upper()}{path}".encode()
        signature = base64.b64encode(
            self._private.sign(message, self._padding, self._hash)
        ).decode()
        return {
            "KALSHI-ACCESS-KEY": self._key_id or "",
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "Content-Type": "application/json",
        }


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class CompositeBasis:
    """Tracks how far one venue sits from a multi-venue USD composite.

    BRTI is a composite of several USD spot venues, so any single venue carries
    a persistent premium or discount to it. Measured over an hour of live data,
    Coinbase ran about 1.7 bps *below* a Kraken/Bitstamp/Gemini median - small
    in absolute terms, but for a 15-minute contract sigma*sqrt(tau) is only
    ~13 bps, so a 1.7 bps offset is ~13% of a standard deviation and moves fair
    value by roughly five points of probability. That was the dominant source
    of phantom edge in the first long run: the model sat 2.5c below market mid
    on 59% of samples.

    The basis moves on the timescale of exchange flow, not ticks, so polling it
    every 20s over REST and smoothing is plenty. The websocket tape stays the
    low-latency source; this only removes its level error.
    """

    def __init__(
        self,
        session: aiohttp.ClientSession,
        halflife_polls: float = 5.0,
        sources: tuple[tuple[str, str], ...] | None = None,
        label: str = "BTC",
    ) -> None:
        self._session = session
        # Resolved lazily: the source tables are defined below this class.
        self._sources = sources if sources is not None else COMPOSITE_SOURCES
        self._label = label
        self._alpha = 1.0 - 0.5 ** (1.0 / max(halflife_polls, 1.0))
        self.offset: float = 0.0  # add this to the venue tick to reach composite
        self.samples: int = 0
        self.last_composite: float | None = None

    async def poll_once(self) -> float | None:
        """One composite reading, as the median of the reachable USD venues."""
        prices: list[float] = []
        for name, url in self._sources:
            try:
                async with self._session.get(
                    url, timeout=aiohttp.ClientTimeout(total=8)
                ) as resp:
                    if resp.status != 200:
                        continue
                    data = json_loads(await resp.read())
                prices.append(_extract_price(name, data))
            except Exception:  # noqa: BLE001 - a venue being down is routine
                continue
        prices = [p for p in prices if p and p > 0]
        if len(prices) < 2:
            return None
        prices.sort()
        mid = len(prices) // 2
        composite = prices[mid] if len(prices) % 2 else (prices[mid - 1] + prices[mid]) / 2.0
        self.last_composite = composite
        return composite

    async def run(self, buffer, interval: float = 20.0) -> None:
        while True:
            try:
                composite = await self.poll_once()
                tick = buffer.last()
                if composite is not None and tick is not None and tick.price > 0:
                    raw = composite - tick.price
                    # Seed on the first reading, then smooth.
                    self.offset = raw if self.samples == 0 else (
                        self.offset + self._alpha * (raw - self.offset)
                    )
                    self.samples += 1
                    if self.samples == 1 or self.samples % 15 == 0:
                        log.info(
                            "%s feed basis vs USD composite: %+.2f USD (%+.1f bps), n=%d",
                            self._label,
                            self.offset,
                            self.offset / composite * 1e4,
                            self.samples,
                        )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.debug("Basis poll failed: %s", exc)
            await asyncio.sleep(interval)

    def correct(self, price: float) -> float:
        return price + self.offset


def _extract_price(name: str, data) -> float | None:
    try:
        if name == "coinbase":
            return float(data["data"]["amount"])
        if name == "kraken":
            return float(list(data["result"].values())[0]["c"][0])
        if name == "bitstamp":
            return float(data["last"])
        if name == "gemini":
            return float(data["last"])
    except Exception:  # noqa: BLE001
        return None
    return None


#: USD spot venues used to approximate the BRTI composite. BTC by default,
#: kept as a module constant for the many callers that predate multi-asset.
COMPOSITE_SOURCES = (
    ("coinbase", "https://api.coinbase.com/v2/prices/BTC-USD/spot"),
    ("kraken", "https://api.kraken.com/0/public/Ticker?pair=XBTUSD"),
    ("bitstamp", "https://www.bitstamp.net/api/v2/ticker/btcusd/"),
    ("gemini", "https://api.gemini.com/v1/pubticker/btcusd"),
)

ETH_COMPOSITE_SOURCES = (
    ("coinbase", "https://api.coinbase.com/v2/prices/ETH-USD/spot"),
    ("kraken", "https://api.kraken.com/0/public/Ticker?pair=ETHUSD"),
    ("bitstamp", "https://www.bitstamp.net/api/v2/ticker/ethusd/"),
    ("gemini", "https://api.gemini.com/v1/pubticker/ethusd"),
)


@dataclass(frozen=True, slots=True)
class Asset:
    """A tradeable underlying and every feed that must match it.

    The series and the spot feed are bound together here on purpose. They are
    the one pair in this system that must never be mixed: pricing an ETH
    contract off the BTC tape compares a ~$63,000 spot against a ~$1,880
    strike, concludes YES is certain, and reports a colossal edge on every
    single observation. Making the pairing a single object means a caller
    cannot express the mismatch, rather than being trusted not to.
    """

    name: str  # "BTC"
    series: str  # "KXBTC15M"
    coinbase_product: str  # "BTC-USD"
    composite: tuple[tuple[str, str], ...]
    rest: tuple[tuple[str, str], ...]


ASSETS: dict[str, Asset] = {
    "BTC": Asset("BTC", BTC_15M_SERIES, "BTC-USD", COMPOSITE_SOURCES, USD_SPOT_REST),
    "ETH": Asset(
        "ETH",
        "KXETH15M",
        "ETH-USD",
        ETH_COMPOSITE_SOURCES,
        (
            ("Coinbase", "https://api.coinbase.com/v2/prices/ETH-USD/spot"),
            ("Bitstamp", "https://www.bitstamp.net/api/v2/ticker/ethusd/"),
        ),
    ),
}


def asset_for(name_or_series: str) -> Asset:
    """Resolve "BTC", "ETH" or a series ticker to its Asset.

    Raises rather than guessing. A wrong guess here silently prices one
    instrument off another's tape, which is the most expensive mistake this
    codebase can make.
    """
    key = (name_or_series or "").strip().upper()
    if key in ASSETS:
        return ASSETS[key]
    for asset in ASSETS.values():
        if asset.series.upper() == key:
            return asset
    raise ValueError(
        f"no spot feed is configured for {name_or_series!r}; "
        f"known assets: {', '.join(sorted(ASSETS))}. Refusing to price a "
        "contract off another instrument's tape."
    )


class CoinbaseSpotStream:
    """USD-quoted BTC tape, aligned with what Kalshi actually settles on.

    Coinbase is a BRTI constituent and quotes in real USD, so it carries none
    of the USDT basis that makes a Binance feed unusable here. Reuses the same
    PriceBuffer as the Polymarket bot, so every downstream calculation - spike
    detection, volatility, the tripwire - is unchanged.
    """

    def __init__(
        self,
        buffer,
        url: str = COINBASE_WS,
        product: str = COINBASE_PRODUCT,
        basis: "CompositeBasis | None" = None,
    ) -> None:
        self._buffer = buffer
        self._url = url
        self._product = product
        self._connected = False
        #: Optional level correction toward the BRTI-like composite. Without it
        #: the tape carries this venue's persistent premium or discount, which
        #: on a 15-minute contract is worth several points of probability.
        self._basis = basis

    @property
    def connected(self) -> bool:
        return self._connected

    async def run(self) -> None:
        import json as _json

        try:
            from websockets.asyncio.client import connect as ws_connect
        except ImportError:  # pragma: no cover
            from websockets.client import connect as ws_connect  # type: ignore

        backoff = 1.0
        subscribe = _json.dumps(
            {"type": "subscribe", "product_ids": [self._product], "channels": ["ticker"]}
        )
        while True:
            try:
                async with ws_connect(self._url, ping_interval=20, ping_timeout=20) as ws:
                    await ws.send(subscribe)
                    log.info("Coinbase %s (USD) tape connected", self._product)
                    self._connected = True
                    backoff = 1.0
                    async for raw in ws:
                        t1 = time.perf_counter_ns()
                        try:
                            msg = json_loads(raw)
                            if msg.get("type") != "ticker":
                                continue
                            price = float(msg["price"])
                        except (ValueError, KeyError, TypeError):
                            continue
                        if price > 0:
                            if self._basis is not None:
                                price = self._basis.correct(price)
                            self._buffer.add(price, 0, t1)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep the tape alive
                log.warning("Coinbase tape dropped (%s)", exc)
            finally:
                self._connected = False
                self._buffer.reset("coinbase tape disconnect")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)


async def usd_spot_rest(
    session: aiohttp.ClientSession,
    sources: tuple[tuple[str, str], ...] = USD_SPOT_REST,
) -> float | None:
    """One-shot USD spot, for sanity-checking a stream against REST."""
    for name, url in sources:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    continue
                data = json_loads(await resp.read())
            if name == "Coinbase":
                return float(data["data"]["amount"])
            return float(data["last"])
        except Exception:  # noqa: BLE001 - try the next source
            continue
    return None


class KalshiClient:
    """Read-only Kalshi client. Market data needs no credentials.

    No create/cancel/modify surface exists here, and the module issues no
    POST/PUT/DELETE - both asserted by the test suite.
    """

    def __init__(
        self,
        session: aiohttp.ClientSession,
        creds: KalshiCredentials | None = None,
        host: str = API_HOST,
    ) -> None:
        self._session = session
        self._creds = creds or KalshiCredentials.from_env()
        self._host = host.rstrip("/")
        self._signer: RsaPssSigner | None = None

    @property
    def credentials(self) -> KalshiCredentials:
        return self._creds

    @property
    def authenticated(self) -> bool:
        return self._signer is not None

    def authenticate(self) -> None:
        self._signer = RsaPssSigner(self._creds)
        log.info("Kalshi signer ready (%s)", self._creds.describe())

    async def _get(self, path: str, params: Mapping[str, Any] | None = None, signed: bool = False):
        query = ""
        if params:
            flat = [(k, str(v)) for k, v in params.items() if v is not None]
            if flat:
                query = "?" + urlencode(flat)
        full = API_BASE + path
        url = self._host + full + query

        headers: dict[str, str] = {}
        if signed:
            if self._signer is None:
                raise KalshiAuthError("authenticate() has not been called")
            # Query string deliberately excluded from the signed message.
            headers = self._signer.headers("GET", full)

        async def once():
            async with self._session.get(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                body = await resp.read()
                if resp.status in RETRYABLE_STATUS:
                    raise RetryableError(
                        f"{path} -> HTTP {resp.status}",
                        status=resp.status,
                        retry_after=_header_seconds(resp.headers.get("Retry-After")),
                    )
                if resp.status in (401, 403):
                    raise KalshiAuthError(
                        f"{path} -> HTTP {resp.status}: {body[:200].decode('utf-8', 'replace')}"
                    )
                if resp.status != 200:
                    raise RuntimeError(
                        f"{path} -> HTTP {resp.status}: {body[:200].decode('utf-8', 'replace')}"
                    )
                return json_loads(body) if body else None

        return await retry_async(once, attempts=3, label=f"kalshi {path}")

    # -- public market data ------------------------------------------------- #

    async def exchange_status(self):
        return await self._get("/exchange/status")

    async def markets(self, **filters: Any):
        return await self._get("/markets", filters)

    async def orderbook(self, ticker: str, depth: int | None = None):
        return await self._get(f"/markets/{ticker}/orderbook", {"depth": depth})

    async def series_list(self, category: str = "Crypto"):
        return await self._get("/series", {"category": category})

    async def live_markets(self, series_ticker: str = BTC_15M_SERIES) -> list[KalshiMarket]:
        """Currently-open markets for a series, soonest close first."""
        payload = await self.markets(series_ticker=series_ticker, status="open", limit=50)
        raw = (payload or {}).get("markets") or []
        out = [m for m in (parse_market(r) for r in raw) if m is not None]
        out.sort(key=lambda m: m.close_ts)
        return out

    async def book(self, ticker: str, depth: int | None = None) -> KalshiBook:
        return KalshiBook.from_payload(await self.orderbook(ticker, depth) or {})

    # -- authenticated reads ------------------------------------------------ #

    async def balance(self):
        """Account cash balance in cents."""
        return await self._get("/portfolio/balance", signed=True)

    async def positions(self, **filters: Any):
        return await self._get("/portfolio/positions", filters, signed=True)

    async def fills(self, **filters: Any):
        """Executed trades. The direct record of what actually filled.

        A position snapshot is a derived, eventually-consistent view and can
        read empty for seconds after a fill; a fill is the event itself, and
        carries the `side` the venue booked - which is exactly what the side
        mapping check needs to confirm.
        """
        return await self._get("/portfolio/fills", filters, signed=True)
