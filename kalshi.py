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
import contextlib
import math
import os
import statistics
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence
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

#: The websocket lives on its own host. The signed message is
#: `timestamp + "GET" + WS_PATH`, exactly as for a REST GET.
WS_HOST = "wss://api.elections.kalshi.com"
WS_PATH = "/trade-api/ws/v2"

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

#: Cap on the realized model's certainty, in sigma. Nothing about a synthetic
#: reference tape justifies more than "as settled as we can tell", and an
#: unbounded z here would let a division by a vanishing remaining window print
#: absurd numbers into logs and gates.
_SETTLED_Z = 8.0

#: How much of the already-elapsed averaging window our own tape must cover
#: before we are willing to price against the realized part of it. Below this we
#: would be extrapolating a partial observation across the whole elapsed slice,
#: which is a worse error than falling back to the terminal model.
MIN_TWAP_COVERAGE = 0.8


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
    """Marginal fee at this price, in dollars per contract (UNROUNDED).

    Correct only in the limit of a large order. Kalshi rounds the fee up to the
    next cent on the WHOLE order, so on the small sizes a small account
    actually trades this understates the real cost badly:

        price 0.99, 1 contract  -> 0.00069 modelled vs 0.01000 charged (14.4x)
        price 0.99, 3 contracts -> 0.00069 modelled vs 0.00333 charged  (4.8x)
        price 0.50, 7 contracts -> 0.01750 modelled vs 0.01857 charged  (1.1x)

    Prefer `fee_at_size` anywhere the order size is known - which is every
    decision that leads to an order. This remains for the size-independent
    breakeven reporting where no size exists yet.
    """
    if not (0.0 < price < 1.0):
        return 0.0
    return FEE_COEFFICIENT * price * (1.0 - price)


def fee_at_size(price: float, contracts: float) -> float:
    """Fee per contract for an order of exactly `contracts`, rounding included.

    This is what the account is actually charged, divided by the size. On a
    3-contract order at 0.99 it is 0.33c per contract where the unrounded model
    says 0.07c - a difference larger than the entire edge such a trade is
    usually chasing.
    """
    if contracts <= 0 or not (0.0 < price < 1.0):
        return 0.0
    return trading_fee(price, contracts) / contracts


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
        self,
        spot: float,
        sigma_per_sqrt_s: float,
        now: float | None = None,
        realized: tuple[float, float] | None = None,
        reference_error: float = 0.0,
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

        Pass `realized` once the averaging window has opened to replace that
        approximation with what the tape actually printed - see
        `fair_value_realized`. It degrades to this terminal model by returning
        None whenever the observation cannot support the better estimate, so
        callers never have to choose between the two.
        """
        if spot <= 0 or not self.strike_known:
            return None
        if realized is not None:
            settled = self.fair_value_realized(
                spot, sigma_per_sqrt_s, realized, now, reference_error
            )
            if settled is not None:
                return settled
        denom = sigma_per_sqrt_s * math.sqrt(self.effective_tau(now))
        if denom <= 0.0:
            return None
        return clamp_prob(norm_cdf(math.log(spot / self.strike) / denom))

    def fair_value_realized(
        self,
        spot: float,
        sigma_per_sqrt_s: float,
        realized: tuple[float, float],
        now: float | None = None,
        reference_error: float = 0.0,
    ) -> float | None:
        """`realized_z` as a probability, or None when it cannot be computed."""
        z = self.realized_z(
            spot, sigma_per_sqrt_s, realized, now, reference_error
        )
        return None if z is None else clamp_prob(norm_cdf(z))

    def realized_z(
        self,
        spot: float,
        sigma_per_sqrt_s: float,
        realized: tuple[float, float],
        now: float | None = None,
        reference_error: float = 0.0,
    ) -> float | None:
        """Standard deviations between spot and the MOVING effective strike.

        Kept separate from the probability because `clamp_prob` floors at 0.01
        and caps at 0.99, whose z is only 2.33 - inverting the clamped
        probability to recover a z would make every reading look like at most
        2.33 sigma and would silently disqualify anything gated on a higher
        threshold. Callers that need certainty in sigma units take this; callers
        that need a price take `fair_value_realized`.

        The contract settles on the mean of the final `L` seconds. Once the
        window has opened, some of that mean is no longer random - the tape has
        already printed it. Pricing against the ORIGINAL strike throughout
        treats known observations as unknown, and it is least accurate exactly
        when ENDGAME trades.

        With `e` seconds elapsed at observed mean `m`, and `r = L - e` still to
        come, settlement is YES when

            (e*m + r*mean_remaining) / L  >=  strike

        so the remaining stub must itself average at least

            required = (L*strike - e*m) / r

        which is a MOVING effective strike: it drifts away as the realized part
        runs against us, and collapses toward the current price as `r` shrinks.
        The mean of a driftless walk over the remaining `r` seconds has variance
        sigma^2*r/3, giving the usual normal probability.

        `realized` is `(observed_mean, covered_seconds)` as returned by
        `PriceBuffer.mean_since()`: the mean our tape printed since the window
        opened, and how many seconds of tape that mean actually spans. Returns
        None when the window has not opened or when coverage is too thin, so the
        caller falls back to the terminal model.

        Two honest limitations, neither hidden: our tape is a synthetic USD
        composite rather than the BRTI index Kalshi actually settles on, and the
        arithmetic mean here is mixed with a log-normal volatility, which is a
        good approximation only because 60 seconds of BTC is a small move. This
        is a better estimate, not the settlement calculation.
        """
        observed_mean, covered = realized
        lookback = self.twap_lookback
        if lookback <= 0.0 or covered <= 0.0 or observed_mean <= 0.0:
            return None
        remaining = self.seconds_remaining(now)

        # `e` is DEFINITIONAL, not "however much tape we happen to hold": the
        # window opened `lookback - r` seconds ago whether or not we watched it.
        # Deriving `e` from coverage instead breaks the identity e + r == L and
        # silently produces a nonsense `required` - which is precisely how this
        # returned a pinned 0.01 for every input during development. When the
        # tape does not reach back far enough we refuse to price and the caller
        # falls back to the terminal model, which is wrong-but-bounded rather
        # than confidently wrong.
        r = max(min(remaining, lookback), 0.0)
        e = lookback - r
        if e <= 0.0:
            return None  # the averaging window has not started yet
        if covered < e * MIN_TWAP_COVERAGE:
            return None  # we did not watch enough of the realized part

        if remaining <= 0.0:
            # Everything that decides this has printed.
            return _SETTLED_Z if observed_mean >= self.strike else -_SETTLED_Z

        required = (lookback * self.strike - e * observed_mean) / r
        if required <= 0.0:
            return _SETTLED_Z  # cannot settle below zero; we have already won

        # Price-space volatility of the remaining stub's own mean.
        sigma_abs = sigma_per_sqrt_s * spot
        var_stub = (sigma_abs * sigma_abs) * (r / 3.0)

        # Our tape is not BRTI, and the moving strike AMPLIFIES that gap: an
        # error of d dollars in `observed_mean` moves `required` by d*e/r. With
        # 10 seconds left that is a 5x multiplier, so a $3 reference error
        # becomes a $15 error in the level we are pricing against. Carrying it
        # as variance rather than ignoring it is what stops this model from
        # getting *more* confident exactly as it becomes least reliable - in the
        # last seconds the term dominates and fair value collapses toward 0.5,
        # which is the honest answer there.
        amplified = abs(reference_error) * (e / r)
        denom = math.sqrt(var_stub + amplified * amplified)
        if denom <= 0.0:
            return None
        return max(-_SETTLED_Z, min(_SETTLED_Z, (spot - required) / denom))


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
    #: Full ladders, ascending by price: [(price, contracts), ...]. Kept so
    #: execution can ask what a given SIZE actually costs rather than assuming
    #: the whole order fills at the top level.
    yes_levels: tuple[tuple[float, float], ...] = ()
    no_levels: tuple[tuple[float, float], ...] = ()
    #: Identity and age of this snapshot.
    #:
    #: `version` distinguishes one book from the next. Without it a confirmation
    #: gate counts evaluation passes rather than new information: the evaluator
    #: runs every 0.1s while REST refreshes every 0.4s, so a single unchanged
    #: snapshot was being counted as four independent confirmations. Kalshi's
    #: REST book carries no sequence number, so this falls back to a hash of the
    #: ladder contents - which is exactly the property wanted, since a book that
    #: has not changed should not count twice.
    seq: int | None = None
    version: int = 0
    received_mono: float = 0.0
    received_ts: float = 0.0

    @property
    def age(self) -> float:
        """Seconds since this snapshot arrived."""
        return time.monotonic() - self.received_mono if self.received_mono else 0.0

    def cost_for(self, side: str, contracts: float) -> tuple[float, float] | None:
        """VWAP and filled quantity for taking `contracts` of `side`.

        Top-of-book is what a 1-lot pays. Asking for 20 when 2 are offered at
        0.63 and the rest sit at 0.67 costs materially more than 0.63, and the
        edge is computed against the price we would ACTUALLY pay. Returns None
        if no depth is visible; returns less than `contracts` when the book
        cannot fill the whole order.
        """
        # Buying YES lifts the NO bid ladder (and vice versa), converted to the
        # price of the outcome we want. Best price first.
        source = self.no_levels if side == "YES" else self.yes_levels
        offers = sorted(
            ((round(1.0 - p, 4), q) for p, q in source if 0.0 < p < 1.0),
            key=lambda pq: pq[0],
        )
        taken = 0.0
        spend = 0.0
        for price, qty in offers:
            if taken >= contracts:
                break
            lot = min(qty, contracts - taken)
            spend += lot * price
            taken += lot
        if taken <= 0:
            return None
        return spend / taken, taken

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

        raw_seq = payload.get("seq") or payload.get("sequence")
        try:
            seq = int(raw_seq) if raw_seq is not None else None
        except (TypeError, ValueError):
            seq = None

        return cls(
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            yes_bid_size=next((q for p, q in yes if p == yes_bid), 0.0),
            yes_ask_size=next((q for p, q in no if p == no_bid), 0.0),
            yes_levels=tuple(yes),
            no_levels=tuple(no),
            seq=seq,
            # Identity is the CONTENT, never the sequence number. `seq` counts
            # messages, and a websocket delta that moves a level ten deep bumps
            # it without changing the quote we would trade against - counting
            # that as a new book would re-break the confirmation rule that
            # `version` exists to enforce ("three confirmations means three
            # books"). `seq` is kept alongside, for stream continuity only.
            version=hash((tuple(yes), tuple(no))),
            received_mono=time.monotonic(),
            received_ts=time.time(),
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
        self.updated_mono: float = 0.0
        #: How far the venues disagreed on the last poll, in dollars. A wide
        #: spread means the "composite" is an average of prices that are not
        #: describing the same instant, and the correction it produces is not
        #: trustworthy. Downstream this is the measured size of our reference
        #: error, so it has to mean venue DISAGREEMENT and not our own
        #: sampling skew - hence the concurrent poll below.
        self.dispersion: float = 0.0
        self.venues: int = 0
        #: Per-venue detail from the last poll: name -> (price, seconds late).
        self.venue_prices: dict[str, tuple[float, float]] = {}
        #: Spread between the earliest and latest venue reply, in seconds.
        self.venue_skew: float = 0.0
        #: Venues dropped for disagreeing with the median, and for replying too
        #: late to describe the same instant as the others.
        self.rejected_outlier: int = 0
        self.rejected_late: int = 0
        #: Smoothed 1-sigma estimate of how far our reference may sit from the
        #: settlement index, in dollars. See `reference_error`.
        self._error_ewma: float = 0.0
        self._error_samples: int = 0

    @property
    def age(self) -> float | None:
        """Seconds since the composite last updated, or None if never."""
        return time.monotonic() - self.updated_mono if self.updated_mono else None

    @property
    def reference_error(self) -> float:
        """Smoothed 1-sigma dollars our reference may sit from the settlement index.

        Not `dispersion / 2`, though that was the first version and is close on
        average. The max-min RANGE of three or four samples is a high-variance
        estimator: measured live it swung between $12 and $59 across six polls a
        few seconds apart while no venue was systematically off by more than a
        basis point. Feeding that straight into pricing would make fair value
        jump for reasons that have nothing to do with the market.

        So: standard deviation across the venues rather than the range, then
        smoothed across polls, because the level of genuine cross-venue
        disagreement moves on the timescale of exchange flow and not of our
        polling. With only two venues there is no usable standard deviation, so
        half the gap is used instead.
        """
        return self._error_ewma

    #: A venue replying this much later than the fastest one is describing a
    #: different instant, not a different price.
    MAX_VENUE_SKEW = 2.0

    async def _fetch(self, name: str, url: str) -> tuple[str, float, float] | None:
        """One venue's price, stamped with when the reply actually landed."""
        try:
            async with self._session.get(
                url, timeout=aiohttp.ClientTimeout(total=8)
            ) as resp:
                if resp.status != 200:
                    return None
                data = json_loads(await resp.read())
        except Exception:  # noqa: BLE001 - a venue being down is routine
            return None
        price = _extract_price(name, data)
        if not price or price <= 0:
            return None
        return name, float(price), time.monotonic()

    async def poll_once(self) -> float | None:
        """One composite reading, as the median of the reachable USD venues.

        The venues are polled CONCURRENTLY and each reply is timestamped. Polled
        in sequence, four venues at an 8s timeout could span half a minute, and
        BTC moves enough in half a minute that the resulting spread would mostly
        measure our own sampling lag. Since that spread is what tells the
        settlement model how far our reference might be from BRTI, a number
        inflated by our own polling would make the model refuse good trades and
        trust bad ones in the same breath.
        """
        results = await asyncio.gather(
            *(self._fetch(name, url) for name, url in self._sources),
            return_exceptions=True,
        )
        observations = [
            r for r in results
            if isinstance(r, tuple) and len(r) == 3
        ]
        if len(observations) < 2:
            return None

        # Drop venues that answered far later than the rest: their price
        # describes a different moment, which is a staleness problem wearing a
        # disagreement costume.
        earliest = min(mono for _, _, mono in observations)
        latest = max(mono for _, _, mono in observations)
        timely = [o for o in observations if o[2] - earliest <= self.MAX_VENUE_SKEW]
        self.rejected_late = len(observations) - len(timely)
        if len(timely) < 2:
            return None

        prices = sorted(p for _, p, _ in timely)
        composite = _median(prices)

        # Discard venues far from the median before committing. A single stale
        # or wrong quote drags a mean and can drag a two-venue median outright;
        # 50 bps is far wider than these venues ever legitimately diverge.
        kept_obs = [o for o in timely if abs(o[1] - composite) / composite < 0.005]
        self.rejected_outlier = len(timely) - len(kept_obs)
        if len(kept_obs) < 2:
            # With fewer than two venues agreeing there is no composite worth
            # the name. Refusing leaves the previous offset in place and lets
            # the reference-age check notice, which beats publishing a number
            # built from one venue and a disagreement.
            return None

        kept = sorted(p for _, p, _ in kept_obs)
        composite = _median(kept)

        self.dispersion = kept[-1] - kept[0]
        # The number that feeds pricing is smoothed and uses a lower-variance
        # statistic than the range; see `reference_error`.
        spread = (
            statistics.stdev(kept) if len(kept) >= 3
            else (kept[-1] - kept[0]) / 2.0
        )
        self._error_ewma = spread if self._error_samples == 0 else (
            self._error_ewma + self._alpha * (spread - self._error_ewma)
        )
        self._error_samples += 1
        self.venues = len(kept_obs)
        self.venue_skew = latest - earliest
        self.venue_prices = {
            name: (price, mono - earliest) for name, price, mono in kept_obs
        }
        self.last_composite = composite
        self.updated_mono = time.monotonic()
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
                            "%s feed basis vs USD composite: %+.2f USD (%+.1f bps), "
                            "n=%d | %d venues spread $%.2f, replies within %.1fs",
                            self._label,
                            self.offset,
                            self.offset / composite * 1e4,
                            self.samples,
                            self.venues,
                            self.dispersion,
                            self.venue_skew,
                        )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.debug("Basis poll failed: %s", exc)
            await asyncio.sleep(interval)

    def correct(self, price: float) -> float:
        return price + self.offset


def _median(values: Sequence[float]) -> float:
    """Median of an already-sorted sequence."""
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


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
                            # Coinbase stamps every ticker with its own event
                            # time. Passing 0 here made the buffer's feed-latency
                            # diagnostic compute `now - 0` - an epoch-sized
                            # number - so a latency-sensitive strategy had no
                            # working measure of how late its own tape was.
                            exch_ms = _iso_ms(msg.get("time"))
                        except (ValueError, KeyError, TypeError):
                            continue
                        if price > 0:
                            if self._basis is not None:
                                price = self._basis.correct(price)
                            self._buffer.add(price, exch_ms, t1)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep the tape alive
                log.warning("Coinbase tape dropped (%s)", exc)
            finally:
                self._connected = False
                self._buffer.reset("coinbase tape disconnect")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)


class KalshiBookStream:
    """Live order books over the websocket: one snapshot, then deltas.

    REST polling tops out at a few hundred milliseconds per market and returns
    the same snapshot most of the time, so the book we priced against was
    routinely older than the quote we were trying to take - the single largest
    source of "the ask moved before the order landed". The websocket pushes
    every price-level change, so the book is current by construction.

    What this class is careful about:

    * A delta can only be applied to a book we already have. Kalshi numbers
      every message on the subscription with `seq`; if one is missed the local
      book is a fiction from that point on, silently and permanently. On a gap
      we DISCARD the affected state and re-request a snapshot rather than
      carrying on, because a book that is quietly wrong is worse than no book:
      the caller has a freshness check for missing data and none at all for
      data that merely lies.
    * REST is not retired. It bootstraps, it covers the stream while it is
      down, and `KalshiClient.book()` remains the resync and diagnostic path.
      This is an accelerator, not a replacement.
    * Nothing here places, cancels, or modifies anything. The only frames sent
      are `subscribe` and `update_subscription` for market data.
    """

    #: Rebuild the connection after this many gaps inside GAP_WINDOW seconds.
    #: Counting CONSECUTIVE gaps does not work: a resnapshot arrives in sequence
    #: immediately after every gap and resets the count, so the threshold would
    #: never be reached no matter how badly the socket was behaving. What
    #: matters is the rate.
    MAX_GAPS_IN_WINDOW = 3
    GAP_WINDOW = 60.0

    #: Rebuild the connection after this many rejected subscription attempts.
    MAX_RESUBSCRIBES = 3

    def __init__(
        self,
        signer: "RsaPssSigner",
        tickers: Iterable[str] = (),
        host: str = WS_HOST,
        path: str = WS_PATH,
        on_book: Callable[[str, "KalshiBook"], None] | None = None,
        on_reset: Callable[[], None] | None = None,
    ) -> None:
        self._signer = signer
        self._host = host.rstrip("/")
        self._path = path
        self._on_book = on_book
        #: Fired whenever local books are discarded - a gap, a drop, a
        #: reconnect. Callers holding a reference to a previously streamed book
        #: need to know it is no longer being maintained; without this the book
        #: object simply stops updating and still looks like a book.
        self._on_reset = on_reset
        self._desired: set[str] = {t for t in tickers if t}
        self._subscribed: set[str] = set()
        #: ticker -> side ("yes"/"no") -> price -> contracts
        self._levels: dict[str, dict[str, dict[float, float]]] = {}
        self._seq: int | None = None
        self._cmd_id = 0
        self._ws: Any = None
        self._recent_gaps: deque[float] = deque(maxlen=32)
        #: Server-assigned subscription id. `update_subscription` is rejected
        #: without it, so until this arrives the only way to change the tracked
        #: set is a fresh `subscribe`.
        self._sid: int | None = None
        self._resubscribes = 0

        self.books: dict[str, KalshiBook] = {}
        self.connected = False
        self.reconnects = 0
        self.gaps = 0
        self.snapshots = 0
        self.deltas = 0
        self.last_message_mono: float = 0.0

    # -- public surface ----------------------------------------------------- #

    def track(self, *tickers: str) -> None:
        """Replace the set of markets we want books for.

        Takes effect on the next pass of the read loop. Books for markets that
        are no longer wanted are dropped immediately so a stale contract cannot
        be read back out of `books` after it stops being maintained.
        """
        wanted = {t for t in tickers if t}
        for gone in self._desired - wanted:
            self._levels.pop(gone, None)
            self.books.pop(gone, None)
        self._desired = wanted

    def book(self, ticker: str) -> "KalshiBook | None":
        return self.books.get(ticker)

    def age(self, ticker: str) -> float | None:
        book = self.books.get(ticker)
        return None if book is None else book.age

    async def run(self) -> None:
        try:
            from websockets.asyncio.client import connect as ws_connect
            header_kw = "additional_headers"
        except ImportError:  # pragma: no cover - older websockets
            from websockets.client import connect as ws_connect  # type: ignore
            header_kw = "extra_headers"

        url = self._host + self._path
        backoff = 1.0
        while True:
            try:
                headers = self._signer.headers("GET", self._path)
                headers.pop("Content-Type", None)
                async with ws_connect(
                    url, ping_interval=20, ping_timeout=20, **{header_kw: headers}
                ) as ws:
                    self._ws = ws
                    self.connected = True
                    backoff = 1.0
                    self._reset_state()
                    log.info("Kalshi book stream connected")
                    await self._read_until_closed(ws)
            except asyncio.CancelledError:
                await self._close_quietly()
                raise
            except Exception as exc:  # noqa: BLE001 - keep the book stream alive
                log.warning("Kalshi book stream dropped (%s)", exc)
            finally:
                self.connected = False
                self._ws = None
                self._reset_state()
            self.reconnects += 1
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)

    # -- internals ---------------------------------------------------------- #

    def _reset_state(self) -> None:
        """Forget every local book. Called on connect, disconnect, and gaps.

        Clearing `books` is the point: a disconnected stream must not leave a
        book behind that looks current to a caller checking only for presence.
        """
        self._levels.clear()
        self.books.clear()
        self._subscribed.clear()
        self._seq = None
        self._sid = None
        self._resubscribes = 0
        self._notify_reset()

    def _notify_reset(self) -> None:
        if self._on_reset is not None:
            with contextlib.suppress(Exception):
                self._on_reset()

    async def _close_quietly(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()

    async def _send(self, ws, cmd: str, params: Mapping[str, Any]) -> None:
        self._cmd_id += 1
        await ws.send(_json_dumps({"id": self._cmd_id, "cmd": cmd, "params": dict(params)}))

    async def _sync_subscription(self, ws) -> None:
        """Make the venue's subscription match what we actually want.

        `update_subscription` carries the server-assigned `sid`. Without it the
        venue answers "Exactly one subscription ID is required" and silently
        keeps the old market set - which in a live session meant the stream
        stayed pinned to contracts that had already expired and never delivered
        another book after the first roll. The REST fallback covered it, so the
        run looked healthy while pricing off a half-second-old book for thirty
        minutes.
        """
        if self._desired == self._subscribed:
            return
        if not self._desired:
            return
        if not self._subscribed or self._sid is None:
            # No subscription yet, or no id to amend one with: start clean.
            await self._send(ws, "subscribe", {
                "channels": ["orderbook_delta"],
                "market_tickers": sorted(self._desired),
            })
            self._levels.clear()
            self.books.clear()
            self._notify_reset()
        else:
            added = sorted(self._desired - self._subscribed)
            removed = sorted(self._subscribed - self._desired)
            if added:
                await self._send(ws, "update_subscription", {
                    "sid": self._sid, "action": "add_markets",
                    "market_tickers": added,
                })
            if removed:
                await self._send(ws, "update_subscription", {
                    "sid": self._sid, "action": "delete_markets",
                    "market_tickers": removed,
                })
        for gone in self._subscribed - self._desired:
            self._levels.pop(gone, None)
            self.books.pop(gone, None)
        self._subscribed = set(self._desired)

    async def _force_resubscribe(self, ws) -> None:
        """Throw the subscription away so the next sync rebuilds it from scratch.

        Bounded: if rebuilding keeps failing, the connection itself is the
        problem and reconnecting is the only remaining move. Without the bound
        this would resubscribe every half second forever, which is how a
        degraded stream turns into a busy one.
        """
        self._subscribed.clear()
        self._sid = None
        self._levels.clear()
        self.books.clear()
        self._notify_reset()
        self._resubscribes += 1
        if self._resubscribes >= self.MAX_RESUBSCRIBES:
            raise ConnectionError(
                f"subscription rejected {self._resubscribes} times; reconnecting"
            )

    async def _resnapshot(self, ws) -> None:
        """Throw away local books and ask for fresh ones, without resubscribing."""
        self._levels.clear()
        self.books.clear()
        self._notify_reset()
        if self._desired and self._sid is not None:
            await self._send(ws, "update_subscription", {
                "sid": self._sid,
                "action": "get_snapshot",
                "market_tickers": sorted(self._desired),
            })
        elif self._desired:
            # No id to ask against; rebuilding the subscription is the only way
            # to get a snapshot back.
            self._subscribed.clear()

    async def _read_until_closed(self, ws) -> None:
        while True:
            await self._sync_subscription(ws)
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
            except asyncio.TimeoutError:
                continue  # idle: loop back so subscription changes take effect
            self.last_message_mono = time.monotonic()
            try:
                msg = json_loads(raw)
            except Exception:  # noqa: BLE001 - a malformed frame is not fatal
                continue
            if not isinstance(msg, dict):
                continue
            kind = msg.get("type")
            if kind == "subscribed":
                self._sid = _int_or_none((msg.get("msg") or {}).get("sid"))
                self._resubscribes = 0
                self._seq = _int_or_none(msg.get("seq")) or self._seq
                log.info("Kalshi book stream subscribed (sid %s)", self._sid)
                continue
            if kind == "error":
                # Not just logged. A rejected command means the venue's idea of
                # what we are subscribed to no longer matches ours, and every
                # book we hold is from that point on unmaintained. Tear the
                # subscription state down so the next pass rebuilds it, rather
                # than continuing to serve books nobody is updating.
                log.warning("Kalshi book stream error frame: %s", msg.get("msg"))
                await self._force_resubscribe(ws)
                continue
            # The sequence numbers every message on the subscription, not just
            # the book ones - so it must be checked BEFORE filtering by type. A
            # live session logged nine gaps, every single one exactly n -> n+2,
            # and every one immediately after a market roll: the venue's
            # acknowledgement of our own subscription change was being dropped
            # here without its sequence number ever being counted. Three of
            # those phantom gaps inside a minute tripped the reconnect guard.
            if not await self._check_sequence(ws, msg):
                continue
            if kind not in ("orderbook_snapshot", "orderbook_delta"):
                continue
            # Every book frame carries the sid too, so the id is recovered even
            # if the confirmation was missed.
            if self._sid is None:
                self._sid = _int_or_none(msg.get("sid"))
            body = msg.get("msg") or {}
            if kind == "orderbook_snapshot":
                self._apply_snapshot(body, msg.get("seq"))
            else:
                self._apply_delta(body, msg.get("seq"))

    async def _check_sequence(self, ws, msg: Mapping[str, Any]) -> bool:
        """False when this message cannot be trusted to follow the last one."""
        try:
            seq = int(msg.get("seq"))
        except (TypeError, ValueError):
            return True  # no sequence to check against
        if self._seq is not None and seq != self._seq + 1:
            self.gaps += 1
            log.warning(
                "Kalshi book stream sequence gap (%s -> %s); rebuilding from a "
                "snapshot", self._seq, seq,
            )
            self._seq = seq
            now = time.monotonic()
            self._recent_gaps.append(now)
            recent = sum(1 for t in self._recent_gaps if now - t <= self.GAP_WINDOW)
            if recent >= self.MAX_GAPS_IN_WINDOW:
                self._recent_gaps.clear()
                raise ConnectionError(
                    f"{recent} sequence gaps in {self.GAP_WINDOW:.0f}s; reconnecting"
                )
            await self._resnapshot(ws)
            # A snapshot is on its way; this message predates it.
            return False
        self._seq = seq
        return True

    def _apply_snapshot(self, body: Mapping[str, Any], seq: Any) -> None:
        ticker = body.get("market_ticker")
        if not ticker:
            return
        self._levels[ticker] = {
            "yes": _level_map(body.get("yes_dollars_fp") or body.get("yes")),
            "no": _level_map(body.get("no_dollars_fp") or body.get("no")),
        }
        self.snapshots += 1
        self._publish(ticker, seq)

    def _apply_delta(self, body: Mapping[str, Any], seq: Any) -> None:
        ticker = body.get("market_ticker")
        side = str(body.get("side") or "").lower()
        sides = self._levels.get(ticker) if ticker else None
        if sides is None or side not in sides:
            # No snapshot for this market yet, so there is nothing to patch.
            # Deltas are meaningless in isolation and guessing a base book from
            # one would invent depth that was never quoted.
            return
        try:
            price = float(body["price_dollars"])
            delta = float(body["delta_fp"])
        except (KeyError, TypeError, ValueError):
            return
        levels = sides[side]
        qty = levels.get(price, 0.0) + delta
        if qty > 1e-9:
            levels[price] = qty
        else:
            levels.pop(price, None)
        self.deltas += 1
        self._publish(ticker, seq)

    def _publish(self, ticker: str, seq: Any) -> None:
        sides = self._levels.get(ticker)
        if sides is None:
            return
        book = KalshiBook.from_payload({
            "seq": seq,
            "orderbook_fp": {
                "yes_dollars": sorted(sides["yes"].items()),
                "no_dollars": sorted(sides["no"].items()),
            },
        })
        self.books[ticker] = book
        if self._on_book is not None:
            with contextlib.suppress(Exception):
                self._on_book(ticker, book)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _level_map(raw: Any) -> dict[float, float]:
    """Kalshi's [[price, contracts], ...] as a price -> contracts dict."""
    out: dict[float, float] = {}
    for entry in raw or []:
        try:
            price, qty = float(entry[0]), float(entry[1])
        except (TypeError, ValueError, IndexError):
            continue
        if qty > 0:
            out[price] = qty
    return out


def _json_dumps(payload: Mapping[str, Any]) -> str:
    import json as _json

    return _json.dumps(payload)


def _iso_ms(value: Any) -> int:
    """Coinbase's RFC3339 event time as epoch milliseconds; 0 if unusable."""
    if not value:
        return 0
    try:
        import datetime as _dt

        text = str(value).replace("Z", "+00:00")
        return int(_dt.datetime.fromisoformat(text).timestamp() * 1000.0)
    except (ValueError, TypeError):
        return 0


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

    @property
    def signer(self) -> "RsaPssSigner | None":
        """The signer, for the websocket handshake. Read-only either way."""
        return self._signer

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
