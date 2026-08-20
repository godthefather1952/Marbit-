#!/usr/bin/env python3
"""Trading strategies for Kalshi's 15-minute BTC up/down markets.

Three strategies, ordered by how much they depend on our model being right.
That ordering matters: the two live bugs so far (a USDT-quoted feed, then an
unreliable volatility estimate) were both *model* errors that manufactured
edge. A strategy that needs less of the model is worth more than one whose
backtest looks better.

    CROSS     model-free. Buys both sides when the book's own bids cross,
              locking a profit at settlement regardless of what BTC does.
              Cannot be wrong; can only be rare.

    STALE     the original latency thesis, rebuilt to be immune to the two
              bugs. It anchors on the MARKET's price and the MARKET's implied
              volatility, and trades only the *change* implied by a spot move.
              Feed level errors and sigma errors cancel out of a difference.

    ENDGAME   near expiry, when the settlement average is largely locked in
              and the outcome is close to decided, buys the near-certain side
              if it is still offered below its true worth.

Every signal carries the numbers needed to grade it later against actual
settlement, which is what `kalshi_score.py` does. Until a strategy has been
graded on real outcomes, its "edge" is a hypothesis.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import NormalDist

from kalshi import KalshiBook, KalshiMarket, fee_at_size, fee_per_contract, trading_fee

_N = NormalDist()


def _clamp(p: float, lo: float = 0.001, hi: float = 0.999) -> float:
    return min(max(p, lo), hi)


# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Leg:
    side: str  # "YES" | "NO"
    price: float  # the ask we would pay
    size: float


@dataclass(slots=True)
class Signal:
    strategy: str
    ticker: str
    legs: list[Leg]
    #: Model probability that the YES side settles yes, at signal time.
    fair_yes: float
    #: Expected profit in dollars, already net of Kalshi's taker fee.
    expected_net: float
    #: Worst case in dollars, if every leg loses.
    max_loss: float
    spot: float
    strike: float
    seconds_left: float
    sigma_used: float
    #: Which book snapshot produced this signal. The confirmation gate counts
    #: DISTINCT books rather than evaluation passes, so an unchanged snapshot
    #: cannot confirm itself repeatedly.
    book_version: int | None = None
    note: str = ""
    ts: float = field(default_factory=time.time)

    def describe(self) -> str:
        legs = ", ".join(f"{leg.side} {leg.size:g} @ {leg.price:.3f}" for leg in self.legs)
        return (
            f"[{self.strategy}] {self.ticker} | {legs} | "
            f"exp net ${self.expected_net:+.2f} | risk ${self.max_loss:.2f} | "
            f"{self.seconds_left:.0f}s left"
        )


# --------------------------------------------------------------------------- #
# Market-implied volatility - the key robustness tool
# --------------------------------------------------------------------------- #


def implied_sigma(market: KalshiMarket, book: KalshiBook, spot: float) -> float | None:
    """Back the volatility out of the market's own quote.

    Our realized-vol estimator has been wrong by 2-4x, and the model is acutely
    sensitive to it. The market, quoting a 1c spread with hundreds of thousands
    of contracts traded, has a far better estimate than five minutes of
    one-second bars from a single venue.

    Taking sigma from the quote turns the model from "what should this be
    worth" into "given what the market believes, what does a spot move imply" -
    a much smaller, much safer claim.
    """
    mid = book.yes_mid
    if mid is None or spot <= 0 or market.strike <= 0:
        return None
    tau = market.effective_tau()
    if tau <= 0:
        return None
    lm = math.log(spot / market.strike)
    z = _N.inv_cdf(_clamp(mid))
    # Near the money the inversion is a ratio of two numbers that are both
    # approximately zero, and the result is garbage with real consequences: a
    # graded session produced sigma = 0.06 bps/s from a 0.535 mid (z = 0.09),
    # which made a 0.1 bps spot wiggle read as a 2-point repricing and bought a
    # losing trade with it. |z| >= 0.25 (mid outside roughly 0.40-0.60) is the
    # region where the quote actually carries volatility information.
    if abs(z) < 0.25 or abs(lm) < 1e-7:
        return None  # too close to the money to invert reliably
    sigma = lm / (z * math.sqrt(tau))
    # Absolute plausibility. 1e-5 per sqrt-second is ~5.6% annualized; crypto
    # does not trade there, so anything below it is an artefact rather than a
    # measurement. A live session inverted a 0.9935 mid sitting almost exactly
    # AT the strike into 0.03 bps/s, and STALE turned that into a claimed
    # $19.83 edge on a $0.15 risk.
    #
    # The divergence is not noise, it is structural: near expiry the market
    # knows how much of the settlement TWAP is already realized and we only
    # know spot, so the quote can be confident while ln(S/K) is ~0. Inverting
    # it there asks the quote a question it is not answering.
    return sigma if 1e-5 < sigma < 1e-2 else None


def vol_agreement(measured: float, implied: float | None) -> float | None:
    """How far our measured volatility sits from the market's, as a ratio >= 1.

    This is the single most load-bearing number in the project. Our only
    possible informational advantage over the book is the spot price, and the
    book sees that at least as fast as we do; the strike is published and the
    clock is public. So on the absolute fair-value model there is nothing left
    for an edge to come from *except* a difference of opinion about sigma - and
    the market's opinion, backed by a one-cent spread and real size, is better
    than five minutes of one-second bars from one venue.

    That makes a large ratio the opposite of an opportunity: it is the
    signature of a broken estimator, and every previous "edge" this project
    found turned out to be exactly that. Returns None when the quote cannot be
    inverted, which counts as unvalidated, not as agreement.
    """
    if implied is None or implied <= 0.0 or measured <= 0.0:
        return None
    return max(measured / implied, implied / measured)


# --------------------------------------------------------------------------- #
# CROSS - model-free
# --------------------------------------------------------------------------- #


def scan_cross(
    market: KalshiMarket, book: KalshiBook, size: float, min_profit: float = 0.01
) -> Signal | None:
    """Buy YES and NO together when the pair costs less than the $1 they pay.

    On Kalshi both ladders are bids, so the YES ask is `1 - best NO bid`. The
    pair costs `2 - (yes_bid + no_bid)`, which drops below $1 exactly when the
    two bids cross. Settlement always pays one side $1, so the profit is locked
    the moment both fills happen - no view on BTC, no volatility estimate, no
    reliance on our spot feed being right.

    The fee is what makes this rare: at the money it costs ~1.75c per contract
    per leg, so the bids must cross by more than ~3.5c to be worth taking.
    """
    if book.yes_ask is None or book.no_ask is None:
        return None
    cost_per_pair = book.yes_ask + book.no_ask
    fees = trading_fee(book.yes_ask, size) + trading_fee(book.no_ask, size)
    profit = size * (1.0 - cost_per_pair) - fees
    if profit < min_profit:
        return None

    return Signal(
        strategy="CROSS",
        ticker=market.ticker,
        book_version=book.version,
        legs=[Leg("YES", book.yes_ask, size), Leg("NO", book.no_ask, size)],
        fair_yes=book.yes_mid or 0.5,
        expected_net=profit,
        max_loss=0.0,  # one side always pays $1
        spot=0.0,
        strike=market.strike,
        seconds_left=market.seconds_remaining(),
        sigma_used=0.0,
        note=f"pair costs {cost_per_pair:.4f}, pays 1.0000",
    )


# --------------------------------------------------------------------------- #
# STALE - the latency thesis, made robust
# --------------------------------------------------------------------------- #


def scan_stale(
    market: KalshiMarket,
    book: KalshiBook,
    anchor_price: float,
    anchor_mid: float,
    spot: float,
    size: float,
    min_edge: float,
    fallback_sigma: float,
    require_implied: bool = True,
    min_move_bps: float = 8.0,
    max_vol_ratio: float = 0.0,
    max_edge: float = 0.35,
    anchor_tau: float = 0.0,
) -> Signal | None:
    """Trade the repricing a spot move implies, not the level.

    `anchor_mid` is the market's own mid from before the move, and `anchor_price`
    the spot at that same moment. Both the starting probability and the
    volatility come from the market, so:

      * a constant error in our spot feed cancels in ln(spot / anchor_price)
      * an error in our volatility estimate mostly cancels, because sigma is
        taken from the quote rather than measured

    What is left is the honest question: BTC moved this much, the market has
    not repriced yet, and is the gap bigger than the fee.

    `require_implied` defaults on. The fallback is our measured volatility, and
    a measurement that reads low scales `delta / (sigma * sqrt(tau))` up, so
    every spot wiggle is reported as a bigger repricing than it is. Falling
    back reintroduces the exact error this strategy exists to be immune to, so
    when the quote cannot be inverted the honest answer is no signal.

    `min_move_bps` is what keeps this strategy being itself. Without a real
    spot move there are two ways the numbers can still show "edge", and both
    were bought and graded in a live session (1 winner in 4):

      * the BOOK moved away from its own anchor mid while spot sat still, so
        the model - anchored on the old mid - fades the book's repricing. The
        book moves on order flow we cannot see; taking the other side of that
        is adverse selection, the opposite of the latency thesis.
      * a 1-3 bps wiggle of pure noise, amplified by sigma*sqrt(tau) in the
        denominator, reads as a 10-point repricing.

    Requiring the spot to have moved by more than noise makes the claim being
    traded the honest one: BTC actually jumped, and the book has not caught up.
    """
    if anchor_price <= 0 or spot <= 0 or not market.strike_known:
        return None
    delta = math.log(spot / anchor_price)
    if abs(delta) < min_move_bps / 10_000.0:
        return None

    sigma = implied_sigma(market, book, spot)
    if sigma is None:
        if require_implied:
            return None
        sigma = fallback_sigma
    elif max_vol_ratio > 0.0:
        # Taking sigma from the quote was supposed to make this strategy immune
        # to our own estimator being wrong. It does - but only while the quote's
        # sigma is itself sane. When the two disagree wildly, one of them is
        # broken and nothing here can tell which, so the honest move is to sit
        # out. A live session traded a 9.53x disagreement (ours 0.32, market
        # 0.03 bps/s) and bought 93 contracts on a fabricated edge.
        ratio = vol_agreement(fallback_sigma, sigma)
        if ratio is not None and ratio > max_vol_ratio:
            return None
    tau = market.effective_tau()
    denom = sigma * math.sqrt(tau)
    if denom <= 0:
        return None

    # The anchor's probability was measured when the contract had MORE time
    # left, and z is ln(S/K)/(sigma*sqrt(tau)) - so the same moneyness is a
    # larger z as tau shrinks. Carrying z0 forward unscaled silently assumes
    # time has not passed, which understates how decided the outcome has
    # become. The error is small early and large late:
    #
    #   tau 800 -> 780 : a 0.70 anchor is really 0.702   (+0.2 points)
    #   tau 200 -> 150 : a 0.70 anchor is really 0.728   (+2.8 points)
    #   tau  90 ->  30 : a 0.70 anchor is really 0.818  (+11.8 points)
    #
    # Correct transform, from z = ln(S/K)/(sigma*sqrt(tau)):
    #     z1 = z0*sqrt(tau0/tau1) + delta/(sigma*sqrt(tau1))
    z0 = _N.inv_cdf(_clamp(anchor_mid))
    if anchor_tau > 0.0 and tau > 0.0:
        z0 *= math.sqrt(anchor_tau / tau)
    fair_yes = _clamp(_N.cdf(z0 + delta / denom))

    if delta > 0:
        side, ask, fair_side = "YES", book.yes_ask, fair_yes
    else:
        side, ask, fair_side = "NO", book.no_ask, 1.0 - fair_yes
    if ask is None or not (0.0 < ask < 1.0):
        return None

    # Size-aware: Kalshi rounds the fee up on the whole order, so the marginal
    # rate understates what a small order is actually charged.
    edge = fair_side - ask - fee_at_size(ask, size)
    if edge < min_edge:
        return None
    # An edge this large on a liquid book is a model error, not an opportunity.
    # Every one this project has produced turned out to be. The trade that
    # motivated the cap claimed $19.83 of expected profit against $0.15 of
    # risk - a 132:1 return that no real market offers.
    if max_edge > 0.0 and edge > max_edge:
        return None

    return Signal(
        strategy="STALE",
        ticker=market.ticker,
        book_version=book.version,
        legs=[Leg(side, ask, size)],
        fair_yes=fair_yes,
        expected_net=edge * size,
        max_loss=size * ask + trading_fee(ask, size),
        spot=spot,
        strike=market.strike,
        seconds_left=market.seconds_remaining(),
        sigma_used=sigma,
        note=(
            f"spot {delta * 1e4:+.1f} bps vs anchor, market mid was {anchor_mid:.3f}, "
            f"implied sigma {sigma * 1e4:.2f} bps/s"
            + (f", tau {anchor_tau:.0f}->{tau:.0f}s" if anchor_tau > 0.0 else "")
        ),
    )


# --------------------------------------------------------------------------- #
# ENDGAME - buy near-certainty into expiry
# --------------------------------------------------------------------------- #


def scan_endgame(
    market: KalshiMarket,
    book: KalshiBook,
    spot: float,
    size: float,
    sigma: float,
    max_seconds_left: float = 120.0,
    min_z: float = 3.0,
    min_edge: float = 0.005,
    max_price: float = 0.99,
    require_implied: bool = True,
    realized: tuple[float, float] | None = None,
    reference_error: float = 0.0,
) -> Signal | None:
    """Late in the window, buy the side that is nearly decided.

    As expiry approaches, `effective_tau` collapses and a given distance from
    the strike becomes an ever larger number of standard deviations. At 3 sigma
    the outcome is ~99.9% settled, yet the favoured side often still trades a
    cent or two below $1.

    The economics are deliberately unglamorous: many small wins and rare, large
    losses. A 2c gain on a 98c contract is +2%, but the loss when it turns is
    -98%, so roughly one reversal in fifty wipes out the profit. This needs the
    strictest sizing of the three, and a `min_z` well above what feels
    necessary.

    `sigma` must be MEASURED, not implied. Feeding it the market's own implied
    volatility makes the strategy degenerate: implied sigma is by definition
    the value that reproduces the quote, so z comes back as the market's own z,
    fair value equals the mid exactly, and the edge collapses to half the
    spread minus the fee - negative by construction, always.

    So ENDGAME is irreducibly a bet that our volatility beats the market's, and
    the graded record says it does not. A live session bought NO at 0.975 on a
    "2.8 sigma" reading while our sigma sat 1.45x BELOW the market's - just
    inside the 1.50x gate. On the market's sigma the same setup was 1.97 sigma,
    under any sane threshold. BTC then moved $38 in 37 seconds, the market
    settled YES, and the trade lost $8.79 - more than every ENDGAME win in the
    project's history combined.

    Hence `require_implied` and the conservative sigma below: when the market
    thinks the world is more volatile than we do, we use ITS number. That
    removes the overconfidence which is the only thing that ever made this
    strategy look profitable. It will fire far less often, and the honest
    reading of that is not that the filter is too strict - it is that the edge
    was the error.
    """
    left = market.seconds_remaining()
    if left > max_seconds_left or left <= 0 or spot <= 0 or not market.strike_known:
        return None

    implied = implied_sigma(market, book, spot)
    if implied is None and require_implied:
        return None
    # Never let an under-estimate of volatility manufacture certainty. Taking
    # the larger sigma can only ever move fair value toward 0.5, so it cannot
    # invent an edge - it can only refuse one.
    sigma_used = max(sigma, implied) if implied is not None else sigma

    # Once the settlement window has opened, price against the part of the
    # average that has already printed rather than against the original strike.
    # This is where ENDGAME lives by definition, and it is the regime where the
    # terminal model is furthest from the truth: it can read 0.99 on a contract
    # whose settlement average is already lost. Returns None when the tape
    # cannot support it, which drops us back to the terminal model below.
    settled_z = market.realized_z(
        spot, sigma_used, realized, reference_error=reference_error
    ) if realized is not None else None

    if settled_z is not None:
        # `min_z` keeps meaning "this is nearly decided" on both paths: the
        # realized model reports its certainty in the same sigma units.
        z = settled_z
        fair_yes = _clamp(_N.cdf(z))
    else:
        tau = market.effective_tau()
        denom = sigma_used * math.sqrt(tau)
        if denom <= 0:
            return None
        z = math.log(spot / market.strike) / denom
        fair_yes = _clamp(_N.cdf(z))
    if abs(z) < min_z:
        return None

    if z > 0:
        side, ask, fair_side = "YES", book.yes_ask, fair_yes
    else:
        side, ask, fair_side = "NO", book.no_ask, 1.0 - fair_yes
    if ask is None or not (0.0 < ask <= max_price):
        return None

    edge = fair_side - ask - fee_at_size(ask, size)
    if edge < min_edge:
        return None

    return Signal(
        strategy="ENDGAME",
        ticker=market.ticker,
        book_version=book.version,
        legs=[Leg(side, ask, size)],
        fair_yes=fair_yes,
        expected_net=edge * size,
        max_loss=size * ask + trading_fee(ask, size),
        spot=spot,
        strike=market.strike,
        seconds_left=left,
        sigma_used=sigma_used,
        note=(
            f"{abs(z):.1f} sigma from the "
            + ("moving strike (settlement average part-printed)"
               if settled_z is not None else "strike")
            + f" with {left:.0f}s left "
            f"(sigma {sigma_used * 1e4:.2f} bps/s"
            + (f", ours {sigma * 1e4:.2f}, market {implied * 1e4:.2f}"
               if implied is not None else "")
            + ")"
        ),
    )


# --------------------------------------------------------------------------- #
# Paper ledger - what turns a run into evidence
# --------------------------------------------------------------------------- #


class PaperLedger:
    """Append-only record of every signal, for grading against settlement.

    One JSON object per line so a run can be scored while it is still going,
    and so a crash costs at most the last line.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.count = 0
        self._seen: set[tuple[str, str]] = set()

    def record(self, signal: Signal, once_per_market: bool = True) -> bool:
        """Log a paper trade. Returns False if skipped as a duplicate.

        One entry per (strategy, market) by default: the same edge persists for
        many evaluation passes, and counting it fifty times would make an
        hour's data look like fifty independent successes when it is one.
        """
        key = (signal.strategy, signal.ticker)
        if once_per_market and key in self._seen:
            return False
        self._seen.add(key)

        row = asdict(signal)
        row["legs"] = [asdict(leg) for leg in signal.legs]
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        self.count += 1
        return True

    def record_execution(
        self,
        signal: Signal,
        outcome: str,
        count: float = 0.0,
        price: float = 0.0,
        detail: str = "",
        signal_price: float | None = None,
        elapsed_ms: float | None = None,
    ) -> None:
        """Record what actually happened to a signal when it reached the venue.

        A recorded signal is a hypothesis; only a filled order is money. A live
        session graded five signals as wins worth +$3.85 while the account moved
        ten cents, because two were simulated during warm-up and three were
        skipped - and nothing downstream could tell the difference. Every signal
        now gets a companion row saying which it was.

        `outcome` is one of: "filled" (real money), "simulated" (dry run),
        "skipped" (never sent), "rejected" (sent, no fill), "closed" (exited
        before expiry).

        `signal_price` and `elapsed_ms` are written here rather than left to be
        reconstructed by joining rows later: slippage is the gap between the
        quote that justified the trade and the price we actually paid, and a
        strategy can be perfectly right about direction while losing all of its
        edge in that gap. Neither number is recoverable after the fact if the
        signal row and the fill row disagree about which leg they describe.
        """
        row = {
            "kind": "execution",
            "ts": time.time(),
            "strategy": signal.strategy,
            "ticker": signal.ticker,
            "outcome": outcome,
            "count": count,
            "price": price,
            "detail": detail,
            "signal_price": signal_price,
            "elapsed_ms": elapsed_ms,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def reset_dedupe(self) -> None:
        """Forget which (strategy, market) pairs have been recorded.

        Used at the paper-to-live promotion: without this, a market that
        signalled during warm-up could never trade live, because its key is
        already marked seen. The file keeps both phases' rows; their
        timestamps separate them.
        """
        self._seen.clear()
