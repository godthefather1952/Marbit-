#!/usr/bin/env python3
"""Does a spot move predict the outcome? The question everything else rests on.

    python kalshi_analyze.py                     # every log in logs/
    python kalshi_analyze.py logs/L_0819*.log
    python kalshi_analyze.py --anchor-age 20 --take-profit 1.5

STALE's entire thesis is that when spot moves, the book has not repriced yet.
That has never been measured. This reads the heartbeats already in your logs,
reconstructs the trade STALE would have taken at every point in time, and
grades it against what the market actually settled at.

The headline it exists to produce: **the smallest spot move, in basis points,
whose win rate beats the price paid plus the taker fee**. Below that number the
strategy is paying Kalshi for the privilege of guessing; above it there may be
something real.

Two policies are graded side by side, because they can disagree:

    HOLD          bought and kept to settlement - the raw predictive value of
                  the signal, with nothing else mixed in
    TAKE-PROFIT   sold on the way up at --take-profit, or at the model's fair
                  value, whichever came first; otherwise held

Reading them together answers the second question directly: did the quick
exits preserve capital, or did they cut winners short?

A caution the report repeats, because it decides how much to believe: a
15-minute market produces ~60 heartbeats that all share ONE settlement. Those
are not 60 independent samples. The effective sample size is the number of
MARKETS, not observations, and it is reported next to every figure.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import datetime as dt
import glob
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import aiohttp

from kalshi import KalshiClient, fee_per_contract
from kalshi_score import settlement

TRACK = re.compile(
    r"^(\S+ \S+).*\[(\w+)\] Tracking (\S+) \| strike \$([\d,.]+)"
)
HEARTBEAT = re.compile(
    r"^(\S+ \S+).*hb \| \[(\w+)\] (\S+) \| spot=\$([\d,.]+) strike=\$([\d,.]+|PENDING)"
    r".*yes (\S+)/(\S+) \|.*?(-?[\d.]+)s left"
)

DEFAULT_BUCKETS = (0.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0)


@dataclass(slots=True)
class Obs:
    ts: float
    asset: str
    ticker: str
    spot: float
    strike: float
    yes_bid: float | None
    yes_ask: float | None
    left: float

    @property
    def no_ask(self) -> float | None:
        return round(1.0 - self.yes_bid, 4) if self.yes_bid is not None else None

    @property
    def no_bid(self) -> float | None:
        return round(1.0 - self.yes_ask, 4) if self.yes_ask is not None else None


@dataclass(slots=True)
class Sample:
    """One trade STALE would have taken, and everything needed to grade it."""

    ticker: str
    asset: str
    move_bps: float
    side: str  # "YES" | "NO"
    entry: float
    left: float
    peak: float  # best mark seen afterwards, for the exit policy
    fair: float  # model fair value for this side at entry
    path: tuple[float, ...] = ()  # marks after entry, in order


def _f(text: str) -> float:
    return float(text.replace(",", ""))


def _time(stamp: str) -> float:
    try:
        return dt.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S.%f").timestamp()
    except ValueError:
        return 0.0


def parse_log(path: Path) -> list[Obs]:
    """Heartbeats, resolved to full tickers via the Tracking lines around them."""
    full: dict[tuple[str, str], str] = {}  # (asset, suffix) -> full ticker
    out: list[Obs] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        track = TRACK.match(line)
        if track:
            _, asset, ticker, _strike = track.groups()
            parts = ticker.split("-")
            if len(parts) >= 2:
                full[(asset, parts[1])] = ticker
            continue
        hb = HEARTBEAT.match(line)
        if not hb:
            continue
        stamp, asset, suffix, spot, strike, ybid, yask, left = hb.groups()
        if strike == "PENDING":
            continue  # floor_strike had not posted; nothing is priceable yet
        out.append(Obs(
            ts=_time(stamp), asset=asset,
            ticker=full.get((asset, suffix), f"{asset}:{suffix}"),
            spot=_f(spot), strike=_f(strike),
            yes_bid=None if ybid == "-" else float(ybid),
            yes_ask=None if yask == "-" else float(yask),
            left=float(left),
        ))
    return out


def build_samples(
    obs: list[Obs], anchor_age: float, min_left: float, tolerance: float = 12.0
) -> list[Sample]:
    """Reconstruct the trade STALE would have taken at each heartbeat.

    The rule is deliberately the live one: look back `anchor_age` seconds, take
    the log return, buy the side the move favours at the price then on offer.
    Nothing here uses information from after the entry except to measure the
    peak, which only the exit policy consults.
    """
    by_ticker: dict[str, list[Obs]] = collections.defaultdict(list)
    for o in obs:
        by_ticker[o.ticker].append(o)

    samples: list[Sample] = []
    for ticker, series in by_ticker.items():
        series.sort(key=lambda o: o.ts)
        for i, now in enumerate(series):
            if now.left < min_left or now.spot <= 0 or now.strike <= 0:
                continue
            # Nearest earlier observation about `anchor_age` back.
            anchor = None
            for j in range(i - 1, -1, -1):
                gap = now.ts - series[j].ts
                if gap >= anchor_age - tolerance:
                    anchor = series[j] if abs(gap - anchor_age) <= tolerance else None
                    break
            if anchor is None or anchor.spot <= 0:
                continue

            move = math.log(now.spot / anchor.spot) * 1e4
            side = "YES" if move > 0 else "NO"
            entry = now.yes_ask if side == "YES" else now.no_ask
            if entry is None or not (0.0 < entry < 1.0):
                continue

            # Best mark available afterwards, for the take-profit policy. This
            # is the only forward-looking value, and only the exit policy uses
            # it - the win rate is computed from settlement alone.
            peak = 0.0
            path = []
            for later in series[i + 1:]:
                mark = later.yes_bid if side == "YES" else later.no_bid
                if mark is not None:
                    peak = max(peak, mark)
                    path.append(mark)

            samples.append(Sample(
                ticker=ticker, asset=now.asset, move_bps=move, side=side,
                entry=entry, left=now.left, peak=peak,
                fair=_fair(now, side), path=tuple(path),
            ))
    return samples


def _fair(o: Obs, side: str) -> float:
    """The market's own mid for this side - the thesis-complete exit target."""
    if o.yes_bid is None or o.yes_ask is None:
        return 0.0
    mid = (o.yes_bid + o.yes_ask) / 2.0
    return mid if side == "YES" else 1.0 - mid


def pnl(sample: Sample, won: bool, take_profit: float) -> tuple[float, float]:
    """Per-contract PnL under (hold, take-profit), net of fees both ways."""
    cost = sample.entry + fee_per_contract(sample.entry)
    held = (1.0 if won else 0.0) - cost

    # Would an exit have triggered on the way up? The live rules are a net
    # multiple of cost, or the book reaching the model's fair value.
    target = min(
        (t for t in (
            _price_for_multiple(sample.entry, take_profit),
            sample.fair if sample.fair > sample.entry else None,
        ) if t is not None),
        default=None,
    )
    if target is not None and sample.peak >= target:
        exited = target - fee_per_contract(target) - cost
    else:
        exited = held
    return held, exited


def _price_for_multiple(entry: float, multiple: float) -> float | None:
    """Smallest sale price whose NET proceeds are `multiple` x net cost."""
    if multiple <= 0:
        return None
    cost = entry + fee_per_contract(entry)
    target = multiple * cost
    # fee_per_contract is monotone-ish and small; one correction pass is plenty.
    return min(target + fee_per_contract(min(target, 0.99)), 0.999)


def _exit_pnl(sale: float, cost: float) -> float:
    return sale - fee_per_contract(sale) - cost


def simulate(s: Sample, won: bool, policy: str, param: float, take_profit: float) -> float:
    """PnL per contract for one exit policy, walking the real post-entry path.

    Every policy sees the same prices in the same order and may only use what
    has happened so far - no policy is allowed to know the peak in advance,
    which is the mistake that makes backtested exits look free.
    """
    cost = s.entry + fee_per_contract(s.entry)
    settle = (1.0 if won else 0.0) - cost

    if policy == "hold":
        return settle

    if policy == "fixed":
        target = _price_for_multiple(s.entry, param)
        for mark in s.path:
            if target is not None and mark >= target:
                return _exit_pnl(mark, cost)
        return settle

    if policy == "trail":
        # Let it run, but give back at most `param` of the best gain so far.
        # This is the direct answer to "a strict TP would have capped the
        # winner": there is no ceiling, only a floor that ratchets up.
        peak = s.entry
        armed = False
        for mark in s.path:
            peak = max(peak, mark)
            if peak >= s.entry * 1.25:      # only arm once genuinely ahead
                armed = True
            if armed and mark <= peak - param * (peak - s.entry):
                return _exit_pnl(mark, cost)
        return settle

    if policy == "half":
        # Sell half at the target, ride the rest. Buys certainty on part of the
        # position without surrendering the tail.
        target = _price_for_multiple(s.entry, take_profit)
        for mark in s.path:
            if target is not None and mark >= target:
                return 0.5 * _exit_pnl(mark, cost) + 0.5 * settle
        return settle

    if policy == "stop":
        # A floor on the loss. Nothing else in this comparison touches the
        # downside - every policy shares the same worst case - so this is the
        # only lever that changes it.
        floor = s.entry * param
        for mark in s.path:
            if mark <= floor:
                return _exit_pnl(mark, cost)
        return settle

    if policy == "tp_stop":
        # Both ends: take the gain at `take_profit`, cut the loss at `param`.
        target = _price_for_multiple(s.entry, take_profit)
        floor = s.entry * param
        for mark in s.path:
            if target is not None and mark >= target:
                return _exit_pnl(mark, cost)
            if mark <= floor:
                return _exit_pnl(mark, cost)
        return settle

    if policy == "adaptive":
        # Take profit only on weak signals; let strong ones run. The bucket
        # table is what suggests this: exits help below ~10 bps and hurt above.
        if abs(s.move_bps) < param:
            target = _price_for_multiple(s.entry, take_profit)
            for mark in s.path:
                if target is not None and mark >= target:
                    return _exit_pnl(mark, cost)
        return settle

    raise ValueError(policy)


def compare_policies(graded, min_move: float, take_profit: float) -> None:
    live = [g for g in graded if abs(g[0].move_bps) >= min_move]
    if not live:
        return
    policies = [
        ("hold to settlement", "hold", 0.0),
        (f"fixed TP {take_profit:g}x", "fixed", take_profit),
        ("fixed TP 2.0x", "fixed", 2.0),
        ("fixed TP 3.0x", "fixed", 3.0),
        ("trail: give back 25%", "trail", 0.25),
        ("trail: give back 40%", "trail", 0.40),
        ("trail: give back 60%", "trail", 0.60),
        (f"sell half at {take_profit:g}x", "half", 0.0),
        ("adaptive: TP under 10 bps", "adaptive", 10.0),
        ("adaptive: TP under 15 bps", "adaptive", 15.0),
        ("stop at 50% of cost", "stop", 0.50),
        ("stop at 65% of cost", "stop", 0.65),
        (f"TP {take_profit:g}x + stop 50%", "tp_stop", 0.50),
        (f"TP {take_profit:g}x + stop 65%", "tp_stop", 0.65),
    ]
    print(f"\n{'=' * 78}")
    print(f" EXIT POLICIES, on the {len(live):,} entries at or above "
          f"{min_move:.0f} bps ({len({g[0].ticker for g in live})} markets)")
    print("=" * 78)
    print(f"  {'policy':<28s} {'$/contract':>11s} {'vs hold':>9s} {'worst':>8s}")
    print("  " + "-" * 60)
    base = statistics.fmean(simulate(g[0], g[1], "hold", 0.0, take_profit) for g in live)
    rows = []
    for label, kind, param in policies:
        vals = [simulate(g[0], g[1], kind, param, take_profit) for g in live]
        mean = statistics.fmean(vals)
        rows.append((mean, label))
        print(f"  {label:<28s} {mean:>+11.4f} {mean - base:>+9.4f} "
              f"{min(vals):>+8.3f}")
    best = max(rows)
    print("  " + "-" * 60)
    print(f"  best on this data: {best[1]} at {best[0]:+.4f}/contract")


def report(samples, settled, buckets, take_profit: float, min_move: float = 0.0):
    graded = []
    for s in samples:
        market = settled.get(s.ticker)
        if not market:
            continue
        result = str(market.get("result", "")).lower()
        if result not in ("yes", "no"):
            continue
        won = (s.side == "YES" and result == "yes") or (s.side == "NO" and result == "no")
        held, exited = pnl(s, won, take_profit)
        graded.append((s, won, held, exited))

    if not graded:
        print("Nothing to grade: no sampled market has settled yet.", file=sys.stderr)
        return

    print("=" * 78)
    print(" DOES A SPOT MOVE PREDICT THE OUTCOME?")
    print("=" * 78)
    print(f" {len(graded):,} reconstructed entries across "
          f"{len({s.ticker for s, *_ in graded})} settled markets\n")
    print(" A win rate only counts if it beats the price paid plus the fee. That")
    print(" break-even is shown next to it; EDGE is the gap between them.\n")

    header = (f"  {'move':>12s} {'n':>6s} {'mkts':>5s} {'win%':>7s} "
              f"{'avg px':>7s} {'need%':>7s} {'EDGE':>7s} {'HOLD $/c':>9s} "
              f"{'EXIT $/c':>9s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    for lo, hi in zip(buckets, list(buckets[1:]) + [float("inf")]):
        band = [g for g in graded if lo <= abs(g[0].move_bps) < hi]
        if not band:
            continue
        n = len(band)
        markets = len({g[0].ticker for g in band})
        wins = sum(1 for g in band if g[1])
        win_rate = wins / n
        avg_px = statistics.fmean(g[0].entry for g in band)
        need = statistics.fmean(
            g[0].entry + fee_per_contract(g[0].entry) for g in band
        )
        hold = statistics.fmean(g[2] for g in band)
        exit_ = statistics.fmean(g[3] for g in band)
        label = f"{lo:.0f}-{hi:.0f} bps" if hi != float("inf") else f"{lo:.0f}+ bps"
        rows.append((lo, win_rate, need, hold, exit_, markets, n))
        print(f"  {label:>12s} {n:>6,} {markets:>5} {win_rate * 100:>6.1f}% "
              f"{avg_px:>7.3f} {need * 100:>6.1f}% {(win_rate - need) * 100:>+6.1f}% "
              f"{hold:>+9.4f} {exit_:>+9.4f}")

    # -- the headline ------------------------------------------------------- #
    print()
    positive = [r for r in rows if r[3] > 0]
    positive_exit = [r for r in rows if r[4] > 0]
    print("-" * 78)
    if positive:
        print(f" HOLD is net positive from {positive[0][0]:.0f} bps up "
              f"({positive[0][5]} settled markets in that band).")
    else:
        print(" HOLD never turns net positive at any move size in this data.")
    if positive_exit:
        print(f" TAKE-PROFIT is net positive from {positive_exit[0][0]:.0f} bps up "
              f"({positive_exit[0][5]} settled markets).")
    else:
        print(" TAKE-PROFIT never turns net positive at any move size either.")

    # The headline covers only what the bot would actually trade. Averaging in
    # the sub-threshold entries buries the tradeable range under thousands of
    # rows the live filter already rejects.
    live = [g for g in graded if abs(g[0].move_bps) >= min_move]
    if live:
        lh = statistics.fmean(g[2] for g in live)
        le = statistics.fmean(g[3] for g in live)
        lw = sum(1 for g in live if g[1]) / len(live)
        print(f"\n AT THE LIVE THRESHOLD (>= {min_move:.0f} bps): {len(live):,} entries, "
              f"{len({g[0].ticker for g in live})} markets, {lw * 100:.1f}% win rate")
        print(f"   hold        {lh:+.4f}/contract")
        print(f"   take-profit {le:+.4f}/contract")
        better = "TAKE-PROFIT" if le > lh else "holding"
        print(f"   -> {better} wins by {abs(le - lh):.4f}/contract on the trades "
              f"the bot would actually take")

    below = [g for g in graded if abs(g[0].move_bps) < min_move]
    if below:
        bh = statistics.fmean(g[2] for g in below)
        print(f"\n Below the threshold ({len(below):,} entries the bot correctly "
              f"skips): hold {bh:+.4f}/contract")

    compare_policies(graded, min_move, take_profit)

    markets_total = len({s.ticker for s, *_ in graded})
    print("-" * 78)
    print(f"\n SAMPLE SIZE: {markets_total} settled markets. Every heartbeat on one")
    print(" market shares that market's single outcome, so the {:,} rows above are"
          .format(len(graded)))
    print(" NOT independent - the effective sample is closer to the market count.")
    if markets_total < 30:
        print(" Under ~30 markets, treat every number here as a direction to")
        print(" investigate rather than a result to trade on.")


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="*", help="session logs (default: every logs/L_*.log)")
    p.add_argument("--anchor-age", type=float, default=20.0,
                   help="lookback for the spot move, seconds (match --anchor-age)")
    p.add_argument("--min-seconds-left", type=float, default=20.0,
                   help="ignore entries closer than this to expiry")
    p.add_argument("--min-move", type=float, default=6.0,
                   help="the live --stale-min-move. Entries below it are shown for "
                        "contrast but excluded from the headline, because the bot "
                        "would never take them - averaging them in buries the "
                        "tradeable range under thousands of noise entries.")
    p.add_argument("--take-profit", type=float, default=1.5,
                   help="net multiple of cost at which the exit policy sells")
    p.add_argument("--buckets", default=",".join(f"{b:g}" for b in DEFAULT_BUCKETS),
                   help="comma-separated bucket edges in bps")
    args = p.parse_args()

    paths = [Path(x) for x in (args.logs or sorted(glob.glob("logs/L_*.log")))]
    paths = [x for x in paths if x.is_file()]
    if not paths:
        print("No logs found. Run a session first.", file=sys.stderr)
        return 1

    obs: list[Obs] = []
    for path in paths:
        found = parse_log(path)
        obs.extend(found)
        print(f"  {path.name}: {len(found):,} observations", file=sys.stderr)
    if not obs:
        print("No parseable heartbeats in those logs.", file=sys.stderr)
        return 1

    samples = build_samples(obs, args.anchor_age, args.min_seconds_left)
    print(f"  reconstructed {len(samples):,} entries from {len(obs):,} observations\n",
          file=sys.stderr)
    if not samples:
        print("No entries could be reconstructed.", file=sys.stderr)
        return 1

    async with aiohttp.ClientSession(
        headers={"User-Agent": "kalshi-analyze/1.0"}
    ) as session:
        client = KalshiClient(session)
        settled = await settlement(client, sorted({s.ticker for s in samples}))

    buckets = tuple(float(b) for b in args.buckets.split(",") if b.strip())
    report(samples, settled, buckets, args.take_profit, args.min_move)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
