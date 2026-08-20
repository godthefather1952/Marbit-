#!/usr/bin/env python3
"""Grade paper trades against what actually settled.

    python kalshi_score.py                    # scores logs/paper_*.jsonl
    python kalshi_score.py logs/paper_X.jsonl

Every other number this project has produced compares our model to the
market's price. This is the only one that compares it to reality: Kalshi
publishes `result` ("yes"/"no") and the settlement value for every expired
market, so each recorded trade can be marked won or lost and turned into an
actual dollar figure, fees included.

Read the hit rate before the PnL. A strategy needs its win rate to beat the
price it paid: buying at 0.60 needs better than 60% to break even before fees,
and buying at 0.97 needs better than 97%. A high hit rate on cheap-looking
contracts can still lose money.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import glob
import json
import sys
from pathlib import Path

import aiohttp

from kalshi import KalshiClient, trading_fee


async def settlement(client: KalshiClient, tickers: list[str]) -> dict[str, dict]:
    """Fetch settled outcomes for the tickers we traded on paper."""
    out: dict[str, dict] = {}
    for ticker in tickers:
        try:
            payload = await client._get(f"/markets/{ticker}")
            market = (payload or {}).get("market") or {}
            if market.get("result"):
                out[ticker] = market
        except Exception as exc:  # noqa: BLE001 - unsettled or unknown is normal
            print(f"  ({ticker}: {str(exc)[:60]})", file=sys.stderr)
    return out


def _execution_index(executions: list[dict]) -> dict[tuple[str, str], dict]:
    """What actually happened per (strategy, ticker): outcome, size, prices.

    Two things this has to carry beyond the outcome, because grading from the
    SIGNAL alone was wrong in both directions:

    `count`/`price` - the signal names a nominal size (--size, 20 by default)
    and the ask it saw. What we actually got was 3 contracts at a crossed
    limit. Scoring the nominal size overstated a real +$1.75 session as
    +$52.72, roughly thirtyfold, on an account that only holds $23.

    `exit_price` - a position closed early does not settle. Grading it as
    held-to-expiry can invert the result: a contract bought at 0.963 and sold
    at 0.974 made 1.1c whatever the market did afterwards.
    """
    rank = {"filled": 3, "rejected": 2, "simulated": 1, "skipped": 0}
    out: dict[tuple[str, str], dict] = {}
    for row in executions:
        key = (row.get("strategy", ""), row.get("ticker", ""))
        rec = out.setdefault(key, {"outcome": "", "count": 0.0, "price": 0.0,
                                   "exit_price": None, "signal_price": None,
                                   "fill_ms": None, "entry_ts": None,
                                   "exit_ts": None, "attempts": 0})
        outcome = str(row.get("outcome") or "")
        if outcome == "closed":
            rec["exit_price"] = row.get("price")
            rec["exit_ts"] = row.get("ts")
            continue
        if outcome in ("filled", "simulated", "rejected"):
            rec["attempts"] += 1
        if rank.get(outcome, -1) > rank.get(rec["outcome"], -1):
            rec["outcome"] = outcome
            if outcome in ("filled", "simulated"):
                rec["count"] = float(row.get("count") or 0.0)
                rec["price"] = float(row.get("price") or 0.0)
                rec["signal_price"] = row.get("signal_price")
                rec["fill_ms"] = row.get("elapsed_ms")
                rec["entry_ts"] = row.get("ts")
    return out


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def score(
    rows: list[dict],
    settled: dict[str, dict],
    executions: list[dict] | None = None,
) -> None:
    # Keyed by "ASSET/STRATEGY" when the ledger carries an asset tag, so a
    # BTC+ETH session is graded as two experiments rather than one blended
    # number that can hide a losing instrument behind a winning one.
    by_strategy: dict[str, list] = collections.defaultdict(list)
    executed = _execution_index(executions or [])
    real_keys: set[str] = set()
    pending = 0

    for row in rows:
        market = settled.get(row["ticker"])
        if not market:
            pending += 1
            continue
        result = str(market.get("result", "")).lower()  # "yes" | "no"
        rec = executed.get((row["strategy"], row["ticker"]))

        gross = 0.0
        cost = 0.0
        fees = 0.0
        exit_price = rec.get("exit_price") if rec else None
        for leg in row["legs"]:
            # Prefer what actually filled over what the signal proposed.
            size = rec["count"] if rec and rec.get("count") else leg["size"]
            price = rec["price"] if rec and rec.get("price") else leg["price"]
            side = leg["side"]
            cost += size * price
            fees += trading_fee(price, size)
            if exit_price is not None:
                # Sold before expiry: the exit price IS the outcome, and the
                # sale pays its own fee.
                gross += size * float(exit_price)
                fees += trading_fee(float(exit_price), size)
            else:
                won = (side == "YES" and result == "yes") or (
                    side == "NO" and result == "no"
                )
                gross += size * (1.0 if won else 0.0)
        note = str(row.get("note") or "")
        asset = note[1:note.index("]")] if note.startswith("[") and "]" in note else ""
        key = f"{asset}/{row['strategy']}" if asset else row["strategy"]
        outcome = rec["outcome"] if rec else "unknown"
        if outcome == "filled":
            real_keys.add(key)
        by_strategy[key].append(
            {
                "ticker": row["ticker"],
                "net": gross - cost - fees,
                "cost": cost,
                "fees": fees,
                "won": gross > cost,
                "exited": exit_price is not None,
                "result": result,
                "fair_yes": row.get("fair_yes"),
                "expected": row.get("expected_net", 0.0),
                "legs": row["legs"],
                "outcome": outcome,
                "contracts": sum(
                    (rec["count"] if rec and rec.get("count") else leg["size"])
                    for leg in row["legs"]
                ),
                "attempts": (rec or {}).get("attempts", 0),
                "signal_price": (rec or {}).get("signal_price"),
                "fill_price": (rec or {}).get("price"),
                "fill_ms": (rec or {}).get("fill_ms"),
                "entry_ts": (rec or {}).get("entry_ts"),
                "exit_ts": (rec or {}).get("exit_ts"),
            }
        )

    print("=" * 72)
    print(" SETTLED RESULTS - real outcomes, split by what actually executed")
    print("=" * 72)
    if pending:
        print(f"\n {pending} trade(s) not yet settled; re-run later to include them.")

    grand_net = grand_cost = grand_exp = 0.0
    total_n = total_wins = 0

    for strategy, trades in sorted(by_strategy.items()):
        net = sum(t["net"] for t in trades)
        cost = sum(t["cost"] for t in trades)
        fees = sum(t["fees"] for t in trades)
        expected = sum(t["expected"] for t in trades)
        wins = sum(1 for t in trades if t["net"] > 0)
        avg_price = (
            sum(leg["price"] * leg["size"] for t in trades for leg in t["legs"])
            / max(sum(leg["size"] for t in trades for leg in t["legs"]), 1e-9)
        )
        grand_net += net
        grand_cost += cost
        grand_exp += expected
        total_n += len(trades)
        total_wins += wins

        outcomes = collections.Counter(t["outcome"] for t in trades)
        real = outcomes.get("filled", 0)
        tag = "REAL MONEY" if real else "PAPER ONLY - no money moved"
        print(f"\n {strategy}   [{tag}]")
        if real and real < len(trades):
            print(f"   {real} of {len(trades)} actually filled; the rest were "
                  f"{', '.join(f'{v} {k}' for k, v in outcomes.items() if k != 'filled')}")
        elif not real:
            print(f"   ({', '.join(f'{v} {k}' for k, v in outcomes.items())})")
        print(f"   signals         {len(trades)}")
        print(f"   hit rate        {wins}/{len(trades)} ({wins / len(trades) * 100:.1f}%)")
        print(f"   avg price paid  {avg_price:.3f}   (needs a {avg_price * 100:.0f}%+ hit "
              f"rate just to break even before fees)")
        print(f"   staked          ${cost:,.2f}")
        print(f"   fees            ${fees:,.2f}")
        print(f"   ACTUAL net      ${net:+,.2f}   ({net / cost * 100:+.1f}% on stake)"
              if cost else f"   ACTUAL net      ${net:+,.2f}")
        print(f"   model predicted ${expected:+,.2f}")

        # Execution quality, separate from whether the thesis was right. A
        # strategy can call direction correctly on every trade and still lose,
        # if the price it pays is worse than the quote that justified it.
        contracts = sum(t["contracts"] for t in trades)
        attempts = sum(t["attempts"] for t in trades)
        got = sum(1 for t in trades if t["outcome"] in ("filled", "simulated"))
        if attempts:
            print(f"   fill rate       {got}/{attempts} attempts "
                  f"({got / attempts * 100:.0f}%), {contracts:g} contracts")
        slips = [
            (t["fill_price"] - t["signal_price"]) * 100.0
            for t in trades
            if t.get("fill_price") and t.get("signal_price")
        ]
        if slips:
            print(f"   slippage        {sum(slips) / len(slips):+.2f}c/contract "
                  f"vs the quote that produced the signal (n={len(slips)})")
        fill_ms = _median([t["fill_ms"] for t in trades if t.get("fill_ms")])
        if fill_ms is not None:
            print(f"   time to fill    {fill_ms / 1000.0:.1f}s median")
        holds = _median([
            t["exit_ts"] - t["entry_ts"] for t in trades
            if t.get("exit_ts") and t.get("entry_ts")
        ])
        if holds is not None:
            print(f"   time to exit    {holds:.0f}s median "
                  f"({sum(1 for t in trades if t['exited'])} closed early)")
        if contracts:
            print(f"   per contract    predicted {expected / contracts:+.4f} "
                  f"vs realized {net / contracts:+.4f}")

        if expected > 0:
            verdict = (
                "model was OPTIMISTIC" if net < expected * 0.5
                else "model roughly tracked reality" if net > 0
                else "model was WRONG - predicted profit, took a loss"
            )
            print(f"   -> {verdict}")

    if total_n:
        real_trades = [t for ts in by_strategy.values() for t in ts if t["outcome"] == "filled"]
        real_net = sum(t["net"] for t in real_trades)
        real_cost = sum(t["cost"] for t in real_trades)
        print("\n" + "=" * 72)
        # The headline is what the account actually did. A previous session
        # reported +$3.85 across five "wins" while the balance moved ten cents,
        # because two were simulated during warm-up and three were never sent.
        if real_trades:
            print(f" REAL MONEY : {len(real_trades)} filled order(s), "
                  f"${real_cost:,.2f} staked, ${real_net:+,.2f}")
        else:
            print(" REAL MONEY : nothing filled. Your balance did not move on any")
            print("              of these - they were simulated or never sent.")
        hypo = total_n - len(real_trades)
        if hypo:
            print(f" HYPOTHETICAL: {hypo} signal(s) graded as if taken, "
                  f"${grand_cost - real_cost:,.2f} would have been staked for "
                  f"${grand_net - real_net:+,.2f}")
        print("=" * 72)
        print(f" all {total_n} graded, {total_wins} won ({total_wins / total_n * 100:.1f}%), "
              f"model predicted ${grand_exp:+,.2f}")
        if grand_cost:
            print(f" combined return on stake: {grand_net / grand_cost * 100:+.1f}%")
        if total_n < 30:
            print("\n With fewer than ~30 settled trades this is noise, not a result.")
            print(" A coin flip clears 60% often enough at this sample size.")
    else:
        print("\n Nothing settled yet.")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", nargs="*",
                        help="paper ledger .jsonl file(s). Default: EVERY ledger "
                             "in logs/, so evidence accumulates across sessions.")
    parser.add_argument("--latest", action="store_true",
                        help="score only the most recent session")
    args = parser.parse_args()

    paths = args.ledger or sorted(glob.glob("logs/paper_*.jsonl"))
    if not paths:
        print("No paper ledger found. Run kalshi_main.py first.", file=sys.stderr)
        return 1
    if args.latest:
        paths = paths[-1:]

    # Reading every session by default is the point. A single run yields a
    # handful of graded trades, which is noise; the question "does this
    # strategy make money" only gets answered by pooling them, and scoring one
    # file at a time silently threw away every previous session's evidence.
    rows, executions = [], []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            (executions if obj.get("kind") == "execution" else rows).append(obj)
    if not rows:
        print("No signals recorded in those ledgers.", file=sys.stderr)
        return 1

    print(f"Scoring {len(rows)} recorded signals from {len(paths)} session(s)")
    if len(paths) > 1:
        print(f"  {Path(paths[0]).name} ... {Path(paths[-1]).name}")
    print()
    async with aiohttp.ClientSession() as session:
        client = KalshiClient(session)
        settled = await settlement(client, sorted({r["ticker"] for r in rows}))
    score(rows, settled, executions)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
