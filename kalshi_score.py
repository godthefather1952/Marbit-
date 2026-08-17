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


def _execution_index(executions: list[dict]) -> dict[tuple[str, str], str]:
    """Best-known outcome per (strategy, ticker).

    "filled" is the only outcome that moved money and wins over everything
    else, because a signal can be skipped once and filled on a later pass.
    """
    rank = {"filled": 3, "rejected": 2, "simulated": 1, "skipped": 0}
    best: dict[tuple[str, str], str] = {}
    for row in executions:
        key = (row.get("strategy", ""), row.get("ticker", ""))
        outcome = str(row.get("outcome") or "")
        if rank.get(outcome, -1) > rank.get(best.get(key, ""), -1):
            best[key] = outcome
    return best


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

        gross = 0.0
        cost = 0.0
        fees = 0.0
        for leg in row["legs"]:
            size, price, side = leg["size"], leg["price"], leg["side"]
            cost += size * price
            fees += trading_fee(price, size)
            won = (side == "YES" and result == "yes") or (side == "NO" and result == "no")
            gross += size * (1.0 if won else 0.0)
        note = str(row.get("note") or "")
        asset = note[1:note.index("]")] if note.startswith("[") and "]" in note else ""
        key = f"{asset}/{row['strategy']}" if asset else row["strategy"]
        outcome = executed.get((row["strategy"], row["ticker"]), "unknown")
        if outcome == "filled":
            real_keys.add(key)
        by_strategy[key].append(
            {
                "ticker": row["ticker"],
                "net": gross - cost - fees,
                "cost": cost,
                "fees": fees,
                "won": gross > cost,
                "result": result,
                "fair_yes": row.get("fair_yes"),
                "expected": row.get("expected_net", 0.0),
                "legs": row["legs"],
                "outcome": outcome,
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
    parser.add_argument("ledger", nargs="?", help="paper ledger .jsonl (default: newest in logs/)")
    args = parser.parse_args()

    path = args.ledger
    if not path:
        candidates = sorted(glob.glob("logs/paper_*.jsonl"))
        if not candidates:
            print("No paper ledger found. Run kalshi_monitor.py first.", file=sys.stderr)
            return 1
        path = candidates[-1]

    rows, executions = [], []
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
        print(f"{path} holds no trades.", file=sys.stderr)
        return 1

    print(f"Scoring {len(rows)} recorded signals from {path}\n")
    async with aiohttp.ClientSession() as session:
        client = KalshiClient(session)
        settled = await settlement(client, sorted({r["ticker"] for r in rows}))
    score(rows, settled, executions)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
