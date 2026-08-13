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


def score(rows: list[dict], settled: dict[str, dict]) -> None:
    by_strategy: dict[str, list] = collections.defaultdict(list)
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
        by_strategy[row["strategy"]].append(
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
            }
        )

    print("=" * 72)
    print(" SETTLED RESULTS - actual outcomes, not model estimates")
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

        print(f"\n {strategy}")
        print(f"   trades          {len(trades)}")
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
        print("\n" + "-" * 72)
        print(f" TOTAL  {total_n} trades, {total_wins} won ({total_wins / total_n * 100:.1f}%), "
              f"staked ${grand_cost:,.2f}")
        print(f"        ACTUAL ${grand_net:+,.2f}   vs model's ${grand_exp:+,.2f}")
        if grand_cost:
            print(f"        return on stake: {grand_net / grand_cost * 100:+.1f}%")
        print("-" * 72)
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

    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        print(f"{path} holds no trades.", file=sys.stderr)
        return 1

    print(f"Scoring {len(rows)} paper trades from {path}\n")
    async with aiohttp.ClientSession() as session:
        client = KalshiClient(session)
        settled = await settlement(client, sorted({r["ticker"] for r in rows}))
    score(rows, settled)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
