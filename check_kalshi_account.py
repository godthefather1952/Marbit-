#!/usr/bin/env python3
"""Verify a Kalshi API key end to end. Reads only - it cannot trade.

    # kalshi.com -> Profile -> API Keys -> create a key, download the RSA .pem
    export KALSHI_API_KEY_ID=...
    export KALSHI_PRIVATE_KEY_PATH=/path/to/kalshi-key.pem
    python check_kalshi_account.py

Or put them in .env (gitignored).

Checks in layers, so a failure names the layer that broke:

    1. exchange reachable and trading open      (no credentials needed)
    2. a live KXBTC15M market with its strike   (no credentials needed)
    3. credentials present and the PEM loads
    4. authenticated read -> your balance
    5. positions
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

import aiohttp

from btc_polymarket_arb import load_dotenv
from kalshi import (
    BTC_15M_SERIES,
    KalshiAuthError,
    KalshiClient,
    KalshiCredentials,
    fee_per_contract,
    usd_spot_rest,
)

OK, BAD, WARN = "  ok ", " FAIL", " warn"


async def run(env_file: str, verbose: bool) -> int:
    load_dotenv(env_file)
    creds = KalshiCredentials.from_env()
    print("Kalshi account check (read-only)\n")

    async with aiohttp.ClientSession() as session:
        client = KalshiClient(session, creds)

        # 1. exchange ------------------------------------------------------- #
        try:
            status = await client.exchange_status()
        except Exception as exc:  # noqa: BLE001
            print(f"{BAD} cannot reach Kalshi: {exc}")
            return 1
        trading = bool((status or {}).get("trading_active"))
        print(f"{OK} exchange reachable (trading_active={trading})")
        if not trading:
            print(f"{WARN} trading is currently closed; books may be empty")

        # 2. a live market -------------------------------------------------- #
        try:
            markets = await client.live_markets(BTC_15M_SERIES)
        except Exception as exc:  # noqa: BLE001
            print(f"{BAD} market lookup failed: {exc}")
            return 1

        now = time.time()
        live = [m for m in markets if m.is_live(now)]
        if not live:
            print(f"{WARN} no open {BTC_15M_SERIES} market this instant (they roll every 15m)")
        else:
            m = live[0]
            print(f"{OK} live market {m.ticker}")
            print(f"        strike ${m.strike:,.2f}   {m.seconds_remaining(now):.0f}s to close")
            print(f"        volume {m.volume:,.0f} contracts")
            try:
                book = await client.book(m.ticker)
                if book.yes_ask is not None:
                    print(
                        f"        yes {book.yes_bid:.3f} / {book.yes_ask:.3f}"
                        f"   fee at ask {fee_per_contract(book.yes_ask) * 100:.2f}c/contract"
                    )
                else:
                    print("        book is empty right now")
            except Exception as exc:  # noqa: BLE001
                print(f"{WARN} book unavailable: {exc}")

        spot = await usd_spot_rest(session)
        if spot:
            print(f"{OK} USD spot reference (BRTI-aligned): ${spot:,.2f}")

        # 3. credentials ---------------------------------------------------- #
        problems = creds.problems()
        if problems:
            for issue in problems:
                print(f"{BAD} {issue}")
            print(
                "\nMarket data works without credentials, so the monitor is already usable:\n"
                "    python kalshi_monitor.py\n"
                "For balances, create a key at kalshi.com (Profile -> API Keys) and set\n"
                "KALSHI_API_KEY_ID plus KALSHI_PRIVATE_KEY_PATH."
            )
            return 1
        print(f"{OK} credentials present ({creds.describe()})")

        try:
            client.authenticate()
        except KalshiAuthError as exc:
            print(f"{BAD} could not load the private key: {exc}")
            return 1
        print(f"{OK} RSA private key loaded and signer built")

        # 4. authenticated read --------------------------------------------- #
        try:
            balance = await client.balance()
        except KalshiAuthError as exc:
            print(f"{BAD} authentication rejected: {exc}")
            print(
                "\nThe request reached Kalshi and was refused, so transport is fine.\n"
                "Usual causes: key deleted, the wrong .pem, or a clock more than a\n"
                "few seconds off (signatures are timestamped)."
            )
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"{BAD} balance request failed: {exc}")
            return 1

        print(f"{OK} authenticated request accepted")
        cents = (balance or {}).get("balance")
        if isinstance(cents, (int, float)):
            print(f"\nBalance   ${cents / 100:,.2f}")
        else:
            print(f"\n  raw balance payload: {balance}")
        if verbose:
            print(f"  raw: {balance}")

        # 5. positions ------------------------------------------------------ #
        try:
            positions = await client.positions()
            rows = (positions or {}).get("market_positions") or []
            print(f"\n{OK} positions readable ({len(rows)} open)")
            if verbose and rows:
                for p in rows[:5]:
                    print(f"        {p}")
        except Exception as exc:  # noqa: BLE001
            print(f"{WARN} positions unavailable: {exc}")

    print("\nAuth, transport and account reads all work. This tool cannot place orders.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    try:
        return asyncio.run(run(args.env_file, args.verbose))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
