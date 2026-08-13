#!/usr/bin/env python3
"""Verify a Polymarket US API key end to end. Reads only - it cannot trade.

    export POLYMARKET_US_KEY_ID=...
    export POLYMARKET_US_SECRET_KEY=...
    python check_us_account.py

Or put them in .env (gitignored) and run `python check_us_account.py`.

Checks, in order, so a failure tells you which layer broke:

    1. credentials present and the secret decodes to a valid Ed25519 seed
    2. local clock within the venue's 30s signature window
    3. public gateway reachable (no auth involved)
    4. authenticated read succeeds -> your balance
    5. positions
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import aiohttp

from btc_polymarket_arb import load_dotenv
from polymarket_us import (
    AuthError,
    PolymarketUSClient,
    USCredentials,
    clock_skew_warning,
    summarize_balances,
)

OK, BAD, WARN = "  ok ", " FAIL", " warn"


async def run(env_file: str, verbose: bool) -> int:
    load_dotenv(env_file)
    creds = USCredentials.from_env()

    print("Polymarket US account check (read-only)\n")

    # 1. credentials -------------------------------------------------------- #
    problems = creds.problems()
    if problems:
        for issue in problems:
            print(f"{BAD} {issue}")
        print(
            "\nGet keys at polymarket.us/developer, then set POLYMARKET_US_KEY_ID and\n"
            "POLYMARKET_US_SECRET_KEY in your environment or .env."
        )
        return 1
    print(f"{OK} credentials present ({creds.describe()})")
    print(f"{OK} secret decodes to a valid 32-byte Ed25519 seed")

    # 2. clock -------------------------------------------------------------- #
    skew = clock_skew_warning()
    print(f"{WARN} {skew}" if skew else f"{OK} local clock within the 30s signature window")

    async with aiohttp.ClientSession() as session:
        client = PolymarketUSClient(session, creds)

        # 3. public gateway ------------------------------------------------- #
        try:
            markets = await client.markets(categories="crypto", limit=3)
            rows = markets.get("markets", []) if isinstance(markets, dict) else []
            print(f"{OK} public gateway reachable ({len(rows)} crypto markets sampled)")
            if verbose:
                for m in rows:
                    print(f"        {m.get('slug')} | {(m.get('question') or '')[:50]}")
        except Exception as exc:  # noqa: BLE001
            print(f"{BAD} public gateway unreachable: {exc}")
            return 1

        # 4. authenticated read --------------------------------------------- #
        try:
            client.authenticate()
        except AuthError as exc:
            print(f"{BAD} could not build signer: {exc}")
            return 1

        try:
            balances = await client.account_balances()
        except AuthError as exc:
            print(f"{BAD} authentication rejected: {exc}")
            print(
                "\nThe request reached the venue and was refused, so the transport is fine.\n"
                "Usual causes: key revoked or regenerated, keys from a different sign-in\n"
                "method than the app, or identity verification not yet approved."
            )
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"{BAD} balance request failed: {exc}")
            return 1

        print(f"{OK} authenticated request accepted")
        cash = summarize_balances(balances)
        if cash:
            print("\nBalances")
            for currency, amount in cash:
                print(f"        {amount:>12,.2f} {currency}")
        else:
            print(f"{WARN} balance response parsed but held no recognizable cash field")
        if verbose or not cash:
            print(f"\n  raw: {balances}")

        # 5. positions ------------------------------------------------------ #
        try:
            positions = await client.positions()
            rows = positions.get("positions", []) if isinstance(positions, dict) else positions
            print(f"\n{OK} positions readable ({len(rows or [])} open)")
            if verbose and rows:
                for p in list(rows)[:5]:
                    print(f"        {p}")
        except Exception as exc:  # noqa: BLE001
            print(f"{WARN} positions unavailable: {exc}")

    print("\nAuth, transport and account reads all work. This tool cannot place orders.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env", help="dotenv path")
    parser.add_argument("--verbose", action="store_true", help="print raw payloads")
    args = parser.parse_args()
    try:
        return asyncio.run(run(args.env_file, args.verbose))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
