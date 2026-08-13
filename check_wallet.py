#!/usr/bin/env python3
"""Check a Polymarket wallet's USDC balance and CTF Exchange allowance.

Takes a *public address only* - no API key, no secret, no private key. Balances
are public on-chain data.

    python check_wallet.py 0xYourWalletAddress
    python check_wallet.py 0xYour... --rpc https://polygon-mainnet.g.alchemy.com/v2/KEY

Use this to confirm your funds and approvals are in place before arming the bot
with --live.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import aiohttp

from btc_polymarket_arb import PolygonRpc, json_loads

#: Public Polygon endpoints, tried in order. Override with --rpc (a private
#: Alchemy/QuickNode/Chainstack endpoint is faster and rate-limit free).
PUBLIC_RPCS = (
    "https://polygon-bor-rpc.publicnode.com",
    "https://1rpc.io/matic",
    "https://polygon.drpc.org",
)

DATA_API = "https://data-api.polymarket.com/value"


async def onchain(session: aiohttp.ClientSession, address: str, rpcs) -> bool:
    from py_clob_client.config import get_contract_config

    for url in rpcs:
        try:
            rpc = PolygonRpc(session, url)
            balance = await asyncio.wait_for(rpc.usdc_balance(address), 20)
            results = [("standard", get_contract_config(137, False).exchange),
                       ("neg-risk", get_contract_config(137, True).exchange)]
            allowances = []
            for label, exchange in results:
                allowances.append((label, exchange, await rpc.usdc_allowance(address, exchange)))
        except Exception as exc:  # noqa: BLE001 - try the next endpoint
            print(f"  {url} -> {type(exc).__name__}: {str(exc)[:70]}")
            continue

        print(f"\nOn-chain (via {url})")
        print(f"  USDC.e balance                 ${balance:,.2f}")
        for label, exchange, allowance in allowances:
            note = "  <-- NOT APPROVED" if allowance <= 0 else ""
            print(f"  allowance -> {label:<9} exchange  ${allowance:,.2f}{note}")

        if balance <= 0:
            print("\n  No USDC at this address. If you funded through the Polymarket UI,")
            print("  the money sits in your PROXY wallet, not your signing wallet -")
            print("  use the deposit address shown in the UI, and set POLYMARKET_FUNDER.")
        if all(a <= 0 for _, _, a in allowances):
            print("\n  Zero allowance: the exchange cannot pull your USDC, so every order")
            print("  would be rejected on-chain. Approve it in the Polymarket UI first.")
        return True
    return False


async def portfolio(session: aiohttp.ClientSession, address: str) -> None:
    try:
        async with session.get(
            DATA_API, params={"user": address}, timeout=aiohttp.ClientTimeout(total=15)
        ) as resp:
            if resp.status != 200:
                print(f"\nPolymarket data API returned HTTP {resp.status}")
                return
            rows = json_loads(await resp.read())
    except Exception as exc:  # noqa: BLE001
        print(f"\nPolymarket data API unavailable: {exc}")
        return

    if isinstance(rows, list) and rows:
        print(f"\nPolymarket portfolio value          ${float(rows[0].get('value') or 0):,.2f}")
        print("  (open positions marked to market, not spendable cash)")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("address", help="public wallet address (0x...)")
    parser.add_argument("--rpc", help="private Polygon RPC URL (recommended)")
    args = parser.parse_args()

    address = args.address.strip()
    if not address.startswith("0x") or len(address) != 42:
        print(f"'{address}' is not a 0x-prefixed 20-byte address", file=sys.stderr)
        return 2

    rpcs = [args.rpc] if args.rpc else list(PUBLIC_RPCS)
    print(f"Wallet {address}")

    async with aiohttp.ClientSession() as session:
        ok = await onchain(session, address, rpcs)
        if not ok:
            print("\nEvery RPC endpoint failed. Pass a private one with --rpc.")
        await portfolio(session, address)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
