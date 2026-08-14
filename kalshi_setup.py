#!/usr/bin/env python3
"""First-run credential setup for live Kalshi trading.

    python kalshi_setup.py            # interactive: prompt, verify, save
    python kalshi_setup.py --check    # non-interactive: verify what .env holds

Called automatically by `kalshi_monitor.py --live` when credentials are missing
or fail verification. The flow is deliberately strict:

    1. ask for the API key ID and the RSA private key (a .pem path, or pasted)
    2. prove the pair works by fetching the account balance over a signed call
    3. show the balance and ask the human to confirm it matches their account -
       the one check that catches "valid key, wrong account"
    4. only then write .env, and re-prompt from the top on any failure

Nothing is persisted until a signed request has succeeded AND the human has
recognised the balance. A pasted PEM is stored in its own chmod-600 file
(`kalshi_key.pem`), not inline in .env: the .env parser is line-based and a
multi-line PEM would silently truncate to its first line - a corrupted key
discovered only at order time.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import stat
import sys
from pathlib import Path

import aiohttp

from btc_polymarket_arb import load_dotenv
from kalshi import KalshiAuthError, KalshiClient, KalshiCredentials

PEM_FILE = "kalshi_key.pem"
MAX_ATTEMPTS = 3


# --------------------------------------------------------------------------- #
# Verification - a signed round trip, not a format check
# --------------------------------------------------------------------------- #


async def verify(creds: KalshiCredentials) -> tuple[float | None, str]:
    """Prove the credentials against the live API. Returns (balance, error).

    A PEM that parses is not a key that works: the key ID may belong to a
    different key, be revoked, or belong to someone else's account. The only
    real test is a signed request, and `balance` is the cheapest one that also
    yields the number the human can recognise as theirs.
    """
    problems = creds.problems()
    if problems:
        return None, "; ".join(problems)
    async with aiohttp.ClientSession(
        headers={"User-Agent": "kalshi-setup/1.0"}
    ) as session:
        client = KalshiClient(session, creds)
        try:
            client.authenticate()
        except KalshiAuthError as exc:
            return None, f"key rejected locally: {exc}"
        try:
            payload = await client.balance()
        except KalshiAuthError as exc:
            return None, f"Kalshi rejected the signature: {exc}"
        except Exception as exc:  # noqa: BLE001
            return None, f"could not reach Kalshi: {exc}"
    cents = (payload or {}).get("balance")
    if not isinstance(cents, (int, float)):
        return None, f"balance endpoint returned an unexpected payload: {payload!r}"
    return float(cents) / 100.0, ""


# --------------------------------------------------------------------------- #
# Prompting
# --------------------------------------------------------------------------- #


def _prompt_key_id() -> str:
    while True:
        key_id = input("Kalshi API key ID (Profile -> API Keys): ").strip()
        if key_id:
            return key_id
        print("  A key ID is required.")


def _prompt_private_key() -> tuple[str | None, str | None]:
    """Returns (path, pem_text) - exactly one is set.

    Accepts either a path to the downloaded .pem or the PEM pasted directly.
    Pasting reads until the -----END line, because a PEM is multi-line and a
    single input() would keep only its header.
    """
    while True:
        raw = input(
            "RSA private key - path to the .pem file, or press Enter to paste it: "
        ).strip()
        if raw:
            path = Path(raw).expanduser()
            if not path.is_file():
                print(f"  No file at {path}.")
                continue
            text = path.read_text(encoding="utf-8")
            if "PRIVATE KEY" not in text:
                print(f"  {path} does not look like a PEM private key.")
                continue
            return str(path), None

        print("  Paste the PEM now (it ends with an -----END ... KEY----- line):")
        lines: list[str] = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            lines.append(line)
            if "-----END" in line:
                break
        pem = "\n".join(lines).strip() + "\n"
        if "PRIVATE KEY" in pem and "-----END" in pem:
            return None, pem
        print("  That did not parse as a complete PEM block; try again.")


def _confirm(question: str) -> bool:
    return input(f"{question} [y/N]: ").strip().lower() in ("y", "yes")


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def save_credentials(
    env_file: str | Path,
    key_id: str,
    pem_path: str | None = None,
    pem_text: str | None = None,
) -> None:
    """Write the verified credentials to .env, preserving everything else.

    Called only after verification has passed - a failed key must never be
    saved, or the next run inherits the failure with no prompt to fix it.
    """
    env_path = Path(env_file)
    if pem_text is not None:
        key_file = env_path.parent / PEM_FILE
        key_file.write_text(pem_text, encoding="utf-8")
        # Owner read/write only: this file IS the trading authority.
        key_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
        pem_path = str(key_file)
        print(f"  Private key written to {key_file} (permissions 600).")

    updates = {
        "KALSHI_API_KEY_ID": key_id,
        "KALSHI_PRIVATE_KEY_PATH": pem_path or "",
    }
    lines = (
        env_path.read_text(encoding="utf-8").splitlines() if env_path.is_file() else []
    )
    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        name = line.split("=", 1)[0].strip().removeprefix("export ").strip()
        if name in updates:
            out.append(f"{name}={updates[name]}")
            seen.add(name)
        else:
            out.append(line)
    missing = [k for k in updates if k not in seen]
    if missing:
        if out and out[-1].strip():
            out.append("")
        out.append("# Kalshi credentials (written by kalshi_setup.py after verification)")
        out.extend(f"{k}={updates[k]}" for k in missing)
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    with contextlib.suppress(OSError):
        env_path.chmod(stat.S_IRUSR | stat.S_IWUSR)

    # The process that prompted should see them immediately, without a restart.
    os.environ["KALSHI_API_KEY_ID"] = key_id
    os.environ["KALSHI_PRIVATE_KEY_PATH"] = pem_path or ""
    os.environ.pop("KALSHI_PRIVATE_KEY", None)
    print(f"  Saved to {env_path}.")


# --------------------------------------------------------------------------- #
# The flow
# --------------------------------------------------------------------------- #


def ensure_credentials(env_file: str | Path = ".env") -> KalshiCredentials | None:
    """Return verified credentials, prompting and saving as needed.

    Synchronous on purpose: it runs before the event loop and the market tasks
    start, so blocking on input() costs nothing and cannot stall a live tape.
    """
    load_dotenv(env_file)
    creds = KalshiCredentials.from_env()

    if creds.complete:
        print(f"Verifying stored Kalshi credentials ({creds.describe()})...")
        balance, error = asyncio.run(verify(creds))
        if balance is not None:
            print(f"  ok - authenticated, balance ${balance:,.2f}")
            if _confirm("  Is that the balance you expect on this account?"):
                return creds
            print(
                "  Treating that as the wrong account - re-enter the credentials."
            )
        else:
            print(f"  Stored credentials failed: {error}")

    if not sys.stdin.isatty():
        print("No terminal to prompt on; fill .env by hand or run kalshi_setup.py "
              "from an interactive shell.")
        return None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\nKalshi credential setup (attempt {attempt}/{MAX_ATTEMPTS})")
        key_id = _prompt_key_id()
        pem_path, pem_text = _prompt_private_key()
        candidate = KalshiCredentials(
            key_id=key_id,
            private_key_pem=pem_text
            or Path(pem_path).read_text(encoding="utf-8"),
        )
        print("  Verifying with a signed balance request...")
        balance, error = asyncio.run(verify(candidate))
        if balance is None:
            print(f"  FAILED: {error}")
            continue
        print(f"  Authenticated. Account balance: ${balance:,.2f}")
        if not _confirm("  Is that the balance you expect on this account?"):
            # Valid key, unrecognised balance = probably the wrong account's
            # key. Saving it would aim every order at someone else's money.
            print("  Not saving - re-enter the credentials for the right account.")
            continue
        save_credentials(env_file, key_id, pem_path=pem_path, pem_text=pem_text)
        return KalshiCredentials.from_env()

    print(f"\nGiving up after {MAX_ATTEMPTS} attempts. Nothing was saved.")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--check", action="store_true",
                        help="verify what .env already holds; never prompt")
    args = parser.parse_args()

    if args.check:
        load_dotenv(args.env_file)
        creds = KalshiCredentials.from_env()
        balance, error = asyncio.run(verify(creds))
        if balance is None:
            print(f"FAILED: {error}")
            return 1
        print(f"ok - authenticated, balance ${balance:,.2f}")
        return 0

    if not sys.stdin.isatty():
        print("No terminal to prompt on; run interactively or fill .env by hand.")
        return 1
    return 0 if ensure_credentials(args.env_file) is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
