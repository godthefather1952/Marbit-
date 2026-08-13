#!/usr/bin/env python3
"""Read-only Polymarket US venue adapter.

Polymarket US (`polymarket.us`) is a separate, CFTC-regulated exchange from the
global `polymarket.com` CLOB that `btc_polymarket_arb.py` trades. Different
markets, different auth, different money:

    global (.com)   EIP-712 orders signed by a Polygon wallet, USDC on-chain,
                    HMAC-SHA256 L2 headers, api_key/secret/passphrase triplet
    US (.us)        Ed25519 request signatures, fiat balances, key id + secret

Nothing here is interchangeable with the global client.

**This module cannot place, modify, or cancel an order.** It exposes reads
only - balances, positions, markets, books. That is deliberate: the first thing
you point at a funded account should not be able to spend it. Order support is
a separate, later decision.

Auth (per docs.polymarket.us/api-reference/authentication):

    message   = f"{timestamp_ms}{METHOD}{path}"
    signature = base64(ed25519_sign(message))
    headers   = X-PM-Access-Key / X-PM-Timestamp / X-PM-Signature

The secret decodes to 64 bytes; the first 32 are the Ed25519 seed. Timestamps
must be within 30 seconds of server time.

Credentials come from the environment (never arguments, never literals):

    POLYMARKET_US_KEY_ID
    POLYMARKET_US_SECRET_KEY
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode

import aiohttp

from btc_polymarket_arb import (
    RETRYABLE_STATUS,
    RetryableError,
    _header_seconds,
    json_loads,
    log,
    retry_async,
)

AUTH_HOST = "https://api.polymarket.us"
GATEWAY_HOST = "https://gateway.polymarket.us"

#: Signatures are rejected outside this window, so a skewed clock reads as an
#: auth failure. Checked explicitly so the error says what it actually is.
MAX_CLOCK_SKEW_SECONDS = 30.0


class AuthError(RuntimeError):
    """Credentials missing, malformed, or rejected by the venue."""


@dataclass(slots=True)
class USCredentials:
    key_id: str | None = None
    secret_key: str | None = None

    @classmethod
    def from_env(cls) -> "USCredentials":
        return cls(
            key_id=(os.getenv("POLYMARKET_US_KEY_ID") or "").strip() or None,
            secret_key=(os.getenv("POLYMARKET_US_SECRET_KEY") or "").strip() or None,
        )

    @property
    def complete(self) -> bool:
        return bool(self.key_id and self.secret_key)

    def problems(self) -> list[str]:
        issues: list[str] = []
        if not self.key_id:
            issues.append("POLYMARKET_US_KEY_ID is not set")
        if not self.secret_key:
            issues.append("POLYMARKET_US_SECRET_KEY is not set")
        elif self.seed_error() is not None:
            issues.append(f"POLYMARKET_US_SECRET_KEY is malformed: {self.seed_error()}")
        return issues

    def seed_error(self) -> str | None:
        try:
            self.seed()
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            return str(exc)
        return None

    def seed(self) -> bytes:
        """The 32-byte Ed25519 seed. The venue issues 64 bytes; take the first half."""
        if not self.secret_key:
            raise AuthError("no secret key")
        try:
            raw = base64.b64decode(self.secret_key, validate=True)
        except Exception as exc:  # noqa: BLE001
            raise AuthError(f"not valid base64 ({exc})") from exc
        if len(raw) < 32:
            raise AuthError(f"decoded to {len(raw)} bytes, need at least 32")
        return raw[:32]

    def describe(self) -> str:
        """Safe to log: identifies the key without revealing the secret."""
        key = f"{self.key_id[:8]}..." if self.key_id else "MISSING"
        return f"key_id={key} secret={'set' if self.secret_key else 'MISSING'}"


class Ed25519Signer:
    """Signs requests. Holds the private key and never exposes or logs it."""

    __slots__ = ("_key_id", "_private")

    def __init__(self, creds: USCredentials) -> None:
        problems = creds.problems()
        if problems:
            raise AuthError("; ".join(problems))

        try:
            from cryptography.hazmat.primitives.asymmetric import ed25519
        except ImportError as exc:  # pragma: no cover
            raise AuthError("pip install cryptography (needed for Ed25519)") from exc

        self._key_id = creds.key_id
        self._private = ed25519.Ed25519PrivateKey.from_private_bytes(creds.seed())

    def headers(self, method: str, path: str) -> dict[str, str]:
        """Auth headers for `METHOD path`.

        `path` must be exactly the path sent on the wire, query string included
        when there is one - the signature covers the string the server rebuilds.
        """
        timestamp = str(int(time.time() * 1000))
        message = f"{timestamp}{method.upper()}{path}".encode()
        signature = base64.b64encode(self._private.sign(message)).decode()
        return {
            "X-PM-Access-Key": self._key_id or "",
            "X-PM-Timestamp": timestamp,
            "X-PM-Signature": signature,
            "Content-Type": "application/json",
        }


class PolymarketUSClient:
    """Read-only client. Authenticated reads plus the public gateway.

    There is intentionally no create/modify/cancel surface here - see the module
    docstring. Adding one should be a deliberate, reviewed change, not something
    that arrives by accident alongside a balance check.
    """

    def __init__(
        self,
        session: aiohttp.ClientSession,
        creds: USCredentials | None = None,
        *,
        auth_host: str = AUTH_HOST,
        gateway_host: str = GATEWAY_HOST,
        sign_query_string: bool = True,
    ) -> None:
        self._session = session
        self._creds = creds or USCredentials.from_env()
        self._auth_host = auth_host.rstrip("/")
        self._gateway_host = gateway_host.rstrip("/")
        #: Whether the signed path includes the query string. The docs only show
        #: a bare path; if authenticated GETs with filters 401, flip this.
        self._sign_query_string = sign_query_string
        self._signer: Ed25519Signer | None = None

    @property
    def credentials(self) -> USCredentials:
        return self._creds

    @property
    def authenticated(self) -> bool:
        return self._signer is not None

    def authenticate(self) -> None:
        """Build the signer. Raises AuthError with a specific reason."""
        self._signer = Ed25519Signer(self._creds)
        log.info("Polymarket US signer ready (%s)", self._creds.describe())

    # -- transport ---------------------------------------------------------- #

    async def _request(
        self, host: str, path: str, params: Mapping[str, Any] | None, signed: bool
    ) -> Any:
        query = ""
        if params:
            flat: list[tuple[str, str]] = []
            for key, value in params.items():
                if value is None:
                    continue
                if isinstance(value, (list, tuple)):
                    flat.extend((key, str(v)) for v in value)
                else:
                    flat.append((key, str(value)))
            if flat:
                query = "?" + urlencode(flat)

        full_path = path + query
        url = host + full_path
        headers: dict[str, str] = {}
        if signed:
            if self._signer is None:
                raise AuthError("authenticate() has not been called")
            headers = self._signer.headers("GET", full_path if self._sign_query_string else path)

        async def once() -> Any:
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
                    raise AuthError(
                        f"{path} -> HTTP {resp.status}: {body[:200].decode('utf-8', 'replace')}"
                    )
                if resp.status != 200:
                    raise RuntimeError(
                        f"{path} -> HTTP {resp.status}: {body[:200].decode('utf-8', 'replace')}"
                    )
                return json_loads(body) if body else None

        return await retry_async(once, attempts=3, label=f"US {path}")

    # -- authenticated reads ------------------------------------------------ #

    async def account_balances(self) -> Any:
        """Cash balance, buying power, and security values."""
        return await self._request(self._auth_host, "/v1/account/balances", None, signed=True)

    async def positions(self, **filters: Any) -> Any:
        return await self._request(
            self._auth_host, "/v1/portfolio/positions", filters, signed=True
        )

    async def activities(self, **filters: Any) -> Any:
        return await self._request(
            self._auth_host, "/v1/portfolio/activities", filters, signed=True
        )

    # -- public gateway reads ----------------------------------------------- #

    async def markets(self, **filters: Any) -> Any:
        """List markets. `categories` is the real filter - `category` is ignored.

        Unknown query parameters are silently dropped by the gateway and a
        default feed is returned instead of an error, so a typo looks like a
        genuine empty result. Verify a filter is honoured before trusting it.
        """
        return await self._request(self._gateway_host, "/v1/markets", filters, signed=False)

    async def market_book(self, slug: str) -> Any:
        return await self._request(
            self._gateway_host, f"/v1/markets/{slug}/book", None, signed=False
        )

    async def market_bbo(self, slug: str) -> Any:
        return await self._request(
            self._gateway_host, f"/v1/markets/{slug}/bbo", None, signed=False
        )

    async def series(self, **filters: Any) -> Any:
        return await self._request(self._gateway_host, "/v1/series", filters, signed=False)


def clock_skew_warning() -> str | None:
    """Local clock drift large enough to make every signature fail.

    Cheap to check and it turns a baffling 401 into an obvious diagnosis.
    """
    import datetime as _dt

    drift = abs(time.time() - _dt.datetime.now(_dt.timezone.utc).timestamp())
    if drift > MAX_CLOCK_SKEW_SECONDS:
        return f"local clock is {drift:.0f}s off UTC; signatures expire after 30s"
    return None


def summarize_balances(payload: Any) -> list[tuple[str, float]]:
    """Pull (currency, cash) pairs out of a balances response.

    Tolerant of shape: the response nests balances under a key whose name has
    moved before, so several are accepted rather than assuming one and
    reporting $0.00 for a funded account.
    """
    rows: Sequence[Any] = ()
    if isinstance(payload, dict):
        for key in ("balances", "accountBalances", "data", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                rows = value
                break
        else:
            rows = [payload]
    elif isinstance(payload, list):
        rows = payload

    out: list[tuple[str, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        currency = str(row.get("currency") or row.get("ccy") or "USD")
        for key in ("cashBalance", "cash", "balance", "amount", "fiatBalance"):
            if key in row:
                try:
                    out.append((currency, float(row[key])))
                except (TypeError, ValueError):
                    continue
                break
    return out
