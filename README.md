# BTC Spot ↔ Polymarket Short-Term Binary Dislocation Scanner

Detects and trades latency-arbitrage windows between Binance BTC spot and
Polymarket's rolling 5-minute / 15-minute **"Bitcoin Up or Down"** binaries.

**It runs dry by default** — simulating fills and logging expected PnL without
ever touching the order API. Live trading requires an explicit `--live` flag
*and* credentials.

```bash
pip install -r requirements.txt
python btc_polymarket_arb.py                  # dry run, no credentials needed
cp .env.example .env && $EDITOR .env          # then, to trade for real:
python btc_polymarket_arb.py --live --size 20 --max-notional 25
```

---

## The trade

Each contract resolves **Up** if the Chainlink BTC/USD TWAP over the tail of the
window is ≥ the reference price at the window open, else **Down**. Because the
settlement reference is a spot BTC price, the contract's fair value is a
deterministic function of `(spot, strike, vol, time remaining)`.

Binance is the fastest venue in the complex. The Polymarket CLOB is a
quoted order book that reprices on a lag. When spot dislocates hard over a
sub-3-second horizon, resting asks are briefly **stale** — they still reflect
the pre-spike distribution. That staleness window is the edge.

## Model

The scanner deliberately does **not** try to pin down the absolute strike. You
usually attach to a market mid-window and cannot observe the exact Chainlink
reference the resolver will use. Instead it runs a *relative* (delta-shift)
model, which is what the staleness trade actually needs:

1. **Anchor** on the last order book seen strictly *before* the spike began. Its
   mid `p₀` embeds the market's own consensus about moneyness:

   ```
   z₀ = Φ⁻¹(p₀)
   ```

2. **Shift.** The spike moves log-spot by `d = ln(S_now / S_pre)`. Under
   driftless GBM the standardized distance to the strike shifts by:

   ```
   z₁ = z₀ + d / (σ · √τ_eff)
   p_fair = Φ(z₁)
   ```

3. **Compare** `p_fair` against the *current* best ask. If the ask has not
   caught up, the window is open.

### TWAP variance haircut

The contract settles on the average price over the final `L` seconds, not the
terminal point, so `τ_eff` is **piecewise** in how much of the averaging window
is still ahead of us. `L` is read live from each market's
`cryptoMarketConfig.twapLookbackSeconds` (currently 60s).

**`τ ≥ L`** — the whole averaging window is in the future. A spot move of `d`
shifts the expected mean by the full `d`:

```
Var[mean] = σ²·(τ − L + L/3) = σ²·(τ − 2L/3)     ⟹   τ_eff = τ − 2L/3
```

**`τ < L`** — part of the window has *already been realized* and is locked in.
Only the remaining stub is random, and a spot move now shifts the expected mean
by only `d·(τ/L)`:

```
Var[mean] = σ²·τ³/(3L²)                          (Monte-Carlo verified)
z-shift   = d·(τ/L) / (σ·τ^1.5/(L√3))            ⟹   τ_eff = τ/3
```

The `L` cancels between numerator and denominator, and the two branches agree
exactly at `τ = L` (both give `L/3`), so the curve is continuous.

> **This one bites.** Naively extending the first branch below `τ = L` drives
> the haircut *negative*; clamping that to a floor pins fair value at ~1.0 in
> the closing seconds of every window and manufactures phantom edge on markets
> that are about to settle. It was caught in live dry-run testing, where a
> market with 10s left reported `tau_eff 1s` and a fabricated 98¢ edge.

Averaging damps terminal variance, so the contract is somewhat *more* sensitive
to a given spot move than a naive terminal model implies.

### Volatility

EWMA (120s half-life) of 1-second log returns over a 5-minute history. The
10-second buffer mandated for spike detection is far too short to estimate σ
from, so a separate longer return series is kept. σ is floored at 20% of a 45%
annualized prior so a dead tape cannot collapse the denominator and explode the
z-shift.

### Sanity check on the numbers

BTC realized vol ≈ 0.8 bps/√s, so a 4-minute window carries ≈ `0.8·√240` ≈ 12
bps of terminal σ. **A 12 bps move is therefore ~1σ of the remaining window** —
which is why it moves a coin-flip to ≈ 0.83, and to ≈ 0.97 with only 60s left.
The default threshold is calibrated, not arbitrary.

## Signals

| Signal | Meaning |
|---|---|
| `ARBITRAGE WINDOW OPEN` | Model edge: spot spiked, the corresponding ask is stale and below the `--max-yes-ask` gate. |
| `RISK-FREE CROSS-BOOK ARB` | Model-free: `ask(Up) + ask(Down) < $1.00`. Lift both legs for a locked profit at settlement, independent of any vol assumption. |

Both carry a per-market, per-side cooldown (default 5s).

## Market discovery

These series embed the window-open unix epoch in the slug
(`btc-updown-5m-1786517100`), so the live market is addressable directly with
one tiny Gamma query rather than paging the whole event list. If the convention
ever changes, the scanner falls back to a `series_slug` scan filtered
client-side.

> **Non-obvious detail:** Polymarket pre-creates these markets roughly **24
> hours** ahead of their trading window. Sorting by `startDate` descending
> returns markets pinned at 0.50/0.51 with no volume — *not* the live one.
> Selection is therefore always done on `eventStartTime <= now < endDate`.

Outcomes are `["Up", "Down"]` (not `["Yes", "No"]`); tokens are mapped **by
name**, so a reordering upstream cannot silently invert every signal.

> **Order book gotcha:** the CLOB returns **both** `bids` *and* `asks` sorted
> ascending by price, so the best ask is the *last* element. The code takes an
> explicit `min()`/`max()` to stay ordering-agnostic.

## Execution

### Credentials

Read from the environment or a `.env` file (see `.env.example`). `.env` is
gitignored — **never commit a private key**.

| Variable | Purpose |
|---|---|
| `POLYMARKET_PK` | Polygon wallet private key. Level 1 auth; required for `--live`. |
| `POLYMARKET_API_KEY` / `POLYMARKET_SECRET` / `POLYMARKET_PASSPHRASE` | Level 2 auth for posting orders. **Optional** — if absent, derived from `POLYMARKET_PK` at startup via `create_or_derive_api_creds()`. |
| `POLYMARKET_SIGNATURE_TYPE` | `0` = EOA (default), `1` = email/magic proxy, `2` = browser-wallet proxy. |
| `POLYMARKET_FUNDER` | Address holding the USDC. **Required** for signature types 1 and 2. |

Credentials are validated *before* any market data is consumed, so a
misconfigured wallet fails fast rather than at the moment an edge appears. Dry
runs report what live mode *would* reject, so the first live run is never the
first time you discover the wallet is wrong.

### Order semantics

Orders are **FOK marketable limits**: signed at the ask via `create_order`, then
posted with `OrderType.FOK`. FOK is the correct default here — the edge *is* a
stale resting ask, so the order must take it in full right now or die. A partial
fill or a resting remainder converts a latency arb into an unhedged directional
bet at exactly the moment the book is repricing against you.

`tick_size` and `neg_risk` are carried on `TrackedMarket` from the Gamma
payload and passed via `PartialCreateOrderOptions`, which avoids two extra
network round-trips per order on the latency-critical path. Prices are snapped
onto the venue tick grid (buys round up, sells round down) — the CLOB rejects
off-grid prices, and `0.45/0.01 == 44.99999...` in binary would otherwise bump a
correct price a full tick.

Every venue call goes through `asyncio.to_thread`: `py-clob-client` is
synchronous, and a blocking HTTP round-trip on the event loop would stall the
Binance tape, which is the one thing this system cannot afford.

### Safety gates

| Gate | Behavior |
|---|---|
| **Dry run** | Default. Simulates fills, books expected PnL, never builds a signing client. |
| **Position gate** | `active_positions` holds at most one open trade per market window. A spike stays above threshold for many consecutive 10 Hz ticks; without this the bot would re-bet the same spike dozens of times. Slots are freed when a market retires. |
| **Notional cap** | `--max-notional` blocks any leg above the cap outright, before signing. |
| **Circuit breaker** | Three consecutive execution errors halts submission for the rest of the session. |
| **Credential preflight** | `--live` aborts at startup on a missing key or a proxy signature type with no funder. |

### Leg risk on the cross-book trade

Both legs are submitted concurrently via `asyncio.gather` — sequencing them
would leave the second exposed to the book moving between round-trips, which is
exactly the risk the trade exists to avoid. If one leg fills and the other does
not, the hedge is gone and what remains is a naked directional position. That is
logged at **ERROR** with an explicit call for manual intervention rather than
buried, because it is the worst outcome of this trade and the bot does not
attempt to auto-unwind it.

## Configuration

```
--spike-bps       spot move threshold, basis points        (default 12 = 0.12%)
--spike-window    spike horizon, seconds                   (default 3)
--max-yes-ask     staleness gate: only flag asks below     (default 0.52)
--min-edge        minimum fair-value edge over the ask     (default 0.03)
--no-ask-ceiling  flag on model edge alone, ignore gate
--book-interval   book poll interval, seconds              (default 1.0)
--cooldown        per-signal cooldown, seconds             (default 5.0)
--ws-url          spot tape endpoint; repeat for failover
--verbose         debug logging

execution:
--live            ARM LIVE TRADING with real funds         (default: dry run)
--size            contracts per order leg                  (default 20)
--max-notional    hard USDC cap per leg                    (default 50)
--slippage-ticks  extra ticks above the ask to cross       (default 0)
--env-file        dotenv path                              (default .env)
```

## Spot tape failover

The canonical endpoint `wss://stream.binance.com:9443/ws/btcusdt@trade` is the
default. Binance geo-blocks some regions — the REST API returns **HTTP 451
"restricted location"** and the websocket resets the connection outright.

The scanner therefore rotates through a failover chain after two consecutive
failures:

1. `stream.binance.com:9443` — canonical
2. `data-stream.binance.vision` — Binance's public market-data mirror, carrying
   the byte-identical `btcusdt@trade` payload (verified: same schema, same
   prices, ~50ms latency)
3. `fstream.binance.com` — futures tape; a different contract that will
   basis-drift from spot, but usable for pure spike *detection*

Override with `--ws-url` (repeatable) to pin your own ordering.

## Architecture

Five cooperating `asyncio` tasks:

| Task | Cadence | Role |
|---|---|---|
| `binance` | event-driven | WS tape → rolling 10s buffer, auto-reconnect + failover |
| `discovery` | 20s | Gamma poll, market roll-over, retire expired windows |
| `books` | 1s | Batched CLOB book snapshots per tracked token |
| `eval` | 10 Hz | Spike detection → forced book refresh → signal → execution |
| `heartbeat` | 15s | Liveness: mode, spot, σ, WS state, latency, order stats, quotes |

`py-clob-client` is synchronous, so its calls are dispatched via
`asyncio.to_thread` to keep the event loop — and therefore the tape —
unblocked. On spike detection the engine forces a *fresh* book fetch
(rate-limited to 4/s) before judging: the entire question is whether the book
has already repriced, so acting on a stale local snapshot would manufacture
phantom edge.

## Tests

```bash
python test_signal_path.py
```

29 checks. Spins up a local websocket server speaking the Binance trade payload
format, replays a scripted spike against a stubbed stale book, and asserts each
gate independently.

**Signals** — UP spike + stale ask fires; DOWN spike fires on the NO leg; an
already-repriced book is suppressed by the edge gate; an ask above 0.52 is
suppressed by the ceiling gate; asks summing to 0.95 fire the cross-book arb; a
flat tape stays silent.

**Dry-run execution** — one simulated order per model signal, two legs per
cross-book signal, expected PnL booked correctly, and *no signing client is ever
constructed*.

**Position gate** — with the cooldown disabled so the gate is the only thing
that can suppress, a sustained spike produces 9 signal banners but exactly **1**
order, with all 8 duplicates logged as gate-suppressed. Releasing the slot
mid-run permits exactly one further entry. Retiring a market frees its slot.

**Safety** — notional cap blocks oversized legs; live mode refuses to arm
without a key, or with a proxy signature type and no funder; the circuit breaker
trips after three consecutive failures and rejects everything after; prices snap
onto the tick grid.

Beyond the suite, the execution path was driven end-to-end against **live
Polymarket books** by injecting a synthetic spike through `--ws-url` while
discovery, order books, tick sizes and `neg_risk` all came from the real venue.

## Caveats before trading this

- **Edges are gross.** Taker fees, gas, and slippage are not deducted. Check
  `takerBaseFee` / `feeSchedule` on each market. The `--min-edge` default of
  0.03 is a placeholder for cost, not a measured one.
- **"Expected PnL" is model PnL, not realized PnL.** It is `size × (fair − paid)`
  under this model's own fair value. It is not marked to settlement, and the
  bot does not track resolution outcomes. Treat it as a diagnostic, not a P&L
  statement.
- **No inventory or balance checks.** The bot does not verify USDC balance or
  allowance before submitting. A rejected order trips toward the circuit
  breaker rather than being pre-empted.
- **No unwind logic.** Positions are held to settlement. There is no stop, no
  exit, and no auto-hedge if a cross-book leg misses.
- **Displayed size is not fillable size.** The scanner reads top-of-book price
  only; it does not walk the book for depth at your clip.
- **Settlement is Chainlink, not Binance.** The scanner uses Binance as a fast
  proxy for the settlement reference. Basis between the two is a real risk that
  this model does not carry.
- **A stale ask may be a wide ask.** In thin books the resting ask can be stale
  because nobody is quoting, not because there is free money — you may be the
  liquidity, and adverse selection applies.
- The 0.52 gate is a heuristic from the original spec. `--no-ask-ceiling` runs
  on model edge alone, which is the more principled criterion.
