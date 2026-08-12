# BTC Spot ↔ Polymarket Short-Term Binary Dislocation Scanner

Detects latency-arbitrage windows between Binance BTC spot and Polymarket's
rolling 5-minute / 15-minute **"Bitcoin Up or Down"** binaries.

The scanner is **read-only** — it discovers markets, prices them, and logs
signals. It places no orders and needs no API credentials.

```bash
pip install -r requirements.txt
python btc_polymarket_arb.py
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
terminal point. For a Brownian path the variance of that average, seen from now:

```
Var = σ²·(τ − L + L/3) = σ²·(τ − 2L/3)        ⟹   τ_eff = τ − 2L/3
```

`L` is read live from each market's `cryptoMarketConfig.twapLookbackSeconds`
(currently 60s). Averaging damps terminal variance, so the contract is slightly
*more* sensitive to a given spot move than a naive terminal model implies.

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
| `eval` | 10 Hz | Spike detection → forced book refresh → signal |
| `heartbeat` | 15s | Liveness: spot, σ, WS state, latency, live quotes |

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

Spins up a local websocket server speaking the Binance trade payload format,
replays a scripted spike against a stubbed stale book, and asserts each gate
independently:

- UP spike + stale ask → fires
- DOWN spike + stale NO ask → fires on the NO leg
- book already repriced → edge gate suppresses
- ask above the 0.52 ceiling → ceiling gate suppresses
- asks summing to 0.95 → cross-book arb fires
- flat tape → silent

## Caveats before trading this

- **Edges are gross.** Taker fees, gas, and slippage are not deducted. Check
  `takerBaseFee` / `feeSchedule` on each market.
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
