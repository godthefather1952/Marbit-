# BTC Spot ↔ Polymarket Short-Term Binary Dislocation Scanner

Detects and trades latency-arbitrage windows between Binance BTC spot and
Polymarket's rolling 5-minute / 15-minute **"Bitcoin Up or Down"** binaries.

**It runs dry by default** — simulating fills and logging expected PnL without
ever touching the order API. Live trading requires an explicit `--live` flag
*and* credentials.

## Quickstart (VS Code)

Requires **Python 3.10+** (3.11 recommended).

```bash
git clone <this repo> && cd Marbit-
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then open the folder in VS Code, press **F5**, and pick a configuration. The
one to start with is **"Kalshi monitor (live data, no orders)"** — it needs no
credentials at all and streams a real 15-minute BTC market immediately.

| Run configuration | Needs credentials? | What it does |
|---|---|---|
| **Kalshi monitor (live data, no orders)** | no | Live BTC 15-min market, fair value vs book, net of fees |
| Kalshi monitor (verbose) | no | Same, reports every edge including negative ones |
| Kalshi monitor (WRONG feed) | no | Reproduces the USDT-basis bug on purpose |
| Polymarket scanner (dry run) | no | The original scanner, simulated fills |
| Tests (165 checks) | no | Full suite, ~3 minutes |
| Check Kalshi account | yes | Verifies your key, shows balance |
| Check Polymarket US account | yes | Verifies your key, shows balance |
| Check a Polygon wallet | no | Balance from a public address |

Nothing in `.vscode/launch.json` can place an order.

### Two install tiers

`requirements.txt` covers Kalshi and everything shared — four packages, quick
and portable. The Polymarket global CLOB additionally needs the eth-account /
web3 stack, which is a slower and more fragile install, so it lives separately:

```bash
pip install -r requirements-polymarket.txt   # only to trade polymarket.com
```

The SDK is imported optionally. Without it the Kalshi tools, the model and the
risk engine all still run; only Polymarket execution raises, and it says
exactly what to install rather than failing with an ImportError at startup.

### If you want credentials

```bash
cp .env.example .env     # then fill in only the venue you use; .env is gitignored
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

`tick_size` and `neg_risk` are carried on `TrackedMarket` and passed via a
cached `PartialCreateOrderOptions`.

> Passing those options does **not**, on its own, keep `create_order` off the
> network. It tests `if options and options.neg_risk` — a truthiness check — so
> a legitimately `False` neg-risk flag falls straight through to a blocking
> `get_neg_risk()` GET, and `get_tick_size()` is called unconditionally to
> validate. The only thing that actually removes those round-trips is warming
> the client's internal caches at discovery. See [Latency](#latency).

Prices are snapped onto the venue tick grid (buys round up, sells round down) —
the CLOB rejects off-grid prices, and `0.45/0.01 == 44.99999...` in binary would
otherwise bump a correct price a full tick.

Every venue call goes through `asyncio.to_thread`: `py-clob-client` is
synchronous, and a blocking HTTP round-trip on the event loop would stall the
Binance tape, which is the one thing this system cannot afford.

### Safety gates

| Gate | Behavior |
|---|---|
| **Dry run** | Default. Simulates fills, books expected PnL, never builds a signing client. |
| **Position gate** | `active_positions` holds at most one open trade per market window. A spike stays above threshold for many consecutive 10 Hz ticks; without this the bot would re-bet the same spike dozens of times. Slots are freed when a market retires. |
| **Notional cap** | `--max-notional` blocks any leg above the cap outright, before signing. |
| **Balance pre-flight** | Every order reserves its notional against free funds before signing. Insufficient funds → refused, never sent. |
| **Drawdown breaker** | Peak-to-trough session drawdown past the limit pauses new entries; repeated trips halt the session. |
| **Consecutive losses** | Three losing closes or failed executions in a row halts the session permanently. |
| **Execution breaker** | Three consecutive execution *errors* halts submission for the rest of the session. |
| **Credential preflight** | `--live` aborts at startup on a missing key, a proxy signature type with no funder, or an unreadable balance. |

## Risk management

### Bankroll and dynamic sizing

On startup in live mode the bot reads the wallet's USDC balance and its
allowance to the CTF Exchange, and **fails closed** — no balance, no trading.
Reading through the CLOB's balance/allowance endpoint rather than a direct RPC
keeps the number aligned with the venue's own view (it is what actually decides
whether an order is accepted) and avoids a web3 dependency plus RPC config. A
zero allowance aborts startup, since every order would be rejected on-chain.

Position size comes from the risk budget, not a fixed count:

```
budget = equity × max_risk_pct        capped by --max-notional and free funds
size   = budget / price               floored at the venue's orderMinSize
```

For a binary the entire premium is at risk, so `risk == size × price`. Sizing
off **current equity** rather than the opening balance means risk per trade
scales down as a session loses ground; realized losses also reduce buying power
directly. A `$500` bankroll at 2% buys exactly `$10` of premium whether the ask
is `0.14` or `0.50`.

Cross-book pairs are sized off the *combined* leg cost, so a hedged pair is one
risk unit rather than two.

### Capital reservation

Orders reserve their notional before signing and hold it until
`risk.open_position()` converts it into position cost. The handoff is atomic on
purpose: releasing at submission time would leave a window where spent capital
looks free, and the cross-book path submits both legs concurrently — without a
reservation each leg would independently see the full free balance and together
commit more than the wallet holds.

### Session breakers

Equity is `bankroll + realized + unrealized`, with open positions marked to the
**best bid** — the price you could actually exit at. Marking to the mid would
flatter the book and let a real drawdown hide.

> When a leg has no bid at all, it marks off the complement of the opposite ask
> (`1 − ask`), since the pair must sum to $1. Without this a position that has
> gone worthless keeps its entry mark and stays invisible to the breaker right
> up until settlement — exactly when the breaker most needs to see it.

Drawdown is measured **peak-to-trough**, and the *tighter* of `--max-drawdown-pct`
(of peak equity) and `--max-drawdown-usd` binds. A trip pauses new entries for
`--risk-cooldown`, then re-baselines the peak and resumes; `--max-drawdown-trips`
caps how many times that can happen before the session halts for good.

> Re-baselining means each cooldown permits another full drawdown, so total
> session loss can reach roughly `trips × limit`. That is why trips are counted
> and capped — set `--max-drawdown-trips 1` for a hard single-strike stop.

A tripped breaker suppresses **signal generation**, not just order submission:
logging tradeable edges the bot has no intention of taking trains the operator
to ignore the banner.

PnL is realized when a market retires, using the last mark. By expiry the book
has converged to ~0 or ~1, so the mark is a close proxy — but it is a proxy, not
settlement-confirmed accounting.

## Kalshi — the venue that actually has this instrument

Kalshi is CFTC-regulated (so US-accessible) and lists **`KXBTC15M` — "BTC price
up in next 15 mins?"**, structurally the same contract as Polymarket's
`btc-updown-15m`. Verified live: 1c spreads, ~870k contracts traded in a single
15-minute window, an order of magnitude deeper than the Polymarket equivalent.

Settlement, verbatim from their API:

> *"If the simple average of the sixty seconds of CF Benchmarks' BRTI before
> HH:15 is at least the simple average of the sixty seconds of BRTI before
> HH:00, then the market resolves to Yes."*

Same 60-second TWAP structure, so the piecewise `effective_tau` haircut derived
for Polymarket transfers unchanged. And Kalshi **publishes the strike**
(`floor_strike`), so fair value is computed outright instead of via the
strike-free delta-shift workaround Polymarket forced.

```bash
python kalshi_monitor.py                 # read-only, no orders, no credentials
python kalshi_monitor.py --min-edge 0.02
```

### The two things that decide viability

**1. Fees are not a rounding error.** Kalshi charges `0.07 x contracts x P x
(1-P)`, rounded up to the cent — quadratic, so it *peaks at 50c*, which is
exactly where these markets open.

| ask | fee/contract | share of a 3c gross edge |
|---|---:|---:|
| $0.05 | 0.33c | 11% |
| $0.20 | 1.12c | 37% |
| **$0.50** | **1.75c** | **58%** |

Every edge the monitor prints is net of this. Polymarket was effectively free;
here the trade is much better at the extremes than near the money.

**2. The reference feed must be USD-quoted, not USDT.**

> This one produced a phantom 29-cent edge that persisted for minutes on a
> market with 870k volume — which is how it was caught. Settlement is BRTI, a
> **USD** index. Binance quotes **USDT**, and USDT/USD routinely drifts 5-15
> bps from parity. Measured live: Binance $63,841.94 against a USD composite of
> $63,766.69, a **+11.8 bps premium** — the same order of magnitude as an
> entire 15-minute BTC move, and a *biased* error rather than noise that
> averages out.

With Binance the model said `fair 0.617` while the market quoted `0.31/0.32`.
Switching to Coinbase BTC-USD (a BRTI constituent) it tracks within a cent:

```
spot=$63,772 strike=$63,777 | fair=0.478 | market 0.470/0.480
spot=$63,766 strike=$63,777 | fair=0.453 | market 0.470/0.480
```

`kalshi_monitor.py --binance` reproduces the error deliberately.

### Read-only, enforced

`kalshi.py` has no order-placing method and issues zero POST/PUT/DELETE
requests, both asserted by the suite. Auth is RSA-PSS SHA256 over
`{timestamp}{METHOD}{path}` with the query string **excluded** — differing from
Polymarket US, which includes it.

## Two different Polymarket exchanges

`polymarket.com` and `polymarket.us` are **separate venues** with separate
accounts, credentials and markets. Getting this wrong wastes a lot of time, so:

| | polymarket.com (global) | polymarket.us |
|---|---|---|
| Auth | EIP-712 orders signed by a Polygon wallet | Ed25519 request signatures |
| Credential | `POLYMARKET_PK` (wallet private key) | Key ID + secret from the developer portal |
| Money | USDC on Polygon | Fiat balance |
| Short-dated BTC | **Yes** — rolling 5m/15m Up/Down | **No** — shortest crypto horizon is ~525 h |
| Traded by | `btc_polymarket_arb.py` | `polymarket_us.py` (read-only) |

**This bot's strategy only exists on polymarket.com.** Polymarket US has 51 BTC
markets, but they are 2026-expiry price milestones ("When will Bitcoin hit
$150k?"), not sub-minute binaries. There is no latency edge on a market that
expires next year, so the scanner cannot be pointed at Polymarket US and work.

### Read-only Polymarket US adapter

`polymarket_us.py` covers the US venue for balances, positions, markets and
books. It **cannot place, modify or cancel an order** — there is no such method
and the module issues zero POST/PUT/DELETE requests, both asserted by the test
suite. The first thing you point at a funded account should not be able to
spend it.

```bash
export POLYMARKET_US_KEY_ID=...
export POLYMARKET_US_SECRET_KEY=...
python check_us_account.py
```

It checks credentials, clock skew (signatures expire after 30 s), the public
gateway, then your balance and positions — in that order, so a failure tells you
which layer broke.

> **Gateway gotcha:** `gateway.polymarket.us` silently drops unknown query
> parameters and returns a default sports feed instead of erroring. Filtering by
> `category` (singular) or `seriesSlug` on `/v1/markets` looks like a genuine
> empty result. The real parameter is `categories`. Always confirm a filter is
> honoured before trusting a zero.

## Checking your wallet

Balances and approvals are public on-chain data, so verifying them needs **only
your address** — no API key, no secret, no private key:

```bash
python check_wallet.py 0xYourWalletAddress
python check_wallet.py 0xYour... --rpc https://polygon-mainnet.g.alchemy.com/v2/KEY
```

It reports USDC.e balance and the allowance to both exchange contracts, and
flags the two setups that silently break live trading: funds sitting in the
proxy wallet rather than the signing wallet, and a zero allowance.

> **The CLOB does not use `X-PM-*` API keys.** Polymarket's newer developer
> keys (`X-PM-Access-Key` / `X-PM-Signature`, Ed25519) are a different product
> from the CLOB trading API, which authenticates with HMAC-SHA256 over
> `POLY_ADDRESS` / `POLY_API_KEY` / `POLY_PASSPHRASE` / `POLY_SIGNATURE`. The
> CLOB triplet also includes a **passphrase**, which the Ed25519 keys do not
> issue. They are not interchangeable, and neither replaces `POLYMARKET_PK`:
> orders are EIP-712 signed by the wallet key itself, so no API credential of
> any kind can place a trade.

## Latency

### Where the time actually goes

Measured, not assumed. The controllable pipeline — parse, buffer, spike scan,
fair value, sizing, risk gates, payload construction — is **0.15 ms**. Everything
above that in a live run is round-trip time.

| Stage | Before | After | Note |
|---|---:|---:|---|
| Order metadata on a new market | **1035 ms** | **~0 ms** | pre-warmed at discovery |
| EIP-712 order signing | 7.44 ms | **0.83 ms** | `coincurve` backend |
| Tick deserialization | 2.47 µs | **0.91 µs** | `orjson` |
| Spike scan (2k ticks) | 355 µs | **147 µs** | min/max instead of per-tick `log()` |
| Eval scheduling delay | ≤100 ms | **~0 ms** | event-driven wake |
| **T1→T2, network removed** | — | **0.15 ms** | full local pipeline |

> **The big one.** `create_order` internally calls `get_tick_size`,
> `get_neg_risk` and `get_fee_rate_bps`, each of which is a **blocking HTTP GET
> on a token it has not seen before**. These markets roll every five minutes
> with brand-new token ids, so without pre-warming, the first order in every
> window — the one the whole system exists to send quickly — paid **1,035 ms**
> of round-trips before signing even began. `prepare_market()` now warms all
> three at discovery, off the hot path.

> **Install `coincurve`.** Without it `eth-keys` falls back to a pure-Python
> elliptic curve implementation and every order costs 7.4 ms to sign instead of
> 0.8 ms. The startup banner reports which backend is live; it prints
> `SLOW - pip install coincurve` if you are on the fallback.

### Reaching <100 ms end-to-end

Local compute is 0.15 ms, so the budget is **entirely network**:

```
T1→T2 = 0.15 ms local + one book-confirmation round-trip
T2→T3 = 0.83 ms signing + one order POST round-trip
```

The book confirmation is not removable without changing the design: the whole
claim is that the resting ask is stale, and you cannot assert that without
looking at the book *after* the spike. So end-to-end is roughly `2 × RTT`, and
<100 ms requires **RTT under ~45 ms** — routine from a well-peered host near
Polymarket's infrastructure, impossible from a distant one no matter how fast
the code is. On the sandbox these numbers were developed on, RTT is ~160 ms
(proxied and geographically distant), giving T1→T2 ≈ 167 ms of which 0.15 ms is
ours.

The next structural win, if you need it, is the CLOB market websocket
(`wss://ws-subscriptions-clob.polymarket.com/ws/market`): a streamed book is
always fresh, which removes the confirmation round-trip from the hot path
entirely and halves the budget. That is not implemented here.

### Instrumentation

Three checkpoints, `time.perf_counter_ns()` throughout — monotonic and immune to
wall-clock adjustment, which `time.time()` is not:

| | Checkpoint |
|---|---|
| **T1** | Binance websocket frame received (stamped before parsing, so deserialization is counted, not excluded) |
| **T2** | Signal evaluated, payload constructed, funds reserved |
| **T3** | Order response received from the CLOB |

Recording costs ~196 ns into a bounded deque. The heartbeat reports medians and
p95:

```
lat[n=2] T1->T2 166.65ms T2->T3 0.0ms p95 166.7ms
```

### Event-driven evaluation

The evaluator no longer polls on a fixed timer. Each tick is compared against
two pre-armed price levels — `low × e^(+thr)` and `high × e^(-thr)`, recomputed
after each pass — so detecting a candidate spike costs **two float comparisons
per tick, no logarithm and no scan**. Crossing one wakes the evaluator
immediately, removing up to a full polling interval of dead time. The timer
remains as a floor for housekeeping.

The tripwire is a *hint*, never the decision: `evaluate()` always recomputes the
real move, so a false wake costs one wasted pass and a missed wake is bounded by
the periodic timer.

### Connection reuse

One pooled `aiohttp.ClientSession` serves every REST call this process makes
(Gamma discovery and the private Polygon RPC), with keepalive, DNS caching and
explicit timeouts. CLOB traffic goes through py-clob-client's shared
`httpx.Client`, which already pools over HTTP/2; the bot retunes it at startup
with a trading-appropriate timeout budget, since the stock blanket 5 s timeout
is far too long for an order that is only valuable for a few hundred
milliseconds.

### Private RPC

`POLYGON_RPC_URL` (Alchemy / QuickNode / Chainstack) switches balance and
allowance reads to direct `eth_call`s against USDC on Polygon — faster than the
Polymarket API and independent of its uptime. Implemented as two raw ERC-20
selectors over the shared session rather than a web3 dependency; falls back to
the CLOB endpoint if the RPC is absent or fails.

`POLYGON_WS_URL` is accepted, validated and logged, but the trading path is
REST-only and does not use it. It is plumbed through for on-chain
subscriptions.

## Resilience

### Retry and backoff

Idempotent reads — Gamma discovery, CLOB book fetches, balance queries — retry
on `408/425/429/500/502/503/504` and on connection-level faults, with
exponential backoff, full jitter, and a cap. A server-sent `Retry-After` always
wins over the computed delay. A 4xx that is not rate limiting fails immediately
rather than burning attempts on something backoff cannot fix.

> **Order submission is deliberately excluded.** A POST that times out may still
> have executed, so retrying risks a double fill — far worse than a missed
> trade. Only reads retry.

### Tape reconnects

The websocket reconnects with exponential backoff and endpoint failover, and on
every disconnect it **resets the price buffer** and re-enters a warm-up equal to
the spike lookback.

> Without the reset, ticks captured before the drop stay in the 10s buffer and
> get compared against the first tick after reconnect. Any drift during the
> outage reads as an instantaneous spike — firing a signal, and now a live
> order, against a book that had every opportunity to reprice.

The reset is scoped to market data only. `active_positions`, open positions,
session PnL and the breakers live on the Executor and RiskManager, so a
reconnect never disturbs risk state. The volatility estimator separately skips
any 1-second return spanning a gap, so a stalled tape cannot inflate σ.

### Live venue metadata

Tick size, minimum order size and `neg_risk` are taken from the order book
response and override the Gamma snapshot. Polymarket tightens the tick for
extreme prices near expiry, and quantizing a `0.001` ask onto a stale `0.01`
grid would price an order an order of magnitude away from the liquidity being
taken.

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
--size            contracts per leg when sizing is fixed   (default 20)
--max-notional    hard USDC cap per leg                    (default 50)
--slippage-ticks  extra ticks above the ask to cross       (default 0)
--env-file        dotenv path                              (default .env)

risk management:
--max-risk-pct         percent of equity risked per trade  (default 2.0)
--fixed-size           disable dynamic sizing, use --size
--max-drawdown-pct     percent drawdown that trips         (default 5.0)
--max-drawdown-usd     absolute USDC drawdown that trips   (default 100)
--max-consecutive-losses  losing/failed trades to halt     (default 3)
--risk-cooldown        pause after a trip, seconds         (default 300)
--max-drawdown-trips   trips before permanent halt         (default 3)
--paper-bankroll       simulated bankroll for dry runs     (default 1000)
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
| `binance` | event-driven | WS tape → rolling 10s buffer, tripwire, auto-reconnect + failover |
| `discovery` | 20s | Gamma poll, market roll-over, retire expired windows |
| `books` | 1s | Batched CLOB book snapshots per tracked token |
| `eval` | tripwire-woken, 10 Hz floor | Spike detection → book confirmation → signal → execution |
| `heartbeat` | 15s | Liveness: mode, spot, σ, WS state, T1→T2/T2→T3 latency, order stats, risk, quotes |

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

120 checks. Spins up a local websocket server speaking the Binance trade payload
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

**Balance & sizing** — USDC parses across every response shape the CLOB returns
(raw 6-decimal, plain decimal, nested allowances map); 2% of $1000 sizes to
exactly $20 of premium; the notional cap and venue minimum both bind correctly;
realized losses shrink buying power and the risk budget; the allowance caps
available funds; an unfundable order is refused before signing; and two
concurrent legs cannot over-commit the same capital.

**Risk breakers** — drawdown inside the limit does not trip; past it enters a
timed cooldown and re-baselines the peak; exhausting the trip budget halts
permanently; three failed executions halt; a losing close increments the streak
and a winning close resets it; a halted session emits **zero** signal banners
and announces the suppression once rather than per tick.

**Retry & rate limits** — 429/503 retryable, 400 not; `PolyApiException` status
classified correctly; backoff recovers through 429s; a non-retryable status
fails without burning attempts; retries are bounded; `Retry-After` overrides the
computed delay.

**Latency** — a JSON backend is selected at import and parses both str and
bytes frames; malformed input raises `ValueError` on all three backends and a
corrupt frame never kills the tape; templates stay isolated per market and a
tightened live tick rebuilds the cached options; retiring a market evicts only
its own templates; the profiler reports T1→T2/T2→T3 in ms, bounds its sample
window, and records in ~196 ns; the tripwire wakes on a qualifying tick,
debounces until re-armed, agrees with the authoritative scan, and survives an
unreachable threshold; the min/max spike scan returns the same answer as the
per-tick search it replaced; and **T1→T2 with the network removed is 0.15 ms**.

**Resilience** — a reset drops the stale tick window and blocks signals during
warm-up; `largest_move` cannot see across a gap; a dropped tape resets the
buffer while `active_positions`, open positions and PnL survive untouched; a
bidless leg marks off the opposite ask so the loss reaches the breaker; live
book tick size overrides the Gamma snapshot; and the vol estimator ignores
returns spanning a gap.

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
- **No unwind logic.** Positions are held to settlement. The breakers stop the
  bot from opening *new* risk; they do not close existing positions. There is no
  stop-loss exit and no auto-hedge if a cross-book leg misses.
- **Realized PnL is mark-based, not settlement-confirmed.** Positions are closed
  at their last observed mark when a window retires, not against the Chainlink
  resolution. Close by construction, but not an accounting record.
- **Balance is read once at startup.** It is not re-polled, so fills, deposits
  or withdrawals during a session are not reflected until restart. Session
  accounting tracks the delta from that opening snapshot.
- **Depth is still not checked.** Sizing respects your bankroll, not the book.
  A computed size can exceed the resting liquidity at the ask, in which case
  the FOK simply fails rather than filling.
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
