#!/usr/bin/env python3
"""The one script: a staged autopilot for a full Kalshi session.

    python kalshi_main.py                        # interactive
    python kalshi_main.py --mode live --warmup-min 20
    python kalshi_main.py --mode dry -- --min-edge 0.03   # extra monitor flags

One run walks through the whole session so nothing has to be remembered:

    PREFLIGHT   exchange up, live market found, spot feed answering. In live
                mode, credentials are prompted for and proven with a signed
                balance request before anything else starts.

    WARM-UP     paper-only monitoring for the chosen time frame, while the
                volatility estimator builds from the tape. Orders cannot be
                sent in this phase no matter what the signals say.

    PROMOTION   live mode arms real execution ONLY if the warm-up passed the
                quality gates - volatility in agreement with the market's own
                quote, feed healthy, market live. A failed gate keeps the
                session on paper and says exactly why, then re-checks
                periodically: the time frame is a proving period, not a timer.

    SESSION     trades (live) or records (paper) until Ctrl+C or
                --session-min elapses. All risk limits apply.

    SCORECARD   on shutdown, waits for the markets that were traded to settle
                and grades every trade against the real outcome - the only
                number that has ever meant anything in this project. A second
                Ctrl+C skips the wait (score later with kalshi_score.py).

Every phase, signal, order, gate report and the final scorecard land in one
log file (logs/L_MMDDYY_HHMMSS.log), named at start so it can be read - or
sent - while the run is still going.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import signal as _signal
import statistics
import sys
import time
from pathlib import Path

import aiohttp

from btc_polymarket_arb import log
from kalshi import KalshiClient
from kalshi_monitor import Monitor, parse_args as monitor_parse_args
from kalshi_score import score as score_trades, settlement
from run_log import start_run_log

GATE_RECHECK_SECONDS = 300.0
SETTLE_POLL_SECONDS = 30.0


# --------------------------------------------------------------------------- #
# Startup prompts
# --------------------------------------------------------------------------- #


def prompt_mode(preset: str | None) -> str:
    if preset in ("dry", "live"):
        return preset
    if not sys.stdin.isatty():
        print("No terminal to prompt on; defaulting to a DRY run.")
        return "dry"
    print(
        "\nHow should this session run?\n"
        "  1) Dry run - paper trades only, no credentials needed (default)\n"
        "  2) LIVE    - real orders with real money, after warm-up gates pass\n"
    )
    while True:
        raw = input("Mode [1/2]: ").strip()
        if raw in ("", "1", "dry"):
            return "dry"
        if raw in ("2", "live"):
            confirm = input("Type LIVE to confirm real-money trading: ").strip()
            if confirm == "LIVE":
                return "live"
            print("  Not confirmed; staying on the menu.")
        else:
            print("  1 or 2.")


def prompt_warmup(preset: float | None, mode: str) -> float:
    """Minutes of paper-only monitoring before execution can arm.

    Floored at 6: the volatility estimator needs ~5 minutes of tape before it
    is a measurement rather than an assumption, and promoting on an assumption
    is how phantom edges get funded.
    """
    if preset is not None:
        return max(preset, 6.0)
    default = 20.0
    if not sys.stdin.isatty() or mode == "dry":
        return default
    raw = input(f"Warm-up minutes before execution can arm [default {default:.0f}]: ").strip()
    try:
        return max(float(raw), 6.0) if raw else default
    except ValueError:
        print(f"  Not a number; using {default:.0f}.")
        return default


# --------------------------------------------------------------------------- #
# Quality gates - what "warmed up" actually means
# --------------------------------------------------------------------------- #


def evaluate_gates(monitor: Monitor, vol_ratio_max: float) -> tuple[bool, list[str]]:
    """Decide whether the warm-up proved enough to trade real money on.

    Time passing is not evidence. These are the conditions under which the
    session's signals are distinguishable from the failure modes this project
    has actually hit: a dead feed, an unmeasured volatility prior, and a sigma
    estimate the market itself contradicts.

    With several instruments running, EVERY instrument must pass. They share
    one account and one set of risk limits, so a broken ETH feed can spend the
    balance that BTC was supposed to trade - there is no such thing as going
    live on half the book.
    """
    lines: list[str] = []
    passed = True
    for inst in monitor.instruments:
        ok, block = _instrument_gates(inst, vol_ratio_max)
        passed = passed and ok
        if len(monitor.instruments) > 1:
            lines.append(f"   {inst.name} ({inst.series}):")
            lines.extend("   " + line for line in block)
        else:
            lines.extend(block)
    lines.append(
        "   => ALL GATES PASSED" if passed else "   => NOT READY - staying on paper"
    )
    return passed, lines


def _instrument_gates(inst, vol_ratio_max: float) -> tuple[bool, list[str]]:
    checks: list[tuple[str, bool, str]] = []

    stream_ok = inst.stream is not None and inst.stream.connected
    checks.append(("spot feed connected", stream_ok, inst.asset.coinbase_product))

    basis = inst.basis
    basis_ok = basis is None or basis.samples >= 3
    checks.append((
        "USD basis correction settled",
        basis_ok,
        f"{basis.samples} polls, {basis.offset:+.2f} USD" if basis else "disabled",
    ))

    market_ok = inst.market is not None and inst.market.strike_known
    checks.append((
        "live market with a published strike",
        market_ok,
        inst.market.ticker if inst.market else "none",
    ))

    checks.append((
        "volatility measured from tape (not the assumed prior)",
        inst.buffer.vol_is_measured, "",
    ))

    obs_ok = inst.observations >= 100
    checks.append(("enough evaluated observations", obs_ok, f"{inst.observations}"))

    ratios = inst.vol_ratios[-200:]
    invertible_ok = len(ratios) >= 30
    checks.append((
        "market quote invertible often enough to validate sigma",
        invertible_ok,
        f"{len(ratios)} recent samples",
    ))

    if invertible_ok:
        med = statistics.median(ratios)
        agree_ok = med <= vol_ratio_max
        detail = f"median {med:.2f}x vs limit {vol_ratio_max:.2f}x"
    else:
        # Unvalidatable is not the same as validated. Fail closed.
        agree_ok, detail = False, "cannot check without invertible quotes"
    checks.append(("our sigma agrees with the market's", agree_ok, detail))

    passed = all(ok for _, ok, _ in checks)
    lines = [f"   [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({d})" if d else "")
             for name, ok, d in checks]
    return passed, lines


# --------------------------------------------------------------------------- #
# Scorecard - grade the session against reality before letting go
# --------------------------------------------------------------------------- #


async def settle_and_score(
    ledger_path: Path, skip: asyncio.Event, wait_minutes: float
) -> list[str]:
    """Wait for traded markets to settle, then grade every recorded trade."""
    rows = []
    if ledger_path.is_file():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            with contextlib.suppress(json.JSONDecodeError):
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        return ["scorecard: no trades were recorded this session"]

    tickers = sorted({r["ticker"] for r in rows})
    settled: dict[str, dict] = {}
    deadline = time.monotonic() + wait_minutes * 60.0

    async with aiohttp.ClientSession(
        headers={"User-Agent": "kalshi-main/1.0"}
    ) as session:
        client = KalshiClient(session)
        while not skip.is_set():
            try:
                settled = await settlement(client, tickers)
            except Exception as exc:  # noqa: BLE001
                log.warning("Settlement poll failed: %s", exc)
            missing = [t for t in tickers if t not in settled]
            if not missing:
                break
            if time.monotonic() >= deadline:
                log.warning(
                    "Gave up waiting on %d market(s); score later with kalshi_score.py",
                    len(missing),
                )
                break
            log.info(
                "Waiting for %d market(s) to settle (%s)... re-checking in %.0fs, "
                "Ctrl+C skips",
                len(missing), ", ".join(missing[:3]), SETTLE_POLL_SECONDS,
            )
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(skip.wait(), timeout=SETTLE_POLL_SECONDS)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        score_trades(rows, settled)
    text = buffer.getvalue()
    print(text)
    return ["SCORECARD (actual settled outcomes)"] + text.splitlines()


# --------------------------------------------------------------------------- #
# The autopilot
# --------------------------------------------------------------------------- #


class _StopControl:
    """First Ctrl+C ends the session; the second skips the settlement wait."""

    def __init__(self) -> None:
        self.stop_session = asyncio.Event()
        self.skip_scorecard = asyncio.Event()

    def on_signal(self) -> None:
        if not self.stop_session.is_set():
            log.warning("Stopping the session... (Ctrl+C again to skip the scorecard)")
            self.stop_session.set()
        else:
            self.skip_scorecard.set()


async def _sleep_unless(stop: asyncio.Event, seconds: float, runner: asyncio.Task) -> None:
    """Wait out `seconds`, returning early if the session stops or the monitor dies."""
    waiter = asyncio.create_task(stop.wait())
    done, _ = await asyncio.wait(
        {waiter, runner}, timeout=seconds, return_when=asyncio.FIRST_COMPLETED
    )
    waiter.cancel()
    if runner in done:
        runner.result()  # surface the monitor's exception instead of hanging


async def autopilot(args: argparse.Namespace, margs: argparse.Namespace,
                    monitor: Monitor, control: _StopControl) -> list[str]:
    """Run the phases. Returns extra lines for the session summary."""
    notes: list[str] = [f"mode          : {args.mode.upper()}"]
    runner = asyncio.create_task(monitor.run())
    stop = control.stop_session

    # -- warm-up ------------------------------------------------------------- #
    warm_s = args.warmup_min * 60.0
    log.warning(
        "PHASE: WARM-UP | paper only for %.0f minutes while volatility is "
        "measured from the tape", args.warmup_min,
    )
    end = time.monotonic() + warm_s
    while time.monotonic() < end and not stop.is_set() and not runner.done():
        await _sleep_unless(stop, min(60.0, end - time.monotonic()), runner)

    # -- promotion ----------------------------------------------------------- #
    promoted = False
    if args.mode == "live":
        while not stop.is_set() and not runner.done():
            passed, report = evaluate_gates(monitor, margs.vol_ratio_max)
            log.warning("PHASE: PROMOTION | quality gates:\n%s", "\n".join(report))
            if passed:
                if await monitor.trader.go_live():
                    monitor.reset_for_live()
                    promoted = True
                    log.warning(
                        "PHASE: SESSION | LIVE - real orders enabled, balance "
                        "$%.2f, max stake $%.2f/trade",
                        monitor.trader.starting_balance, monitor.trader.max_stake(),
                    )
                else:
                    log.error("Arming failed; continuing on paper")
                break
            log.warning(
                "Staying on paper; gates re-check in %.0f minutes",
                GATE_RECHECK_SECONDS / 60.0,
            )
            await _sleep_unless(stop, GATE_RECHECK_SECONDS, runner)
        notes.append(f"promotion     : {'LIVE' if promoted else 'stayed on paper'}")
    else:
        log.warning("PHASE: SESSION | dry run continues on paper")

    # -- session ------------------------------------------------------------- #
    if args.session_min > 0:
        session_end = time.monotonic() + args.session_min * 60.0
        while time.monotonic() < session_end and not stop.is_set() and not runner.done():
            await _sleep_unless(stop, min(60.0, session_end - time.monotonic()), runner)
        if not stop.is_set():
            log.warning("Session length reached (%.0f min); stopping", args.session_min)
    else:
        while not stop.is_set() and not runner.done():
            await _sleep_unless(stop, 3600.0, runner)

    # A final gate report in the summary, whatever mode ran: it is the honest
    # description of whether this session's signals were worth anything.
    _, final_report = evaluate_gates(monitor, margs.vol_ratio_max)
    notes.append("final gate state:")
    notes.extend(line.strip() for line in final_report)

    runner.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await runner

    # -- scorecard ----------------------------------------------------------- #
    if monitor.ledger is not None:
        log.warning("PHASE: SCORECARD | grading recorded trades against settlement")
        notes.append("")
        notes.extend(
            await settle_and_score(
                monitor.ledger.path, control.skip_scorecard, args.settle_wait_min
            )
        )
    return notes


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #


def parse_main_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=["dry", "live"],
                   help="skip the mode prompt")
    p.add_argument("--warmup-min", type=float, default=None,
                   help="minutes of paper-only warm-up before execution can arm "
                        "(min 6; the vol estimator needs ~5 minutes of tape)")
    p.add_argument("--session-min", type=float, default=0.0,
                   help="stop after this many minutes; 0 = run until Ctrl+C")
    p.add_argument("--settle-wait-min", type=float, default=25.0,
                   help="how long the scorecard waits for markets to settle")
    p.add_argument("--assets", default=None,
                   help="comma-separated underlyings to trade concurrently, e.g. "
                        "'BTC,ETH'. Each gets its own spot feed; they share one "
                        "account, so risk limits apply across all of them.")
    p.add_argument("--aggressive", action="store_true",
                   help="loosen the opportunity thresholds (see kalshi_monitor.py "
                        "--aggressive). The correctness gates are unchanged.")
    p.add_argument("--env-file", default=".env")
    known, extra = p.parse_known_args()
    # Forward the shared flags to the monitor's parser too.
    if known.assets:
        extra += ["--assets", known.assets]
    if known.aggressive:
        extra += ["--aggressive"]
    # Anything after `--` (or any unrecognised monitor flag) passes through to
    # the monitor's own parser, so one script exposes every knob.
    return known, [a for a in extra if a != "--"]


def main() -> int:
    args, extra = parse_main_args()
    args.mode = prompt_mode(args.mode)
    args.warmup_min = prompt_warmup(args.warmup_min, args.mode)

    if args.mode == "live":
        # Credentials before anything else: prompted for if missing, proven
        # with a signed balance request, confirmed by the human, only then
        # saved. See kalshi_setup.py.
        from kalshi_setup import ensure_credentials

        if ensure_credentials(args.env_file) is None:
            print("Credential setup failed; not starting.")
            return 1

    # The monitor always starts in paper mode - even for live sessions, where
    # promotion happens only after the gates pass. Passing --live here would
    # arm execution from the first observation, which is the opposite of the
    # point of the warm-up.
    margs = monitor_parse_args(["--env-file", args.env_file, *extra])
    monitor = Monitor(margs)

    run_log = start_run_log(
        log,
        directory=margs.log_dir,
        title=f"KALSHI AUTOPILOT ({args.mode.upper()}"
              f"{', AGGRESSIVE' if margs.aggressive else ''}"
              f"{', gates before live' if args.mode == 'live' else ', paper only'})",
        context=[
            f"assets   : {', '.join(i.name + ' (' + i.series + ' <- ' + i.asset.coinbase_product + ')' for i in monitor.instruments)}",
            f"preset   : {'AGGRESSIVE' if margs.aggressive else 'standard'}",
            f"warm-up  : {args.warmup_min:.0f} min, then quality gates",
            f"min edge : {margs.min_edge:+.4f}/contract net of fees",
            f"confirm  : {margs.confirm_seconds:.1f}s / {margs.confirm_passes} passes, "
            f"book every {margs.book_interval:.2f}s",
        ],
    )
    monitor.run_log = run_log
    # The ledger exists even when file logging is off: the scorecard grades
    # from it, and a session that cannot be graded teaches nothing.
    from run_log import run_log_name
    from strategies import PaperLedger

    stem = run_log.path.stem if run_log else run_log_name().replace(".log", "")
    monitor.ledger = PaperLedger(Path(margs.log_dir) / f"paper_{stem}.jsonl")
    log.info("Recording trades to %s", monitor.ledger.path)
    if not margs.no_replay:
        # Separate from the ledger on purpose: the ledger holds trades to be
        # graded, this holds the market state every decision was made from,
        # including the passes where we decided not to act.
        monitor.attach_replay(
            Path(margs.log_dir) / f"replay_{stem}.jsonl", margs.replay_interval
        )

    control = _StopControl()
    notes: list[str] = []

    async def amain() -> None:
        loop = asyncio.get_running_loop()
        for sig in (_signal.SIGINT, _signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, control.on_signal)
        notes.extend(await autopilot(args, margs, monitor, control))

    try:
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(amain())
    finally:
        monitor.close_replay()
        summary = monitor.build_summary() + [""] + notes
        if run_log is not None:
            run_log.close(summary)
            print(f"\nSession log   : {run_log.path}")
        if monitor.ledger is not None:
            print(f"Paper ledger  : {monitor.ledger.path}")
        if monitor.replay is not None:
            print(f"Replay record : {monitor.replay.path} ({monitor.replay.rows:,} rows)")
        print("Send the session log file to review the run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
