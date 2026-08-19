#!/usr/bin/env python3
"""Operate the Kalshi autopilot from Telegram.

    python kalshi_telegram.py            # then message the bot from your phone

Commands (send them to the bot in Telegram):

    /start              a dry run, default warm-up
    /start live         real money, after the warm-up gates pass
    /start live 30      real money, 30-minute warm-up
    /stop               end the session and post the summary
    /status             balance, open positions, funnel, fill rate
    /score              the pooled scorecard across every session
    /help               this list

Between /start and /stop the session runs exactly as `kalshi_main.py` does -
same phases, same gates, same risk limits. This module only wraps it, so the
trading path is untouched: notifications are lifted off the existing log stream
by a logging handler rather than by new hooks in the strategies or the
executor. Nothing here can change what or how the bot trades.

Setup, once:

  1. Message @BotFather on Telegram, send /newbot, copy the token.
  2. Put it in .env as TELEGRAM_BOT_TOKEN=...
  3. Run this script and message your bot anything. It prints the chat id and
     refuses to act until you add TELEGRAM_CHAT_ID=... to .env.

That second step is not ceremony. The bot spends real money, and a token that
leaks is a stranger able to send /start. Commands from any chat other than the
allowlisted one are logged and ignored.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import html
import io
import logging
import os
import time
from pathlib import Path

import aiohttp

from btc_polymarket_arb import load_dotenv, log
from kalshi_main import _StopControl, autopilot
from kalshi_monitor import Monitor, parse_args as monitor_parse_args
from run_log import run_log_name, start_run_log
from strategies import PaperLedger

API = "https://api.telegram.org/bot{token}/{method}"

#: Log lines worth waking a phone for. Everything else stays in the file.
NOTIFY = (
    "LIVE TRADING ARMED",
    "PHASE:",
    "PAPER TRADE",
    "execution:",
    "EXITED",
    "SETTLED",
    "TRADING HALTED",
    "Side mapping",
    "AGGRESSIVE preset",
    "Order not filled",
)

MAX_MESSAGE = 3800  # Telegram's limit is 4096; leave room for formatting.


class Notifier(logging.Handler):
    """Forwards interesting log records to Telegram, batched.

    A handler rather than callbacks scattered through the trading code: the
    monitor already logs every fill, exit, settlement and halt at WARNING, so
    there is nothing to add and no risk of a notification bug reaching the
    order path. Batching matters because a busy second can produce a dozen
    lines and Telegram rate-limits per chat - and because twelve buzzes is
    worse than one.
    """

    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__(level=logging.INFO)
        self._queue = queue
        self._loop = loop

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = record.getMessage()
        except Exception:  # noqa: BLE001 - logging must never raise
            return
        if not any(marker in text for marker in NOTIFY):
            return
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._queue.put_nowait, text.strip())


class TelegramBot:
    def __init__(self, token: str, chat_id: str | None, args: argparse.Namespace) -> None:
        self._token = token
        self.chat_id = chat_id
        self._args = args
        self._session: aiohttp.ClientSession | None = None
        self._offset = 0
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._notifier: Notifier | None = None

        # Live session state.
        self.task: asyncio.Task | None = None
        self.monitor: Monitor | None = None
        self.control: _StopControl | None = None
        self.run_log = None
        self.started_at = 0.0
        self.mode = "dry"

    # -- transport ---------------------------------------------------------- #

    async def call(self, method: str, **payload):
        assert self._session is not None
        url = API.format(token=self._token, method=method)
        try:
            async with self._session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=45)
            ) as resp:
                return await resp.json()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a chat outage must not stop trading
            log.warning("Telegram %s failed: %s", method, exc)
            return None

    async def send(self, text: str, chat_id: str | None = None) -> None:
        target = chat_id or self.chat_id
        if not target:
            return
        for chunk in _split(text, MAX_MESSAGE):
            await self.call(
                "sendMessage", chat_id=target,
                text=f"<pre>{html.escape(chunk)}</pre>", parse_mode="HTML",
            )

    # -- loops -------------------------------------------------------------- #

    async def _drain_notifications(self) -> None:
        """Batch queued log lines into one message every couple of seconds."""
        while True:
            first = await self._queue.get()
            await asyncio.sleep(2.0)
            lines = [first]
            while not self._queue.empty() and len("\n".join(lines)) < MAX_MESSAGE:
                lines.append(self._queue.get_nowait())
            await self.send("\n".join(lines))

    async def poll(self) -> None:
        """Long-poll for commands. Survives network blips without dying."""
        while True:
            try:
                data = await self.call(
                    "getUpdates", offset=self._offset, timeout=30,
                    allowed_updates=["message"],
                )
                for update in (data or {}).get("result", []):
                    self._offset = max(self._offset, update.get("update_id", 0) + 1)
                    await self._on_update(update)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("Telegram poll error: %s", exc)
                await asyncio.sleep(5.0)

    async def _on_update(self, update: dict) -> None:
        message = update.get("message") or {}
        text = str(message.get("text") or "").strip()
        chat = str((message.get("chat") or {}).get("id") or "")
        if not text or not chat:
            return

        if not self.chat_id:
            # Onboarding: tell the operator their id, but refuse to act.
            log.warning("Message from unregistered chat %s: %r", chat, text[:40])
            await self.send(
                f"Not configured yet.\n\nYour chat id is:\n  {chat}\n\n"
                f"Add this to .env and restart:\n  TELEGRAM_CHAT_ID={chat}",
                chat_id=chat,
            )
            return
        if chat != self.chat_id:
            # The token alone must not be enough to spend money.
            log.warning("IGNORED command from chat %s (not allowlisted): %r",
                        chat, text[:40])
            return

        await self._command(text)

    # -- commands ----------------------------------------------------------- #

    async def _command(self, text: str) -> None:
        parts = text.split()
        cmd = parts[0].lower().lstrip("/").split("@")[0]
        rest = parts[1:]
        handler = {
            "start": self._cmd_start, "run": self._cmd_start,
            "stop": self._cmd_stop,
            "status": self._cmd_status,
            "score": self._cmd_score,
            "help": self._cmd_help, "commands": self._cmd_help,
        }.get(cmd)
        if handler is None:
            await self.send(f"Unknown command: {cmd}\nSend /help for the list.")
            return
        try:
            await handler(rest)
        except Exception as exc:  # noqa: BLE001
            log.exception("Command /%s failed", cmd)
            await self.send(f"/{cmd} failed: {type(exc).__name__}: {exc}")

    async def _cmd_help(self, _rest: list[str]) -> None:
        await self.send(__doc__.split("Setup, once:")[0].strip())

    async def _cmd_start(self, rest: list[str]) -> None:
        if self.task is not None and not self.task.done():
            await self.send(
                f"Already running ({self.mode.upper()}, "
                f"{(time.time() - self.started_at) / 60:.0f} min in). "
                f"Send /stop first."
            )
            return

        mode = "live" if rest and rest[0].lower() in ("live", "real") else "dry"
        warmup = 20.0
        for token in rest[1:] or rest:
            with contextlib.suppress(ValueError):
                warmup = max(float(token), 6.0)

        margs = monitor_parse_args(
            ["--env-file", self._args.env_file, *self._args.monitor_flags]
        )
        if mode == "live":
            from kalshi import KalshiCredentials

            problems = KalshiCredentials.from_env().problems()
            if problems:
                await self.send(
                    "Cannot run live: credentials are not set up.\n  "
                    + "\n  ".join(problems)
                    + "\n\nRun `python kalshi_setup.py` on the host once; it "
                      "cannot be done from here because it needs your private key."
                )
                return

        monitor = Monitor(margs)
        run_log = start_run_log(
            log,
            directory=margs.log_dir,
            title=f"KALSHI AUTOPILOT ({mode.upper()} via Telegram"
                  f"{', AGGRESSIVE' if margs.aggressive else ''})",
            context=[
                f"assets   : {', '.join(i.name for i in monitor.instruments)}",
                f"warm-up  : {warmup:.0f} min, then quality gates",
                f"min edge : {margs.min_edge:+.4f}/contract net of fees",
            ],
        )
        monitor.run_log = run_log
        stem = run_log.path.stem if run_log else run_log_name().replace(".log", "")
        monitor.ledger = PaperLedger(Path(margs.log_dir) / f"paper_{stem}.jsonl")

        session_args = argparse.Namespace(
            mode=mode, warmup_min=warmup,
            session_min=self._args.session_min,
            settle_wait_min=self._args.settle_wait_min,
            env_file=self._args.env_file,
        )
        self.monitor, self.run_log = monitor, run_log
        self.control = _StopControl()
        self.mode, self.started_at = mode, time.time()
        self.task = asyncio.create_task(
            self._run_session(session_args, margs, monitor, self.control)
        )

        await self.send(
            f"Started: {mode.upper()}"
            f"{' (AGGRESSIVE)' if margs.aggressive else ''}\n"
            f"assets   : {', '.join(i.name for i in monitor.instruments)}\n"
            f"warm-up  : {warmup:.0f} min of paper before the gates are checked\n"
            f"log      : {run_log.path if run_log else 'console only'}\n\n"
            + ("Real orders can start once the warm-up gates pass."
               if mode == "live" else "Paper only - no orders will be sent.")
            + "\nSend /stop to end it, /status any time."
        )

    async def _run_session(self, args, margs, monitor, control) -> None:
        notes: list[str] = []
        try:
            notes = await autopilot(args, margs, monitor, control)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("Session failed")
            await self.send(f"Session ERROR: {type(exc).__name__}: {exc}")
        finally:
            summary = monitor.build_summary() + [""] + notes
            if self.run_log is not None:
                self.run_log.close(summary)
            await self.send("SESSION ENDED\n\n" + "\n".join(summary))
            self.task = None

    async def _cmd_stop(self, _rest: list[str]) -> None:
        if self.task is None or self.task.done():
            await self.send("Nothing is running.")
            return
        await self.send("Stopping... settling open positions, then the summary.")
        # The same graceful path Ctrl+C takes: finish the phase, grade, report.
        self.control.on_signal()

    async def _cmd_status(self, _rest: list[str]) -> None:
        monitor = self.monitor
        if self.task is None or self.task.done() or monitor is None:
            await self.send("Idle. Send /start or /start live to begin.")
            return
        trader = monitor.trader
        lines = [
            f"{self.mode.upper()} session, "
            f"{(time.time() - self.started_at) / 60:.0f} min in",
            f"execution : {trader.stats() if trader else 'not armed yet'}",
            "funnel    : " + " -> ".join(
                f"{k} {v}" for k, v in monitor._funnel.items() if k != "unpriceable"
            ),
        ]
        for inst in monitor.instruments:
            tick = inst.buffer.last()
            market = inst.market
            lines.append(
                f"{inst.name:4s} {market.ticker.split('-')[-2] if market else 'no market':<14s}"
                f" spot {('$%,.2f' % tick.price) if tick else 'n/a':>12s}"
                f" | obs {inst.observations:,}"
            )
        if trader is not None:
            for order in trader.open_positions():
                lines.append(
                    f"  open: {order.outcome} x{order.count} @ {order.price:.3f} "
                    f"on {order.ticker}"
                )
        await self.send("\n".join(lines))

    async def _cmd_score(self, _rest: list[str]) -> None:
        await self.send("Scoring every recorded session against settlement...")
        from kalshi_score import main as score_main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            await score_main()
        await self.send(buf.getvalue() or "Nothing settled yet.")

    # -- lifecycle ---------------------------------------------------------- #

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        self._notifier = Notifier(self._queue, loop)
        log.addHandler(self._notifier)
        async with aiohttp.ClientSession() as session:
            self._session = session
            me = await self.call("getMe")
            name = ((me or {}).get("result") or {}).get("username", "?")
            log.info("Telegram bot @%s connected", name)
            if self.chat_id:
                await self.send(
                    f"Kalshi bot @{name} is up.\n"
                    f"Send /start for a dry run, /start live for real money, "
                    f"/help for everything."
                )
            else:
                log.warning(
                    "TELEGRAM_CHAT_ID is not set. Message the bot once and it "
                    "will reply with the id to put in .env."
                )
            drain = asyncio.create_task(self._drain_notifications())
            try:
                await self.poll()
            finally:
                drain.cancel()
                log.removeHandler(self._notifier)


def _split(text: str, limit: int) -> list[str]:
    """Break a long report on line boundaries so nothing is silently truncated."""
    if len(text) <= limit:
        return [text]
    out, current = [], ""
    for line in text.splitlines(keepends=True):
        if len(current) + len(line) > limit and current:
            out.append(current)
            current = ""
        current += line
    if current:
        out.append(current)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--session-min", type=float, default=0.0,
                        help="auto-stop a session after this many minutes; "
                             "0 means it runs until /stop")
    parser.add_argument("--settle-wait-min", type=float, default=25.0)
    parser.add_argument("monitor_flags", nargs="*",
                        help="flags passed straight through to the monitor, "
                             "e.g. --assets BTC,ETH --aggressive")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("websockets").setLevel(logging.WARNING)
    load_dotenv(args.env_file)

    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        print(
            "TELEGRAM_BOT_TOKEN is not set.\n\n"
            "  1. Message @BotFather on Telegram, send /newbot\n"
            "  2. Put the token in .env as TELEGRAM_BOT_TOKEN=...\n"
            "  3. Run this again and message your bot; it will tell you the\n"
            "     chat id to add as TELEGRAM_CHAT_ID."
        )
        return 1
    chat_id = (os.getenv("TELEGRAM_CHAT_ID") or "").strip() or None

    bot = TelegramBot(token, chat_id, args)
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(bot.run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
