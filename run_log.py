#!/usr/bin/env python3
"""Per-run log files.

Every run writes one file named:

    L_MMDDYY_HHMMSS.log

- `L`      literal prefix
- `MMDDYY` UTC date at initialization
- `HHMMSS` UTC time at initialization

e.g. `logs/L_081326_063245.log` for 2026-08-13 06:32:45 UTC. Naming on the
initialization instant (not on close) means the file exists and is readable
while the run is still going, which matters for a monitor left running for
hours.

The file gets a header describing what was run, the same stream of records
that goes to the console, and a footer summary written even if the run is
interrupted.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

#: Console stays terse; the file carries full timestamps and levels.
FILE_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s %(message)s"
FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"


def run_log_name(started: dt.datetime | None = None, prefix: str = "L") -> str:
    """`L_MMDDYY_HHMMSS.log`, from a UTC instant."""
    started = started or dt.datetime.now(dt.timezone.utc)
    started = started.astimezone(dt.timezone.utc)
    return f"{prefix}_{started:%m%d%y}_{started:%H%M%S}.log"


class _UtcFormatter(logging.Formatter):
    """Timestamps in UTC, so the log agrees with the filename and the venues."""

    converter = staticmethod(lambda secs: dt.datetime.fromtimestamp(secs, dt.timezone.utc).timetuple())


@dataclass(slots=True)
class RunLog:
    path: Path
    started: dt.datetime
    handler: logging.Handler
    logger: logging.Logger
    _closed: bool = field(default=False, repr=False)

    @property
    def elapsed(self) -> dt.timedelta:
        return dt.datetime.now(dt.timezone.utc) - self.started

    def section(self, title: str, lines: Sequence[str] = ()) -> None:
        """Write a titled block straight to the file and the console."""
        bar = "=" * max(20, min(70, len(title) + 8))
        body = "\n".join(f" {line}" for line in lines)
        self.logger.info("\n%s\n %s\n%s\n%s", bar, title, bar, body if body else "")

    def close(self, summary: Sequence[str] = ()) -> None:
        """Write the footer and detach. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        ended = dt.datetime.now(dt.timezone.utc)
        elapsed = ended - self.started
        hours, rem = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(rem, 60)

        lines = list(summary) + [
            "",
            f"started : {self.started:%Y-%m-%d %H:%M:%S} UTC",
            f"ended   : {ended:%Y-%m-%d %H:%M:%S} UTC",
            f"elapsed : {hours}h {minutes:02d}m {seconds:02d}s",
            f"log     : {self.path}",
        ]
        self.section("SESSION SUMMARY", lines)
        self.handler.flush()
        self.logger.removeHandler(self.handler)
        self.handler.close()


def start_run_log(
    logger: logging.Logger,
    *,
    directory: str | os.PathLike[str] = "logs",
    prefix: str = "L",
    title: str = "",
    context: Sequence[str] = (),
    level: int = logging.DEBUG,
) -> RunLog | None:
    """Attach a per-run file handler. Returns None if the file can't be opened.

    Logging must never be the reason a run fails, so an unwritable directory
    degrades to console-only with a warning rather than raising.
    """
    started = dt.datetime.now(dt.timezone.utc)
    path = Path(directory) / run_log_name(started, prefix)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not open a run log at %s (%s); console only", path, exc)
        return None

    handler.setFormatter(_UtcFormatter(FILE_FORMAT, datefmt=FILE_DATEFMT))
    handler.setLevel(level)
    logger.addHandler(handler)
    # The file is only as complete as the logger is permissive.
    if logger.level > level:
        logger.setLevel(level)

    run = RunLog(path=path, started=started, handler=handler, logger=logger)
    run.section(
        title or "RUN START",
        [
            f"started : {started:%Y-%m-%d %H:%M:%S} UTC",
            f"command : {' '.join(sys.argv)}",
            f"python  : {platform.python_version()} on {platform.system()}",
            *context,
        ],
    )
    logger.info("Writing run log to %s", path)
    return run
