"""
Central logging configuration for S.A.R.A.

Every module obtains its logger via ``get_logger(__name__)`` rather than
configuring its own handlers. This keeps log formatting, rotation, and
output destinations defined in exactly one place (this module), which is
what ``config/settings.yaml`` controls via the ``logging:`` section.

We use ``loguru`` instead of the stdlib ``logging`` module because it gives
us structured, leveled logging with sane defaults (rotation, retention,
readable formatting) without needing a `logging.config.dictConfig` block —
and it plays nicely with binding module-scoped context (see
``get_logger``).

Usage
-----
    from utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("SARA core initialized")
    logger.bind(agent="coding_agent").debug("dispatching task")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger as _loguru_logger

_CONFIGURED = False


def configure_logging(
    *,
    log_dir: str | Path = "logs",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    rotation: str = "10 MB",
    retention: str = "14 days",
) -> None:
    """Configure the global loguru sinks. Must be called once at startup.

    This is invoked by ``core.app.Application.startup`` before any other
    module logs anything. Calling it more than once is safe — subsequent
    calls are no-ops — so individual modules/tests can call it defensively
    without worrying about duplicate sinks.

    Args:
        log_dir: Directory where rotating log files are written.
        console_level: Minimum level shown in the console/stderr sink.
        file_level: Minimum level written to the rotating file sink.
        rotation: loguru rotation policy (size- or time-based).
        retention: How long rotated log files are kept before deletion.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    _loguru_logger.remove()  # drop loguru's default stderr sink; we define our own

    _loguru_logger.add(
        sys.stderr,
        level=console_level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[module]}</cyan> - <level>{message}</level>"
        ),
        colorize=True,
    )

    _loguru_logger.add(
        log_path / "sara.log",
        level=file_level,
        rotation=rotation,
        retention=retention,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{extra[module]} - {message}"
        ),
        encoding="utf-8",
        enqueue=True,  # process-safe: background threads/agents can log concurrently
    )

    # Default 'module' binding so format strings above never KeyError before
    # a module-scoped logger has bound its own name.
    _loguru_logger.configure(extra={"module": "sara"})

    _CONFIGURED = True
    get_logger(__name__).info(
        "Logging configured (console={}, file={}, dir={})",
        console_level,
        file_level,
        log_path.resolve(),
    )


def get_logger(module_name: str) -> Any:
    """Return a logger bound to ``module_name`` for use in log output.

    Args:
        module_name: Conventionally ``__name__`` of the calling module.

    Returns:
        A loguru logger instance with the module name bound into ``extra``
        so every line is attributable to its source module.
    """
    if not _CONFIGURED:
        # Fail-safe default so importing a module before startup.configure_logging()
        # runs (e.g. in a unit test) still produces usable output instead of
        # a KeyError from the format string's {extra[module]}.
        configure_logging()
    return _loguru_logger.bind(module=module_name)
