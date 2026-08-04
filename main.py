"""
S.A.R.A. entry point.

Phase 1 scope: boot the core (config, logging, event bus, service
registry) and the Context Engine, log confirmation, and idle until
interrupted. GUI startup (``ui/``), voice, and agents are wired in here in
later phases — this file's job is the boot sequence, not feature logic.
"""

from __future__ import annotations

import signal
import sys
import time
import types

from core.app import Application
from utils.logger import get_logger

logger = get_logger(__name__)


def main() -> int:
    app = Application(config_path="config/settings.yaml")

    def _handle_sigint(signum: int, frame: types.FrameType | None) -> None:
        logger.info("Received interrupt signal, shutting down...")
        app.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        app.startup()
    except Exception:
        logger.exception("S.A.R.A. failed to start")
        return 1

    logger.info("S.A.R.A. is running. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        app.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
