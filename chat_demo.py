"""
S.A.R.A. — Phase 3 terminal demo.

Boots the full core (same ``core.app.Application`` used by ``main.py``)
and drops into a simple REPL: type a prompt, it goes through
``executive_agent.submit_task("conversation.chat", ...)`` — Task ->
Executive Agent -> Capability Registry -> ConversationAgent ->
OllamaProvider -> Executive Agent -> Result — and the response (or a
clear error) is printed.

This is a demonstration script, not a permanent UI (that's ``ui/``,
Phase 2 of the original roadmap) — its only job is proving the pipeline
works end-to-end with a human typing at one end.

Usage:
    python chat_demo.py
    (requires a running Ollama server with the configured model pulled —
    see config/settings.yaml's `llm:` section for host/model)

Type 'exit' or 'quit' to stop, Ctrl+C also works.
"""

from __future__ import annotations

from core.app import Application
from utils.logger import get_logger

logger = get_logger(__name__)

_EXIT_COMMANDS = {"exit", "quit", "q"}


def main() -> int:
    app = Application(config_path="config/settings.yaml")

    print("Starting S.A.R.A. core...")
    try:
        app.startup()
    except Exception as exc:  # noqa: BLE001 - top-level demo entry point
        print(f"Failed to start S.A.R.A.: {exc}")
        return 1

    assert app.executive_agent is not None  # guaranteed by a successful startup()

    print(f"S.A.R.A. is ready (model: {app.config.llm.model}, host: {app.config.llm.host})")
    if not app.llm_provider.is_available():
        print(
            "Warning: Ollama does not appear to be reachable right now. "
            "You can still type a prompt — it will fail with a clear error "
            "instead of hanging, demonstrating the failure path."
        )
    print("Type a message and press Enter. Type 'exit' to quit.\n")

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue
            if user_input.lower() in _EXIT_COMMANDS:
                break

            result = app.executive_agent.submit_task(
                "conversation.chat",
                user_input,
                timeout_seconds=app.config.llm.request_timeout_seconds + 5.0,
            )

            if result.success:
                print(f"S.A.R.A.: {result.result['response']}\n")
            else:
                print(f"S.A.R.A. [error - {result.reason}]: {result.error_message}\n")
    except KeyboardInterrupt:
        print()  # tidy newline after ^C
    finally:
        print("Shutting down S.A.R.A....")
        app.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
