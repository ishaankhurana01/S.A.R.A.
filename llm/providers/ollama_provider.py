"""
Ollama LLM provider.

``OllamaProvider`` is the only piece of code in the whole project that
speaks to Ollama. It implements ``core.interfaces.LLMProvider`` (see
``llm.providers.base_provider`` for why that ABC lives in ``core``, not
here), and is consumed exclusively by
``agents.conversation_agent.ConversationAgent`` — no other agent holds a
reference to it, which is what requirement #5 ("no other agent should
communicate directly with Ollama") means structurally rather than just as
a coding convention: nothing else in ``core.service_registry`` resolves
``LLMProvider`` except the Conversation Agent.

Implementation notes
---------------------
- Talks to Ollama's REST API directly over HTTP (``/api/generate`` and
  ``/api/tags``) via the ``requests`` library, rather than the official
  ``ollama`` Python package. This is a deliberate choice for this phase:
  it keeps the dependency surface small, makes every failure mode
  (connection refused, timeout, HTTP error, malformed JSON) directly
  mockable in unit tests without a real Ollama server, and the wire
  format is simple enough that wrapping it ourselves costs little.
- ``stream: False`` is always sent — per requirement #7, streaming is an
  explicit non-goal for this phase; the provider returns one complete
  string.
- Error mapping is deliberately conservative: Ollama's exact HTTP status
  code for "model not found" has varied across versions (some return 404,
  older ones return 400/500 with an ``"error"`` field), so we inspect the
  response body's error text rather than trusting a single status code.
"""

from __future__ import annotations

from typing import Any

import requests

from core.interfaces import LLMProvider
from utils.exceptions import (
    LLMError,
    LLMInvalidResponseError,
    LLMModelNotFoundError,
    LLMProviderUnavailableError,
    LLMTimeoutError,
)
from utils.logger import get_logger

logger = get_logger(__name__)

_GENERATE_PATH = "/api/generate"
_TAGS_PATH = "/api/tags"


class OllamaProvider(LLMProvider):
    """Talks to a local Ollama server to generate chat completions.

    Example:
        provider = OllamaProvider(
            host="http://localhost:11434",
            model="qwen2.5:7b",
            request_timeout_seconds=30.0,
        )
        text = provider.generate("What's a good name for a cat?")
    """

    def __init__(
        self,
        *,
        host: str,
        model: str,
        request_timeout_seconds: float = 30.0,
        default_temperature: float = 0.7,
    ) -> None:
        """
        Args:
            host: Base URL of the Ollama server, e.g. ``"http://localhost:11434"``.
                No trailing slash expected; one is stripped defensively if present.
            model: Model tag to request, e.g. ``"qwen2.5:7b"``.
            request_timeout_seconds: Per-request HTTP timeout. This is
                independent of (and typically shorter than) any timeout
                the Executive Agent's reasoning loop applies at the task
                level — see ``config.config_schema.LLMConfig.request_timeout_seconds``.
            default_temperature: Used when ``generate()`` is called
                without an explicit ``temperature`` override.
        """
        self._host = host.rstrip("/")
        self._model = model
        self._request_timeout_seconds = request_timeout_seconds
        self._default_temperature = default_temperature

    def generate(self, prompt: str, *, temperature: float | None = None) -> str:
        """Generate a complete (non-streamed) response for ``prompt``.

        Args:
            prompt: The user prompt to send to the model.
            temperature: Overrides the provider's default temperature for
                this call only.

        Returns:
            The model's complete response text.

        Raises:
            LLMProviderUnavailableError: The Ollama server could not be
                reached at all (connection refused, DNS failure, etc.) —
                this is the "Ollama isn't running" case.
            LLMTimeoutError: The request exceeded ``request_timeout_seconds``.
            LLMModelNotFoundError: The configured model is not available
                on the server.
            LLMInvalidResponseError: The server responded but the body
                was not valid JSON, or lacked a usable response field.
            LLMError: Any other non-success response from the server.
        """
        effective_temperature = temperature if temperature is not None else self._default_temperature
        url = f"{self._host}{_GENERATE_PATH}"
        request_body = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": effective_temperature},
        }

        logger.debug("Requesting completion from {} (model={})", url, self._model)
        try:
            response = requests.post(url, json=request_body, timeout=self._request_timeout_seconds)
        except requests.exceptions.Timeout as exc:
            raise LLMTimeoutError(
                f"Ollama request timed out after {self._request_timeout_seconds}s",
                context={"host": self._host, "model": self._model},
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise LLMProviderUnavailableError(
                f"Could not reach Ollama at {self._host}. Is the Ollama server running?",
                context={"host": self._host},
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise LLMError(
                f"Unexpected error contacting Ollama: {exc}",
                context={"host": self._host},
            ) from exc

        return self._parse_generate_response(response)

    def _parse_generate_response(self, response: "requests.Response") -> str:
        """Validate the HTTP response and extract the generated text.

        Split out from ``generate`` so error-path logic (shared shape
        between success/failure bodies) is unit-testable against a bare
        ``requests.Response``-like object without a real HTTP round trip.
        """
        body: Any
        try:
            body = response.json()
        except ValueError as exc:
            raise LLMInvalidResponseError(
                f"Ollama returned a non-JSON response (HTTP {response.status_code})",
                context={"status_code": response.status_code, "text_preview": response.text[:200]},
            ) from exc

        if response.status_code != 200:
            error_text = ""
            if isinstance(body, dict):
                error_text = str(body.get("error", ""))
            if "not found" in error_text.lower():
                raise LLMModelNotFoundError(
                    f"Model '{self._model}' is not available on this Ollama server: {error_text}",
                    context={"model": self._model, "status_code": response.status_code},
                )
            raise LLMError(
                f"Ollama returned HTTP {response.status_code}: {error_text or body}",
                context={"status_code": response.status_code, "model": self._model},
            )

        if not isinstance(body, dict):
            raise LLMInvalidResponseError(
                "Ollama response body was not a JSON object",
                context={"body_type": type(body).__name__},
            )

        text = body.get("response")
        if not isinstance(text, str) or not text.strip():
            raise LLMInvalidResponseError(
                "Ollama response did not contain usable text in the 'response' field",
                context={"keys_present": list(body.keys())},
            )

        return text

    def is_available(self) -> bool:
        """Return whether the Ollama server is reachable right now.

        Uses ``GET /api/tags`` (lists installed models) as a cheap
        liveness probe rather than issuing a real generation request.
        Any failure (connection error, timeout, non-200) is treated as
        "not available" — this method is designed to never raise, so
        callers (e.g. a future health-check UI element) can use it
        directly in a boolean context.
        """
        url = f"{self._host}{_TAGS_PATH}"
        try:
            response = requests.get(url, timeout=self._request_timeout_seconds)
        except requests.exceptions.RequestException as exc:
            logger.debug("Ollama availability check failed: {}", exc)
            return False
        return response.status_code == 200
