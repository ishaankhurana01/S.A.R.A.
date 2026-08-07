"""
Unit tests for ``llm.providers.ollama_provider.OllamaProvider``.

Ollama is never actually running in this test environment — every test
mocks ``requests.post``/``requests.get`` at the module level
(``llm.providers.ollama_provider.requests``) so the provider's error
mapping can be verified deterministically, including failure modes
(connection refused, timeout) that would be awkward to trigger reliably
against a real server.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from llm.providers.ollama_provider import OllamaProvider
from utils.exceptions import (
    LLMError,
    LLMInvalidResponseError,
    LLMModelNotFoundError,
    LLMProviderUnavailableError,
    LLMTimeoutError,
)


def _make_provider(**overrides) -> OllamaProvider:
    defaults = dict(host="http://localhost:11434", model="qwen2.5:7b", request_timeout_seconds=5.0)
    defaults.update(overrides)
    return OllamaProvider(**defaults)


def _mock_response(status_code: int, json_body: object) -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.status_code = status_code
    response.json.return_value = json_body
    response.text = str(json_body)
    return response


# --------------------------------------------------------------------------- #
# Successful generation
# --------------------------------------------------------------------------- #
@patch("llm.providers.ollama_provider.requests.post")
def test_generate_returns_response_text(mock_post) -> None:
    mock_post.return_value = _mock_response(200, {"response": "Hello there!", "done": True})
    provider = _make_provider()

    result = provider.generate("Say hello")

    assert result == "Hello there!"
    called_url = mock_post.call_args.args[0]
    assert called_url == "http://localhost:11434/api/generate"
    sent_body = mock_post.call_args.kwargs["json"]
    assert sent_body["model"] == "qwen2.5:7b"
    assert sent_body["prompt"] == "Say hello"
    assert sent_body["stream"] is False


@patch("llm.providers.ollama_provider.requests.post")
def test_generate_uses_temperature_override(mock_post) -> None:
    mock_post.return_value = _mock_response(200, {"response": "ok"})
    provider = _make_provider(default_temperature=0.7)

    provider.generate("test", temperature=0.1)

    sent_body = mock_post.call_args.kwargs["json"]
    assert sent_body["options"]["temperature"] == 0.1


@patch("llm.providers.ollama_provider.requests.post")
def test_generate_falls_back_to_default_temperature(mock_post) -> None:
    mock_post.return_value = _mock_response(200, {"response": "ok"})
    provider = _make_provider(default_temperature=0.42)

    provider.generate("test")

    sent_body = mock_post.call_args.kwargs["json"]
    assert sent_body["options"]["temperature"] == 0.42


@patch("llm.providers.ollama_provider.requests.post")
def test_host_trailing_slash_is_stripped(mock_post) -> None:
    mock_post.return_value = _mock_response(200, {"response": "ok"})
    provider = _make_provider(host="http://localhost:11434/")

    provider.generate("test")

    assert mock_post.call_args.args[0] == "http://localhost:11434/api/generate"


# --------------------------------------------------------------------------- #
# Ollama server unavailable
# --------------------------------------------------------------------------- #
@patch("llm.providers.ollama_provider.requests.post")
def test_connection_error_raises_provider_unavailable(mock_post) -> None:
    mock_post.side_effect = requests.exceptions.ConnectionError("refused")
    provider = _make_provider()

    with pytest.raises(LLMProviderUnavailableError):
        provider.generate("hello")


@patch("llm.providers.ollama_provider.requests.get")
def test_is_available_false_on_connection_error(mock_get) -> None:
    mock_get.side_effect = requests.exceptions.ConnectionError("refused")
    provider = _make_provider()

    assert provider.is_available() is False


@patch("llm.providers.ollama_provider.requests.get")
def test_is_available_true_on_200(mock_get) -> None:
    mock_get.return_value = _mock_response(200, {"models": []})
    provider = _make_provider()

    assert provider.is_available() is True


@patch("llm.providers.ollama_provider.requests.get")
def test_is_available_false_on_non_200(mock_get) -> None:
    mock_get.return_value = _mock_response(500, {"error": "internal error"})
    provider = _make_provider()

    assert provider.is_available() is False


# --------------------------------------------------------------------------- #
# Timeout
# --------------------------------------------------------------------------- #
@patch("llm.providers.ollama_provider.requests.post")
def test_timeout_raises_llm_timeout_error(mock_post) -> None:
    mock_post.side_effect = requests.exceptions.Timeout("timed out")
    provider = _make_provider(request_timeout_seconds=2.0)

    with pytest.raises(LLMTimeoutError):
        provider.generate("hello")


# --------------------------------------------------------------------------- #
# Unknown model
# --------------------------------------------------------------------------- #
@patch("llm.providers.ollama_provider.requests.post")
def test_model_not_found_via_404(mock_post) -> None:
    mock_post.return_value = _mock_response(404, {"error": "model 'ghost-model' not found"})
    provider = _make_provider(model="ghost-model")

    with pytest.raises(LLMModelNotFoundError):
        provider.generate("hello")


@patch("llm.providers.ollama_provider.requests.post")
def test_model_not_found_via_500_with_error_text(mock_post) -> None:
    # Older Ollama versions surface a missing model as HTTP 500 rather
    # than 404 — the provider must detect this from the error text, not
    # just the status code.
    mock_post.return_value = _mock_response(
        500, {"error": "model 'ghost-model' not found, try pulling it first"}
    )
    provider = _make_provider(model="ghost-model")

    with pytest.raises(LLMModelNotFoundError):
        provider.generate("hello")


# --------------------------------------------------------------------------- #
# Invalid / malformed responses
# --------------------------------------------------------------------------- #
@patch("llm.providers.ollama_provider.requests.post")
def test_non_json_body_raises_invalid_response(mock_post) -> None:
    response = MagicMock(spec=requests.Response)
    response.status_code = 200
    response.json.side_effect = ValueError("not json")
    response.text = "<html>not json</html>"
    mock_post.return_value = response
    provider = _make_provider()

    with pytest.raises(LLMInvalidResponseError):
        provider.generate("hello")


@patch("llm.providers.ollama_provider.requests.post")
def test_missing_response_field_raises_invalid_response(mock_post) -> None:
    mock_post.return_value = _mock_response(200, {"done": True})  # no "response" key
    provider = _make_provider()

    with pytest.raises(LLMInvalidResponseError):
        provider.generate("hello")


@patch("llm.providers.ollama_provider.requests.post")
def test_empty_response_text_raises_invalid_response(mock_post) -> None:
    mock_post.return_value = _mock_response(200, {"response": "   "})
    provider = _make_provider()

    with pytest.raises(LLMInvalidResponseError):
        provider.generate("hello")


@patch("llm.providers.ollama_provider.requests.post")
def test_non_dict_json_body_raises_invalid_response(mock_post) -> None:
    mock_post.return_value = _mock_response(200, ["not", "a", "dict"])
    provider = _make_provider()

    with pytest.raises(LLMInvalidResponseError):
        provider.generate("hello")


# --------------------------------------------------------------------------- #
# Generic provider failure
# --------------------------------------------------------------------------- #
@patch("llm.providers.ollama_provider.requests.post")
def test_unexpected_http_error_raises_generic_llm_error(mock_post) -> None:
    mock_post.return_value = _mock_response(400, {"error": "bad request: malformed prompt"})
    provider = _make_provider()

    with pytest.raises(LLMError) as exc_info:
        provider.generate("hello")
    assert not isinstance(exc_info.value, LLMModelNotFoundError)


@patch("llm.providers.ollama_provider.requests.post")
def test_other_request_exception_raises_generic_llm_error(mock_post) -> None:
    mock_post.side_effect = requests.exceptions.RequestException("weird network issue")
    provider = _make_provider()

    with pytest.raises(LLMError):
        provider.generate("hello")
