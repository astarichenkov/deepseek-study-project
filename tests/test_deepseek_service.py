"""Unit tests for DeepSeekService with a fully mocked OpenAI client.

The OpenAI client is replaced by a fake, so nothing here touches the
network or spends API balance.
"""
import asyncio

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from app.config import Settings
from app.services.deepseek import (
    DeepSeekAuthenticationError,
    DeepSeekError,
    DeepSeekMalformedResponseError,
    DeepSeekNetworkError,
    DeepSeekRateLimitError,
    DeepSeekTimeoutError,
    DeepSeekService,
)


# ---------------------------------------------------------------------------
# Fakes that mimic the shape of OpenAI SDK responses
# ---------------------------------------------------------------------------
class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, message):
        self.message = message


class FakeChatCompletion:
    def __init__(self, choices):
        self.choices = choices


class FakeCompletions:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeAsyncOpenAI:
    """Drop-in replacement for ``AsyncOpenAI`` (kept for reference/typing).

    Tests use :class:`RecordingAsyncOpenAI` (defined in the ``make_service``
    fixture) which also captures constructor kwargs.
    """

    def __init__(self, result=None, **kwargs):
        self.constructor_kwargs = kwargs
        self.completions = FakeCompletions(result)
        self.chat = FakeChat(self.completions)


@pytest.fixture
def make_service(monkeypatch):
    """Build a DeepSeekService whose client is a RecordingAsyncOpenAI.

    The fake records the constructor kwargs (api_key / base_url / timeout)
    and the ``chat.completions.create`` calls, and returns ``result``
    (or raises it if it is an exception).
    """

    def _make(result):
        class RecordingAsyncOpenAI:
            instances = []

            def __init__(self, **kwargs):
                self.constructor_kwargs = kwargs
                self.completions = FakeCompletions(result)
                self.chat = FakeChat(self.completions)
                RecordingAsyncOpenAI.instances.append(self)

        monkeypatch.setattr(
            "app.services.deepseek.AsyncOpenAI", RecordingAsyncOpenAI
        )
        service = DeepSeekService(Settings(deepseek_api_key="test-key"))
        return service, RecordingAsyncOpenAI.instances[-1]

    return _make


def _request():
    """A minimal httpx.Request object required by OpenAI SDK errors."""
    return httpx.Request("POST", "https://api.deepseek.com/chat/completions")


def _status_error(error_cls, code):
    """Build a real OpenAI SDK status error (needs a real httpx.Response)."""
    return error_cls(
        f"error {code}",
        response=httpx.Response(code, request=_request()),
        body=None,
    )


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------
def test_success_returns_answer_and_uses_correct_config(make_service):
    completion = FakeChatCompletion(
        [FakeChoice(FakeMessage("Dependency injection is a design pattern."))]
    )
    service, fake_client = make_service(completion)

    result = asyncio.run(service.chat("Explain dependency injection"))

    assert result.answer == "Dependency injection is a design pattern."
    call = fake_client.completions.calls[0]
    assert call["model"] == "deepseek-v4-flash"
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][1] == {"role": "user", "content": "Explain dependency injection"}
    assert fake_client.constructor_kwargs["base_url"] == "https://api.deepseek.com"


def test_success_strips_surrounding_whitespace(make_service):
    completion = FakeChatCompletion([FakeChoice(FakeMessage("  answer text  "))])
    service, _ = make_service(completion)

    result = asyncio.run(service.chat("hi"))
    assert result.answer == "answer text"


# ---------------------------------------------------------------------------
# Malformed provider responses
# ---------------------------------------------------------------------------
def test_empty_choices_raises_malformed(make_service):
    service, _ = make_service(FakeChatCompletion([]))
    with pytest.raises(DeepSeekMalformedResponseError):
        asyncio.run(service.chat("hi"))


def test_missing_choices_raises_malformed(make_service):
    service, _ = make_service(FakeChatCompletion(None))
    with pytest.raises(DeepSeekMalformedResponseError):
        asyncio.run(service.chat("hi"))


def test_blank_content_raises_malformed(make_service):
    service, _ = make_service(FakeChatCompletion([FakeChoice(FakeMessage("   "))]))
    with pytest.raises(DeepSeekMalformedResponseError):
        asyncio.run(service.chat("hi"))


def test_none_content_raises_malformed(make_service):
    service, _ = make_service(FakeChatCompletion([FakeChoice(FakeMessage(None))]))
    with pytest.raises(DeepSeekMalformedResponseError):
        asyncio.run(service.chat("hi"))


# ---------------------------------------------------------------------------
# Error mapping from OpenAI SDK exceptions to domain errors
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("exc", "expected", "expected_status"),
    [
        (_status_error(AuthenticationError, 401), DeepSeekAuthenticationError, 401),
        (_status_error(RateLimitError, 429), DeepSeekRateLimitError, 429),
        (APITimeoutError(request=_request()), DeepSeekTimeoutError, 504),
        (
            APIConnectionError(message="connection refused", request=_request()),
            DeepSeekNetworkError,
            502,
        ),
        (APIError("internal server error", request=_request(), body=None), DeepSeekError, 502),
        (RuntimeError("unexpected"), DeepSeekError, 500),
    ],
)
def test_exception_mapping(make_service, exc, expected, expected_status):
    service, _ = make_service(exc)
    with pytest.raises(expected) as excinfo:
        asyncio.run(service.chat("hi"))
    assert excinfo.value.status_code == expected_status
