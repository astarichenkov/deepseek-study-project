"""Service-level tests for the compare flow with a fully mocked client.

These tests verify the CORE contract of the assignment:

* exactly TWO provider calls per comparison;
* the SAME original user prompt is used for both;
* the unrestricted call has NO custom response_format / max_tokens / stop;
* the controlled call forwards the exact validated ``response_format``
  object (e.g. ``{"type": "json_object"}``), ``max_tokens`` and
  ``stop=["<END>"]`` to the mocked OpenAI client;
* the controlled request adds the JSON-output mention and the explicit
  termination instruction;
* finish_reason comes from the (fake) provider response.

No network traffic happens here.
"""
import asyncio

import httpx
import pytest
from openai import AuthenticationError, RateLimitError

from app.config import Settings
from app.schemas.compare import CompareRequest, ResponseFormat
from app.services.deepseek import (
    DEFAULT_MAX_TOKENS,
    DeepSeekAuthenticationError,
    DeepSeekError,
    DeepSeekService,
)

PROMPT = "Назови три преимущества REST API и кратко объясни каждое."


def _completion(answer: str, finish_reason: str | None = "stop"):
    """A fake ChatCompletion object shaped like the OpenAI SDK response."""
    class _Message:
        pass

    class _Choice:
        pass

    class _Completion:
        pass

    message = _Message()
    message.content = answer
    choice = _Choice()
    choice.message = message
    choice.finish_reason = finish_reason
    completion = _Completion()
    completion.choices = [choice]
    return completion


def _status_error(error_cls, code):
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    return error_cls(f"error {code}", response=httpx.Response(code, request=request), body=None)


class _FakeCompletions:
    """Returns queued results (or raises queued exceptions) per create call."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.results:
            raise RuntimeError("no more queued fake results")
        item = self.results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def make_service(monkeypatch):
    """DeepSeekService whose client records constructor kwargs + create calls."""

    def _make(results):
        class _RecordingAsyncOpenAI:
            instances = []

            def __init__(self, **kwargs):
                self.constructor_kwargs = kwargs
                self.completions = _FakeCompletions(results)
                self.chat = type("_Chat", (), {"completions": self.completions})()
                _RecordingAsyncOpenAI.instances.append(self)

        monkeypatch.setattr(
            "app.services.deepseek.AsyncOpenAI", _RecordingAsyncOpenAI
        )
        service = DeepSeekService(Settings(deepseek_api_key="test-key"))
        return service, _RecordingAsyncOpenAI.instances[-1]

    return _make


def _sample_request(**overrides):
    base = dict(
        message=PROMPT,
        response_format=ResponseFormat(type="json_object"),
        max_tokens=150,
        stop_sequence="<END>",
    )
    base.update(overrides)
    return CompareRequest(**base)


def test_compare_makes_exactly_two_provider_calls(make_service):
    service, client = make_service(
        [_completion("Свободный ответ..."), _completion('{"advantages": [...]}')]
    )

    response = asyncio.run(service.compare(_sample_request()))

    assert len(client.completions.calls) == 2, "comparison must make exactly 2 calls"
    assert response.unrestricted.answer == "Свободный ответ..."
    assert response.controlled.answer == '{"advantages": [...]}'


def test_compare_uses_same_original_prompt_for_both_calls(make_service):
    service, client = make_service(
        [_completion("A"), _completion("B")]
    )

    asyncio.run(service.compare(_sample_request()))

    unrestricted, controlled = client.completions.calls
    assert unrestricted["messages"][1] == {"role": "user", "content": PROMPT}
    assert controlled["messages"][1] == {"role": "user", "content": PROMPT}


def test_unrestricted_call_has_no_custom_controls(make_service):
    service, client = make_service([_completion("A"), _completion("B")])

    asyncio.run(service.compare(_sample_request()))

    unrestricted = client.completions.calls[0]
    # No response_format, no stop, no extra instruction message
    assert "response_format" not in unrestricted
    assert "stop" not in unrestricted
    assert len(unrestricted["messages"]) == 2
    # Only the app-level default cap (NOT the form's max_tokens)
    assert unrestricted["max_tokens"] == DEFAULT_MAX_TOKENS
    assert unrestricted["max_tokens"] != 150


def test_controlled_call_forwards_exact_response_format(make_service):
    service, client = make_service([_completion("A"), _completion("B")])

    asyncio.run(service.compare(_sample_request()))

    controlled = client.completions.calls[1]
    # The exact validated object must reach the mocked OpenAI client
    assert controlled["response_format"] == {"type": "json_object"}
    assert isinstance(controlled["response_format"], dict)


def test_controlled_call_receives_max_tokens(make_service):
    service, client = make_service([_completion("A"), _completion("B")])

    asyncio.run(service.compare(_sample_request(max_tokens=300)))

    assert client.completions.calls[1]["max_tokens"] == 300
    assert client.completions.calls[0]["max_tokens"] == DEFAULT_MAX_TOKENS


def test_controlled_call_receives_stop(make_service):
    service, client = make_service([_completion("A"), _completion("B")])

    asyncio.run(service.compare(_sample_request(stop_sequence="<END>")))

    assert client.completions.calls[1]["stop"] == ["<END>"]
    assert "stop" not in client.completions.calls[0]


def test_controlled_call_receives_termination_instruction(make_service):
    service, client = make_service([_completion("A"), _completion("B")])

    asyncio.run(service.compare(_sample_request(stop_sequence="<END>")))

    instruction = client.completions.calls[1]["messages"][2]["content"]
    assert "<END>" in instruction
    assert "маркер" in instruction
    # In JSON mode the marker is placed after the closing JSON brace
    assert "закрывающей фигурной скобки" in instruction


def test_text_mode_stop_uses_plain_termination_instruction(make_service):
    service, client = make_service([_completion("A"), _completion("B")])

    asyncio.run(service.compare(
        _sample_request(response_format=ResponseFormat(type="text"), stop_sequence="<END>")
    ))

    instruction = client.completions.calls[1]["messages"][2]["content"]
    assert "<END>" in instruction
    assert "Заверши ответ маркером" in instruction


def test_json_output_instruction_added_in_json_mode(make_service):
    """DeepSeek JSON Output expects the word 'json' in the messages."""
    service, client = make_service([_completion("A"), _completion("B")])

    asyncio.run(service.compare(_sample_request(response_format=ResponseFormat(type="json_object"))))

    instruction = client.completions.calls[1]["messages"][2]["content"]
    assert "json" in instruction.lower()
    # ...and the ORIGINAL prompt message is untouched
    assert client.completions.calls[1]["messages"][1]["content"] == PROMPT


def test_text_mode_has_no_json_mention(make_service):
    service, client = make_service([_completion("A"), _completion("B")])

    asyncio.run(service.compare(
        _sample_request(response_format=ResponseFormat(type="text"), stop_sequence=None)
    ))

    controlled = client.completions.calls[1]
    assert controlled["response_format"] == {"type": "text"}
    instruction = controlled["messages"][2]["content"]
    assert "json" not in instruction.lower()
    assert "Заверши ответ маркером" not in instruction


def test_no_stop_means_no_stop_param_and_no_termination_instruction(make_service):
    service, client = make_service([_completion("A"), _completion("B")])

    response = asyncio.run(service.compare(_sample_request(stop_sequence=None)))

    controlled = client.completions.calls[1]
    assert "stop" not in controlled
    instruction = controlled["messages"][2]["content"]
    assert "маркер" not in instruction
    assert response.controlled.settings.stop is None


def test_finish_reason_propagates_from_provider(make_service):
    service, _ = make_service([_completion("A", "stop"), _completion("B", "length")])

    response = asyncio.run(service.compare(_sample_request()))

    assert response.unrestricted.finish_reason == "stop"
    assert response.controlled.finish_reason == "length"


def test_controlled_finish_reason_stop(make_service):
    service, _ = make_service([_completion("A", "stop"), _completion("B", "stop")])
    response = asyncio.run(service.compare(_sample_request()))
    assert response.controlled.finish_reason == "stop"


def test_settings_echo_mirrors_api_parameters(make_service):
    service, _ = make_service([_completion("A"), _completion("B")])

    response = asyncio.run(service.compare(_sample_request()))

    settings = response.controlled.settings
    assert settings.response_format == {"type": "json_object"}
    assert settings.max_tokens == 150
    assert settings.stop == ["<END>"]
    assert response.unrestricted.settings is None


def test_partial_failure_controlled_side(make_service):
    service, _ = make_service(
        [_completion("Unrestricted OK"), _status_error(RateLimitError, 429)]
    )

    response = asyncio.run(service.compare(_sample_request()))

    assert response.unrestricted.answer == "Unrestricted OK"
    assert response.controlled is None
    assert response.controlled_error and "rate limit" in response.controlled_error.lower()


def test_unrestricted_failure_raises(make_service):
    service, _ = make_service(
        [_status_error(AuthenticationError, 401), _completion("never reached")]
    )

    with pytest.raises(DeepSeekAuthenticationError):
        asyncio.run(service.compare(_sample_request()))


def test_timeout_maps_to_deepseek_error(make_service):
    from openai import APITimeoutError

    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    service, _ = make_service(
        [APITimeoutError(request=request), _completion("never reached")]
    )
    from app.services.deepseek import DeepSeekTimeoutError

    with pytest.raises(DeepSeekTimeoutError):
        asyncio.run(service.compare(_sample_request()))


def test_network_failure_maps_to_deepseek_error(make_service):
    from openai import APIConnectionError

    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    service, _ = make_service(
        [APIConnectionError(message="refused", request=request), _completion("never reached")]
    )
    from app.services.deepseek import DeepSeekNetworkError

    with pytest.raises(DeepSeekNetworkError):
        asyncio.run(service.compare(_sample_request()))


def test_malformed_provider_response_raises(make_service):
    service, _ = make_service([_completion("", "stop"), _completion("never reached")])
    from app.services.deepseek import DeepSeekMalformedResponseError

    with pytest.raises(DeepSeekMalformedResponseError):
        asyncio.run(service.compare(_sample_request()))


def test_unexpected_provider_error_raises_deepseek_error_500(make_service):
    service, _ = make_service([RuntimeError("boom"), _completion("never reached")])

    with pytest.raises(DeepSeekError) as excinfo:
        asyncio.run(service.compare(_sample_request()))
    assert excinfo.value.status_code == 500


def test_compare_does_not_retry(make_service):
    """One failed controlled call must NOT trigger extra provider calls."""
    service, client = make_service(
        [_completion("Unrestricted OK"), _status_error(RateLimitError, 429)]
    )

    asyncio.run(service.compare(_sample_request()))

    assert len(client.completions.calls) == 2
