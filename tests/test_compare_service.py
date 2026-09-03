"""Service-level tests for the compare flow with a fully mocked client.

Verifies the CORE contract of the assignment:

* exactly TWO provider calls per comparison;
* the SAME original user prompt is used for both calls;
* the unrestricted call has NO response_format / custom max_tokens / stop /
  JSON-structure / termination instruction;
* the controlled call forwards ``response_format={"type": "json_object"}``
  and the user ``max_tokens``;
* stop is OPTIONAL: omitted (not sent) when empty, passed as ``stop=[...]``
  only when the user supplies a sequence;
* the controlled instruction does NOT inject an artificial END marker;
* finish_reason comes from the (fake) provider response.

No network traffic happens here.
"""
import json

import httpx
import pytest
from openai import AuthenticationError, RateLimitError

from app.config import Settings
from app.schemas.compare import CompareRequest
from app.services.deepseek import (
    DEFAULT_MAX_TOKENS,
    DeepSeekAuthenticationError,
    DeepSeekService,
)

PROMPT = "Напиши список основных продуктов для приготовления борща на 4 порции."
DEFAULT_STRUCTURE = {
    "products": [
        {
            "name": "Название продукта",
            "count": "Количество",
            "unit": "Единица измерения",
        }
    ]
}
CUSTOM_STRUCTURE = {
    "ingredients": [
        {
            "product": "Название",
            "amount": "Количество",
            "unit": "Единица измерения",
            "optional": "Необязательный ингредиент",
        }
    ]
}


def _completion(answer: str | None, finish_reason: str | None = "stop"):
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
    def _make(results):
        class _RecordingAsyncOpenAI:
            instances = []

            def __init__(self, **kwargs):
                self.constructor_kwargs = kwargs
                self.completions = _FakeCompletions(results)
                self.chat = type("_Chat", (), {"completions": self.completions})()
                _RecordingAsyncOpenAI.instances.append(self)

        monkeypatch.setattr("app.services.deepseek.AsyncOpenAI", _RecordingAsyncOpenAI)
        service = DeepSeekService(Settings(deepseek_api_key="test-key"))
        return service, _RecordingAsyncOpenAI.instances[-1]

    return _make


def _sample_request(**overrides):
    base = dict(
        message=PROMPT,
        json_structure=DEFAULT_STRUCTURE,
        max_tokens=500,
        stop_sequence=None,
    )
    base.update(overrides)
    return CompareRequest(**base)


def run(coro):
    import asyncio

    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Two calls / same prompt
# ---------------------------------------------------------------------------
def test_compare_makes_exactly_two_provider_calls(make_service):
    service, client = make_service(
        [_completion("Свободный текст..."), _completion('{"products": []}')]
    )
    response = run(service.compare(_sample_request()))
    assert len(client.completions.calls) == 2
    assert response.unrestricted.answer == "Свободный текст..."
    assert response.controlled.answer == '{"products": []}'


def test_compare_uses_same_original_prompt_in_both(make_service):
    service, client = make_service([_completion("A"), _completion("B")])
    run(service.compare(_sample_request()))
    unrestricted, controlled = client.completions.calls
    assert unrestricted["messages"][1] == {"role": "user", "content": PROMPT}
    assert controlled["messages"][1]["content"].startswith(PROMPT)
    assert PROMPT in controlled["messages"][1]["content"]


def test_unrestricted_call_has_no_controls(make_service):
    service, client = make_service([_completion("A"), _completion("B")])
    run(service.compare(_sample_request()))
    unrestricted = client.completions.calls[0]
    assert "response_format" not in unrestricted
    assert "stop" not in unrestricted
    assert len(unrestricted["messages"]) == 2
    assert unrestricted["max_tokens"] == DEFAULT_MAX_TOKENS


# ---------------------------------------------------------------------------
# Controlled controls
# ---------------------------------------------------------------------------
def test_controlled_call_forwards_json_mode(make_service):
    service, client = make_service([_completion("A"), _completion("B")])
    run(service.compare(_sample_request()))
    assert client.completions.calls[1]["response_format"] == {"type": "json_object"}


def test_controlled_call_forwards_user_max_tokens(make_service):
    service, client = make_service([_completion("A"), _completion("B")])
    run(service.compare(_sample_request(max_tokens=400)))
    assert client.completions.calls[1]["max_tokens"] == 400
    assert client.completions.calls[0]["max_tokens"] == DEFAULT_MAX_TOKENS


def test_default_controlled_request_omits_stop(make_service):
    """stop is optional and empty by default -> the provider is NOT sent stop."""
    service, client = make_service([_completion("A"), _completion("B")])
    run(service.compare(_sample_request()))  # stop_sequence=None
    assert "stop" not in client.completions.calls[1]
    assert "stop" not in client.completions.calls[0]


def test_non_empty_stop_is_passed_to_provider(make_service):
    service, client = make_service([_completion("A"), _completion("B")])
    run(service.compare(_sample_request(stop_sequence="###END###")))
    assert client.completions.calls[1]["stop"] == ["###END###"]
    assert "stop" not in client.completions.calls[0]


def test_whitespace_stop_is_omitted(make_service):
    service, client = make_service([_completion("A"), _completion("B")])
    run(service.compare(_sample_request(stop_sequence="   ")))
    assert "stop" not in client.completions.calls[1]


# ---------------------------------------------------------------------------
# Instruction content
# ---------------------------------------------------------------------------
def _controlled_instruction(make_service):
    service, client = make_service([_completion("A"), _completion("B")])
    run(service.compare(_sample_request()))
    return client.completions.calls[1]["messages"][1]["content"]


def test_controlled_instruction_mentions_json_and_structure(make_service):
    instruction = _controlled_instruction(make_service)
    assert "JSON" in instruction  # DeepSeek json mode requires the word "json"
    assert '"products"' in instruction
    assert '"name"' in instruction
    assert '"count"' in instruction
    assert '"unit"' in instruction


def test_controlled_instruction_bounds_the_list(make_service):
    instruction = _controlled_instruction(make_service)
    assert "максимум 7" in instruction


def test_controlled_instruction_has_no_artificial_end_marker(make_service):
    """Default JSON must have a natural ending — no END marker injected."""
    instruction = _controlled_instruction(make_service)
    assert "END_OF_RESPONSE" not in instruction
    assert "сгенерируй маркер" not in instruction
    assert "После завершения JSON" not in instruction


def test_custom_json_fields_embedded(make_service):
    service, client = make_service([_completion("A"), _completion("B")])
    run(service.compare(_sample_request(json_structure=CUSTOM_STRUCTURE)))
    instruction = client.completions.calls[1]["messages"][1]["content"]
    assert "ingredients" in instruction
    assert '"product"' in instruction
    assert '"optional"' in instruction


def test_json_structure_serialized_for_instruction(make_service):
    instruction = _controlled_instruction(make_service)
    expected = json.dumps(DEFAULT_STRUCTURE, ensure_ascii=False, indent=2)
    assert expected in instruction


# ---------------------------------------------------------------------------
# finish_reason
# ---------------------------------------------------------------------------
def test_finish_reason_propagates(make_service):
    service, _ = make_service([_completion("A", "stop"), _completion("B", "length")])
    response = run(service.compare(_sample_request()))
    assert response.unrestricted.finish_reason == "stop"
    assert response.controlled.finish_reason == "length"


def test_controlled_finish_reason_stop(make_service):
    service, _ = make_service([_completion("A", "stop"), _completion("B", "stop")])
    assert run(service.compare(_sample_request())).controlled.finish_reason == "stop"


def test_settings_echo_mirrors_api(make_service):
    service, _ = make_service([_completion("A"), _completion("B")])
    response = run(service.compare(_sample_request(max_tokens=500, stop_sequence="STOP")))
    s = response.controlled.settings
    assert s.response_format == {"type": "json_object"}
    assert s.max_tokens == 500
    assert s.stop == ["STOP"]
    assert s.json_structure == DEFAULT_STRUCTURE
    assert response.settings == s
    assert response.unrestricted.settings is None


# ---------------------------------------------------------------------------
# Result passthrough / provider anomalies
# ---------------------------------------------------------------------------
def test_controlled_valid_json_answer_passthrough(make_service):
    raw = '{"products":[{"name":"Свёкла","count":"1","unit":"шт"}]}'
    service, _ = make_service([_completion("текст", "stop"), _completion(raw, "stop")])
    response = run(service.compare(_sample_request()))
    assert response.controlled.answer == raw


def test_controlled_invalid_json_answer_passthrough(make_service):
    raw = '{"products": [unclosed'
    service, _ = make_service([_completion("текст", "stop"), _completion(raw, "stop")])
    response = run(service.compare(_sample_request()))
    assert response.controlled.answer == raw
    assert response.controlled.finish_reason == "stop"


def test_controlled_empty_content_yields_partial_failure(make_service):
    service, _ = make_service([_completion("Unrestricted OK"), _completion("", "length")])
    response = run(service.compare(_sample_request()))
    assert response.unrestricted.answer == "Unrestricted OK"
    assert response.controlled is None
    assert response.controlled_error


def test_controlled_none_content_yields_partial_failure(make_service):
    service, _ = make_service([_completion("Unrestricted OK"), _completion(None, "stop")])
    response = run(service.compare(_sample_request()))
    assert response.unrestricted.answer == "Unrestricted OK"
    assert response.controlled is None


def test_content_none_with_stop_is_still_malformed(make_service):
    """content=None with stop configured is still a malformed provider
    response (partial failure), never fabricated."""
    service, _ = make_service(
        [
            _completion("Unrestricted OK", "stop"),
            _completion(None, "stop"),  # stop-configured controlled call
        ]
    )
    response = run(service.compare(_sample_request(stop_sequence="STOP")))
    assert response.unrestricted.answer == "Unrestricted OK"
    assert response.controlled is None
    assert response.controlled_error
    assert response.settings.stop == ["STOP"]


def test_valid_stop_terminated_response_is_accepted(make_service):
    """A non-empty provider response with a stop-configured request is accepted
    normally (not treated as malformed just because stop is set)."""
    service, _ = make_service(
        [
            _completion("Unrestricted OK", "stop"),
            _completion('{"products": []}', "stop"),
        ]
    )
    response = run(service.compare(_sample_request(stop_sequence="STOP")))
    assert response.controlled is not None
    assert response.controlled.answer == '{"products": []}'
    assert response.controlled.finish_reason == "stop"


def test_requested_controls_survive_controlled_failure(make_service):
    service, _ = make_service(
        [_completion("Unrestricted OK"), _status_error(RateLimitError, 429)]
    )
    response = run(service.compare(_sample_request(json_structure=CUSTOM_STRUCTURE, stop_sequence="STOP")))
    assert response.controlled is None
    assert response.controlled_error
    assert response.settings.response_format == {"type": "json_object"}
    assert response.settings.max_tokens == 500
    assert response.settings.stop == ["STOP"]
    assert response.settings.json_structure == CUSTOM_STRUCTURE


def test_unrestricted_failure_raises(make_service):
    service, _ = make_service([_status_error(AuthenticationError, 401), _completion("never")])
    with pytest.raises(DeepSeekAuthenticationError):
        run(service.compare(_sample_request()))


def test_unexpected_provider_error_raises_500(make_service):
    service, _ = make_service([RuntimeError("boom"), _completion("never")])
    from app.services.deepseek import DeepSeekError

    with pytest.raises(DeepSeekError) as excinfo:
        run(service.compare(_sample_request()))
    assert excinfo.value.status_code == 500


def test_compare_does_not_retry(make_service):
    service, client = make_service(
        [_completion("Unrestricted OK"), _status_error(RateLimitError, 429)]
    )
    run(service.compare(_sample_request()))
    assert len(client.completions.calls) == 2
