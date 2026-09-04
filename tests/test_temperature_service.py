"""Service-level tests for Day 4 temperature (provider fully mocked)."""
import asyncio

import httpx
import pytest
from openai import RateLimitError

from app.config import Settings
from app.schemas.temperature import TemperatureRequest
from app.services.deepseek import (
    DeepSeekMalformedResponseError,
    DeepSeekRateLimitError,
    DeepSeekService,
)

PROMPT = "Объясни школьнику, что такое искусственный интеллект."


def _completion(content, finish_reason="stop", usage=None):
    class _M:
        pass

    class _C:
        pass

    class _R:
        pass

    m = _M(); m.content = content
    c = _C(); c.message = m; c.finish_reason = finish_reason
    r = _R(); r.choices = [c]; r.usage = usage
    return r


def _usage(pt=10, ct=20, tt=30):
    class U:
        pass

    u = U(); u.prompt_tokens = pt; u.completion_tokens = ct; u.total_tokens = tt
    return u


def _status_error(cls, code):
    req = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    return cls(f"err {code}", response=httpx.Response(code, request=req), body=None)


class _FakeCompletions:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def make_service(monkeypatch):
    def _make(results):
        class _Rec:
            instances = []

            def __init__(self, **kwargs):
                self.completions = _FakeCompletions(results)
                self.chat = type("_Chat", (), {"completions": self.completions})()
                _Rec.instances.append(self)

        monkeypatch.setattr("app.services.deepseek.AsyncOpenAI", _Rec)
        return DeepSeekService(Settings(deepseek_api_key="test-key")), _Rec.instances[-1]

    return _make


def _req(temperature=0.7, **over):
    base = dict(message=PROMPT, temperature=temperature, max_tokens=700, stop_sequence=None)
    base.update(over)
    return TemperatureRequest(**base)


def run(coro):
    return asyncio.run(coro)


def test_temperature_passed_to_provider_exactly(make_service):
    service, client = make_service([_completion("a"), _completion("b"), _completion("c")])
    for temp in (0.0, 0.7, 1.2):
        run(service.complete_with_temperature(_req(temp)))
        call = client.completions.calls[-1]
        assert call["temperature"] == temp
        assert call["max_tokens"] == 700
        assert "response_format" not in call


def test_same_message_for_all_temperatures(make_service):
    service, client = make_service([_completion("a"), _completion("b"), _completion("c")])
    for temp in (0.0, 0.7, 1.2):
        run(service.complete_with_temperature(_req(temp)))
    msgs = [c["messages"][1]["content"] for c in client.completions.calls]
    assert msgs == [PROMPT, PROMPT, PROMPT]  # same prompt, only temperature differs


def test_max_tokens_passed(make_service):
    service, client = make_service([_completion("x")])
    run(service.complete_with_temperature(_req(0.5, max_tokens=500)))
    assert client.completions.calls[0]["max_tokens"] == 500


def test_empty_stop_omitted_and_nonempty_passed(make_service):
    service, client = make_service([_completion("x"), _completion("x")])
    run(service.complete_with_temperature(_req(0.5)))
    assert "stop" not in client.completions.calls[0]
    run(service.complete_with_temperature(_req(0.5, stop_sequence="STOP")))
    assert client.completions.calls[1]["stop"] == ["STOP"]


def test_finish_reason_and_usage_propagated(make_service):
    service, _ = make_service([_completion("Ответ", "stop", _usage())])
    resp = run(service.complete_with_temperature(_req(1.2)))
    assert resp.finish_reason == "stop"
    assert resp.usage == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    assert resp.applied_parameters["temperature"] == 1.2
    assert resp.applied_parameters["max_tokens"] == 700
    assert resp.applied_parameters["stop"] is None


def test_empty_response_raises_malformed(make_service):
    service, _ = make_service([_completion("", "stop")])
    with pytest.raises(DeepSeekMalformedResponseError):
        run(service.complete_with_temperature(_req(0.7)))


def test_provider_rate_limit_maps(make_service):
    service, _ = make_service([_status_error(RateLimitError, 429)])
    with pytest.raises(DeepSeekRateLimitError):
        run(service.complete_with_temperature(_req(0.7)))


def test_day4_sends_thinking_disabled_extra_body(make_service):
    service, client = make_service([_completion("Ответ")])
    run(service.complete_with_temperature(_req(0.7)))
    call = client.completions.calls[0]
    assert call.get("extra_body") == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in call
    assert call["temperature"] == 0.7
    assert call["max_tokens"] == 700
