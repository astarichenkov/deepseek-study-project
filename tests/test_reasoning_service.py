"""Service-level tests for Day 3 reasoning strategies (provider fully mocked)."""
import asyncio

import httpx
import pytest
from openai import RateLimitError

from app.config import Settings
from app.schemas.reasoning import ReasoningRequest
from app.services.deepseek import (
    DeepSeekMalformedResponseError,
    DeepSeekRateLimitError,
    DeepSeekService,
)
from app.services.grading import DEFAULT_REASONING_TASK

TASK = DEFAULT_REASONING_TASK


def _completion(content, finish_reason="stop", usage=None):
    class _Message:
        pass

    class _Choice:
        pass

    class _Completion:
        pass

    msg = _Message(); msg.content = content
    ch = _Choice(); ch.message = msg; ch.finish_reason = finish_reason
    comp = _Completion(); comp.choices = [ch]; comp.usage = usage
    return comp


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


def _req(method, **overrides):
    base = dict(method=method, task=TASK, max_tokens=1000, stop_sequence=None)
    base.update(overrides)
    return ReasoningRequest(**base)


def run(coro):
    return asyncio.run(coro)


# ---------------- DIRECT ----------------
def test_direct_sends_original_task_only(make_service):
    service, client = make_service([_completion("Ответ 1")])
    resp = run(service.reasoning(_req("direct", stop_sequence="STOP")))
    call = client.completions.calls[0]
    assert call["messages"] == [{"role": "user", "content": TASK}]
    assert "response_format" not in call
    assert call["max_tokens"] == 1000
    assert call["stop"] == ["STOP"]
    assert resp.kind == "solution"
    assert resp.prompt_sent == TASK
    assert resp.solution == "Ответ 1"


def test_direct_omits_stop_when_empty(make_service):
    service, client = make_service([_completion("Ответ")])
    run(service.reasoning(_req("direct")))
    assert "stop" not in client.completions.calls[0]


# ---------------- STEP-BY-STEP ----------------
def test_step_by_step_includes_task_and_instruction(make_service):
    service, client = make_service([_completion("Ответ")])
    run(service.reasoning(_req("step_by_step")))
    content = client.completions.calls[0]["messages"][0]["content"]
    assert "пошагово" in content
    assert TASK in content  # same original task present
    assert "response_format" not in client.completions.calls[0]


# ---------------- GENERATE PROMPT ----------------
def test_generate_prompt_asks_for_prompt_and_includes_task(make_service):
    service, client = make_service([_completion("Твой улучшенный промпт")])
    resp = run(service.reasoning(_req("generate_prompt")))
    content = client.completions.calls[0]["messages"][0]["content"]
    assert "Составь эффективный промпт" in content
    assert "Не решай саму задачу" in content
    assert TASK in content
    assert resp.kind == "generated_prompt"
    assert resp.generated_prompt == "Твой улучшенный промпт"


# ---------------- USE PROMPT ----------------
def test_use_prompt_sends_generated_prompt_and_keeps_task(make_service):
    service, client = make_service([_completion("Ответ по промпту")])
    gp = f"Реши задачу.\n\nЗадача:\n{TASK}"
    resp = run(service.reasoning(_req("use_prompt", generated_prompt=gp)))
    assert len(client.completions.calls) == 1
    sent = client.completions.calls[0]["messages"][0]["content"]
    assert sent == gp  # prompt already contains the task -> sent as-is
    assert resp.prompt_sent == gp
    assert resp.solution == "Ответ по промпту"


def test_use_prompt_appends_task_when_missing(make_service):
    service, client = make_service([_completion("Ответ")])
    gp = "Просто реши логическую задачу."
    resp = run(service.reasoning(_req("use_prompt", generated_prompt=gp)))
    sent = client.completions.calls[0]["messages"][0]["content"]
    assert TASK in sent  # original task remains available
    assert resp.prompt_sent == sent


# ---------------- EXPERTS ----------------
def test_experts_one_call_with_roles_and_task(make_service):
    service, client = make_service([_completion("АНАЛИТИК:\n... ИТОГОВОЕ РЕШЕНИЕ: ...")])
    resp = run(service.reasoning(_req("experts")))
    assert len(client.completions.calls) == 1
    content = client.completions.calls[0]["messages"][0]["content"]
    assert TASK in content
    for token in ("Аналитик", "Инженер", "Критик", "ИТОГОВОЕ РЕШЕНИЕ"):
        assert token in content
    assert "response_format" not in client.completions.calls[0]
    assert resp.solution


# ---------------- COMMON ----------------
def test_finish_reason_and_usage_propagated(make_service):
    service, _ = make_service([_completion("Ответ", "stop", _usage())])
    resp = run(service.reasoning(_req("direct")))
    assert resp.finish_reason == "stop"
    assert resp.usage == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}


def test_usage_none_when_absent(make_service):
    service, _ = make_service([_completion("Ответ")])
    assert run(service.reasoning(_req("direct"))).usage is None


def test_default_task_graded(make_service):
    service, _ = make_service(
        [_completion("Борис — понедельник. Виктор — вторник. Анна — среда.")]
    )
    resp = run(service.reasoning(_req("direct")))
    assert resp.status == "correct"


def test_custom_task_not_graded(make_service):
    service, _ = make_service([_completion("любой ответ")])
    resp = run(service.reasoning(_req("direct", task="Моя своя задача на 5 строк о погоде.")))
    assert resp.status is None


def test_empty_response_raises_malformed(make_service):
    service, _ = make_service([_completion("", "stop")])
    with pytest.raises(DeepSeekMalformedResponseError):
        run(service.reasoning(_req("direct")))


def test_provider_rate_limit_maps(make_service):
    service, _ = make_service([_status_error(RateLimitError, 429)])
    with pytest.raises(DeepSeekRateLimitError):
        run(service.reasoning(_req("direct")))


def test_reasoning_sends_no_thinking_override(make_service):
    service, client = make_service([_completion("x")])
    run(service.reasoning(_req("direct")))
    call = client.completions.calls[0]
    assert "extra_body" not in call
    assert "reasoning_effort" not in call
