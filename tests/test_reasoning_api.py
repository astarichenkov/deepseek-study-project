"""API-level tests for POST /api/reasoning (provider calls fully mocked)."""
import pytest

from app.schemas.reasoning import ReasoningResponse
from app.services.grading import DEFAULT_REASONING_TASK
from app.services.deepseek import DeepSeekAuthenticationError

TASK = DEFAULT_REASONING_TASK
VALID = {
    "method": "direct",
    "task": TASK,
    "max_tokens": 1000,
    "stop_sequence": None,
}


def test_reasoning_direct_valid(client, fake_service):
    r = client.post("/api/reasoning", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert body["method"] == "direct"
    assert body["solution"]
    assert len(fake_service.reasoning_calls) == 1
    req = fake_service.reasoning_calls[0]
    assert req.task == TASK
    assert req.max_tokens == 1000


def test_reasoning_generate_prompt_returns_prompt(client):
    r = client.post("/api/reasoning", json={**VALID, "method": "generate_prompt"})
    assert r.status_code == 200
    assert r.json()["kind"] == "generated_prompt"
    assert r.json()["generated_prompt"]


def test_reasoning_use_prompt_requires_generated_prompt(client, fake_service):
    payload = {**VALID, "method": "use_prompt", "generated_prompt": None}
    assert client.post("/api/reasoning", json=payload).status_code == 422
    payload["generated_prompt"] = "   "
    assert client.post("/api/reasoning", json=payload).status_code == 422
    # a valid one is accepted
    ok = {**VALID, "method": "use_prompt", "generated_prompt": "Реши: " + TASK}
    assert client.post("/api/reasoning", json=ok).status_code == 200
    assert len(fake_service.reasoning_calls) == 1


# ---------- validation -> zero provider calls ----------
@pytest.mark.parametrize("task", ["", "   ", "\t\n", "a" * 4001])
def test_reasoning_task_validation(client, task):
    assert client.post("/api/reasoning", json={**VALID, "task": task}).status_code == 422


@pytest.mark.parametrize("max_tokens", [15, 2001, "abc", -1])
def test_reasoning_max_tokens_validation(client, max_tokens):
    assert client.post("/api/reasoning", json={**VALID, "max_tokens": max_tokens}).status_code == 422


@pytest.mark.parametrize("bad_stop", ["x", "}", "###"])
def test_reasoning_stop_validation(client, bad_stop):
    assert client.post("/api/reasoning", json={**VALID, "stop_sequence": bad_stop}).status_code == 422


def test_reasoning_validation_zero_provider_calls(client, fake_service):
    bads = [
        {**VALID, "task": ""},
        {**VALID, "max_tokens": 5},
        {**VALID, "method": "use_prompt", "generated_prompt": None},
    ]
    for p in bads:
        client.post("/api/reasoning", json=p)
    assert len(fake_service.reasoning_calls) == 0


# ---------- errors ----------
def test_reasoning_provider_failure_maps_401(client, fake_service):
    fake_service.raise_error = DeepSeekAuthenticationError()
    assert client.post("/api/reasoning", json=VALID).status_code == 401


def test_reasoning_status_propagated(client, fake_service):
    fake_service.reasoning_result = ReasoningResponse(
        method="direct", kind="solution", prompt_sent="p", solution="a",
        finish_reason="length", status="incorrect",
    )
    body = client.post("/api/reasoning", json=VALID).json()
    assert body["status"] == "incorrect"
    assert body["finish_reason"] == "length"


def test_reasoning_error_never_leaks_secrets(client, fake_service, settings):
    fake_service.raise_error = DeepSeekAuthenticationError()
    raw = client.post("/api/reasoning", json=VALID).text
    assert settings.deepseek_api_key not in raw
    assert "Traceback" not in raw
