"""API-level tests for POST /api/compare (provider calls fully mocked).

Focus: request validation (incl. response_format whitelisting), HTTP
mapping and the response contract. The "exactly two provider calls with
the right parameters" behaviour is tested at the service level in
test_compare_service.py.
"""
import pytest

from app.schemas.compare import CompareResponse, CompareResult, ControlledSettings
from app.services.deepseek import DeepSeekAuthenticationError

VALID_PAYLOAD = {
    "message": "Назови три преимущества REST API.",
    "response_format": {"type": "json_object"},
    "max_tokens": 150,
    "stop_sequence": "<END>",
}


def test_compare_success(client, fake_service):
    response = client.post("/api/compare", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["unrestricted"]["answer"] == "Unrestricted answer."
    assert body["controlled"]["answer"] == "Controlled answer."
    assert body["unrestricted"]["finish_reason"] == "stop"
    assert body["controlled"]["finish_reason"] == "length"
    # Applied API parameters echoed on the controlled side AND at top level
    settings = body["controlled"]["settings"]
    assert settings["response_format"] == {"type": "json_object"}
    assert settings["max_tokens"] == 150
    assert settings["stop"] == ["<END>"]
    assert body["settings"] == settings
    assert body["unrestricted"]["settings"] is None
    assert body["controlled_error"] is None
    # Backend received exactly the validated request
    assert len(fake_service.compare_calls) == 1
    req = fake_service.compare_calls[0]
    assert req.message == VALID_PAYLOAD["message"]
    assert req.response_format.as_api_param() == {"type": "json_object"}
    assert req.max_tokens == 150
    assert req.stop_sequence == "<END>"


def test_compare_defaults_when_fields_missing(client):
    payload = {"message": "Назови три преимущества REST API."}
    response = client.post("/api/compare", json=payload)
    assert response.status_code == 200
    settings = response.json()["controlled"]["settings"]
    assert settings["response_format"] == {"type": "json_object"}
    assert settings["max_tokens"] == 150
    assert settings["stop"] is None


# ---------------------------------------------------------------------------
# response_format validation
# ---------------------------------------------------------------------------
def test_response_format_accepts_text(client):
    payload = {**VALID_PAYLOAD, "response_format": {"type": "text"}}
    assert client.post("/api/compare", json=payload).status_code == 200


@pytest.mark.parametrize(
    "response_format",
    [
        {"type": "json_schema"},          # OpenAI-only, not supported here
        {"type": "xml"},                  # unsupported type
        {"type": "json_object", "extra": 1},  # arbitrary SDK args rejected
        {"type": 123},
        "json_object",                    # must be an object
        42,
        [{"type": "json_object"}],
        {},
    ],
)
def test_response_format_invalid_rejected(client, response_format):
    payload = {**VALID_PAYLOAD, "response_format": response_format}
    assert client.post("/api/compare", json=payload).status_code == 422


# ---------------------------------------------------------------------------
# Other field validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "payload",
    [
        {**VALID_PAYLOAD, "message": ""},
        {**VALID_PAYLOAD, "message": "   "},
        {**VALID_PAYLOAD, "message": "a" * 4001},
        {**VALID_PAYLOAD, "message": 12345},
    ],
)
def test_compare_message_validation(client, payload):
    assert client.post("/api/compare", json=payload).status_code == 422


@pytest.mark.parametrize("max_tokens", [15, 2001, 0, -5, "abc", 1.5, None])
def test_compare_max_tokens_validation(client, max_tokens):
    payload = {**VALID_PAYLOAD, "max_tokens": max_tokens}
    assert client.post("/api/compare", json=payload).status_code == 422


def test_compare_stop_sequence_too_long(client):
    payload = {**VALID_PAYLOAD, "stop_sequence": "x" * 51}
    assert client.post("/api/compare", json=payload).status_code == 422


def test_compare_whitespace_stop_sequence_becomes_none(client):
    payload = {**VALID_PAYLOAD, "stop_sequence": "   "}
    response = client.post("/api/compare", json=payload)
    assert response.status_code == 200
    assert response.json()["controlled"]["settings"]["stop"] is None


def test_validation_failure_causes_zero_service_calls(client, fake_service):
    """Invalid payloads must be rejected before any service/provider call."""
    bad_payloads = [
        {**VALID_PAYLOAD, "response_format": {"type": "json_schema"}},
        {**VALID_PAYLOAD, "max_tokens": 5},
        {**VALID_PAYLOAD, "message": ""},
    ]
    for payload in bad_payloads:
        client.post("/api/compare", json=payload)
    assert len(fake_service.compare_calls) == 0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
def test_compare_provider_failure_maps_to_http(client, fake_service):
    fake_service.raise_error = DeepSeekAuthenticationError()
    response = client.post("/api/compare", json=VALID_PAYLOAD)
    assert response.status_code == 401
    assert "detail" in response.json()


def test_compare_partial_failure_returns_200_with_controlled_error(client, fake_service):
    fake_service.compare_result = CompareResponse(
        unrestricted=CompareResult(answer="Unrestricted answer.", finish_reason="stop"),
        settings=ControlledSettings(
            response_format={"type": "json_object"},
            max_tokens=150,
            stop=["<END>"],
        ),
        controlled=None,
        controlled_error="DeepSeek rate limit exceeded. Please wait a moment and try again.",
    )
    response = client.post("/api/compare", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["unrestricted"]["answer"] == "Unrestricted answer."
    assert body["controlled"] is None
    assert "rate limit" in body["controlled_error"]
    # Requested/applied parameters must survive a controlled failure
    assert body["settings"]["response_format"] == {"type": "json_object"}
    assert body["settings"]["max_tokens"] == 150
    assert body["settings"]["stop"] == ["<END>"]
    # No fabricated finish_reason on the failed side
    assert "finish_reason" not in (body["controlled"] or {})


def test_compare_unexpected_error_returns_generic_500(client, fake_service):
    fake_service.raise_error = RuntimeError("secret internal detail")
    response = client.post("/api/compare", json=VALID_PAYLOAD)
    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "Internal server error. Please try again later."
    assert "secret internal detail" not in body["detail"]


def test_compare_error_never_leaks_secrets(client, fake_service, settings):
    fake_service.raise_error = DeepSeekAuthenticationError()
    response = client.post("/api/compare", json=VALID_PAYLOAD)
    raw = response.text
    assert settings.deepseek_api_key not in raw
    assert "Traceback" not in raw
