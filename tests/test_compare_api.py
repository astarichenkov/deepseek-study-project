"""API-level tests for POST /api/compare (provider calls fully mocked)."""
import pytest

from app.schemas.compare import (
    DEFAULT_JSON_STRUCTURE,
    MAX_JSON_STRUCTURE_BYTES,
    CompareResponse,
    CompareResult,
    ControlledSettings,
)
from app.services.deepseek import DeepSeekAuthenticationError

VALID_PAYLOAD = {
    "message": "Напиши список основных продуктов для приготовления борща на 4 порции.",
    "json_structure": DEFAULT_JSON_STRUCTURE,
    "max_tokens": 500,
    "stop_sequence": None,
}


def test_default_borscht_comparison_request(client):
    response = client.post("/api/compare", json={"message": VALID_PAYLOAD["message"]})
    assert response.status_code == 200
    body = response.json()
    assert body["settings"]["json_structure"] == DEFAULT_JSON_STRUCTURE
    assert body["settings"]["max_tokens"] == 500
    assert body["settings"]["stop"] is None  # optional stop omitted by default


def test_compare_success(client, fake_service):
    response = client.post("/api/compare", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["unrestricted"]["answer"]
    assert body["controlled"]["answer"]
    settings = body["controlled"]["settings"]
    assert settings["response_format"] == {"type": "json_object"}
    assert settings["max_tokens"] == 500
    assert settings["stop"] is None
    assert settings["json_structure"] == DEFAULT_JSON_STRUCTURE
    assert body["settings"] == settings
    assert body["controlled_error"] is None
    req = fake_service.compare_calls[0]
    assert req.message == VALID_PAYLOAD["message"]
    assert req.max_tokens == 500
    assert req.stop_sequence is None


def test_non_empty_stop_passed_through(client, fake_service):
    payload = {**VALID_PAYLOAD, "stop_sequence": "STOP"}
    response = client.post("/api/compare", json=payload)
    assert response.status_code == 200
    assert response.json()["settings"]["stop"] == ["STOP"]
    assert fake_service.compare_calls[0].stop_sequence == "STOP"


def test_whitespace_stop_normalized_to_null(client):
    payload = {**VALID_PAYLOAD, "stop_sequence": "   "}
    response = client.post("/api/compare", json=payload)
    assert response.status_code == 200
    assert response.json()["settings"]["stop"] is None


def test_extra_response_format_is_ignored(client):
    payload = {**VALID_PAYLOAD, "response_format": {"type": "text"}}
    response = client.post("/api/compare", json=payload)
    assert response.status_code == 200
    assert response.json()["settings"]["response_format"] == {"type": "json_object"}


def test_custom_json_fields_accepted(client):
    custom = {"ingredients": [{"product": "Название", "amount": "Количество",
                               "unit": "Единица измерения", "optional": "Необязательный ингредиент"}]}
    response = client.post("/api/compare", json={**VALID_PAYLOAD, "json_structure": custom})
    assert response.status_code == 200
    assert response.json()["settings"]["json_structure"] == custom


# ---------------------------------------------------------------------------
# Validation (all 422 before any service call)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("message", ["", "   ", "\t\n", "a" * 4001, 12345])
def test_compare_message_validation(client, message):
    assert client.post("/api/compare", json={**VALID_PAYLOAD, "message": message}).status_code == 422


@pytest.mark.parametrize(
    "json_structure",
    [[], "products", 42, [{"a": 1}], {}],
)
def test_compare_json_structure_must_be_non_empty_object(client, json_structure):
    assert client.post("/api/compare", json={**VALID_PAYLOAD, "json_structure": json_structure}).status_code == 422


def test_compare_json_structure_too_large_rejected(client):
    huge = {"x": "a" * (MAX_JSON_STRUCTURE_BYTES + 50)}
    assert client.post("/api/compare", json={**VALID_PAYLOAD, "json_structure": huge}).status_code == 422


def test_compare_message_missing_rejected(client):
    payload = dict(VALID_PAYLOAD)
    payload.pop("message")
    assert client.post("/api/compare", json=payload).status_code == 422


@pytest.mark.parametrize("max_tokens", [15, 2001, 0, -5, "abc", 1.5])
def test_compare_max_tokens_validation(client, max_tokens):
    assert client.post("/api/compare", json={**VALID_PAYLOAD, "max_tokens": max_tokens}).status_code == 422


def test_compare_stop_sequence_too_long(client):
    assert client.post("/api/compare", json={**VALID_PAYLOAD, "stop_sequence": "x" * 51}).status_code == 422


@pytest.mark.parametrize("short_stop", ["x", "END", "}", "###", "$$$"])
def test_compare_short_or_symbol_only_stop_rejected(client, short_stop):
    """Too-short or symbol-only stop sequences can terminate inside JSON
    (verified against DeepSeek json mode) -> rejected with 422."""
    assert client.post("/api/compare", json={**VALID_PAYLOAD, "stop_sequence": short_stop}).status_code == 422


def test_compare_distinctive_stop_accepted(client):
    for stop in ["STOP", "###END###", "<STOP_JSON>"]:
        r = client.post("/api/compare", json={**VALID_PAYLOAD, "stop_sequence": stop})
        assert r.status_code == 200
        assert r.json()["settings"]["stop"] == [stop]


def test_validation_failure_causes_zero_service_calls(client, fake_service):
    bad_payloads = [
        {**VALID_PAYLOAD, "json_structure": []},
        {**VALID_PAYLOAD, "json_structure": {}},
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


def test_compare_partial_failure_keeps_requested_controls(client, fake_service):
    fake_service.compare_result = CompareResponse(
        unrestricted=CompareResult(answer="Unrestricted answer.", finish_reason="stop"),
        settings=ControlledSettings(
            response_format={"type": "json_object"},
            max_tokens=500,
            stop=None,
            json_structure=DEFAULT_JSON_STRUCTURE,
        ),
        controlled=None,
        controlled_error="DeepSeek returned an unexpected response. Please try again.",
    )
    response = client.post("/api/compare", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["unrestricted"]["answer"] == "Unrestricted answer."
    assert body["controlled"] is None
    assert "unexpected response" in body["controlled_error"]
    assert body["settings"]["response_format"] == {"type": "json_object"}
    assert body["settings"]["max_tokens"] == 500
    assert body["settings"]["json_structure"] == DEFAULT_JSON_STRUCTURE


def test_compare_output_limit_partial_surfaces_finish_reason(client, fake_service):
    fake_service.compare_result = CompareResponse(
        unrestricted=CompareResult(answer="Unrestricted answer.", finish_reason="stop"),
        settings=ControlledSettings(
            response_format={"type": "json_object"},
            max_tokens=150,
            stop=None,
            json_structure=DEFAULT_JSON_STRUCTURE,
        ),
        controlled=None,
        controlled_error="Ответ не поместился в заданный лимит max_tokens. Увеличьте максимальную длину ответа.",
        controlled_finish_reason="length",
    )
    response = client.post("/api/compare", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["controlled"] is None
    assert body["controlled_finish_reason"] == "length"
    assert "max_tokens" in body["controlled_error"]
    # requested parameters visible
    assert body["settings"]["max_tokens"] == 150
    assert body["settings"]["response_format"] == {"type": "json_object"}


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
