"""API-level tests for POST /api/compare (provider calls fully mocked).

Focus: request validation (incl. json_structure as a JSON object), HTTP
mapping and the response contract.
"""
import pytest

from app.schemas.compare import (
    DEFAULT_JSON_STRUCTURE,
    CompareResponse,
    CompareResult,
    ControlledSettings,
)
from app.services.deepseek import DeepSeekAuthenticationError

VALID_PAYLOAD = {
    "message": "Напиши список продуктов для приготовления борща на 4 порции.",
    "json_structure": DEFAULT_JSON_STRUCTURE,
    "max_tokens": 300,
    "stop_sequence": "END_OF_RESPONSE",
}


def test_default_borscht_comparison_request(client):
    """Only message sent -> defaults applied (json_structure, max_tokens)."""
    response = client.post("/api/compare", json={"message": VALID_PAYLOAD["message"]})
    assert response.status_code == 200
    body = response.json()
    assert body["settings"]["json_structure"] == DEFAULT_JSON_STRUCTURE
    assert body["settings"]["max_tokens"] == 800


def test_compare_success(client, fake_service):
    response = client.post("/api/compare", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["unrestricted"]["answer"] == "Unrestricted answer."
    assert body["unrestricted"]["finish_reason"] == "stop"
    assert body["controlled"]["answer"] == '{"products": []}'
    assert body["controlled"]["finish_reason"] == "length"
    # Controlled settings echo on controlled + top level
    assert body["controlled"]["settings"]["response_format"] == {"type": "json_object"}
    assert body["controlled"]["settings"]["max_tokens"] == 300
    assert body["controlled"]["settings"]["stop"] == ["END_OF_RESPONSE"]
    assert body["controlled"]["settings"]["json_structure"] == DEFAULT_JSON_STRUCTURE
    assert body["settings"] == body["controlled"]["settings"]
    assert body["controlled_error"] is None
    # backend received the validated request
    assert len(fake_service.compare_calls) == 1
    req = fake_service.compare_calls[0]
    assert req.message == VALID_PAYLOAD["message"]
    assert req.json_structure == DEFAULT_JSON_STRUCTURE
    assert req.max_tokens == 300
    assert req.stop_sequence == "END_OF_RESPONSE"


def test_extra_response_format_is_ignored(client):
    """response_format is fixed server-side; a client-supplied one is ignored."""
    payload = {**VALID_PAYLOAD, "response_format": {"type": "text"}}
    response = client.post("/api/compare", json=payload)
    assert response.status_code == 200
    assert response.json()["settings"]["response_format"] == {"type": "json_object"}


def test_custom_json_fields_accepted(client):
    custom = {
        "ingredients": [
            {
                "product": "Название",
                "amount": "Количество",
                "unit": "Единица измерения",
                "optional": "Обязательный ли ингредиент",
            }
        ]
    }
    payload = {**VALID_PAYLOAD, "json_structure": custom}
    response = client.post("/api/compare", json=payload)
    assert response.status_code == 200
    assert response.json()["settings"]["json_structure"] == custom


# ---------------------------------------------------------------------------
# Validation (all 422 before any service call)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "message", ["", "   ", "\t\n", "a" * 4001, 12345]
)
def test_compare_message_validation(client, message):
    assert client.post("/api/compare", json={**VALID_PAYLOAD, "message": message}).status_code == 422


@pytest.mark.parametrize(
    "json_structure",
    [
        [],                     # array, not object
        "products",             # string
        42,                     # scalar
        [{"a": 1}],             # array of objects
        {},                     # empty object
    ],
)
def test_compare_json_structure_must_be_non_empty_object(client, json_structure):
    payload = {**VALID_PAYLOAD, "json_structure": json_structure}
    assert client.post("/api/compare", json=payload).status_code == 422


def test_compare_message_missing_rejected(client):
    payload = dict(VALID_PAYLOAD)
    payload.pop("message")
    assert client.post("/api/compare", json=payload).status_code == 422


@pytest.mark.parametrize("max_tokens", [15, 2001, 0, -5, "abc", 1.5])
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
    assert response.json()["settings"]["stop"] is None


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
            max_tokens=300,
            stop=["END_OF_RESPONSE"],
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
    # requested controls visible on failure
    assert body["settings"]["response_format"] == {"type": "json_object"}
    assert body["settings"]["max_tokens"] == 300
    assert body["settings"]["stop"] == ["END_OF_RESPONSE"]
    assert body["settings"]["json_structure"] == DEFAULT_JSON_STRUCTURE


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
