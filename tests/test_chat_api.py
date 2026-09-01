"""API-level tests for POST /api/chat (DeepSeek calls are fully mocked)."""
import pytest

from app.schemas.chat import MAX_MESSAGE_LENGTH
from app.services.deepseek import (
    DeepSeekAuthenticationError,
    DeepSeekError,
    DeepSeekMalformedResponseError,
    DeepSeekNetworkError,
    DeepSeekRateLimitError,
    DeepSeekTimeoutError,
)


def test_chat_success(client, fake_service):
    response = client.post("/api/chat", json={"message": "What is dependency injection?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Mocked answer from DeepSeek."
    assert fake_service.last_message == "What is dependency injection?"


@pytest.mark.parametrize(
    "payload",
    [
        {"message": ""},
        {"message": "   "},
        {"message": "\t\n"},
        {"message": " \n\t "},
    ],
)
def test_chat_rejects_empty_or_whitespace_only(client, payload):
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_chat_rejects_too_long_message(client):
    payload = {"message": "a" * (MAX_MESSAGE_LENGTH + 1)}
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 422


def test_chat_accepts_max_length_message(client, fake_service):
    payload = {"message": "a" * MAX_MESSAGE_LENGTH}
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200


def test_chat_rejects_missing_field(client):
    response = client.post("/api/chat", json={})
    assert response.status_code == 422


def test_chat_rejects_non_string_message(client):
    response = client.post("/api/chat", json={"message": 12345})
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (DeepSeekAuthenticationError(), 401),
        (DeepSeekRateLimitError(), 429),
        (DeepSeekTimeoutError(), 504),
        (DeepSeekNetworkError(), 502),
        (DeepSeekMalformedResponseError(), 502),
        (DeepSeekError("generic provider error", 502), 502),
    ],
)
def test_chat_maps_service_errors_to_http(client, fake_service, error, expected_status):
    fake_service.raise_error = error
    response = client.post("/api/chat", json={"message": "hello"})
    assert response.status_code == expected_status
    body = response.json()
    assert "detail" in body
    assert body["detail"]  # non-empty, user-friendly message


def test_chat_error_body_does_not_leak_internals(client, fake_service):
    fake_service.raise_error = DeepSeekAuthenticationError()
    response = client.post("/api/chat", json={"message": "hello"})
    body = response.json()
    assert "sk-" not in body["detail"]
    assert "Traceback" not in body["detail"]
    assert "test-key" not in body["detail"]


def test_chat_unexpected_error_returns_generic_500(client, fake_service):
    fake_service.raise_error = RuntimeError("secret internal detail")
    response = client.post("/api/chat", json={"message": "hello"})
    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "Internal server error. Please try again later."
    assert "secret internal detail" not in body["detail"]
    assert "Traceback" not in body["detail"]
