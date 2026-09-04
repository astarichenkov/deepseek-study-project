"""API-level tests for POST /api/temperature (provider calls fully mocked)."""
import pytest

from app.services.deepseek import DeepSeekAuthenticationError

PROMPT = "Объясни школьнику, что такое искусственный интеллект. Приведи один простой пример и одну запоминающуюся аналогию."
VALID = {"message": PROMPT, "temperature": 0.7, "max_tokens": 700, "stop_sequence": None}


@pytest.mark.parametrize("temp", [0, 0.0, 0.7, 1.2, 1.5])
def test_valid_temperatures_accepted(client, temp):
    r = client.post("/api/temperature", json={**VALID, "temperature": temp})
    assert r.status_code == 200
    assert r.json()["applied_parameters"]["temperature"] == float(temp)


@pytest.mark.parametrize("temp", [2.5, -0.1, 99, "abc", None])
def test_out_of_range_or_invalid_temperature_rejected(client, temp):
    assert client.post("/api/temperature", json={**VALID, "temperature": temp}).status_code == 422


def test_temperature_zero_provider_calls_on_validation(client, fake_service):
    bads = [
        {**VALID, "temperature": 3},
        {**VALID, "max_tokens": 5},
        {**VALID, "message": ""},
        {**VALID, "stop_sequence": "}"},
    ]
    for p in bads:
        client.post("/api/temperature", json=p)
    assert len(fake_service.temperature_calls) == 0


def test_request_reaches_service_with_expected_fields(client, fake_service):
    r = client.post("/api/temperature", json={**VALID, "temperature": 1.2})
    assert r.status_code == 200
    req = fake_service.temperature_calls[0]
    assert req.temperature == 1.2
    assert req.max_tokens == 700
    assert req.message == PROMPT
    assert req.stop_sequence is None


def test_nonempty_stop_passed(client, fake_service):
    r = client.post("/api/temperature", json={**VALID, "stop_sequence": "STOP"})
    assert r.status_code == 200
    assert fake_service.temperature_calls[0].stop_sequence == "STOP"
    assert r.json()["applied_parameters"]["stop"] == ["STOP"]


def test_provider_failure_maps_401(client, fake_service):
    fake_service.raise_error = DeepSeekAuthenticationError()
    assert client.post("/api/temperature", json=VALID).status_code == 401


def test_error_never_leaks_secrets(client, fake_service, settings):
    fake_service.raise_error = DeepSeekAuthenticationError()
    raw = client.post("/api/temperature", json=VALID).text
    assert settings.deepseek_api_key not in raw
    assert "Traceback" not in raw
