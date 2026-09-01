"""Tests for configuration loading and environment handling."""
import pytest

from app.config import Settings, get_settings
from app.schemas.chat import MAX_MESSAGE_LENGTH


def test_defaults():
    s = Settings(deepseek_api_key="")
    assert s.deepseek_base_url == "https://api.deepseek.com"
    assert s.deepseek_model == "deepseek-v4-flash"
    assert s.deepseek_timeout_seconds == 30.0
    assert s.max_message_length == MAX_MESSAGE_LENGTH
    assert s.system_prompt  # a system prompt is always configured
    assert s.has_deepseek_api_key is False


def test_reads_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123")
    s = Settings()
    assert s.deepseek_api_key == "sk-test-123"
    assert s.has_deepseek_api_key is True


def test_env_overrides_other_settings(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://custom.example.com")
    monkeypatch.setenv("DEEPSEEK_TIMEOUT_SECONDS", "7.5")
    s = Settings(deepseek_api_key="k")
    assert s.deepseek_model == "deepseek-chat"
    assert s.deepseek_base_url == "https://custom.example.com"
    assert s.deepseek_timeout_seconds == 7.5


def test_get_settings_is_cached():
    assert get_settings() is get_settings()


def test_get_settings_can_be_recomputed_after_env_change(monkeypatch):
    get_settings.cache_clear()
    try:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-cached-test")
        assert get_settings().deepseek_api_key == "sk-cached-test"
    finally:
        get_settings.cache_clear()


def test_deepseek_service_uses_settings(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("app.services.deepseek.AsyncOpenAI", FakeClient)

    s = Settings(
        deepseek_api_key="sk-client-test",
        deepseek_base_url="https://custom.example.com",
        deepseek_timeout_seconds=12.5,
    )
    from app.services.deepseek import DeepSeekService

    DeepSeekService(s)
    assert captured["api_key"] == "sk-client-test"
    assert captured["base_url"] == "https://custom.example.com"
    assert captured["timeout"] == 12.5


def test_placeholder_key_when_missing(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("app.services.deepseek.AsyncOpenAI", FakeClient)
    from app.services.deepseek import DeepSeekService

    DeepSeekService(Settings(deepseek_api_key=""))
    assert captured["api_key"] == "missing-api-key"
