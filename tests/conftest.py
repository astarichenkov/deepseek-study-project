"""Shared fixtures for the test suite.

No test in this suite ever touches the real DeepSeek API: every normal
DeepSeek call is replaced either by dependency override (API tests) or by
mocking the OpenAI client (service unit tests). The only exception is the
explicitly-marked integration smoke test (``pytest -m integration``).
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import get_deepseek_service
from app.config import Settings, get_settings
from app.main import create_app
from app.schemas.chat import ChatResponse


class FakeDeepSeekService:
    """In-memory stand-in for ``DeepSeekService``.

    Records the last message it received and can be told to raise a
    specific exception to exercise error handling in the API layer.
    """

    def __init__(self, answer: str = "Mocked answer from DeepSeek.") -> None:
        self.answer = answer
        self.last_message: str | None = None
        self.raise_error: Exception | None = None

    async def chat(self, message: str) -> ChatResponse:
        self.last_message = message
        if self.raise_error is not None:
            raise self.raise_error
        return ChatResponse(answer=self.answer)


@pytest.fixture
def settings() -> Settings:
    return Settings(deepseek_api_key="test-key", environment="test")


@pytest.fixture
def fake_service() -> FakeDeepSeekService:
    return FakeDeepSeekService()


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings=settings)


@pytest.fixture
def client(app: FastAPI, settings: Settings, fake_service: FakeDeepSeekService):
    """TestClient with the DeepSeek service swapped for a fake."""
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_deepseek_service] = lambda: fake_service
    # Starlette 1.x re-raises handled server errors by design; the app ships
    # a global error handler, so capture its response instead.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()
