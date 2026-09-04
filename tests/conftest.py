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
from app.schemas.compare import (
    FIXED_RESPONSE_FORMAT,
    CompareRequest,
    CompareResponse,
    CompareResult,
    ControlledSettings,
)
from app.schemas.reasoning import ReasoningRequest, ReasoningResponse
from app.schemas.temperature import TemperatureRequest, TemperatureResponse

class FakeDeepSeekService:
    """In-memory stand-in for ``DeepSeekService``.

    Records the last message it received and can be told to raise a
    specific exception to exercise error handling in the API layer.
    ``compare`` returns a canned comparison unless ``compare_result`` is
    set (e.g. to simulate a partial failure).
    """

    def __init__(self, answer: str = "Mocked answer from DeepSeek.") -> None:
        self.answer = answer
        self.last_message: str | None = None
        self.raise_error: Exception | None = None
        self.compare_calls: list[CompareRequest] = []
        self.compare_result: CompareResponse | None = None
        self.reasoning_calls: list[ReasoningRequest] = []
        self.reasoning_result: ReasoningResponse | None = None
        self.temperature_calls: list[TemperatureRequest] = []

    async def chat(self, message: str) -> ChatResponse:
        self.last_message = message
        if self.raise_error is not None:
            raise self.raise_error
        return ChatResponse(answer=self.answer)

    async def compare(self, request: CompareRequest) -> CompareResponse:
        self.compare_calls.append(request)
        if self.raise_error is not None:
            raise self.raise_error
        if self.compare_result is not None:
            return self.compare_result
        stop_list = [request.stop_sequence] if request.stop_sequence else None
        settings = ControlledSettings(
            response_format=dict(FIXED_RESPONSE_FORMAT),
            max_tokens=request.max_tokens,
            stop=stop_list,
            json_structure=request.json_structure,
        )
        return CompareResponse(
            unrestricted=CompareResult(
                answer="Unrestricted answer.", finish_reason="stop"
            ),
            settings=settings,
            controlled=CompareResult(
                answer='{"products": []}',
                finish_reason="length",
                settings=settings,
            ),
        )

    async def reasoning(self, request: ReasoningRequest) -> ReasoningResponse:
        self.reasoning_calls.append(request)
        if self.raise_error is not None:
            raise self.raise_error
        if self.reasoning_result is not None:
            return self.reasoning_result
        if request.method == "generate_prompt":
            return ReasoningResponse(
                method="generate_prompt",
                kind="generated_prompt",
                prompt_sent="stage1 prompt",
                generated_prompt="Реши задачу: " + request.task[:20],
                finish_reason="stop",
                usage={"completion_tokens": 40},
            )
        return ReasoningResponse(
            method=request.method,
            kind="solution",
            prompt_sent="prompt",
            solution="Ответ (mock)",
            finish_reason="stop",
            status="indeterminate",
        )

    async def complete_with_temperature(self, request: TemperatureRequest) -> TemperatureResponse:
        self.temperature_calls.append(request)
        if self.raise_error is not None:
            raise self.raise_error
        return TemperatureResponse(
            answer=self.answer,
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 30, "total_tokens": 40},
            applied_parameters={
                "model": "deepseek-v4-flash",
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stop": [request.stop_sequence] if request.stop_sequence else None,
            },
        )


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
