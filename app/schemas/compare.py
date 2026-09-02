"""Pydantic schemas for the response-control comparison API.

The comparison sends the SAME user prompt twice:

* ``unrestricted`` — no custom controls (only the application's normal
  system message and its default token cap);
* ``controlled``   — with the REAL API response-control parameters:
  ``response_format`` (structured output), ``max_tokens`` (output-token
  limit) and ``stop`` (stop sequence).

``response_format`` is a validated JSON object that is forwarded to the
DeepSeek API as the actual ``response_format`` parameter — it is NOT a
prompt instruction.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.chat import MAX_MESSAGE_LENGTH, ensure_not_blank

MAX_STOP_SEQUENCE_LENGTH = 50

MIN_MAX_TOKENS = 16
MAX_MAX_TOKENS = 2000
DEFAULT_MAX_TOKENS = 150


class ResponseFormat(BaseModel):
    """Validated ``response_format`` API configuration.

    Only the mechanisms actually supported by the DeepSeek
    OpenAI-compatible endpoint are whitelisted:

    * ``{"type": "json_object"}`` — JSON Output mode (the model is guided
      to produce a valid JSON object; DeepSeek/OpenAI recommend the word
      "json" appear in the messages);
    * ``{"type": "text"}``       — default free-text output.

    ``json_schema`` (OpenAI-only) and arbitrary extra keys are rejected,
    so no unsupported/arbitrary SDK arguments can be injected.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["json_object", "text"] = Field(
        ...,
        description=(
            "Механизм формата ответа: 'json_object' (JSON Output) или 'text'."
        ),
    )

    def as_api_param(self) -> dict:
        """The exact dict forwarded to ``client.chat.completions.create``."""
        return {"type": self.type}


class CompareRequest(BaseModel):
    """Payload for ``POST /api/compare``."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
        description="Оригинальный запрос — один и тот же для обоих вызовов.",
    )
    response_format: ResponseFormat = Field(
        default_factory=lambda: ResponseFormat(type="json_object"),
        description=(
            "Реальная API-конфигурация формата ответа "
            "(передаётся как response_format в DeepSeek API)."
        ),
    )
    max_tokens: int = Field(
        DEFAULT_MAX_TOKENS,
        ge=MIN_MAX_TOKENS,
        le=MAX_MAX_TOKENS,
        description="Лимит выходных токенов контролируемого запроса (API max_tokens).",
    )
    stop_sequence: str | None = Field(
        None,
        description=(
            "Стоп-последовательность контролируемого запроса (API stop). "
            "Пустая/пробельная строка трактуется как «не задана»."
        ),
    )

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        return ensure_not_blank(value)

    @field_validator("stop_sequence")
    @classmethod
    def clean_stop_sequence(cls, value: str | None) -> str | None:
        """Trim; blank becomes None; enforce length after trimming."""
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        if len(value) > MAX_STOP_SEQUENCE_LENGTH:
            raise ValueError(
                f"stop_sequence is too long (max {MAX_STOP_SEQUENCE_LENGTH} characters)"
            )
        return value


class ControlledSettings(BaseModel):
    """Echo of the actual API parameters applied to the controlled request."""

    response_format: dict
    max_tokens: int
    stop: list[str] | None = None


class CompareResult(BaseModel):
    """One side of the comparison.

    ``settings`` is set only on the controlled side and mirrors the actual
    API parameters that were sent.
    """

    answer: str
    finish_reason: str | None = None
    settings: ControlledSettings | None = None


class CompareResponse(BaseModel):
    """Result of one comparison operation (two provider calls).

    ``settings`` echoes the REQUESTED controlled parameters and is always
    present — the UI needs them even when the controlled provider call
    fails (they are known before the call).

    ``controlled`` is ``None`` (with a friendly ``controlled_error``) when
    the controlled request failed but the unrestricted one succeeded.
    """

    unrestricted: CompareResult
    settings: ControlledSettings
    controlled: CompareResult | None = None
    controlled_error: str | None = None
