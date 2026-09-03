"""Pydantic schemas for the response-control comparison API.

The comparison sends the SAME user prompt twice:

* ``unrestricted`` — no custom controls (only the application's normal
  system message and its default token cap);
* ``controlled``   — with REAL API response controls:

  * ``response_format={"type": "json_object"}`` — fixed by the application
    (JSON Output mode; the frontend never supplies it);
  * ``json_structure`` — user-editable JSON object describing the desired
    output structure, embedded in the controlled ``messages`` as an
    instruction;
  * ``max_tokens`` — output-token limit (real API parameter);
  * ``stop`` / ``stop_sequence`` — generation-termination marker (real API
    parameter) plus a dynamic instruction to emit the marker.

``response_format`` is NOT a prompt instruction and NOT arbitrary SDK JSON —
the UI exposes it read-only and the backend hard-codes
``{"type": "json_object"}``.
"""
import json

from pydantic import BaseModel, Field, field_validator

from app.schemas.chat import MAX_MESSAGE_LENGTH, ensure_not_blank

MAX_STOP_SEQUENCE_LENGTH = 50
MIN_MAX_TOKENS = 16
MAX_MAX_TOKENS = 2000
# NOTE (verified against DeepSeek): JSON Output mode returns EMPTY or
# truncated-invalid content whenever the model's intended output is large
# (a full Russian data list), regardless of max_tokens. Keeping the list
# small (the instruction bounds it) and giving a generous budget makes it
# work reliably. 800 is the working default.
DEFAULT_MAX_TOKENS = 800

# response_format is fixed by the application for the controlled request.
FIXED_RESPONSE_FORMAT: dict = {"type": "json_object"}

# Default editable JSON structure (educational borscht example).
DEFAULT_JSON_STRUCTURE: dict = {
    "products": [
        {
            "name": "Название продукта",
            "count": "Количество",
            "unit": "Единица измерения",
        }
    ]
}


class CompareRequest(BaseModel):
    """Payload for ``POST /api/compare``."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
        description="Оригинальный запрос — один и тот же для обоих вызовов.",
    )
    json_structure: dict = Field(
        default_factory=lambda: json.loads(json.dumps(DEFAULT_JSON_STRUCTURE)),
        description=(
            "Желаемая структура JSON-ответа (корневой JSON-объект). "
            "Встраивается в контролируемый запрос как инструкция (messages), "
            "а не как API-параметр."
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

    @field_validator("json_structure")
    @classmethod
    def json_structure_must_be_non_empty_object(cls, value: dict) -> dict:
        """Root must be a non-empty JSON object (a bare list/scalar or an
        empty object is not a usable structure)."""
        if not isinstance(value, dict):
            raise ValueError("json_structure must be a JSON object")
        if not value:
            raise ValueError("json_structure must not be empty")
        return value

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
    """Echo of the requested controlled parameters.

    Present even when the controlled provider call fails, so the UI can
    show what was requested.
    """

    response_format: dict
    max_tokens: int
    stop: list[str] | None = None
    json_structure: dict


class CompareResult(BaseModel):
    """One side of the comparison.

    ``settings`` is set only on the controlled side.
    """

    answer: str
    finish_reason: str | None = None
    settings: ControlledSettings | None = None


class CompareResponse(BaseModel):
    """Result of one comparison operation (two provider calls)."""

    unrestricted: CompareResult
    settings: ControlledSettings
    controlled: CompareResult | None = None
    controlled_error: str | None = None
