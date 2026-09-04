"""Schemas for Day 4 — temperature experiment.

One and the same message is sent with different ``temperature`` values (a
REAL DeepSeek API parameter). No JSON mode. ``response_format`` is not used.
"""
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.chat import MAX_MESSAGE_LENGTH, ensure_not_blank
from app.schemas.compare import (
    MAX_MAX_TOKENS,
    MAX_STOP_SEQUENCE_LENGTH,
    MIN_MAX_TOKENS,
    MIN_STOP_SEQUENCE_LENGTH,
)

# DeepSeek's documented temperature range (OpenAI-compatible endpoint).
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
DEFAULT_MAX_TOKENS_DAY4 = 700


class TemperatureRequest(BaseModel):
    """Payload for ``POST /api/temperature``."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
        description="The (common) user prompt sent for this temperature.",
    )
    temperature: float = Field(
        ...,
        ge=MIN_TEMPERATURE,
        le=MAX_TEMPERATURE,
        description=(
            "Real API temperature (variability). Editable by the user; "
            f"accepted provider range {MIN_TEMPERATURE}..{MAX_TEMPERATURE}."
        ),
    )
    max_tokens: int = Field(
        DEFAULT_MAX_TOKENS_DAY4,
        ge=MIN_MAX_TOKENS,
        le=MAX_MAX_TOKENS,
        description="Output-token limit (real API max_tokens).",
    )
    stop_sequence: str | None = Field(
        None,
        description="Optional stop sequence; empty/blank is omitted.",
    )

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        return ensure_not_blank(value)

    @field_validator("stop_sequence")
    @classmethod
    def clean_stop_sequence(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        if len(value) > MAX_STOP_SEQUENCE_LENGTH:
            raise ValueError(
                f"stop_sequence is too long (max {MAX_STOP_SEQUENCE_LENGTH} characters)"
            )
        if len(value) < MIN_STOP_SEQUENCE_LENGTH or not any(ch.isalnum() for ch in value):
            raise ValueError(
                "stop_sequence must be a distinctive marker of at least "
                f"{MIN_STOP_SEQUENCE_LENGTH} characters (letters/digits)"
            )
        return value


class TemperatureResponse(BaseModel):
    """Result of one temperature run."""

    answer: str
    finish_reason: str | None = None
    usage: dict | None = None
    applied_parameters: dict  # {model, temperature, max_tokens, stop}
