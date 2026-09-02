"""Pydantic schemas for the chat API."""

from pydantic import BaseModel, Field, field_validator

# Sensible maximum input size (in characters) for a user question.
MAX_MESSAGE_LENGTH = 4000


def ensure_not_blank(value: str) -> str:
    """Reject empty/whitespace-only strings (shared by chat & compare schemas)."""
    if not value.strip():
        raise ValueError("Value must not be empty or whitespace-only")
    return value


class ChatRequest(BaseModel):
    """Validated payload for ``POST /api/chat``."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
        description="The question the user wants to ask DeepSeek.",
    )

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        """Reject empty and whitespace-only messages (e.g. ``"   "``)."""
        return ensure_not_blank(value)


class ChatResponse(BaseModel):
    """Successful response body for ``POST /api/chat``."""

    answer: str


class ErrorResponse(BaseModel):
    """Uniform error body returned by the API (never contains secrets)."""

    detail: str
