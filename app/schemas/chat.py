"""Pydantic schemas for the chat API."""

from pydantic import BaseModel, Field, field_validator

# Sensible maximum input size (in characters) for a user question.
MAX_MESSAGE_LENGTH = 4000


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
        if not value.strip():
            raise ValueError("Message must not be empty or whitespace-only")
        return value


class ChatResponse(BaseModel):
    """Successful response body for ``POST /api/chat``."""

    answer: str


class ErrorResponse(BaseModel):
    """Uniform error body returned by the API (never contains secrets)."""

    detail: str
