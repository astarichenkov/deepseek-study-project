"""Pydantic schemas for Day 3 — reasoning-strategy comparison.

Four ways to solve the SAME task through the DeepSeek API (no JSON mode,
no editable structure — those belong to Day 2):

1. direct          — original task without extra reasoning instructions;
2. step_by_step    — same task plus "решай пошагово";
3. generate_prompt — stage 1: model creates an improved prompt;
   use_prompt      — stage 2: solve using that (user-editable) prompt;
4. experts         — one prompt with several expert roles inside.

Validation must happen before any provider call.
"""
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.chat import MAX_MESSAGE_LENGTH, ensure_not_blank
from app.schemas.compare import (
    MAX_MAX_TOKENS,
    MAX_STOP_SEQUENCE_LENGTH,
    MIN_MAX_TOKENS,
    MIN_STOP_SEQUENCE_LENGTH,
)

ReasoningMethod = Literal[
    "direct", "step_by_step", "generate_prompt", "use_prompt", "experts"
]

DEFAULT_MAX_TOKENS_DAY3 = 1000
MAX_GENERATED_PROMPT_LENGTH = 12000


class ReasoningRequest(BaseModel):
    """Payload for ``POST /api/reasoning``."""

    method: ReasoningMethod = Field(..., description="Which reasoning strategy to run.")
    task: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
        description="The original task (shared by every strategy).",
    )
    max_tokens: int = Field(
        DEFAULT_MAX_TOKENS_DAY3,
        ge=MIN_MAX_TOKENS,
        le=MAX_MAX_TOKENS,
        description="Output-token limit (real API max_tokens).",
    )
    stop_sequence: str | None = Field(
        None,
        description="Optional stop sequence; empty/blank is omitted.",
    )
    generated_prompt: str | None = Field(
        None,
        max_length=MAX_GENERATED_PROMPT_LENGTH,
        description="Required only for method='use_prompt' (the prompt to solve with).",
    )

    @model_validator(mode="after")
    def check_fields(self) -> "ReasoningRequest":
        task = self.task.strip()
        if not task:
            raise ValueError("task must not be empty or whitespace-only")

        if self.method == "use_prompt":
            gp = (self.generated_prompt or "").strip()
            if not gp:
                raise ValueError(
                    "generated_prompt is required for method='use_prompt'"
                )
        return self

    @model_validator(mode="after")
    def clean_stop_sequence(self) -> "ReasoningRequest":
        value = self.stop_sequence
        if value is None:
            return self
        value = value.strip()
        if not value:
            self.stop_sequence = None
            return self
        if len(value) > MAX_STOP_SEQUENCE_LENGTH:
            raise ValueError(
                f"stop_sequence is too long (max {MAX_STOP_SEQUENCE_LENGTH} characters)"
            )
        if len(value) < MIN_STOP_SEQUENCE_LENGTH or not any(
            ch.isalnum() for ch in value
        ):
            raise ValueError(
                "stop_sequence must be a distinctive marker of at least "
                f"{MIN_STOP_SEQUENCE_LENGTH} characters (letters/digits)"
            )
        return self


class ReasoningResponse(BaseModel):
    """Result of one reasoning request."""

    method: str
    kind: Literal["solution", "generated_prompt"] = "solution"
    prompt_sent: str | None = None  # exact user prompt that was sent
    solution: str | None = None  # solution text (solution kinds)
    generated_prompt: str | None = None  # produced prompt (generate_prompt)
    finish_reason: str | None = None
    usage: dict | None = None
    status: Literal["correct", "incorrect", "indeterminate"] | None = None
