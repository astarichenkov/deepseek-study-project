"""DeepSeek LLM client wrapper built on the official OpenAI Python SDK.

The class is deliberately isolated from FastAPI so it can be unit-tested
with a fully mocked HTTP layer and swapped for a fake via dependency
injection in the API tests.

Two public operations:

* ``chat(message)``          — single unrestricted request (POST /api/chat);
* ``compare(request)``       — the SAME prompt twice: once unrestricted and
  once with REAL API response controls (``response_format`` = JSON mode,
  editable ``json_structure`` instruction in messages, ``max_tokens`` and
  ``stop``). Exactly two provider calls per comparison.
"""
import json
import logging

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

from app.config import Settings
from app.schemas.chat import ChatResponse
from app.schemas.compare import (
    FIXED_RESPONSE_FORMAT,
    CompareRequest,
    CompareResponse,
    CompareResult,
    ControlledSettings,
)

logger = logging.getLogger("app.services.deepseek")

# Application-level default output-token cap used by /api/chat and by the
# UNRESTRICTED side of a comparison. It keeps normal responses bounded and
# is independent of the student-configurable max_tokens of the controlled
# request (so the comparison isolates the effect of response controls).
DEFAULT_MAX_TOKENS = 1024


class DeepSeekError(Exception):
    """Base class for all DeepSeek service failures.

    * ``message`` -- user-friendly text, safe to expose to the browser
      (no secrets, no stack traces, no internal details).
    * ``status_code`` -- the HTTP status the API layer should return.
    """

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class DeepSeekAuthenticationError(DeepSeekError):
    """The API key is missing or invalid."""

    def __init__(self) -> None:
        super().__init__(
            "DeepSeek authentication failed. Please check the API key.", 401
        )


class DeepSeekRateLimitError(DeepSeekError):
    """DeepSeek throttled the request (HTTP 429)."""

    def __init__(self) -> None:
        super().__init__(
            "DeepSeek rate limit exceeded. Please wait a moment and try again.", 429
        )


class DeepSeekTimeoutError(DeepSeekError):
    """The upstream request exceeded the configured timeout."""

    def __init__(self) -> None:
        super().__init__(
            "DeepSeek took too long to respond. Please try again.", 504
        )


class DeepSeekNetworkError(DeepSeekError):
    """The upstream host could not be reached (DNS/TCP/TLS failure)."""

    def __init__(self) -> None:
        super().__init__(
            "Could not reach the DeepSeek service. Please try again later.", 502
        )


class DeepSeekMalformedResponseError(DeepSeekError):
    """The provider returned something we could not parse."""

    def __init__(self) -> None:
        super().__init__(
            "DeepSeek returned an unexpected response. Please try again.", 502
        )


class DeepSeekService:
    """Thin wrapper around ``AsyncOpenAI`` configured for DeepSeek."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # A placeholder key keeps the client constructible even without
        # configuration; a real (missing/invalid) key surfaces as a
        # friendly 401 from the API layer instead of a client crash.
        self._client = AsyncOpenAI(
            api_key=settings.deepseek_api_key or "missing-api-key",
            base_url=settings.deepseek_base_url,
            timeout=settings.deepseek_timeout_seconds,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def chat(self, message: str) -> ChatResponse:
        """Single unrestricted request (used by ``POST /api/chat``)."""
        result = await self._complete(
            [
                {"role": "system", "content": self._settings.system_prompt},
                {"role": "user", "content": message},
            ],
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        return ChatResponse(answer=result.answer)

    async def compare(self, request: CompareRequest) -> CompareResponse:
        """Send the SAME prompt twice and compare the answers.

        Exactly two provider calls:

        1. ``unrestricted`` — system + original user prompt only, app default
           cap. NO response_format, NO custom max_tokens, NO stop, NO JSON
           structure / termination instruction.
        2. ``controlled``   — the SAME original user prompt plus a JSON
           control instruction, and REAL API parameters:

           * ``response_format={"type": "json_object"}`` (fixed);
           * ``max_tokens=<from UI>``;
           * ``stop=[<from UI>]`` only when the user provided a sequence
             (empty is omitted — no artificial stop marker by default).

           The control instruction is appended to the SAME final user message
           as the original prompt (kept verbatim). It explicitly mentions
           JSON, embeds the user's editable JSON structure and asks the model
           to fill it with a bounded list. No artificial END marker is
           injected, so the JSON keeps its natural ending.

        Failure policy (partial-failure handling):

        * if the UNRESTRICTED call fails -> raise (HTTP error; nothing to
          compare);
        * if the CONTROLLED call fails -> the unrestricted result is still
          returned with ``controlled=None``, a friendly ``controlled_error``
          and the requested ``settings`` so the UI can show what was
          requested.

        No retries; one comparison never spends more than two provider calls.
        """
        json_struct_size = len(json.dumps(request.json_structure, ensure_ascii=False))
        logger.info(
            "comparison started message_length=%s json_structure_present=%s "
            "json_structure_size=%s response_format=json_object max_tokens=%s "
            "stop_configured=%s",
            len(request.message),
            True,
            json_struct_size,
            request.max_tokens,
            bool(request.stop_sequence),
        )

        system = {"role": "system", "content": self._settings.system_prompt}
        user = {"role": "user", "content": request.message}

        # 1) Unrestricted request.
        logger.info("unrestricted call started")
        unrestricted = await self._complete(
            [system, user],
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        logger.info(
            "unrestricted call completed finish_reason=%s content_length=%s",
            unrestricted.finish_reason,
            len(unrestricted.answer),
        )

        # 2) Controlled request.
        # Empirically DeepSeek json mode is most reliable when the final user
        # message itself carries the JSON instruction (a bare prompt following
        # a separate instruction message is more likely to yield empty output).
        # The original user prompt is kept VERBATIM as the prefix of the final
        # message, then the control instruction is appended.
        instruction = self._build_control_instruction(request)
        final_controlled_message = request.message + "\n\n" + instruction
        controlled_messages = [system, {"role": "user", "content": final_controlled_message}]
        stop_list = [request.stop_sequence] if request.stop_sequence else None
        settings_echo = ControlledSettings(
            response_format=dict(FIXED_RESPONSE_FORMAT),
            max_tokens=request.max_tokens,
            stop=stop_list,
            json_structure=request.json_structure,
        )
        logger.info(
            "controlled call started response_format=json_object max_tokens=%s "
            "stop_configured=%s",
            request.max_tokens,
            bool(request.stop_sequence),
        )
        try:
            controlled = await self._complete(
                controlled_messages,
                response_format=FIXED_RESPONSE_FORMAT,
                max_tokens=request.max_tokens,
                stop=request.stop_sequence,
            )
        except DeepSeekError as exc:
            logger.warning(
                "controlled call failed exception_class=%s", type(exc).__name__
            )
            return CompareResponse(
                unrestricted=self._result(unrestricted),
                settings=settings_echo,
                controlled=None,
                controlled_error=exc.message,
            )

        logger.info(
            "controlled call completed finish_reason=%s content_length=%s",
            controlled.finish_reason,
            len(controlled.answer),
        )
        logger.info("comparison completed")
        return CompareResponse(
            unrestricted=self._result(unrestricted),
            settings=settings_echo,
            controlled=self._result(controlled, settings=settings_echo),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _result(
        self,
        result: CompareResult,
        settings: ControlledSettings | None = None,
    ) -> CompareResult:
        return CompareResult(
            answer=result.answer,
            finish_reason=result.finish_reason,
            settings=settings,
        )

    def _build_control_instruction(self, request: CompareRequest) -> str:
        """The controlled-request instruction (a real ``messages`` message).

        * explicitly mentions JSON (DeepSeek JSON Output requires the word
          "json" in the messages when response_format=json_object is used);
        * embeds the user's editable JSON structure verbatim;
        * asks the model to fill the structure with a BOUNDED list (max ~7
          items) so the JSON stays compact and json mode completes it;
        * does NOT inject an artificial END marker — the JSON keeps its
          natural ending (stop is optional and separate).

        The original user message is NOT altered.
        """
        structure_text = json.dumps(
            request.json_structure, ensure_ascii=False, indent=2
        )
        parts = [
            "Верни ответ только в формате JSON.",
            "Используй следующую структуру JSON:\n" + structure_text,
            (
                "Заполни структуру реальными данными по запросу пользователя. "
                "Укажи только основные ингредиенты — максимум 7 позиций. "
                "Ответ должен содержать только корректный JSON."
            ),
        ]
        return "\n\n".join(parts)

    async def _complete(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict | None = None,
        max_tokens: int | None = None,
        stop: str | None = None,
    ) -> CompareResult:
        """Run one Chat Completions call with optional response controls.

        Only validated, explicitly requested parameters are forwarded:

        * ``response_format`` — structured-output configuration (forwarded
          as-is to the API, e.g. ``{"type": "json_object"}``);
        * ``max_tokens``      — output-token limit;
        * ``stop``            — sent as ``stop=[...]``.

        Returns the answer plus the REAL ``finish_reason`` reported by the
        provider (``stop`` / ``length`` / whatever DeepSeek returns) — the
        value is never invented by the application.
        """
        params: dict = {
            "model": self._settings.deepseek_model,
            "messages": messages,
            "temperature": 0.7,
        }
        if response_format is not None:
            params["response_format"] = response_format
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if stop:
            params["stop"] = [stop]

        try:
            response = await self._client.chat.completions.create(**params)
        except AuthenticationError as exc:
            logger.warning("DeepSeek auth error (type=%s)", type(exc).__name__)
            raise DeepSeekAuthenticationError() from exc
        except RateLimitError as exc:
            logger.warning("DeepSeek rate limit (type=%s)", type(exc).__name__)
            raise DeepSeekRateLimitError() from exc
        except APITimeoutError as exc:
            logger.warning("DeepSeek timeout (type=%s)", type(exc).__name__)
            raise DeepSeekTimeoutError() from exc
        except APIConnectionError as exc:
            logger.warning("DeepSeek connection error (type=%s)", type(exc).__name__)
            raise DeepSeekNetworkError() from exc
        except APIError as exc:
            logger.warning(
                "DeepSeek API error (type=%s, status=%s)",
                type(exc).__name__,
                getattr(exc, "status_code", "?"),
            )
            raise DeepSeekError(
                "The DeepSeek API reported an error. Please try again.", 502
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive catch-all
            logger.exception(
                "Unexpected DeepSeek client error (type=%s)", type(exc).__name__
            )
            raise DeepSeekError(
                "An unexpected error occurred while contacting DeepSeek.", 500
            ) from exc

        # Validate the shape of the provider response before using it.
        try:
            choice = response.choices[0]
            content = choice.message.content
            finish_reason = choice.finish_reason
        except (AttributeError, IndexError, TypeError) as exc:
            try:
                choices_len = len(response.choices)
            except Exception:
                choices_len = "?"
            try:
                first_message = response.choices[0].message
                message_present = first_message is not None
            except Exception:
                first_message = None
                message_present = "?"
            try:
                first_finish = response.choices[0].finish_reason
            except Exception:
                first_finish = "?"
            logger.warning(
                "Malformed DeepSeek response structure (type=%s, choices_len=%s, "
                "first_choice_has_message=%s, first_finish_reason=%s) — "
                "choices missing/empty or message absent",
                type(response).__name__,
                choices_len,
                message_present,
                first_finish,
            )
            raise DeepSeekMalformedResponseError() from exc

        if content is None or not content.strip():
            # Real DeepSeek behaviour: with response_format=json_object the
            # provider sometimes returns EMPTY content (finish_reason
            # "length"/"stop"). Log safe diagnostics; do not fake data.
            logger.warning(
                "DeepSeek returned empty content (content_is_none=%s, "
                "content_length=0, finish_reason=%r, response_format=%s, "
                "max_tokens=%s, stop=%s)",
                content is None,
                finish_reason,
                params.get("response_format"),
                params.get("max_tokens"),
                params.get("stop"),
            )
            raise DeepSeekMalformedResponseError() from None

        return CompareResult(answer=content.strip(), finish_reason=finish_reason)
