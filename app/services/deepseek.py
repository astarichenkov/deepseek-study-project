"""DeepSeek LLM client wrapper built on the official OpenAI Python SDK.

The class is deliberately isolated from FastAPI so it can be unit-tested
with a fully mocked HTTP layer and swapped for a fake via dependency
injection in the API tests.

Two public operations:

* ``chat(message)``          — single unrestricted request (POST /api/chat);
* ``compare(request)``       — the SAME prompt twice: once unrestricted and
  once with REAL API response-control parameters (``response_format``,
  ``max_tokens``, ``stop``). Exactly two provider calls per comparison.
"""
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
    CompareRequest,
    CompareResponse,
    CompareResult,
    ControlledSettings,
)

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
        """Single unrestricted request (used by ``POST /api/chat``).

        Raises a ``DeepSeekError`` subclass on every failure mode; the API
        layer converts those into proper HTTP responses.
        """
        result = await self._complete(
            self._messages(message, instruction=None),
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        return ChatResponse(answer=result.answer)

    async def compare(self, request: CompareRequest) -> CompareResponse:
        """Send the SAME prompt twice and compare the answers.

        Exactly two provider calls:

        1. ``unrestricted`` — system + user prompt only, app default cap.
           NO custom ``response_format``, NO custom ``max_tokens``, NO
           ``stop``, NO extra instructions.
        2. ``controlled``   — the SAME user prompt, plus:
           * ``response_format=<validated JSON from the UI>`` (real API
             parameter, e.g. ``{"type": "json_object"}``);
           * ``max_tokens=<value from the UI>``;
           * ``stop=[<stop sequence>]`` when provided;
           * when JSON mode is on, an instruction message that mentions
             "json" (DeepSeek JSON Output requirement) — added ONLY to the
             controlled request;
           * when a stop sequence is set, an explicit termination
             instruction — added ONLY to the controlled request.

        Failure policy (partial-failure handling):

        * if the UNRESTRICTED call fails -> raise (the API layer maps the
          error to HTTP; there is nothing useful to compare);
        * if the CONTROLLED call fails -> the unrestricted result is still
          returned with ``controlled=None`` and a friendly
          ``controlled_error`` message.

        No retries are performed; one comparison never spends more than two
        provider calls.
        """
        system = {"role": "system", "content": self._settings.system_prompt}
        user = {"role": "user", "content": request.message}

        # 1) Unrestricted request.
        unrestricted = await self._complete(
            [system, user],
            max_tokens=DEFAULT_MAX_TOKENS,
        )

        # 2) Controlled request.
        instruction = self._build_control_instruction(request)
        controlled_messages = [system, user, {"role": "user", "content": instruction}]
        stop_list = [request.stop_sequence] if request.stop_sequence else None
        try:
            controlled = await self._complete(
                controlled_messages,
                response_format=request.response_format.as_api_param(),
                max_tokens=request.max_tokens,
                stop=request.stop_sequence,
            )
        except DeepSeekError as exc:
            return CompareResponse(
                unrestricted=self._result(unrestricted),
                controlled=None,
                controlled_error=exc.message,
            )

        return CompareResponse(
            unrestricted=self._result(unrestricted),
            controlled=self._result(
                controlled,
                settings=ControlledSettings(
                    response_format=request.response_format.as_api_param(),
                    max_tokens=request.max_tokens,
                    stop=stop_list,
                ),
            ),
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

    def _messages(
        self, message: str, instruction: str | None
    ) -> list[dict[str, str]]:
        """Build the ``messages`` list.

        The original user prompt is ALWAYS kept verbatim as its own
        message; the control instruction, when present, is appended as a
        SEPARATE user message so the original prompt is not altered.
        """
        messages = [
            {"role": "system", "content": self._settings.system_prompt},
            {"role": "user", "content": message},
        ]
        if instruction:
            messages.append({"role": "user", "content": instruction})
        return messages

    def _build_control_instruction(self, request: CompareRequest) -> str:
        """Instructions added ONLY to the controlled request.

        * JSON mode (``response_format={"type": "json_object"}``): an
          explicit message mentioning "json" — DeepSeek's JSON Output mode
          expects the word "json" to appear in the messages. This is a
          prompt-level requirement that accompanies the API parameter; the
          JSON mode itself is enabled through the real ``response_format``
          parameter.
        * Stop sequence: an explicit termination instruction telling the
          model to finish with the marker (the sequence is ALSO sent as the
          real ``stop`` API parameter). In JSON mode the marker is placed
          right after the closing brace of the JSON object.

        Empirically verified against DeepSeek: a contradictory instruction
        ("nothing outside JSON" + marker after JSON) yields empty output,
        so JSON-mode wording explicitly allows the trailing marker.
        """
        parts = []
        if request.response_format.type == "json_object":
            parts.append(
                "Сначала напиши валидный JSON-объект (JSON) с краткими пунктами, "
                'например: {"advantages": ["...", "...", "..."]}.'
            )
        if request.stop_sequence:
            if request.response_format.type == "json_object":
                parts.append(
                    f"В самом конце, сразу после закрывающей фигурной скобки JSON, "
                    f"добавь маркер {request.stop_sequence}."
                )
            else:
                parts.append(
                    f"Заверши ответ маркером {request.stop_sequence} "
                    "и не пиши ничего после него."
                )
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
            raise DeepSeekAuthenticationError() from exc
        except RateLimitError as exc:
            raise DeepSeekRateLimitError() from exc
        except APITimeoutError as exc:
            raise DeepSeekTimeoutError() from exc
        except APIConnectionError as exc:
            raise DeepSeekNetworkError() from exc
        except APIError as exc:
            raise DeepSeekError(
                "The DeepSeek API reported an error. Please try again.", 502
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive catch-all
            raise DeepSeekError(
                "An unexpected error occurred while contacting DeepSeek.", 500
            ) from exc

        # Validate the shape of the provider response before using it.
        try:
            choice = response.choices[0]
            content = choice.message.content
            finish_reason = choice.finish_reason
        except (AttributeError, IndexError, TypeError):
            raise DeepSeekMalformedResponseError() from None

        if not content or not content.strip():
            raise DeepSeekMalformedResponseError()

        return CompareResult(answer=content.strip(), finish_reason=finish_reason)
