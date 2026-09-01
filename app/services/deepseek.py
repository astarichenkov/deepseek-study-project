"""DeepSeek LLM client wrapper built on the official OpenAI Python SDK.

The class is deliberately isolated from FastAPI so it can be unit-tested
with a fully mocked HTTP layer and swapped for a fake via dependency
injection in the API tests.
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

    async def chat(self, message: str) -> ChatResponse:
        """Send ``message`` to DeepSeek and return the generated answer.

        Raises a ``DeepSeekError`` subclass on every failure mode; the API
        layer converts those into proper HTTP responses.
        """
        try:
            response = await self._client.chat.completions.create(
                model=self._settings.deepseek_model,
                messages=[
                    {"role": "system", "content": self._settings.system_prompt},
                    {"role": "user", "content": message},
                ],
                temperature=0.7,
                max_tokens=1024,
            )
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
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            raise DeepSeekMalformedResponseError() from None

        if not content or not content.strip():
            raise DeepSeekMalformedResponseError()

        return ChatResponse(answer=content.strip())
