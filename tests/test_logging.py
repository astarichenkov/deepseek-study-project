"""Tests for persistent logging and safe diagnostics.

These do NOT touch the real DeepSeek API.
"""
import asyncio
import logging

import pytest

from app.config import Settings
from app.schemas.compare import CompareRequest
from app.services.deepseek import DeepSeekService


def _make(monkeypatch, results):
    """results = list of (content, finish_reason)."""
    class _Msg:
        pass

    class _Ch:
        pass

    class _R:
        pass

    def build(content, finish_reason="stop"):
        msg = _Msg(); msg.content = content
        ch = _Ch(); ch.message = msg; ch.finish_reason = finish_reason
        r = _R(); r.choices = [ch]
        return r

    class _Comp:
        def __init__(self):
            self.queue = [build(c, f) for (c, f) in results]

        async def create(self, **kwargs):
            item = self.queue.pop(0)
            return item

    class _Chat:
        completions = _Comp()

    class _Client:
        chat = _Chat()

    monkeypatch.setattr("app.services.deepseek.AsyncOpenAI", lambda **k: _Client())
    service = DeepSeekService(Settings(deepseek_api_key="test-key"))
    return service


def test_configure_logging_writes_records(tmp_path):
    from app.main import configure_logging

    log_file = str(tmp_path / "app.log")
    root = configure_logging(log_file)
    logging.getLogger("app.services.deepseek").warning("comparison started max_tokens=500")
    for handler in root.handlers:
        handler.flush()
    content = (tmp_path / "app.log").read_text(encoding="utf-8")
    assert "comparison started max_tokens=500" in content


def test_configure_logging_graceful_when_unwritable(tmp_path):
    """A logging failure must never break the application."""
    from app.main import configure_logging

    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")  # a FILE, so mkdir under it fails
    bad_path = str(blocker / "sub" / "app.log")
    root = configure_logging(bad_path)  # must not raise
    assert root is not None


def test_empty_provider_content_logs_safe_diagnostics(monkeypatch, caplog):
    service = _make(
        monkeypatch,
        [
            ("Unrestricted OK", "stop"),
            (None, "length"),  # controlled returns empty/None
        ],
    )
    request = CompareRequest(message="Тест", json_structure={"products": []}, max_tokens=500)

    with caplog.at_level(logging.WARNING, logger="app.services.deepseek"):
        resp = asyncio.run(service.compare(request))

    assert resp.controlled is None
    assert resp.unrestricted.answer == "Unrestricted OK"
    # safe, structured diagnostics present; not a single useless line
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "empty content" in messages
    assert "content_is_none=True" in messages
    assert "finish_reason" in messages
    # no secrets
    assert "test-key" not in messages
    assert "sk-" not in messages
    assert "Authorization" not in messages
