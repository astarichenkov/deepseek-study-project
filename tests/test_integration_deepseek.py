"""Optional real-API smoke test for DeepSeek.

This test is EXCLUDED from the normal test run (see ``addopts`` in
``pyproject.toml``). Run it explicitly with:

    pytest -m integration

It is skipped unless ``DEEPSEEK_API_KEY`` is present in the environment
(or in the local ``.env`` file). It performs exactly ONE tiny request, so
it consumes only a negligible amount of API balance.
"""
import os

import pytest

from app.config import get_settings
from app.services.deepseek import DeepSeekService

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY is not set; skipping real API smoke test",
)
async def test_real_deepseek_smoke():
    settings = get_settings()
    service = DeepSeekService(settings)
    answer = await service.chat("Reply with exactly: OK")
    assert answer.answer.strip(), "DeepSeek returned an empty answer"
    assert "OK" in answer.answer.upper()
