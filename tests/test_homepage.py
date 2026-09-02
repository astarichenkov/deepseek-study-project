"""Tests for the homepage and static frontend assets."""


def test_homepage_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    # Russian educational comparison UI
    assert "DeepSeek API — управление ответом модели" in html
    assert "Один и тот же запрос с разным уровнем контроля ответа через API" in html
    assert "Настройки ответа с ограничениями" in html
    assert "Параметры API" in html
    assert "Эти параметры будут применены только к ответу «С ограничениями»" in html
    assert 'id="compare-form"' in html
    assert 'id="message"' in html
    assert 'id="response-format"' in html
    assert 'id="reset-json"' in html
    assert 'id="max-tokens"' in html
    assert 'id="stop-sequence"' in html
    assert 'id="api-preview"' in html
    assert 'id="compare-button"' in html
    assert 'id="loading"' in html
    assert 'id="card-unrestricted"' in html
    assert 'id="card-controlled"' in html
    assert 'id="applied-settings"' in html
    assert "Без ограничений" in html
    assert "С ограничениями" in html
    assert "Что изменилось?" in html
    assert "Сравнение выполняет 2 запроса к DeepSeek API" in html
    # Tooltips + real API parameter names
    assert "info-btn" in html
    assert "tooltip" in html
    assert "json_object" in html
    assert "response_format" in html
    assert "max_tokens" in html
    assert "stop" in html
    assert "finish_reason" in html


def test_homepage_does_not_embed_api_key(client, settings):
    response = client.get("/")
    assert settings.deepseek_api_key not in response.text


def test_homepage_does_not_leak_credentials_patterns(client):
    response = client.get("/")
    html = response.text
    assert "Authorization" not in html
    assert "sk-" not in html
    assert "DEEPSEEK_API_KEY" not in html


def test_static_css_served(client):
    response = client.get("/static/css/style.css")
    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]


def test_static_js_served(client):
    response = client.get("/static/js/app.js")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]


def test_all_js_dom_references_exist_in_homepage(client):
    """Regression: every element id that app.js requires must exist in the
    rendered homepage, otherwise the frontend silently breaks (null refs)."""
    import re
    from pathlib import Path

    js_path = Path(__file__).resolve().parents[1] / "app" / "static" / "js" / "app.js"
    js = js_path.read_text(encoding="utf-8")
    referenced = set(re.findall(r'getElementById\("([^"]+)"\)', js))
    assert referenced, "no getElementById calls found in app.js"

    html = client.get("/").text
    present = set(re.findall(r'id="([^"]+)"', html))
    missing = sorted(referenced - present)
    assert not missing, f"app.js references missing element ids: {missing}"


def test_js_is_wrapped_for_dom_ready_and_missing_element_guard(client):
    """Regression: init must be deferred safely and fail clearly if elements
    are missing instead of throwing mid-way (leaving buttons dead)."""
    from pathlib import Path

    js = (Path(__file__).resolve().parents[1] / "app" / "static" / "js" / "app.js").read_text(
        encoding="utf-8"
    )
    assert "DOMContentLoaded" in js
    assert "requiredIds" in js
    assert "missing DOM elements" in js
