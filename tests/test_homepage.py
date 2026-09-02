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
