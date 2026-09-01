"""Tests for the homepage and static frontend assets."""


def test_homepage_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    assert "DeepSeek Study Assistant" in html
    assert "DeepSeek" in html
    assert 'id="chat-form"' in html
    assert 'id="send-button"' in html
    assert 'id="loading"' in html
    assert 'id="result"' in html


def test_homepage_does_not_embed_api_key(client, settings):
    response = client.get("/")
    assert settings.deepseek_api_key not in response.text


def test_static_css_served(client):
    response = client.get("/static/css/style.css")
    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]


def test_static_js_served(client):
    response = client.get("/static/js/app.js")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
