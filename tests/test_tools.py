"""
Характеризационные тесты tools.py: rate limiter и read_url.
Сеть не трогаем — httpx.Client подменяется фейком.
"""
import tools


# ---------- _check_rate_limit ----------

def test_rate_limit_allows_up_to_max_then_blocks():
    tools._search_timestamps.clear()
    allowed = [tools._check_rate_limit() for _ in range(tools.MAX_SEARCHES_PER_MINUTE)]
    assert all(allowed)
    # следующий сверх лимита — запрещён
    assert tools._check_rate_limit() is False
    tools._search_timestamps.clear()


def test_web_search_reports_when_rate_limited(monkeypatch):
    # Забиваем лимит вручную
    tools._search_timestamps.clear()
    for _ in range(tools.MAX_SEARCHES_PER_MINUTE):
        tools._check_rate_limit()
    out = tools.web_search("что угодно")
    assert "лимит" in out.lower()
    tools._search_timestamps.clear()


# ---------- read_url ----------

def test_read_url_empty():
    assert tools.read_url("") == "Пустой URL."
    assert tools.read_url("   ") == "Пустой URL."


class _FakeResponse:
    def __init__(self, url):
        self.requested_url = url
        self.headers = {"content-type": "text/html"}
        self.text = "<html><head><title>T</title></head><body>Привет мир</body></html>"

    def raise_for_status(self):
        pass


class _FakeClient:
    """Ловит URL, с которым реально пошёл запрос."""
    captured = {}

    def __init__(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url):
        _FakeClient.captured["url"] = url
        return _FakeResponse(url)


def test_read_url_prepends_https_scheme(monkeypatch):
    monkeypatch.setattr(tools.httpx, "Client", _FakeClient)
    _FakeClient.captured.clear()
    result = tools.read_url("example.com")
    assert _FakeClient.captured["url"] == "https://example.com"
    assert "Привет мир" in result


def test_read_url_keeps_existing_scheme(monkeypatch):
    monkeypatch.setattr(tools.httpx, "Client", _FakeClient)
    _FakeClient.captured.clear()
    tools.read_url("http://example.com/page")
    assert _FakeClient.captured["url"] == "http://example.com/page"
