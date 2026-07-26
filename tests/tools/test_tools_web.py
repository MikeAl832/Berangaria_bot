"""
Характеризационные тесты tools.py: rate limiter и read_url.
Сеть не трогаем — httpx.Client подменяется фейком.
"""
from berangaria.tools import web as tools


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


HTML = b"<html><head><title>T</title></head><body>\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82 \xd0\xbc\xd0\xb8\xd1\x80</body></html>"


class _FakeResponse:
    def __init__(self, url, *, body=HTML, status_code=200, headers=None):
        self.url = url
        self.status_code = status_code
        self.headers = headers if headers is not None else {"content-type": "text/html"}
        self.encoding = "utf-8"
        self._body = body
        self.body_read = False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_bytes(self):
        self.body_read = True
        for offset in range(0, len(self._body), 64):
            yield self._body[offset:offset + 64]

    def raise_for_status(self):
        pass


class _FakeClient:
    """Ловит запрос, с которым реально пошёл read_url."""
    captured = {}
    responses = []

    def __init__(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def stream(self, method, url, **kwargs):
        _FakeClient.captured = {
            "method": method,
            "url": url,
            "headers": kwargs.get("headers") or {},
            "extensions": kwargs.get("extensions") or {},
        }
        _FakeClient.responses.append(_FakeClient.captured)
        return self.make_response(url)

    def make_response(self, url):
        return _FakeResponse(url)


def _resolve_to(monkeypatch, *addresses):
    """Подменяет DNS фиксированным списком адресов."""
    queue = list(addresses)

    def fake_getaddrinfo(host, port, **kw):
        value = queue.pop(0) if len(queue) > 1 else queue[0]
        return [(tools.socket.AF_INET, tools.socket.SOCK_STREAM, 6, "", (value, port))]

    monkeypatch.setattr(tools.socket, "getaddrinfo", fake_getaddrinfo)


def test_read_url_prepends_https_scheme(monkeypatch):
    monkeypatch.setattr(tools.httpx, "Client", _FakeClient)
    _resolve_to(monkeypatch, "93.184.216.34")
    _FakeClient.responses.clear()

    result = tools.read_url("example.com")

    # Подключаемся к проверенному адресу, имя хоста уходит в Host/SNI.
    assert _FakeClient.captured["url"] == "https://93.184.216.34"
    assert _FakeClient.captured["headers"]["Host"] == "example.com"
    assert _FakeClient.captured["extensions"]["sni_hostname"] == "example.com"
    assert "Привет мир" in result


def test_read_url_keeps_existing_scheme_and_path(monkeypatch):
    monkeypatch.setattr(tools.httpx, "Client", _FakeClient)
    _resolve_to(monkeypatch, "93.184.216.34")
    _FakeClient.responses.clear()

    tools.read_url("http://example.com/page")

    assert _FakeClient.captured["url"] == "http://93.184.216.34/page"
    assert _FakeClient.captured["headers"]["Host"] == "example.com"


def test_read_url_rejects_private_address(monkeypatch):
    monkeypatch.setattr(tools.httpx, "Client", _FakeClient)
    _FakeClient.responses.clear()

    result = tools.read_url("http://127.0.0.1:6333/collections")

    assert "отклонён" in result.lower()
    assert "локальным" in result.lower()
    assert _FakeClient.responses == []


def test_read_url_pins_validated_ip_against_dns_rebinding(monkeypatch):
    """Второй resolve не должен существовать.

    Хост с чередующимися A-записями (публичный / 127.0.0.1) иначе выигрывает
    примерно каждую вторую попытку: валидация видит публичный адрес, а httpx
    делает собственный getaddrinfo и уходит на внутренний.
    """
    monkeypatch.setattr(tools.httpx, "Client", _FakeClient)
    _resolve_to(monkeypatch, "93.184.216.34", "127.0.0.1")
    _FakeClient.responses.clear()

    tools.read_url("http://rebind.example/page")

    assert _FakeClient.captured["url"] == "http://93.184.216.34/page"
    assert "127.0.0.1" not in _FakeClient.captured["url"]


def test_read_url_revalidates_redirect_target(monkeypatch):
    class RedirectClient(_FakeClient):
        def make_response(self, url):
            return _FakeResponse(
                url,
                status_code=302,
                headers={"location": "http://127.0.0.1/private"},
            )

    monkeypatch.setattr(tools.httpx, "Client", RedirectClient)
    _resolve_to(monkeypatch, "93.184.216.34")
    RedirectClient.responses.clear()

    result = tools.read_url("https://example.com/redirect")

    assert "отклонён" in result.lower()
    assert len(RedirectClient.responses) == 1


def test_read_url_rejects_oversized_body(monkeypatch):
    class HugeClient(_FakeClient):
        def make_response(self, url):
            return _FakeResponse(url, body=b"x" * (tools.READ_URL_MAX_BYTES + 1024))

    monkeypatch.setattr(tools.httpx, "Client", HugeClient)
    _resolve_to(monkeypatch, "93.184.216.34")

    result = tools.read_url("https://example.com/huge")

    assert "слишком большая" in result.lower()


def test_read_url_trusts_content_length_before_reading(monkeypatch):
    class DeclaredHugeClient(_FakeClient):
        def make_response(self, url):
            return _FakeResponse(
                url,
                headers={
                    "content-type": "text/html",
                    "content-length": str(tools.READ_URL_MAX_BYTES * 10),
                },
            )

    monkeypatch.setattr(tools.httpx, "Client", DeclaredHugeClient)
    _resolve_to(monkeypatch, "93.184.216.34")

    result = tools.read_url("https://example.com/declared-huge")

    assert "слишком большая" in result.lower()


def test_read_url_does_not_read_body_of_non_text_response(monkeypatch):
    captured_response = {}

    class BinaryClient(_FakeClient):
        def make_response(self, url):
            response = _FakeResponse(
                url,
                body=b"\x00" * 1024,
                headers={"content-type": "application/octet-stream"},
            )
            captured_response["response"] = response
            return response

    monkeypatch.setattr(tools.httpx, "Client", BinaryClient)
    _resolve_to(monkeypatch, "93.184.216.34")

    result = tools.read_url("https://example.com/file.bin")

    assert "не текстовая страница" in result.lower()
    assert captured_response["response"].body_read is False


def test_read_url_rejects_non_http_scheme(monkeypatch):
    monkeypatch.setattr(tools.httpx, "Client", _FakeClient)
    _FakeClient.responses.clear()

    result = tools.read_url("gopher://example.com/x")

    # Схема дописывается только при её отсутствии, поэтому gopher доходит до
    # валидатора и отклоняется там.
    assert "отклонён" in result.lower()
    assert _FakeClient.responses == []


def test_read_url_rejects_credentials_in_url(monkeypatch):
    monkeypatch.setattr(tools.httpx, "Client", _FakeClient)
    _resolve_to(monkeypatch, "93.184.216.34")
    _FakeClient.responses.clear()

    result = tools.read_url("https://user:pass@example.com/page")

    assert "отклонён" in result.lower()
    assert _FakeClient.responses == []


def test_read_url_stops_after_max_redirects(monkeypatch):
    class LoopClient(_FakeClient):
        def make_response(self, url):
            return _FakeResponse(
                url,
                status_code=302,
                headers={"location": "https://example.com/next"},
            )

    monkeypatch.setattr(tools.httpx, "Client", LoopClient)
    _resolve_to(monkeypatch, "93.184.216.34")
    LoopClient.responses.clear()

    result = tools.read_url("https://example.com/start")

    assert "перенаправлений" in result.lower()
    assert len(LoopClient.responses) == tools.MAX_URL_REDIRECTS + 1


def test_read_url_client_disables_automatic_redirects(monkeypatch):
    """follow_redirects=False обязателен: иначе httpx уйдёт по цепочке сам."""
    captured_kwargs = {}

    class RecordingClient(_FakeClient):
        def __init__(self, *a, **kw):
            captured_kwargs.update(kw)

    monkeypatch.setattr(tools.httpx, "Client", RecordingClient)
    _resolve_to(monkeypatch, "93.184.216.34")

    tools.read_url("https://example.com/page")

    assert captured_kwargs["follow_redirects"] is False
