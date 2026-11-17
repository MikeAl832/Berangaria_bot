"""Web search and SSRF-hardened URL reading (DeepSeek Search/Fetch + direct fallback)."""

from __future__ import annotations

import ipaddress
import logging
import socket
import time
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup

from berangaria.config import DEEPSEEK_API_BASE, DEEPSEEK_API_KEY, SEARCH_PROVIDER

logger = logging.getLogger(__name__)

# Process-global rate limiter for web_search: protects shared search budget.
_search_timestamps: list[float] = []
MAX_SEARCHES_PER_MINUTE = 10

RATE_LIMIT_PREFIX = "⚠️ Превышен лимит поисковых запросов"

DEEPSEEK_SEARCH_URL = f"{DEEPSEEK_API_BASE}/v1/tools/search"
DEEPSEEK_FETCH_URL = f"{DEEPSEEK_API_BASE}/v1/tools/fetch"

READ_URL_MAX_CHARS = 4000
MAX_URL_REDIRECTS = 5
READ_URL_MAX_BYTES = 4 * 1024 * 1024
READ_URL_TOTAL_TIMEOUT = 30.0


def _check_rate_limit() -> bool:
    """Return True when another search call is allowed within the 60s window."""
    current_time = time.time()
    _search_timestamps[:] = [
        ts for ts in _search_timestamps if current_time - ts < 60
    ]
    if len(_search_timestamps) >= MAX_SEARCHES_PER_MINUTE:
        return False
    _search_timestamps.append(current_time)
    return True


def _extract_search_items(data: object) -> list[dict]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []
    for key in ("results", "data", "web_pages", "items"):
        val = data.get(key)
        if isinstance(val, list):
            return [item for item in val if isinstance(item, dict)]
        if isinstance(val, dict):
            for nested in ("results", "items", "value"):
                sub = val.get(nested)
                if isinstance(sub, list):
                    return [item for item in sub if isinstance(item, dict)]
    return []


def _format_search_items(items: list[dict], *, max_results: int, query: str) -> str:
    if not items:
        return f"По запросу '{query}' ничего не найдено."
    lines: list[str] = []
    for i, item in enumerate(items[:max_results], 1):
        href = str(item.get("url") or item.get("href") or item.get("link") or "").strip()
        title = str(
            item.get("title")
            or item.get("name")
            or item.get("site_name")
            or href
            or "Без заголовка"
        ).strip()
        raw_date = str(item.get("date") or "").strip()
        date_prefix = f"[{raw_date[:10]}] " if len(raw_date) >= 10 else ""
        snippet = str(
            item.get("snippet")
            or item.get("body")
            or item.get("content")
            or item.get("summary")
            or ""
        ).strip()
        if len(snippet) > 500:
            snippet = snippet[:500].rstrip() + "…"
        lines.append(f"{i}. {date_prefix}{title}\n{snippet}\n{href}".strip())
    return "\n\n".join(lines).strip()


def _deepseek_web_search(query: str, max_results: int = 5) -> str:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query[:128],
        "count": max(1, min(int(max_results or 5), 10)),
        "need_content": False,
    }
    with httpx.Client(timeout=15.0) as client:
        resp = client.post(DEEPSEEK_SEARCH_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    return _format_search_items(
        _extract_search_items(data), max_results=max_results, query=query
    )


def _ddgs_web_search(
    query: str,
    max_results: int = 5,
    timelimit: str | None = None,
    region: str = "ru-ru",
) -> str:
    from ddgs import DDGS

    with DDGS() as ddgs:
        results = list(
            ddgs.text(
                query,
                max_results=max_results,
                region=region,
                timelimit=timelimit,
                safesearch="off",
            )
        )
    return _format_search_items(results, max_results=max_results, query=query)


def web_search(
    query: str,
    max_results: int = 5,
    timelimit: str | None = None,
    region: str = "ru-ru",
) -> str:
    if not _check_rate_limit():
        return f"{RATE_LIMIT_PREFIX} ({MAX_SEARCHES_PER_MINUTE}/мин). Попробуйте позже."

    query = (query or "").strip()
    if not query:
        return "Пустой поисковый запрос."

    if SEARCH_PROVIDER == "deepseek" and DEEPSEEK_API_KEY:
        try:
            return _deepseek_web_search(query, max_results=max_results)
        except Exception as exc:
            logger.warning(
                "🔍 [yellow]DeepSeek Search API недоступен (%s), пробуем fallback[/]",
                exc,
            )
            try:
                return _ddgs_web_search(
                    query,
                    max_results=max_results,
                    timelimit=timelimit,
                    region=region,
                )
            except Exception as fallback_exc:
                return f"Ошибка поиска: {fallback_exc}"

    try:
        return _ddgs_web_search(
            query,
            max_results=max_results,
            timelimit=timelimit,
            region=region,
        )
    except Exception as exc:
        return f"Ошибка поиска: {exc}"


def _validate_public_url(url: str) -> str:
    """Reject URLs targeting non-HTTP schemes, credentials, or private/local IPs.

    Returns the first validated public IP so direct HTTP requests can pin it and
    prevent DNS rebinding between check and connect.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Разрешены только HTTP и HTTPS URL.")
    if not parsed.hostname:
        raise ValueError("В URL отсутствует имя хоста.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL с логином или паролем не поддерживаются.")

    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise ValueError("Некорректный порт в URL.") from exc

    try:
        literal = ipaddress.ip_address(parsed.hostname.split("%", 1)[0])
        addresses = [str(literal)]
    except ValueError:
        try:
            addr_info = socket.getaddrinfo(
                parsed.hostname, port, type=socket.SOCK_STREAM
            )
        except socket.gaierror as exc:
            raise ValueError("Не удалось определить адрес сайта.") from exc
        addresses = list(
            dict.fromkeys(item[4][0].split("%", 1)[0] for item in addr_info)
        )
    if not addresses:
        raise ValueError("Сайт не имеет доступных IP-адресов.")

    for raw_ip in addresses:
        ip = ipaddress.ip_address(raw_ip)
        if not ip.is_global:
            raise ValueError("Доступ к локальным и служебным сетевым адресам запрещён.")

    return addresses[0]


def _pinned_request_args(url: str, validated_ip: str) -> tuple[str, dict[str, str]]:
    """Connect directly to the validated IP while preserving Host/SNI."""
    parsed = urlparse(url)
    host_header = parsed.netloc.split("@", 1)[-1]
    ip = ipaddress.ip_address(validated_ip)
    literal = f"[{validated_ip}]" if ip.version == 6 else validated_ip
    netloc = f"{literal}:{parsed.port}" if parsed.port else literal
    return urlunparse(parsed._replace(netloc=netloc)), {"Host": host_header}


def _read_body_within_limit(response, max_bytes: int) -> bytes:
    """Stream response body while enforcing a strict byte ceiling."""
    declared = response.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes:
        raise ValueError(
            f"Страница слишком большая ({int(declared) // 1024} КБ)."
        )
    body = bytearray()
    for piece in response.iter_bytes():
        body.extend(piece)
        if len(body) > max_bytes:
            raise ValueError(
                f"Страница слишком большая (> {max_bytes // 1024} КБ)."
            )
    return bytes(body)


def _deepseek_fetch_url(url: str, *, max_chars: int) -> str:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=20.0) as client:
        resp = client.post(DEEPSEEK_FETCH_URL, json={"url": url}, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    if isinstance(data, dict):
        payload = data.get("data") if isinstance(data.get("data"), dict) else data
        title = str(payload.get("title") or "").strip()
        content = str(
            payload.get("content")
            or payload.get("markdown")
            or payload.get("text")
            or ""
        ).strip()
    else:
        title = ""
        content = ""

    if not content:
        raise ValueError("DeepSeek Fetch вернул пустой текст")

    result = f"Заголовок: {title}\n\n" if title else ""
    result += content[:max_chars]
    if len(content) > max_chars:
        result += "…"
    return result


def _direct_read_url(url: str, *, max_chars: int = READ_URL_MAX_CHARS) -> str:
    """SSRF-hardened direct HTTP fetch with pinned IP and HTML-to-text cleanup."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    timeout = httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=5.0)
    deadline = time.monotonic() + READ_URL_TOTAL_TIMEOUT

    with httpx.Client(
        timeout=timeout, follow_redirects=False, headers=headers
    ) as client:
        for _ in range(MAX_URL_REDIRECTS + 1):
            if time.monotonic() > deadline:
                return "Превышено общее время чтения страницы."
            validated_ip = _validate_public_url(url)
            request_url, host_header = _pinned_request_args(url, validated_ip)
            with client.stream(
                "GET",
                request_url,
                headers=host_header,
                extensions={"sni_hostname": urlparse(url).hostname},
            ) as r:
                status_code = getattr(r, "status_code", 200)
                location = r.headers.get("location")
                if status_code in {301, 302, 303, 307, 308} and location:
                    url = urljoin(url, location)
                    continue

                r.raise_for_status()
                content_type = r.headers.get("content-type", "").lower()
                if "html" not in content_type and "text" not in content_type:
                    return (
                        "Это не текстовая страница "
                        f"(тип: {content_type or 'неизвестен'})."
                    )
                raw_body = _read_body_within_limit(r, READ_URL_MAX_BYTES)
                encoding = getattr(r, "encoding", None) or "utf-8"
            break
        else:
            return f"Слишком много перенаправлений (>{MAX_URL_REDIRECTS})."

        soup = BeautifulSoup(
            raw_body.decode(encoding, errors="replace"), "html.parser"
        )
        for tag in soup(
            [
                "script",
                "style",
                "noscript",
                "header",
                "footer",
                "nav",
                "aside",
                "svg",
                "form",
                "iframe",
            ]
        ):
            tag.decompose()

        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        text = " ".join(soup.get_text(separator=" ").split())
        if not text:
            return "Страница загрузилась, но текста на ней нет."

        result = f"Заголовок: {title}\n\n" if title else ""
        result += text[:max_chars]
        if len(text) > max_chars:
            result += "…"
        return result


def read_url(url: str, max_chars: int = READ_URL_MAX_CHARS) -> str:
    """Validate URL against SSRF, then read via DeepSeek Fetch or direct HTTP."""
    url = (url or "").strip()
    if not url:
        return "Пустой URL."
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        # Security boundary: always validate scheme, credentials, and public IP first.
        _validate_public_url(url)

        if SEARCH_PROVIDER == "deepseek" and DEEPSEEK_API_KEY:
            try:
                return _deepseek_fetch_url(url, max_chars=max_chars)
            except Exception as exc:
                logger.warning(
                    "🔗 [yellow]DeepSeek Fetch API недоступен (%s), читаем напрямую[/]",
                    exc,
                )

        return _direct_read_url(url, max_chars=max_chars)
    except ValueError as e:
        return f"URL отклонён: {e}"
    except httpx.HTTPStatusError as e:
        return f"Страница недоступна (HTTP {e.response.status_code})."
    except Exception as e:
        return f"Ошибка чтения страницы: {e}"
