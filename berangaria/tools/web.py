import time
import ipaddress
import socket
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from ddgs import DDGS
from bs4 import BeautifulSoup

# Простой rate limiter для web_search. Общий на процесс: один разошедшийся чат
# тратит лимит всех остальных, поэтому у хода есть ещё и свой потолок (dispatch).
_search_timestamps = []
MAX_SEARCHES_PER_MINUTE = 10

# По этому префиксу диспетчер отличает отказ лимитера от честного «не нашлось»:
# для модели это разные ситуации, а текст один и тот же.
RATE_LIMIT_PREFIX = "⚠️ Превышен лимит поисковых запросов"

def _check_rate_limit() -> bool:
    """Проверяет, не превышен ли лимит запросов. Возвращает True если можно делать запрос."""
    current_time = time.time()
    # Удаляем запросы старше 60 секунд
    _search_timestamps[:] = [ts for ts in _search_timestamps if current_time - ts < 60]
    
    if len(_search_timestamps) >= MAX_SEARCHES_PER_MINUTE:
        return False
    
    _search_timestamps.append(current_time)
    return True


def web_search(query: str, max_results: int = 5, timelimit: str = None, region: str = "ru-ru") -> str:
    # Проверка rate limit
    if not _check_rate_limit():
        return f"{RATE_LIMIT_PREFIX} ({MAX_SEARCHES_PER_MINUTE}/мин). Попробуйте позже."
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                query,
                max_results=max_results,
                region=region,
                timelimit=timelimit,
                safesearch="off"
            ))

        if not results:
            return f"По запросу '{query}' ничего не найдено."

        output = ""
        for i, r in enumerate(results, 1):
            output += f"{i}. {r['title']}\n{r['body']}\n{r['href']}\n\n"
        return output.strip()

    except Exception as e:
        return f"Ошибка поиска: {e}"


READ_URL_MAX_CHARS = 4000  # сколько символов текста страницы отдавать модели
MAX_URL_REDIRECTS = 5
READ_URL_MAX_BYTES = 4 * 1024 * 1024   # потолок тела ответа
READ_URL_TOTAL_TIMEOUT = 30.0          # дедлайн на всю цепочку редиректов


def _validate_public_url(url: str) -> str:
    """Отклоняет URL, способные обратиться к локальной/внутренней сети.

    Возвращает провалидированный IP-адрес. Возврат обязателен: если подключаться
    по имени, httpx сделает собственный resolve, и хост с чередующимися
    A-записями (DNS rebinding) подставит внутренний адрес уже после проверки.
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
        # IP-литерал проверяем напрямую: это заодно не даёт тестовым/системным
        # DNS-резолверам подменить смысл 127.0.0.1 или ::1.
        literal = ipaddress.ip_address(parsed.hostname.split("%", 1)[0])
        addresses = [str(literal)]
    except ValueError:
        try:
            addr_info = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise ValueError("Не удалось определить адрес сайта.") from exc
        # dict.fromkeys вместо set: порядок резолвера сохраняется, и мы
        # подключаемся именно к тому адресу, который проверили первым.
        addresses = list(dict.fromkeys(item[4][0].split("%", 1)[0] for item in addr_info))
    if not addresses:
        raise ValueError("Сайт не имеет доступных IP-адресов.")

    for raw_ip in addresses:
        ip = ipaddress.ip_address(raw_ip)
        if not ip.is_global:
            raise ValueError("Доступ к локальным и служебным сетевым адресам запрещён.")

    return addresses[0]


def _pinned_request_args(url: str, validated_ip: str) -> tuple[str, dict[str, str]]:
    """Возвращает URL с закреплённым адресом и заголовок Host для него.

    Подключаемся к уже проверенному IP, а имя хоста передаём в Host/SNI —
    между проверкой и соединением не остаётся второго DNS-запроса.
    """
    parsed = urlparse(url)
    host_header = parsed.netloc.split("@", 1)[-1]
    ip = ipaddress.ip_address(validated_ip)
    literal = f"[{validated_ip}]" if ip.version == 6 else validated_ip
    netloc = f"{literal}:{parsed.port}" if parsed.port else literal
    return urlunparse(parsed._replace(netloc=netloc)), {"Host": host_header}


def _read_body_within_limit(response, max_bytes: int) -> bytes:
    """Читает тело потоком, обрывая передачу на достижении потолка."""
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

def read_url(url: str, max_chars: int = READ_URL_MAX_CHARS) -> str:
    """Скачивает страницу и возвращает её текст (заголовок + основной контент)."""
    url = (url or "").strip()
    if not url:
        return "Пустой URL."
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    timeout = httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=5.0)
    deadline = time.monotonic() + READ_URL_TOTAL_TIMEOUT
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False, headers=headers) as client:
            for _ in range(MAX_URL_REDIRECTS + 1):
                if time.monotonic() > deadline:
                    return "Превышено общее время чтения страницы."
                validated_ip = _validate_public_url(url)
                request_url, host_header = _pinned_request_args(url, validated_ip)
                # stream: content-type и размер проверяем ДО чтения тела, иначе
                # ссылка на многогигабайтный файл выкачивается целиком впустую.
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

            soup = BeautifulSoup(raw_body.decode(encoding, errors="replace"), "html.parser")

            # Выкидываем неинформативные блоки
            for tag in soup(["script", "style", "noscript", "header", "footer",
                             "nav", "aside", "svg", "form", "iframe"]):
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

    except ValueError as e:
        return f"URL отклонён: {e}"
    except httpx.HTTPStatusError as e:
        return f"Страница недоступна (HTTP {e.response.status_code})."
    except Exception as e:
        return f"Ошибка чтения страницы: {e}"
