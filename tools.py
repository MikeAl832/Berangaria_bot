import time
import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import httpx
from ddgs import DDGS
from bs4 import BeautifulSoup

# Простой rate limiter для web_search
_search_timestamps = []
MAX_SEARCHES_PER_MINUTE = 10

def _check_rate_limit() -> bool:
    """Проверяет, не превышен ли лимит запросов. Возвращает True если можно делать запрос."""
    current_time = time.time()
    # Удаляем запросы старше 60 секунд
    _search_timestamps[:] = [ts for ts in _search_timestamps if current_time - ts < 60]
    
    if len(_search_timestamps) >= MAX_SEARCHES_PER_MINUTE:
        return False
    
    _search_timestamps.append(current_time)
    return True

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Mandatory search for prices, specs, news, dates after 2023. Then give an answer with numbers — don't say 'rumored' or 'no data'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query in the most relevant language"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results, 3-8",
                        "default": 5
                    },
                    "timelimit": {
                        "type": "string",
                        "description": "Time filter: 'd'=day, 'w'=week, 'm'=month, 'y'=year",
                        "enum": ["d", "w", "m", "y"]
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "react_to_message",
            "description": (
                "Put an emoji reaction badge on a message (NOT in your text). "
                "This is a real Telegram action — the emoji appears next to that message. "
                "Use it freely and often to show emotions: agreement 👍, laughter 😂, shock 😱, "
                "trolling 🤡, approval 🔥. PREFER reaction-only (no text) for simple acknowledgment. "
                "Add text only if you have something specific to say. "
                "By default it reacts to the latest message; pass the [#N] handle as 'id' to react to a specific earlier one. "
                "Do NOT react again to a message you already reacted to (history notes look like "
                "'Ты поставила реакцию 🤡 на [#N] …') — pick another [#N], write text, send a sticker, or stay silent. "
                "NEVER fake it in text (no '*reacts with 🔥*') — call this function instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "emoji": {
                        "type": "string",
                        "description": "A single emoji character to react with (a common Telegram reaction)."
                    },
                    "id": {
                        "type": "integer",
                        "description": "Optional [#N] handle of the message to react to. Omit to react to the latest message."
                    }
                },
                "required": ["emoji"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reply_to_message",
            "description": (
                "Send your message as a Telegram REPLY to a specific earlier message, identified by its [#N] handle "
                "(shown at the very start of each incoming message, e.g. [#7]). "
                "In a normal back-and-forth you do NOT need this — just answer with plain text. "
                "Use it only when you deliberately want to answer an EARLIER or different message than the latest one "
                "(e.g. you were pinged and want to pick up something said a few messages ago). "
                "Pass the number from [#N] as 'id' and your reply as 'text'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "The [#N] handle number of the message you want to reply to."
                    },
                    "text": {
                        "type": "string",
                        "description": "Your reply text (plain text, no service tags, no emoji)."
                    }
                },
                "required": ["id", "text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_stickers",
            "description": (
                "Search your sticker collection by vibe. Returns a NUMBERED list of matching stickers "
                "with descriptions and tags — it does NOT send anything. Use it when you feel like reacting "
                "with a sticker: browse the options, then send the one that best fits via send_sticker(id). "
                "You may refine the search up to 3 times per turn with different wording; after that pick "
                "from already found numbers or answer without a sticker. If none fit, don't send. Don't overuse stickers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Vivid description of the sticker's mood/content, in Russian, e.g. "
                            "'недоумение, кто-то сморозил глупость' or 'ржу в голос' or 'одобряю, огонь'."
                        )
                    },
                    "count": {
                        "type": "integer",
                        "description": "How many candidates to return, 3-10 (default 6)."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_sticker",
            "description": (
                "Send ONE specific sticker to the chat, chosen from the latest find_stickers results by its number. "
                "Call this only AFTER find_stickers, passing the id of the option you liked best. "
                "After a successful send the turn ends — the sticker IS the full reply (no extra text round). "
                "NEVER describe a sticker in text ('*кидает стикер*') — send it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "The number of the sticker from find_stickers results (e.g. 3)."
                    }
                },
                "required": ["id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_url",
            "description": (
                "Download a web page by URL and read its text content. "
                "Use when the user sends a link or asks to analyze/comment on a specific URL. "
                "Don't use for general questions — use web_search for those."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full page URL (with http:// or https://)"
                    }
                },
                "required": ["url"]
            }
        }
    }
]

# Разрешённые Telegram эмодзи для реакций (для валидации перед вызовом API)
ALLOWED_REACTIONS = {
    "👍", "👎", "❤", "🔥", "🥰", "👏", "😁", "🤔", "🤯", "😱", "🤬", "😢", "🎉",
    "🤩", "🤮", "💩", "🙏", "👌", "🕊", "🤡", "🥱", "🥴", "😍", "🐳", "❤‍🔥", "🌚",
    "🌭", "💯", "🤣", "⚡", "🍌", "🏆", "💔", "🤨", "😐", "🍓", "🍾", "💋", "🖕",
    "😈", "😴", "😭", "🤓", "👻", "👨‍💻", "👀", "🎃", "🙈", "😇", "😨", "🤝", "✍",
    "🤗", "🫡", "🎅", "🎄", "☃", "💅", "🤪", "🗿", "🆒", "💘", "🙉", "🦄", "😘",
    "💊", "🙊", "😎", "👾", "🤷‍♂", "🤷", "🤷‍♀", "😡",
}

def web_search(query: str, max_results: int = 5, timelimit: str = None, region: str = "ru-ru") -> str:
    # Проверка rate limit
    if not _check_rate_limit():
        return f"⚠️ Превышен лимит поисковых запросов ({MAX_SEARCHES_PER_MINUTE}/мин). Попробуйте позже."
    
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


def _validate_public_url(url: str) -> None:
    """Отклоняет URL, способные обратиться к локальной/внутренней сети."""
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
        addresses = {str(literal)}
    except ValueError:
        try:
            addr_info = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise ValueError("Не удалось определить адрес сайта.") from exc
        addresses = {item[4][0].split("%", 1)[0] for item in addr_info}
    if not addresses:
        raise ValueError("Сайт не имеет доступных IP-адресов.")

    for raw_ip in addresses:
        ip = ipaddress.ip_address(raw_ip)
        if not ip.is_global:
            raise ValueError("Доступ к локальным и служебным сетевым адресам запрещён.")

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
    try:
        with httpx.Client(timeout=15.0, follow_redirects=False, headers=headers) as client:
            for _ in range(MAX_URL_REDIRECTS + 1):
                _validate_public_url(url)
                r = client.get(url)
                status_code = getattr(r, "status_code", 200)
                location = r.headers.get("location")
                if status_code in {301, 302, 303, 307, 308} and location:
                    url = urljoin(url, location)
                    continue
                break
            else:
                return f"Слишком много перенаправлений (>{MAX_URL_REDIRECTS})."

            # Защита от клиента/прокси, который мог последовать редиректу сам.
            final_url = str(getattr(r, "url", url))
            _validate_public_url(final_url)
            r.raise_for_status()

            content_type = r.headers.get("content-type", "").lower()
            if "html" not in content_type and "text" not in content_type:
                return f"Это не текстовая страница (тип: {content_type or 'неизвестен'})."

            soup = BeautifulSoup(r.text, "html.parser")

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
