import time
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
                "Put an emoji reaction badge on the user's message (NOT in your text). "
                "This is a real Telegram action — the emoji appears next to their message. "
                "Use it freely and often to show emotions: agreement 👍, laughter 😂, shock 😱, "
                "trolling 🤡, approval 🔥. PREFER reaction-only (silent, no text) for simple acknowledgment. "
                "Add text only if you have something specific to say. "
                "NEVER fake it in text (no '*reacts with 🔥*') — call this function instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "emoji": {
                        "type": "string",
                        "description": "A single emoji character to react with (a common Telegram reaction)."
                    }
                },
                "required": ["emoji"]
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
        with httpx.Client(timeout=15.0, follow_redirects=True, headers=headers) as client:
            r = client.get(url)
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

    except httpx.HTTPStatusError as e:
        return f"Страница недоступна (HTTP {e.response.status_code})."
    except Exception as e:
        return f"Ошибка чтения страницы: {e}"
