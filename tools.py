import time
from ddgs import DDGS

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
    }
]

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
