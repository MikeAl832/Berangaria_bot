from ddgs import DDGS

def web_search(query: str, max_results: int = 5, timelimit: str = None, region: str = "ru-ru") -> str:
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
