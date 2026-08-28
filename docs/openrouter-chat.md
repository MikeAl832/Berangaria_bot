# OpenRouter chat cache

Один шлюз: `https://openrouter.ai/api/v1/chat/completions`, модель
`openai/gpt-5.6-terra`, ключ `OPENROUTER_API_KEY`. Прямого xAI в коде нет.

Кэш здесь двухслойный. OpenRouter клеит запросы к **одному хостеру**. OpenAI
кэширует **байт-стабильный префикс** messages. Без первого слоя второй живёт
только пока балансировщик случайно попадёт на ту же машину.

## Что уходит в каждый запрос

Тело (чат и суммаризация):

```json
{
  "model": "openai/gpt-5.6-terra",
  "session_id": "berangaria-<sha256(history_key)>",
  "provider": { "only": ["openai"], "allow_fallbacks": false },
  "reasoning": { "effort": "low" },
  "messages": [ ... ],
  "tools": [ ... ]
}
```

Заголовки:

```
Authorization: Bearer <OPENROUTER_API_KEY>
x-session-id: <тот же session_id>
HTTP-Referer: https://github.com/MikeAl832/Berangaria_bot
X-Title: Berangaria
```

`session_id` в теле важнее заголовка, если вдруг разойдутся. Один id на весь
чат (`private_X` / `group_Y`), не на ход и не на tool-round. Длина ≤256.

| Поле | Зачем |
|---|---|
| `session_id` / `x-session-id` | sticky routing с первого успешного ответа, не после первого cache hit |
| `provider.only: ["openai"]` | не прыгать на Azure/Bedrock — у них другой KV-кэш |
| `allow_fallbacks: false` | падение OpenAI не «спасается» чужим хостером ценой холодного кэша |
| `reasoning.effort: low` | чат+tools по гайду OpenAI; `none` — если снова упрётесь в CoT-налог |
| суммаризация `high` | отдельный запрос, как на Grok/Luna; не наследует low чата |

В логе: `🧭 Маршрут: provider=OpenAI ... cache=`. После прогрева цель 80–90%.
Первый запрос чата холодный. Пила 90/0/90 при том же `session_id` — хостер сменился
или префикс messages переписали.

Не путай с Mem0: `🧠 Память` к prompt cache не относится.

## Что держит сам префикс messages

Это уже не поля OpenRouter, а как собран prompt. Ломать нельзя:

- Personality system — первое сообщение, без даты.
- Календарный день — вторым `system`, не внутри personality.
- Память дописывается в **последнее** user-сообщение, не в начало.
- Уже отправленный префикс не переписывается (реакции — в хвост).
- `provider_messages` — сырой assistant/tool след, `content` — то, что ушло в Telegram.

OpenAI автоматически кэширует префикс, если он ≥1024 токенов. System prompt + tools
это закрывают. Меняется хвост (новый user, tool results текущего хода) — это
нормальный miss только на хвосте; начало должно оставаться cache hit.

## Чего не делать

| Делать | Не делать |
|---|---|
| `provider.only: ["openai"]` | `provider.order` — выключает sticky routing |
| `allow_fallbacks: false` | `sort: "price"` — прыжки OpenAI ↔ Azure ↔ Bedrock |
| Один `session_id` на history key | Новый id на ход или tool-round |
| `reasoning.effort: low` в чате | `medium`/`high` на каждый пинг в группе |
| `high` только в суммаризации | `top_k` / `min_p` — OpenAI их не ест |

Точки в коде: `berangaria/config.py` (`chat_api_headers`, `apply_chat_gateway`),
`berangaria/chat/llm_client.py`, `berangaria/chat/summarization.py`.
