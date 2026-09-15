# OpenRouter chat route

Один шлюз: `https://openrouter.ai/api/v1/chat/completions`, модель
`meta/muse-spark-1.3`, хостер Meta, ключ `OPENROUTER_API_KEY`.
Другого chat-шлюза в коде нет.

Модель поддерживает tools, structured outputs, `temperature` и
настраиваемый reasoning. Reasoning обязателен; обычный чат
использует `low`, суммаризация — `high`.

## Что уходит в каждый запрос

Тело чата:

```json
{
  "model": "meta/muse-spark-1.3",
  "session_id": "berangaria-<sha256(history_key)>",
  "provider": {
    "only": ["meta"],
    "allow_fallbacks": false,
    "require_parameters": true
  },
  "reasoning": { "effort": "low" },
  "messages": [ ... ],
  "tools": [ ... ]
}
```

Заголовки:

```text
Authorization: Bearer <OPENROUTER_API_KEY>
x-session-id: <тот же session_id>
HTTP-Referer: https://github.com/MikeAl832/Berangaria_bot
X-Title: Berangaria
```

`session_id` один на весь чат (`private_X` / `group_Y`), а не на ход или
tool-round. В теле и заголовке используется одно значение.

| Поле | Зачем |
|---|---|
| `session_id` / `x-session-id` | стабильная routing affinity и диагностика одного чата |
| `provider.only: ["meta"]` | разрешить только хостер Meta |
| `allow_fallbacks: false` | не переходить на другого хостера |
| `require_parameters: true` | отклонить маршрут, который не поддерживает поля запроса |
| `reasoning.effort: low` | баланс tool-надёжности, цены и задержки |
| суммаризация `high` | отдельный запрос для точного сжатия фактов |

В логе: `🧭 Маршрут: provider=Meta model=... session=... cache=...`.
Ответ с другим provider создаёт критический алерт владельцу. Строка
токенов разделяет cache read и cache write, а строка цены отмечает
`usage.cost` или локальную оценку.

Текущий Meta endpoint не объявляет implicit caching. Нельзя считать
стабильный `session_id` доказательством cache hit: его подтверждают
только `usage.prompt_tokens_details.cached_tokens` и фактический
`usage.cost`.

Не путать с Mem0: `🧠 Память` к prompt cache не относится.

## Что держит сам префикс messages

- Personality system — первое сообщение, без даты.
- Календарный день — вторым `system`, а не внутри personality.
- Память дописывается в последнее user-сообщение, а не в начало.
- Уже отправленный префикс не переписывается; реакции добавляются в хвост.
- `provider_messages` хранит точный assistant/tool след, `content` — текст из Telegram.
- `reasoning_details` передаётся обратно без пересборки или сокращения.

## Чего не делать

| Делать | Не делать |
|---|---|
| `provider.only: ["meta"]` | `provider.order` или `sort`, разрешающие дрейф endpoint |
| `allow_fallbacks: false` | скрыто переходить к другому хостеру |
| `require_parameters: true` | молча терять неподдерживаемые поля |
| один `session_id` на history key | новый id на ход или tool-round |
| `reasoning.effort: low` в чате | `none`, который Muse не поддерживает |
| `high` только в суммаризации | поднимать весь обычный чат до `medium`/`high` |
| обычный `meta/muse-spark-1.3` | `-contributor` без отдельного решения о data policy |

Точки в коде: `berangaria/config.py` (`chat_api_headers`,
`apply_chat_gateway`), `berangaria/chat/llm_client.py`,
`berangaria/chat/summarization.py`.

Актуальные capabilities и цены: [OpenRouter Muse Spark 1.3](https://openrouter.ai/meta/muse-spark-1.3).
