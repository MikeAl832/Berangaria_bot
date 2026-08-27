# Переход чата на OpenRouter

Штатный чат — прямой `api.x.ai`. OpenRouter в коде **нет**: ни `session_id` в теле, ни `provider.only`, ни ключа `OPENROUTER_API_KEY`. Это нарочно. Развилка «вдруг вернёмся» раздувала `config.py` и ломала кэш Grok, потому что шлюз пинит хостера, а KV-кэш живёт на конкретной реплике xAI.

Вернуть OpenRouter можно. Это не «поменять URL». Нужны yaml, секрет и снова несколько строк в транспорте. Этот файл — что именно и почему.

## Зачем вообще уходили

Grok кэширует префикс **на сервере**. Заголовок `x-grok-conv-id` клеит чат к одной машине. OpenRouter `session_id` клеит только к провайдеру `xAI`. Дальше их балансировщик кидает ARN ↔ HEL (и разные реплики внутри региона). Префикс байт-в-байт тот же, `cached_tokens` прыгает 97% → 0%.

Официальный passthrough заголовков у OpenRouter — вроде `x-anthropic-beta`. `x-grok-conv-id` в этом списке нет. Пока шлюз его не форвардит, на Grok через OpenRouter кэш почти наверняка снова будет пилой. Имеет смысл возвращаться ради фолбэка на Bedrock, единого счёта или другой модели — не ради hit-rate Grok.

## Что уже есть и трогать не надо

Это общее для обоих хостов. Ломать нельзя, иначе кэш умрёт и на прямом xAI:

- Стабильный id чата: `berangaria.chat.llm_client._chat_session_id(history_key)` — sha256 ключа истории, без Telegram id, ≤256 символов.
- Дата — вторым `system`, не внутри personality prompt.
- Память дописывается в **последнее** user-сообщение, не в начало.
- Уже отправленный префикс не переписывается (реакции — в хвост).
- `reasoning_details` эхом как пришли.
- `provider_messages` — сырой assistant/tool след, `content` — то, что ушло в Telegram.

В логе смотри `🧭 Маршрут: ... cache=`. Цель после прогрева 70–90%. Первый запрос чата холодный. Редкий мисс после простоя — норма. Пила 97/0/97 на соседних репликах — нет.

## Переменные и yaml

`.env`:

```env
OPENROUTER_API_KEY=<ключ с openrouter.ai/keys>
```

Сейчас загрузчик читает только `XAI_API_KEY` или `CHAT_API_KEY`. Пока код не научится брать `OPENROUTER_API_KEY`, временно можно:

```env
CHAT_API_KEY=<тот же ключ OpenRouter>
```

`config.yaml`:

```yaml
model: "x-ai/grok-4.6"
chat_api_url: "https://openrouter.ai/api/v1/chat/completions"
```

Slug `grok-4.6` на шлюзе не работает — нужен `x-ai/grok-4.6`.

Цены (`price_prompt_cache_*`, `price_completion`) для Grok 4.6 те же: $2 / $0.50 cached / $6 за 1M, ниже 200K prompt. Если OpenRouter вернёт `usage.cost`, лог берёт его.

DeepSeek `API_KEY` и Gemini не связаны с этим переключением.

## Что обязательно вернуть в код

Иначе получится «URL OpenRouter, кэш как без affinity».

### 1. Ключ

В `berangaria/config.py` для URL OpenRouter читать `OPENROUTER_API_KEY` (с запасным `CHAT_API_KEY`). Не слать ключ xAI на шлюз и наоборот.

### 2. Тело запроса — `session_id`, не `provider.order`

В `send_llm_request` и суммаризации, тот же id что сейчас в заголовке:

```python
payload["session_id"] = session_id  # _chat_session_id(key)
```

Липкость OpenRouter: один `session_id` → один хостер, сбрасывается после ~10 минут тишины. Без него шлюз хеширует первые сообщения и может сменить endpoint.

Хост ограничивать так:

```python
payload["provider"] = {
    "only": ["xai"],
    "allow_fallbacks": False,
}
```

| Делать | Не делать |
|---|---|
| `provider.only` | `provider.order` — **выключает** sticky routing |
| `allow_fallbacks: false`, если нужен только xAI | `sort: "price"` — прыгает между равноценными пулами xAI и сбрасывает кэш |
| Один стабильный `session_id` на history key | Новый id на каждый ход или tool-round |

Фолбэк `amazon-bedrock` (~+10% к цене) — только если сознательно принимаешь холодный кэш при уходе с xAI.

### 3. Заголовки

`chat_api_headers` на пути OpenRouter:

```
Authorization: Bearer <OPENROUTER_API_KEY>
Content-Type: application/json
x-session-id: <тот же session_id>
x-grok-conv-id: <тот же>          # на случай если шлюз начнёт форвардить
HTTP-Referer: ...                 # опционально, атрибуция
X-Title: Berangaria               # опционально
X-OpenRouter-Metadata: enabled    # опционально, диагностика region/endpoint
```

Прямому `api.x.ai` **нельзя** слать `session_id` и `provider` в JSON — неизвестные поля могут дать 400. Если снова появится развилка, режь её по URL, не держи оба набора полей «на всякий».

Точки: `berangaria/config.py` (`CHAT_API_KEY`, `chat_api_headers`), `berangaria/chat/llm_client.py` (тело чата), `berangaria/chat/summarization.py` (тоже Completions).

## Как проверить, что кэш живой

После выкладки 10–15 реплик подряд в одном ЛС:

```
🧭 Маршрут: provider=... model=x-ai/grok-4.6 session=............ cache=...
```

- `cache=` 70–90% на втором и дальше запросах того же чата — ок.
- Снова 97% ↔ 0%/2% при том же `session=` — шлюз не держит реплику xAI. Это ожидаемо для Grok через OpenRouter, пока нет форварда `x-grok-conv-id`. Тогда либо мириться с ценой, либо обратно на `api.x.ai`.
- Если включил `X-OpenRouter-Metadata`, в ответе будет `openrouter_metadata` (region/endpoint). Скачки ARN↔HEL при том же session — тот же диагноз.

Не путай с Mem0: `🧠 Память` к prompt cache не относится.

## Обратно на прямой xAI

```yaml
model: "grok-4.6"
chat_api_url: "https://api.x.ai/v1/chat/completions"
```

```env
XAI_API_KEY=<ключ с console.x.ai>
```

В запросе снова только `x-grok-conv-id`. Без `session_id` и без `provider` в теле.
