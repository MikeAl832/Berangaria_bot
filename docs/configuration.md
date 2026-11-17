# Configuration Reference

Complete reference for Berangaria Bot configuration and deployment.

## Environment Variables (`.env`)

Copy `.env.example` to `.env`:

```env
TELEGRAM_BOT_TOKEN=<bot_token_from_botfather>
DEEPSEEK_API_KEY=<deepseek_api_key>
GEMINI_API_KEY=<gemini_api_key>
# Optional — required only when chat_provider: "openrouter"
# OPENROUTER_API_KEY=<openrouter_api_key>
```

*Legacy note*: `API_KEY` is still accepted as a fallback alias for `DEEPSEEK_API_KEY`.

| Variable | Required | Purpose |
|----------|----------|---------|
| `TELEGRAM_BOT_TOKEN` | Yes | Telegram Bot API authentication (`@BotFather`) |
| `DEEPSEEK_API_KEY` | Yes | DeepSeek chat, summarization, memory extractor/verifier, and Search/Fetch API |
| `GEMINI_API_KEY` | Yes | Gemini video/audio understanding, optional fallback image descriptions, and Qdrant embeddings |
| `OPENROUTER_API_KEY` | Only if `chat_provider: "openrouter"` | OpenRouter Chat Completions gateway |

---

## `config.yaml` Parameters

### Main LLM (`deepseek` or `openrouter`)

```yaml
chat_provider: "deepseek"                 # "deepseek" | "openrouter"
model: "deepseek-flash"
chat_api_url: "https://api.deepseek.com/chat/completions"
max_context_tokens: 32000
max_reply_tokens: 4096

generation_params:
  temperature: 1.0
  thinking:
    type: "disabled"

summary_generation_params:
  temperature: 0.3
  thinking:
    type: "enabled"

openrouter_provider: null
```

- `chat_provider` (`BOT_CHAT_PROVIDER` env override):
  - `"deepseek"` (default): sends standard Chat Completions requests to `api.deepseek.com` using `DEEPSEEK_API_KEY`. No OpenRouter-specific `session_id` or `provider` fields are injected.
  - `"openrouter"`: sends requests to OpenRouter using `OPENROUTER_API_KEY`, attaches `HTTP-Referer`, `X-Title`, sticky `x-session-id` / `session_id`, and (when `openrouter_provider` is a mapping, e.g. `{"only": ["deepseek"], "allow_fallbacks": false}`) pins the upstream provider.
- `model`: Required chat/summarization model identifier.
- `generation_params`: Sampling parameters merged into ordinary chat requests.
- `summary_generation_params`: Sampling parameters merged into `/summarize` and automatic history compression requests.
- **Prefix caching invariants**: Persisted history separates display `content` from raw `provider_messages` and tracks `provider_sent`. Already-sent rows are never mutated by incoming emoji reactions or late Telegram edits, preserving provider KV-cache hits (`prompt_cache_hit_tokens` on DeepSeek / `cached_tokens` on OpenRouter).

### Vision & Media

```yaml
vision_mode: true
image_vision_provider: "chat"             # "chat" | "gemini"
gemini_model: "gemini-3.5-flash-lite"
video_max_duration_sec: 300
audio_max_duration_sec: 300
```

- `vision_mode`: Master switch for image, sticker, video, and audio understanding.
- `image_vision_provider` (`BOT_IMAGE_VISION_PROVIDER` env override):
  - `"chat"` (default): photos, `media_group_id` albums, and static WebP stickers are cached in bounded memory (`media_image_cache`) and attached directly to the chat model as OpenAI `image_url` (`data:image/...;base64,...`) blocks. Cached images remain attached in the active history window until scheduled summarization compresses older turns, preserving KV-cache prefixes.
  - `"gemini"`: describes photos/albums/stickers in Russian via `gemini_model` and injects `[Image description: ...]`. Use this when switching `model` to a text-only LLM.
- **Videos, circles, video stickers, and voice/audio**: Analyzed/transcribed via Gemini (`gemini_model`). Media downloads use the cloud Bot API `getFile` path (up to 20 MiB; configurable down via `BOT_VIDEO_MAX_FILE_SIZE_BYTES` and timeout via `TELEGRAM_MEDIA_TIMEOUT_SECONDS`).

### Seeing Other Bots in Groups (Native Bot API)

No Telethon client or user session is needed. To allow Berangaria to see messages from another bot in an allowlisted group:
1. Open `@BotFather` and enable **Bot-to-Bot Communication Mode** on **both** bots.
2. Keep Berangaria as a group admin (or disable Group Privacy).
3. Incoming messages with `from_user.is_bot == True` are automatically tagged `[Bot: Name]` in chat history and excluded from long-term memory. Unauthorized private DMs from stranger bots are dropped silently to prevent bot-to-bot loops.

### Web Search & URL Reading

```yaml
search_provider: "deepseek"               # "deepseek" | "ddgs"
web_search_max_per_turn: 2
read_url_max_per_turn: 4
web_tool_max_per_turn: 6
```

- `search_provider` (`BOT_SEARCH_PROVIDER` env override):
  - `"deepseek"` (default): uses DeepSeek's Search (`POST /v1/tools/search`) and Fetch (`POST /v1/tools/fetch`) endpoints with `DEEPSEEK_API_KEY`, falling back to `ddgs` / direct SSRF-pinned HTTP HTML extraction on transient errors.
  - `"ddgs"`: uses DuckDuckGo (`ddgs`) and the direct SSRF-pinned HTTP reader.
- `read_url` always validates `http`/`https` scheme, rejects URL credentials, and rejects private/loopback/link-local IPs before any request.

### Memory (`Mem0` + `Qdrant` + `SQLite`)

```yaml
mem0_llm_model: "deepseek-flash"
embedding_model: "gemini-embedding-2"
embedding_dims: 768
memory_search_limit: 8
memory_min_score: 0.3
memory_max_chars: 1200
memory_flush_interval_seconds: 900
```

- Every non-bot, non-forwarded user message is queued in SQLite (`memory_sources`). While a message is still `waiting` or `pending` (before history is frozen/processed), a Telegram edit updates its queued text; once frozen or processed, edits do not rewrite memory.
- Only confirmed turns (`DELIVERED` or intentional `SILENT`) release sources to the background worker; failed/interrupted turns abandon them.
- `deepseek-flash` extracts candidates with verbatim quotes, an independent verifier approves or discards them fail-closed, and approved facts are committed atomically to SQLite (`memory_facts`) + Mem0/Qdrant (`infer=False`).
- Changing `embedding_model` or `embedding_dims` requires recreating both `mem0` and `stickers` Qdrant collections.
- For Docker-free local runs, set `QDRANT_PATH=qdrant_local` in `.env` or your shell: both Mem0 and sticker search share an embedded on-disk Qdrant directory instead of connecting to `QDRANT_HOST:QDRANT_PORT`.

### Stickers & Multi-Message Bursts

```yaml
sticker_enabled: true
sticker_collection: "stickers"
sticker_dims: 768
sticker_min_score: 0.25
sticker_top_k: 8
sticker_auto_sync: true
sticker_sync_file: "data/stickers_clean.json"
sticker_index_version: 3
sticker_send_max_per_turn: 2

multi_message_max: 5
multi_message_max_chars: 280
multi_message_max_total_chars: 900
```

- `send_sticker(query)` performs a single-step vector search in Qdrant and posts a random match among the top hits above `sticker_min_score`.
- `send_messages` sends 1–5 short bubbles, each with an optional `reply_to` (`[#N]` handle from current chat history).

### Access Control, Alerts & Cost Tracking

```yaml
allowed_users: [1217938322, 1809564460, 5589809444]
allowed_groups: [-5578136180]
admin_alert_chat_id: null

price_prompt_cache_miss: 0.15
price_prompt_cache_hit: 0.003
price_prompt_cache_write: 0.15
price_completion: 0.60
```

- The first ID in `allowed_users` is the authenticated owner (`OWNER_USER_ID`), represented by server-generated `[Owner: Name]` metadata.
- `admin_alert_chat_id`: `null` sends throttled critical alerts to `OWNER_USER_ID`; set a single integer chat ID to override.
- `price_*` (USD per 1M tokens) are required on startup. Provider-reported `usage.cost` takes precedence when present.

---

## Universal Deployment

### Standard VPS / Docker Compose

1. Clone the repository on any Linux server with Docker & Docker Compose installed:
   ```bash
   git clone https://github.com/MikeAl832/Berangaria_bot.git
   cd Berangaria_bot
   cp .env.example .env
   nano .env
   docker compose up -d --build
   ```
2. Persistent state lives in `./bot_data` (`bot_state.db`, `bot.log`, `loop_heartbeat`) and `./qdrant_storage`.

### Optional GitHub Actions SSH Auto-Deploy

`.github/workflows/deploy.yml` runs `py_compile`, `ruff`, `bandit`, and `pytest` on every push and PR.
To enable automatic SSH deployment on pushes to `main`:
1. Set repository variable `DEPLOY_ENABLED=true` in **GitHub → Settings → Secrets and variables → Actions → Variables**.
2. Add repository secrets `DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY`, `DEPLOY_PATH` (and optional `DEPLOY_PORT`).
