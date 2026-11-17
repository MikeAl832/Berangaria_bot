# Berangaria Bot

Telegram bot built on Python 3.11 and `python-telegram-bot` (cloud Bot API), designed as a sharp-witted equal participant for 1-on-1 DMs and group chats. Combines prefix-cache-first conversation state, strict fail-closed long-term memory, native multimodal vision, and live web search.

## Stack & Architecture

- **Chat & Summarization Gateway**:
  - **Default (`chat_provider: "deepseek"`)**: Direct **DeepSeek API** (`deepseek-flash` / DeepSeek V4.1 Flash) — low-latency chat with `thinking: disabled`, scheduled history compression with `thinking: enabled` (`temperature: 0.3`), and native OpenAI-compatible `image_url` vision.
  - **Configurable alternative (`chat_provider: "openrouter"`)**: One-line switch in `config.yaml` with custom `generation_params`, `summary_generation_params`, and optional `openrouter_provider` routing rules.
- **Web Search & URL Fetching**: Native DeepSeek Search (`POST /v1/tools/search`) and Fetch (`POST /v1/tools/fetch`) APIs with mandatory SSRF validation (HTTP/HTTPS only, no credentials, public global IPs only) and automatic `ddgs` / DNS-pinned direct HTTP fallback.
- **Strict Long-Term Memory**: Durable SQLite source queue (`memory_sources`) -> DeepSeek candidate extractor -> independent fail-closed DeepSeek verifier with literal `source_quote` provenance -> atomic SQLite registry (`memory_facts`) + Mem0/Qdrant vector index.
- **Media & Embeddings**: Google Gemini (`gemini-3.5-flash-lite` for video, voice/audio transcription, and optional `image_vision_provider: "gemini"` fallback; `gemini-embedding-2` 768d vectors for memory and stickers).
- **Vector Store**: Qdrant (`qdrant/qdrant:v1.20.0` in Docker Compose, or embedded on-disk mode via `QDRANT_PATH` for zero-Docker local runs).

## Key Features

- **Prefix-cache-first state (`~95–98%` KV-cache hit rate)**:
  - Static system prompt is separated from the daily date message so the core prefix stays cached across turns.
  - Exact provider/tool transcripts (`provider_messages`), native `image_url` blocks, and post-send incoming reactions (appended as tail events once `provider_sent: true`) never mutate already-cached prefixes until scheduled summarization (`05:00`, `10:00`, `14:00`, `20:00` `Europe/Moscow`).
- **Humanized conversational style**: Compact, uncensored prompt surface tuned against common LLM writing tells (no em-dash spam, book-quote formatting, empty "not X, but Y" contrasts, or aphoristic summary closers).
- **Switchable image vision (`image_vision_provider: "chat" | "gemini"`)**:
  - `"chat"` (default): photos, `media_group_id` photo albums, and static WebP stickers go directly into the chat model as `image_url` blocks via bounded in-memory LRU cache (`media_image_cache`) without bloating SQLite.
  - `"gemini"`: describes images via Gemini for text-only chat models.
- **Video & voice understanding**: videos, video notes (circles), video stickers, and voice/audio messages are processed via cloud Bot API (`<= 20 MiB`) and analyzed/transcribed by Gemini.
- **Native Bot-to-Bot visibility**: supports Telegram's native Bot-to-Bot Communication Mode (`@BotFather`), tagging peer bots as `[Bot: Name]`, excluding them from long-term memory, and silently ignoring unauthorized stranger bots in private chat to prevent loops.
- **Natural Telegram actions**:
  - `send_sticker`: single-step semantic vector search in Qdrant (`200+` indexed Russian emotion/occasion tags).
  - `react_to_message`: native Telegram emoji badges with duplicate-reaction guard.
  - `reply_to_message`: targeted reply to `[#N]` handles with optional exact UTF-16 `quote` highlighting.
  - `send_messages`: addressed bursts of 1–5 short bubbles, each with independent `reply_to`.
  - Streaming drafts (`send_message_draft`) in private chats; clean single-delivery turns in groups.
- **Content-free analytics**: `/stats`, `/top` chat leaderboards, `/ping` health check, and owner-only `/dashboard` in DM.

## Quick Start

### 1. Environment variables (`.env`)

```bash
cp .env.example .env
```

```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
DEEPSEEK_API_KEY=your_deepseek_api_key
GEMINI_API_KEY=your_gemini_api_key
# Optional (only when chat_provider: "openrouter" in config.yaml):
# OPENROUTER_API_KEY=your_openrouter_api_key
```

Review `allowed_users` (first entry is `OWNER_USER_ID`) and `allowed_groups` in [`config.yaml`](config.yaml).

### 2. Run with Docker Compose (Production / VPS)

```bash
docker compose up -d --build
```

Starts `qdrant`, `bot` (persisting SQLite + rotated `bot.log` + `loop_heartbeat` in `./bot_data`), `log-tail`, and Dozzle log viewer bound to `127.0.0.1:9999`.

### 3. Run Locally without Docker (Windows / Linux / macOS)

Using [`uv`](https://docs.astral.sh/uv/) and Python 3.11 with embedded on-disk Qdrant (`QDRANT_PATH`):

```bash
uv venv .venv --python 3.11
uv pip install --python .venv -r requirements.txt -r requirements-dev.txt

# Linux / macOS:
QDRANT_PATH=qdrant_local .venv/bin/python -m berangaria

# Windows (PowerShell):
$env:QDRANT_PATH = "qdrant_local"
.venv\Scripts\python.exe -m berangaria
# (or simply run scripts\start.bat)
```

*(If running Qdrant in Docker locally instead of embedded mode, omit `QDRANT_PATH` and pass `QDRANT_HOST=127.0.0.1`.)*

## Commands

| Command | Description | Access |
|---------|-------------|--------|
| `/start` | Overview and available capabilities | Allowed users/groups |
| `/clear` | Clear chat history (durable SQLite write-through) | All / Admin\* |
| `/stats` | Context window, token, and memory stats | Allowed users/groups |
| `/ping` | Process uptime, polling identity, and update counters | Allowed users/groups |
| `/top` | Current-chat activity leaderboards | Allowed users/groups |
| `/summarize` | Compress older chat history immediately | All / Admin\* |
| `/random <0-100>` | Set base random reply probability in groups | All / Admin\* |
| `/dashboard` | Token, cost, cache-hit, and alert analytics | Owner DM only |

\* Admin-only in groups when `admin_mode: true` in `config.yaml`.

## Validation & Tests

```bash
.venv/bin/python -m py_compile $(git ls-files '*.py')
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/bandit -q -ll -r . -x ./.venv,./venv,./tests
git diff --check
```

## Project Structure

```text
Berangaria_bot/
├── berangaria/                  # Runtime package (python -m berangaria)
│   ├── __main__.py              # Entry point
│   ├── app.py                   # Startup, handlers, scheduled jobs, shutdown
│   ├── config.py                # config.yaml + env loader, gateway & Qdrant config
│   ├── prompts.py               # SYSTEM_PROMPT, vision suffix, Mem0 instructions
│   ├── analytics/               # Content-free SQLite usage & leaderboard store
│   ├── core/                    # state.py, utils.py, paths.py, logging_setup.py, alerts.py
│   ├── chat/                    # Handlers, debounce queue, media, transport, delivery
│   ├── memory/                  # pipeline.py (strict fail-closed queue), store.py (Mem0)
│   ├── media/                   # vision.py (Gemini video/audio) + telegram_download.py
│   ├── stickers/                # store.py (Gemini embeddings + Qdrant sticker index)
│   └── tools/                   # schemas.py, web.py (DeepSeek Search/Fetch), dispatch.py
├── tests/                       # Pytest suite (isolated temp DB & logs)
├── scripts/                     # start.sh, start.bat, logs.sh, sticker CLI utilities
├── docs/                        # Configuration, logging, Dozzle viewer, ADRs
├── data/stickers_clean.json     # Curated sticker metadata synced into Qdrant
├── .github/workflows/deploy.yml # CI checks + optional SSH auto-deploy
├── config.yaml                  # Non-secret runtime defaults
├── .env.example                 # Secret & local override template
├── docker-compose.yml           # Universal Docker Compose stack
├── Dockerfile                   # Python 3.11-slim container image
└── pyproject.toml               # Ruff, pytest, and Bandit configuration
```

## Documentation

- **[docs/configuration.md](docs/configuration.md)** — Provider switching (`deepseek` / `openrouter`), vision & search modes, embedded vs Docker Qdrant, memory tuning, and CI/CD SSH deployment.
- **[docs/logging.md](docs/logging.md)** & **[docs/log-viewer.md](docs/log-viewer.md)** — Split console/file audit logging and Dozzle setup.
- **[CONTEXT.md](CONTEXT.md)** & **[docs/adr/](docs/adr/)** — Domain language and architecture decision records for the fail-closed memory pipeline.

## License

GPL-3.0
