# Configuration Reference

Complete reference for Berangaria Bot configuration options.

## Configuration Files

### config.yaml

Main configuration file with all bot settings.

### .env

Environment variables for sensitive data:

```env
TELEGRAM_BOT_TOKEN=<token>
API_KEY=<deepseek_key>
GEMINI_API_KEY=<gemini_key>
# Optional Fish Audio TTS (send_voice tool). Without both, voice stays off.
FISH_API_KEY=<fish_key>
FISH_VOICE_ID=<voice_model_id>
# Telegram API application credentials for the local Bot API server:
TELEGRAM_API_ID=<api_id>
TELEGRAM_API_HASH=<api_hash>
# Optional local Telegram Bot API server:
TELEGRAM_BOT_API_BASE_URL=http://127.0.0.1:8081
TELEGRAM_BOT_API_LOCAL_MODE=true
BOT_VIDEO_MAX_FILE_SIZE_BYTES=2147483648
# Optional user bridge (read-only MTProto — see other bots in groups):
# USER_BRIDGE_SESSION=<string from scripts/user_bridge_login.py>
# USER_BRIDGE_ENABLED=true   # or set user_bridge_enabled in config.yaml
```

The production Compose file starts `aiogram/telegram-bot-api` with
`TELEGRAM_LOCAL=1` and persists its files under `/var/lib/telegram-bot-api`.
When local mode is enabled, `TELEGRAM_BOT_API_BASE_FILE_URL` defaults to
`<TELEGRAM_BOT_API_BASE_URL>/file`. The bot container must be able to read the
same absolute file paths returned by the API server; the provided Compose file
mounts `/var/lib/telegram-bot-api` read-only and uses host networking.

### berangaria/config.py

Python configuration loader. Reads `config.yaml`, applies environment overrides, and builds the Mem0
config. Relative paths (`config.yaml`, `log_file`, `sticker_sync_file`, `BOT_DB_PATH`) are resolved
against the repository root via `berangaria/core/paths.py`, so the bot behaves the same whether it is
started with `python -m berangaria`, from Docker, or from a script in `scripts/`.
Prompt texts live separately in `berangaria/prompts.py`.

## config.yaml Parameters

### Main LLM (DeepSeek)

```yaml
model: "deepseek-v4-flash"
max_context_tokens: 32000
max_reply_tokens: 4096
generation_params:
  temperature: 0.9
  top_p: 0.95
```

**Parameters:**
- `model`: DeepSeek model identifier
- `max_context_tokens`: Maximum conversation history size
- `max_reply_tokens`: Maximum response length
- `generation_params`: Model sampling parameters (temperature, top_p)

Additional supported parameters: `top_k`, `min_p`, `presence_penalty`, `repetition_penalty`

### Vision (Gemini)

```yaml
vision_mode: true
gemini_model: "gemini-3.5-flash-lite"
video_max_duration_sec: 300
```

**Parameters:**
- `vision_mode`: Enable/disable image and video understanding
- `gemini_model`: Gemini model for vision tasks
- `video_max_duration_sec`: Maximum video length in seconds (shipped: 300)
- `video_max_file_size_bytes`: Weight ceiling for a downloaded video. Defaults to 20 MB on
  the cloud Bot API and to 2 GiB when `TELEGRAM_BOT_API_LOCAL_MODE=true`, which is what the
  production Compose file runs. Override with `BOT_VIDEO_MAX_FILE_SIZE_BYTES`.
- `audio_max_duration_sec`: Maximum voice/audio length in seconds (shipped: 300)
- `gemini_upload_max_wait_sec` / `gemini_upload_backoff_initial` / `gemini_upload_backoff_max`:
  Files API upload polling budget and backoff

**Available Gemini models:**
- `gemini-3.5-flash-lite` (free tier, current default; GA since 21 July 2026)
- `gemini-3.1-flash-lite` (deprecated, earliest shutdown 7 May 2027 — `gemini-3.5-flash-lite` is Google's named replacement)
- `gemini-2.5-flash` (shutdown 16 October 2026)
- `gemini-2.0-flash` (already shut down — do not use)

Do not pass `temperature`, `topP`, or `topK` to Gemini 3.x models: they are tuned for the default
temperature of 1.0, and lower values are documented to cause looping and degraded output. The vision
requests in `berangaria/media/vision.py` deliberately send only `maxOutputTokens`.

### TTS / Voice notes (Fish Audio)

```yaml
tts_enabled: true
tts_model: "s2.1-pro-free"
tts_format: "opus"
tts_latency: "normal"
tts_max_chars: 400
tts_timeout_seconds: 45
tts_max_per_turn: 1
tts_default_emotion: "calm"
```

**Parameters:**
- `tts_enabled`: Master switch. Effective only when `FISH_API_KEY` and `FISH_VOICE_ID` are set in `.env`.
- `tts_model`: Fish model header (`s2.1-pro-free` is the free tier; paid fallback `s2.1-pro`).
- `tts_format`: Prefer `opus` for Telegram voice notes.
- `tts_latency`: `normal` | `balanced` | `low`.
- `tts_max_chars`: Hard cap on spoken text passed to Fish.
- `tts_timeout_seconds`: HTTP timeout for one synthesis request.
- `tts_max_per_turn`: Max successful `send_voice` attempts per LLM turn (usually 1).
- `tts_default_emotion`: Applied when the model omits `emotion` (`calm`, or empty for none).

The model calls `send_voice(text, emotion?)`. On success the turn ends (mutex with sticker / multi / reply tools). Emotion is a whitelist only: calm, sarcastic, disdainful, bored, indifferent, confident, sighing, chuckling, none.

Smoke test (same client as the bot):

```bash
python scripts/fish_tts_smoke.py --suite
```

### Memory (Mem0 + Embeddings)

```yaml
mem0_llm_model: "deepseek-v4-flash"
embedding_model: "gemini-embedding-2"
embedding_dims: 768
memory_search_limit: 10
memory_min_score: 0.3
memory_max_chars: 1200
memory_flush_interval_seconds: 300
memory_query_min_chars: 12
memory_query_recent_messages: 3
memory_queue_batch_size: 20
memory_waiting_max_age_seconds: 1800
memory_source_retention_seconds: 2592000
```

**Parameters:**
- `mem0_llm_model`: DeepSeek model used by the strict extractor and independent verifier
- `embedding_model`: Gemini embedding model (name as shipped, without a "models/" prefix)
- `embedding_dims`: Vector dimensions (768 for gemini-embedding-2). Changing the model or the dimension invalidates every existing vector: both the `mem0` and `stickers` collections must be re-embedded, otherwise recall silently degrades because old and new vectors share a space they were not trained in.
- `memory_search_limit`: Number of facts retrieved per query
- `memory_min_score`: Vector-score floor (0.0-1.0); SQLite approval and current-topic matching remain mandatory
- `memory_max_chars`: Maximum total length of memory context
- `memory_flush_interval_seconds`: Periodic retry interval for the durable SQLite queue
- `memory_query_min_chars`: Minimum meaningful query length before memory search runs
- `memory_query_recent_messages`: Recent meaningful user messages combined for memory search
- `memory_queue_batch_size`: Maximum source messages processed per worker pass
- `memory_waiting_max_age_seconds`: Age at which a source still in `waiting` is treated as
  the remains of an interrupted turn and abandoned (raw text erased). A turn cannot outlive
  debounce plus the full LLM retry budget, so keep this comfortably above that; the floor is
  60 s. This is a safety net behind the compensation in `handlers.wait_and_process` — a
  `waiting` source blocks the FIFO queue of its own memory scope until it is resolved.
- `memory_source_retention_seconds`: How long finished (`completed`/`abandoned`) queue rows are
  kept before pruning. `dead` rows are never pruned — they are the only forensic trace of why a
  source did not become memory. Keep this comfortably above the window in which
  `INSERT OR IGNORE` still protects against re-queueing the same message.

**Relevance threshold guide:**
- `0.1`: Very permissive (includes many facts)
- `0.2`: Soft filtering
- `0.27`: Permissive legacy value
- `0.3`: Balanced — **current default**
- `0.5`: Stricter than shipped — drops approved facts scoring 0.3–0.5

### Bot Behavior

```yaml
bot_names: ["Бер", "Ber"]
random_reply_chance: 10
summary_interval: 10
timezone: "Europe/Moscow"
summary_hours: [5]
message_debounce_seconds: 4.0
max_buffered_messages: 30
max_buffered_chars: 20000
random_reply_cooldown: 10
admin_mode: false
streaming_enabled: true
stream_update_interval_seconds: 0.8
stream_preview_min_chars: 12
debug: true
full_debug_logs: true
verbose: false
log_file: bot.log
log_max_bytes: 10485760
log_backup_count: 5
```

**Parameters:**
- `bot_names`: Names that trigger bot responses in groups
- `random_reply_chance`: Default probability (0-100) of spontaneous group replies. Runtime changes via `/random` are saved in SQLite and survive restarts.
- `summary_interval`: Messages preserved after summarization
- `timezone`: Bot timezone for `[Time:]` tags, CURRENT TIME in the system prompt, and scheduled summarization (default `Europe/Moscow`)
- `summary_hours`: Local hours when automatic history compression runs (shipped: `[5]` → 05:00; default if omitted: `[5, 14]`)
- `message_debounce_seconds`: Timeout for merging consecutive messages (seconds)
- `max_buffered_messages` / `max_buffered_chars`: Budget for one debounce buffer. Every message
  restarts the debounce window, so without a budget a continuous stream merges into a single
  unbounded history entry. On reaching either limit the buffer is flushed instead of extended.
- `random_reply_cooldown`: Minimum interval between random replies (seconds)
- `admin_mode`: Restrict management commands to group admins
- `streaming_enabled`: Enable DeepSeek SSE and private-chat Telegram draft previews; group chats receive one final message
- `stream_update_interval_seconds`: Minimum delay between private Telegram draft updates; clamped to 0.25-5 seconds
- `stream_preview_min_chars`: Minimum buffered answer length before the first private draft update
- `debug`: Enable detailed logging
- `full_debug_logs`: Write detailed prompts, model replies, memory facts, and vision descriptions to DEBUG logs without enabling DEBUG output in Docker console.
- `verbose`: Super-detailed logs (HTTP, TLS, H2 - includes debug)
- `log_file`: Local log file path. Docker overrides this to `/data/bot.log` (`./bot_data/bot.log` on the host).
- `log_max_bytes`: One log file size before rotation. Set `0` to disable rotation.
- `log_backup_count`: Number of rotated log files to keep.

### Tools: Search and Stickers

```yaml
factual_temperature: 0.4
web_search_max_per_turn: 2
multi_message_max: 3
multi_message_max_chars: 280
multi_message_max_total_chars: 600
multi_message_delay_min: 0.4
multi_message_delay_max: 2.0
multi_message_delay_total_cap: 5.0
sticker_enabled: true
sticker_min_score: 0.25
sticker_top_k: 8
sticker_send_max_per_turn: 2
sticker_sync_file: data/stickers_clean.json
sticker_index_version: 3
```

**Parameters:**
- `factual_temperature`: Sampling temperature used for the rest of a turn once `web_search` or
  `read_url` ran — lower means fewer invented numbers. It is applied *only* for those two tools:
  reactions and sticker searches no longer cool the turn down, because that silently flattened the
  persona every time the bot merely looked for a sticker.
- `web_search_max_per_turn`: Ceiling on `web_search` calls in one reply (one query plus one refined
  retry). The prompt asks the bot to verify facts aggressively, and the DuckDuckGo rate limiter
  (10/min in `berangaria/tools/web.py`) is process-global, so one runaway turn would otherwise
  break search for every chat. On overflow the tool returns a refusal telling the model to answer
  with what it already has.
- `multi_message_*`: Caps and typing pauses for the terminal `send_messages` tool (2–3 short
  Telegram bubbles with `typing` between them). Delays scale with bubble length and are capped by
  `multi_message_delay_total_cap`. Mutex with `reply_to_message` and `send_sticker`.
- `sticker_min_score`: Vector-score floor for sticker search. Below it a sticker is not offered.
  Lowering it widens the menu but risks off-vibe stickers; raise it back if that happens.
- `sticker_top_k`: How many vector hits `send_sticker` considers before picking one at random
  (all above `sticker_min_score`).
- `sticker_send_max_per_turn`: Max one-shot `send_sticker(query)` attempts per reply (search+send).
  Default 2 — one refined query if the first misses. Success ends the turn; no separate find step.
  Legacy key `sticker_find_max_per_turn` is still read as a fallback.
- `sticker_sync_file`: Catalogue path (`.json` array or `.jsonl`). Relative paths resolve from the
  repository root.
- `sticker_index_version`: Bump when the embed text formula or catalogue schema changes. On startup
  the bot recreates the Qdrant sticker collection and re-embeds the whole catalogue once, then
  writes the marker under the bot data dir.

`MAX_TOOL_ROUNDS` in `berangaria/config.py` (12) bounds consecutive tool-call rounds. A turn that
verifies a fact and sends a sticker uses far fewer rounds than the old find→pick→send flow; the
ceiling still has headroom so overflow does not abort ordinary turns.

### Access Control

```yaml
allowed_users: [1217938322, 1809564460]
allowed_groups: [-1002263830880]
admin_alert_chat_id: 1217938322
```

**Parameters:**
- `allowed_users`: Telegram user IDs with bot access
- `allowed_groups`: Telegram group IDs where bot operates
- `admin_alert_chat_id`: One Telegram chat ID for throttled critical-error alerts, or `null` to disable alerts. This value is a scalar, not a list.

Get IDs by sending `/start` to the bot with debug mode enabled.

Access is checked before photos, stickers, videos, or audio are downloaded or sent to Gemini, so denied users cannot consume external API quota.

### Cost Tracking

```yaml
price_prompt_cache_miss: 0.435
price_prompt_cache_hit: 0.003625
price_completion: 0.87
```

**Parameters (per 1M tokens):**
- `price_prompt_cache_miss`: Regular input tokens
- `price_prompt_cache_hit`: Cached input tokens
- `price_completion`: Output tokens

Current prices for DeepSeek v4 Flash. Update when prices change.

## Memory Configuration

### Strict fail-closed pipeline

1. Every non-empty original text message, including short and low-signal text, is stored as an atomic SQLite queue item with chat scope, author, Telegram message ID, timestamp, and source text. A later Telegram edit with the same message ID does not overwrite it.
2. DeepSeek extracts structured candidates containing a normalized fact, stable `fact_key`, and exact source quote.
3. A separate verifier independently checks self-attribution, stability, source entailment, modality, and sensitive-data exclusions.
4. Deterministic validation confirms the quote exists verbatim in the source and rejects malformed or unsafe candidates.
5. All candidates receive a final decision before storage. Mem0 receives each approved fact with `infer=False`; one SQLite transaction publishes all facts from the source, partial failure compensates new/replaced vectors, and a retry reconciles crash leftovers by `source_id` before new writes.
6. A newer fact with the same scope, subject, and `fact_key` updates the existing vector in place.
7. Topical retrieval cross-checks the Mem0 ID and exact fact text against the SQLite approval registry for the same chat scope, applies the score threshold, and requires a topical match with the latest meaningful user message before adding a fact to the prompt. Explicit general recall questions read the approved SQLite registry for that scope directly, bypassing vector score and topical matching but never approval, scope, or output limits.

The worker starts after the buffered Telegram turn completes, so verification is not on the reply's critical path. The pipeline is fail-closed. Any DeepSeek, Mem0, Qdrant, parsing, or validation failure creates no memory. Technical failures retry from SQLite in FIFO order exactly five times, then become dead-letter records. Full raw source text is retained only while a retry is possible; completed and dead-letter records are redacted, while approved facts keep only their exact evidence quote and Telegram provenance.

### What Gets Remembered

**Included:**
- Clear, stable self-statements by the message author
- Durable preferences, biographical facts, habits, and long-lived projects
- Facts backed by an exact quote from the same source message

**Excluded:**
- Statements about other people, group-level “we” claims, questions, plans, guesses, quotes, and irony
- Temporary states and low-signal messages
- Credentials, precise addresses/documents, financial or medical data
- Forwarded messages, vision/audio descriptions, and media-only messages

## Debug Mode

Enable with `debug: true` in config.yaml.

### Startup Output

```
Mem0 configuration:
{
  "llm_provider": "deepseek",
  "llm_model": "deepseek-v4-flash",
  "embedder_provider": "gemini",
  "embedder_model": "gemini-embedding-2",
  "vector_store": "qdrant"
}
Mem0 initialized
```

### Memory Operations

```
Память: факт одобрен (source_id=17, scope=private_123456, key=software.os)
Память: обработано источников 3, одобрено 1, отброшено 2, retry 0, dead-letter 0
Память: найдено 5 → загружено 2 фактов (86 символов)
```

### Request Tracking

```
Tokens: request=1234 (cache=980), reply=567, total=1801
Request cost: $0.000285
```

## Troubleshooting

### Automatic Summarization

**How it works:**
- Runs at every hour listed in `summary_hours` (shipped: 05:00 local time)
- Summarizes chats longer than `summary_interval + 1` messages (result must be strictly shorter)
- Keeps the last `summary_interval` messages intact
- Compresses older history into a brief summary
- Uses DeepSeek with thinking enabled (`reasoning_effort: high`); long client timeout and a larger `max_tokens` budget so CoT does not starve the final summary

**Manual trigger:**
Use `/summarize` command to compress chat history immediately.

**Configuration:**
- Schedule is set by `summary_hours` in config.yaml (shipped: `[5]`)
- Minimum messages required: more than `summary_interval + 1` (result is always 1 summary + `summary_interval` recent)

**Logs:**
```
⏰ Следующая суммаризация в 05:00 11.06.2026 (через 8.5ч)
📝 Запуск суммаризации для 3 активных чатов...
  ✅ group_-1002263830880: 25 → 11 сообщений
📝 Суммаризировано 1 из 3 чатов
```

### Vision Not Working

**Symptoms:** Bot doesn't respond to images/videos

**Solutions:**
1. Check `GEMINI_API_KEY` exists in .env
2. Verify `vision_mode: true` in config.yaml
3. Confirm key is from aistudio.google.com
4. Check logs for Gemini API errors

### Memory Errors

**Symptoms:** Mem0 initialization fails

**Solutions:**
1. Verify Qdrant running: `docker ps`
2. Check `GEMINI_API_KEY` in .env
3. Install dependencies: `pip install google-generativeai langchain-google-genai langchain-core`
4. Verify embedding model name: `gemini-embedding-2`

**Common error:** "Unsupported embedding provider: googleai"
- Fix: In `berangaria/config.py`, provider should be "gemini" (not "googleai" or "google")

### High API Costs

**Symptoms:** Unexpected DeepSeek charges

**Solutions:**
1. Check cache hit rate in logs (target: 70-90%)
2. Verify using `deepseek-v4-flash` (not reasoning models)
3. Reduce `max_context_tokens` if conversations too long
4. Use `/summarize` to compress long chats

### Poor Memory Recall

**Symptoms:** Bot doesn't remember previous conversations

**Solutions:**
1. Enable `full_debug_logs` and check memory saves/retrieval
2. Check periodic queue reports for retries or dead-letter sources
3. Confirm the fact passed extractor, verifier, and deterministic validation
4. Keep `memory_min_score` at `0.3` unless an audited retrieval benchmark supports a change
5. Increase `memory_search_limit` only if the context budget allows it
6. Verify Qdrant has data: check qdrant_storage/ directory size
7. Ensure `fastembed` is installed so Qdrant BM25 keyword search is enabled

## Migration Guide

### From LM Studio Version

**Configuration changes:**

Removed parameters:
- `mem0_model` (use main model)
- `provider` (hardcoded in `berangaria/config.py`)
- `embedding_provider` (hardcoded)
- `base_url` (not needed for cloud APIs)
- `vision_model` (local models removed)
- `video_max_frames` (Gemini handles full video)

Added parameters:
- `embedding_model` (Gemini model name)
- `embedding_dims` (vector dimensions)

**Code changes:**

All LM Studio references removed from:
- `berangaria/config.py`
- `berangaria/chat/llm_client.py`
- `berangaria/chat/handlers.py`

**Data migration:**

Qdrant data (qdrant_storage/) requires recreation if embedding dimensions changed. Backup old data if needed:

```bash
docker-compose down
mv qdrant_storage qdrant_storage.backup
docker-compose up -d
```

Bot will rebuild memory from new conversations.

## Performance Optimization

### Token Economy

- Use `summary_interval` to compress long chats
- Monitor cache hit rate (70%+ is good)
- Adjust `max_context_tokens` based on typical conversation length

### Memory Quality

- Keep `memory_min_score: 0.3` unless an audited retrieval benchmark justifies changing it
- Enable debug mode to audit what's saved
- Monitor pending/dead-letter queue counts and verifier rejection reasons
- Keep media descriptions and raw conversation batches out of Mem0

### System Performance

- Qdrant runs locally (fast, no network latency)
- Gemini embeddings are free tier
- DeepSeek v4 flash is optimized for speed

## Advanced Configuration

### Custom System Prompt

Edit `SYSTEM_PROMPT` in `berangaria/prompts.py`. Current prompt defines personality, communication style, and behavior rules. The tool descriptions in `berangaria/tools/schemas.py` are part of the prompt too — the model decides whether to search or send a sticker by reading them.

### Mem0 Configuration

Full Mem0 config in `MEM0_CONFIG` dictionary (`berangaria/config.py`):

```python
MEM0_CONFIG = {
    "version": "v1.1",
    "custom_instructions": MEM0_CUSTOM_INSTRUCTIONS,
    "llm": {
        "provider": "deepseek",
        "config": {
            "model": MEM0_LLM_MODEL,
            "api_key": DEEPSEEK_API_KEY,
            "temperature": 0.1,
            "max_tokens": 2000,
        }
    },
    "embedder": {
        "provider": "gemini",
        "config": {
            "model": EMBEDDING_MODEL,
            "api_key": GEMINI_API_KEY
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "127.0.0.1",
            "port": 6333,
            "collection_name": "mem0",
            "embedding_model_dims": EMBEDDING_DIMS
        }
    }
}
```

## Environment Variables Reference

| Variable | Required | Purpose | Source |
|----------|----------|---------|--------|
| `TELEGRAM_BOT_TOKEN` | Yes | Bot authentication | @BotFather |
| `API_KEY` | Yes | DeepSeek API access | platform.deepseek.com |
| `GEMINI_API_KEY` | Yes | Gemini vision + embeddings | aistudio.google.com |
| `FISH_API_KEY` | No | Fish Audio TTS | fish.audio/app/api-keys |
| `FISH_VOICE_ID` | No | Fish voice model id for `send_voice` | fish.audio library |

### Production: secrets without SSH

The VPS `.env` is untracked and is not wiped by deploy `git reset --hard`.
`deploy.yml` can upsert the following from
**GitHub → Settings → Secrets and variables → Actions** into that file before
`docker compose up`. Empty secrets are skipped (existing server values kept),
and empty shell exports are unset so Compose still reads the server `.env`
(shell environment wins over the project file for `${VAR}` interpolation):

- `FISH_API_KEY` / `FISH_VOICE_ID` (TTS)
- `TELEGRAM_API_ID` / `TELEGRAM_API_HASH` (local Bot API + user bridge)
- `USER_BRIDGE_SESSION` (Telethon StringSession for the optional user bridge)

`TELEGRAM_API_ID` / `TELEGRAM_API_HASH` must exist either as GitHub secrets or
already in the VPS `.env` — `docker-compose.yml` requires them for
`telegram-bot-api`. Other bot keys may still live only in the server `.env`
until the same pattern is extended. Do not commit `.env` or put keys in
`config.yaml`.

### User bridge (optional)

Bot API does not deliver messages from other bots in groups. When enabled, a
**read-only** Telethon client on your user account listens in allowlisted groups
and injects those messages into the normal debounce → LLM → Bot API reply path.

| Setting | Default | Meaning |
|---------|---------|---------|
| `user_bridge_enabled` | `false` | Master switch (`USER_BRIDGE_ENABLED` env override) |
| `user_bridge_chat_ids` | `[]` | Empty = use `allowed_groups`; else explicit chat ids |
| `user_bridge_reconnect_seconds` | `5` | Pause after bridge disconnect |
| `user_bridge_media_timeout_seconds` | `60` | Cap for download + vision per bot media |
| `user_bridge_dedup_ttl_seconds` | `300` | Drop duplicate `(chat_id, message_id)` |

Rules enforced in code:

- groups only; only `sender.bot`; skip Berangaria’s own bot id
- **no** long-term memory enqueue for bridge traffic
- history uses `[Bot: name]` (not `[User: …]`)
- replies / tools stay on Bot API
- bridge errors reconnect; they never stop Bot API polling

One-time session: `python scripts/user_bridge_login.py` → put `USER_BRIDGE_SESSION=…`
in `.env` and the matching GitHub Actions secret.
## File Structure

```
Berangaria_bot/
├── config.yaml                      # Main configuration
├── .env                             # Secrets (gitignored)
├── berangaria/
│   ├── __main__.py                  # Entry point (python -m berangaria)
│   ├── app.py                       # Startup, handlers, jobs, shutdown
│   ├── config.py                    # Config loader and Mem0 setup
│   ├── prompts.py                   # System prompt, vision suffix, Mem0 instructions
│   ├── core/paths.py                # Project-root-relative path resolution
│   ├── core/state.py                # In-memory and SQLite state
│   ├── core/utils.py                # Helper functions
│   ├── core/logging_setup.py        # Logging configuration
│   ├── chat/handlers.py             # Telegram event handlers
│   ├── chat/llm_client.py           # DeepSeek client
│   ├── chat/streaming.py            # SSE reconstruction and drafts
│   ├── memory/pipeline.py           # Strict memory verification pipeline
│   ├── memory/store.py              # Mem0 initialization
│   ├── media/vision.py              # Gemini image/video/audio
│   ├── media/tts.py                 # Fish Audio TTS (send_voice)
│   ├── stickers/store.py            # Sticker embeddings and Qdrant search
│   └── tools/                       # schemas.py, web.py, dispatch.py
├── data/stickers_clean.json         # Sticker catalogue (JSON array; .jsonl also supported)
├── scripts/                         # start.sh, start.bat, logs.sh, sticker CLI, fish_tts_smoke.py
├── docker-compose.yml               # Bot, Qdrant, local Bot API, log viewer
├── requirements.txt                 # Python dependencies
└── qdrant_storage/                  # Vector DB data (auto-created)
```

## References

- [DeepSeek API Docs](https://platform.deepseek.com/docs)
- [Google AI Studio](https://aistudio.google.com)
- [Mem0 Documentation](https://docs.mem0.ai)
- [Qdrant Documentation](https://qdrant.tech/documentation)
