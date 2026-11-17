# AGENTS.md

This file applies to the entire repository.

## Project overview

Berangaria is a Python 3.11 Telegram bot built on `python-telegram-bot` (cloud Bot API only).
By default it runs on the **DeepSeek API** (`deepseek-flash` / DeepSeek V4.1 Flash: chat with `thinking: disabled`, summarization with `thinking: enabled`, native `image_url` vision for photos/stickers, native Search & Fetch API for web grounding, and strict two-step memory extraction/verification), while retaining first-class configurable support for **OpenRouter** (`chat_provider: "openrouter"`). Gemini (`gemini-3.5-flash-lite` + `gemini-embedding-2`) handles video/audio understanding, optional fallback image descriptions (`image_vision_provider: "gemini"`), and embeddings; Mem0 + Qdrant store approved long-term memories and sticker vectors; SQLite persists conversation history, verification queues, runtime settings, and content-free analytics.

## Prompt and model baseline

- Shipped chat uses `chat_provider: "deepseek"`, `model: "deepseek-flash"`, `temperature: 1.0`, and `thinking: {"type": "disabled"}`; `summary_generation_params` enables `thinking: {"type": "enabled"}` at `temperature: 0.3`. When switching to `chat_provider: "openrouter"`, model-specific `generation_params`, `summary_generation_params`, and optional `openrouter_provider` routing rules live in `config.yaml`.
- `SYSTEM_PROMPT` is intentionally compact and uncensored for a strong model. Do not reintroduce exhaustive forbidden-catchphrase lists, weak-model personality cookbooks, anti-swearing/anti-insult censorship, or repeated all-caps constraints. Keep positive direction toward wit, jokes, banter, and equal-participant behavior.
- Prompt slimming must preserve actual protocol and security invariants: authenticated owner metadata, service-tag boundaries, untrusted web data, memory provenance, tool argument rules, and terminal tool paths. Protect those contracts with focused tests.
- Judge prompt or sampling changes with equivalent real-dialogue A/B trials. Passing unit tests proves wiring and invariants, not conversational quality.

All runtime code lives in the `berangaria` package; the entry point is `python -m berangaria`.
Tests under `tests/` mirror the package layout. One-off CLI tools live in `scripts/`.

Key modules:

- `berangaria/app.py`: application startup, handler registration, scheduled jobs, and graceful shutdown.
- `berangaria/config.py`: `config.yaml` + environment loader (`deepseek` / `openrouter` gateway, vision/search providers, and Mem0 config).
- `berangaria/prompts.py`: `SYSTEM_PROMPT`, `VISION_PROMPT_SUFFIX`, and `MEM0_CUSTOM_INSTRUCTIONS`. The tool `description` fields in `berangaria/tools/schemas.py` are prompt surface too — change them as deliberately.
- `berangaria/core/paths.py`: resolves relative config/db/log paths against the repository root.
- `berangaria/core/state.py`: shared in-memory state, LRU caches for media descriptions and native image bytes, per-chat locks, and SQLite persistence.
- `berangaria/core/utils.py`, `berangaria/core/logging_setup.py`: helpers and logging configuration.
- `berangaria/core/polling_diagnostics.py`: process-local polling identity, heartbeat stamp (`/data/loop_heartbeat`), and Conflict context (`host=` is who caught HTTP 409).
- `berangaria/core/alerts.py`: throttled owner DMs; put volatile uptime/since_update in `detail`, not `message`.
- `berangaria/chat/handlers.py`: Telegram command/event entry points (`/start`, `/clear`, `/stats`, `/ping`, `/top`, `/dashboard`, `/random`, `/summarize`), access control, and compatibility wrappers.
- `berangaria/chat/message_queue.py`: message normalization (including native Bot-to-Bot `is_bot` tagging), debounce buffering, history commit, and turn dispatch.
- `berangaria/chat/media_handlers.py`: photo albums plus photo, video, sticker, and voice processing (`image_vision_provider: "chat" | "gemini"`).
- `berangaria/chat/llm_client.py`: memory retrieval orchestration, LLM retries, tool rounds, and final turn branching.
- `berangaria/chat/completion_transport.py`: one streaming or non-streaming provider request and Telegram preview handoff.
- `berangaria/chat/assistant_turn.py`: confirmed assistant text/actions and Telegram message-ID persistence.
- `berangaria/chat/llm_diagnostics.py`: full audit payload/response logging plus token and cost reporting (supports both DeepSeek `prompt_cache_hit_tokens` and OpenAI/OpenRouter `cached_tokens`).
- `berangaria/chat/reply_delivery.py`: Telegram HTML/plain delivery, chunking, addressed multi-message bursts, and partial-failure handling.
- `berangaria/chat/memory_context.py`, `berangaria/chat/summarization.py`: approved-memory context filtering and history compression.
- `berangaria/chat/history_rendering.py`, `berangaria/chat/reply_formatting.py`: pure history-to-provider rendering (including prefix-preserving native `image_url` blocks) and Telegram reply formatting/cleanup helpers.
- `berangaria/chat/streaming.py`: OpenAI-compatible SSE reconstruction and throttled private Telegram drafts.
- `berangaria/memory/pipeline.py`: durable source queue, extraction, independent verification, exact Mem0 indexing, retries, and dead-letter handling.
- `berangaria/memory/store.py`: retryable Mem0 initialization.
- `berangaria/tools/schemas.py` / `web.py` / `dispatch.py`: tool definitions, DeepSeek Search & Fetch API + SSRF-hardened direct fallback, and dispatcher for reactions, replies, addressed bursts, and stickers.
- `berangaria/media/vision.py`, `berangaria/media/telegram_download.py`: Gemini video/audio/fallback-image requests and cloud Bot API media downloads (up to 20 MiB).
- `berangaria/stickers/store.py`: Gemini embeddings and Qdrant-backed sticker search/indexing.

## Setup and validation

Use Python 3.11, matching CI and Docker:

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt -r requirements-dev.txt
```

Checker versions are pinned in `requirements-dev.txt` and configured in `pyproject.toml`. Bump pins deliberately and resolve new findings in the same change.

Tests supply placeholder API keys in `tests/conftest.py`; never add real credentials to tests or the repository.

Run before handing work off:

```bash
.venv/bin/python -m py_compile $(git ls-files '*.py')
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/bandit -q -ll -r . -x ./.venv,./venv,./tests
git diff --check
```

Bandit may report intentional low-severity best-effort exception handling and non-cryptographic randomness. Any medium/high finding must be resolved.

## Concurrency and history invariants

- A complete chat turn is serialized with `state.get_turn_lock(history_key)`: append user input, call LLM/tools, deliver the reply, then persist the assistant entry.
- Whenever both locks are needed, acquire the turn lock before `state.get_history_lock`. Never reverse this order.
- Commands or jobs that clear, summarize, replace, or delete history must use the same turn lock. Reactions and edits to committed history also use it.
- Do not hold the narrow history lock across network I/O. The turn lock protects the logical transaction; the history lock protects short in-memory/SQLite mutations.
- Group debounce buffers are per user, but group history is shared. Do not remove per-history turn serialization.
- Summarization must work on copies of retained entries. Never renumber or otherwise mutate live history before the API request succeeds.
- Deliver Telegram text before adding the assistant entry to history. A delivery failure must not create a response the user never received or retry the whole LLM turn.
- Telegram-only cleanup and formatting must never rewrite the provider transcript. Store confirmed display text in `content`, retain the exact assistant/tool sequence in `provider_messages`, and render that sequence unmodified on later LLM turns. Terminal Telegram tools need a persisted tool result so replayed tool-call history stays valid.
- New persisted prompt rows carry `provider_sent: false`. Immediately before the first provider request, atomically persist them as sent; legacy rows without the flag count as sent. An incoming reaction may update an unsent assistant row in place, but after that row has been sent it must be appended as a structured tail event so the cached prompt prefix never changes.

## Async and external-I/O rules

- Never call blocking HTTP, file reads of large media, Qdrant, embedding, or search code directly from an async handler. Use an async client or `asyncio.to_thread`.
- Apply `_check_access_permissions` before downloading media or calling external APIs. Stranger bots (`is_bot=True`) in private chat must be ignored silently without a rejection reply to avoid bot-to-bot loops.
- `tools.read_url` handles untrusted model/user URLs. Always validate HTTP(S)-only scheme, reject credentials, and reject non-global IPs before calling DeepSeek Fetch or the direct HTTP fallback; keep `follow_redirects=False`, DNS-pinned IP connection, and redirect revalidation on the direct path.
- Keep transport/API retries separate from LLM tool-call round limits. Tool use must not consume network recovery attempts, and every loop must have an explicit terminal error.
- Tool arguments are untrusted JSON generated by a model. Parse defensively and return a tool error rather than replaying already executed actions.
- Streaming previews may expose only final-answer `delta.content`. Never send `reasoning_content`, tool names, tool arguments, or raw SSE events to Telegram.
- Persist structured `reasoning_details` (or `reasoning_content` / `reasoning`) only after confirmed final delivery and echo it unmodified on later chat turns. Keep opaque provider reasoning out of Telegram previews, memory extraction, and summaries.
- Private chats use `send_message_draft`. Group turns must not create persistent streaming previews: groups receive one final response only.
- `telegram.error.Conflict` on `getUpdates` is fatal: log CRITICAL (and flush), notify the owner with a stable fingerprint, then `os._exit(1)` in `finally` so Docker `restart: always` recovers. Keep `get_updates_read_timeout` aligned with outgoing Bot API `read_timeout`.

## Persistence and memory rules

- SQLite writes are write-through for history changes. A reported reset or clear must be durable before confirming it to the user.
- Use `berangaria.memory.store.memory` dynamically (`from berangaria.memory import store as memory_store`); do not import the object by value.
- Messages from other bots (`is_bot=True`), forwards, and media descriptions must never enter `memory_sources`.
- A source may be completed only after every approved fact is confirmed by Mem0 and atomically written to the SQLite approval registry. Immediately compensate a partial Mem0 source-transaction; before any retried source can extract or store again, reconcile all Mem0 records tagged with that source ID against the authoritative SQLite registry. Technical failures return the source to the durable SQLite queue in FIFO order; after five failed attempts, terminal sources are dead-lettered without approved or retrievable memory.
- A turn that never reaches confirmed delivery must abandon its sources, not release them. The FIFO gate is scope-local.
- Do not commit `.env`, API keys, bot tokens, `bot_data/`, Qdrant storage, databases, logs, or generated virtual environments.

## Configuration and deployment

- `config.yaml` is the runtime source of non-secret defaults. Environment variables override secrets and selected deployment paths. Chat `model` and `price_*` have no code fallback: missing keys fail startup. Changing the default model updates `price_*`, `test_shipped_chat_model_is_deepseek_flash`, and the operator docs.
- The first entry in `allowed_users` is the authenticated owner (`OWNER_USER_ID`). Only server-generated `[Owner: Name]` metadata may identify the owner, and the owner bypasses user/group allowlists.
- `admin_alert_chat_id` is one integer Telegram chat ID or `null` (which routes alerts to `OWNER_USER_ID`).
- Docker uses `BOT_DB_PATH=/data/bot_state.db` and `BOT_LOG_FILE=/data/bot.log`; preserve the mounted `/data` volume (`./bot_data:/data`) and `./qdrant_storage:/qdrant/storage`.

## Change discipline

- Preserve unrelated user changes in a dirty worktree.
- Add regression tests for bug fixes, especially races, failure ordering, security boundaries, and persistence behavior.
- Prefer focused patches over broad rewrites unless protected by tests.
- Update `README.md` or `docs/configuration.md` when configuration, startup, commands, deployment, or externally visible behavior changes.

## Agent skills

- **Issue tracker**: GitHub Issues (`docs/agents/issue-tracker.md`).
- **Triage labels**: Five default triage role labels (`docs/agents/triage-labels.md`).
- **Domain docs**: `CONTEXT.md` at the root and ADRs under `docs/adr/` (`docs/agents/domain.md`).
