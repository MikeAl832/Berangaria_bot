# Berangaria Bot

Telegram bot with long-term memory, vision understanding, and web search capabilities. Built with python-telegram-bot, featuring a sharp-witted personality and multi-modal understanding.

## Architecture

- **Main LLM**: DeepSeek v4 Flash (chat, summarization, strict memory extraction/verification)
  - Alternative: DeepSeek v4 Pro for better instruction following
  - Alternative: Grok-4.3 for superior humor and character consistency
- **Vision**: Google Gemini 3.5 Flash Lite (image/video/audio understanding)
- **Embeddings**: Google Gemini Embedding v2 (memory vectors)
- **Vector Store**: Qdrant (local Docker container)
- **Memory**: SQLite durable verification queue + Mem0/Qdrant approved-fact index

## Key Features

- **Long-term memory** scoped per chat (shared in groups, private in DMs)
- **Multi-modal understanding**: images, videos, stickers, voice messages, and audio
- **Fact-checking by default**: the system prompt treats built-in knowledge as undated gossip. Any reply resting on a number, date, name or a checkable claim someone else made goes through `web_search` first — including before the bot corrects or mocks anyone. Opinions, jokes and anything about the chat itself are explicitly excluded, the search is capped per turn, and the mechanics stay invisible in the reply.
- **Stickers as first-class replies**: when the answer is mostly emotion, the bot sends a sticker instead of typing it. Reactions acknowledge, stickers reply — the prompt now states that tiebreaker instead of leaving the two tools competing for the same situations.
- **Telegram reactions** via function calling for natural emoji responses
- **Automatic conversation summarization** with token budget management
- **Smart message buffering** for rapid consecutive messages (4-second debounce)
- **Streaming replies** with native Telegram drafts in private chats and one final delivery in groups
- **Configurable random replies** in groups with cooldown
- **Persistent runtime settings**: `/random` changes are saved in SQLite and survive restarts
- **Token usage tracking** and cost estimation with cache hit rates
- **Persistent rotated logs** in Docker (`bot_data/bot.log`) with helper commands
- **Media description caching** to avoid re-processing repeated content
- **Emoji-sparse text responses** — the system prompt discourages emoji in prose; reactions go through the reaction tool. Emoji are deliberately NOT stripped from the text: when an emoji IS the whole answer, it must survive.

## Prerequisites

- Python 3.10+
- Docker (for Qdrant)
- DeepSeek API key
- Google Gemini API key (for vision and embeddings)
- Telegram bot token

## Installation

### 1. Clone and setup Qdrant

```bash
git clone https://github.com/MikeAl832/Berangaria_bot.git
cd Berangaria_bot
docker-compose up -d
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables

Create `.env` file:

```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
API_KEY=your_deepseek_api_key
GEMINI_API_KEY=your_gemini_api_key
# Telegram API application credentials for the local Bot API server:
TELEGRAM_API_ID=your_telegram_api_id
TELEGRAM_API_HASH=your_telegram_api_hash
```

The production Compose configuration starts the local Telegram Bot API server
at `127.0.0.1:8081` in local mode and mounts its persistent file directory
read-only into the bot. `TELEGRAM_API_ID` and `TELEGRAM_API_HASH` are required
for this service. This allows media larger than the cloud Bot API `getFile`
limit to be processed.

API keys:
- Telegram: [@BotFather](https://t.me/botfather)
- DeepSeek: [platform.deepseek.com](https://platform.deepseek.com)
- Gemini: [aistudio.google.com](https://aistudio.google.com)

### 4. Configure bot settings

Edit `config.yaml` - see [docs/configuration.md](docs/configuration.md) for detailed options.

Key settings:
- `model`: DeepSeek model name
- `vision_mode`: enable/disable vision
- `embedding_model`: Gemini embedding model
- `mem0_llm_model`: DeepSeek model used by the strict memory extractor and verifier
- `memory_search_limit`: facts injected into context
- `allowed_users` / `allowed_groups`: access control

### 5. Run

```bash
python -m berangaria
```

Windows users can use `scripts\start.bat`, Linux users `scripts/start.sh`.
Both resolve the project root themselves, so they work from any directory.

## How It Works

### Memory System

Memory is partitioned by chat:
- **Groups**: shared memory for entire chat (`group_<chat_id>`)
- **Private**: per-user memory (`private_<user_id>`)

Each original text message, including short ones, is queued durably in SQLite with its author, chat scope, Telegram message ID, timestamp, and source text. A later Telegram edit cannot replace that source. Verification starts only after the buffered Telegram turn succeeds, so waiting sources cannot be claimed before reply delivery. DeepSeek first extracts candidates with literal source quotes, then a separate verifier must approve all decisions before storage begins; malformed output, ambiguity, sensitive data, forwarded text, media-only context, or any service failure is fail-closed. Mem0 writes from one source are published atomically in SQLite and compensated on partial failure; retries first reconcile any records left by a crash. Failed sources retry in FIFO order exactly five times. Mem0 receives only approved facts with `infer=False`, and later statements replace the same fact in place. Completed raw source text is erased; topical retrieval accepts only literal ID/text matches from the chat-scoped SQLite approval registry, the strict relevance threshold, and a topical match with the latest meaningful user message. Explicit general recall questions read the same approved, scope-limited SQLite registry directly because an unscoped vector score has no meaningful topic. Original TikTok URLs remain in provenance but are removed from the LLM-facing copy.

### Vision Processing

When media is received:
1. **Images/Stickers**: Gemini analyzes and provides natural conversational description
2. **Videos**: Full video processing (no frame extraction) with timeline understanding
3. **Voice/Audio**: Speech transcription with diarization support
4. Description injected as `[Image description: ...]`, `[Video description: ...]`, or transcript
5. Main LLM responds as if it observed the media directly

**Supported formats:**
- Images: JPEG, PNG, WebP (static stickers)
- Video: MP4, WebM (including video stickers and circles)
- Audio: OGG, MP3, WAV (voice messages and audio files)

**Smart features:**
- Inline processing for small files (<18MB)
- Resumable upload for larger files via Gemini Files API
- Automatic cleanup of temporary files
- Media description caching (keyed by `file_unique_id`)
- Duration limits: 300s for video, 300s for audio (`video_max_duration_sec` / `audio_max_duration_sec`)

**Natural language prompts:**
Vision prompts redesigned for conversational output instead of structured reports. Gemini describes media as if explaining to a friend, making integration with chat LLM seamless.

### Conversation Management

- **Token budgeting** per chat with automatic summarization at 85% capacity (configurable via `max_context_tokens`)
- **Manual summarization** via `/summarize` command (admin-only in groups if `admin_mode: true`)
- **Summary generation** uses DeepSeek with specialized prompt preserving key facts
- **History preservation** as `[Previous conversation summary: ...]` message
- **Message debouncing** (4 seconds) to merge rapid consecutive messages from same user
- **Smart media handling**: descriptions truncated at sentence boundaries (max 800 chars/item)
- **Random reply system**: system-level instructions for natural spontaneous responses
- **Time-aware context**: 3+ hour gaps treated as new conversations
- **Streaming delivery**: DeepSeek SSE content is previewed through native drafts in private chats; groups wait for one final answer so an ambiguous Telegram timeout cannot leave a duplicate partial message. Reasoning and tool arguments remain private, and only the final answer is persisted

## Commands

| Command | Description | Access |
|---------|-------------|--------|
| `/start` | Show help | All |
| `/clear` | Clear chat history | All/Admin* |
| `/stats` | Token usage statistics | All |
| `/summarize` | Compress conversation | All/Admin* |
| `/random <0-100>` | Set random reply chance | All/Admin* |

\* When `admin_mode: true` in config

## Logs

Docker writes full DEBUG logs to `/data/bot.log`, mounted on the host as `./bot_data/bot.log`. The file rotates at 10 MB and keeps 5 backups by default.
The production compose stack also runs Dozzle on `127.0.0.1:9999`; host Nginx exposes it at `logs.titlo10.fun` with Basic Auth.

```bash
./scripts/logs.sh          # live Docker logs for the bot
./scripts/logs.sh file     # last lines from bot_data/bot.log
./scripts/logs.sh tail     # follow bot_data/bot.log
./scripts/logs.sh errors   # recent warnings/errors from bot_data/bot.log
```

See [docs/log-viewer.md](docs/log-viewer.md) for the self-hosted browser log viewer setup.

## Configuration

See [docs/configuration.md](docs/configuration.md) for complete configuration reference including:
- All config.yaml parameters
- Memory tuning options
- Debug mode details
- Troubleshooting guide
- Migration notes

## Project Structure

```
Berangaria_bot/
├── berangaria/                  # The bot package (python -m berangaria)
│   ├── __main__.py              # Entry point
│   ├── app.py                   # Startup, handler registration, jobs, shutdown
│   ├── config.py                # config.yaml + env loader, Mem0 config
│   ├── prompts.py               # SYSTEM_PROMPT, vision suffix, Mem0 instructions
│   ├── core/                    # state.py, utils.py, paths.py, logging_setup.py
│   ├── chat/                    # handlers.py, llm_client.py, streaming.py
│   ├── memory/                  # pipeline.py (durable queue), store.py (Mem0)
│   ├── media/                   # vision.py (Gemini image/video/audio)
│   ├── stickers/                # store.py (embeddings + Qdrant search)
│   └── tools/                   # schemas.py, web.py, dispatch.py
├── tests/                       # Mirrors the package layout
├── scripts/                     # start.sh, start.bat, logs.sh, sticker CLI tools
├── docs/                        # Configuration, logging, ADRs, agent docs
├── data/stickers_clean.jsonl    # Sticker source for the Qdrant index
├── deploy/                      # Nginx config for the log viewer
├── config.yaml                  # Main configuration
├── .env                         # Secrets (not committed)
├── docker-compose.yml           # Bot, Qdrant, local Bot API, log viewer
└── requirements.txt             # Python dependencies
```

## Cost Estimation

### DeepSeek v4 Flash (per 1M tokens)
- Regular input: $0.14
- Cached input: $0.0028 (50x cheaper)
- Output: $0.28
- **Typical usage**: 1000 messages ≈ $0.4 (with 70-80% cache hit rate)

### DeepSeek v4 Pro (per 1M tokens)
- Regular input: $0.435
- Cached input: $0.003625
- Output: $0.87
- **Typical usage**: 1000 messages ≈ $1.27 (better instruction following)

### Grok-4.3 (per 1M tokens)
- Input: $1.25
- Output: $2.50
- **Typical usage**: 1000 messages ≈ $4 (best for character/humor)

### Gemini (Vision & Embeddings)
- Vision: Free tier (15 requests/min, 1500 requests/day)
- Embeddings: Free tier (1500 requests/day)
- Files API: Free tier (20GB storage)

**Model selection guide:**
- **Flash**: Best value for general conversations ($0.4/1000)
- **Pro**: Better instruction following, less drift ($1.27/1000)
- **Grok**: Superior character consistency and humor ($4/1000)

## Debug Mode

Set `debug: true` in config.yaml for detailed logging:
- Mem0 configuration on startup
- Memory search queries and results
- Facts being saved
- Token usage per request
- Full prompts sent to LLM

## Troubleshooting

**Vision not working**: 
- Check `GEMINI_API_KEY` in .env
- Verify `vision_mode: true` in config.yaml
- Check Gemini API quota (free tier: 15 req/min)

**Memory errors**: 
- Ensure Qdrant is running: `docker ps`
- Verify embedding model: `gemini-embedding-2` (768 dimensions)
- Check Qdrant logs: `docker logs qdrant`

**High costs**: 
- Check cache hit rate in logs (should be 70-90% after warmup)
- Consider switching to DeepSeek v4 Pro if Flash hallucinates
- Monitor token usage with `/stats` command

**Bot uses emojis in text**:
- This is prompt-driven, not filtered in code. `_clean_reply()` deliberately leaves emoji
  intact so that an emoji-only reply is not turned into silence by the silence-placeholder
  rule. Adjust `SYSTEM_PROMPT` in `berangaria/prompts.py` if the model overuses them.
- Ensure bot restarted after recent updates

**Random replies too frequent/rare**:
- Adjust with `/random <0-100>` command
- `/random` changes are saved in `bot_data/bot_state.db` and survive container restarts
- Check cooldown settings in config.yaml
- Verify `random_reply_chance` and `random_reply_cooldown`

**Media descriptions cut off**:
- Descriptions auto-truncate at sentence boundaries (800 chars)
- Increase `MAX_DESC_CHARS` in `berangaria/chat/handlers.py` if needed
- Check logs for truncation warnings

For detailed troubleshooting, see [docs/configuration.md](docs/configuration.md).

## Documentation

- **[docs/configuration.md](docs/configuration.md)**: Complete configuration reference
- **[docs/logging.md](docs/logging.md)**: Logging system documentation
- **[docs/logging-cheatsheet.md](docs/logging-cheatsheet.md)**: Quick logging reference
- **[docs/log-viewer.md](docs/log-viewer.md)**: Self-hosted Dozzle/Nginx log viewer

## Recent Improvements (January 2026)

**System prompt optimization**:
- Clearer separation: "read tags vs don't write tags" to prevent confusion
- New `EMOJIS AND REACTIONS` section with explicit examples
- Removed duplicate CRITICAL RULE statements
- `_clean_reply()` strips internal service tags (`<think>`, `[#N]`, `[Context from memory: ...]`) — emoji are intentionally preserved

**Vision prompt redesign**:
- Natural conversational style instead of structured reports
- Prompts designed for seamless integration with chat LLM
- Removed technical section headers (DETAILS/RECOGNITION/SUMMARY)
- Better recognition instructions with "похоже на..." for uncertainty

**Smart truncation**:
- Media descriptions now truncate at sentence boundaries
- Prevents broken thoughts in context injection
- Falls back to comma-based or hard truncation if needed

**Random reply improvements**:
- Moved from user message injection to system-level instruction
- Clearer directive: "reply only to current, ignore older"
- Added "don't mention silence" to avoid meta-commentary

**Tools optimization**:
- Shortened `react_to_message` description (180→80 words)
- Cross-reference between SYSTEM_PROMPT and tool definitions
- Emphasis on reaction frequency and silent-reaction option

See [docs/adr/](docs/adr/) for the decisions behind the current design.

## License

GPL-3.0

## Author

MikeAl832
