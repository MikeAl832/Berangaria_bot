# Berangaria Bot

Telegram bot built with `python-telegram-bot`. A sharp-witted chat personality with long-term memory, image and video understanding, and web search. Uses a multi-model setup: a main chat LLM (DeepSeek), a vision provider (Google Gemini or a local model), and a local model for memory extraction.

## Features

- LLM-powered chat with a configurable personality system prompt
- Long-term memory using Mem0AI + Qdrant vector store, scoped per chat (shared memory in groups)
- Image and video understanding (vision mode) via Gemini API or a local vision model
- Web search tool (`web_search`) for up-to-date facts
- Automatic and manual conversation summarization
- Configurable random reply chance in groups
- Message buffering (combines rapid consecutive messages)
- Access control via allowed users/groups lists
- Admin mode for restricting management commands
- Token usage and cost tracking

## How vision works

When a user sends a photo or video, it is **not** sent to the main chat model. Instead:

1. A dedicated vision provider analyzes the media and produces a structured text description (`DETAILS / RECOGNITION / SUMMARY`).
2. That description is injected into the conversation as `[Image description: ...]` / `[Video description: ...]`.
3. The main chat model reacts to the description as if it saw the media itself.

Two vision providers are supported (`vision_provider` in `config.yaml`):

- **`gemini`** — Google Gemini API. Free tier, native video understanding (no frame extraction), best character/meme recognition. Requires `GEMINI_API_KEY`.
- **`lmstudio`** — local vision model (e.g. Qwen3-VL) via LM Studio. For video, frames are extracted with ffmpeg (`imageio-ffmpeg`) and sent as images.

Video length is capped by `video_max_duration_sec`; longer videos are rejected.

## Memory model

- Memory is partitioned by chat: in groups it is **shared across the whole chat** (`group_<chat_id>`), in private chats it is per-user (`private_<user_id>`).
- Only the user's own message text is stored — bot replies and media descriptions are excluded.
- Fact extraction uses a local non-thinking model (`mem0_model`) with custom Russian instructions, so only meaningful facts about people are kept (no greetings, reactions, or meta-comments).
- Retrieval is filtered by relevance score and capped in count and length.

## Tech Stack

- Python 3.10+
- python-telegram-bot (with job-queue)
- Mem0AI + Qdrant (vector database)
- DeepSeek API (main chat) — any OpenAI-compatible endpoint also works
- Google Gemini API (vision) and/or LM Studio (local vision + memory model)
- imageio-ffmpeg (video frame extraction for local vision)
- PyYAML, python-dotenv, httpx, ddgs, fastembed, Pillow

## Installation

### Qdrant (required, via Docker Compose)

```bash
git clone https://github.com/MikeAl832/Berangaria_bot.git
cd Berangaria_bot
docker-compose up -d   # starts Qdrant on localhost:6333
```

### Bot dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Windows users**: you can use the provided `start.bat`. It creates the virtual environment (if missing), installs dependencies, and starts the bot. Linux users have `start.sh`.

## Configuration

### Environment variables

Create a `.env` file:

```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
API_KEY=your_deepseek_api_key            # main chat model (DeepSeek)
GEMINI_API_KEY=your_gemini_api_key       # required only when vision_provider=gemini
```

### config.yaml

Main chat:
- `model`: main chat model name (DeepSeek API)
- `generation_params`: temperature, top_p, etc. for the main model
- `max_context_tokens`, `max_reply_tokens`

Vision:
- `vision_mode`: enable/disable image and video analysis
- `vision_provider`: `gemini` or `lmstudio`
- `gemini_model`: Gemini model when provider is `gemini` (e.g. `gemini-2.0-flash`, `gemini-2.5-flash`)
- `vision_model`: local vision model name when provider is `lmstudio` (e.g. `qwen3-vl-8b-thinking`)
- `video_max_duration_sec`: max video length accepted
- `video_max_frames`: frames sampled from video (local provider only; Gemini gets the whole video)

Memory (Mem0):
- `mem0_model`: local non-thinking model for memory extraction & summarization (e.g. `gigachat3.1-10b-a1.8b`); leave empty to fall back to `model`
- `provider`: backend for the memory/vision local models (`lmstudio` or `deepseek`)
- `base_url`: LM Studio API base URL
- `embedding_model`: embedding model for the vector store
- `memory_search_limit`: number of facts injected into context (top-N by relevance)
- `memory_min_score`: minimum relevance score for a fact (0..1)
- `memory_max_chars`: max total length of the injected memory block

General:
- `admin_mode`: restrict `/clear`, `/summarize`, `/random` to group admins
- `allowed_users`, `allowed_groups`: access lists (Telegram IDs)
- `bot_names`: triggers for mentioning the bot in groups
- `random_reply_chance`: probability of a spontaneous reply in groups
- `summary_interval`: messages kept after summarization
- `debug`: when true, logs full prompts and full vision descriptions to `bot.log`
- `price_*`: per-1M-token prices used for cost estimation

**Note**: Qdrant must be running (default `localhost:6333`). Use the provided `docker-compose.yml`.

> **LM Studio note**: load the `mem0_model` (and the local `vision_model`, if used) with a context length of at least **16384** tokens. The memory extraction prompt plus dialogue easily exceeds the default 4096.

## Running

```bash
# Windows
start.bat

# Linux
./start.sh

# Manual start (any OS)
python -m main
```

Logs are written to `bot.log`.

## Commands

| Command              | Description                                      | Notes                     |
|----------------------|--------------------------------------------------|---------------------------|
| `/start`             | Show help message                                | -                         |
| `/clear`             | Clear conversation history for current chat      | Admin mode restricts it   |
| `/stats`             | Show message count and token usage               | -                         |
| `/summarize`         | Manually trigger conversation summarization      | Admin mode restricts it   |
| `/random <0-100>`    | Set random reply probability in groups           | Admin mode restricts it   |

## Architecture Notes

- `main.py` — entry point, logging setup, handler registration
- `handlers.py` — command/message handlers, message buffering, photo and video handling
- `llm_client.py` — main LLM request loop, summarization, memory search/save
- `vision_provider.py` — image/video description via Gemini or LM Studio
- `memory_store.py` — Mem0 initialization
- `utils.py` — media download, video frame extraction, mention/random-reply helpers
- `state.py` — in-memory conversation histories, token tracking, message buffers
- `tools.py` — tool definitions (`web_search`)
- `config.py` + `config.yaml` — centralized configuration (including Mem0 config and system prompt)

The bot maintains per-chat history with token budgeting and optional automatic summarization.

## License

MIT

## Author

MikeAl832
