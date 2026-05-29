# Berangaria Bot

Telegram bot built with `python-telegram-bot`. Uses LLM (DeepSeek or local OpenAI-compatible endpoint) with long-term memory via Mem0 + Qdrant.

## Features

- LLM-powered chat with configurable system prompt
- Long-term memory using Mem0AI and Qdrant vector store
- Automatic and manual conversation summarization
- Optional vision support (image analysis)
- Configurable random reply chance in groups
- Message buffering (combines rapid messages)
- Access control via allowed users/groups lists
- Admin mode for restricting management commands
- Token usage tracking and statistics

## Tech Stack

- Python 3.10+
- python-telegram-bot (with job-queue)
- Mem0AI
- Qdrant (vector database)
- DeepSeek API or local LLM (LM Studio / any OpenAI-compatible server)
- PyYAML, python-dotenv, httpx, ddgs, fastembed

## Installation

### Using Docker Compose (recommended for Qdrant)

```bash
git clone https://github.com/MikeAl832/Berangaria_bot.git
cd Berangaria_bot
docker-compose up -d
```

### Local installation

```bash
git clone https://github.com/MikeAl832/Berangaria_bot.git
cd Berangaria_bot

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Windows users**: You can use the provided `start.bat` script. It automatically creates a virtual environment (if missing), installs dependencies, and starts the bot.

## Configuration

### Environment variables

Create `.env` file:

```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
API_KEY=your_deepseek_api_key          # or leave empty for local models
```

### config.yaml

Main settings:

- `provider`: `deepseek` or `lmstudio`
- `model`: main chat model (DeepSeek API)
- `mem0_model`: local non-thinking model for memory extraction & summarization (e.g. `qwen3-4b-instruct-2507@q8_0`); leave empty to use `model`
- `vision_provider`: `gemini` (free Google API, native video) or `lmstudio` (local Qwen3-VL)
- `vision_model`: local vision model name when `vision_provider=lmstudio`
- `gemini_model`: Gemini model when `vision_provider=gemini`
- `vision_mode`: enable/disable image and video analysis
- `admin_mode`: restrict `/clear`, `/summarize`, `/random` to group admins
- `allowed_users`, `allowed_groups`: access lists (Telegram IDs)
- `bot_names`: triggers for mentioning the bot in groups
- Memory settings (Qdrant host/port, embedding model)
- Generation parameters (temperature, max tokens, etc.)

**Note**: Qdrant must be running (default: `localhost:6333`). Use the provided `docker-compose.yml`.

## Running

```bash
# Windows (recommended)
start.bat

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

- `main.py` — application entry point and handler registration
- `handlers.py` — command and message handlers + message buffering logic
- `llm_client.py` — LLM request handling
- `memory_store.py` — Mem0 integration
- `state.py` — in-memory conversation histories and token tracking
- `tools.py` — tool definitions (web_search)
- `config.py` + `config.yaml` — centralized configuration

The bot maintains per-chat history with token budgeting and optional automatic summarization.

## License

MIT

## Author

MikeAl832
