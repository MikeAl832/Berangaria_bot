# Berangaria Bot

Telegram bot with long-term memory, vision understanding, and web search capabilities. Built with python-telegram-bot, featuring a sharp-witted personality and multi-modal understanding.

## Architecture

- **Main LLM**: DeepSeek v4 Flash (chat, summarization, memory extraction)
- **Vision**: Google Gemini API (image/video understanding)
- **Embeddings**: Google Gemini API (memory vectors)
- **Vector Store**: Qdrant (local Docker container)
- **Memory**: Mem0 (fact extraction and retrieval)

## Key Features

- Persistent memory scoped per chat (shared in groups, private in DMs)
- Vision mode for analyzing images and videos
- Web search tool for current information
- Automatic conversation summarization
- Token usage tracking and cost estimation
- Configurable random replies in groups
- Message buffering for rapid consecutive messages

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
```

API keys:
- Telegram: [@BotFather](https://t.me/botfather)
- DeepSeek: [platform.deepseek.com](https://platform.deepseek.com)
- Gemini: [aistudio.google.com](https://aistudio.google.com)

### 4. Configure bot settings

Edit `config.yaml` - see [CONFIG_README.md](CONFIG_README.md) for detailed options.

Key settings:
- `model`: DeepSeek model name
- `vision_mode`: enable/disable vision
- `embedding_model`: Gemini embedding model
- `memory_search_limit`: facts injected into context
- `allowed_users` / `allowed_groups`: access control

### 5. Run

```bash
python main.py
```

Windows users can use `start.bat`, Linux users `start.sh`.

## How It Works

### Memory System

Memory is partitioned by chat:
- **Groups**: shared memory for entire chat (`group_<chat_id>`)
- **Private**: per-user memory (`private_<user_id>`)

Only user messages are stored (bot replies and media descriptions excluded). Facts are extracted using DeepSeek with minimal custom instructions - Mem0 decides what's important. Retrieval uses semantic similarity with configurable thresholds.

### Vision Processing

When media is received:
1. Gemini API analyzes the image/video
2. Structured description generated (DETAILS / RECOGNITION / SUMMARY)
3. Description injected as `[Image description: ...]` or `[Video description: ...]`
4. Main LLM responds to description as if it saw the media

Video processing: Gemini handles full video natively (no frame extraction needed). Max duration controlled by `video_max_duration_sec`.

### Conversation Management

- Token budgeting per chat with automatic summarization at 85% capacity
- Manual summarization via `/summarize` command
- Summary generation uses DeepSeek with specialized prompt
- History preserved as `[Previous conversation summary: ...]`

## Commands

| Command | Description | Access |
|---------|-------------|--------|
| `/start` | Show help | All |
| `/clear` | Clear chat history | All/Admin* |
| `/stats` | Token usage statistics | All |
| `/summarize` | Compress conversation | All/Admin* |
| `/random <0-100>` | Set random reply chance | All/Admin* |

\* When `admin_mode: true` in config

## Configuration

See [CONFIG_README.md](CONFIG_README.md) for complete configuration reference including:
- All config.yaml parameters
- Memory tuning options
- Debug mode details
- Troubleshooting guide
- Migration notes

## Project Structure

```
Berangaria_bot/
├── main.py              # Entry point
├── handlers.py          # Telegram handlers
├── llm_client.py        # DeepSeek API client
├── vision_provider.py   # Gemini vision
├── memory_store.py      # Mem0 initialization
├── state.py             # In-memory state
├── utils.py             # Utilities
├── tools.py             # Tool definitions
├── config.py            # Configuration loader
├── config.yaml          # Main configuration
├── .env                 # Secrets (not committed)
├── docker-compose.yml   # Qdrant container
└── requirements.txt     # Python dependencies
```

## Cost Estimation

### DeepSeek (per 1M tokens)
- Regular input: $0.14
- Cached input: $0.0028 (50x cheaper)
- Output: $0.28

Typical usage: 1000 messages ~$1.00 (with 80% cache hit rate)

### Gemini
- Vision: Free (free tier)
- Embeddings: Free (free tier)

## Debug Mode

Set `debug: true` in config.yaml for detailed logging:
- Mem0 configuration on startup
- Memory search queries and results
- Facts being saved
- Token usage per request
- Full prompts sent to LLM

## Troubleshooting

**Vision not working**: Check `GEMINI_API_KEY` in .env and `vision_mode: true` in config.yaml

**Memory errors**: Ensure Qdrant is running (`docker ps`) and embedding model name is correct (`models/text-embedding-004`)

**High costs**: Check cache hit rate in logs (should be 70-90% after warmup)

**Nothing remembered**: Enable debug mode, check memory saves in logs, lower `memory_min_score` to 0.1

For detailed troubleshooting, see [CONFIG_README.md](CONFIG_README.md).

## Changes from Previous Version

**Removed**:
- LM Studio support (full cloud migration)
- Local vision models (Qwen3-VL)
- Local embedding models (HuggingFace)
- Multi-provider complexity

**Simplified**:
- Single LLM provider (DeepSeek)
- Single vision provider (Gemini)
- Single embedding provider (Gemini)
- Cleaner configuration structure

## License

GPL-3.0

## Author

MikeAl832

