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
```

### config.py

Python configuration loader. Auto-generates Mem0 config and system prompt. No manual editing required.

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
vision_provider: "gemini"
gemini_model: "gemini-3.1-flash-lite"
video_max_duration_sec: 60
```

**Parameters:**
- `vision_mode`: Enable/disable image and video understanding
- `vision_provider`: Must be "gemini" (LM Studio removed)
- `gemini_model`: Gemini model for vision tasks
- `video_max_duration_sec`: Maximum video length in seconds

**Available Gemini models:**
- `gemini-2.0-flash` (free tier)
- `gemini-2.5-flash` (free tier)
- `gemini-3.1-flash-lite` (free tier, current)

### Memory (Mem0 + Embeddings)

```yaml
embedding_model: "models/text-embedding-004"
embedding_dims: 768
memory_search_limit: 10
memory_min_score: 0.2
memory_max_chars: 1200
```

**Parameters:**
- `embedding_model`: Gemini embedding model (must include "models/" prefix)
- `embedding_dims`: Vector dimensions (fixed at 768 for text-embedding-004)
- `memory_search_limit`: Number of facts retrieved per query
- `memory_min_score`: Relevance threshold (0.0-1.0)
- `memory_max_chars`: Maximum total length of memory context

**Relevance threshold guide:**
- `0.1`: Very permissive (includes many facts)
- `0.2`: Soft filtering (current default)
- `0.3`: Balanced
- `0.5`: Strict (only highly relevant facts)

### Bot Behavior

```yaml
bot_names: ["Бер", "Ber"]
random_reply_chance: 10
summary_interval: 10
admin_mode: false
debug: true
```

**Parameters:**
- `bot_names`: Names that trigger bot responses in groups
- `random_reply_chance`: Probability (0-100) of spontaneous group replies
- `summary_interval`: Messages preserved after summarization
- `admin_mode`: Restrict management commands to group admins
- `debug`: Enable detailed logging

### Access Control

```yaml
allowed_users: [1217938322, 1809564460]
allowed_groups: [-1002263830880]
```

**Parameters:**
- `allowed_users`: Telegram user IDs with bot access
- `allowed_groups`: Telegram group IDs where bot operates

Get IDs by sending `/start` to the bot with debug mode enabled.

### Cost Tracking

```yaml
price_prompt_cache_miss: 0.14
price_prompt_cache_hit: 0.0028
price_completion: 0.28
```

**Parameters (per 1M tokens):**
- `price_prompt_cache_miss`: Regular input tokens
- `price_prompt_cache_hit`: Cached input tokens
- `price_completion`: Output tokens

Current prices for DeepSeek v4 Flash. Update when prices change.

## Memory Configuration

### Custom Instructions

Located in `config.py` as `MEM0_CUSTOM_INSTRUCTIONS`. Current version uses minimal instructions:

```python
MEM0_CUSTOM_INSTRUCTIONS = """
Extract facts about people in Russian.
Group chats: name at message start
Don't extract: bot meta-comments
"""
```

### Three Configuration Options

**Option 1: Minimal (current, recommended)**
- Short instructions (~10 lines)
- Mem0 decides what's important
- Group chat attribution handled

**Option 2: No instructions**
```python
# Comment out in config.py:
# "custom_instructions": MEM0_CUSTOM_INSTRUCTIONS,
```
- Standard Mem0 behavior
- May be less accurate for Russian

**Option 3: Detailed**
- Add comprehensive extraction rules
- More control over what's saved
- Risk of over-filtering

### What Gets Remembered

**Included:**
- Personal information and facts
- Preferences and opinions
- Questions (show interests)
- Current activities and projects
- Technical details
- Communication style

**Excluded:**
- Bare greetings
- Meaningless interjections
- Bot meta-comments (API, models, settings)

## Debug Mode

Enable with `debug: true` in config.yaml.

### Startup Output

```
Mem0 configuration:
{
  "llm_provider": "deepseek",
  "llm_model": "deepseek-v4-flash",
  "embedder_provider": "gemini",
  "embedder_model": "models/text-embedding-004",
  "vector_store": "qdrant"
}
Mem0 initialized
```

### Memory Operations

```
Mem0 search: query='configure bot', scope=private_123456
Mem0 results: found 5 facts
Memory: 5 facts, 345 chars
Facts:
- User works with DeepSeek API
- User studies embeddings
...

Mem0 save: 'configured Gemini embeddings...' (scope=private_123456)
Memory saved: 2 facts (scope=private_123456)
```

### Request Tracking

```
Tokens: request=1234 (cache=980), reply=567, total=1801
Request cost: $0.000285
```

## Troubleshooting

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
4. Verify embedding model name: `models/text-embedding-004`

**Common error:** "Unsupported embedding provider: googleai"
- Fix: In config.py, provider should be "gemini" (not "googleai" or "google")

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
1. Enable debug mode and check memory saves
2. Lower `memory_min_score` to 0.1
3. Increase `memory_search_limit` to 15-20
4. Try removing custom_instructions
5. Verify Qdrant has data: check qdrant_storage/ directory size

## Migration Guide

### From LM Studio Version

**Configuration changes:**

Removed parameters:
- `mem0_model` (use main model)
- `provider` (hardcoded in config.py)
- `embedding_provider` (hardcoded)
- `base_url` (not needed for cloud APIs)
- `vision_model` (local models removed)
- `video_max_frames` (Gemini handles full video)

Added parameters:
- `embedding_model` (Gemini model name)
- `embedding_dims` (vector dimensions)

**Code changes:**

All LM Studio references removed from:
- config.py
- vision_provider.py
- llm_client.py
- handlers.py

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

- Start with `memory_min_score: 0.2`
- Enable debug mode to audit what's saved
- Adjust custom_instructions if needed
- Use minimal instructions for best results

### System Performance

- Qdrant runs locally (fast, no network latency)
- Gemini embeddings are free tier
- DeepSeek v4 flash is optimized for speed

## Advanced Configuration

### Custom System Prompt

Edit `SYSTEM_PROMPT` in config.py. Current prompt defines personality, communication style, and behavior rules.

### Mem0 Configuration

Full Mem0 config in `MEM0_CONFIG` dictionary (config.py):

```python
MEM0_CONFIG = {
    "version": "v1.1",
    "custom_instructions": MEM0_CUSTOM_INSTRUCTIONS,
    "llm": {
        "provider": "deepseek",
        "config": {
            "model": MODEL,
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

## File Structure

```
Berangaria_bot/
├── config.yaml          # Main configuration
├── .env                 # Secrets (gitignored)
├── config.py            # Config loader and Mem0 setup
├── main.py              # Entry point
├── handlers.py          # Telegram event handlers
├── llm_client.py        # DeepSeek client
├── vision_provider.py   # Gemini vision client
├── memory_store.py      # Mem0 initialization
├── state.py             # In-memory state
├── utils.py             # Helper functions
├── tools.py             # Tool definitions
├── docker-compose.yml   # Qdrant container
├── requirements.txt     # Python dependencies
└── qdrant_storage/      # Vector DB data (auto-created)
```

## References

- [DeepSeek API Docs](https://platform.deepseek.com/docs)
- [Google AI Studio](https://aistudio.google.com)
- [Mem0 Documentation](https://docs.mem0.ai)
- [Qdrant Documentation](https://qdrant.tech/documentation)
