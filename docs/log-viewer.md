# Self-Hosted Log Viewer (Dozzle)

`docker-compose.yml` includes a local [Dozzle](https://dozzle.dev/) instance bound strictly to `127.0.0.1:9999` for browser-based container and audit-log inspection.

## Runtime Layout

The full bot audit log (`bot.log`) is streamed through a lightweight `log-tail` sidecar so Dozzle can display both clean INFO console output and full DEBUG audit traces side by side:

```text
bot       -> writes /data/bot.log (./bot_data/bot.log on host)
log-tail  -> tail -n 300 -F /logs/bot.log
dozzle    -> reads Docker stdout of containers labeled dev.berangaria.logs=visible
```

Containers visible in Dozzle:
- `*-bot-1`: clean `INFO` console stream (incoming previews, token/cache stats, tool summaries, alerts).
- `*-log-tail-1`: full `DEBUG` audit stream (`bot_data/bot.log`, including prompts and responses when `full_debug_logs: true`).
- `*-qdrant-1`: Qdrant vector database logs.

## Secure Access

Because `full_debug_logs: true` records full chat prompts and replies, Dozzle binds to `127.0.0.1:9999` only and is never exposed on `0.0.0.0` by default.

### Option A: SSH Port Forwarding (Recommended, Zero Extra Setup)

From your workstation:

```bash
ssh -L 9999:127.0.0.1:9999 user@your-server
```

Then open `http://127.0.0.1:9999` in your local browser.

### Option B: Reverse Proxy with TLS & Authentication

If you expose Dozzle on a custom domain (via Nginx, Caddy, or Cloudflare Tunnel), always enforce authentication (e.g., Basic Auth, Cloudflare Access, or Dozzle's `DOZZLE_AUTH_PROVIDER=simple` with a mounted `users.yml`) before routing traffic to `http://127.0.0.1:9999`.

## Debug Flags (`config.yaml` / env)

- `debug: false`: keeps console stream at `INFO` while `bot.log` still captures `DEBUG`.
- `full_debug_logs: true` (`BOT_FULL_DEBUG_LOGS`): writes full LLM payloads, model replies, memory blocks, and media descriptions to `bot.log`. Set to `false` if you want technical `DEBUG` events without prompt/message bodies.
- `verbose: false`: suppresses noisy low-level SDK/HTTP/TLS frame logs (`httpx`, `httpcore2`, `google_genai`, `apscheduler`).
