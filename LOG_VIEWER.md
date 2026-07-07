# Self-hosted log viewer

Berangaria uses a local Dozzle instance for browser-based log viewing.

## Runtime layout

```text
https://logs.titlo10.fun
  -> host Nginx with Basic Auth
  -> 127.0.0.1:9999
  -> dozzle
  -> Docker logs
```

The full bot audit log is streamed through a dedicated `log-tail` container:

```text
bot -> /data/bot.log
./bot_data/bot.log on host
log-tail container -> tail -n 300 -F /logs/bot.log
dozzle -> reads log-tail stdout
```

Dozzle only sees containers labeled with `dev.berangaria.logs=visible`.

## What to watch

- `berangaria_bot-log-tail-1`: full `bot_data/bot.log`, including detailed LLM DEBUG logs.
- `berangaria_bot-bot-1`: regular Docker console logs.
- `berangaria_bot-qdrant-1`: Qdrant logs.

## Debug levels

- `debug: false`: keeps Docker console readable.
- `full_debug_logs: true`: writes detailed prompts, model replies, memory facts, and vision descriptions to `bot_data/bot.log`.
- `verbose: false`: keeps third-party HTTP/Telegram SDK noise out.

Docker overrides `full_debug_logs` with `BOT_FULL_DEBUG_LOGS=true`.

## DNS and TLS

Create a Cloudflare DNS record:

```text
logs.titlo10.fun -> VPS public IPv4
```

After DNS resolves to the VPS, run on the server:

```bash
sudo certbot --nginx -d logs.titlo10.fun
```

Until TLS is issued, the host Nginx config serves the log viewer on HTTP. If
Cloudflare proxying is enabled, keep Basic Auth at the origin anyway.

## Credentials

Nginx Basic Auth is configured on the host, outside the repository:

```text
/etc/nginx/.htpasswd-berangaria-logs
```

The generated initial password is stored on the server in:

```text
/root/berangaria-logs-credentials.txt
```

Rotate it with:

```bash
printf 'logs:%s\n' "$(openssl passwd -apr1 'new-password')" | sudo tee /etc/nginx/.htpasswd-berangaria-logs >/dev/null
sudo nginx -t && sudo systemctl reload nginx
```
