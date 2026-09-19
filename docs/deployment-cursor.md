# Deployment on cursor

Production lives in `/home/box/Berangaria_bot` on the Grok Bot VM (`cursor` or `grok-bot-vm-*`), running as `box`.
Pushes to `main` and manual workflow runs first execute tests, Ruff and Bandit
on GitHub-hosted runners. The deploy job then runs on the repository's
self-hosted runner with the `berangaria-cursor` label.

The runner connects directly to GitHub over outbound HTTPS. Deployment does
not use the old VPS, its SSH keys, or the reverse SSH tunnel. Keep the cursor
environment, Docker daemon and runner running. The workflow has no pull-request
trigger, and the production job additionally requires `refs/heads/main`.

The runner is installed in `/home/box/berangaria-actions-runner`. Its registration
requires a repository owner/admin registration token from GitHub Settings →
Actions → Runners → New self-hosted runner. The one-time token expires after
one hour; the configured runner maintains its own credentials afterward.

To register the prepared runner, run as `box` on the production VM (substitute the token
from the repository settings):

```sh
cd /home/box/berangaria-actions-runner
./config.sh --unattended --url https://github.com/MikeAl832/Berangaria_bot \
  --token REGISTRATION_TOKEN --name berangaria-cursor \
  --labels berangaria-cursor --work _work
```

The installed `supervise-runner.sh` process waits for registration and starts
`run.sh` automatically, restarting it if it exits. If the cursor environment
itself is restarted, start Docker and the supervisor again:

```sh
setsid -f /home/box/berangaria-actions-runner/supervise-runner.sh </dev/null >/dev/null 2>&1
```

The supervisor uses a lock to prevent duplicate instances. Its log is
`/home/box/berangaria-actions-runner/supervisor.log`.

`scripts/deploy-cursor.sh` requires the migrated databases and a clean tracked
checkout, fetches the exact workflow commit, and checks it out without deleting
untracked state. Existing `.env` values survive missing GitHub secrets. The bot
is rebuilt before containers are updated; readiness requires Telegram
authentication and a subsequent stability check. Failure output contains
container status, not private chat logs.

`deploy/compose.cursor.yml` uses host networking for builds and services because
cursor's Docker daemon has bridge NAT disabled. Qdrant and
Dozzle bind only to loopback. Dependency image tags point to the exact images
copied during the 2026-09-14 migration; deployment does not pull newer versions.

For manual operations:

```sh
cd /home/box/Berangaria_bot
docker compose -f docker-compose.yml -f deploy/compose.cursor.yml ps
docker compose -f docker-compose.yml -f deploy/compose.cursor.yml logs --tail 100 bot
```

SQLite, logs and the bot Telethon media session remain in `bot_data/`, Qdrant
in `qdrant_storage/`. The local Bot API server has been removed; follow the
[one-time migration](configuration.md#telegram-media-downloads) before the first deploy. Never run the old VPS bot alongside
this installation. The old `VPS_*` GitHub secrets are no longer used by the
workflow and may be removed separately.

### Telegram Conflict and stalled polling

HTTP 409 (`telegram.error.Conflict`) means two concurrent `getUpdates` long-polls
on this bot token. The owner DM field `host=` is the process that *caught* the
409, not the identity of the other poller. After switching off the local Bot API
(15 Sep 2026) this process talks to `api.telegram.org` directly; a short
getUpdates HTTP read timeout on us-west-2 can abort a still-registered long-poll
and the PTB retry then 409s itself. The builder sets `get_updates_*_timeout` to
the same 60s budget as outgoing Bot API calls. After Conflict the process logs
CRITICAL, writes `analytics_alerts`, notifies once per 60s (volatile
`uptime` / `since_update` / `updates` are not part of the cooldown fingerprint),
and `os._exit(1)` so Docker `restart: always` starts a clean poller.

A heartbeat every 10 minutes logs `since_update=` and pets a daemon watchdog
thread. If this process has already seen updates and then goes silent for 30
minutes, `bot.log` gets a WARNING (`Polling stalled`) only — no owner DM
(quiet nights are normal). If the asyncio loop itself stops (no heartbeat for
20 minutes) the watchdog `os._exit(1)` so Docker restarts — a live PID with a
frozen loop will not recover on its own. Edit/delete handlers and the Telethon
user bridge are not a second Bot API poller.

The old `logs.titlo10.fun` website still uses Nginx and a forwarding service on
the old VPS. Moving its domain/TLS endpoint is separate from deployment; local
Dozzle on cursor (`127.0.0.1:9999`) does not depend on the VPS.
