# Deployment on cursor

Production lives in `/opt/Berangaria_bot` on `cursor`, running as `box`.
Pushes to `main` and manual workflow runs first execute tests, Ruff and Bandit
on GitHub-hosted runners. The deploy job then runs on the repository's
self-hosted runner with the `berangaria-cursor` label.

The runner connects directly to GitHub over outbound HTTPS. Deployment does
not use the old VPS, its SSH keys, or the reverse SSH tunnel. Keep the cursor
environment, Docker daemon and runner running. The workflow has no pull-request
trigger, and the production job additionally requires `refs/heads/main`.

The runner is installed in `/opt/berangaria-actions-runner`. Its registration
requires a repository owner/admin registration token from GitHub Settings →
Actions → Runners → New self-hosted runner. The one-time token expires after
one hour; the configured runner maintains its own credentials afterward.

To register the prepared runner, run as `box` on cursor (substitute the token
from the repository settings):

```sh
cd /opt/berangaria-actions-runner
./config.sh --unattended --url https://github.com/MikeAl832/Berangaria_bot \
  --token REGISTRATION_TOKEN --name berangaria-cursor \
  --labels berangaria-cursor --work _work
```

The installed `supervise-runner.sh` process waits for registration and starts
`run.sh` automatically, restarting it if it exits. If the cursor environment
itself is restarted, start Docker and the supervisor again:

```sh
setsid -f /opt/berangaria-actions-runner/supervise-runner.sh </dev/null >/dev/null 2>&1
```

The supervisor uses a lock to prevent duplicate instances. Its log is
`/opt/berangaria-actions-runner/supervisor.log`.

`scripts/deploy-cursor.sh` requires the migrated databases and a clean tracked
checkout, fetches the exact workflow commit, and checks it out without deleting
untracked state. Existing `.env` values survive missing GitHub secrets. The bot
is rebuilt before containers are updated; readiness requires Telegram
authentication and a subsequent stability check. Failure output contains
container status, not private chat logs.

`deploy/compose.cursor.yml` uses host networking for builds and services because
cursor's Docker daemon has bridge NAT disabled. Qdrant, Telegram Bot API and
Dozzle bind only to loopback. Dependency image tags point to the exact images
copied during the 2026-09-14 migration; deployment does not pull newer versions.

For manual operations:

```sh
cd /opt/Berangaria_bot
docker compose -f docker-compose.yml -f deploy/compose.cursor.yml ps
docker compose -f docker-compose.yml -f deploy/compose.cursor.yml logs --tail 100 bot
```

SQLite and logs remain in `bot_data/`, Qdrant in `qdrant_storage/`, and Telegram
Bot API data in `/var/lib/telegram-bot-api`. Never run the old VPS bot alongside
this installation. The old `VPS_*` GitHub secrets are no longer used by the
workflow and may be removed separately.

The old `logs.titlo10.fun` website still uses Nginx and a forwarding service on
the old VPS. Moving its domain/TLS endpoint is separate from deployment; local
Dozzle on cursor (`127.0.0.1:9999`) does not depend on the VPS.
