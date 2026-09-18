#!/usr/bin/env bash
set -euo pipefail

case "$(hostname)" in
  cursor|grok-bot-vm-*) ;;
  *)
    echo "Deploy preflight failed: runner host is not an allowed production host (cursor or grok-bot-vm-*)."
    exit 1
    ;;
esac
test "$(id -un)" = box || {
  echo "Deploy preflight failed: runner user is not box."
  exit 1
}
[[ "${DEPLOY_SHA:-}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "Deploy preflight failed: DEPLOY_SHA is invalid."
  exit 1
}
cd /home/box/Berangaria_bot || {
  echo "Deploy preflight failed: /home/box/Berangaria_bot is unavailable."
  exit 1
}

# Require the migrated installation; never create an empty production database.
test -d .git || {
  echo "Deploy preflight failed: production Git checkout is missing."
  exit 1
}
test -s .env || {
  echo "Deploy preflight failed: production .env is missing or empty."
  exit 1
}
test -s bot_data/bot_state.db || {
  echo "Deploy preflight failed: production database is missing or empty."
  exit 1
}
test -d qdrant_storage || {
  echo "Deploy preflight failed: Qdrant storage is missing."
  exit 1
}
git diff --quiet || {
  echo "Deploy preflight failed: production checkout has unstaged tracked changes."
  echo "Changed tracked files:"
  git diff --name-only
  exit 1
}
git diff --cached --quiet || {
  echo "Deploy preflight failed: production checkout has staged changes."
  exit 1
}
git fetch origin "$DEPLOY_SHA"
git checkout --detach "$DEPLOY_SHA"
test "$(git rev-parse HEAD)" = "$DEPLOY_SHA"

compose() {
  docker compose -f docker-compose.yml -f deploy/compose.cursor.yml "$@"
}

# Keep existing values when an optional GitHub secret is absent. Unset empty
# exports so they cannot override valid values from Compose's .env file.
upsert_env() {
  local key="$1" value="$2"
  if [ -z "$value" ]; then
    echo "Keeping existing ${key}"
    unset "$key"
    return
  fi
  if grep -qE "^${key}=" .env; then
    grep -vE "^${key}=" .env > .env.upsert.tmp || true
    mv .env.upsert.tmp .env
  fi
  printf '%s=%s\n' "$key" "$value" >> .env
  echo "Updated ${key}"
}

umask 077
chmod 600 .env
upsert_env FISH_API_KEY "${FISH_API_KEY:-}"
upsert_env FISH_VOICE_ID "${FISH_VOICE_ID:-}"
upsert_env TELEGRAM_API_ID "${TELEGRAM_API_ID:-}"
upsert_env TELEGRAM_API_HASH "${TELEGRAM_API_HASH:-}"
upsert_env USER_BRIDGE_SESSION "${USER_BRIDGE_SESSION:-}"
upsert_env OPENROUTER_API_KEY "${OPENROUTER_API_KEY:-}"
upsert_env XAI_API_KEY "${XAI_API_KEY:-}"

compose config --quiet
compose build bot
# Stop the old poller before starting the new one to avoid Telegram Conflict
# from overlapping getUpdates during recreate.
compose stop bot || true
compose up -d --no-build --pull never --remove-orphans

bot_id="$(compose ps -q bot)"
test -n "$bot_id"
started_at="$(docker inspect -f '{{.State.StartedAt}}' "$bot_id")"
initial_restarts="$(docker inspect -f '{{.RestartCount}}' "$bot_id")"

check_running() {
  local status restart_count
  status="$(docker inspect -f '{{.State.Status}}' "$bot_id")"
  restart_count="$(docker inspect -f '{{.RestartCount}}' "$bot_id")"
  if [ "$status" != running ] || [ "$restart_count" != "$initial_restarts" ]; then
    echo "Bot stopped or restarted during deployment."
    compose ps --all
    exit 1
  fi
}

echo "Waiting for Telegram initialization..."
ready=0
for attempt in $(seq 1 30); do
  check_running
  # Match only the current process's logs. Do not publish private chat logs
  # to GitHub Actions on failure. Avoid grep -q/SIGPIPE with pipefail enabled.
  if docker logs --since "$started_at" "$bot_id" 2>&1 | grep 'Бот запущен' > /dev/null; then
    ready=1
    break
  fi
  sleep 4
done
if [ "$ready" -ne 1 ]; then
  echo "Bot did not authenticate with Telegram within 120 seconds."
  compose ps --all
  exit 1
fi

for attempt in $(seq 1 5); do
  sleep 4
  check_running
done
echo "Bot initialized and remained stable at ${DEPLOY_SHA}."
