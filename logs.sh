#!/usr/bin/env bash
# Удобный просмотр логов Berangaria_bot на сервере и локально.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

LINES="${LINES:-200}"
SERVICE="${SERVICE:-bot}"
HOST_LOG_FILE="${HOST_LOG_FILE:-bot_data/bot.log}"

usage() {
    cat <<'EOF'
Usage: ./logs.sh [follow|file|tail|errors|qdrant|path]

Commands:
  follow   Show live Docker logs for the bot service (default)
  file     Print the persistent host log file from bot_data/bot.log
  tail     Follow the persistent host log file
  errors   Print recent WARNING/ERROR/CRITICAL lines from the persistent log
  qdrant   Show live Docker logs for Qdrant
  path     Print the persistent host log path

Environment:
  LINES=200                 Number of lines to show
  SERVICE=bot               Docker Compose service for follow mode
  HOST_LOG_FILE=bot_data/bot.log
EOF
}

compose_cmd() {
    if docker compose version >/dev/null 2>&1; then
        printf 'docker compose'
    elif command -v docker-compose >/dev/null 2>&1; then
        printf 'docker-compose'
    else
        printf 'docker compose'
    fi
}

require_log_file() {
    if [[ ! -f "$HOST_LOG_FILE" ]]; then
        printf 'Log file not found: %s\n' "$HOST_LOG_FILE" >&2
        printf 'Use ./logs.sh follow until the container writes persistent logs.\n' >&2
        exit 1
    fi
}

mode="${1:-follow}"

case "$mode" in
    follow|docker)
        DC="$(compose_cmd)"
        $DC logs --tail "$LINES" -f "$SERVICE"
        ;;
    file)
        require_log_file
        tail -n "$LINES" "$HOST_LOG_FILE"
        ;;
    tail|follow-file)
        require_log_file
        tail -n "$LINES" -F "$HOST_LOG_FILE"
        ;;
    errors)
        require_log_file
        (grep -Ei '\[(WARNING|ERROR|CRITICAL)\]' "$HOST_LOG_FILE" || true) | tail -n "$LINES"
        ;;
    qdrant)
        DC="$(compose_cmd)"
        $DC logs --tail "$LINES" -f qdrant
        ;;
    path)
        printf '%s\n' "$HOST_LOG_FILE"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
