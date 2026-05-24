#!/usr/bin/env bash
# Запуск Berangaria_bot под Linux
set -euo pipefail

# Переходим в директорию скрипта, чтобы относительные пути работали
cd "$(dirname "$(readlink -f "$0")")"

VENV_DIR="venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

log() {
    printf '\033[1;32m[start.sh]\033[0m %s\n' "$*"
}

err() {
    printf '\033[1;31m[start.sh]\033[0m %s\n' "$*" >&2
}

# 1. Проверяем наличие .env
if [[ ! -f ".env" ]]; then
    err "Файл .env не найден. Создай его и добавь TELEGRAM_BOT_TOKEN и API_KEY."
    exit 1
fi

# 2. Поднимаем Qdrant через docker compose, если он ещё не запущен
if command -v docker >/dev/null 2>&1; then
    if docker compose version >/dev/null 2>&1; then
        DC="docker compose"
    elif command -v docker-compose >/dev/null 2>&1; then
        DC="docker-compose"
    else
        err "docker compose не найден. Установи Docker Compose plugin или docker-compose."
        exit 1
    fi

    log "Запускаю Qdrant ($DC up -d)..."
    $DC up -d qdrant

    # Ждём, пока Qdrant начнёт отвечать на 6333
    log "Ожидаю готовность Qdrant на 127.0.0.1:6333..."
    for i in {1..30}; do
        if curl -fsS http://127.0.0.1:6333/readyz >/dev/null 2>&1 \
            || curl -fsS http://127.0.0.1:6333/ >/dev/null 2>&1; then
            log "Qdrant готов."
            break
        fi
        sleep 1
        if [[ $i -eq 30 ]]; then
            err "Qdrant не поднялся за 30 секунд. Смотри: $DC logs qdrant"
            exit 1
        fi
    done
else
    err "Docker не найден. Установи Docker, либо подними Qdrant вручную на 127.0.0.1:6333."
    exit 1
fi

# 3. Создаём venv при необходимости
if [[ ! -d "$VENV_DIR" ]]; then
    log "Создаю виртуальное окружение в $VENV_DIR..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# 4. Активируем venv
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# 5. Ставим зависимости, если есть requirements.txt и они ещё не установлены
if [[ -f "requirements.txt" ]]; then
    STAMP="$VENV_DIR/.requirements.sha256"
    CURRENT_HASH="$(sha256sum requirements.txt | awk '{print $1}')"
    if [[ ! -f "$STAMP" || "$(cat "$STAMP" 2>/dev/null)" != "$CURRENT_HASH" ]]; then
        log "Устанавливаю зависимости из requirements.txt..."
        pip install --upgrade pip
        pip install -r requirements.txt
        echo "$CURRENT_HASH" > "$STAMP"
    else
        log "Зависимости уже актуальны."
    fi
fi

# 6. Запускаем бота
log "Запускаю бота..."
exec python main.py
