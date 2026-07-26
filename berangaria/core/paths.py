"""Единая точка разрешения путей проекта.

Модуль ничего не импортирует из проекта, поэтому его может использовать и
`berangaria.config`, и `berangaria.core.state`, не создавая цикла.

Относительные пути в конфиге и переменных окружения считаются от корня
репозитория, а не от текущей рабочей директории: бот запускается как
`python -m berangaria`, из контейнера и из скриптов в `scripts/`, и все три
должны попадать в один и тот же config.yaml, bot_state.db и bot.log.
"""
from pathlib import Path

# berangaria/core/paths.py -> berangaria/core -> berangaria -> корень репозитория
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value: str) -> str:
    """Абсолютный путь: относительное значение считается от корня репозитория."""
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else PROJECT_ROOT / path)
