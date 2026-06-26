import os
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Подставляем фиктивные ключи до импорта config (он падает без них)
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("API_KEY", "test-deepseek-key")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
