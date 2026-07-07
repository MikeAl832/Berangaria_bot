import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=False)

COLOR_MAP = {
    "red": Fore.RED,
    "green": Fore.GREEN,
    "yellow": Fore.YELLOW,
    "blue": Fore.BLUE,
    "cyan": Fore.CYAN,
    "magenta": Fore.MAGENTA,
    "white": Fore.WHITE,
    "bright_red": Fore.RED + Style.BRIGHT,
    "bright_green": Fore.GREEN + Style.BRIGHT,
}

LEVEL_COLORS = {
    logging.DEBUG: Fore.CYAN,
    logging.INFO: Fore.GREEN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: Fore.RED + Style.BRIGHT,
}


DIM = Style.DIM
RESET = Style.RESET_ALL

# Регулярка для поиска [color]текст[/] — используется и для подсветки, и для очистки.
_COLOR_PATTERN = re.compile(r'\[(\w+)\](.*?)\[/\]', re.DOTALL)


class ColorFormatter(logging.Formatter):
    """Форматтер с поддержкой локального выделения цветов."""

    def format(self, record):
        # Базовый цвет уровня
        level_color = LEVEL_COLORS.get(record.levelno, "")
        
        asctime = self.formatTime(record, self.datefmt)
        levelname = f"{level_color}{record.levelname}{RESET}"
        
        message = record.getMessage()
        
        # Применяем локальные цвета внутри сообщения
        message = self._apply_inline_colors(message)
        
        # Основная строка
        line = f"{DIM}{asctime}{RESET} [{levelname}] {message}"
        
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        
        return line

    def _apply_inline_colors(self, text: str) -> str:
        """Заменяет [color]текст[/] на цветной текст."""
        def replace_match(match):
            color_name = match.group(1).lower()
            content = match.group(2)
            
            color_code = COLOR_MAP.get(color_name, "")
            if color_code:
                return f"{color_code}{content}{RESET}"
            return match.group(0)  # если цвет неизвестен — оставляем как есть
        
        return _COLOR_PATTERN.sub(replace_match, text)


class PlainFormatter(logging.Formatter):
    """Форматтер для файла: убирает теги [color]...[/], оставляя чистый текст."""

    def format(self, record):
        original_msg = record.msg
        original_args = record.args
        try:
            # Подменяем сообщение на версию без цветовых тегов
            record.msg = _strip_color_tags(record.getMessage())
            record.args = None  # getMessage уже подставил аргументы
            return super().format(record)
        finally:
            record.msg = original_msg
            record.args = original_args


def _strip_color_tags(text: str) -> str:
    """Удаляет разметку [color]текст[/], оставляя только содержимое."""
    return _COLOR_PATTERN.sub(lambda m: m.group(2), text)


def _ensure_log_parent(log_file: str) -> None:
    parent = Path(log_file).expanduser().parent
    if parent != Path("."):
        parent.mkdir(parents=True, exist_ok=True)


def setup_logging(
    level: int = logging.INFO,
    log_file: str = "bot.log",
    debug: bool = False,
    verbose: bool = False,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """
    Настраивает цветной вывод в консоль и чистый (без тегов) вывод в файл.
    
    ВАРИАНТ 3A:
    - Консоль: INFO (чистый вывод) или DEBUG (если debug=True)
    - Файл: ВСЕГДА DEBUG (полный аудит для разбора)
    
    Args:
        level: Базовый уровень логирования (по умолчанию INFO)
        log_file: Путь к файлу логов
        debug: Если True, устанавливает уровень DEBUG в консоли
        verbose: Если True, включает DEBUG для всех библиотек (HTTP, TLS, H2)
    """
    root = logging.getLogger()
    
    # Root logger всегда на DEBUG, чтобы файл всё ловил
    root.setLevel(logging.DEBUG)

    # Убираем возможные старые хендлеры (на случай повторного вызова)
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    datefmt = "%Y-%m-%d %H:%M:%S"

    # ========================================
    # КОНСОЛЬ — с цветом и поддержкой инлайн-тегов
    # ========================================
    console = logging.StreamHandler()
    console.setFormatter(ColorFormatter(datefmt=datefmt))
    
    # Уровень консоли зависит от debug флага
    if debug or verbose:
        console.setLevel(logging.DEBUG)
    else:
        console.setLevel(logging.INFO)
    
    root.addHandler(console)

    # ========================================
    # ФАЙЛ — ВСЕГДА DEBUG для полного аудита
    # ========================================
    _ensure_log_parent(log_file)
    if max_bytes > 0:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=max(0, backup_count),
            encoding="utf-8",
        )
    else:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(
        PlainFormatter("%(asctime)s [%(levelname)s] %(message)s", datefmt=datefmt)
    )
    file_handler.setLevel(logging.DEBUG)  # Файл всегда ловит всё
    root.addHandler(file_handler)

    # Глушим избыточные библиотечные логи (если verbose=False, то глушим)
    if not verbose:
        # HTTP клиенты
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        
        # OpenAI/DeepSeek клиент
        logging.getLogger("openai").setLevel(logging.WARNING)
        
        # H2 protocol (HTTP/2)
        logging.getLogger("h2").setLevel(logging.WARNING)
        logging.getLogger("hpack").setLevel(logging.WARNING)
        
        # TLS/rustls/rquest (если есть)
        logging.getLogger("rustls").setLevel(logging.WARNING)
        logging.getLogger("rquest").setLevel(logging.WARNING)
        
        # fastembed (уже есть warning filter, но на всякий случай)
        logging.getLogger("fastembed").setLevel(logging.WARNING)
        
        # Telegram bot библиотека (бесконечные getUpdates)
        logging.getLogger("telegram").setLevel(logging.INFO)
        logging.getLogger("telegram.ext").setLevel(logging.INFO)
