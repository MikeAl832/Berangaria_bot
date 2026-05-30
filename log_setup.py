import logging
import re
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
        original = record.msg
        try:
            # Подменяем сообщение на версию без цветовых тегов
            record.msg = _strip_color_tags(record.getMessage())
            record.args = None  # getMessage уже подставил аргументы
            return super().format(record)
        finally:
            record.msg = original


def _strip_color_tags(text: str) -> str:
    """Удаляет разметку [color]текст[/], оставляя только содержимое."""
    return _COLOR_PATTERN.sub(lambda m: m.group(2), text)


def setup_logging(level=logging.INFO, log_file="bot.log"):
    """Настраивает цветной вывод в консоль и чистый (без тегов) вывод в файл."""
    root = logging.getLogger()
    root.setLevel(level)

    # Убираем возможные старые хендлеры (на случай повторного вызова)
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    datefmt = "%Y-%m-%d %H:%M:%S"

    # Консоль — с цветом и поддержкой инлайн-тегов
    console = logging.StreamHandler()
    console.setFormatter(ColorFormatter(datefmt=datefmt))
    root.addHandler(console)

    # Файл — чистый текст, теги [color]...[/] вырезаются
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(
        PlainFormatter("%(asctime)s [%(levelname)s] %(message)s", datefmt=datefmt)
    )
    root.addHandler(file_handler)
