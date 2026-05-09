import logging
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from config import (
    TELEGRAM_TOKEN, RANDOM_REPLY_CHANCE, MAX_CONTEXT_TOKENS,
    MAX_REPLY_TOKENS, VISION_MODE
)
import config
from handlers import (
    start, clear, stats, random_chance, summarize_command,
    handle_message, handle_media, error_handler
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def main():
    logger.info("🤖 Бот запускается...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("random", random_chance))
    app.add_handler(CommandHandler("summarize", summarize_command))

    # Текстовые сообщения
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Медиа
    app.add_handler(MessageHandler(filters.PHOTO, handle_media))

    app.add_error_handler(error_handler)

    logger.info(f"🎲 Шанс случайного ответа: {config.RANDOM_REPLY_CHANCE}%")
    logger.info(f"📝 Максимальный контекст: {MAX_CONTEXT_TOKENS} токенов")
    logger.info(f"💬 Максимум токенов в ответе: {MAX_REPLY_TOKENS}")
    logger.info(f"👁️ Vision mode: {VISION_MODE}")
    logger.info("🔧 Команды: /start, /clear, /stats, /random X, /summarize")
    logger.info("✅ Бот запущен!")
    
    app.run_polling(drop_pending_updates=True)

if __name__ == '__main__':
    main()
