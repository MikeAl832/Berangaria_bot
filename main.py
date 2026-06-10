import logging
import warnings
import asyncio

try:
    from telegram.warnings import PTBDeprecationWarning
    warnings.filterwarnings("ignore", category=PTBDeprecationWarning)
except ImportError:
    pass

# Suppress fastembed metadata warnings (non-critical)
warnings.filterwarnings("ignore", message="Local file sizes do not match the metadata")

from log_setup import setup_logging
from config import DEBUG, VERBOSE

setup_logging(level=logging.INFO, log_file='bot.log', debug=DEBUG, verbose=VERBOSE)

from telegram.ext import Application, CommandHandler, MessageHandler, filters

from config import (
    TELEGRAM_TOKEN, RANDOM_REPLY_CHANCE, MAX_CONTEXT_TOKENS,
    MAX_REPLY_TOKENS, VISION_MODE, GEMINI_MODEL, DEBUG, SUMMARY_INTERVAL
)
import state
from handlers import (
    start, clear, stats, random_chance, summarize_command,
    handle_message, handle_media, handle_video, error_handler
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


async def periodic_summarization():
    """Суммаризирует активные чаты раз в сутки в 5:00."""
    from datetime import datetime, timedelta
    from llm_client import summarize_history
    
    while True:
        # Вычисляем время до следующего запуска в 5:00
        now = datetime.now()
        target = now.replace(hour=5, minute=0, second=0, microsecond=0)
        
        # Если сейчас уже после 5:00, берём следующие сутки
        if now >= target:
            target += timedelta(days=1)
        
        wait_seconds = (target - now).total_seconds()
        logger.info(f"⏰ [cyan]Следующая суммаризация в {target.strftime('%H:%M %d.%m.%Y')}[/] (через {wait_seconds/3600:.1f}ч)")
        
        await asyncio.sleep(wait_seconds)
        
        try:
            summarized_count = 0
            total_chats = len(state.histories)
            
            logger.info(f"📝 [yellow]Запуск суммаризации для {total_chats} активных чатов...[/]")
            
            for key in list(state.histories.keys()):
                async with state.get_history_lock(key):
                    history = state.histories.get(key, [])
                    
                    # Суммаризируем только если история достаточно длинная
                    if len(history) >= SUMMARY_INTERVAL:
                        old_len = len(history)
                        new_history = await summarize_history(history)
                        
                        if new_history is not history and len(new_history) < old_len:
                            state.histories[key] = new_history
                            summarized_count += 1
                            logger.info(f"  ✅ {key}: {old_len} → {len(new_history)} сообщений")
            
            if summarized_count > 0:
                logger.info(f"📝 [green]Суммаризировано {summarized_count} из {total_chats} чатов[/]")
            else:
                logger.info(f"📝 [dim]Нет чатов для суммаризации[/]")
                
        except Exception as e:
            logger.error(f"❌ [red]Ошибка при суммаризации чатов:[/] {e}", exc_info=True)

def main():
    logger.info("🤖 [cyan]Бот запускается...[/]")
    
    # Инициализируем изменяемый шанс из config
    state.random_reply_chance = RANDOM_REPLY_CHANCE
    
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
    app.add_handler(MessageHandler(
        filters.VIDEO | filters.VIDEO_NOTE | filters.ANIMATION,
        handle_video
    ))

    app.add_error_handler(error_handler)

    logger.info(f"🎲 Шанс случайного ответа: [yellow]{state.random_reply_chance}%[/]")
    logger.info(f"📝 Максимальный контекст: [yellow]{MAX_CONTEXT_TOKENS}[/] токенов")
    logger.info(f"💬 Максимум токенов в ответе: [yellow]{MAX_REPLY_TOKENS}[/]")
    logger.info(f"👁️ Vision mode: [yellow]{VISION_MODE}[/]")
    if VISION_MODE:
        logger.info(f"🖼️ Vision provider: [cyan]Gemini[/] ([magenta]{GEMINI_MODEL}[/])")
    logger.info("🔧 Команды: /start, /clear, /stats, /random X, /summarize")
    logger.info("📝 Автосуммаризация активных чатов: каждый день в 5:00")
    logger.info("✅ [bright_green]Бот запущен![/]")
    
    # Запускаем фоновую задачу суммаризации
    loop = asyncio.get_event_loop()
    summarization_task = loop.create_task(periodic_summarization())
    
    try:
        app.run_polling(drop_pending_updates=True)
    except KeyboardInterrupt:
        logger.info("🛑 [yellow]Получен сигнал остановки...[/]")
    finally:
        # Graceful shutdown
        summarization_task.cancel()
        logger.info("👋 [green]Бот остановлен[/]")

if __name__ == '__main__':
    main()
