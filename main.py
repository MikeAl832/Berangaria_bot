import logging
import warnings
import asyncio

# Логирование намеренно настраивается до импорта модулей, создающих клиентов.
# ruff: noqa: E402

try:
    from telegram.warnings import PTBDeprecationWarning
    warnings.filterwarnings("ignore", category=PTBDeprecationWarning)
except ImportError:
    pass

# Suppress fastembed metadata warnings (non-critical)
warnings.filterwarnings("ignore", message="Local file sizes do not match the metadata")

from log_setup import setup_logging
from config import DEBUG, FULL_DEBUG_LOGS, LOG_BACKUP_COUNT, LOG_FILE, LOG_MAX_BYTES, VERBOSE

setup_logging(
    level=logging.INFO,
    log_file=LOG_FILE,
    debug=DEBUG,
    verbose=VERBOSE,
    max_bytes=LOG_MAX_BYTES,
    backup_count=LOG_BACKUP_COUNT,
)

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, MessageReactionHandler, filters

from config import (
    TELEGRAM_TOKEN, RANDOM_REPLY_CHANCE, MAX_CONTEXT_TOKENS,
    MAX_REPLY_TOKENS, VISION_MODE, GEMINI_MODEL, SUMMARY_INTERVAL,
    MEMORY_FLUSH_INTERVAL_SECONDS, SUMMARY_HOURS, TIMEZONE_NAME,
)
import state
import memory_store
from handlers import (
    start, clear, stats, random_chance, summarize_command,
    handle_message, handle_media, handle_video, handle_sticker, handle_voice,
    handle_edited_message, handle_chat_event, handle_message_reaction, error_handler
)
from utils import now_local, next_summary_run

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


async def periodic_summarization():
    """Суммаризирует активные чаты в заданные часы (по умолчанию 05:00 и 14:00 МСК)."""
    from llm_client import summarize_history

    while True:
        now = now_local()
        target = next_summary_run(now)
        wait_seconds = max(1.0, (target - now).total_seconds())
        hours_label = ", ".join(f"{h:02d}:00" for h in SUMMARY_HOURS)
        logger.info(
            f"⏰ [cyan]Следующая суммаризация в {target.strftime('%H:%M %d.%m.%Y')} "
            f"({TIMEZONE_NAME})[/] (через {wait_seconds/3600:.1f}ч; слоты: {hours_label})"
        )

        try:
            await asyncio.sleep(wait_seconds)
        except asyncio.CancelledError:
            raise

        try:
            summarized_count = 0
            total_chats = len(state.histories)

            logger.info(f"📝 [yellow]Запуск суммаризации для {total_chats} активных чатов...[/]")

            for key in list(state.histories.keys()):
                async with state.get_turn_lock(key):
                    history = state.histories.get(key, [])

                    # Суммаризируем только если история достаточно длинная
                    if len(history) >= SUMMARY_INTERVAL:
                        old_len = len(history)
                        new_history = await summarize_history(history)

                        if new_history is not history and len(new_history) < old_len:
                            async with state.get_history_lock(key):
                                state.histories[key] = new_history
                                state.save_history(key)
                            summarized_count += 1
                            logger.info(f"  ✅ {key}: {old_len} → {len(new_history)} сообщений")

            if summarized_count > 0:
                logger.info(f"📝 [green]Суммаризировано {summarized_count} из {total_chats} чатов[/]")
            else:
                logger.info("📝 [dim]Нет чатов для суммаризации[/]")

            # Чистим данные чатов, неактивных больше 72 часов (предотвращает рост словарей в памяти)
            removed = state.cleanup_old_chats(max_age_hours=72)
            if removed > 0:
                logger.info(f"🧹 [green]Очищено {removed} неактивных чатов[/]")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"❌ [red]Ошибка при суммаризации чатов:[/] {e}", exc_info=True)


async def sync_stickers_on_start():
    """
    При старте синхронизирует стикеры в Qdrant (в потоке, чтобы не блокировать loop).
      - если версия формата эмбеддинга в коде новее записанной → ОДИН РАЗ переэмбеддит всё
        (миграция без стирания коллекции), затем пишет новую версию;
      - иначе просто доливает недостающие. Если новых нет — почти бесплатно.
    Rate limit переживается внутри (retry на 429), рестарты не нужны.
    """
    import os
    from config import (
        STICKER_ENABLED, STICKER_AUTO_SYNC, STICKER_SYNC_FILE,
        STICKER_SYNC_MAX_PER_START, STICKER_INDEX_VERSION,
    )
    if not (STICKER_ENABLED and STICKER_AUTO_SYNC):
        return
    if not os.path.exists(STICKER_SYNC_FILE):
        logger.warning(f"🎨 [yellow]Файл стикеров '{STICKER_SYNC_FILE}' не найден — синк пропущен[/]")
        return
    try:
        from sticker_store import sync_from_file, get_applied_version, set_applied_version
        applied = get_applied_version()
        if applied < STICKER_INDEX_VERSION:
            # Миграция формата: переэмбеддить ВСЕ (без лимита, retry вытянет через rate limit).
            logger.info(
                f"🎨 [cyan]Миграция индекса стикеров: формат v{applied} → v{STICKER_INDEX_VERSION}, "
                f"переэмбеддинг всех стикеров (один раз)...[/]"
            )
            res = await asyncio.to_thread(
                lambda: sync_from_file(STICKER_SYNC_FILE, limit=None, force_all=True)
            )
            set_applied_version(STICKER_INDEX_VERSION)
            logger.info(
                f"🎨 [green]Миграция завершена: переэмбеддено {res['added']}, "
                f"всего в коллекции {res['total']}. Формат v{STICKER_INDEX_VERSION} записан.[/]"
            )
        else:
            limit = STICKER_SYNC_MAX_PER_START or None
            logger.info(f"🎨 [cyan]Синхронизация стикеров из '{STICKER_SYNC_FILE}'...[/]")
            res = await asyncio.to_thread(lambda: sync_from_file(STICKER_SYNC_FILE, limit=limit))
            if res["added"]:
                logger.info(f"🎨 [green]Стикеры: добавлено {res['added']}, всего в коллекции {res['total']}[/]")
            else:
                logger.info(f"🎨 [dim]Стикеры: новых нет (в коллекции {res['already']})[/]")
    except Exception as e:
        # Версию НЕ пишем → на следующем старте попробует снова (миграция идемпотентна).
        logger.error(f"🎨 [red]Синк/миграция стикеров при старте не удались:[/] {e}", exc_info=True)


async def periodic_memory_flush():
    """Периодически сохраняет короткие накопленные реплики в Mem0."""
    if MEMORY_FLUSH_INTERVAL_SECONDS <= 0:
        logger.info("🧠 [dim]Периодический flush памяти выключен[/]")
        return

    from llm_client import flush_pending_memory

    while True:
        await asyncio.sleep(MEMORY_FLUSH_INTERVAL_SECONDS)
        try:
            if memory_store.memory is None:
                await asyncio.to_thread(
                    memory_store.initialize_memory,
                    attempts=3,
                    delay_seconds=2.0,
                )
            if memory_store.memory is None:
                continue
            flushed = await flush_pending_memory()
            if flushed:
                logger.info(f"🧠 [green]Периодический flush памяти:[/] {flushed} реплик")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"❌ [red]Ошибка периодического flush памяти:[/] {e}", exc_info=True)


def main():
    logger.info("🤖 [cyan]Бот запускается...[/]")

    # Инициализируем БД, runtime-настройки и сохранённые истории диалогов
    state.init_db()
    # Compose запускает контейнеры по порядку, но Qdrant может ещё не принимать
    # соединения. Повторяем инициализацию, не оставляя память выключенной навсегда.
    memory_store.initialize_memory(attempts=10, delay_seconds=2.0)
    runtime_settings = state.load_runtime_settings(default_random_reply_chance=RANDOM_REPLY_CHANCE)
    loaded_chats = state.load_all_histories()
    logger.info(f"⚙️ [green]Runtime-настройки загружены:[/] random_reply_chance=[yellow]{runtime_settings.random_reply_chance}%[/]")
    logger.info(f"💾 [green]Загружено историй из БД:[/] [yellow]{loaded_chats}[/] чатов")

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("random", random_chance))
    app.add_handler(CommandHandler("summarize", summarize_command))

    # Правки сообщений: ловим раньше основных хендлеров, чтобы обновить текст в буфере,
    # пока сообщение ещё не ушло в DeepSeek (фильтр матчит только edited_message)
    app.add_handler(MessageHandler(filters.UpdateType.EDITED_MESSAGE, handle_edited_message))

    # Текстовые сообщения
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Медиа
    app.add_handler(MessageHandler(filters.PHOTO, handle_media))
    app.add_handler(MessageHandler(
        filters.VIDEO | filters.VIDEO_NOTE | filters.ANIMATION,
        handle_video
    ))
    app.add_handler(MessageHandler(filters.Sticker.ALL, handle_sticker))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))

    # Служебные события группы: смена названия / фото / удаление фото
    app.add_handler(MessageHandler(
        filters.StatusUpdate.NEW_CHAT_TITLE | filters.StatusUpdate.NEW_CHAT_PHOTO | filters.StatusUpdate.DELETE_CHAT_PHOTO,
        handle_chat_event
    ))

    # Реакции на сообщения бота (пассивно фиксируем, чтобы он о них знал).
    # Требует allowed_updates с message_reaction (ниже) и админства бота в группах.
    app.add_handler(MessageReactionHandler(handle_message_reaction))

    app.add_error_handler(error_handler)

    logger.info(f"🎲 Шанс случайного ответа: [yellow]{state.random_reply_chance}%[/]")
    logger.info(f"📝 Максимальный контекст: [yellow]{MAX_CONTEXT_TOKENS}[/] токенов")
    logger.info(f"💬 Максимум токенов в ответе: [yellow]{MAX_REPLY_TOKENS}[/]")
    logger.info(f"👁️ Vision mode: [yellow]{VISION_MODE}[/]")
    logger.info(f"🧾 Full debug logs: [yellow]{FULL_DEBUG_LOGS}[/]")
    if VISION_MODE:
        logger.info(f"🖼️ Vision provider: [cyan]Gemini[/] ([magenta]{GEMINI_MODEL}[/])")
    logger.info("🔧 Команды: /start, /clear, /stats, /random X, /summarize")
    hours_label = ", ".join(f"{h:02d}:00" for h in SUMMARY_HOURS)
    logger.info(f"🕒 Часовой пояс: [yellow]{TIMEZONE_NAME}[/]")
    logger.info(f"📝 Автосуммаризация: [yellow]{hours_label}[/] ({TIMEZONE_NAME})")
    logger.info("✅ [bright_green]Бот запущен![/]")
    
    # Запускаем фоновые задачи суммаризации и синхронизации стикеров
    loop = asyncio.get_event_loop()
    summarization_task = loop.create_task(periodic_summarization())
    sticker_sync_task = loop.create_task(sync_stickers_on_start())
    memory_flush_task = loop.create_task(periodic_memory_flush())
    
    try:
        # allowed_updates=ALL_TYPES — иначе Telegram НЕ присылает message_reaction.
        # Лишние типы без хендлеров просто игнорируются.
        app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES, close_loop=False)
    except KeyboardInterrupt:
        logger.info("🛑 [yellow]Получен сигнал остановки...[/]")
    finally:
        # Graceful shutdown
        background_tasks = [summarization_task, sticker_sync_task, memory_flush_task]
        for task in background_tasks:
            task.cancel()
        try:
            if not loop.is_closed():
                loop.run_until_complete(asyncio.gather(*background_tasks, return_exceptions=True))
        except RuntimeError as e:
            logger.debug(f"Не удалось дождаться фоновых задач при остановке: {e}")

        # Финальный flush историй на диск (страховка поверх write-through)
        try:
            for k in list(state.histories.keys()):
                state.save_history(k)
            logger.info("💾 [green]Истории сохранены в БД[/]")
        except Exception as e:
            logger.error(f"❌ Ошибка финального сохранения историй: {e}")
        # Флаш остатков буфера долговременной памяти
        try:
            from llm_client import (
                flush_pending_memory,
                flush_pending_memory_blocking,
                wait_for_memory_flush_tasks,
            )
            if not loop.is_closed():
                loop.run_until_complete(wait_for_memory_flush_tasks())
                flushed = loop.run_until_complete(flush_pending_memory())
                logger.info(f"🧠 [green]Остатки памяти сохранены[/] ({flushed} реплик)")
            else:
                flush_pending_memory_blocking()
                logger.info("🧠 [green]Остатки памяти сохранены[/]")
        except Exception as e:
            logger.error(f"❌ Ошибка финального сохранения памяти: {e}")
        logger.info("👋 [green]Бот остановлен[/]")

if __name__ == '__main__':
    main()
