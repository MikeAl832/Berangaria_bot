import logging
import time
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

from berangaria.core.logging_setup import setup_logging
from berangaria.config import DEBUG, FULL_DEBUG_LOGS, LOG_BACKUP_COUNT, LOG_FILE, LOG_MAX_BYTES, VERBOSE

setup_logging(
    level=logging.INFO,
    log_file=LOG_FILE,
    debug=DEBUG,
    verbose=VERBOSE,
    max_bytes=LOG_MAX_BYTES,
    backup_count=LOG_BACKUP_COUNT,
)

from telegram import Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    MessageReactionHandler,
    filters,
)

from berangaria.config import (
    TELEGRAM_TOKEN, RANDOM_REPLY_CHANCE, MAX_CONTEXT_TOKENS,
    MAX_REPLY_TOKENS, VISION_MODE, GEMINI_MODEL,
    MEMORY_FLUSH_INTERVAL_SECONDS, MEMORY_WAITING_MAX_AGE_SECONDS,
    MEMORY_SOURCE_RETENTION_SECONDS,
    SUMMARY_HOURS, SUMMARY_INTERVAL, SUMMARY_MIN_EXTRA, SUMMARY_QUIET_SECONDS,
    TIMEZONE_NAME,
    STREAMING_ENABLED, MODEL, CHAT_API_URL,
)
from berangaria.core import state
from berangaria.core import alerts
from berangaria.memory import store as memory_store
from berangaria.chat.handlers import (
    start, clear, stats, top, dashboard, dashboard_callback,
    random_chance, summarize_command,
    handle_message, handle_media, handle_video, handle_sticker, handle_voice,
    handle_edited_message, handle_chat_event, handle_message_reaction, error_handler
)
from berangaria.core.utils import now_local, next_summary_run
from berangaria.user_bridge import start_user_bridge, stop_user_bridge
from berangaria.media.telegram_download import stop_media_downloader

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("telethon").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


async def _telegram_post_init(application: Application) -> None:
    """Emit readiness only after PTB has initialized the Telegram bot.

    ``Application.initialize`` performs the first authenticated Telegram API
    request.  Keeping the deploy marker in ``main`` made an invalid token or an
    unreachable Bot API look like a successful start because ``run_polling``
    had not begun yet.

    The user bridge starts here so ``ExtBot.id`` is available for self-skip.
    ``start_user_bridge`` only spawns the supervisor task; Telethon I/O must
    not block polling. Failures stay fail-open.
    """
    logger.info("✅ [bright_green]Бот запущен![/]")
    try:
        await start_user_bridge(application)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("👀 [red]User bridge: ошибка запуска (бот продолжает работу)[/]")


async def _summarize_scheduled_keys(keys: list[str]) -> tuple[int, list[str]]:
    """Compress chats that pass the scheduled gates. Returns (done, postponed)."""
    from berangaria.chat.llm_client import summarize_history
    from berangaria.chat.summarization import scheduled_summary_status

    summarized_count = 0
    postponed: list[str] = []
    for key in keys:
        async with state.get_turn_lock(key):
            history = state.histories.get(key, [])
            status = scheduled_summary_status(
                len(history),
                last_activity=state.last_activity.get(key),
                now=time.time(),
                interval=SUMMARY_INTERVAL,
                min_extra=SUMMARY_MIN_EXTRA,
                quiet_seconds=SUMMARY_QUIET_SECONDS,
            )
            if status == "too_short":
                continue
            if status == "recent":
                postponed.append(key)
                continue
            old_len = len(history)
            new_history = await summarize_history(history, key=key)
            if new_history is not history and len(new_history) < old_len:
                async with state.get_history_lock(key):
                    state.histories[key] = new_history
                    state.save_history(key)
                summarized_count += 1
                logger.info(f"  ✅ {key}: {old_len} → {len(new_history)} сообщений")
    return summarized_count, postponed


async def run_scheduled_summarization_slot(*, sleep=asyncio.sleep) -> int:
    """One opportunity slot: skip short chats, postpone recently active once."""
    total_chats = len(state.histories)
    logger.info(f"📝 [yellow]Запуск суммаризации для {total_chats} активных чатов...[/]")
    summarized_count, postponed = await _summarize_scheduled_keys(
        list(state.histories.keys())
    )
    if postponed:
        logger.info(
            "📝 [dim]Откладываю %s чат(ов) на %.0fс — недавно писали[/]",
            len(postponed),
            SUMMARY_QUIET_SECONDS,
        )
        await sleep(SUMMARY_QUIET_SECONDS)
        extra, still_active = await _summarize_scheduled_keys(postponed)
        summarized_count += extra
        for key in still_active:
            logger.info(
                "📝 [dim]Пропуск %s до следующего слота — всё ещё пишут[/]",
                key,
            )
    if summarized_count > 0:
        logger.info(
            f"📝 [green]Суммаризировано {summarized_count} из {total_chats} чатов[/]"
        )
    else:
        logger.info("📝 [dim]Нет чатов для суммаризации[/]")
    return summarized_count


async def periodic_summarization(bot=None):
    """Суммаризирует активные чаты в заданные часы (по умолчанию 05:00 и 14:00 МСК)."""
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
            await run_scheduled_summarization_slot()

            # Чистим данные чатов, неактивных больше 72 часов (предотвращает рост словарей в памяти)
            removed = state.cleanup_old_chats(max_age_hours=72)
            if removed > 0:
                logger.info(f"🧹 [green]Очищено {removed} неактивных чатов[/]")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"❌ [red]Ошибка при суммаризации чатов:[/] {e}", exc_info=True)
            await alerts.notify_owner(
                bot,
                category="Summarization job",
                message="периодическая суммаризация завершилась ошибкой",
                error=e,
            )


async def sync_stickers_on_start():
    """
    Sync stickers into Qdrant on startup (in a worker thread).
      - if the embed-format version in config is newer than the marker on disk →
        one-shot full rewrite (recreate collection + re-embed catalogue), then
        write the new version marker;
      - else only upsert missing file_ids (cheap when the catalogue is unchanged).
    Rate limits are retried inside the embed helper; version is written only on success.
    """
    import os
    from berangaria.config import (
        STICKER_ENABLED, STICKER_AUTO_SYNC, STICKER_SYNC_FILE,
        STICKER_SYNC_MAX_PER_START, STICKER_INDEX_VERSION,
    )
    if not (STICKER_ENABLED and STICKER_AUTO_SYNC):
        return
    if not os.path.exists(STICKER_SYNC_FILE):
        logger.warning(f"🎨 [yellow]Файл стикеров '{STICKER_SYNC_FILE}' не найден — синк пропущен[/]")
        return
    try:
        from berangaria.stickers.store import sync_from_file, get_applied_version, set_applied_version
        applied = get_applied_version()
        if applied < STICKER_INDEX_VERSION:
            # Format migration: wipe orphans from older packs, re-embed the whole
            # catalogue. Version marker is written only after success.
            logger.info(
                f"🎨 [cyan]Миграция индекса стикеров: формат v{applied} → v{STICKER_INDEX_VERSION}, "
                f"полная перезапись коллекции (один раз)...[/]"
            )
            res = await asyncio.to_thread(
                lambda: sync_from_file(
                    STICKER_SYNC_FILE,
                    limit=None,
                    force_all=True,
                    recreate=True,
                )
            )
            set_applied_version(STICKER_INDEX_VERSION)
            logger.info(
                f"🎨 [green]Миграция завершена: залито {res['added']}, "
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


async def periodic_memory_flush(bot=None):
    """Периодически подбирает durable memory-очередь после фоновых попыток."""
    if MEMORY_FLUSH_INTERVAL_SECONDS <= 0:
        logger.info("🧠 [dim]Периодический retry памяти выключен[/]")
        return

    from berangaria.memory.pipeline import process_pending_memory

    while True:
        await asyncio.sleep(MEMORY_FLUSH_INTERVAL_SECONDS)
        try:
            # Подстраховка к компенсации в handlers.wait_and_process: ход не
            # живёт дольше debounce + retry-бюджета, поэтому зависшее заметно
            # дольше 'waiting' — след оборванного процесса. Такой источник
            # блокирует очередь своей области памяти, пока его не похоронить.
            reaped = state.reap_stale_waiting_sources(MEMORY_WAITING_MAX_AGE_SECONDS)
            if reaped:
                logger.warning(
                    "🧠 [yellow]Память: похоронено зависших источников: %s[/]", reaped
                )
            pruned = state.prune_memory_sources(MEMORY_SOURCE_RETENTION_SECONDS)
            if pruned:
                logger.info("🧠 [dim]Память: удалено старых строк очереди: %s[/]", pruned)
            if memory_store.memory is None:
                await asyncio.to_thread(
                    memory_store.initialize_memory,
                    attempts=3,
                    delay_seconds=2.0,
                )
            if memory_store.memory is None:
                continue
            report = await process_pending_memory()
            if report.processed:
                logger.info(
                    "🧠 [green]Память: обработано источников %s, одобрено %s, "
                    "отброшено %s, retry %s, dead-letter %s[/]",
                    report.processed,
                    report.approved,
                    report.discarded,
                    report.retried,
                    report.dead_lettered,
                )
            if report.dead_lettered:
                await alerts.notify_owner(
                    bot,
                    category="Memory dead-letter",
                    message=(
                        f"источников окончательно отклонено: {report.dead_lettered}; "
                        f"обработано в проходе: {report.processed}"
                    ),
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"❌ [red]Ошибка периодического flush памяти:[/] {e}", exc_info=True)
            await alerts.notify_owner(
                bot,
                category="Memory job",
                message="периодический flush памяти завершился ошибкой",
                error=e,
            )


def build_telegram_application() -> Application:
    """Use Telegram's cloud Bot API for updates and outgoing messages."""
    # Leave enough time for outgoing media uploads and slow Telegram responses.
    builder = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .connect_timeout(10.0)
        .read_timeout(60.0)
        .write_timeout(60.0)
        .pool_timeout(5.0)
        .media_write_timeout(120.0)
        .post_init(_telegram_post_init)
        .post_shutdown(stop_media_downloader)
    )
    return builder.build()


def register_handlers(app: Application) -> None:
    """Регистрирует хендлеры на приложении.

    Вынесено из main() отдельной функцией, чтобы инвариант «пассивные хендлеры
    не держат слот обновлений» можно было проверить тестом.
    """
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("top", top))
    app.add_handler(CommandHandler("dashboard", dashboard))
    app.add_handler(CommandHandler("random", random_chance))
    app.add_handler(CommandHandler("summarize", summarize_command))
    app.add_handler(CallbackQueryHandler(dashboard_callback, pattern=r"^dashboard:"))

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

    # Служебные события группы: смена названия / фото / удаление фото.
    # block=False — см. комментарий к реакциям ниже.
    app.add_handler(MessageHandler(
        filters.StatusUpdate.NEW_CHAT_TITLE | filters.StatusUpdate.NEW_CHAT_PHOTO | filters.StatusUpdate.DELETE_CHAT_PHOTO,
        handle_chat_event,
        block=False,
    ))

    # Реакции на сообщения бота (пассивно фиксируем, чтобы он о них знал).
    # Требует allowed_updates с message_reaction (ниже) и админства бота в группах.
    #
    # block=False обязателен: приложение работает с дефолтным
    # SimpleUpdateProcessor(1), то есть апдейты обрабатываются по одному инлайн.
    # Оба хендлера берут turn-lock чата, и пока в этом чате идёт LLM-ход
    # (клиент живёт до 600 с плюс tool-раунды), блокирующий хендлер держал бы
    # единственный слот и морозил весь бот: другие группы, все личные чаты и все
    # команды. Оба хендлера пассивные, порядок их выполнения ни на что не влияет.
    #
    # Глобальный concurrent_updates(True) здесь НЕ подходит: он распараллелит и
    # приём сообщений, а debounce-буфер собирает combined_text в порядке прихода.
    app.add_handler(MessageReactionHandler(handle_message_reaction, block=False))

    app.add_error_handler(error_handler)


def main():
    logger.info("🤖 [cyan]Бот запускается...[/]")

    # Инициализируем БД, runtime-настройки и сохранённые истории диалогов
    state.init_db()
    # Compose запускает контейнеры по порядку, но Qdrant может ещё не принимать
    # соединения. Повторяем инициализацию, не оставляя память выключенной навсегда.
    memory_store.initialize_memory(attempts=10, delay_seconds=2.0)
    runtime_settings = state.load_runtime_settings(default_random_reply_chance=RANDOM_REPLY_CHANCE)
    loaded_chats = state.load_all_histories()
    logger.info(
        "⚙️ [green]Runtime-настройки загружены:[/] "
        f"base_random_reply_chance=[yellow]{runtime_settings.random_reply_chance}%[/]"
    )
    logger.info(f"💾 [green]Загружено историй из БД:[/] [yellow]{loaded_chats}[/] чатов")

    app = build_telegram_application()
    register_handlers(app)

    logger.info(
        f"🎲 Базовый шанс случайного ответа: [yellow]{state.random_reply_chance}%[/]"
    )
    logger.info(f"🧠 Чат-модель: [cyan]{MODEL}[/] ([magenta]{CHAT_API_URL}[/])")
    logger.info(f"📝 Максимальный контекст: [yellow]{MAX_CONTEXT_TOKENS}[/] токенов")
    logger.info(f"💬 Максимум токенов в ответе: [yellow]{MAX_REPLY_TOKENS}[/]")
    logger.info(f"👁️ Vision mode: [yellow]{VISION_MODE}[/]")
    logger.info(f"🧾 Full debug logs: [yellow]{FULL_DEBUG_LOGS}[/]")
    logger.info(f"🌊 Потоковые ответы: [yellow]{STREAMING_ENABLED}[/]")
    if VISION_MODE:
        logger.info(f"🖼️ Vision provider: [cyan]Gemini[/] ([magenta]{GEMINI_MODEL}[/])")
    logger.info(
        "🔧 Команды: /start, /clear, /stats, /top, /dashboard, /random X, /summarize"
    )
    hours_label = ", ".join(f"{h:02d}:00" for h in SUMMARY_HOURS)
    logger.info(f"🕒 Часовой пояс: [yellow]{TIMEZONE_NAME}[/]")
    logger.info(f"📝 Автосуммаризация: [yellow]{hours_label}[/] ({TIMEZONE_NAME})")
    # Запускаем фоновые задачи суммаризации и синхронизации стикеров
    loop = asyncio.get_event_loop()
    summarization_task = loop.create_task(periodic_summarization(app.bot))
    sticker_sync_task = loop.create_task(sync_stickers_on_start())
    memory_flush_task = loop.create_task(periodic_memory_flush(app.bot))
    # User bridge is started from post_init (after ExtBot.initialize). The
    # long-lived supervisor lives inside the user_bridge module and is stopped
    # explicitly below. Missing secrets / Telethon errors never block polling.

    try:
        # allowed_updates=ALL_TYPES — иначе Telegram НЕ присылает message_reaction.
        # Лишние типы без хендлеров просто игнорируются.
        app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES, close_loop=False)
    except KeyboardInterrupt:
        logger.info("🛑 [yellow]Получен сигнал остановки...[/]")
    finally:
        # Graceful shutdown
        background_tasks = [summarization_task, sticker_sync_task, memory_flush_task]
        try:
            if not loop.is_closed():
                loop.run_until_complete(stop_user_bridge())
        except Exception as e:
            logger.debug(f"User bridge stop: {e}")
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
        # Финальная попытка обработать durable-очередь долговременной памяти.
        try:
            from berangaria.memory.pipeline import process_pending_memory, wait_for_memory_worker
            if not loop.is_closed():
                # Проход по очереди делает сетевые вызовы (DeepSeek, Mem0) и без
                # ограничения может не уложиться в stop_grace_period. SIGKILL
                # посреди прохода сжигает попытку источника впустую, поэтому
                # укладываемся сами и с запасом.
                loop.run_until_complete(
                    asyncio.wait_for(wait_for_memory_worker(), timeout=30)
                )
                report = loop.run_until_complete(
                    asyncio.wait_for(process_pending_memory(), timeout=60)
                )
                logger.info(
                    "🧠 [green]Остатки памяти обработаны[/] "
                    f"(источников={report.processed}, одобрено={report.approved})"
                )
            else:
                logger.info("🧠 [green]Очередь памяти сохранена в SQLite[/]")
        except asyncio.TimeoutError:
            logger.warning(
                "🧠 [yellow]Финальный проход памяти не уложился в отведённое время — "
                "очередь осталась в SQLite и будет продолжена после старта[/]"
            )
        except Exception as e:
            logger.error(f"❌ Ошибка финального сохранения памяти: {e}")
        logger.info("👋 [green]Бот остановлен[/]")

if __name__ == '__main__':
    main()
