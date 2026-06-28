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
    handle_message, handle_media, handle_video, handle_sticker, handle_voice,
    handle_edited_message, handle_chat_event, error_handler
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
                            state.save_history(key)
                            summarized_count += 1
                            logger.info(f"  ✅ {key}: {old_len} → {len(new_history)} сообщений")
            
            if summarized_count > 0:
                logger.info(f"📝 [green]Суммаризировано {summarized_count} из {total_chats} чатов[/]")
            else:
                logger.info(f"📝 [dim]Нет чатов для суммаризации[/]")

            # Чистим данные чатов, неактивных больше 72 часов (предотвращает рост словарей в памяти)
            removed = state.cleanup_old_chats(max_age_hours=72)
            if removed > 0:
                logger.info(f"🧹 [green]Очищено {removed} неактивных чатов[/]")

        except Exception as e:
            logger.error(f"❌ [red]Ошибка при суммаризации чатов:[/] {e}", exc_info=True)


async def poll_tiktok_queue(bot):
    """Фоновая задача опроса очереди новых видео/слайд-шоу из TikTok."""
    import os
    import json
    import glob
    from handlers import get_history_key, get_history_lock, touch_activity
    from vision_provider import describe_video, describe_image_bytes
    from llm_client import generate_and_send_tiktok_review
    from state import histories

    QUEUE_DIR = "/tmp/tiktok_queue"
    os.makedirs(QUEUE_DIR, exist_ok=True)
    try:
        os.chmod(QUEUE_DIR, 0o777)
    except Exception:
        pass

    logger.info("⏱️ [cyan]Запущен опрос очереди TikTok для Berangaria...[/]")

    while True:
        await asyncio.sleep(5)
        try:
            json_files = glob.glob(os.path.join(QUEUE_DIR, "*.json"))
            for json_file in json_files:
                logger.info(f"📦 Найдена запись в очереди TikTok: {json_file}")
                
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception as e:
                    logger.error(f"❌ Ошибка чтения JSON {json_file}: {e}")
                    continue

                video_id = data.get("video_id")
                media_type = data.get("type")
                media_files = data.get("media_files", [])
                chat_id = data.get("chat_id")
                message_id = data.get("message_id")
                message_thread_id = data.get("message_thread_id")
                username = data.get("username")

                # Убедимся, что все медиафайлы существуют
                media_exists = all(os.path.exists(f) for f in media_files)
                if not media_exists:
                    logger.warning(f"⚠️ Медиафайлы для {video_id} еще не записаны на диск. Пропускаем.")
                    continue

                logger.info(f"👁️ Описание медиа {video_id} ({media_type}) через Gemini Vision...")
                description = ""

                if media_type == "video" and media_files:
                    video_path = media_files[0]
                    try:
                        description = await describe_video(
                            video_path=video_path,
                            mime="video/mp4",
                            caption="",
                            duration=0.0
                        )
                    except Exception as e:
                        logger.error(f"❌ Ошибка describe_video: {e}")
                        description = "(не удалось распознать видео)"
                elif media_type == "slideshow" and media_files:
                    descriptions = []
                    for idx, img_path in enumerate(media_files[:3]):
                        try:
                            with open(img_path, 'rb') as f_img:
                                img_bytes = f_img.read()
                            desc = await describe_image_bytes(img_bytes, "image/jpeg", caption="")
                            descriptions.append(f"Кадр {idx+1}: {desc}")
                        except Exception as e:
                            logger.error(f"❌ Ошибка describe_image {img_path}: {e}")
                    description = "\n\n".join(descriptions) if descriptions else "(не удалось распознать изображения)"

                if not description:
                    description = "(медиафайл пустой или не распознан)"

                # Формируем контекст истории
                key = get_history_key(chat_id, False, 0)
                user_msg = f"[User: {username or 'TikTok'}] [New Liked TikTok Media]"
                if media_type == "video":
                    user_msg += f"\n[Video description: {description}]"
                else:
                    user_msg += f"\n[Image description: {description}]"

                # Загружаем в историю
                async with get_history_lock(key):
                    if key not in histories:
                        histories[key] = []
                    history = histories[key]
                    history.append({"role": "user", "content": user_msg})
                    histories[key] = history
                    touch_activity(key)
                    state.save_history(key)

                # Генерируем рецензию и отправляем в группу
                logger.info(f"🧠 Генерация ответа DeepSeek для {video_id}...")
                await generate_and_send_tiktok_review(bot, chat_id, message_id, key, history, message_thread_id)

                # Удаляем обработанные файлы
                try:
                    for f in media_files:
                        if os.path.exists(f):
                            os.remove(f)
                    if os.path.exists(json_file):
                        os.remove(json_file)
                    logger.info(f"🗑️ Запись очереди {video_id} обработана и очищена.")
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка очистки файлов очереди {video_id}: {e}")

        except Exception as e:
            logger.error(f"❌ Ошибка в цикле опроса очереди TikTok: {e}", exc_info=True)


def main():
    logger.info("🤖 [cyan]Бот запускается...[/]")

    # Инициализируем изменяемый шанс из config
    state.random_reply_chance = RANDOM_REPLY_CHANCE

    # Инициализируем БД и подгружаем сохранённые истории диалогов
    state.init_db()
    loaded_chats = state.load_all_histories()
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
    
    # Запускаем фоновые задачи суммаризации и опроса TikTok
    loop = asyncio.get_event_loop()
    summarization_task = loop.create_task(periodic_summarization())
    tiktok_queue_task = loop.create_task(poll_tiktok_queue(app.bot))
    
    try:
        app.run_polling(drop_pending_updates=True)
    except KeyboardInterrupt:
        logger.info("🛑 [yellow]Получен сигнал остановки...[/]")
    finally:
        # Graceful shutdown
        summarization_task.cancel()
        tiktok_queue_task.cancel()
        # Финальный flush историй на диск (страховка поверх write-through)
        try:
            for k in list(state.histories.keys()):
                state.save_history(k)
            logger.info("💾 [green]Истории сохранены в БД[/]")
        except Exception as e:
            logger.error(f"❌ Ошибка финального сохранения историй: {e}")
        # Флаш остатков буфера долговременной памяти
        try:
            from llm_client import flush_pending_memory_blocking
            flush_pending_memory_blocking()
            logger.info("🧠 [green]Остатки памяти сохранены[/]")
        except Exception as e:
            logger.error(f"❌ Ошибка финального сохранения памяти: {e}")
        logger.info("👋 [green]Бот остановлен[/]")

if __name__ == '__main__':
    main()
