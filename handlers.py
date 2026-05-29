import logging
import asyncio
from datetime import datetime
from telegram import Update
from telegram.ext import ContextTypes

from config import (
    ADMIN_MODE, RANDOM_REPLY_CHANCE, SUMMARY_INTERVAL, MAX_CONTEXT_TOKENS,
    ALLOWED_USERS, ALLOWED_GROUPS, VISION_MODE, VIDEO_MAX_DURATION_SEC, VIDEO_MAX_FRAMES,
    VISION_PROVIDER
)
from state import histories, get_history_key, message_buffer, chat_tokens, api_call_count
import config
from llm_client import summarize_history, send_llm_request
from vision_provider import describe_image_bytes, describe_video, is_native_video_supported
from utils import (
    escape_user_text, is_bot_mentioned, should_reply_randomly,
    download_media_as_base64, extract_video_frames, download_video_to_file
)

logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    user_name = update.effective_user.first_name
    is_group = update.effective_chat.type in ['group', 'supergroup']

    if is_group:
        await update.message.reply_text(
            f"👋 Привет, {user_name}!\n\n"
            f"📌 Как ко мне обращаться:\n"
            f"• Ответить на моё сообщение (reply)\n"
            f"• Написать @{context.bot.username}\n"
            f"• Назвать меня {context.bot.first_name}\n\n"
            f"🎲 Шанс случайного ответа: {config.RANDOM_REPLY_CHANCE}%\n"
            f"Команды:\n"
            f"/clear — очистить историю\n"
            f"/stats — статистика\n"
            f"/summarize — сжатие истории\n"
            f"/random X — изменить шанс случайных ответов"
        )
    else:
        await update.message.reply_text(
            f"👋 Привет, {user_name}!\n\n"
            f"Команды:\n"
            f"/clear — очистить историю\n"
            f"/stats — статистика\n"
            f"/summarize — сжатие истории"
        )

async def random_chance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    is_group = update.effective_chat.type in ['group', 'supergroup']
    if not is_group:
        await update.message.reply_text("Эта команда работает только в группах!")
        return

    if not context.args:
        await update.message.reply_text(f"Текущий шанс: {config.RANDOM_REPLY_CHANCE}%\nИспользуйте: /random 0-100")
        return

    try:
        new_chance = int(context.args[0])
        if not 0 <= new_chance <= 100:
            await update.message.reply_text("Шанс должен быть от 0 до 100!")
            return

        chat_id = update.effective_chat.id
        user_id = update.effective_user.id

        if ADMIN_MODE:
            try:
                chat_member = await context.bot.get_chat_member(chat_id, user_id)
                is_admin = chat_member.status in ['administrator', 'creator']
            except:
                is_admin = False

            if not is_admin:
                await update.message.reply_text("❌ Только администраторы могут менять шанс!")
                return

        config.RANDOM_REPLY_CHANCE = new_chance
        await update.message.reply_text(f"✅ Шанс изменён на {config.RANDOM_REPLY_CHANCE}%")

    except ValueError:
        await update.message.reply_text("Укажите число от 0 до 100")

async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    is_group = update.effective_chat.type in ['group', 'supergroup']
    key = get_history_key(chat_id, not is_group, user_id)

    if is_group and ADMIN_MODE:
        try:
            chat_member = await context.bot.get_chat_member(chat_id, user_id)
            is_admin = chat_member.status in ['administrator', 'creator']
        except:
            is_admin = False

        if not is_admin:
            await update.message.reply_text("❌ Только администраторы могут очищать историю!")
            return

    if key in histories:
        del histories[key]
        await update.message.reply_text("🧹 История очищена!")
    else:
        await update.message.reply_text("История и так пуста!")

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    is_group = update.effective_chat.type in ['group', 'supergroup']
    key = get_history_key(chat_id, not is_group, user_id)

    history = histories.get(key, [])
    msg_count = len(history)
    token_count = chat_tokens.get(key, 0)
    chat_type = "группы" if is_group else "личного чата"

    await update.message.reply_text(
        f"📊 Статистика {chat_type}:\n"
        f"Сообщений в истории: {msg_count}\n"
        f"Токенов (с учетом системного промпта): {token_count}/{MAX_CONTEXT_TOKENS}\n"
        f"Вызовов API: {api_call_count.get(key, 0)}\n"
        f"🎲 Шанс случайного ответа: {config.RANDOM_REPLY_CHANCE}%"
    )

async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    is_group = update.effective_chat.type in ['group', 'supergroup']
    key = get_history_key(chat_id, not is_group, user_id)

    if is_group and ADMIN_MODE:
        try:
            chat_member = await context.bot.get_chat_member(chat_id, user_id)
            is_admin = chat_member.status in ['administrator', 'creator']
        except:
            is_admin = False
        
        if not is_admin:
            await update.message.reply_text("❌ Только администраторы могут сжимать историю.")
            return

    if key not in histories or len(histories[key]) < SUMMARY_INTERVAL:
        await update.message.reply_text("📝 История слишком короткая для суммаризации (нужно минимум 10 сообщений).")
        return

    history = histories[key]
    old_len = len(history)
    
    status_msg = await update.message.reply_text("📝 Создаю краткое содержание диалога...")
    
    try:
        new_history = await summarize_history(history)
        
        if new_history is history:
            await status_msg.edit_text("❌ Не удалось создать резюме.")
            return
        
        histories[key] = new_history
        new_len = len(new_history)
        
        await status_msg.edit_text(
            f"✅ Диалог сжат: {old_len} → {new_len} сообщений.\n"
            f"Суть разговора сохранена, последние реплики остались нетронутыми."
        )
        logger.info(f"📝 Ручная суммаризация: {old_len} → {new_len} сообщений для {key}")
        
    except Exception as e:
        await status_msg.edit_text(f"❌ Ошибка при суммаризации: {e}")
        logger.error(f"❌ Ошибка ручной суммаризации: {e}")

# ========== ЛОГИКА СКЛЕИВАНИЯ СООБЩЕНИЙ ==========

async def process_buffered_messages(buffer_key: str, update: Update, context: ContextTypes.DEFAULT_TYPE, key: str, is_group: bool, user_id: int, user_name: str, mentioned: bool, random_reply: bool, reason: str):
    # Эта функция вызывается после задержки
    data = message_buffer.get(buffer_key)
    if not data:
        return

    messages = data["messages"]
    del message_buffer[buffer_key]

    if not messages:
        return

    # Склеиваем сообщения
    # Описания медиа (картинок/видео) добавляем как [Image description: ...] или [Video description: ...]
    media_items = [
        (m.get("media_kind", "image"), m["media_description"])
        for m in messages if m.get("media_description")
    ]

    if is_group:
        # Для группы мы просто берем базовую часть из первого сообщения и добавляем все тексты
        first_msg = messages[0]
        timestamp = first_msg["timestamp"]
        
        message_parts = [f"[User: {user_name}] [Time: {timestamp}]"]
        
        # Добавляем reply context из первого сообщения, если он был
        if first_msg["reply_to_name"]:
            message_parts.append(f"[Reply to: {first_msg['reply_to_name']}]")
            message_parts.append(f"[Quoted message: {first_msg['reply_to_text']}]")

        combined_text = "\n".join([m["text"] for m in messages if m["text"]])
        if combined_text:
            message_parts.append(f"[Message: {escape_user_text(combined_text)}]")
        else:
            message_parts.append(f"[Message: (сообщение без текста)]")

        for kind, desc in media_items:
            tag = "Video description" if kind == "video" else "Image description"
            message_parts.append(f"[{tag}: {escape_user_text(desc)}]")

        message_content = " ".join(message_parts)
    else:
        first_msg = messages[0]
        timestamp = first_msg["timestamp"]
        message_parts = [f"[User: {user_name}] [Time: {timestamp}]"]
        
        combined_text = "\n".join([m["text"] for m in messages if m["text"]])
        if combined_text:
            message_parts.append(f"[Message: {escape_user_text(combined_text)}]")
        else:
            message_parts.append("[Message: (без текста)]")

        for kind, desc in media_items:
            tag = "Video description" if kind == "video" else "Image description"
            message_parts.append(f"[{tag}: {escape_user_text(desc)}]")

        message_content = " ".join(message_parts)

    if key not in histories:
        histories[key] = []
    
    history = histories[key]
    history.append({"role": "user", "content": message_content})
    histories[key] = history

    if not (mentioned or random_reply):
        return

    await update.message.chat.send_action(action="typing")
    await send_llm_request(update, context, key, history, user_name, user_id, is_group, random_reply, reason)


async def queue_message(update: Update, context: ContextTypes.DEFAULT_TYPE,
                        text: str, media_description: str = None, media_kind: str = None):
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name
    is_group = update.effective_chat.type in ['group', 'supergroup']
    
    key = get_history_key(chat_id, not is_group, user_id)
    buffer_key = f"{chat_id}_{user_id}"

    if is_group and ALLOWED_GROUPS and chat_id not in ALLOWED_GROUPS:
        return
    if not is_group and ALLOWED_USERS and user_id not in ALLOWED_USERS:
        await update.message.reply_text("Не разговариваю с незнакомцами.")
        return

    now = datetime.now()
    timestamp = f"{now.hour:02d}:{now.minute:02d}"

    reply_to_name = None
    reply_to_text = None
    if update.message.reply_to_message:
        reply_to_name = update.message.reply_to_message.from_user.first_name
        reply_to_text = (update.message.reply_to_message.text or "сообщение без текста")[:80]

    mentioned, reason = is_bot_mentioned(update, context)
    random_reply = should_reply_randomly(chat_id) if is_group else False
    if not is_group:
        mentioned = True

    msg_data = {
        "text": text,
        "media_description": media_description,
        "media_kind": media_kind,
        "timestamp": timestamp,
        "reply_to_name": reply_to_name,
        "reply_to_text": reply_to_text
    }

    if buffer_key in message_buffer:
        # Отменяем предыдущий таймер
        message_buffer[buffer_key]["task"].cancel()
        message_buffer[buffer_key]["messages"].append(msg_data)
        
        # Обновляем флаги вызова
        if mentioned:
            message_buffer[buffer_key]["mentioned"] = True
            message_buffer[buffer_key]["reason"] = reason
        if random_reply:
            message_buffer[buffer_key]["random_reply"] = True
    else:
        message_buffer[buffer_key] = {
            "messages": [msg_data],
            "mentioned": mentioned,
            "random_reply": random_reply,
            "reason": reason
        }

    # Запускаем новый таймер на 4 секунды
    # Если за 4 секунды от пользователя не будет новых сообщений, сработает process_buffered_messages
    async def wait_and_process():
        try:
            await asyncio.sleep(4.0)
            data = message_buffer.get(buffer_key)
            if data:
                await process_buffered_messages(
                    buffer_key, update, context, key, is_group, user_id, user_name,
                    data["mentioned"], data["random_reply"], data["reason"]
                )
        except asyncio.CancelledError:
            pass # Таймер был отменен из-за нового сообщения

    message_buffer[buffer_key]["task"] = asyncio.create_task(wait_and_process())


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE): 
    if update.message is None:
        return
    
    chat_id = update.effective_chat.id
    chat_type = update.effective_chat.type
    chat_title = update.effective_chat.title or "ЛС"
    user_text = update.message.text or ""
    
    logger.info(f"📨 [{chat_title} | {chat_id} | {chat_type}] {update.effective_user.first_name}: {user_text or '(пусто)'}")

    if update.effective_user.id == context.bot.id:
        return

    await queue_message(update, context, text=user_text)


async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    if update.effective_user.id == context.bot.id:
        return

    caption = update.message.caption or ""

    if not VISION_MODE:
        # Vision выключен — просто пропустим текст подписи (если есть)
        if caption:
            await queue_message(update, context, text=caption)
        return

    # Показываем "печатает", пока vision-модель работает с картинкой
    try:
        await update.message.chat.send_action(action="typing")
    except Exception:
        pass

    image_description = None
    try:
        photo = update.message.photo[-1]
        image_bytes, mime = await download_media_as_base64(photo.file_id, context, return_bytes=True)
        image_description = await describe_image_bytes(image_bytes, mime, caption=caption)
    except Exception as e:
        logger.error(f"Ошибка обработки фото: {e}")

    if not image_description:
        # Если описать не удалось — оставим хотя бы пометку
        image_description = "(не удалось разобрать изображение)"

    await queue_message(update, context, text=caption,
                        media_description=image_description, media_kind="image")


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    if update.effective_user.id == context.bot.id:
        return

    caption = update.message.caption or ""

    if not VISION_MODE:
        if caption:
            await queue_message(update, context, text=caption)
        return

    # Telegram отдаёт видео либо как .video, либо как .video_note (кружочки), либо как .animation (gif)
    video_obj = update.message.video or update.message.video_note or update.message.animation
    if video_obj is None:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    is_group = update.effective_chat.type in ['group', 'supergroup']

    # Применяем те же ACL, что и в queue_message — чтобы не качать видео впустую
    if is_group and ALLOWED_GROUPS and chat_id not in ALLOWED_GROUPS:
        return
    if not is_group and ALLOWED_USERS and user_id not in ALLOWED_USERS:
        await update.message.reply_text("Не разговариваю с незнакомцами.")
        return

    # Раннее отсечение по метаданным Telegram (быстрее, чем качать)
    tg_duration_raw = getattr(video_obj, "duration", 0) or 0
    # PTB начиная с v22.2 постепенно мигрирует duration с int на datetime.timedelta.
    # Поддерживаем оба варианта, чтобы не зависеть от версии.
    if hasattr(tg_duration_raw, "total_seconds"):
        tg_duration = tg_duration_raw.total_seconds()
    else:
        tg_duration = float(tg_duration_raw)

    if tg_duration and tg_duration > VIDEO_MAX_DURATION_SEC:
        await update.message.reply_text(
            f"Видео длиннее {VIDEO_MAX_DURATION_SEC} сек — не буду смотреть."
        )
        return

    try:
        await update.message.chat.send_action(action="upload_video")
    except Exception:
        pass

    video_description = None

    if is_native_video_supported():
        # Gemini принимает видео целиком — без извлечения кадров
        video_path = None
        try:
            video_path, mime, duration = await download_video_to_file(
                video_obj.file_id, context,
                max_duration_sec=VIDEO_MAX_DURATION_SEC
            )
            if video_path is None and duration > VIDEO_MAX_DURATION_SEC:
                await update.message.reply_text(
                    f"Видео длиннее {VIDEO_MAX_DURATION_SEC} сек ({duration:.0f}с) — не буду смотреть."
                )
                return
            if video_path:
                try:
                    await update.message.chat.send_action(action="typing")
                except Exception:
                    pass
                video_description = await describe_video(
                    video_path=video_path, mime=mime,
                    caption=caption, duration=duration
                )
        except Exception as e:
            logger.error(f"Ошибка обработки видео (gemini): {e}", exc_info=True)
        finally:
            if video_path:
                try:
                    import os as _os
                    _os.remove(video_path)
                except OSError:
                    pass
    else:
        # LM Studio: извлекаем кадры через ffmpeg и шлём как набор картинок
        try:
            frames, duration = await extract_video_frames(
                video_obj.file_id, context,
                num_frames=VIDEO_MAX_FRAMES,
                max_duration_sec=VIDEO_MAX_DURATION_SEC
            )

            if not frames and duration > VIDEO_MAX_DURATION_SEC:
                await update.message.reply_text(
                    f"Видео длиннее {VIDEO_MAX_DURATION_SEC} сек ({duration:.0f}с) — не буду смотреть."
                )
                return

            if not frames:
                logger.warning("Не удалось извлечь кадры из видео")
            else:
                try:
                    await update.message.chat.send_action(action="typing")
                except Exception:
                    pass
                video_description = await describe_video(
                    frames_data_urls=frames, caption=caption, duration=duration
                )
        except Exception as e:
            logger.error(f"Ошибка обработки видео (lmstudio): {e}", exc_info=True)

    if not video_description:
        video_description = "(не удалось разобрать видео)"

    await queue_message(update, context, text=caption,
                        media_description=video_description, media_kind="video")


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"❌ Глобальная ошибка: {context.error}", exc_info=True)
    try:
        if update and update.effective_message:
            await update.effective_message.reply_text("Произошла ошибка. Попробуйте /clear.")
    except Exception as e:
        logger.error(f"Не удалось отправить сообщение об ошибке: {e}")
