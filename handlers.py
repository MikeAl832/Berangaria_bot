import logging
import asyncio
from datetime import datetime
from functools import wraps
from telegram import Update
from telegram.ext import ContextTypes

from config import (
    ADMIN_MODE, SUMMARY_INTERVAL, MAX_CONTEXT_TOKENS,
    ALLOWED_USERS, ALLOWED_GROUPS, VISION_MODE, VIDEO_MAX_DURATION_SEC,
    MESSAGE_DEBOUNCE_SECONDS, RANDOM_REPLY_COOLDOWN, MAX_MEDIA_ITEMS_IN_CONTEXT
)
from state import histories, get_history_key, message_buffer, chat_tokens, api_call_count, get_history_lock, _buffer_lock, touch_activity
import state
from llm_client import summarize_history, send_llm_request
from vision_provider import describe_image_bytes, describe_video
from utils import (
    escape_user_text, is_bot_mentioned, should_reply_randomly,
    download_media_as_base64, download_video_to_file, get_video_duration
)

logger = logging.getLogger(__name__)


# ========== ДЕКОРАТОР ДЛЯ ПРОВЕРКИ ПРАВ АДМИНИСТРАТОРА ==========

def admin_required(func):
    """
    Декоратор для команд, требующих прав администратора в группах.
    В личных чатах пропускает всех. В группах проверяет ADMIN_MODE.
    """
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.message is None:
            return
        
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id
        is_group = update.effective_chat.type in ['group', 'supergroup']
        
        # В личных чатах разрешаем всем
        if not is_group:
            return await func(update, context)
        
        # В группах проверяем ADMIN_MODE
        if ADMIN_MODE:
            try:
                chat_member = await context.bot.get_chat_member(chat_id, user_id)
                is_admin = chat_member.status in ['administrator', 'creator']
            except Exception:
                is_admin = False
            
            if not is_admin:
                await update.message.reply_text("❌ Только администраторы могут использовать эту команду!")
                return
        
        return await func(update, context)
    
    return wrapper

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
            f"🎲 Шанс случайного ответа: {state.random_reply_chance}%\n"
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

@admin_required
async def random_chance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    is_group = update.effective_chat.type in ['group', 'supergroup']
    if not is_group:
        await update.message.reply_text("Эта команда работает только в группах!")
        return

    if not context.args:
        await update.message.reply_text(f"Текущий шанс: {state.random_reply_chance}%\nИспользуйте: /random 0-100")
        return

    try:
        new_chance = int(context.args[0])
        if not 0 <= new_chance <= 100:
            await update.message.reply_text("Шанс должен быть от 0 до 100!")
            return

        state.random_reply_chance = new_chance
        await update.message.reply_text(f"✅ Шанс изменён на {state.random_reply_chance}%")

    except ValueError:
        await update.message.reply_text("Укажите число от 0 до 100")

@admin_required
async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    is_group = update.effective_chat.type in ['group', 'supergroup']
    key = get_history_key(chat_id, not is_group, user_id)

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
        f"🎲 Шанс случайного ответа: {state.random_reply_chance}%"
    )

@admin_required
async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    is_group = update.effective_chat.type in ['group', 'supergroup']
    key = get_history_key(chat_id, not is_group, user_id)

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
        logger.info(f"📝 Ручная суммаризация: [green]{old_len} → {new_len}[/] сообщений для {key}")
        
    except Exception as e:
        await status_msg.edit_text(f"❌ Ошибка при суммаризации: {e}")
        logger.error(f"❌ [red]Ошибка ручной суммаризации:[/] {e}")

# ========== ЛОГИКА СКЛЕИВАНИЯ СООБЩЕНИЙ ==========

async def process_buffered_messages(buffer_key: str, update: Update, context: ContextTypes.DEFAULT_TYPE, key: str, is_group: bool, user_id: int, user_name: str, mentioned: bool, random_reply: bool, reason: str):
    # Эта функция вызывается после задержки
    async with _buffer_lock:
        data = message_buffer.get(buffer_key)
        if not data:
            return

        messages = data["messages"]
        del message_buffer[buffer_key]

    # Формируем единое сообщение из буфера
    first_msg = messages[0]
    timestamp = first_msg["timestamp"]
    
    message_parts = [f"[User: {user_name}] [Time: {timestamp}]"]
    
    # Добавляем информацию о пересылке, если она есть
    if first_msg.get("forward_info"):
        message_parts.append(f"[{first_msg['forward_info']}]")
    
    # Reply context (только для групп, в личке не нужен)
    if is_group and first_msg["reply_to_name"]:
        message_parts.append(f"[Reply to: {first_msg['reply_to_name']}]")
        message_parts.append(f"[Quoted message: {first_msg['reply_to_text']}]")

    # Склеиваем текст из всех сообщений
    combined_text = "\n".join([m["text"] for m in messages if m["text"]])
    if combined_text:
        message_parts.append(f"[Message: {escape_user_text(combined_text)}]")
    else:
        message_parts.append(f"[Message: ({'сообщение без текста' if is_group else 'без текста'})]")

    # Добавляем описания медиа (ограничиваем для экономии токенов)
    MAX_DESC_CHARS = 800  # Максимум символов на одно описание медиа
    media_items = [
        (m.get("media_kind", "image"), m["media_description"])
        for m in messages if m.get("media_description")
    ]
    for kind, desc in media_items[:MAX_MEDIA_ITEMS_IN_CONTEXT]:
        tag = "Video description" if kind == "video" else "Image description"
        # Обрезаем слишком длинные описания
        desc_truncated = desc[:MAX_DESC_CHARS] + ("..." if len(desc) > MAX_DESC_CHARS else "")
        message_parts.append(f"[{tag}: {escape_user_text(desc_truncated)}]")
    
    if len(media_items) > MAX_MEDIA_ITEMS_IN_CONTEXT:
        message_parts.append(f"[+{len(media_items) - MAX_MEDIA_ITEMS_IN_CONTEXT} more media items]")

    message_content = " ".join(message_parts)

    async with get_history_lock(key):
        if key not in histories:
            histories[key] = []
        
        history = histories[key]
        history.append({"role": "user", "content": message_content})
        histories[key] = history
        touch_activity(key)  # Обновляем время активности

    if not (mentioned or random_reply):
        return

    await update.message.chat.send_action(action="typing")
    await send_llm_request(update, context, key, history, user_name, user_id, is_group, random_reply, reason)


def _check_access_permissions(chat_id: int, user_id: int, is_group: bool) -> bool:
    """Проверяет права доступа к боту для пользователя/группы."""
    if is_group and ALLOWED_GROUPS and chat_id not in ALLOWED_GROUPS:
        return False
    if not is_group and ALLOWED_USERS and user_id not in ALLOWED_USERS:
        return False
    return True


def _extract_forward_info(message) -> str | None:
    """Извлекает информацию о пересылке сообщения."""
    if not message.forward_origin:
        return None
    
    origin = message.forward_origin
    forward_type = origin.type
    
    if forward_type == "user":
        return f"Forwarded from user: {origin.sender_user.first_name}"
    elif forward_type == "hidden_user":
        return f"Forwarded from: {origin.sender_user_name}"
    elif forward_type == "chat":
        chat_title = origin.sender_chat.title if origin.sender_chat else "Unknown chat"
        return f"Forwarded from chat: {chat_title}"
    elif forward_type == "channel":
        chat_title = origin.chat.title if origin.chat else "Unknown channel"
        return f"Forwarded from channel: {chat_title}"
    
    return None


def _extract_reply_context(message) -> tuple[str | None, str | None]:
    """Извлекает контекст reply-сообщения (на кого отвечает)."""
    if not message.reply_to_message:
        return None, None
    
    reply_to_name = message.reply_to_message.from_user.first_name
    reply_to_text = (message.reply_to_message.text or "сообщение без текста")[:80]
    return reply_to_name, reply_to_text


async def queue_message(update: Update, context: ContextTypes.DEFAULT_TYPE,
                        text: str, media_description: str = None, media_kind: str = None):
    """
    Добавляет сообщение в буфер для склейки последовательных сообщений от одного пользователя.
    Если за MESSAGE_DEBOUNCE_SECONDS не будет новых сообщений, буфер обрабатывается.
    """
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name
    is_group = update.effective_chat.type in ['group', 'supergroup']
    
    key = get_history_key(chat_id, not is_group, user_id)
    buffer_key = f"{chat_id}_{user_id}"

    # Проверка прав доступа (используем общую функцию)
    if not _check_access_permissions(chat_id, user_id, is_group):
        if not is_group:
            await update.message.reply_text("Не разговариваю с незнакомцами.")
        return

    # Формируем метаданные сообщения
    now = datetime.now()
    timestamp = f"{now.hour:02d}:{now.minute:02d}"
    reply_to_name, reply_to_text = _extract_reply_context(update.message)
    forward_info = _extract_forward_info(update.message)

    # Определяем, должен ли бот ответить
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
        "reply_to_text": reply_to_text,
        "forward_info": forward_info
    }

    # Запускаем новый таймер
    async def wait_and_process():
        try:
            await asyncio.sleep(MESSAGE_DEBOUNCE_SECONDS)
            data = message_buffer.get(buffer_key)
            if data:
                await process_buffered_messages(
                    buffer_key, update, context, key, is_group, user_id, user_name,
                    data["mentioned"], data["random_reply"], data["reason"]
                )
        except asyncio.CancelledError:
            pass  # Таймер был отменен из-за нового сообщения

    # Добавляем в буфер с блокировкой (все операции атомарны)
    async with _buffer_lock:
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
        
        # Создаём таску внутри блокировки для атомарности
        message_buffer[buffer_key]["task"] = asyncio.create_task(wait_and_process())


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE): 
    if update.message is None:
        return
    
    chat_id = update.effective_chat.id
    chat_type = update.effective_chat.type
    chat_title = update.effective_chat.title or "ЛС"
    user_text = update.message.text or ""
    
    # Обрезаем длинные сообщения для INFO режима (в DEBUG будет полное)
    log_text = user_text if len(user_text) <= 80 else f"{user_text[:77]}..."
    logger.info(f"📨 [[blue]{chat_title} | {chat_id} | {chat_type}[/]] [cyan]{update.effective_user.first_name}[/]: {log_text or '(пусто)'}")

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

    image_description = None
    try:
        photo = update.message.photo[-1]
        image_bytes, mime = await download_media_as_base64(photo.file_id, context, return_bytes=True)
        image_description = await describe_image_bytes(image_bytes, mime, caption=caption)
    except Exception as e:
        logger.error(f"❌ [red]Ошибка обработки фото:[/] {e}")

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

    # Проверка доступа (единственная, queue_message также проверит)
    if not _check_access_permissions(chat_id, user_id, is_group):
        if not is_group:
            await update.message.reply_text("Не разговариваю с незнакомцами.")
        return

    # Раннее отсечение по метаданным Telegram (быстрее, чем качать)
    tg_duration = get_video_duration(video_obj)
    
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
    video_path = None

    try:
        video_path, mime, duration = await download_video_to_file(
            video_obj.file_id, context
        )
        
        if not video_path:
            video_description = "(не удалось скачать видео)"
        else:
            # describe_video удаляет файл в своём finally блоке
            video_description = await describe_video(
                video_path=video_path, mime=mime,
                caption=caption, duration=duration
            )
            video_path = None  # Файл удалён в describe_video
    except Exception as e:
        logger.error(f"❌ [red]Ошибка обработки видео:[/] {e}", exc_info=True)
        video_description = "(не удалось разобрать видео)"
    finally:
        # Гарантия удаления файла, если он не был обработан
        if video_path:
            try:
                import os
                if os.path.exists(video_path):
                    os.remove(video_path)
                    logger.debug(f"🗑️ Удалён временный файл (fallback): {video_path}")
            except OSError as err:
                logger.warning(f"⚠️ Не удалось удалить временный файл {video_path}: {err}")

    await queue_message(update, context, text=caption,
                        media_description=video_description, media_kind="video")


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"❌ [bright_red]Глобальная ошибка:[/] {context.error}", exc_info=True)
    try:
        if update and update.effective_message:
            await update.effective_message.reply_text("Произошла ошибка. Попробуйте /clear.")
    except Exception as e:
        logger.error(f"❌ [red]Не удалось отправить сообщение об ошибке:[/] {e}")
