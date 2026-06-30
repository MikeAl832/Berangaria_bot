import logging
import asyncio
from datetime import datetime
from functools import wraps
from telegram import Update, ReactionTypeEmoji
from telegram.ext import ContextTypes

import time

from config import (
    ADMIN_MODE, SUMMARY_INTERVAL, MAX_CONTEXT_TOKENS,
    ALLOWED_USERS, ALLOWED_GROUPS, VISION_MODE, VIDEO_MAX_DURATION_SEC,
    VIDEO_MAX_FILE_SIZE_BYTES,
    AUDIO_MAX_DURATION_SEC, MESSAGE_DEBOUNCE_SECONDS, RANDOM_REPLY_COOLDOWN,
    MAX_MEDIA_ITEMS_IN_CONTEXT, ADMIN_ALERT_CHAT_ID
)
from state import histories, get_history_key, message_buffer, chat_tokens, api_call_count, get_history_lock, _buffer_lock, touch_activity, save_history
import state
from llm_client import summarize_history, send_llm_request, record_user_memory
from vision_provider import describe_image_bytes, describe_video, transcribe_audio
from utils import (
    escape_user_text, is_bot_mentioned, should_reply_randomly,
    download_media_as_base64, download_video_to_file, download_audio_to_file, get_video_duration
)

logger = logging.getLogger(__name__)


def truncate_at_sentence(text: str, max_chars: int) -> str:
    """Обрезает текст по последнему полному предложению, чтобы не ломать мысль."""
    if len(text) <= max_chars:
        return text
    
    truncated = text[:max_chars]
    
    # Ищем последнюю точку, вопросительный или восклицательный знак
    for delimiter in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
        last_pos = truncated.rfind(delimiter)
        if last_pos > max_chars * 0.6:  # нашли в последних 40%
            return truncated[:last_pos + 1]
    
    # Фоллбэк: ищем запятую
    last_comma = truncated.rfind(', ')
    if last_comma > max_chars * 0.7:
        return truncated[:last_comma] + "..."
    
    # Последний фоллбэк: обрезаем жёстко
    return truncated + "..."


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
        state.delete_history(key)  # Чистим и в БД
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
        state.save_history(key)
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

async def process_buffered_messages(buffer_key: str, update: Update, context: ContextTypes.DEFAULT_TYPE, key: str, is_group: bool, user_id: int, user_name: str, mentioned: bool, random_reply: bool):
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
        if kind == "video":
            tag = "Video description"
        elif kind == "audio":
            tag = "Audio description"
        else:
            tag = "Image description"
        # Обрезаем слишком длинные описания по полным предложениям
        desc_truncated = truncate_at_sentence(desc, MAX_DESC_CHARS)
        message_parts.append(f"[{tag}: {escape_user_text(desc_truncated)}]")
    
    if len(media_items) > MAX_MEDIA_ITEMS_IN_CONTEXT:
        message_parts.append(f"[+{len(media_items) - MAX_MEDIA_ITEMS_IN_CONTEXT} more media items]")

    message_content = " ".join(message_parts)

    async with get_history_lock(key):
        if key not in histories:
            histories[key] = []

        history = histories[key]
        # Стабильный уникальный номер сообщения для reply по [#N] (не меняется до суммаризации/clear)
        # и telegram message_id последнего сообщения буфера — на него ляжет реплай.
        next_sid = max((m.get("sid", 0) for m in history), default=0) + 1
        last_mid = messages[-1].get("message_id")
        history.append({"role": "user", "content": message_content, "sid": next_sid, "mid": last_mid})
        histories[key] = history
        touch_activity(key)  # Обновляем время активности
        state.save_history(key)  # Персистим историю на диск

    # Копим слова юзера для долговременной памяти (медиа-описания не сохраняем).
    # Делаем это для ВСЕХ сообщений, даже если бот не отвечает — память про всю беседу.
    record_user_memory(key, combined_text, user_name, is_group)

    if not (mentioned or random_reply):
        return

    # «печатает…» показываем только при прямом обращении: ambient-пинг может закончиться
    # молчанием, и призрачный индикатор «Ber печатает» без сообщения выглядел бы как баг.
    if mentioned:
        await update.message.chat.send_action(action="typing")
    await send_llm_request(update, context, key, history, user_name, user_id, mentioned)


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
    mentioned, _ = is_bot_mentioned(update, context)
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
        "forward_info": forward_info,
        "message_id": update.message.message_id
    }

    # Запускаем новый таймер
    async def wait_and_process():
        try:
            await asyncio.sleep(MESSAGE_DEBOUNCE_SECONDS)
            data = message_buffer.get(buffer_key)
            if data:
                await process_buffered_messages(
                    buffer_key, update, context, key, is_group, user_id, user_name,
                    data["mentioned"], data["random_reply"]
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
            if random_reply:
                message_buffer[buffer_key]["random_reply"] = True
        else:
            message_buffer[buffer_key] = {
                "messages": [msg_data],
                "mentioned": mentioned,
                "random_reply": random_reply,
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


async def handle_edited_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Пользователь отредактировал сообщение. Если оно ещё НЕ ушло в DeepSeek
    (лежит в дебаунс-буфере), обновляем его текст по message_id, чтобы модель
    увидела финальную версию. Уже сброшенные в историю/отправленные — не трогаем.
    """
    edited = update.edited_message
    if edited is None or edited.from_user is None:
        return
    if edited.from_user.id == context.bot.id:
        return

    chat_id = edited.chat.id
    user_id = edited.from_user.id
    new_text = edited.text or edited.caption or ""
    buffer_key = f"{chat_id}_{user_id}"

    async with _buffer_lock:
        data = message_buffer.get(buffer_key)
        if not data:
            # Буфер уже сброшен — правка опоздала, ничего не делаем
            return
        for m in data["messages"]:
            if m.get("message_id") == edited.message_id:
                old_text = m.get("text", "")
                m["text"] = new_text
                logger.info(
                    f"✏️ [cyan]Правка в буфере[/] (msg_id={edited.message_id}): "
                    f"'{old_text[:40]}' → '{new_text[:40]}'"
                )
                break


async def handle_chat_event(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Служебные события группы: смена названия / фото / удаление фото.
    Пишем событие в историю (бот в курсе изменений) и всегда реагируем в характере.
    При смене фото — прогоняем новое фото через vision, чтобы Ber комментировал картинку.
    """
    msg = update.message
    if msg is None or msg.from_user is None:
        return
    if msg.from_user.id == context.bot.id:
        return

    chat_id = msg.chat.id
    user_id = msg.from_user.id
    user_name = msg.from_user.first_name
    is_group = msg.chat.type in ['group', 'supergroup']
    if not is_group:
        return

    if not _check_access_permissions(chat_id, user_id, is_group):
        return

    # Описываем событие
    media_desc = None
    if msg.new_chat_title:
        event_text = f'changed the group name to "{escape_user_text(msg.new_chat_title)}"'
    elif msg.delete_chat_photo:
        event_text = "removed the group photo"
    elif msg.new_chat_photo:
        event_text = "changed the group photo"
        if VISION_MODE:
            try:
                photo = msg.new_chat_photo[-1]  # самый крупный размер
                image_bytes, mime = await download_media_as_base64(photo.file_id, context, return_bytes=True)
                media_desc = await describe_image_bytes(image_bytes, mime, caption="Это новое фото группы.")
            except Exception as e:
                logger.error(f"❌ [red]Не удалось разобрать новое фото группы:[/] {e}")
    else:
        return

    logger.info(f"📢 [magenta]Событие группы[/] [[blue]{msg.chat.title or chat_id}[/]] {user_name}: {event_text}")

    now = datetime.now()
    timestamp = f"{now.hour:02d}:{now.minute:02d}"

    parts = [f"[User: {user_name}] [Time: {timestamp}] [Event: {event_text}]"]
    if media_desc:
        parts.append(f"[Image description: {escape_user_text(media_desc)}]")
    content = " ".join(parts)

    key = get_history_key(chat_id, False)

    async with get_history_lock(key):
        if key not in histories:
            histories[key] = []
        history = histories[key]
        next_sid = max((m.get("sid", 0) for m in history), default=0) + 1
        history.append({"role": "user", "content": content, "sid": next_sid, "mid": msg.message_id})
        histories[key] = history
        touch_activity(key)
        state.save_history(key)

    await msg.chat.send_action(action="typing")
    # mentioned=True — событие заметное, реагируем всегда; reply ляжет на служебное сообщение
    await send_llm_request(update, context, key, history, user_name, user_id, True)


async def handle_message_reaction(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Пассивно фиксирует реакции, которые ставят на сообщения САМОГО бота.
    Ничего не отправляет — лишь дописывает структурное поле incoming_reactions
    к assistant-записи (найденной по mid), чтобы бот «узнал» о реакции при следующем
    своём ходе. Сама заметка рендерится эфемерно в _render_history_for_api,
    мимо суммаризации и памяти.
    """
    mr = update.message_reaction
    if mr is None or mr.user is None:
        return  # аноним/канал/не тот апдейт — пропускаем (реакции ботов Telegram и так не шлёт)

    chat_id = mr.chat.id
    is_group = mr.chat.type in ['group', 'supergroup']
    key = get_history_key(chat_id, not is_group, mr.user.id)

    history = histories.get(key)
    if not history:
        return

    def _emojis(reaction_tuple):
        return [r.emoji for r in (reaction_tuple or []) if isinstance(r, ReactionTypeEmoji)]

    new_e = _emojis(mr.new_reaction)
    old_e = _emojis(mr.old_reaction)
    added = [e for e in new_e if e not in old_e]
    removed = [e for e in old_e if e not in new_e]
    if not added and not removed:
        return  # изменились только кастом/платные реакции — нам нечего записывать

    name = mr.user.first_name
    async with get_history_lock(key):
        # Привязываемся к сообщению бота по mid. Не нашли (реакция на чужое сообщение
        # или на наше, отправленное до фичи / ушедшее в резюме) — тихо выходим.
        target = next(
            (m for m in history
             if m.get("role") == "assistant" and m.get("mid") == mr.message_id),
            None,
        )
        if target is None:
            return
        inc = target.setdefault("incoming_reactions", [])
        for e in added:
            inc.append({"from": name, "emoji": e})
        for e in removed:
            for i, rec in enumerate(inc):
                if rec.get("from") == name and rec.get("emoji") == e:
                    inc.pop(i)
                    break
        if not inc:
            target.pop("incoming_reactions", None)
        histories[key] = history
        save_history(key)

    if added:
        logger.info(f"💟 [magenta]Реакция на сообщение бота:[/] {' '.join(added)} от {name}")
    if removed:
        logger.info(f"🚫 [dim]Сняли реакцию с сообщения бота:[/] {' '.join(removed)} ({name})")


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
        cached = state.get_cached_media_description(photo.file_unique_id)
        if cached is not None:
            logger.info(f"♻️ [dim]Фото уже разобрано ранее, берём из кэша[/]")
            image_description = cached
        else:
            image_bytes, mime = await download_media_as_base64(photo.file_id, context, return_bytes=True)
            image_description = await describe_image_bytes(image_bytes, mime, caption=caption)
            if image_description:
                state.cache_media_description(photo.file_unique_id, image_description)
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

    # Telegram не отдаёт боту файлы тяжелее 20 МБ — отсекаем до скачивания
    if video_obj.file_size and video_obj.file_size > VIDEO_MAX_FILE_SIZE_BYTES:
        await update.message.reply_text(
            f"Видео больше {VIDEO_MAX_FILE_SIZE_BYTES // (1024 * 1024)} МБ — Telegram не даёт мне такое скачать."
        )
        return

    # Проверяем кэш (повторное видео/гифку не качаем и не разбираем заново)
    cached = state.get_cached_media_description(video_obj.file_unique_id)
    if cached is not None:
        logger.info(f"♻️ [dim]Видео уже разобрано ранее, берём из кэша[/]")
        await queue_message(update, context, text=caption,
                            media_description=cached, media_kind="video")
        return

    video_description = None
    video_path = None

    try:
        video_path, mime, _ = await download_video_to_file(
            video_obj.file_id, context
        )

        if not video_path:
            video_description = "(не удалось скачать видео)"
        else:
            # describe_video удаляет файл в своём finally блоке.
            # Длительность берём из метаданных Telegram (download возвращает 0.0)
            video_description = await describe_video(
                video_path=video_path, mime=mime,
                caption=caption, duration=tg_duration
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

    # Кэшируем только успешный разбор (плейсхолдеры ошибок не кэшируем)
    if video_description and not video_description.startswith("(не удалось"):
        state.cache_media_description(video_obj.file_unique_id, video_description)

    await queue_message(update, context, text=caption,
                        media_description=video_description, media_kind="video")


async def handle_sticker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    if update.effective_user.id == context.bot.id:
        return

    sticker = update.message.sticker
    if sticker is None:
        return

    emoji = sticker.emoji or ""

    if not VISION_MODE:
        # Vision выключен — передаём хотя бы эмодзи стикера как текст
        await queue_message(update, context, text=(emoji or "(стикер)"))
        return

    # Анимированные .tgs (Lottie/вектор) Gemini не разбирает как картинку — фолбэк на эмодзи
    if sticker.is_animated:
        desc = f"Анимированный стикер с эмодзи {emoji}" if emoji else "Анимированный стикер"
        await queue_message(update, context, text="",
                            media_description=desc, media_kind="image")
        return

    # Проверяем кэш (повторный стикер не разбираем заново)
    cached = state.get_cached_media_description(sticker.file_unique_id)
    if cached is not None:
        logger.info(f"♻️ [dim]Стикер уже разобран ранее, берём из кэша[/]")
        await queue_message(update, context, text="",
                            media_description=cached, media_kind="image")
        return

    sticker_description = None
    sticker_kind = "image"
    hint = f"Это стикер из Telegram с эмодзи {emoji}." if emoji else "Это стикер из Telegram."

    try:
        if sticker.is_video:
            # Видео-стикер .webm — разбираем как короткое видео
            sticker_kind = "video"
            video_path, mime, duration = await download_video_to_file(sticker.file_id, context)
            if not video_path:
                sticker_description = "(не удалось скачать стикер)"
            else:
                sticker_description = await describe_video(
                    video_path=video_path, mime=mime, caption=hint, duration=duration
                )
        else:
            # Статичный стикер .webp — обычная картинка
            image_bytes, mime = await download_media_as_base64(
                sticker.file_id, context, return_bytes=True
            )
            sticker_description = await describe_image_bytes(image_bytes, mime, caption=hint)
    except Exception as e:
        logger.error(f"❌ [red]Ошибка обработки стикера:[/] {e}")

    if not sticker_description:
        sticker_description = f"Стикер с эмодзи {emoji}" if emoji else "(не удалось разобрать стикер)"
    elif not sticker_description.startswith("(не удалось"):
        state.cache_media_description(sticker.file_unique_id, sticker_description)

    await queue_message(update, context, text="",
                        media_description=sticker_description, media_kind=sticker_kind)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    if update.effective_user.id == context.bot.id:
        return

    if not VISION_MODE:
        return

    audio_obj = update.message.voice or update.message.audio
    if audio_obj is None:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    is_group = update.effective_chat.type in ['group', 'supergroup']

    if not _check_access_permissions(chat_id, user_id, is_group):
        if not is_group:
            await update.message.reply_text("Не разговариваю с незнакомцами.")
        return

    # Раннее отсечение слишком длинных аудио по метаданным Telegram
    duration = get_video_duration(audio_obj)
    if duration and duration > AUDIO_MAX_DURATION_SEC:
        await update.message.reply_text(
            f"Аудио длиннее {AUDIO_MAX_DURATION_SEC} сек — слушать не буду."
        )
        return

    caption = update.message.caption or ""

    # Кэш транскрипции (повторное аудио не распознаём заново)
    transcript = state.get_cached_media_description(audio_obj.file_unique_id)

    if transcript is None:
        try:
            await update.message.chat.send_action(action="typing")
        except Exception:
            pass

        try:
            audio_path, mime = await download_audio_to_file(audio_obj.file_id, context)
            if not audio_path:
                transcript = ""
            else:
                # transcribe_audio удаляет файл в своём finally
                transcript = await transcribe_audio(audio_path=audio_path, mime=mime, caption=caption)
        except Exception as e:
            logger.error(f"❌ [red]Ошибка обработки голосового:[/] {e}", exc_info=True)
            transcript = ""

        if transcript:
            state.cache_media_description(audio_obj.file_unique_id, transcript)

    if not transcript:
        # Не смогли распознать — отдадим хотя бы пометку, чтобы бот не молчал
        await queue_message(update, context, text="",
                            media_description="(голосовое сообщение, не удалось распознать)",
                            media_kind="audio")
        return

    logger.info(f"🎤 [cyan]Транскрипция:[/] {transcript[:80]}{'...' if len(transcript) > 80 else ''}")
    # Транскрипт передаём как audio description, чтобы модель понимала что это услышанное
    await queue_message(update, context, text=caption,
                        media_description=transcript, media_kind="audio")


# Троттлинг алертов админу, чтобы не спамить при серии ошибок
_last_alert_ts = 0.0
_ALERT_COOLDOWN_SEC = 60


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global _last_alert_ts
    logger.error(f"❌ [bright_red]Глобальная ошибка:[/] {context.error}", exc_info=True)
    try:
        if update and update.effective_message:
            await update.effective_message.reply_text("Произошла ошибка. Попробуйте /clear.")
    except Exception as e:
        logger.error(f"❌ [red]Не удалось отправить сообщение об ошибке:[/] {e}")

    # Алерт админу (если настроен и не спамим)
    if ADMIN_ALERT_CHAT_ID:
        now = time.time()
        if now - _last_alert_ts >= _ALERT_COOLDOWN_SEC:
            _last_alert_ts = now
            try:
                err_text = str(context.error)[:500]
                await context.bot.send_message(
                    chat_id=ADMIN_ALERT_CHAT_ID,
                    text=f"⚠️ Ошибка бота:\n{err_text}"
                )
            except Exception as e:
                logger.error(f"❌ [red]Не удалось отправить алерт админу:[/] {e}")
