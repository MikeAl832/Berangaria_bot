import logging
from functools import wraps
from telegram import Update, ReactionTypeEmoji
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from berangaria.config import (
    ADMIN_MODE, SUMMARY_INTERVAL, MAX_CONTEXT_TOKENS,
    ALLOWED_USERS, ALLOWED_GROUPS, OWNER_USER_ID, VISION_MODE, VIDEO_MAX_DURATION_SEC,
    VIDEO_MAX_FILE_SIZE_BYTES,
    AUDIO_MAX_DURATION_SEC, MESSAGE_DEBOUNCE_SECONDS, MAX_MEDIA_ITEMS_IN_CONTEXT,
    MAX_BUFFERED_MESSAGES, MAX_BUFFERED_CHARS,
    LOG_MESSAGE_PREVIEW_CHARS,
)
from berangaria.core.state import (
    histories, get_history_key, message_buffer, chat_tokens, api_call_count,
    get_history_lock, get_turn_lock, _buffer_lock, touch_activity, save_history,
)
from berangaria.core import state
from berangaria.core import alerts
from berangaria.chat.llm_client import summarize_history, send_llm_request
from berangaria.chat import media_handlers
from berangaria.chat import message_queue
from berangaria.chat import analytics_ui
from berangaria.analytics import store as analytics_store
from berangaria.memory.pipeline import (
    abandon_memory_sources,
    enqueue_memory_source,
    release_memory_sources,
)
from berangaria.media.vision import (
    describe_image_bytes,
    describe_images,
    describe_video,
    transcribe_audio,
    VISION_FAILED_IMAGE,
)
from berangaria.core.utils import (
    escape_user_text, is_bot_mentioned, should_reply_randomly,
    download_media_as_base64, download_video_to_file, download_audio_to_file, get_video_duration,
    now_local, strip_tiktok_urls,
)

logger = logging.getLogger(__name__)

# Telegram delivers album photos as separate updates with the same media_group_id.
# Wait after the last photo so we can download them all and describe in one Gemini call.
ALBUM_GATHER_SECONDS = 1.2
_album_buffer = media_handlers.album_buffer
_album_lock = media_handlers.album_lock


def _queue_runtime() -> message_queue.QueueRuntime:
    """Snapshot queue settings and patchable boundaries for one enqueue call."""
    return message_queue.QueueRuntime(
        message_debounce_seconds=MESSAGE_DEBOUNCE_SECONDS,
        max_media_items_in_context=MAX_MEDIA_ITEMS_IN_CONTEXT,
        max_buffered_messages=MAX_BUFFERED_MESSAGES,
        max_buffered_chars=MAX_BUFFERED_CHARS,
        owner_user_id=OWNER_USER_ID,
        check_access_permissions=_check_access_permissions,
        truncate_at_sentence=truncate_at_sentence,
        build_memory_text=_build_memory_text,
        extract_forward_info=_extract_forward_info,
        extract_reply_context=_extract_reply_context,
        log_message_preview=_log_message_preview,
        is_bot_mentioned=is_bot_mentioned,
        should_reply_randomly=should_reply_randomly,
        enqueue_memory_source=enqueue_memory_source,
        release_memory_sources=release_memory_sources,
        abandon_memory_sources=abandon_memory_sources,
        send_llm_request=send_llm_request,
        process_buffered_messages=process_buffered_messages,
    )


def _media_runtime() -> media_handlers.MediaRuntime:
    """Snapshot media dependencies while preserving the existing test seams."""
    return media_handlers.MediaRuntime(
        vision_mode=VISION_MODE,
        album_gather_seconds=ALBUM_GATHER_SECONDS,
        max_media_items_in_context=MAX_MEDIA_ITEMS_IN_CONTEXT,
        video_max_duration_sec=VIDEO_MAX_DURATION_SEC,
        video_max_file_size_bytes=VIDEO_MAX_FILE_SIZE_BYTES,
        audio_max_duration_sec=AUDIO_MAX_DURATION_SEC,
        vision_failed_image=VISION_FAILED_IMAGE,
        check_access_permissions=_check_access_permissions,
        queue_message=queue_message,
        download_media_as_base64=download_media_as_base64,
        download_video_to_file=download_video_to_file,
        download_audio_to_file=download_audio_to_file,
        get_video_duration=get_video_duration,
        describe_image_bytes=describe_image_bytes,
        describe_images=describe_images,
        describe_video=describe_video,
        transcribe_audio=transcribe_audio,
    )


def _log_message_preview(text: str) -> str:
    """Bound message text written to INFO logs without changing chat content."""
    if len(text) <= LOG_MESSAGE_PREVIEW_CHARS:
        return text
    return f"{text[:LOG_MESSAGE_PREVIEW_CHARS - 3]}..."


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


def _build_memory_text(combined_text: str, *, is_forwarded: bool = False) -> str:
    """
    Возвращает только пользовательский текст для долговременной памяти.
    Описания vision-модели намеренно не являются источниками фактов, поэтому
    медиа сюда нельзя передать даже параметром.
    """
    if is_forwarded:
        return ""
    return (combined_text or "").strip()


# ========== ДЕКОРАТОРЫ ДОСТУПА К КОМАНДАМ ==========

def access_required(func):
    """Применяет тот же список доступа, что и к обычным сообщениям.

    Без него команды остаются единственным входом, не проходящим
    `_check_access_permissions`: посторонний может добавить бота в свою группу
    и управлять им, хотя ни одно его сообщение бот не обработает. Отвечать
    отказом нельзя — это подтвердило бы присутствие бота, поэтому молча выходим.
    """
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.message is None:
            return

        chat_id = update.effective_chat.id
        user_id = update.effective_user.id
        is_group = update.effective_chat.type in ['group', 'supergroup']

        if not _check_access_permissions(chat_id, user_id, is_group):
            logger.info(
                f"🚫 [dim]Команда от постороннего отклонена "
                f"(chat={chat_id}, user={user_id})[/]"
            )
            return

        return await func(update, context)

    return wrapper


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


def owner_private_required(func):
    """Allow a sensitive command only from the authenticated owner's DM."""
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.message is None:
            return
        user_id = update.effective_user.id
        is_private = update.effective_chat.type == "private"
        if OWNER_USER_ID is None or user_id != OWNER_USER_ID:
            logger.info("🚫 [dim]Owner-команда отклонена (user=%s)[/]", user_id)
            return
        if not is_private:
            await update.message.reply_text("Панель владельца доступна только в личном чате со мной.")
            return
        return await func(update, context)

    return wrapper

@access_required
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
            f"/top — лидерборд чата\n"
            f"/summarize — сжатие истории\n"
            f"/random X — изменить шанс случайных ответов"
        )
    else:
        owner_command = (
            "\n/dashboard — панель владельца"
            if update.effective_user.id == OWNER_USER_ID
            else ""
        )
        await update.message.reply_text(
            f"👋 Привет, {user_name}!\n\n"
            f"Команды:\n"
            f"/clear — очистить историю\n"
            f"/stats — статистика\n"
            f"/top — лидерборд чата\n"
            f"/summarize — сжатие истории"
            f"{owner_command}"
        )

@access_required
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

        saved_chance = state.set_random_reply_chance(new_chance)
        await update.message.reply_text(f"✅ Шанс изменён на {saved_chance}%")

    except ValueError:
        await update.message.reply_text("Укажите число от 0 до 100")

async def _discard_pending_buffers(
    key: str, chat_id: int, user_id: int, is_group: bool
) -> int:
    """Отменяет ждущие debounce-буферы чата и хоронит их источники памяти.

    Источники именно хоронятся, а не освобождаются: их текст пользователь
    только что попросил стереть, а брошенный `waiting` заблокировал бы очередь
    памяти этой области.
    """
    # В группе история общая, но буферы — по пользователям: гасим все буферы чата.
    prefix = f"{chat_id}_" if is_group else f"{chat_id}_{user_id}"
    discarded = 0
    source_ids: list[int | None] = []
    async with _buffer_lock:
        for buffer_key in [k for k in message_buffer if k.startswith(prefix)]:
            data = message_buffer.pop(buffer_key)
            task = data.get("task")
            if task is not None:
                task.cancel()
            source_ids.extend(
                message.get("memory_source_id") for message in data.get("messages", [])
            )
            discarded += 1

    # Album gather buffers have not enqueued memory yet — just cancel timers.
    async with _album_lock:
        for album_key in [k for k in _album_buffer if k.startswith(prefix)]:
            data = _album_buffer.pop(album_key)
            task = data.get("task")
            if task is not None:
                task.cancel()
            discarded += 1

    if source_ids:
        abandon_memory_sources(source_ids)
    if discarded:
        logger.info(
            f"🧹 [dim]Сброшено ждущих буферов для '{key}': {discarded}[/]"
        )
    return discarded


def _album_cache_key(file_unique_ids: list[str]) -> str:
    return media_handlers.album_cache_key(file_unique_ids)


async def _flush_album_after_delay(album_key: str) -> None:
    await media_handlers.flush_album_after_delay(album_key, _media_runtime())


async def _process_photo_album(data: dict) -> None:
    await media_handlers.process_photo_album(data, _media_runtime())


async def _buffer_album_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    media_group_id: str,
    caption: str,
) -> None:
    await media_handlers.buffer_album_photo(
        update, context, media_group_id, caption, _media_runtime()
    )


@access_required
@admin_required
async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    is_group = update.effective_chat.type in ['group', 'supergroup']
    key = get_history_key(chat_id, not is_group, user_id)

    # Сначала гасим debounce-буфер: иначе его таймер проснётся через несколько
    # секунд, пересоздаст историю из доклирового сообщения и ответит на то, что
    # просили стереть. Делаем это ДО взятия turn-lock — вложенности
    # turn -> _buffer_lock нет больше нигде, и заводить её нельзя.
    await _discard_pending_buffers(key, chat_id, user_id, is_group)

    async with get_turn_lock(key):
        async with get_history_lock(key):
            existed = key in histories
            deleted = state.delete_history(key)  # Чистим и в БД
            if deleted:
                histories.pop(key, None)

    if not deleted:
        await update.message.reply_text(
            "⚠️ Не удалось очистить историю — она осталась на месте. Попробуйте позже."
        )
        return

    if existed:
        await update.message.reply_text("🧹 История очищена!")
    else:
        await update.message.reply_text("История и так пуста!")

@access_required
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


@access_required
async def top(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return
    try:
        text = analytics_ui.render_top(update.effective_chat.id)
    except Exception as exc:
        logger.error("❌ Не удалось построить /top: %s", exc, exc_info=True)
        await update.message.reply_text("Статистика временно недоступна.")
        return
    await update.message.reply_text(text)


@owner_private_required
async def dashboard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text, keyboard = analytics_ui.render_dashboard()
    except Exception as exc:
        logger.error("❌ Не удалось построить /dashboard: %s", exc, exc_info=True)
        await update.message.reply_text("Панель временно недоступна.")
        return
    await update.message.reply_text(text, reply_markup=keyboard)


async def dashboard_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if query is None:
        return
    message = query.message
    user_id = query.from_user.id if query.from_user is not None else None
    chat_type = getattr(getattr(message, "chat", None), "type", None)
    if OWNER_USER_ID is None or user_id != OWNER_USER_ID or chat_type != "private":
        await query.answer("Недоступно", show_alert=True)
        return
    page, period = analytics_ui.parse_dashboard_callback(query.data or "")
    try:
        text, keyboard = analytics_ui.render_dashboard(page, period)
    except Exception as exc:
        logger.error("❌ Не удалось обновить /dashboard: %s", exc, exc_info=True)
        await query.answer("Статистика временно недоступна", show_alert=True)
        return
    await query.answer()
    try:
        await query.edit_message_text(text=text, reply_markup=keyboard)
    except BadRequest as exc:
        if "message is not modified" not in str(exc).lower():
            raise

@access_required
@admin_required
async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    is_group = update.effective_chat.type in ['group', 'supergroup']
    key = get_history_key(chat_id, not is_group, user_id)

    async with get_turn_lock(key):
        history = histories.get(key)
        # После сжатия всегда 1 резюме + SUMMARY_INTERVAL свежих = SUMMARY_INTERVAL+1.
        # При len <= SUMMARY_INTERVAL+1 либо сжимать нечего, либо длина не уменьшится —
        # не дёргаем DeepSeek впустую.
        min_for_shrink = SUMMARY_INTERVAL + 1
        if not history or len(history) <= min_for_shrink:
            await update.message.reply_text(
                f"📝 История слишком короткая для суммаризации "
                f"(нужно больше {min_for_shrink} сообщений)."
            )
            return

        old_len = len(history)
        status_msg = await update.message.reply_text("📝 Создаю краткое содержание диалога...")

        try:
            new_history = await summarize_history(history)

            if new_history is history:
                await status_msg.edit_text("❌ Не удалось создать резюме.")
                return

            new_len = len(new_history)
            # Резюме, которое не короче исходника, — не сжатие. Записывать его
            # значит платить за вызов и рапортовать «сжат: 11 → 11».
            if new_len >= old_len:
                await status_msg.edit_text(
                    "❌ Резюме не короче исходной истории — оставила как было."
                )
                logger.warning(
                    f"📝 [yellow]Суммаризация не сократила историю {key}:[/] "
                    f"{old_len} → {new_len}"
                )
                return

            async with get_history_lock(key):
                histories[key] = new_history
                saved = state.save_history(key)
            if not saved:
                await status_msg.edit_text(
                    "⚠️ Резюме создано, но не записалось в БД — после перезапуска "
                    "вернётся прежняя история."
                )
                return

            await status_msg.edit_text(
                f"✅ Диалог сжат: {old_len} → {new_len} сообщений.\n"
                "Суть разговора сохранена, последние реплики остались нетронутыми."
            )
            logger.info(f"📝 Ручная суммаризация: [green]{old_len} → {new_len}[/] сообщений для {key}")

        except Exception as e:
            await status_msg.edit_text(f"❌ Ошибка при суммаризации: {e}")
            logger.error(f"❌ [red]Ошибка ручной суммаризации:[/] {e}")

# ========== ЛОГИКА СКЛЕИВАНИЯ СООБЩЕНИЙ ==========

async def process_buffered_messages(buffer_key: str, update: Update, context: ContextTypes.DEFAULT_TYPE, key: str, is_group: bool, user_id: int, user_name: str, mentioned: bool, random_reply: bool):
    await message_queue.process_buffered_messages(
        buffer_key,
        update,
        context,
        key,
        is_group,
        user_id,
        user_name,
        mentioned,
        random_reply,
        _queue_runtime(),
    )


def _check_access_permissions(chat_id: int, user_id: int, is_group: bool) -> bool:
    """Проверяет права доступа к боту для пользователя/группы."""
    if OWNER_USER_ID is not None and user_id == OWNER_USER_ID:
        return True
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
    raw_reply = message.reply_to_message.text or message.reply_to_message.caption or "сообщение без текста"
    reply_to_text = (strip_tiktok_urls(raw_reply) or "сообщение без текста")[:80]
    return reply_to_name, reply_to_text


async def queue_message(update: Update, context: ContextTypes.DEFAULT_TYPE,
                        text: str, media_description: str = None, media_kind: str = None):
    await message_queue.queue_message(
        update,
        context,
        text,
        media_description,
        media_kind,
        _queue_runtime(),
    )


async def queue_bridge_bot_message(
    update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    text: str,
    media_description: str | None = None,
    media_kind: str | None = None,
    reply_to_name: str | None = None,
    reply_to_text: str | None = None,
    reply_to_user_id: int | None = None,
    created_at: float | None = None,
):
    await message_queue.queue_bridge_bot_message(
        update,
        context,
        text=text,
        media_description=media_description,
        media_kind=media_kind,
        reply_to_name=reply_to_name,
        reply_to_text=reply_to_text,
        reply_to_user_id=reply_to_user_id,
        created_at=created_at,
        runtime=_queue_runtime(),
    )


async def _enqueue_buffered(
    *,
    buffer_key: str,
    msg_data: dict,
    update,
    context: ContextTypes.DEFAULT_TYPE,
    key: str,
    is_group: bool,
    user_id: int,
    user_name: str,
    mentioned: bool,
    random_reply: bool,
):
    await message_queue.enqueue_buffered(
        buffer_key=buffer_key,
        msg_data=msg_data,
        update=update,
        context=context,
        key=key,
        is_group=is_group,
        user_id=user_id,
        user_name=user_name,
        mentioned=mentioned,
        random_reply=random_reply,
        runtime=_queue_runtime(),
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE): 
    if update.message is None:
        return
    
    chat_id = update.effective_chat.id
    chat_type = update.effective_chat.type
    chat_title = update.effective_chat.title or "ЛС"
    user_text = update.message.text or ""
    
    # INFO preview is bounded independently from the content sent to the bot.
    log_text = _log_message_preview(user_text)
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
    new_text = strip_tiktok_urls(edited.text or edited.caption or "")
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

    now = now_local()
    timestamp = f"{now.hour:02d}:{now.minute:02d}"

    author_kind = "Owner" if user_id == OWNER_USER_ID else "User"
    parts = [f"[{author_kind}: {user_name}] [Time: {timestamp}] [Event: {event_text}]"]
    if media_desc:
        parts.append(f"[Image description: {escape_user_text(media_desc)}]")
    content = " ".join(parts)

    key = get_history_key(chat_id, False)

    async with get_turn_lock(key):
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
    async with get_turn_lock(key):
        async with get_history_lock(key):
            # История могла быть очищена, пока реакция ждала turn-lock.
            history = histories.get(key)
            if not history:
                return
            target = next(
                (m for m in history
                 if m.get("role") == "assistant" and m.get("mid") == mr.message_id),
                None,
            )
            if target is None:
                return
            inc = target.setdefault("incoming_reactions", [])
            for e in added:
                inc.append({"from": name, "from_id": mr.user.id, "emoji": e})
            for e in removed:
                for i, rec in enumerate(inc):
                    same_person = (
                        rec.get("from_id") == mr.user.id
                        or (rec.get("from_id") is None and rec.get("from") == name)
                    )
                    if same_person and rec.get("emoji") == e:
                        inc.pop(i)
                        break
            if not inc:
                target.pop("incoming_reactions", None)
            histories[key] = history
            save_history(key)

    for emoji in added:
        analytics_store.record_event(
            "incoming_reaction_added",
            chat_id=chat_id,
            chat_type=mr.chat.type,
            actor_id=mr.user.id,
            actor_name=name,
            actor_kind="owner" if mr.user.id == OWNER_USER_ID else "user",
            message_id=mr.message_id,
            details={"emoji": emoji},
        )
    for emoji in removed:
        analytics_store.record_event(
            "incoming_reaction_removed",
            chat_id=chat_id,
            chat_type=mr.chat.type,
            actor_id=mr.user.id,
            actor_name=name,
            actor_kind="owner" if mr.user.id == OWNER_USER_ID else "user",
            message_id=mr.message_id,
            details={"emoji": emoji},
        )

    if added:
        logger.info(f"💟 [magenta]Реакция на сообщение бота:[/] {' '.join(added)} от {name}")
    if removed:
        logger.info(f"🚫 [dim]Сняли реакцию с сообщения бота:[/] {' '.join(removed)} ({name})")


async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await media_handlers.handle_media(update, context, _media_runtime())


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await media_handlers.handle_video(update, context, _media_runtime())


async def handle_sticker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await media_handlers.handle_sticker(update, context, _media_runtime())


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await media_handlers.handle_voice(update, context, _media_runtime())


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"❌ [bright_red]Глобальная ошибка:[/] {context.error}", exc_info=True)
    try:
        if update and update.effective_message:
            await update.effective_message.reply_text("Произошла ошибка. Попробуйте /clear.")
    except Exception as e:
        logger.error(f"❌ [red]Не удалось отправить сообщение об ошибке:[/] {e}")

    await alerts.notify_owner(
        context.bot,
        category="Unhandled error",
        message=str(context.error)[:500],
        error=context.error,
    )
