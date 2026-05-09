import logging
import asyncio
from datetime import datetime
from telegram import Update
from telegram.ext import ContextTypes

from config import (
    ADMIN_MODE, RANDOM_REPLY_CHANCE, SUMMARY_INTERVAL, MAX_CONTEXT_TOKENS,
    ALLOWED_USERS, ALLOWED_GROUPS, VISION_MODE
)
from state import histories, get_history_key, message_buffer, chat_tokens
import config
from llm_client import summarize_history, send_llm_request
from utils import escape_user_text, is_bot_mentioned, should_reply_randomly, download_media_as_base64

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
            f"Я запоминаю нашу беседу.\n\n"
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
        f"Диалогов: {msg_count // 2}\n"
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
            
        message_content = " ".join(message_parts)

    if key not in histories:
        histories[key] = []
    
    history = histories[key]
    
    # Для картинок логика чуть сложнее. Если были картинки, добавим их.
    # В текущей реализации мы склеиваем текст. Картинки тоже нужно учесть.
    content_parts = []
    has_image = any(m["image_url"] for m in messages)
    
    if has_image:
        content_parts.append({"type": "text", "text": message_content})
        for m in messages:
            if m["image_url"]:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": m["image_url"]}
                })
        history.append({"role": "user", "content": content_parts})
    else:
        history.append({"role": "user", "content": message_content})

    histories[key] = history

    if not (mentioned or random_reply):
        return

    await update.message.chat.send_action(action="typing")
    await send_llm_request(update, context, key, history, user_name, user_id, is_group, random_reply, reason)


async def queue_message(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str, image_url: str = None):
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
        "image_url": image_url,
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
        # Если Vision Mode выключен, мы все равно должны обрабатывать текст из картинки
        if caption:
            await queue_message(update, context, text=caption)
        return  

    try:
        photo = update.message.photo[-1]
        b64, mime = await download_media_as_base64(photo.file_id, context)
        image_url = f"data:{mime};base64,{b64}"
    except Exception as e:
        logger.error(f"Ошибка загрузки фото: {e}")
        image_url = None

    await queue_message(update, context, text=caption, image_url=image_url)


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"❌ Глобальная ошибка: {context.error}", exc_info=True)
    try:
        if update and update.effective_message:
            await update.effective_message.reply_text("Произошла ошибка. Попробуйте /clear.")
    except Exception as e:
        logger.error(f"Не удалось отправить сообщение об ошибке: {e}")
