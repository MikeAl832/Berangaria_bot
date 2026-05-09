import re
import time
import random
import base64
from telegram import Update
from telegram.ext import ContextTypes
from config import BOT_NAMES, RANDOM_REPLY_CHANCE
from state import random_reply_cooldown

def get_bot_real_name(context: ContextTypes.DEFAULT_TYPE) -> str:
    return context.bot.first_name

def should_reply_randomly(chat_id: int) -> bool:
    last_reply = random_reply_cooldown.get(chat_id, 0)
    current_time = time.time()
    if current_time - last_reply < 30:
        return False
    if random.randint(1, 100) <= RANDOM_REPLY_CHANCE:
        random_reply_cooldown[chat_id] = current_time
        return True
    return False

def is_bot_mentioned(update: Update, context: ContextTypes.DEFAULT_TYPE) -> tuple:
    if update.message is None:
        return False, None

    message_text = update.message.text or update.message.caption or ""
    bot_real_name = get_bot_real_name(context)
    bot_username = context.bot.username

    if update.message.reply_to_message:
        if update.message.reply_to_message.from_user.id == context.bot.id:
            return True, "reply"

    if not message_text:
        return False, None

    if message_text.startswith('/'):
        return True, "команда"

    if f"@{bot_username}" in message_text:
        return True, f"@{bot_username}"

    if re.search(rf'\b{re.escape(bot_real_name)}\b', message_text, re.IGNORECASE):
        return True, bot_real_name

    for name in BOT_NAMES:
        if re.search(rf'\b{re.escape(name)}\b', message_text, re.IGNORECASE):
            return True, name

    return False, None

async def download_media_as_base64(file_id: str, context: ContextTypes.DEFAULT_TYPE) -> tuple[str, str]:
    file = await context.bot.get_file(file_id)
    path = file.file_path.lower()

    if path.endswith(('.jpg', '.jpeg')):
        mime = "image/jpeg"
    elif path.endswith('.png'):
        mime = "image/png"
    elif path.endswith('.webp'):
        mime = "image/webp"
    elif path.endswith('.gif'):
        mime = "image/gif"
    else:
        mime = "image/jpeg" 

    buf = bytearray()
    await file.download_as_bytearray(buf)
    b64 = base64.b64encode(bytes(buf)).decode('utf-8')
    return b64, mime

def escape_user_text(text: str) -> str:
    return text.replace('[', '(').replace(']', ')')
