# Глобальные состояния бота

# История сообщений по ключу (private_X или group_Y)
histories = {}

# Тайм-ауты для рандомных ответов
random_reply_cooldown = {}

# Последнее известное количество токенов для чата (из API)
chat_tokens = {}

# Буфер сообщений для склеивания (debounce)
# Формат: { "chat_id_user_id": { "messages": [...], "task": asyncio.Task } }
message_buffer = {}

def get_history_key(chat_id: int, is_private: bool, user_id: int = None) -> str:
    if is_private:
        return f"private_{user_id}"
    return f"group_{chat_id}"
