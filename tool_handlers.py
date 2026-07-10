"""
Обработчики tool_calls, вынесенные из send_llm_request.

Каждая функция получает мутируемое состояние хода `ToolTurn` и, где нужно,
`payload_messages` / `update` / `context`. Логика перенесена БЕЗ изменений —
это чистая декомпозиция «бог-функции», а не смена поведения.

Общие инварианты (сохранены как были):
- каждый нетерминальный инструмент дописывает ОДНО сообщение role="tool" в payload;
- reply_to_message терминальный: он ничего не пишет в payload, только выставляет
  turn.pending_reply, а отправку делает вызывающий код после цикла.
"""
import json
import asyncio
import logging

from config import STICKER_ENABLED, STICKER_FIND_MAX_PER_TURN
from tools import web_search, read_url, ALLOWED_REACTIONS
from sticker_store import search_stickers

logger = logging.getLogger(__name__)


class ToolTurn:
    """
    Мутируемое состояние обработки инструментов в рамках одного запроса.
    Живёт от начала retry-цикла до отправки ответа (переживает несколько
    витков tool_calls внутри одного send_llm_request).
    """

    def __init__(self):
        self.status_message = None   # статусная плашка поиска/чтения ссылки (переиспользуется)
        self.reacted = False         # бот поставил реакцию — допускаем ответ без текста
        self.reactions_made = []     # [{"emoji", "on"}] — реакции этого хода (пишутся в историю)
        self.sticker_sent = False    # бот отправил стикер — тоже допускаем ответ без текста
        self.stickers_made = []      # [{"desc", "emotion"}] — стикеры этого хода
        self.sticker_candidates = {}  # {номер: {"file_id", "desc", "emotion"}} — из find_stickers
        self.sticker_seq = 0          # сквозная нумерация кандидатов (не сбрасывается между поисками хода)
        self.find_stickers_calls = 0  # сколько раз find_stickers вызвали в этом ходе
        self.pending_reply = None     # (target_mid, text, sid) если модель выбрала reply_to_message


async def handle_web_search(turn, payload_messages, update, tool_call, args):
    await update.message.chat.send_action(action="typing")
    status_text = f"🔍 Выполняю поиск: *{args['query']}*..."

    if turn.status_message is None:
        turn.status_message = await update.message.reply_text(
            status_text,
            parse_mode="Markdown"
        )
    else:
        try:
            await turn.status_message.edit_text(
                status_text,
                parse_mode="Markdown"
            )
        except Exception as e:
            logger.warning(f"⚠️ [yellow]Не удалось отредактировать статусное сообщение:[/] {e}")

    logger.info(f"🔍 [blue]Поиск:[/] {args['query']}")

    search_result = web_search(
        query=args['query'],
        max_results=args.get('max_results', 5),
        timelimit=args.get('timelimit', None),
        region=args.get('region', 'ru-ru')
    )

    if turn.status_message:
        try:
            await turn.status_message.edit_text("🔍 Поиск завершён, обрабатываю результаты...")
        except Exception:
            pass

    logger.debug(f"📄 Результат: {repr(search_result[:200])}")

    if not search_result:
        search_result = "Поиск не дал результатов."

    payload_messages.append({
        "role": "tool",
        "tool_call_id": tool_call['id'],
        "content": search_result
    })


async def handle_read_url(turn, payload_messages, update, tool_call, args):
    url = args.get('url', '')
    await update.message.chat.send_action(action="typing")
    status_text = "🔗 Читаю ссылку..."
    if turn.status_message is None:
        turn.status_message = await update.message.reply_text(status_text)
    else:
        try:
            await turn.status_message.edit_text(status_text)
        except Exception:
            pass

    logger.info(f"🔗 [blue]Чтение ссылки:[/] {url}")
    page_text = read_url(url)
    logger.debug(f"📄 Страница: {repr(page_text[:200])}")

    payload_messages.append({
        "role": "tool",
        "tool_call_id": tool_call['id'],
        "content": page_text
    })


async def handle_find_stickers(turn, payload_messages, update, tool_call, args):
    fquery = (args.get('query') or '').strip()
    try:
        fcount = int(args.get('count') or 6)
    except (TypeError, ValueError):
        fcount = 6
    fcount = max(1, min(fcount, 10))
    if not STICKER_ENABLED:
        tool_result = "Стикеры отключены."
    elif not fquery:
        tool_result = "Пустой запрос. Опиши, какой стикер ищешь."
    elif turn.find_stickers_calls >= STICKER_FIND_MAX_PER_TURN:
        logger.info(
            f"🎨 [dim]find_stickers лимит {STICKER_FIND_MAX_PER_TURN}/ход — отказ[/] "
            f"('{fquery[:60]}')"
        )
        if turn.sticker_candidates:
            tool_result = (
                f"Лимит поиска стикеров в этом ходе ({STICKER_FIND_MAX_PER_TURN}). "
                f"Выбери из уже найденных номеров {min(turn.sticker_candidates)}–"
                f"{max(turn.sticker_candidates)} через send_sticker(id) "
                f"или ответь без стикера."
            )
        else:
            tool_result = (
                f"Лимит поиска стикеров в этом ходе ({STICKER_FIND_MAX_PER_TURN}). "
                f"Ответь без стикера."
            )
    else:
        turn.find_stickers_calls += 1
        try:
            await update.message.chat.send_action(action="choose_sticker")
        except Exception:
            pass
        # Поиск синхронный (эмбеддинг + Qdrant) — уводим в поток.
        cands = await asyncio.to_thread(search_stickers, fquery, fcount)
        remaining = STICKER_FIND_MAX_PER_TURN - turn.find_stickers_calls
        if not cands:
            logger.info(f"🎨 [dim]find_stickers '{fquery}' — ничего выше порога[/]")
            if remaining > 0:
                tool_result = (
                    "Под этот запрос ничего не нашлось. "
                    f"Можешь переформулировать ещё (осталось поисков: {remaining}) "
                    "или ответь без стикера."
                )
            else:
                tool_result = (
                    "Под этот запрос ничего не нашлось. "
                    "Лимит поиска исчерпан — ответь без стикера."
                )
        else:
            lines = []
            for c in cands:
                turn.sticker_seq += 1
                turn.sticker_candidates[turn.sticker_seq] = {
                    "file_id": c.get("file_id"),
                    "desc": c.get("description") or fquery,
                    "emotion": c.get("emotion"),
                }
                # При ВЫБОРЕ показываем всё: эмоцию, полное описание и теги —
                # чтобы модель судила по содержанию, а не по обрывку.
                emo = c.get("emotion") or "—"
                desc = (c.get("description") or "").replace("\n", " ").strip()
                kws = ", ".join(c.get("keywords") or [])
                line = f"#{turn.sticker_seq} [{emo}] {desc}"
                if kws:
                    line += f" | теги: {kws}"
                lines.append(line)
            logger.info(
                f"🎨 [magenta]find_stickers:[/] '{fquery}' → {len(cands)} шт. "
                f"({turn.find_stickers_calls}/{STICKER_FIND_MAX_PER_TURN})"
            )
            refine_hint = (
                f" можно уточнить поиск ещё {remaining} раз(а);"
                if remaining > 0
                else " больше искать нельзя — бери из списка или без стикера;"
            )
            tool_result = (
                "Нашла стикеры (выбери подходящий ПО ОПИСАНИЮ и вызови send_sticker "
                f"с его номером;{refine_hint} если ни один не в тему — не отправляй):\n"
                + "\n".join(lines)
            )

    payload_messages.append({
        "role": "tool",
        "tool_call_id": tool_call['id'],
        "content": tool_result
    })


async def handle_send_sticker(turn, payload_messages, update, context, tool_call, args):
    if not STICKER_ENABLED:
        tool_result = "Стикеры отключены."
    else:
        # Основной путь: id из результатов find_stickers.
        chosen = None
        if args.get('id') is not None:
            try:
                chosen = turn.sticker_candidates.get(int(args.get('id')))
            except (TypeError, ValueError):
                chosen = None
        # Совместимость: если вместо id передали query — разовый подбор лучшего.
        if chosen is None and (args.get('query') or '').strip():
            q = args['query'].strip()
            cands = await asyncio.to_thread(search_stickers, q)
            if cands:
                chosen = {"file_id": cands[0].get("file_id"),
                          "desc": cands[0].get("description") or q,
                          "emotion": cands[0].get("emotion")}
        if not chosen or not chosen.get("file_id"):
            tool_result = ("Не поняла, какой стикер слать. Сначала вызови find_stickers "
                           "и передай номер найденного: send_sticker(id).")
        else:
            thread_id = getattr(update.message, "message_thread_id", None)
            try:
                kw = {"chat_id": update.effective_chat.id, "sticker": chosen["file_id"]}
                if thread_id is not None:
                    kw["message_thread_id"] = thread_id
                await context.bot.send_sticker(**kw)
                turn.sticker_sent = True
                turn.stickers_made.append({"desc": chosen.get("desc"),
                                           "emotion": chosen.get("emotion")})
                logger.info(f"🎨 [magenta]Стикер отправлен[/] «{(chosen.get('desc') or '')[:40]}»")
                # tool_result всё равно пишем (на случай, если ход не терминальный
                # из‑за ошибки порядка), но send_llm_request после sticker_sent
                # завершает ход без нового round-trip к API.
                tool_result = "Стикер отправлен. Ход завершён — дополнительный текст не нужен."
            except Exception as e:
                logger.warning(f"⚠️ [yellow]Не удалось отправить стикер:[/] {e}")
                tool_result = "Стикер отправить не удалось. Ответь текстом."

    payload_messages.append({
        "role": "tool",
        "tool_call_id": tool_call['id'],
        "content": tool_result
    })


def _sid_for_mid(sid_to_mid: dict, mid) -> int | None:
    """Актуальный [#sid] по telegram mid (sid_to_mid = {sid: mid})."""
    if mid is None:
        return None
    for sid, m in (sid_to_mid or {}).items():
        if m == mid:
            return sid
    return None


def _find_existing_bot_reaction(history, turn, react_mid):
    """Эмодзи, которое бот уже ставил на это telegram-сообщение, или None."""
    if react_mid is None:
        return None
    for r in turn.reactions_made or []:
        if r.get("on_mid") == react_mid:
            return r.get("emoji") or "?"
    for m in history or []:
        if m.get("role") != "assistant":
            continue
        for r in m.get("reactions") or []:
            if r.get("on_mid") == react_mid:
                return r.get("emoji") or "?"
    return None


def _quote_for_mid(history, react_mid, current_mid) -> str | None:
    """Короткая цитата user-сообщения (для подсказки модели)."""
    if react_mid is None:
        return None
    for _m in history or []:
        if _m.get("role") == "user" and _m.get("mid") == react_mid:
            _t = (_m.get("content") or "").strip()
            if not _t:
                return None
            return (_t[:40] + "…") if len(_t) > 40 else _t
    return None


async def handle_react(turn, payload_messages, update, context, tool_call, args, sid_to_mid, history):
    # Модель часто шлёт эмодзи с вариативным селектором U+FE0F (❤️),
    # а Telegram и ALLOWED_REACTIONS хранят каноничную форму без него (❤).
    # Срезаем FE0F, иначе сердце/☃/✍/🕊/❤‍🔥/🤷‍♂ молча не проходят валидацию.
    emoji = (args.get('emoji') or '').strip().replace(chr(0xFE0F), '')
    try:
        react_sid = int(args.get('id'))
    except (TypeError, ValueError):
        react_sid = None
    react_mid = sid_to_mid.get(react_sid) if react_sid is not None else None
    if react_mid is None:
        react_mid = update.message.message_id
        # Если id не передали / протух — это реакция на текущее; подтянем его sid.
        if react_sid is None:
            react_sid = _sid_for_mid(sid_to_mid, react_mid)

    if emoji not in ALLOWED_REACTIONS:
        tool_result = f"Эмодзи '{emoji}' не разрешён Telegram. Ответь текстом или выбери из списка."
    else:
        already = _find_existing_bot_reaction(history, turn, react_mid)
        if already:
            # Не зовём Telegram повторно — модель часто «забывает» прошлую реакцию.
            sid_hint = f" [#{react_sid}]" if react_sid is not None else ""
            logger.info(
                f"😀 [dim]Реакция уже была[/] {already} на mid={react_mid}"
                f"{sid_hint}; повтор {emoji} отклонён"
            )
            tool_result = (
                f"Ты УЖЕ поставила реакцию {already} на это сообщение"
                f"{sid_hint}. Повторно ставить нельзя (даже другим эмодзи). "
                f"Сделай что-то другое: ответь текстом, поставь реакцию на "
                f"ДРУГОЕ сообщение (другой [#N]), отправь стикер — или промолчи, "
                f"если добавить нечего."
            )
            # Уже реагировали раньше — ход «с реакцией» валиден (можно без текста).
            turn.reacted = True
        else:
            try:
                await context.bot.set_message_reaction(
                    chat_id=update.effective_chat.id,
                    message_id=react_mid,
                    reaction=emoji
                )
                turn.reacted = True
                # Стабильный якорь — telegram mid (не меняется). [#sid] при рендере
                # резолвим из живой истории, чтобы после renumber не врать модели.
                on_quote = _quote_for_mid(history, react_mid, update.message.message_id)
                turn.reactions_made.append({
                    "emoji": emoji,
                    "on_mid": react_mid,
                    "on_sid": react_sid,  # снимок на момент хода (для логов/старых записей)
                    "on": on_quote,
                })
                logger.info(
                    f"😀 [magenta]Реакция:[/] {emoji} → "
                    f"[#{react_sid if react_sid is not None else 'текущее'}] (mid={react_mid})"
                )
                tool_result = (
                    f"Реакция {emoji} поставлена"
                    f"{f' на [#{react_sid}]' if react_sid is not None else ''}. "
                    f"Повторно на это же сообщение не ставь. "
                    f"Если добавить нечего — можешь обойтись без текста."
                )
            except Exception as e:
                logger.warning(f"⚠️ [yellow]Не удалось поставить реакцию {emoji}:[/] {e}")
                tool_result = "Не удалось поставить реакцию, ответь текстом."

    payload_messages.append({
        "role": "tool",
        "tool_call_id": tool_call['id'],
        "content": tool_result
    })


def handle_reply(turn, update, args, sid_to_mid):
    """
    Терминальный инструмент. Только выставляет turn.pending_reply —
    отправку и запись в историю делает вызывающий код после цикла,
    поэтому tool-result в payload НЕ добавляется (нового запроса к API не будет).
    """
    try:
        reply_sid = int(args.get('id'))
    except (TypeError, ValueError):
        reply_sid = None
    reply_text = args.get('text', '') or ''
    reply_mid = sid_to_mid.get(reply_sid)
    if reply_mid is None:
        # Невалидный/устаревший [#N] — отвечаем на текущее сообщение
        reply_mid = update.message.message_id
    turn.pending_reply = (reply_mid, reply_text, reply_sid)


async def dispatch_tool_call(turn, payload_messages, update, context, tool_call, sid_to_mid, history):
    """
    Разбирает один tool_call и направляет в нужный обработчик.
    Полностью повторяет прежнюю if/elif-цепочку из send_llm_request.
    """
    func_name = tool_call['function']['name']
    args = json.loads(tool_call['function']['arguments'])

    if func_name == 'web_search':
        await handle_web_search(turn, payload_messages, update, tool_call, args)
    elif func_name == 'read_url':
        await handle_read_url(turn, payload_messages, update, tool_call, args)
    elif func_name == 'find_stickers':
        await handle_find_stickers(turn, payload_messages, update, tool_call, args)
    elif func_name == 'send_sticker':
        await handle_send_sticker(turn, payload_messages, update, context, tool_call, args)
    elif func_name == 'react_to_message':
        await handle_react(turn, payload_messages, update, context, tool_call, args, sid_to_mid, history)
    elif func_name == 'reply_to_message':
        handle_reply(turn, update, args, sid_to_mid)
    else:
        # Неизвестный инструмент — всё равно отвечаем, иначе API упадёт
        payload_messages.append({
            "role": "tool",
            "tool_call_id": tool_call['id'],
            "content": f"Инструмент '{func_name}' не поддерживается."
        })
