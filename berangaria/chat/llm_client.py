import logging
import asyncio
import random
import httpx
from telegram import Update
from telegram.ext import ContextTypes

from berangaria.config import (
    CHAT_API_URL, SUMMARY_INTERVAL as _SUMMARY_INTERVAL, VISION_MODE,
    MAX_CONTEXT_TOKENS,
    MAX_REPLY_TOKENS, MODEL, GENERATION_PARAMS, FACTUAL_TEMPERATURE, FULL_DEBUG_LOGS,
    PRICE_PROMPT_CACHE_MISS, PRICE_PROMPT_CACHE_HIT, PRICE_PROMPT_CACHE_WRITE,
    PRICE_COMPLETION,
    MEMORY_SEARCH_LIMIT, MEMORY_MIN_SCORE, MEMORY_MAX_CHARS,
    MEMORY_QUERY_MIN_CHARS, MEMORY_QUERY_RECENT_MESSAGES, MAX_API_RETRIES,
    MAX_TOOL_ROUNDS, STREAMING_ENABLED, STREAM_UPDATE_INTERVAL_SECONDS,
    STREAM_PREVIEW_MIN_CHARS,
    MULTI_MESSAGE_DELAY_MIN, MULTI_MESSAGE_DELAY_MAX, MULTI_MESSAGE_DELAY_TOTAL_CAP,
    MULTI_MESSAGE_CHARS_PER_SEC, chat_api_headers, apply_chat_routing,
)
from berangaria.prompts import SYSTEM_PROMPT, VISION_PROMPT_SUFFIX
from berangaria.core.state import histories, chat_tokens, api_call_count, get_history_lock, save_history
from berangaria.memory import store as memory_store
from berangaria.core import state
from berangaria.tools.schemas import TOOLS
from berangaria.tools.dispatch import ToolTurn, dispatch_tool_call
from berangaria.chat.streaming import stream_chat_completion
from berangaria.chat import (
    assistant_turn,
    completion_transport,
    llm_diagnostics,
    memory_context,
    reply_delivery,
    summarization,
)
from berangaria.chat.history_rendering import (
    build_sid_map as _build_sid_map,
    extract_plain_text as _extract_plain_text,
    render_history_for_api as _render_history_for_api,
    renumber_sids,
)
from berangaria.chat.reply_formatting import (
    clean_reply as _clean_reply,
    is_parse_error as _is_parse_error,
    markdown_to_html as _markdown_to_html,
    split_for_telegram,
    strip_markdown as _strip_markdown,
)
from berangaria.core.utils import now_local

_renumber_sids = renumber_sids
SUMMARY_INTERVAL = _SUMMARY_INTERVAL
markdown_to_html = _markdown_to_html
strip_markdown = _strip_markdown
_split_for_telegram = split_for_telegram

logger = logging.getLogger(__name__)


def _message_reasoning_len(message: dict) -> int:
    """Length of provider reasoning, whether DeepSeek- or OpenAI-shaped."""
    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str) and reasoning_content:
        return len(reasoning_content)
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        return len(reasoning)
    if isinstance(reasoning, dict):
        text = reasoning.get("content") or reasoning.get("text") or ""
        return len(text) if isinstance(text, str) else 0
    return 0


def _estimate_request_cost(
    usage: dict,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
    cache_write_tokens: int,
) -> float:
    """Prefer the provider's billed `usage.cost`; otherwise estimate from prices."""
    billed = usage.get("cost")
    if billed is not None:
        try:
            return float(billed)
        except (TypeError, ValueError):
            pass

    cached = max(0, int(cached_tokens or 0))
    written = max(0, int(cache_write_tokens or 0))
    uncached = max(0, int(prompt_tokens or 0) - cached - written)
    return (
        (uncached / 1_000_000) * PRICE_PROMPT_CACHE_MISS
        + (cached / 1_000_000) * PRICE_PROMPT_CACHE_HIT
        + (written / 1_000_000) * PRICE_PROMPT_CACHE_WRITE
        + (int(completion_tokens or 0) / 1_000_000) * PRICE_COMPLETION
    )


ReplyDeliveryError = reply_delivery.ReplyDeliveryError


_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTHS = ["January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December"]


def _current_time_str() -> str:
    """Формирует строку с текущей датой и временем суток для системного промпта (МСК)."""
    now = now_local()
    time_str = f"Today is {_DAYS[now.weekday()]}, {now.day} {_MONTHS[now.month-1]} {now.year} year. "

    if 5 <= now.hour < 12:
        time_of_day = "morning"
    elif 12 <= now.hour < 17:
        time_of_day = "daytime"
    elif 17 <= now.hour < 23:
        time_of_day = "evening"
    else:
        time_of_day = "night"

    time_str += f"Times of Day: {time_of_day}."
    return time_str


def _build_system_prompt() -> str:
    """Собирает полный системный промпт: база + vision (если включён) + текущее время."""
    system_prompt = SYSTEM_PROMPT
    if VISION_MODE:
        system_prompt += VISION_PROMPT_SUFFIX
    system_prompt += f"\n\n=== CURRENT TIME ===\n{_current_time_str()}\n"
    return system_prompt


def _multi_message_delay_seconds(text: str, *, slept_total: float = 0.0) -> float:
    """Пауза перед следующим bubble: длина + jitter, с общим потолком на ход."""
    remaining = MULTI_MESSAGE_DELAY_TOTAL_CAP - slept_total
    if remaining <= 0:
        return 0.0
    base = len(text or "") / MULTI_MESSAGE_CHARS_PER_SEC
    delay = max(MULTI_MESSAGE_DELAY_MIN, min(MULTI_MESSAGE_DELAY_MAX, base))
    delay *= random.uniform(0.85, 1.15)
    return max(0.0, min(delay, remaining))


def _is_meaningful_memory_query(text: str) -> bool:
    return memory_context.is_meaningful_query(
        text, min_chars=MEMORY_QUERY_MIN_CHARS
    )


def _build_memory_search_query(history: list, user_name: str) -> str:
    return memory_context.build_search_query(
        history,
        user_name,
        min_chars=MEMORY_QUERY_MIN_CHARS,
        recent_messages=MEMORY_QUERY_RECENT_MESSAGES,
        extract_plain_text=_extract_plain_text,
    )


def _build_memory_relevance_query(history: list, user_name: str) -> str:
    return memory_context.build_relevance_query(
        history,
        user_name,
        min_chars=MEMORY_QUERY_MIN_CHARS,
        extract_plain_text=_extract_plain_text,
    )


_memory_terms = memory_context._memory_terms
_is_general_memory_recall = memory_context.is_general_recall
_memory_fact_matches_query = memory_context._fact_matches_query


def _approved_memory_recall_results(scope: str) -> dict:
    return memory_context.approved_recall_results(
        scope, search_limit=MEMORY_SEARCH_LIMIT
    )


def _format_memory_block(mem_results: dict, query: str = "") -> str:
    return memory_context.format_memory_block(
        mem_results,
        query,
        min_score=MEMORY_MIN_SCORE,
        max_chars=MEMORY_MAX_CHARS,
        search_limit=MEMORY_SEARCH_LIMIT,
    )


_count_memory_block_facts = memory_context.count_memory_block_facts
_filter_approved_memory_results = memory_context.filter_approved_results


async def summarize_history(history: list) -> list:
    return await summarization.summarize_history(history)


async def send_llm_request(
    update: Update, context: ContextTypes.DEFAULT_TYPE, key: str,
    history: list, user_name: str, user_id: int, mentioned: bool = False):

    # Автосуммаризация при достижении 85% от лимита токенов
    context_threshold = int(MAX_CONTEXT_TOKENS * 0.85)
    if chat_tokens.get(key, 0) > context_threshold:
        logger.info(f"📝 [yellow]Автосуммаризация[/] для key={key}")
        history = await summarize_history(history)
        async with get_history_lock(key):
            histories[key] = history
            save_history(key)

    system_prompt = _build_system_prompt()
    # В payload подставляем reply-хэндлы [#N] (только в копию, история остаётся чистой)
    payload_messages = [{"role": "system", "content": system_prompt}] + _render_history_for_api(history)
    sid_to_mid = _build_sid_map(history)

    try:
        # Валидация ключа для безопасности
        if not state.is_valid_memory_scope(key):
            logger.warning(f"⚠️ [yellow]Невалидный ключ памяти:[/] {key}")
        else:
            query = _build_memory_search_query(history, user_name)
            if not query:
                if FULL_DEBUG_LOGS:
                    logger.debug(f"🔍 Mem0 поиск пропущен: нет содержательного query (scope={key})")
            else:
                relevance_query = _build_memory_relevance_query(history, user_name)
                if _is_general_memory_recall(relevance_query):
                    approved_results = _approved_memory_recall_results(key)
                    results_count = len(approved_results["results"])
                    if FULL_DEBUG_LOGS:
                        logger.debug(f"🔍 Память: общий recall из SQLite (scope={key})")
                elif memory_store.memory:
                    if FULL_DEBUG_LOGS:
                        logger.debug(f"🔍 Mem0 поиск: query='{query[:80]}', scope={key}")

                    # Уменьшен таймаут до 15 секунд для быстрого ответа
                    mem_results = await asyncio.wait_for(
                        asyncio.to_thread(
                            memory_store.memory.search,
                            query,
                            filters={"user_id": key},
                            limit=MEMORY_SEARCH_LIMIT
                        ),
                        timeout=15.0
                    )

                    results_count = len(mem_results.get('results', []))
                    approved_results = _filter_approved_memory_results(mem_results, key)
                else:
                    approved_results = {"results": []}
                    results_count = 0
                    if FULL_DEBUG_LOGS:
                        logger.debug(f"🔍 Mem0 поиск пропущен: хранилище недоступно (scope={key})")

                mem_text = _format_memory_block(
                    approved_results,
                    query=relevance_query,
                )

                if mem_text and payload_messages[-1]["role"] == "user":
                    last_content = payload_messages[-1]["content"]
                    payload_messages[-1] = {
                        "role": "user",
                        "content": f"{last_content}\n\n[Context from memory:\n{mem_text}\n]"
                    }
                    facts_count = _count_memory_block_facts(mem_text)

                    # Краткий лог для INFO, детальный для DEBUG
                    logger.info(f"🧠 Память: найдено {results_count} → загружено {facts_count} фактов ({len(mem_text)} символов)")

                    if FULL_DEBUG_LOGS:
                        logger.debug(f"📝 Факты:\n{mem_text}")

    except asyncio.TimeoutError:
        logger.warning(f"⚠️ [yellow]Память: таймаут поиска (15s), продолжаем без неё[/] scope={key}")
    except Exception as e:
        logger.error(f"⚠️ [red]Ошибка получения памяти:[/] {e}")

    # Мутируемое состояние хода (статусная плашка, реакции, стикеры, pending_reply) —
    # см. tool_handlers.ToolTurn. Живёт весь retry-цикл.
    turn = ToolTurn()
    used_tool = False  # после вызова инструмента (поиск/ссылка) отвечаем с пониженной температурой

    async def _request_completion(client, payload, headers):
        runtime = completion_transport.CompletionRuntime(
            update=update,
            context=context,
            mentioned=mentioned,
            api_url=CHAT_API_URL,
            streaming_enabled=STREAMING_ENABLED,
            update_interval_seconds=STREAM_UPDATE_INTERVAL_SECONDS,
            preview_min_chars=STREAM_PREVIEW_MIN_CHARS,
            stream_chat_completion=stream_chat_completion,
        )
        return await completion_transport.request_completion(
            client, payload, headers, turn, runtime
        )

    async def _delete_turn_status():
        await reply_delivery.delete_turn_status(turn)

    def _delivery_runtime() -> reply_delivery.DeliveryRuntime:
        return reply_delivery.DeliveryRuntime(
            update=update,
            context=context,
            clean_reply=_clean_reply,
            is_parse_error=_is_parse_error,
            multi_message_delay_seconds=_multi_message_delay_seconds,
        )

    async def _deliver(text: str, target_mid, status_msg):
        return await reply_delivery.deliver(
            text, target_mid, status_msg, _delivery_runtime()
        )

    async def _deliver_multi(messages: list[str], target_mid, status_msg):
        return await reply_delivery.deliver_multi(
            messages, target_mid, status_msg, _delivery_runtime()
        )

    async def _save_assistant(text: str):
        return await assistant_turn.save_assistant_turn(
            text, turn=turn, key=key, history=history
        )

    async def _remember_bot_mid(entry, sent_mid):
        await assistant_turn.remember_bot_message_id(entry, sent_mid, key=key)

    async with httpx.AsyncClient(timeout=600.0) as client:
        if FULL_DEBUG_LOGS:
            llm_diagnostics.log_request(payload_messages, enabled=True)

        forced_answer_nudge = False  # один раз подтолкнём ответить, если промолчала при прямом обращении

        api_failures = 0
        tool_rounds = 0
        while True:
            gen_params = dict(GENERATION_PARAMS)
            if used_tool:
                # Факты после поиска/чтения ссылки — холоднее, меньше выдумок
                gen_params["temperature"] = FACTUAL_TEMPERATURE

            payload = apply_chat_routing({
                "model": MODEL,
                "messages": payload_messages,
                "max_tokens": MAX_REPLY_TOKENS,
                "tools": TOOLS,
                **gen_params
            })

            try:
                headers = chat_api_headers()
                
                response = await _request_completion(client, payload, headers)

                if response.status_code == 400:
                    await _delete_turn_status()
                    async with get_history_lock(key):
                        histories[key] = []
                        save_history(key)
                    logger.error(f"[red]400:[/] {response.text}")
                    await update.message.reply_text("⚠️ История сброшена. Напишите ещё раз.")
                    return

                # Обработка rate limiting
                if response.status_code == 429:
                    api_failures += 1
                    if api_failures >= MAX_API_RETRIES:
                        await _delete_turn_status()
                        await update.message.reply_text("❌ API временно перегружен. Попробуйте позже.")
                        return
                    try:
                        retry_after = min(
                            60.0,
                            max(1.0, float(response.headers.get("Retry-After", 5))),
                        )
                    except (TypeError, ValueError):
                        retry_after = 5.0
                    logger.warning(
                        f"⚠️ [yellow]Rate limit (429), ждём {retry_after:g}s перед retry "
                        f"{api_failures}/{MAX_API_RETRIES}[/]"
                    )
                    await asyncio.sleep(retry_after)
                    continue

                if response.status_code != 200:
                    logger.error(f"❌ [red]API error {response.status_code}:[/] {response.text[:200]}")
                    api_failures += 1
                    if api_failures < MAX_API_RETRIES:
                        await asyncio.sleep(2 ** (api_failures - 1))
                        continue
                    await _delete_turn_status()
                    await update.message.reply_text(f"❌ Ошибка API: {response.status_code}")
                    return

                data = response.json()
                choice = data['choices'][0]
                finish_reason = choice.get('finish_reason', '')
                message = choice['message']
                usage = data.get('usage', {})

                if usage:
                    llm_diagnostics.record_usage(
                        usage,
                        key=key,
                        chat_tokens=chat_tokens,
                        estimate_request_cost=_estimate_request_cost,
                    )
                if finish_reason == 'tool_calls' and message.get('tool_calls'):
                    tool_rounds += 1
                    if tool_rounds > MAX_TOOL_ROUNDS:
                        logger.error(f"❌ Превышен лимит tool-call раундов ({MAX_TOOL_ROUNDS})")
                        if turn.status_message:
                            try:
                                await turn.status_message.delete()
                            except Exception:
                                pass
                        await update.message.reply_text(
                            "❌ Не удалось завершить обработку инструментов. Попробуйте переформулировать запрос."
                        )
                        return
                    payload_messages.append(message)
                    # The cold temperature is only right where the reply retells
                    # facts that were looked up. It used to be switched on by ANY
                    # tool, so a sticker search or a reaction silently flattened the
                    # rest of the turn — punishing the model for exactly the
                    # behaviour the prompt is trying to encourage.
                    if any(
                        (tc.get("function") or {}).get("name") in ("web_search", "read_url")
                        for tc in message["tool_calls"]
                    ):
                        used_tool = True
                    turn.pending_reply = None  # (target_mid, text, sid) если модель выбрала reply_to_message
                    turn.pending_messages = None  # list[str] если send_messages

                    for tool_call in message['tool_calls']:
                        try:
                            await dispatch_tool_call(
                                turn, payload_messages, update, context,
                                tool_call, sid_to_mid, history,
                            )
                        except Exception as exc:
                            logger.error(f"❌ Ошибка инструмента: {exc}", exc_info=True)
                            payload_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.get("id", ""),
                                "content": f"Инструмент завершился ошибкой: {exc}",
                            })

                    # reply_to_message терминальный и приоритетный: если модель его вызвала,
                    # отправляем выбранный ответ и завершаем — без ещё одного витка к API
                    # и без дефолтного реплая ниже (двойной отправки не будет).
                    if turn.pending_reply is not None:
                        reply_mid, reply_text, reply_sid = turn.pending_reply
                        try:
                            reply_text = _clean_reply(reply_text)
                        except Exception as exc:
                            # Ход терминальный: payload_messages уже содержит
                            # assistant-сообщение с tool_calls, на которые нет
                            # ответов. Уронить это в общий retry-обработчик значит
                            # переотправить такой payload, получить 400 и стереть
                            # историю чата. Завершаем ход здесь.
                            logger.error(
                                f"❌ Не удалось подготовить текст реплая: {exc}",
                                exc_info=True,
                            )
                            await _delete_turn_status()
                            if turn.reactions_made or turn.stickers_made or turn.voices_made:
                                await _save_assistant("")
                            return
                        api_call_count[key] = api_call_count.get(key, 0) + 1
                        if reply_text:
                            logger.info(f"↩️ [magenta]Ответ реплаем на[/] [#{reply_sid}]")
                            try:
                                sent_mid = await _deliver(reply_text, reply_mid, turn.status_message)
                            except Exception as exc:
                                logger.error(f"❌ Не удалось доставить ответ: {exc}", exc_info=True)
                                if turn.reactions_made or turn.stickers_made or turn.voices_made:
                                    await _save_assistant("")
                                raise ReplyDeliveryError(
                                    "Telegram не подтвердил доставку ответа"
                                ) from exc
                            saved = await _save_assistant(reply_text)
                            await _remember_bot_mid(saved, sent_mid)
                        else:
                            # reply_to_message без текста — отправлять нечего (пустых сообщений не шлём)
                            if turn.reacted or turn.sticker_sent or turn.voice_sent:
                                await _save_assistant("")  # реакция/стикер/голос, текста нет
                            elif not mentioned:
                                logger.info("🤫 [dim]Промолчала (ambient, reply без текста)[/]")
                            else:
                                logger.warning("⚠️ [yellow]reply_to_message без текста при прямом обращении[/]")
                            if turn.status_message:
                                try:
                                    await turn.status_message.delete()
                                except Exception:
                                    pass
                        return

                    # send_messages — terminal burst (2–3 short bubbles + typing pauses).
                    if turn.pending_messages:
                        messages = list(turn.pending_messages)
                        turn.pending_messages = None
                        api_call_count[key] = api_call_count.get(key, 0) + 1
                        target_mid = update.message.message_id if mentioned else None
                        logger.info(
                            f"💬 [magenta]Серия из {len(messages)} сообщений[/]"
                        )
                        try:
                            sent_mid, delivered = await _deliver_multi(
                                messages, target_mid, turn.status_message
                            )
                        except Exception as exc:
                            logger.error(
                                f"❌ Не удалось доставить серию сообщений: {exc}",
                                exc_info=True,
                            )
                            if turn.reactions_made or turn.stickers_made or turn.voices_made:
                                await _save_assistant("")
                            raise ReplyDeliveryError(
                                "Telegram не подтвердил доставку серии сообщений"
                            ) from exc
                        if delivered:
                            # One history row: joined bubbles (prefix cache / summarizer).
                            history_text = "\n".join(delivered)
                            saved = await _save_assistant(history_text)
                            await _remember_bot_mid(saved, sent_mid)
                        elif turn.reactions_made or turn.stickers_made or turn.voices_made:
                            await _save_assistant("")
                        return

                    # Стикер уже в чате — это полный ответ, лишний round-trip к API не нужен.
                    if turn.sticker_sent:
                        api_call_count[key] = api_call_count.get(key, 0) + 1
                        await _save_assistant("")
                        logger.info("🎨 [dim]Ход завершён стикером (без доп. текста)[/]")
                        if turn.status_message:
                            try:
                                await turn.status_message.delete()
                            except Exception:
                                pass
                        return

                    # Голосовое уже в чате — терминальный ответ, без ещё одного round-trip.
                    if turn.voice_sent:
                        api_call_count[key] = api_call_count.get(key, 0) + 1
                        await _save_assistant("")
                        logger.info("🔊 [dim]Ход завершён голосовым (без доп. текста)[/]")
                        if turn.status_message:
                            try:
                                await turn.status_message.delete()
                            except Exception:
                                pass
                        return

                    continue

                reply = message.get('content', '')
                
                # В DEBUG показываем полный ответ модели
                if FULL_DEBUG_LOGS:
                    llm_diagnostics.log_response(
                        reply, finish_reason, enabled=True
                    )
                # Увеличиваем счётчик вызовов API
                api_call_count[key] = api_call_count.get(key, 0) + 1

                reply = _clean_reply(reply)

                if not reply:
                    if turn.reacted or turn.sticker_sent or turn.voice_sent:
                        # Ограничилась реакцией/стикером/голосом — валидный ответ.
                        await _save_assistant("")
                        if turn.status_message:
                            try:
                                await turn.status_message.delete()
                            except Exception:
                                pass
                        return
                    if not mentioned:
                        # Ambient-пинг: к Ber не обращались — осознанное молчание, это норма.
                        logger.info(f"🤫 [dim]Промолчала (ambient)[/] (ключ={key})")
                        if turn.status_message:
                            try:
                                await turn.status_message.delete()
                            except Exception:
                                pass
                        return
                    # Прямое обращение / личка / событие — молчать нельзя. Один раз подталкиваем ответить.
                    if not forced_answer_nudge:
                        forced_answer_nudge = True
                        payload_messages.append({
                            "role": "system",
                            "content": "Тебе адресовали сообщение напрямую — нельзя молчать. Дай короткий ответ в своём стиле."
                        })
                        logger.info(f"↩️ [yellow]Пустой ответ при прямом обращении — подталкиваю ответить[/] (ключ={key})")
                        continue
                    logger.warning(f"⚠️ [yellow]Пустой ответ при прямом обращении даже после напоминания[/] (ключ={key})")
                    if turn.status_message:
                        try:
                            await turn.status_message.delete()
                        except Exception:
                            pass
                    return

                if finish_reason == 'length':
                    logger.warning(f"⚠️ [yellow]Ответ обрезан по лимиту токенов[/] (ключ={key})")
                    reply += "\n\n_(ответ обрезан)_"

                # Долговременная память обрабатывается отдельно из SQLite-очереди;
                # доставка ответа не зависит от extractor/verifier.

                # Модель не выбрала инструмент reply_to_message:
                # адресное обращение → реплай на триггер (как раньше);
                # ambient (случайный пинг) → обычное сообщение без reply.
                target_mid = update.message.message_id if mentioned else None
                try:
                    sent_mid = await _deliver(reply, target_mid, turn.status_message)
                except Exception as exc:
                    logger.error(f"❌ Не удалось доставить ответ: {exc}", exc_info=True)
                    if turn.reactions_made or turn.stickers_made or turn.voices_made:
                        await _save_assistant("")
                    raise ReplyDeliveryError(
                        "Telegram не подтвердил доставку ответа"
                    ) from exc
                saved = await _save_assistant(reply)
                await _remember_bot_mid(saved, sent_mid)
                return

            except ReplyDeliveryError:
                # Доставка — часть логической транзакции хода. Не повторяем LLM/tools
                # и даём debounce-слою оставить memory sources в waiting.
                raise
            except httpx.ConnectError:
                logger.error("❌ [bright_red]API недоступен![/]")
                api_failures += 1
                if api_failures < MAX_API_RETRIES:
                    await asyncio.sleep(2 ** (api_failures - 1))
                    continue
                await _delete_turn_status()
                await update.message.reply_text("❌ API недоступен!")
                return
            except httpx.TimeoutException:
                logger.error("❌ [bright_red]Таймаут запроса к API[/]")
                api_failures += 1
                if api_failures < MAX_API_RETRIES:
                    await asyncio.sleep(2 ** (api_failures - 1))
                    continue
                await _delete_turn_status()
                await update.message.reply_text("❌ Таймаут.")
                return
            except Exception as e:
                logger.error(f"❌ [bright_red]Ошибка в обработке запроса:[/] {e}", exc_info=True)
                api_failures += 1
                if api_failures < MAX_API_RETRIES:
                    await asyncio.sleep(2 ** (api_failures - 1))
                    continue
                await _delete_turn_status()
                await update.message.reply_text("❌ Ошибка при обработке.")
                return
