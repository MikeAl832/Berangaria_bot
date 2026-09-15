import logging
import asyncio
import copy
import hashlib
import json
import random
import httpx
from telegram import Update
from telegram.ext import ContextTypes

from berangaria.config import (
    CHAT_API_URL, SUMMARY_INTERVAL as _SUMMARY_INTERVAL, VISION_MODE,
    MAX_CONTEXT_TOKENS,
    MAX_REPLY_TOKENS, MODEL, GENERATION_PARAMS, FULL_DEBUG_LOGS,
    PRICE_PROMPT_CACHE_MISS, PRICE_PROMPT_CACHE_HIT, PRICE_PROMPT_CACHE_WRITE,
    PRICE_COMPLETION, CHAT_PROVIDER,
    MEMORY_SEARCH_LIMIT, MEMORY_MIN_SCORE, MEMORY_MAX_CHARS,
    MEMORY_QUERY_MIN_CHARS, MEMORY_QUERY_RECENT_MESSAGES, MAX_API_RETRIES,
    MAX_TOOL_ROUNDS, STREAMING_ENABLED, STREAM_UPDATE_INTERVAL_SECONDS,
    STREAM_PREVIEW_MIN_CHARS,
    MULTI_MESSAGE_DELAY_MIN, MULTI_MESSAGE_DELAY_MAX, MULTI_MESSAGE_DELAY_TOTAL_CAP,
    MULTI_MESSAGE_CHARS_PER_SEC, chat_api_headers, apply_chat_gateway,
)
from berangaria.analytics import store as analytics_store
from berangaria.prompts import SYSTEM_PROMPT, VISION_PROMPT_SUFFIX
from berangaria.core.state import histories, chat_tokens, api_call_count, get_history_lock, save_history
from berangaria.memory import store as memory_store
from berangaria.core import state
from berangaria.core import alerts
from berangaria.tools.schemas import TOOLS
from berangaria.tools.dispatch import ToolTurn, available_tools_for_turn, dispatch_tool_call
from berangaria.chat.streaming import stream_chat_completion
from berangaria.chat.chat_actions import ChatActionHeartbeat, effective_message_thread_id
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


def _safe_error_value(value, *, max_chars=160):
    """Compact one provider error field for logs without dumping the raw body."""
    text = " ".join(str(value or "unknown").split())
    return text[:max_chars]


def _provider_error_summary(response):
    """Extract stable OpenRouter error fields from HTTP or in-band responses."""
    try:
        data = response.json()
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    error = data.get("error")
    if not isinstance(error, dict):
        error = {}
    metadata = error.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    headers = getattr(response, "headers", {}) or {}
    generation_id = data.get("id") or headers.get("x-generation-id") or "unknown"
    return {
        "generation_id": _safe_error_value(generation_id, max_chars=96),
        "error_type": _safe_error_value(metadata.get("error_type"), max_chars=64),
        "provider_code": _safe_error_value(metadata.get("provider_code"), max_chars=64),
        "message": _safe_error_value(error.get("message")),
    }


def _rate_limit_retry_delay(response, failure_number):
    """Honor Retry-After, otherwise use bounded exponential backoff with jitter."""
    headers = getattr(response, "headers", {}) or {}
    raw_retry_after = headers.get("Retry-After")
    if raw_retry_after is None:
        raw_retry_after = headers.get("retry-after")
    if raw_retry_after not in (None, ""):
        try:
            return min(60.0, max(1.0, float(raw_retry_after))), "retry-after"
        except (TypeError, ValueError):
            pass
    base_delay = min(30.0, 5.0 * (2 ** max(0, failure_number - 1)))
    return min(30.0, max(1.0, base_delay * random.uniform(0.9, 1.1))), "backoff"


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


def _current_date_str() -> str:
    """Calendar date only. Clock time already sits on each history message as [Time: HH:MM]."""
    now = now_local()
    return (
        f"Today is {_DAYS[now.weekday()]}, {now.day} "
        f"{_MONTHS[now.month - 1]} {now.year}."
    )


def _chat_session_id(history_key: str) -> str:
    """Stable opaque conversation id for one persisted chat scope."""
    digest = hashlib.sha256(str(history_key).encode("utf-8")).hexdigest()
    return f"berangaria-{digest}"


def _build_system_prompt() -> str:
    """Stable system prefix: personality, rules, optional vision suffix. No date."""
    system_prompt = SYSTEM_PROMPT
    if VISION_MODE:
        system_prompt += VISION_PROMPT_SUFFIX
    return system_prompt


def _build_payload_prefix() -> list[dict]:
    """Cached system prompt, then a daily date line that must not sit inside it."""
    return [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "system", "content": _current_date_str()},
    ]


def _provider_trace_for_history(
    payload_messages: list[dict],
    start: int,
    *,
    final_message: dict | None = None,
    terminal_tool_result: str | None = None,
) -> list[dict]:
    """Copy the exact assistant/tool suffix needed for the next cache prefix.

    Normal final assistant messages have not been appended to ``payload_messages``
    yet, so callers pass them separately. Terminal Telegram tools do not make a
    follow-up model request; synthesize their tool results only after delivery is
    confirmed so the persisted transcript remains API-valid without claiming an
    action that the user never received.
    """
    trace = [
        copy.deepcopy(message)
        for message in payload_messages[max(0, start):]
        if isinstance(message, dict)
    ]
    if isinstance(final_message, dict):
        copied_final = copy.deepcopy(final_message)
        if not copied_final.get("role"):
            copied_final["role"] = "assistant"
        trace.append(copied_final)

    if terminal_tool_result is not None:
        answered_ids = {
            message.get("tool_call_id")
            for message in trace
            if message.get("role") == "tool" and message.get("tool_call_id")
        }
        for message in trace:
            if message.get("role") != "assistant":
                continue
            for tool_call in message.get("tool_calls") or []:
                call_id = tool_call.get("id")
                if call_id and call_id not in answered_ids:
                    trace.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": terminal_tool_result,
                    })
                    answered_ids.add(call_id)
    return trace


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


async def summarize_history(history: list, *, key: str | None = None) -> list:
    session_id = _chat_session_id(key) if key else None
    return await summarization.summarize_history(history, session_id=session_id)


async def _mark_history_sent_to_provider(history: list, *, key: str) -> None:
    """Persist the first provider-send boundary for newly created history rows."""
    async with get_history_lock(key):
        pending = [
            message for message in history if message.get("provider_sent") is False
        ]
        if not pending:
            return
        for message in pending:
            message["provider_sent"] = True
        histories[key] = history
        if not save_history(key):
            for message in pending:
                message["provider_sent"] = False
            raise RuntimeError(
                "не удалось сохранить границу отправки истории провайдеру"
            )


async def send_llm_request(
    update: Update, context: ContextTypes.DEFAULT_TYPE, key: str,
    history: list, user_name: str, user_id: int, mentioned: bool = False):
    """Run one selected model turn while keeping Telegram presence current."""
    message = update.message
    chat = getattr(message, "chat", None) or update.effective_chat
    thread_id = effective_message_thread_id(message)
    async with ChatActionHeartbeat(
        chat,
        action="typing",
        message_thread_id=thread_id,
    ) as chat_actions:
        return await _run_llm_turn(
            update,
            context,
            key,
            history,
            user_name,
            user_id,
            mentioned,
            chat_actions=chat_actions,
        )


async def _run_llm_turn(
    update: Update, context: ContextTypes.DEFAULT_TYPE, key: str,
    history: list, user_name: str, user_id: int, mentioned: bool = False,
    *, chat_actions: ChatActionHeartbeat):

    # Автосуммаризация при достижении 85% от лимита токенов
    context_threshold = int(MAX_CONTEXT_TOKENS * 0.85)
    if chat_tokens.get(key, 0) > context_threshold:
        logger.info(f"📝 [yellow]Автосуммаризация[/] для key={key}")
        history = await summarize_history(history, key=key)
        async with get_history_lock(key):
            histories[key] = history
            save_history(key)

    # Reply handles [#N] only in the payload copy; persisted history stays clean.
    # Date is a second system message so the long personality prefix can cache.
    payload_messages = _build_payload_prefix() + _render_history_for_api(history)
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
    turn = ToolTurn(chat_actions=chat_actions)

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

    async def _deliver(
        text: str,
        target_mid,
        status_msg,
        *,
        quote: str | None = None,
        quote_position: int | None = None,
    ):
        return await reply_delivery.deliver(
            text,
            target_mid,
            status_msg,
            _delivery_runtime(),
            quote=quote,
            quote_position=quote_position,
        )

    async def _deliver_multi(messages: list[str], target_mid, status_msg):
        return await reply_delivery.deliver_multi(
            messages, target_mid, status_msg, _delivery_runtime()
        )

    async def _save_assistant(
        text: str,
        *,
        provider_message: dict | None = None,
        provider_messages: list[dict] | None = None,
    ):
        return await assistant_turn.save_assistant_turn(
            text,
            turn=turn,
            key=key,
            history=history,
            provider_message=provider_message,
            provider_messages=provider_messages,
        )

    async def _remember_bot_mid(entry, sent_mid):
        await assistant_turn.remember_bot_message_id(entry, sent_mid, key=key)

    def _target_identity(target_mid: int | None) -> tuple[int | None, str | None]:
        if target_mid is not None:
            target = next(
                (
                    item
                    for item in history
                    if item.get("role") == "user" and item.get("mid") == target_mid
                ),
                None,
            )
            if target is not None and target.get("author_id") is not None:
                try:
                    return int(target["author_id"]), target.get("author_name")
                except (TypeError, ValueError):
                    pass
            if target_mid != getattr(update.message, "message_id", None):
                return None, None
        return user_id, user_name

    def _record_reply(
        sent_mid: int | None,
        *,
        target_mid: int | None,
        mode: str,
        bubbles: int = 1,
    ) -> None:
        target_id, target_name = _target_identity(target_mid)
        analytics_store.record_event(
            "assistant_reply",
            chat_id=update.effective_chat.id,
            chat_type=update.effective_chat.type,
            actor_kind="bot",
            target_user_id=target_id,
            target_user_name=target_name,
            message_id=sent_mid,
            details={"mode": mode, "bubbles": max(1, bubbles)},
        )

    async def _alert(category: str, message: str, error: BaseException | None = None) -> None:
        await alerts.notify_owner(
            context.bot,
            category=category,
            message=f"chat={update.effective_chat.id}: {message}",
            error=error,
        )

    # Everything appended after this boundary is the provider-only side of the
    # current turn (assistant tool calls, tool results, final raw assistant). It is
    # persisted beside the Telegram-visible text after confirmed delivery.
    provider_turn_start = len(payload_messages)

    # From this point on the exact rendered history may reach the provider. Mark
    # explicit new rows before network I/O so an ambiguous timeout cannot make a
    # possibly cached prefix mutable again.
    await _mark_history_sent_to_provider(history, key=key)

    async with httpx.AsyncClient(timeout=600.0) as client:
        if FULL_DEBUG_LOGS:
            llm_diagnostics.log_request(payload_messages, enabled=True)

        api_failures = 0
        tool_rounds = 0
        while True:
            gen_params = dict(GENERATION_PARAMS)

            session_id = _chat_session_id(key)
            turn_tools = available_tools_for_turn(turn, TOOLS)
            request_body = {
                "model": MODEL,
                "messages": payload_messages,
                "max_tokens": MAX_REPLY_TOKENS,
                "tools": turn_tools,
                **gen_params,
            }
            available_tool_names = {
                (tool.get("function") or {}).get("name") for tool in turn_tools
            }
            if not {"web_search", "read_url"} & available_tool_names:
                # Once no web action remains, force one final answer from the
                # collected evidence instead of allowing a hallucinated tool loop.
                request_body["tool_choice"] = "none"
            payload = apply_chat_gateway(
                request_body,
                session_id=session_id,
            )

            try:
                headers = chat_api_headers(session_id=session_id)
                
                response = await _request_completion(client, payload, headers)

                if response.status_code == 400:
                    await _delete_turn_status()
                    logger.error(f"[red]400:[/] {response.text}")
                    await _alert(
                        "LLM bad request",
                        "API вернул 400; история чата сохранена",
                    )
                    await update.message.reply_text(
                        "⚠️ API отклонил запрос. История сохранена; попробуйте ещё раз."
                    )
                    return

                # Обработка rate limiting
                if response.status_code == 429:
                    api_failures += 1
                    error_summary = _provider_error_summary(response)
                    payload_chars = len(json.dumps(payload, ensure_ascii=False, default=str))
                    if api_failures >= MAX_API_RETRIES:
                        logger.error(
                            "Rate limit (429) exhausted: generation=%s type=%s provider_code=%s "
                            "message=%r; tool_rounds=%s web_search=%s read_url=%s "
                            "messages=%s payload_chars=%s",
                            error_summary["generation_id"],
                            error_summary["error_type"],
                            error_summary["provider_code"],
                            error_summary["message"],
                            tool_rounds,
                            turn.web_search_calls,
                            turn.read_url_calls,
                            len(payload_messages),
                            payload_chars,
                        )
                        await _delete_turn_status()
                        await _alert(
                            "LLM rate limit",
                            "исчерпаны повторы после HTTP 429; "
                            f"generation={error_summary['generation_id']} "
                            f"type={error_summary['error_type']} "
                            f"provider_code={error_summary['provider_code']}",
                        )
                        await update.message.reply_text("❌ API временно перегружен. Попробуйте позже.")
                        return
                    retry_after, delay_source = _rate_limit_retry_delay(response, api_failures)
                    logger.warning(
                        "⚠️ Rate limit (429): generation=%s type=%s provider_code=%s "
                        "message=%r; ждём %gs (%s) перед retry %s/%s; "
                        "tool_rounds=%s web_search=%s read_url=%s messages=%s payload_chars=%s",
                        error_summary["generation_id"],
                        error_summary["error_type"],
                        error_summary["provider_code"],
                        error_summary["message"],
                        retry_after,
                        delay_source,
                        api_failures,
                        MAX_API_RETRIES - 1,
                        tool_rounds,
                        turn.web_search_calls,
                        turn.read_url_calls,
                        len(payload_messages),
                        payload_chars,
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
                    await _alert(
                        "LLM API error",
                        f"исчерпаны повторы после HTTP {response.status_code}",
                    )
                    await update.message.reply_text(f"❌ Ошибка API: {response.status_code}")
                    return

                # A successful provider round starts a fresh transport-retry budget;
                # tool rounds have their own independent MAX_TOOL_ROUNDS ceiling.
                api_failures = 0
                data = response.json()
                choice = data['choices'][0]
                finish_reason = choice.get('finish_reason', '')
                message = choice['message']
                usage = data.get('usage', {})
                llm_diagnostics.log_router_metadata(data, response.headers)

                raw_provider = data.get("provider")
                provider_name = str(raw_provider or "unknown")
                if raw_provider is None:
                    logger.warning(
                        "OpenRouter не вернул provider; ожидался %s",
                        CHAT_PROVIDER,
                    )
                elif provider_name.strip().lower() != CHAT_PROVIDER:
                    generation_id = str(data.get("id") or "unknown")
                    logger.error(
                        "OpenRouter нарушил pin провайдера: expected=%s actual=%s generation=%s",
                        CHAT_PROVIDER,
                        provider_name,
                        generation_id,
                    )
                    await _alert(
                        "LLM routing provider",
                        f"OpenRouter вернул provider={provider_name}, "
                        f"ожидался {CHAT_PROVIDER}; generation={generation_id}",
                    )

                if usage:
                    total_cost = llm_diagnostics.record_usage(
                        usage,
                        key=key,
                        chat_tokens=chat_tokens,
                        estimate_request_cost=_estimate_request_cost,
                    )
                    details = usage.get("prompt_tokens_details") or {}
                    model_name = str(data.get("model") or MODEL)
                    prompt_tokens = max(0, int(usage.get("prompt_tokens", 0) or 0))
                    cached_tokens = max(0, int(details.get("cached_tokens", 0) or 0))
                    cache_ratio = (
                        (cached_tokens / prompt_tokens) * 100
                        if prompt_tokens
                        else 0.0
                    )
                    logger.info(
                        "🧭 Маршрут: provider=%s model=%s session=%s cache=%.1f%%",
                        provider_name,
                        model_name,
                        session_id[-12:],
                        cache_ratio,
                    )
                    analytics_store.record_llm_usage(
                        chat_id=update.effective_chat.id,
                        chat_type=update.effective_chat.type,
                        user_id=user_id,
                        user_name=user_name,
                        provider=provider_name,
                        model=model_name,
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        cached_tokens=details.get("cached_tokens", 0),
                        cache_write_tokens=details.get("cache_write_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        total_tokens=usage.get("total_tokens", 0),
                        cost_usd=total_cost,
                    )
                if finish_reason == 'tool_calls' and message.get('tool_calls'):
                    tool_rounds += 1
                    if tool_rounds > MAX_TOOL_ROUNDS:
                        logger.error(f"❌ Превышен лимит tool-call раундов ({MAX_TOOL_ROUNDS})")
                        await _alert(
                            "LLM tool loop",
                            f"превышен лимит tool-call раундов ({MAX_TOOL_ROUNDS})",
                        )
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
                    turn.pending_reply = None
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

                    provider_tool_trace = _provider_trace_for_history(
                        payload_messages,
                        provider_turn_start,
                    )

                    # reply_to_message терминальный и приоритетный: если модель его вызвала,
                    # отправляем выбранный ответ и завершаем — без ещё одного витка к API
                    # и без дефолтного реплая ниже (двойной отправки не будет).
                    if turn.pending_reply is not None:
                        (
                            reply_mid,
                            reply_text,
                            reply_sid,
                            reply_quote,
                            reply_quote_position,
                        ) = turn.pending_reply
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
                                await _save_assistant(
                                    "",
                                    provider_messages=_provider_trace_for_history(
                                        payload_messages,
                                        provider_turn_start,
                                        terminal_tool_result=(
                                            "Текстовый ответ не был отправлен; "
                                            "ход завершён уже выполненным действием."
                                        ),
                                    ),
                                )
                            return
                        api_call_count[key] = api_call_count.get(key, 0) + 1
                        if reply_text:
                            logger.info(f"↩️ [magenta]Ответ реплаем на[/] [#{reply_sid}]")
                            try:
                                sent_mid = await _deliver(
                                    reply_text,
                                    reply_mid,
                                    turn.status_message,
                                    quote=reply_quote,
                                    quote_position=reply_quote_position,
                                )
                            except Exception as exc:
                                logger.error(f"❌ Не удалось доставить ответ: {exc}", exc_info=True)
                                await _alert(
                                    "Telegram delivery",
                                    "не удалось доставить адресный ответ",
                                    exc,
                                )
                                if turn.reactions_made or turn.stickers_made or turn.voices_made:
                                    await _save_assistant(
                                        "",
                                        provider_messages=_provider_trace_for_history(
                                            payload_messages,
                                            provider_turn_start,
                                            terminal_tool_result=(
                                                "Telegram не подтвердил текстовый ответ; "
                                                "ход завершён уже выполненным действием."
                                            ),
                                        ),
                                    )
                                raise ReplyDeliveryError(
                                    "Telegram не подтвердил доставку ответа"
                                ) from exc
                            saved = await _save_assistant(
                                reply_text,
                                provider_messages=_provider_trace_for_history(
                                    payload_messages,
                                    provider_turn_start,
                                    terminal_tool_result=(
                                        "Ответ доставлен в Telegram. Ход завершён."
                                    ),
                                ),
                            )
                            await _remember_bot_mid(saved, sent_mid)
                            _record_reply(
                                sent_mid,
                                target_mid=reply_mid,
                                mode="reply",
                            )
                        else:
                            # reply_to_message без текста — отправлять нечего (пустых сообщений не шлём)
                            if turn.reacted or turn.sticker_sent or turn.voice_sent:
                                await _save_assistant(
                                    "",
                                    provider_messages=_provider_trace_for_history(
                                        payload_messages,
                                        provider_turn_start,
                                        terminal_tool_result=(
                                            "Текстовый ответ пуст; ход завершён "
                                            "уже выполненным действием."
                                        ),
                                    ),
                                )
                                if turn.sticker_sent or turn.voice_sent:
                                    _record_reply(
                                        None,
                                        target_mid=reply_mid,
                                        mode="voice" if turn.voice_sent else "sticker",
                                    )
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

                    # send_messages — terminal burst (2–5 short bubbles + typing pauses).
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
                            await _alert(
                                "Telegram delivery",
                                "не удалось доставить серию сообщений",
                                exc,
                            )
                            if turn.reactions_made or turn.stickers_made or turn.voices_made:
                                await _save_assistant(
                                    "",
                                    provider_messages=_provider_trace_for_history(
                                        payload_messages,
                                        provider_turn_start,
                                        terminal_tool_result=(
                                            "Telegram не подтвердил пакет сообщений; "
                                            "ход завершён уже выполненным действием."
                                        ),
                                    ),
                                )
                            raise ReplyDeliveryError(
                                "Telegram не подтвердил доставку серии сообщений"
                            ) from exc
                        if delivered:
                            # One history row: joined bubbles (prefix cache / summarizer).
                            history_text = "\n".join(delivered)
                            saved = await _save_assistant(
                                history_text,
                                provider_messages=_provider_trace_for_history(
                                    payload_messages,
                                    provider_turn_start,
                                    terminal_tool_result=(
                                        "Сообщения доставлены в Telegram. Ход завершён."
                                    ),
                                ),
                            )
                            await _remember_bot_mid(saved, sent_mid)
                            _record_reply(
                                sent_mid,
                                target_mid=target_mid,
                                mode="multi",
                                bubbles=len(delivered),
                            )
                        elif turn.reactions_made or turn.stickers_made or turn.voices_made:
                            await _save_assistant(
                                "",
                                provider_messages=_provider_trace_for_history(
                                    payload_messages,
                                    provider_turn_start,
                                    terminal_tool_result=(
                                        "Текстовые сообщения не отправлены; ход завершён "
                                        "уже выполненным действием."
                                    ),
                                ),
                            )
                        return

                    # Стикер уже в чате — это полный ответ, лишний round-trip к API не нужен.
                    if turn.sticker_sent:
                        api_call_count[key] = api_call_count.get(key, 0) + 1
                        await _save_assistant(
                            "",
                            provider_messages=provider_tool_trace,
                        )
                        _record_reply(
                            None,
                            target_mid=update.message.message_id,
                            mode="sticker",
                        )
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
                        await _save_assistant(
                            "",
                            provider_messages=provider_tool_trace,
                        )
                        _record_reply(
                            None,
                            target_mid=update.message.message_id,
                            mode="voice",
                        )
                        logger.info("🔊 [dim]Ход завершён голосовым (без доп. текста)[/]")
                        if turn.status_message:
                            try:
                                await turn.status_message.delete()
                            except Exception:
                                pass
                        return

                    continue

                reply = message.get('content', '')
                provider_final_trace = _provider_trace_for_history(
                    payload_messages,
                    provider_turn_start,
                    final_message=message,
                )
                
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
                        await _save_assistant(
                            "",
                            provider_message=message,
                            provider_messages=provider_final_trace,
                        )
                        if turn.sticker_sent or turn.voice_sent:
                            _record_reply(
                                None,
                                target_mid=update.message.message_id,
                                mode="voice" if turn.voice_sent else "sticker",
                            )
                        if turn.status_message:
                            try:
                                await turn.status_message.delete()
                            except Exception:
                                pass
                        return
                    if not mentioned:
                        logger.info(f"🤫 [dim]Промолчала (ambient)[/] (ключ={key})")
                        if turn.status_message:
                            try:
                                await turn.status_message.delete()
                            except Exception:
                                pass
                        return
                    logger.warning(f"⚠️ [yellow]Пустой ответ при прямом обращении[/] (ключ={key})")
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
                    await _alert(
                        "Telegram delivery",
                        "не удалось доставить ответ",
                        exc,
                    )
                    if turn.reactions_made or turn.stickers_made or turn.voices_made:
                        await _save_assistant(
                            "",
                            provider_messages=_provider_trace_for_history(
                                payload_messages,
                                provider_turn_start,
                                terminal_tool_result=(
                                    "Telegram не подтвердил финальный текст; "
                                    "ход завершён уже выполненным действием."
                                ),
                            ),
                        )
                    raise ReplyDeliveryError(
                        "Telegram не подтвердил доставку ответа"
                    ) from exc
                saved = await _save_assistant(
                    reply,
                    provider_message=message,
                    provider_messages=provider_final_trace,
                )
                await _remember_bot_mid(saved, sent_mid)
                _record_reply(sent_mid, target_mid=target_mid, mode="text")
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
                await _alert("LLM connection", "API недоступен после всех повторов")
                await update.message.reply_text("❌ API недоступен!")
                return
            except httpx.TimeoutException:
                logger.error("❌ [bright_red]Таймаут запроса к API[/]")
                api_failures += 1
                if api_failures < MAX_API_RETRIES:
                    await asyncio.sleep(2 ** (api_failures - 1))
                    continue
                await _delete_turn_status()
                await _alert("LLM timeout", "таймаут API после всех повторов")
                await update.message.reply_text("❌ Таймаут.")
                return
            except Exception as e:
                logger.error(f"❌ [bright_red]Ошибка в обработке запроса:[/] {e}", exc_info=True)
                api_failures += 1
                if api_failures < MAX_API_RETRIES:
                    await asyncio.sleep(2 ** (api_failures - 1))
                    continue
                await _delete_turn_status()
                await _alert("LLM failure", "обработка запроса завершилась ошибкой", e)
                await update.message.reply_text("❌ Ошибка при обработке.")
                return
