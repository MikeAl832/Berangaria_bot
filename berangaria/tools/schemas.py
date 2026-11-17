"""The tool descriptions the LLM sees, plus the Telegram reaction whitelist.

These descriptions are part of the prompt: they are what the model reads when it
decides whether to call a tool. Edit them as deliberately as SYSTEM_PROMPT.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Fact-check against the live web. Call BEFORE stating a checkable fact yourself, "
                "and BEFORE agreeing with, repeating, or casually accepting a checkable claim "
                "someone else made (confident tone is not evidence). "
                "Mandatory for numbers, prices, rates, dates, specs, versions, releases, statistics, "
                "real people/companies/products, current status, and anything that could differ from a year ago. "
                "Also call instead of hedging ('вроде', 'кажется') or rubber-stamping ('да, точно') on such claims. "
                "Skip for opinions, jokes, pure absurdity you are mocking, arithmetic, or chat lore. "
                "Answer with the concrete number, name, or date from sources; do not invent or endorse unsupported claims, "
                "and never announce that you searched."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Keywords plus the entity rather than a full sentence. "
                            "Use Russian for local/RU topics, English for tech, global news, and foreign products."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results, 3 to 8.",
                        "default": 5,
                    },
                    "region": {
                        "type": "string",
                        "description": (
                            "'ru-ru' for local and Russian-language topics (default), "
                            "'wt-wt' for global, tech, and English-language topics."
                        ),
                    },
                    "timelimit": {
                        "type": "string",
                        "description": (
                            "Time filter for fast-moving facts (news, prices, rates, standings, live scores): "
                            "'d'=day, 'w'=week, 'm'=month, 'y'=year. Omit for stable facts."
                        ),
                        "enum": ["d", "w", "m", "y"],
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "react_to_message",
            "description": (
                "Put a real Telegram emoji badge on a message (not in your reply text). "
                "Use freely to acknowledge emotion: agreement 👍, laughter 😂, shock 😱, trolling 🤡, approval 🔥. "
                "Prefer reaction-only (no text) when 'seen, noted' is enough; add text only if you have a specific remark. "
                "If your whole reply is an emotional beat where you would type 'ржу' / 'жесть' / 'топ', "
                "call send_sticker(query) instead. "
                "Defaults to the latest message; pass [#N] as 'id' for an earlier one. "
                "Do not react twice to a message you already reacted to in history; pick another [#N], send a sticker, write text, or stay silent. "
                "Never narrate reactions in text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "emoji": {
                        "type": "string",
                        "description": "A single allowed Telegram reaction emoji.",
                    },
                    "id": {
                        "type": "integer",
                        "description": "Optional [#N] handle of the message to react to. Omit for the latest message.",
                    },
                },
                "required": ["emoji"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reply_to_message",
            "description": (
                "Send your message as a Telegram reply to a specific [#N] message handle "
                "(shown at the start of incoming messages, e.g. [#7]). "
                "Not needed in normal 1-on-1 back-and-forth where plain text suffices. "
                "Use when answering an earlier or different message, or when highlighting exact words in the latest message. "
                "Pass the [#N] number as 'id' and your reply as 'text'. To highlight specific words in Telegram's reply header, "
                "optionally pass 'quote' copied EXACTLY from that message's [Message: ...] text; "
                "never paraphrase quote, and omit it when an exact substring is unavailable."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "The [#N] handle number of the target message.",
                    },
                    "text": {
                        "type": "string",
                        "description": "Your reply text (no service tags or emoji; supported Telegram markup is allowed).",
                    },
                    "quote": {
                        "type": "string",
                        "description": "Optional exact substring of the target message to highlight in Telegram's reply header.",
                    },
                },
                "required": ["id", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_sticker",
            "description": (
                "Vector-search your sticker pack and send one matching sticker immediately in a single call. "
                "Use when the whole reply is emotion (laughter, mockery, refusal, shock, fatigue, cringe) "
                "or instead of typing 'ржу' / 'жесть' / 'топ' / 'ну ты дал' / 'не хочу'. "
                "Also use when you have no sentence to add instead of inventing filler or going empty. "
                "Prefer over a badge reaction when replying in the thread; use reaction-only when adding nothing at all. "
                "Stickers are indexed in Russian by emotion, action, occasion, keywords, character, and visual description; "
                "query with short emotion/occasion tags rather than retelling the chat. "
                "Success ends the turn and discards accompanying plain text. "
                "Not for direct questions, help requests, or after web_search. On a miss, try one tighter query or reply in words."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Short Russian emotion or occasion tags matching the index. "
                            "Examples: 'отказ, не хочу', 'ирония, сарказм', 'отчаяние, паника', "
                            "'радость, ржу', 'усталость, всё', 'злость, бесит'."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_url",
            "description": (
                "Fetch a web page by URL and read its text content. "
                "Use when someone shares a link or asks about a specific page, or right after web_search "
                "when a result snippet is truncated or missing the exact detail you need. "
                "For general questions without a URL, call web_search first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full http:// or https:// page URL.",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_messages",
            "description": (
                "Send 1 to 5 short Telegram bubbles in order, each optionally replying to its own [#N] target. "
                "Use when your answer genuinely splits into two or more distinct beats, or when addressing "
                "multiple people/messages in a group. For a single compact thought, reply in plain text instead. "
                "Available after web_search/read_url (address the user who asked, not the tool output). "
                "Omit reply_to for standalone bubbles. Success ends the turn and discards accompanying plain text; "
                "do not combine with reply_to_message or send_sticker."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string", "description": "Bubble text."},
                                "reply_to": {
                                    "type": "integer",
                                    "description": "Optional existing [#N] handle to reply to in this chat.",
                                },
                            },
                            "required": ["text"],
                            "additionalProperties": False,
                        },
                        "description": "Messages in delivery order, each with an optional reply_to.",
                        "minItems": 1,
                        "maxItems": 5,
                    }
                },
                "required": ["messages"],
            },
        },
    },
]

# Разрешённые Telegram эмодзи для реакций (для валидации перед вызовом API)
ALLOWED_REACTIONS = {
    "👍", "👎", "❤", "🔥", "🥰", "👏", "😁", "🤔", "🤯", "😱", "🤬", "😢", "🎉",
    "🤩", "🤮", "💩", "🙏", "👌", "🕊", "🤡", "🥱", "🥴", "😍", "🐳", "❤‍🔥", "🌚",
    "🌭", "💯", "🤣", "⚡", "🍌", "🏆", "💔", "🤨", "😐", "🍓", "🍾", "💋", "🖕",
    "😈", "😴", "😭", "🤓", "👻", "👨‍💻", "👀", "🎃", "🙈", "😇", "😨", "🤝", "✍",
    "🤗", "🫡", "🎅", "🎄", "☃", "💅", "🤪", "🗿", "🆒", "💘", "🙉", "🦄", "😘",
    "💊", "🙊", "😎", "👾", "🤷‍♂", "🤷", "🤷‍♀", "😡",
}
