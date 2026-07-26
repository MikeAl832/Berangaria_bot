"""The tool descriptions the LLM sees, plus the Telegram reaction whitelist.

These descriptions are part of the prompt: they are what the model reads when it
decides whether to call a tool. Edit them as deliberately as SYSTEM_PROMPT.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Verify a checkable fact against the live web BEFORE you state it. Mandatory for numbers, prices, rates, dates, specs, versions, releases, statistics, real people/companies/products, current status, and anything whose answer would differ today from a year ago — and ALWAYS before confirming, denying or mocking a factual claim someone else made. If you were about to hedge ('вроде', 'кажется', 'если не ошибаюсь'), search instead of hedging. NOT for opinions, jokes, hyperbole, arithmetic, or facts about this chat and its people. Then answer with the specific number or date; 'rumored' / 'no exact data' are banned while sources exist. Never mention that you searched.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query: keywords plus the entity, not a full sentence. Use the language of the best source — Russian for local/RU topics, English for tech, global news and foreign products."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results, 3-8",
                        "default": 5
                    },
                    "region": {
                        "type": "string",
                        "description": "'ru-ru' for local and Russian-language topics (default), 'wt-wt' for global, tech and English-language topics. Match it to the query language."
                    },
                    "timelimit": {
                        "type": "string",
                        "description": "Time filter for fast-moving facts (news, prices, rates, standings, live scores): 'd'=day, 'w'=week, 'm'=month, 'y'=year. Omit it for stable facts.",
                        "enum": ["d", "w", "m", "y"]
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "react_to_message",
            "description": (
                "Put an emoji reaction badge on a message (NOT in your text). "
                "This is a real Telegram action — the emoji appears next to that message. "
                "Use it freely and often to acknowledge emotion: agreement 👍, laughter 😂, shock 😱, "
                "trolling 🤡, approval 🔥. PREFER reaction-only (no text) when a message needs nothing beyond 'seen, noted'. "
                "Add text only if you have something specific to say. "
                "If your whole answer IS the emotion — you were about to type 'ржу' / 'жесть' / 'топ' — send a sticker "
                "instead: a reaction acknowledges, a sticker replies. "
                "By default it reacts to the latest message; pass the [#N] handle as 'id' to react to a specific earlier one. "
                "Do NOT react again to a message you already reacted to (history notes look like "
                "'Ты поставила реакцию 🤡 на [#N] …') — pick another [#N], write text, send a sticker, or stay silent. "
                "NEVER fake it in text (no '*reacts with 🔥*') — call this function instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "emoji": {
                        "type": "string",
                        "description": "A single emoji character to react with (a common Telegram reaction)."
                    },
                    "id": {
                        "type": "integer",
                        "description": "Optional [#N] handle of the message to react to. Omit to react to the latest message."
                    }
                },
                "required": ["emoji"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reply_to_message",
            "description": (
                "Send your message as a Telegram REPLY to a specific earlier message, identified by its [#N] handle "
                "(shown at the very start of each incoming message, e.g. [#7]). "
                "In a normal back-and-forth you do NOT need this — just answer with plain text. "
                "Use it only when you deliberately want to answer an EARLIER or different message than the latest one "
                "(e.g. you were pinged and want to pick up something said a few messages ago). "
                "Pass the number from [#N] as 'id' and your reply as 'text'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "The [#N] handle number of the message you want to reply to."
                    },
                    "text": {
                        "type": "string",
                        "description": "Your reply text (plain text, no service tags, no emoji)."
                    }
                },
                "required": ["id", "text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_stickers",
            "description": (
                "Search your sticker collection by vibe. Returns a NUMBERED list of candidates "
                "(emotion tag, what the sticker shows, keywords) — it sends nothing yet. "
                "Use it whenever your reply is mostly emotion: laughter, mockery, approval, shock, fatigue, absurdity — "
                "or whenever you were about to type a short line like 'ржу' / 'жесть' / 'топ'. That line IS a sticker. "
                "Then call send_sticker(id) with the number you liked; one search is normally enough (max 3 per turn). "
                "The one hard limit: a sticker must not stand in for an answer — if the message asks a direct question "
                "or asks for help, answer that in words. Everywhere else the sticker is the better reply."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A Russian emotion word plus 2-3 concrete words — NOT a retelling of the situation. "
                            "GOOD: 'ирония, ухмылка', 'шок, глаза по пять копеек', 'раздражение, достали'. "
                            "BAD: 'стикер про то как человек купил машину и хвастается'."
                        )
                    },
                    "count": {
                        "type": "integer",
                        "description": "How many candidates to return, 3-10. Ask for 8 — more options, more chance one fits.",
                        "default": 8
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_sticker",
            "description": (
                "Send ONE sticker to the chat, chosen from the latest find_stickers results by its number. "
                "A sticker-only reply is the normal case, not an exception — but never in place of an actual "
                "answer: a direct question or a request for help needs words. "
                "Call this only AFTER find_stickers, passing the id of the option you liked best. "
                "After a successful send the turn ends, and plain text written alongside it is discarded — "
                "the sticker is the whole reply. "
                "NEVER describe a sticker in text ('*кидает стикер*') — send it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "The number of the sticker from find_stickers results (e.g. 3)."
                    }
                },
                "required": ["id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_url",
            "description": (
                "Download a web page by URL and read its text content. "
                "Use when the user sends a link or asks about a specific page, AND as the follow-up to "
                "web_search when a result snippet is truncated, vague, or missing the number you need — "
                "open the most credible result URL and read the real page. "
                "Don't use it for a general question with no URL in hand — that is web_search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full page URL (with http:// or https://)"
                    }
                },
                "required": ["url"]
            }
        }
    }
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
