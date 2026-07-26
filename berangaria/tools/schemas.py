"""Описания инструментов, которые видит LLM, и белый список Telegram-реакций.

Тексты описаний — это часть промпта: модель решает, вызывать ли инструмент,
именно по ним. Правь их так же осознанно, как SYSTEM_PROMPT.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Mandatory search for prices, specs, news, dates after 2023. Then give an answer with numbers — don't say 'rumored' or 'no data'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query in the most relevant language"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results, 3-8",
                        "default": 5
                    },
                    "timelimit": {
                        "type": "string",
                        "description": "Time filter: 'd'=day, 'w'=week, 'm'=month, 'y'=year",
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
                "Use it freely and often to show emotions: agreement 👍, laughter 😂, shock 😱, "
                "trolling 🤡, approval 🔥. PREFER reaction-only (no text) for simple acknowledgment. "
                "Add text only if you have something specific to say. "
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
                "Search your sticker collection by vibe. You LOVE using stickers and should do it proactively when the vibe fits. Returns a NUMBERED list of matching stickers"
                "with descriptions and tags — it does NOT send anything. Use it when you feel like reacting "
                "with a sticker: browse the options, then send the one that best fits via send_sticker(id). "
                "You may refine the search up to 3 times per turn with different wording; after that pick "
                "from already found numbers or answer without a sticker. If none fit, don't send. Don't overuse stickers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Vivid description of the sticker's mood/content, in Russian, e.g. "
                            "'недоумение, кто-то сморозил глупость' or 'ржу в голос' or 'одобряю, огонь'."
                        )
                    },
                    "count": {
                        "type": "integer",
                        "description": "How many candidates to return, 3-10 (default 6)."
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
                "Send the sticker that best matches the current emotion/vibe. Sticker-only replies are highly encouraged."
                "Send ONE specific sticker to the chat, chosen from the latest find_stickers results by its number. "
                "Call this only AFTER find_stickers, passing the id of the option you liked best. "
                "After a successful send the turn ends — the sticker IS the full reply (no extra text round). "
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
                "Use when the user sends a link or asks to analyze/comment on a specific URL. "
                "Don't use for general questions — use web_search for those."
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
