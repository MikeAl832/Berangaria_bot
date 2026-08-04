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
                "Fact-check against the live web. Use BEFORE you state a checkable fact yourself, "
                "and BEFORE you agree with, repeat as true, or casually accept a checkable claim "
                "someone else just made — confident tone is not evidence. "
                "Mandatory for numbers, prices, rates, dates, specs, versions, releases, statistics, "
                "real people/companies/products, current status, and anything that would differ a year ago. "
                "Also when about to hedge ('вроде', 'кажется') or rubber-stamp ('да, точно') on such a claim: search instead. "
                "NOT for opinions, jokes, pure absurdity you are only mocking, arithmetic, or facts about this chat and its people. "
                "After results: answer with the specific number or date; do not invent or endorse the user's story if sources disagree or are empty. "
                "Never mention that you searched."
            ),
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
                "If your whole answer IS the emotion — you were about to type 'ржу' / 'жесть' / 'топ' — "
                "call send_sticker(query) instead: a reaction acknowledges, a sticker replies. "
                "By default it reacts to the latest message; pass the [#N] handle as 'id' to react to a specific earlier one. "
                "Do NOT react again to a message you already reacted to (history notes look like "
                "'Ты поставила реакцию 🤡 на [#N] …') — pick another [#N], write text, send_sticker, or stay silent. "
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
            "name": "send_sticker",
            "description": (
                "ONE call: vector-search your sticker pack and post one match immediately "
                "(no second tool, no id pick). "
                "Use when the whole reply is emotion — laughter, mockery, refusal, shock, fatigue, cringe — "
                "or you were about to type 'ржу' / 'жесть' / 'топ' / 'ну ты дал' / 'не хочу'. "
                "Prefer over a reaction when you would have typed something; reaction-only when you would say nothing. "
                "INDEX (Russian): each sticker is stored as emotion + secondary emotions + action + use_cases "
                "(when to send) + situation + keywords + optional character + visual description. "
                "Query that language: occasion/emotion tags, not a chat retelling. "
                "On success the turn ENDS; plain text alongside is discarded. "
                "Not for direct questions, help, or after web_search. Miss → tighter query once or words. "
                "NEVER fake a sticker in text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Short Russian tags matching the index: primary emotion and/or use_case phrases. "
                            "GOOD: 'отказ, не хочу', 'ирония, сарказм', 'отчаяние, паника', 'смущение, обида', "
                            "'радость, ура', 'усталость, всё', 'злость, бесит'. "
                            "BAD: 'стикер про то как человек купил машину и хвастается' (story, no match)."
                        )
                    }
                },
                "required": ["query"]
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
    },
    {
        "type": "function",
        "function": {
            "name": "send_messages",
            "description": (
                "Send 2–3 short Telegram messages in a natural burst (typing pauses between them). "
                "Use RARELY when the reply is naturally two beats — a reaction then a thought, a joke then a question — "
                "not to pad length. Most turns are still ONE plain-text reply or a sticker. "
                "After a successful call the turn ENDS: plain text written alongside is discarded; "
                "do not combine with reply_to_message, send_sticker, or send_voice. "
                "Each string is one bubble: short, no service tags, no emoji. "
                "Not for facts after web_search, lists, instructions, or ambient one-liners."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "2 or 3 short message texts, in order. "
                            "Example: [\"ну ты дал\", \"и это серьёзно?\"]"
                        ),
                        "minItems": 2,
                        "maxItems": 3,
                    }
                },
                "required": ["messages"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_voice",
            "description": (
                "Speak a SHORT reply as a Telegram voice note in your calm, slightly haughty voice. "
                "Use RARELY — only when spoken deadpan lands better than text "
                "(quiet burn, dry one-liner, intimate private-chat beat). "
                "Default remains plain text or send_sticker. "
                "On success the turn ENDS; do not combine with send_sticker, send_messages, or reply_to_message. "
                "Not for long explanations, lists, code, facts after web_search, or pure emotion "
                "(pure emotion → send_sticker). "
                "Optional emotion is a subtle delivery cue only — calm / sarcastic / disdainful / "
                "bored / indifferent / confident / sighing / chuckling. Never invent other tags; "
                "never put [brackets] in text; never announce that this is a voice message."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "What you say aloud: 1–2 short sentences, plain speech, no emoji, "
                            "no service tags, no stage directions."
                        ),
                    },
                    "emotion": {
                        "type": "string",
                        "description": (
                            "Optional delivery cue. Omit for default calm deadpan. "
                            "Allowed: calm, sarcastic, disdainful, bored, indifferent, "
                            "confident, sighing, chuckling, none."
                        ),
                        "enum": [
                            "calm",
                            "sarcastic",
                            "disdainful",
                            "bored",
                            "indifferent",
                            "confident",
                            "sighing",
                            "chuckling",
                            "none",
                        ],
                    },
                },
                "required": ["text"],
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
