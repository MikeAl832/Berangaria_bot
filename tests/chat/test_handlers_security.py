import asyncio
from types import SimpleNamespace

from berangaria.chat import handlers
from berangaria.core import state


class _ForbiddenMediaMessage:
    caption = ""

    def __init__(self):
        self.replies = []

    @property
    def photo(self):
        raise AssertionError("photo must not be accessed before authorization")

    @property
    def sticker(self):
        raise AssertionError("sticker must not be accessed before authorization")

    async def reply_text(self, text, **kwargs):
        self.replies.append(text)


class _User:
    id = 2


class _Chat:
    id = 100
    type = "private"


class _Update:
    def __init__(self):
        self.message = _ForbiddenMediaMessage()
        self.effective_user = _User()
        self.effective_chat = _Chat()


class _Bot:
    id = 999


class _Context:
    bot = _Bot()


class _VoiceChat:
    def __init__(self):
        self.id = -100
        self.type = "supergroup"
        self.title = "Тест"
        self.actions = []

    async def send_action(self, action="typing"):
        self.actions.append(action)


class _VoiceMessage:
    def __init__(self, chat):
        self.chat = chat
        self.message_id = 905
        self.date = None
        self.text = None
        self.caption = None
        self.audio = None
        self.voice = SimpleNamespace(
            file_id="voice-file",
            file_unique_id="voice-unique",
            duration=3,
        )
        self.forward_origin = None
        self.reply_to_message = None
        self.replies = []

    async def reply_text(self, text, **kwargs):
        self.replies.append(text)


def _voice_update_and_context():
    chat = _VoiceChat()
    update = SimpleNamespace(
        message=_VoiceMessage(chat),
        effective_chat=chat,
        effective_user=SimpleNamespace(id=42, first_name="Миша"),
    )
    context = SimpleNamespace(
        bot=SimpleNamespace(
            id=999,
            username="berangaria_bot",
            first_name="Berangaria",
        )
    )
    return chat, update, context


def _run_voice_transcript(monkeypatch, transcript, ambient_selector):
    chat, update, context = _voice_update_and_context()
    actions_during_transcription = []
    llm_calls = []

    monkeypatch.setattr(handlers, "VISION_MODE", True)
    monkeypatch.setattr(handlers, "MESSAGE_DEBOUNCE_SECONDS", 0)
    monkeypatch.setattr(handlers, "_check_access_permissions", lambda *args: True)
    monkeypatch.setattr(
        handlers,
        "download_audio_to_file",
        lambda *args, **kwargs: asyncio.sleep(
            0, result=("voice.ogg", "audio/ogg")
        ),
    )

    async def transcribe_audio(**kwargs):
        actions_during_transcription.append(list(chat.actions))
        return transcript

    async def send_llm_request(*args, **kwargs):
        llm_calls.append((args, kwargs))

    monkeypatch.setattr(handlers, "transcribe_audio", transcribe_audio)
    monkeypatch.setattr(handlers, "should_reply_randomly", ambient_selector)
    monkeypatch.setattr(handlers, "send_llm_request", send_llm_request)
    monkeypatch.setattr(handlers, "release_memory_sources", lambda source_ids: None)

    async def run():
        await handlers.handle_voice(update, context)
        task = state.message_buffer["-100_42"]["task"]
        await task

    asyncio.run(run())
    return chat, actions_during_transcription, llm_calls


def test_log_message_preview_uses_configured_limit(monkeypatch):
    monkeypatch.setattr(handlers, "LOG_MESSAGE_PREVIEW_CHARS", 400)
    text = "я" * 500

    preview = handlers._log_message_preview(text)

    assert len(preview) == 400
    assert preview.endswith("...")
    assert handlers._log_message_preview("коротко") == "коротко"


def test_photo_access_checked_before_media_processing(monkeypatch):
    monkeypatch.setattr(handlers, "ALLOWED_USERS", [1])
    update = _Update()

    asyncio.run(handlers.handle_media(update, _Context()))

    assert update.message.replies == ["Не разговариваю с незнакомцами."]


def test_sticker_access_checked_before_media_processing(monkeypatch):
    monkeypatch.setattr(handlers, "ALLOWED_USERS", [1])
    update = _Update()

    asyncio.run(handlers.handle_sticker(update, _Context()))

    assert update.message.replies == ["Не разговариваю с незнакомцами."]


def test_owner_bypasses_user_and_group_allowlists(monkeypatch):
    monkeypatch.setattr(handlers, "OWNER_USER_ID", 2)
    monkeypatch.setattr(handlers, "ALLOWED_USERS", [1])
    monkeypatch.setattr(handlers, "ALLOWED_GROUPS", [-999])

    assert handlers._check_access_permissions(100, 2, False)
    assert handlers._check_access_permissions(-100, 2, True)
    assert not handlers._check_access_permissions(100, 3, False)
    assert not handlers._check_access_permissions(-100, 3, True)


def test_group_media_invalidates_older_ambient_candidate(monkeypatch):
    monkeypatch.setattr(handlers, "ALLOWED_GROUPS", [-100])
    stale_token = state.record_group_activity(-100)
    update = SimpleNamespace(
        message=SimpleNamespace(),
        effective_chat=SimpleNamespace(id=-100, type="supergroup"),
        effective_user=SimpleNamespace(id=2),
    )
    context = SimpleNamespace(bot=SimpleNamespace(id=999))

    handlers._record_incoming_media_activity(update, context)

    assert not state.is_latest_group_activity(-100, stale_token)


def test_unmentioned_voice_stays_silent_without_early_typing(
    monkeypatch, isolated_db
):
    chat, actions_during_transcription, llm_calls = _run_voice_transcript(
        monkeypatch,
        "Всем привет, как ваши дела?",
        lambda *args: False,
    )

    assert actions_during_transcription == [[]]
    assert chat.actions == []
    assert llm_calls == []


def test_spoken_name_triggers_reply_without_gemini_typing(
    monkeypatch, isolated_db
):
    def should_not_sample_ambient(*args):
        raise AssertionError("explicit spoken mention must bypass ambient sampling")

    chat, actions_during_transcription, llm_calls = _run_voice_transcript(
        monkeypatch,
        "Бер, ты здесь?",
        should_not_sample_ambient,
    )

    assert actions_during_transcription == [[]]
    assert chat.actions == []
    assert len(llm_calls) == 1
    assert llm_calls[0][0][-1] is True
    history = llm_calls[0][0][3]
    assert "[Message: (сообщение без текста)]" in history[-1]["content"]
    assert "[Audio description: Бер, ты здесь?]" in history[-1]["content"]


def test_extract_reply_context_prefers_manual_selected_quote():
    replied = SimpleNamespace(
        from_user=SimpleNamespace(first_name="Миша"),
        sender_chat=None,
        text="Это длинное исходное сообщение",
        caption=None,
    )
    message = SimpleNamespace(
        reply_to_message=replied,
        quote=SimpleNamespace(
            text="длинное исходное",
            position=4,
            is_manual=True,
        ),
    )

    assert handlers._extract_reply_context(message) == (
        "Миша",
        "длинное исходное",
        True,
        4,
    )


def test_extract_reply_context_falls_back_to_original_excerpt():
    replied = SimpleNamespace(
        from_user=SimpleNamespace(first_name="Миша"),
        sender_chat=None,
        text="я" * 100,
        caption=None,
    )
    message = SimpleNamespace(reply_to_message=replied, quote=None)

    assert handlers._extract_reply_context(message) == (
        "Миша",
        "я" * 80,
        False,
        None,
    )
