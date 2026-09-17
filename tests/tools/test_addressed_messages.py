import pytest

from berangaria.tools.dispatch import ToolTurn, handle_send_messages
from berangaria.chat.outgoing_message import OutgoingMessage


def test_model_payload_resolves_each_reply_target():
    turn, payload = ToolTurn(), []
    handle_send_messages(turn, payload, {"id": "batch"}, {
        "messages": [
            {"text": "Ответ первому", "reply_to": 12},
            {"text": "Ответ второму", "reply_to": 18},
            {"text": "Общее замечание"},
        ],
    }, {12: 1012, 18: 1018})
    assert payload == []
    assert turn.pending_messages == [
        OutgoingMessage("Ответ первому", 1012, 12),
        OutgoingMessage("Ответ второму", 1018, 18),
        OutgoingMessage("Общее замечание"),
    ]


@pytest.mark.parametrize("target", [999, True, "12", 12.5])
def test_invalid_target_rejects_whole_batch(target):
    turn, payload = ToolTurn(), []
    handle_send_messages(turn, payload, {"id": "batch"}, {
        "messages": [{"text": "Один", "reply_to": 12}, {"text": "Два", "reply_to": target}],
    }, {12: 1012})
    assert turn.pending_messages is None
    assert payload[0]["tool_call_id"] == "batch"
    assert "reply_to" in payload[0]["content"]
