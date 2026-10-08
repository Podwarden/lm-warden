import base64

import pytest

from app.chat2.history import PRESENT_OPTIONS_TOOL, build_messages
from app.chat2.repo import MessageRow


def _row(seq, role, parts, finish=None):
    return MessageRow(
        id=f"m{seq}",
        chat_id="c",
        seq=seq,
        role=role,
        parts=parts,
        model=None,
        settings_snapshot={},
        usage=None,
        finish_reason=finish,
        error=None,
        created_at="2026-01-01T00:00:00.000Z",
    )


async def _loader(aid: str):
    return ("image/png", b"PNGDATA") if aid == "img1" else None


@pytest.mark.asyncio
async def test_build_messages_shapes() -> None:
    rows = [
        _row(
            1,
            "user",
            [
                {"type": "text", "text": "hi"},
                {"type": "image", "attachmentId": "img1"},
                {"type": "image", "attachmentId": "gone"},
            ],
        ),
        _row(
            2,
            "assistant",
            [
                {"type": "reasoning", "text": "thinking"},
                {
                    "type": "tool_call",
                    "id": "c1",
                    "name": "present_options",
                    "args": {"options": []},
                    "argsText": '{"options":[]}',
                    "client": True,
                },
                {"type": "options", "callId": "c1", "options": [], "multi": False},
                {"type": "text", "text": "pick one"},
            ],
            finish="tool_calls",
        ),
        _row(
            3,
            "tool",
            [{"type": "tool_result", "callId": "c1", "result": {"selected": ["a"]}, "ok": True}],
        ),
    ]
    out = await build_messages(
        rows, {"system_prompt": "be brief"}, supports_vision=True, load_image=_loader
    )
    assert out[0] == {"role": "system", "content": "be brief"}
    user = out[1]
    assert user["role"] == "user" and user["content"][0] == {"type": "text", "text": "hi"}
    b64 = base64.b64encode(b"PNGDATA").decode()
    assert user["content"][1]["image_url"]["url"] == f"data:image/png;base64,{b64}"
    assert user["content"][2] == {"type": "text", "text": "[image expired]"}
    asst = out[2]
    assert "reasoning" not in asst and asst["content"] == "pick one"
    assert asst["tool_calls"] == [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "present_options", "arguments": '{"options":[]}'},
        }
    ]
    assert out[3] == {"role": "tool", "tool_call_id": "c1", "content": '{"selected": ["a"]}'}


@pytest.mark.asyncio
async def test_no_vision_drops_images_and_empty_system_omitted() -> None:
    rows = [
        _row(1, "user", [{"type": "text", "text": "hi"}, {"type": "image", "attachmentId": "img1"}])
    ]
    out = await build_messages(
        rows, {"system_prompt": ""}, supports_vision=False, load_image=_loader
    )
    assert out == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "text", "text": "[image omitted]"},
            ],
        }
    ]
    assert PRESENT_OPTIONS_TOOL["function"]["name"] == "present_options"
