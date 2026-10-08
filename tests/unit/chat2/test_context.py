from app.chat2.context import derive_context
from app.chat2.repo import MessageRow


def _m(seq, role, usage=None):
    return MessageRow(
        id=f"m{seq}",
        chat_id="c",
        seq=seq,
        role=role,
        parts=[],
        model=None,
        settings_snapshot={},
        usage=usage,
        finish_reason=None,
        error=None,
        created_at="2026-01-01T00:00:00.000Z",
    )


def test_derive_context_uses_last_assistant_usage_and_capped_reserve() -> None:
    msgs = [_m(1, "user"), _m(2, "assistant", {"prompt": 4000, "completion": 200}), _m(3, "user")]
    s = derive_context(msgs, {"max_tokens": 32768}, window=8192)
    assert (s.prompt_tokens, s.window, s.full) == (4200, 8192, True)  # 4200 + 4096 >= 8192
    s2 = derive_context(msgs, {"max_tokens": 512}, window=8192)
    assert s2.full is False  # 4200 + 512 < 8192


def test_derive_context_unknown_window_never_full_and_no_usage() -> None:
    assert derive_context([_m(1, "user")], {"max_tokens": 100}, window=None).full is False
    s = derive_context([], {"max_tokens": 100}, window=1000)
    assert (s.prompt_tokens, s.full) == (0, False)
    assert s.as_event() == {"type": "context", "promptTokens": 0, "window": 1000, "full": False}
