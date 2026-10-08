"""Pure-function tests for the god-mode display-only prompt window.

The window NEVER touches the forwarded request — it only shapes the tapped copy
so a huge repeated system prompt doesn't clip the newest turn (which lives at
the tail) out of the viewer.
"""

from app.proxy.routes import _godmode_prompt_window


def test_short_prompt_returned_verbatim():
    text = "hello world"
    out, elided = _godmode_prompt_window(text, max_chars=16000, tail_chars=4000)
    assert out == text
    assert elided is False


def test_exactly_at_limit_is_verbatim():
    text = "x" * 100
    out, elided = _godmode_prompt_window(text, max_chars=100, tail_chars=40)
    assert out == text
    assert elided is False


def test_over_limit_keeps_head_and_tail_with_marker():
    head = "H" * 12000
    middle = "M" * 10000
    tail = "T" * 4000
    text = head + middle + tail
    out, elided = _godmode_prompt_window(text, max_chars=16000, tail_chars=4000)

    assert elided is True
    # Head budget = max_chars - tail_chars = 12000 chars from the front.
    assert out.startswith("H" * 12000)
    # The genuine tail survives in full.
    assert out.endswith("T" * 4000)
    # The elided middle count is len - head_len - tail = 26000 - 12000 - 4000.
    assert "…[10000 chars elided]…" in out
    # Newline-padded marker exactly as the frontend keys on.
    assert "\n\n…[10000 chars elided]…\n\n" in out
    # No 'M' from the dropped middle leaks through.
    assert "M" not in out


def test_tail_zero_is_pure_head_slice_with_marker():
    text = "A" * 100 + "B" * 100
    out, elided = _godmode_prompt_window(text, max_chars=100, tail_chars=0)
    assert elided is True
    assert out.startswith("A" * 100)
    assert out.endswith("\n\n")  # empty tail after the marker
    assert "…[100 chars elided]…" in out


def test_degenerate_tail_larger_than_max_does_not_go_negative():
    # tail_chars >= max_chars must clamp so head_len never goes negative.
    text = "Z" * 500
    out, elided = _godmode_prompt_window(text, max_chars=100, tail_chars=1000)
    assert elided is True
    # tail clamped to max_chars (100) → head empty → keep the last 100 chars.
    assert out.endswith("Z" * 100)
    assert "…[400 chars elided]…" in out
