from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from app.chat2.limits import TITLE_MAX_CHARS

# Spec §4.2 step 9: the title call runs at the LOWEST priority — it is
# cosmetic, and must never sit in front of a user's interactive turn.
#
# vLLM's priority scheduler orders its waiting queue by (priority, arrival)
# ASCENDING, so a HIGHER number is served LATER, and the proxy maps the
# warden's 0..9 token priority (9 = highest) onto `vllm_priority =
# -warden_priority`, i.e. the band -9..0 (app/proxy/routes.py). It also never
# overrides a priority the client set itself. Sending +9 therefore parks the
# title request behind every possible warden-mapped request, including
# default (priority-0) traffic — which a plain `0` would only tie with.
LOWEST_PRIORITY = 9

# How many trailing user/assistant messages the reassessment prompt sees. Six
# is two-to-three exchanges: enough for the model to notice the conversation
# has moved on from its opening question, short enough that the title call
# stays cheap (it is billed to the user like any other turn).
TITLE_CONTEXT_MESSAGES = 6

_PROMPT = (
    "Read the conversation and reply with a topic title of 2 to 5 words for it. "
    "Reply with the title only, no quotes, no punctuation at the end."
)


def first_line_title(text: str, limit: int = TITLE_MAX_CHARS) -> str:
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    if not line:
        return "New chat"
    return line if len(line) <= limit else line[: limit - 1] + "…"


def normalized_title(title: str) -> str:
    """The form two titles are compared in before one replaces the other.

    A reassessment that comes back with the same title in different clothes
    ("Deploy  Checklist") must not count as a rename: it would push the real
    previous title out of `title_prev` and turn the undo affordance into a
    no-op.
    """
    return " ".join(title.split()).casefold()


def title_context(rows: Sequence[Any], limit_chars: int = 2000) -> str:
    """Render the tail of a conversation as the title prompt's user message.

    Walks backwards so the *newest* turns get the character budget — an early
    wall-of-text pasted into turn one must not crowd out the exchange that
    actually shows what the chat is about now. The result is re-ordered
    oldest-first so the model reads it as a conversation.
    """
    picked: list[str] = []
    budget = limit_chars
    for r in reversed(rows):
        if len(picked) >= TITLE_CONTEXT_MESSAGES or budget <= 0:
            break
        if r.role not in ("user", "assistant"):
            continue
        text = " ".join(str(p.get("text", "")) for p in r.parts if p.get("type") == "text").strip()
        if not text:
            continue
        line = f"{r.role.capitalize()}: {text}"[:budget]
        picked.append(line)
        budget -= len(line) + 1  # + the newline this line will be joined with
    return "\n".join(reversed(picked))


async def generate_llm_title(
    *,
    post_json: Callable[[dict[str, Any]], Awaitable[dict[str, Any] | None]],
    model: str,
    context_text: str,
) -> str | None:
    payload = {
        "model": model,
        "stream": False,
        "max_tokens": 16,
        "temperature": 0.2,
        "priority": LOWEST_PRIORITY,
        # A reasoning model would spend all 16 tokens on its preamble and hand
        # back an empty (or null) completion, so the title call never produces
        # a title on exactly the models people most want titles for. This is
        # the same knob the per-chat `enable_thinking: false` setting uses.
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": _PROMPT},
            {"role": "user", "content": context_text},
        ],
    }
    resp = await post_json(payload)
    if not resp:
        return None
    try:
        content = resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    # `str(content)` on a null completion is the *string* "None", which sails
    # through a truthiness check and renames the chat to the literal "None" --
    # displacing a perfectly good title into `title_prev` on the way. An
    # unusable completion has to mean "no rename", so anything that is not a
    # non-empty string is rejected outright.
    if not isinstance(content, str):
        return None
    title = content.strip().strip('"').strip()
    return first_line_title(title) if title else None
