"""Guard leaks found by the second live test (2026-10-30, v2026.10.30.1): a value repeated in
the same first request, values the session matcher split apart, and base64 right after `=`."""

from __future__ import annotations

import base64
import json

from app.guard.config import compile_guard
from app.guard.proxy import SeenSet, run_guard
from app.guard.scanner import scan_text
from tests.unit.guard.corpus import github_token

SECRET = b"k" * 32
G = compile_guard({"mode": "sanitize"})


def _run(msgs, seen, session="s"):
    body = {"model": "m", "messages": [{"role": r, "content": c} for r, c in msgs]}
    return run_guard(body, "openai", G, token_id="t", secret=SECRET, session=session, seen=seen)


def _text(out) -> str:
    return json.dumps(out.body, ensure_ascii=False)


def test_a_value_repeated_in_the_same_first_request_is_caught_everywhere():
    pw = "Zx9qLm2vB7nR4tKw"
    out = _run(
        [
            ("user", f"Remember: the db password is {pw}."),
            ("assistant", "Noted."),
            ("user", f".env:\nDB_PASSWORD={pw}\nReply with just OK."),
        ],
        SeenSet(),
    )
    assert pw not in _text(out)
    assert {h.where.index for h in out.findings} == {1, 3}


def test_values_next_to_delimiters_are_caught_again_later_in_the_session():
    pw = "Qm7vLx2pRt9k"
    phrase = "velvet orbit canyon marble tulip"
    key = base64.b64encode(bytes(range(64))).decode()  # an Azure storage key: 88 chars ending '=='
    seen = SeenSet()
    first = _run(
        [
            (
                "user",
                f'DB_PASSWORD={pw}\nDB_PASSPHRASE="{phrase}"\nAZ="DefaultEndpointsProtocol=https;AccountKey={key};EndpointSuffix=core.windows.net"',
            )
        ],
        seen,
    )
    assert pw not in _text(first) and phrase not in _text(first) and key not in _text(first)
    later = _run(
        [
            ("user", f"connect with sqlplus app/{pw}@ORCL"),
            ("user", f"the passphrase is {phrase}, keep it"),
            ("user", f"key: {key};"),
        ],
        seen,
    )
    text = _text(later)
    assert pw not in text and phrase not in text and key not in text


def test_base64_right_after_an_equals_sign_is_decoded():
    blob = base64.b64encode(f"GITHUB_TOKEN={github_token()}\n".encode()).decode()
    for text in (f"ENV_B64={blob}", f"export ENCODED={blob}"):
        assert [f.value for f in scan_text(text, G.rules)] == [blob]


def test_a_known_value_deep_inside_one_long_minified_line_is_caught():
    pw = "Zx9qLm2vB7nR4tKw"
    seen = SeenSet()
    _run([("user", f"DB_PASSWORD={pw}")], seen)
    line = "a=1;" * 50_000 + f'cfg.db="{pw}";' + "b=2;" * 1000
    out = _run([("tool", line)], seen)
    assert pw not in _text(out)
