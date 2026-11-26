"""Stage 1 canary (A-10, A-3): a sentinel in the system prompt, a tool result
and the reply never reaches a lens column, a log line, or either findings
route's answer. The session runs through the real proxy (kvsig on)."""

from __future__ import annotations

import json
import logging
from typing import Any

from app.sessions import store
from tests.conftest import jwt_login
from tests.unit.proxy.test_lens_hot_path import SESSION, _post, _rows, _sys
from tests.unit.proxy.test_prompt_lens_forward import _ready_gm
from tests.unit.stats.test_forest_routes import _forest_login

CANARY = "CANARY-8e2c41f7-lens"
ME = "/api/me/prompt-lens/requests/{}/findings"
ADMIN = "/api/admin/prompt-lens/requests/{}/findings"
#: Every column the lens added or reads (0045, 0055, 0063), and the shape and calls.
LENS_COLUMNS = (*store._COLS[store._COLS.index("lens_break") :], "tools_out", "cobs2_shape")


def run_session_with_canary(client: Any, plaintext: str) -> None:
    u = {"role": "user", "content": "list the files"}
    said = f"reading it {CANARY}"
    _post(
        client,
        plaintext,
        {"model": "qwen", "messages": [_sys("You help.", f"dir /w/{CANARY}/1"), u]},
        SESSION,
        reply=said,
    )
    call = {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
    t2 = [
        _sys("You help.", f"dir /w/{CANARY}/2"),
        u,
        {"role": "assistant", "content": said, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "c1", "content": f"file text {CANARY}"},
        {"role": "user", "content": "and now"},
    ]
    _post(client, plaintext, {"model": "qwen", "messages": t2}, SESSION, reply=f"done {CANARY}")


def test_canary_never_in_findings_columns_or_logs(
    tmp_data_dir: Any, client: Any, caplog: Any
) -> None:
    caplog.set_level(logging.DEBUG)
    plaintext = _ready_gm(client, tmp_data_dir)
    run_session_with_canary(client, plaintext)
    rows = _rows(tmp_data_dir, 2)
    assert CANARY not in json.dumps([{c: r[c] for c in LENS_COLUMNS} for r in rows])
    _q, snap = client.app.state.prompt_lens_hub.subscribe()
    hub_ids = {e["request_id"]: e["req_id"] for e in snap if e.get("type") == "request_start"}
    me, admin = _forest_login(client, plaintext), jwt_login(client)
    for r in rows:
        # The Prompt Lens id (what the request view holds) and the stored id.
        for rid in (hub_ids[r["id"]], r["id"]):
            a = client.get(ME.format(rid), headers=me)
            b = client.get(ADMIN.format(rid), headers=admin)
            assert a.status_code == b.status_code == 200, (a.text, b.text)
            assert CANARY not in a.text and CANARY not in b.text
    assert CANARY not in caplog.text
