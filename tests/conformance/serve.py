"""Boot vllm-warden for the @podwarden/chat-ui conformance kit.

Real app, real SQLite, real routes — only the model upstream is faked (the
same ``httpx.AsyncClient.send`` seam the unit tests use), so the kit
exercises the contract end to end without a GPU.

Usage: ``python -m tests.conformance.serve`` (see ``make conformance``).
Prints ``READY <port> <token> <csrf_id> <csrf_token>`` on stdout once
/healthz answers, an admin user and one ``status='loaded'`` model row are
seeded, and a JWT has been minted. The kit's injected ``fetch`` must carry
all three credentials: this backend gates mutating requests behind the
double-submit CSRF check in ``app/auth/csrf.py`` exactly as the real
frontend's ``auth-fetch`` does, and the CSRF token is an HMAC of the
``vw_csrf_id`` cookie, so the cookie has to travel with it.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

PORT = int(os.environ.get("CONFORMANCE_PORT", "18080"))

# The one model row the harness seeds. `GET /models` reports the catalog by
# `served_model_name`, which is what a chat's `model` field holds.
MODEL = "qwen"

# See `_fake_stream.aiter_raw` — few long sleeps, not many short ones.
_CHUNK_BYTES = 64
_CHUNK_DELAY_S = 0.045

# A throwaway data dir per boot: the kit deletes the chats it creates, but
# the defaults row, the ledger and the attachment blobs are all real writes.
data = Path(tempfile.mkdtemp(prefix="vw-conformance-"))

# `app/config.py` has no VW_BIND_PORT override (bind_port is pinned to 8080);
# the listening port is passed to uvicorn directly. That mismatch is harmless
# here — the only consumer of `settings.bind_port` on this path is the
# fire-and-forget title call, and it goes through the patched `send` too, so
# nothing is ever dialled on 8080.
os.environ.update(
    {
        "VW_DATA_DIR": str(data),
        "VW_HF_CACHE_DIR": str(data / "hf"),
        "VW_COOKIE_SECRET": "conformance-secret-32-bytes-min-pad!",
        "VW_CONTAINER_GPU_COUNT": "1",
        "VW_CHAT_QUOTA_FREE_FLOOR_BYTES": "0",
        "VW_FRONTEND_ORIGIN": f"http://127.0.0.1:{PORT}",
        "VW_TRUST_PROXY_ORIGIN": "0",
    }
)


def _sse(*objs: Any, done: bool = True) -> bytes:
    out = b"".join(f"data: {json.dumps(o)}\n\n".encode() for o in objs)
    return out + (b"data: [DONE]\n\n" if done else b"")


def _fake_stream(body: bytes, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {"content-type": "text/event-stream"}

    async def aiter_raw():
        # A real suspension between chunks: the `abort`, `turn-in-flight` and
        # `live-replay` groups all need a turn that is still streaming a beat
        # after it opened.
        #
        # The wall-clock budget is spent in FEW, LONG sleeps rather than many
        # short ones. 7-byte chunks at 5 ms meant ~515 scheduler round-trips
        # for the long reply, and each one is a chance for a loaded runner to
        # overshoot its nominal 5 ms -- the error compounds 515 times and the
        # turn can blow the kit's 10 s per-test cap. 64 bytes at 45 ms buys
        # the same ~2.6 s from ~57 iterations, so the same overshoot costs an
        # order of magnitude less.
        for i in range(0, len(body), _CHUNK_BYTES):
            await asyncio.sleep(_CHUNK_DELAY_S)
            yield body[i : i + _CHUNK_BYTES]

    resp.aiter_raw = aiter_raw
    resp.aread = AsyncMock(return_value=body)
    resp.aclose = AsyncMock()
    return resp


def _last_user_text(payload: dict[str, Any]) -> str:
    """The text of the newest user message.

    `app/chat2/routes_turn.py` sends multimodal content — a LIST of parts,
    not a bare string — so a plain `isinstance(content, str)` read here comes
    back empty for every real turn and the `__fail__` trigger never fires.
    """
    for msg in reversed(payload.get("messages") or []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                p.get("text") or ""
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        return ""
    return ""


def _fake_json(payload: dict[str, Any], status: int = 200) -> MagicMock:
    """A non-streaming upstream reply.

    `app/chat2/routes_turn.py:_llm_title` fires a `stream: false` completion
    on its own task and reads `r.json()`. Handing it the SSE mock made every
    turn spray a `chat2 llm title failed` traceback into the harness log (the
    MagicMock survives `resp["choices"]` but dies at `int(usage[...])`), which
    is noise a real failure would then hide in.
    """
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {"content-type": "application/json"}
    resp.json = MagicMock(return_value=payload)
    resp.aread = AsyncMock(return_value=json.dumps(payload).encode())
    resp.aclose = AsyncMock()
    return resp


def _wants_stream(request, body: dict[str, Any]) -> bool:  # noqa: ANN001
    accept = request.headers.get("accept", "")
    return "text/event-stream" in accept or body.get("stream") is True


async def fake_send(self, request, **kw):  # noqa: ANN001, ANN003, ANN201, ARG001
    body = json.loads(request.content or b"{}")
    if not _wants_stream(request, body):
        return _fake_json(
            {
                "choices": [{"message": {"content": "Conformance title"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            }
        )
    text = _last_user_text(body)
    if "__fail__" in text:
        return _fake_stream(b'{"detail":"rate limit"}', 429)
    # A long prompt earns a long answer, so the mid-flight groups have a
    # stream to act on; a short one is answered briskly.
    words = ["Hello", " from", " the", " conformance", " upstream."] * (8 if len(text) > 200 else 1)
    chunks: list[dict[str, Any]] = [
        {"choices": [{"index": 0, "delta": {"content": w}, "finish_reason": None}]} for w in words
    ]
    chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    chunks.append(
        {
            "choices": [],
            "usage": {"prompt_tokens": 12, "completion_tokens": len(words)},
        }
    )
    return _fake_stream(_sse({"choices": [{"index": 0, "delta": {"role": "assistant"}}]}, *chunks))


def _seed(db: Path) -> None:
    from tests.conftest import seed_admin_user

    seed_admin_user(db)
    c = sqlite3.connect(db)
    c.execute(
        # `supports_vision` is tri-state (NULL = "ask the on-disk HF config",
        # which the harness has none of). Seeding an explicit 1 makes
        # `GET /models` report vision, which is what makes the kit run its
        # attachments group against the real upload/signed-URL flow instead of
        # skipping it.
        "INSERT INTO models(id, served_model_name, hf_repo, gpu_indices, "
        "tensor_parallel_size, status, max_model_len, supports_tools, "
        "supports_vision, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,datetime('now'),datetime('now'))",
        (f"{MODEL}-id", MODEL, f"org/{MODEL}", "[0]", 1, "loaded", 8192, 1, 1),
    )
    c.commit()
    c.close()


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    method: str = "POST",
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _credentials(base: str) -> tuple[str, str, str]:
    """(bearer, csrf_id, csrf_token) for the kit's injected fetch."""
    token = _post_json(f"{base}/api/auth/login", {"username": "admin", "password": "hunter2"})[
        "access_token"
    ]
    with urllib.request.urlopen(f"{base}/api/csrf", timeout=10) as r:
        csrf = json.loads(r.read())["csrf"]
        jar = SimpleCookie()
        for header in r.headers.get_all("Set-Cookie") or []:
            jar.load(header)
    csrf_id = jar["vw_csrf_id"].value

    # A fresh account has no default model, so every chat the kit creates
    # without `fixtures.modelId` would open its turn against `model = null`
    # and get a 404 `model_not_loaded`. Pin the one seeded model through the
    # real PUT /defaults route (the `defaults` group round-trips and restores
    # this row, so it has to be a genuine write, not a DB poke).
    _post_json(
        f"{base}/api/chat2/defaults",
        {"model": MODEL, "settings": {}},
        method="PUT",
        headers={
            "Authorization": f"Bearer {token}",
            "X-CSRF-Token": csrf,
            "Cookie": f"vw_csrf_id={csrf_id}",
        },
    )
    return token, csrf_id, csrf


def _announce() -> None:
    base = f"http://127.0.0.1:{PORT}"
    # 90s, matching the CI job's READY budget. 30s was a full application boot
    # on a runner that also builds images, and it failed on a branch that
    # touched neither this harness nor the app. Raising the CI-side READY wait
    # from 30s to 90s simply moved the failure here: the harness now reaches
    # this line and times out one layer down. Both waits are the same budget
    # for the same boot, so they are now the same number.
    deadline = time.monotonic() + 90
    last_err: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"{base}/healthz", timeout=1)
            break
        except (urllib.error.URLError, OSError) as exc:
            last_err = exc
            time.sleep(0.1)
    else:
        # Say WHY, not just that it happened: "never answered" is the same
        # message whether the port was refused for 90 seconds or the app
        # returned an error, and the two want different fixes.
        print(
            f"FAILED /healthz never answered within 90s on {base} " f"(last error: {last_err!r})",
            flush=True,
        )
        os._exit(1)
    try:
        _seed(data / "vllm-warden.db")
        token, csrf_id, csrf = _credentials(base)
    except Exception as exc:  # noqa: BLE001 — the Make target reads this log
        print(f"FAILED {exc!r}", flush=True)
        os._exit(1)
    print("READY", PORT, token, csrf_id, csrf, flush=True)


def main() -> None:
    import httpx
    import uvicorn

    from app.main import build_app

    app = build_app()
    patch.object(httpx.AsyncClient, "send", new=fake_send).start()
    threading.Thread(target=_announce, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    sys.exit(main())
