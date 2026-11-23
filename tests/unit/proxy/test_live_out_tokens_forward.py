"""B1 through the real forward path: a thinking stream's live row counts tokens
of the reasoning channel while it runs, and exposes the decode speed and TTFT."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from app.stats.live_requests import _serialize
from tests.unit.proxy.test_streamed_completion_tokens import _make_fake_tokenizer, _seed_loaded


def test_live_row_counts_reasoning_tokens_mid_stream(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _make_fake_tokenizer(lambda t: len(t) if t else 0)
    seen: list[dict] = []

    async def aiter():
        # Two frames of THINKING only (no content yet), each 10 chars = 10 tokens
        # with the fake tokenizer: the old counter read 0 here.
        yield b'data: {"choices":[{"delta":{"reasoning_content":"0123456789"}}]}\n\n'
        await asyncio.sleep(0.6)
        yield b'data: {"choices":[{"delta":{"reasoning_content":"abcdefghij"}}]}\n\n'
        await asyncio.sleep(0.6)
        yield b'data: {"choices":[{"delta":{"reasoning_content":"k"}}]}\n\n'
        reg = client.app.state.request_registry
        for r in reg.snapshot():
            seen.append(_serialize(r, time.monotonic()))
        yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        yield b"data: [DONE]\n\n"

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "text/event-stream"}
    fake_resp.aiter_bytes = aiter
    fake_resp.aclose = AsyncMock()
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake_resp)):
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "stream": True, "messages": [{"role": "user", "content": "x"}]},
        ) as r:
            b"".join(r.iter_bytes())
    assert len(seen) == 1
    row = seen[0]
    assert row["phase"] == "thinking"
    assert row["completion_tokens"] >= 20  # tokens of the reasoning channel, not frames
    assert row["ttft_s"] is not None and row["ttft_s"] >= 0
    assert row["ttft_total_s"] is not None and row["ttft_total_s"] >= row["ttft_s"]
    assert row["decode_tps_5s"] is not None and row["decode_tps_5s"] > 0
