"""Streaming /v1/chat/completions client with per-request timings.

A second, cookie-free httpx client (bearer auth only) -- the admin session's
cookie jar has no business on the data plane, and `/v1/*` is CSRF-exempt
anyway.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import httpx


@dataclass
class RequestTiming:
    label: str
    t_send: float = 0.0
    t_headers: float | None = None
    t_first_frame: float | None = None
    t_first_content: float | None = None
    t_done: float | None = None
    status: int | None = None
    finish_reason: str | None = None
    n_frames: int = 0
    usage: dict | None = None
    error: str | None = None


class OpenAIClient:
    def __init__(self, base_url: str, *, verify: bool | str = True) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            verify=verify,
            limits=httpx.Limits(max_connections=64),
            timeout=httpx.Timeout(connect=10, read=None, write=30, pool=None),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def stream_chat(
        self,
        *,
        label: str,
        token: str,
        model: str,
        prompt: str,
        max_tokens: int,
        priority_seed: int,
        ignore_eos: bool = False,
        temperature: float = 0.0,
    ) -> RequestTiming:
        timing = RequestTiming(label=label)
        body: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "seed": priority_seed,
        }
        if ignore_eos:
            body["ignore_eos"] = True
        timing.t_send = time.monotonic()
        try:
            async with self._client.stream(
                "POST",
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {token}"},
                json=body,
            ) as resp:
                timing.t_headers = time.monotonic()
                timing.status = resp.status_code
                if resp.status_code != 200:
                    raw = await resp.aread()
                    timing.error = raw.decode(errors="replace")[:500]
                    return timing
                async for raw_line in resp.aiter_lines():
                    if not raw_line.startswith("data:"):
                        continue
                    now = time.monotonic()
                    if timing.t_first_frame is None:
                        timing.t_first_frame = now
                    timing.n_frames += 1
                    payload = raw_line[len("data:") :].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        if delta.get("content") and timing.t_first_content is None:
                            timing.t_first_content = now
                        fr = choices[0].get("finish_reason")
                        if fr:
                            timing.finish_reason = fr
                    if chunk.get("usage"):
                        timing.usage = chunk["usage"]
        except httpx.HTTPError as exc:
            # I6: exception TYPE only -- httpx.HTTPError's str() frequently
            # embeds the full request URL (hostname), which must never reach
            # a report or a failure message.
            timing.error = type(exc).__name__
        finally:
            timing.t_done = time.monotonic()
        return timing
