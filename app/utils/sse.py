"""Helpers for Server-Sent Events (SSE) responses.

Centralised here so every ``StreamingResponse(media_type="text/event-stream")``
in the codebase gets the same anti-buffering headers without each route
re-inventing them — a regression that would let one new endpoint silently
re-introduce the proxy-buffering bug fixed for log streams in #50.

See ``sse_headers`` below for the rationale on each header.
"""

from __future__ import annotations

from typing import Any


def sse_headers() -> dict[str, str]:
    """Return headers that defeat reverse-proxy buffering of SSE streams.

    nginx (and Caddy/Traefik, which honour the same hint) buffer
    ``text/event-stream`` responses by default, causing the client to see
    a wall of buffered chunks at proxy-buffer intervals rather than the
    intended real-time stream. ``X-Accel-Buffering: no`` disables this on
    nginx and its lookalikes; ``Cache-Control: no-cache`` is a defensive
    cover for CDNs/proxies that don't recognise the SSE media type and
    would otherwise apply default caching to the response body.

    Callers should pass the returned dict to ``StreamingResponse`` via
    ``headers=sse_headers()``. If the endpoint needs additional headers,
    merge with ``{**sse_headers(), **extra}`` so the buffering hints
    cannot be accidentally clobbered.
    """
    return {
        "X-Accel-Buffering": "no",
        "Cache-Control": "no-cache",
    }


def delta_reasoning(delta: dict[str, Any]) -> str | None:
    """The reasoning channel of a chat delta, whichever name the engine uses.

    vLLM's ``--reasoning-parser`` output has streamed as
    ``delta.reasoning_content``; newer builds name it ``delta.reasoning``
    (prod finding 2026-09-27: qwen3.8-27b-fp8-model on v2026.09.27.1 emits
    ``{"delta": {"reasoning": "..."}}`` with no ``reasoning_content`` key).
    Every reader of the channel — the completion accounting, the runaway
    detector, god mode, the non-stream re-aggregation, the stress probes and
    the chat normaliser — must go through this helper so neither name is ever
    missed.     When both are present (they are not in practice) the traditional
    ``reasoning_content`` wins. An empty string counts as absent: a delta
    that carried no reasoning text must not be mistaken for one that did.
    """
    return delta.get("reasoning_content") or delta.get("reasoning")


def delta_reasoning_field(delta: dict[str, Any]) -> str:
    """The name the engine used for the reasoning channel in ``delta``.

    Companion to :func:`delta_reasoning` for the one consumer that must emit
    the channel under the engine's OWN name — the non-stream re-aggregation,
    whose output has to match what the engine's non-stream endpoint would
    have produced. It resolves the same way the value does (first
    non-empty name wins, ``reasoning_content`` preferred), so the name and
    the value always agree about which field carried the text. Callers only
    use it when :func:`delta_reasoning` returned something.
    """
    if delta.get("reasoning_content"):
        return "reasoning_content"
    return "reasoning"
