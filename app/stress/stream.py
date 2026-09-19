"""SSE accumulation for engine-direct probes (design §9.1).

Probes go straight to ``127.0.0.1:{engine_port}``, not through the warden's
own ``/v1``. That is not a shortcut: ``PriorityScheduler`` admits
``VW_PROXY_MAX_INFLIGHT`` — default 16 — per engine, so a concurrency run
through the proxy could never test past 16 and would be measuring the warden
rather than the engine. Going direct also sidesteps the priority push-down and
``runaway_mode``'s body rewrite.

The cost is that none of the proxy's streaming machinery is reusable:
``warmup_probe`` is a single non-streaming POST, and ``_feed_detector`` takes
an event already parsed by the proxy's forward path. So the harness needs its
own SSE client. Parsing lives here, separated from transport, so the whole of
it is testable by feeding lines with no network and no GPU.
"""
from __future__ import annotations

import json

from app.stress.gates import ProbeObservation


class StreamAccumulator:
    """Accumulates one OpenAI-compatible streaming response.

    ``detector`` is an optional ``RunawayDetector``. It is None for probes
    whose output is legitimately repetitive — the echo probe repeats a token on
    purpose, and the graded probes emit short structured answers — because the
    detector emits shingles at every character offset and would manufacture
    failures out of correct output.
    """

    def __init__(self, detector=None) -> None:
        self.detector = detector
        self.content = ""
        self.reasoning_content = ""
        self.finish_reason: str | None = None
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        #: Tokens the engine served from its prefix cache. Recorded because
        #: it makes the unique-preamble defeat CHECKABLE rather than a
        #: precaution taken on faith: 50 of 54 tokens came from cache on a
        #: repeated prompt (llama-server b10731, 2026-09-03), so a high
        #: value on a probe means the concurrency axis is measuring
        #: deduplicated KV and its number is meaningless.
        self.cached_tokens: int | None = None
        self.saw_done = False
        self.malformed_frames = 0
        self._delta_count = 0
        self._think_open = False

    # -- feeding ----------------------------------------------------------

    def feed_line(self, line: str) -> None:
        """Consume one raw SSE line.

        A malformed frame is counted and skipped rather than raised: one bad
        frame must not abort a probe that is otherwise fine, or a transient
        encoding hiccup would be recorded as a model limit.
        """
        line = line.strip()
        if not line or line.startswith(":"):
            return
        if not line.startswith("data:"):
            return
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            self.saw_done = True
            return
        try:
            ev = json.loads(payload)
        except (ValueError, TypeError):
            self.malformed_frames += 1
            return
        self._consume(ev)

    def _consume(self, ev: dict) -> None:
        usage = ev.get("usage")
        if isinstance(usage, dict):
            # The engine's own count. Never our tokenizer's: TokenizerCache
            # joins message text without the chat template, so role tokens,
            # BOS and the generation prompt go uncounted and it systematically
            # undercounts.
            if usage.get("prompt_tokens") is not None:
                self.prompt_tokens = int(usage["prompt_tokens"])
            if usage.get("completion_tokens") is not None:
                self.completion_tokens = int(usage["completion_tokens"])
            details = usage.get("prompt_tokens_details")
            if isinstance(details, dict) and details.get("cached_tokens") is not None:
                self.cached_tokens = int(details["cached_tokens"])

        choices = ev.get("choices")
        if not isinstance(choices, list):
            return
        prim = next(
            (c for c in choices if isinstance(c, dict) and c.get("index", 0) == 0),
            None,
        )
        if prim is None:
            return

        if prim.get("finish_reason"):
            self.finish_reason = prim["finish_reason"]

        delta = prim.get("delta")
        if not isinstance(delta, dict):
            return

        reasoning = delta.get("reasoning_content")
        content = delta.get("content")

        # Mirrors app/proxy/routes.py::_feed_detector, and for the same reason:
        # --reasoning-parser qwen3 strips the literal <think>/</think> tags and
        # streams the chain as reasoning_content. The detector's unclosed-think
        # signal is purely textual, so without synthesising the tags an
        # over-reasoning generation — precisely the pathology being hunted —
        # would never open a think block and the signal would be dead.
        if reasoning:
            self.reasoning_content += reasoning
            self._delta_count += 1
            if self.detector is not None:
                if not self._think_open:
                    self.detector.feed("<think>")
                    self._think_open = True
                self.detector.feed(reasoning)

        if content:
            self.content += content
            self._delta_count += 1
            if self.detector is not None:
                if self._think_open:
                    self.detector.feed("</think>")
                    self._think_open = False
                self.detector.feed(content)

    # -- verdicts ---------------------------------------------------------

    @property
    def repetition_tripped(self) -> bool:
        return bool(self.detector is not None and self.detector.tripped)

    def observation(self, *, graded_ok: bool | None) -> ProbeObservation:
        """The gate-ready view of what this probe produced.

        ``finish_reason`` stays None when the stream ended without a terminal
        frame, which the ABRUPT gate reads as a defect — correctly, since the
        wall-clock reaper and the disconnect path both end a stream that way.

        ``output_tokens`` falls back to a delta count when the engine omitted
        usage, because the SHORT gate needs a length and an approximation beats
        nothing. It is never used for the published prompt size.
        """
        return ProbeObservation(
            content=self.content,
            reasoning_content=self.reasoning_content,
            finish_reason=self.finish_reason,
            output_tokens=(
                self.completion_tokens
                if self.completion_tokens is not None
                else self._delta_count
            ),
            repetition_tripped=self.repetition_tripped,
            graded_ok=graded_ok,
        )
