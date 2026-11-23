"""Live out-token counting for in-flight streams (stats redesign B1).

The live requests table shows, per running request, how many tokens it has
generated so far and how fast it is generating now. Both used to come from a
count of SSE frames that carried ``delta.content``: a thinking model read 0 for
as long as it thought, a tool call read 0, and an engine that sends several
tokens a frame (speculative decoding) read low.

``LiveOutCounter`` counts TOKENS of every generated channel instead (content,
reasoning, tool-call name and arguments). The caller feeds each frame's text
and, every live tick (0.5 s), asks for the count: only the text that arrived
since the last tick is tokenized, so the cost per tick is the size of the new
text, never the whole reply. When the engine itself reports
``usage.completion_tokens`` on a frame (a client that asked for
``continuous_usage_stats``, or the final usage frame) that figure is a floor:
the count is the larger of it and ours, and per-frame usage skips tokenizing.

The design also allowed asking the engine for continuous usage on every
stream and stripping it from the client's bytes. That rewrites every frame the
client receives; counting on our side leaves the forwarded bytes untouched,
which every other accounting path in the proxy also guarantees.

Fail-open: a tokenizer failure falls back to the frame count, never raises.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

#: The window of the live decode speed, in seconds.
DECODE_WINDOW_S = 5.0


class LiveOutCounter:
    """Tokens generated so far by one stream. Not thread-safe (one stream)."""

    __slots__ = ("_pending", "_counted", "_engine", "_fresh_engine", "frames")

    def __init__(self) -> None:
        self._pending = ""
        self._counted = 0
        self._engine: int | None = None
        #: The engine reported usage since the last count: its figure covers the
        #: pending text, so tokenizing it again would be wasted work.
        self._fresh_engine = False
        #: Frames that carried generated text: the fallback when counting fails.
        self.frames = 0

    def feed(self, text: str, usage: int | None) -> None:
        """One frame's generated text (all channels joined) and its usage
        figure, if the frame carried one."""
        if text:
            self._pending += text
            self.frames += 1
        if usage is not None and usage >= 0:
            self._engine = max(usage, self._engine or 0)
            self._fresh_engine = usage > 0

    @property
    def dirty(self) -> bool:
        return bool(self._pending)

    async def count(self, count_fn: Callable[[str], Awaitable[int]]) -> int:
        """Tokens so far: the larger of the engine's figure and our count (an
        early ``completion_tokens: 0`` frame must not pin the count at 0)."""
        if self._fresh_engine and (self._engine or 0) >= self._counted:
            # Per-frame usage (continuous_usage_stats) covers the pending text:
            # exact, nothing to tokenize.
            self._pending = ""
            self._counted = self._engine or 0
        self._fresh_engine = False
        if self._pending:
            text, self._pending = self._pending, ""
            try:
                self._counted += max(0, int(await count_fn(text)))
            except Exception:  # noqa: BLE001 -- live display only, fail open
                self._counted = max(self._counted, self.frames)
        return max(self._engine or 0, self._counted)


def decode_tps(
    samples: Sequence[tuple[float, int]],
    *,
    first_token: float | None,
    now: float,
    window: float = DECODE_WINDOW_S,
) -> float | None:
    """This request's decode speed over the last ``window`` seconds.

    ``samples`` are ``(monotonic, tokens_so_far)`` readings, oldest first.
    None before the first token. 0.0 when no token arrived for ``window``
    seconds: a stalled request is not still running at its old speed.
    The origin is the latest sample at or before ``now - window``; a request
    younger than that measures from its first token (0 tokens then).
    """
    if first_token is None:
        return None
    if not samples:
        return 0.0 if now - first_token >= window else None
    t_last, n_last = samples[-1]
    if now - t_last >= window:
        return 0.0
    t_ref, n_ref = first_token, 0
    for t, n in samples:
        if t <= now - window:
            t_ref, n_ref = t, n
        else:
            break
    dt = t_last - t_ref
    if dt <= 0:
        return None
    return max(0.0, (n_last - n_ref) / dt)
