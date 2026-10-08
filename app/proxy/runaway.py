"""Stateful, per-request runaway-generation detector for the warden proxy.

Root cause (see docs/superpowers/specs/2026-07-21-runaway-detector-design.md):
the shared prod reasoner opens ``<think>`` by default and sometimes never
terminates the reasoning chain, running toward ``max_model_len`` while pinning a
KV slot for the whole request wall-clock window. A fixed ``max_tokens`` cap would
truncate legitimate long code output equally, so instead this detector watches the
live token stream and trips only on a confident pathology signal.

The detector is **purely textual and synchronous**. It is fed each decoded text
``delta`` (``feed`` returns a :class:`Verdict`); vLLM streams ~1 token per SSE
chunk, so delta-count ≈ token-count and no live tokenizer calls touch the hot
path. Three independent signals, each threshold-gated with conservative defaults:

* **Unclosed-think budget** — once ``<think>`` is seen and ``</think>`` is not yet
  seen, more than ``think_budget`` deltas inside the open block trips
  ``TRIP_THINK``. Tags may split across delta boundaries, so a small tail buffer
  is prepended before every scan.
* **Repetition loop** — a rolling ~``window_chars`` character window with a
  rolling-hash of ``shingle_size``-char shingles; a shingle recurring more than
  ``repeat_max`` times inside the window trips ``TRIP_REPEAT``.
* **Absolute backstop** — more than ``hard_max`` total deltas trips ``TRIP_HARD``.

``feed`` is O(len(delta)) with memory bounded by the window, and the verdict is
sticky: once tripped it is returned unchanged without further evaluation (the
caller tears the generation down on the first trip anyway).
"""

import enum
from collections import deque

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
# Longest tag we scan for; the tail buffer keeps one char less than this so a tag
# split across a delta boundary is always reunited on the next feed while a tag
# lying wholly inside the retained tail can never re-match (see _scan_tags).
_MAX_TAG_LEN = len(_THINK_CLOSE)  # 8
_TAIL_LEN = _MAX_TAG_LEN - 1  # 7

# 61-bit Mersenne modulus keeps the polynomial rolling hash collision-safe for a
# few-thousand-shingle window while staying inside a machine word.
_HASH_BASE = 257
_HASH_MOD = (1 << 61) - 1


class Verdict(enum.Enum):
    OK = "ok"
    TRIP_THINK = "trip_think"
    TRIP_REPEAT = "trip_repeat"
    TRIP_HARD = "trip_hard"


_FINISH_REASON = {
    Verdict.TRIP_THINK: "runaway_think",
    Verdict.TRIP_REPEAT: "runaway_repeat",
    Verdict.TRIP_HARD: "runaway_hard",
}


def finish_reason_for(verdict: Verdict) -> str | None:
    """Map a trip verdict to its OpenAI ``finish_reason`` string (``None`` for OK)."""
    return _FINISH_REASON.get(verdict)


class RunawayDetector:
    def __init__(
        self,
        *,
        think_budget: int = 24000,
        repeat_max: int = 6,
        hard_max: int = 96000,
        shingle_size: int = 48,
        window_chars: int = 4096,
    ):
        self.think_budget = think_budget
        self.repeat_max = repeat_max
        self.hard_max = hard_max
        self.shingle_size = shingle_size
        self.window_chars = window_chars

        # Delta / think accounting.
        self.total = 0
        self.think_open = False
        self.think_tokens = 0
        self._think_block_start = 0
        self._tail = ""

        # Repetition state — a rolling-hash over a bounded char window.
        self._win: deque[int] = deque()  # char codes currently in the window
        self._shingles: deque[int] = deque()  # hashes of the live (in-window) shingles
        self._counts: dict[int, int] = {}
        self._codes: deque[int] = deque()  # char codes of the current shingle suffix
        self._h = 0
        self._bpow = pow(_HASH_BASE, shingle_size, _HASH_MOD)

        self.verdict = Verdict.OK

    @classmethod
    def from_settings(cls, settings) -> "RunawayDetector":
        return cls(
            think_budget=settings.runaway_think_budget,
            repeat_max=settings.runaway_repeat_max,
            hard_max=settings.runaway_hard_max,
        )

    @property
    def tripped(self) -> bool:
        return self.verdict is not Verdict.OK

    @property
    def finish_reason(self) -> str | None:
        return finish_reason_for(self.verdict)

    def feed(self, delta: str) -> Verdict:
        # Sticky: the caller tears down on the first trip; any later feed just
        # echoes the terminal verdict without re-scanning.
        if self.verdict is not Verdict.OK:
            return self.verdict
        # Role-only / empty content chunks carry no generated token — ignore them
        # so they neither count toward a budget nor perturb the repetition window.
        if not delta:
            return Verdict.OK

        self.total += 1

        # --- think open/close scan, robust to tags split across deltas --------
        was_open = self.think_open
        combined = self._tail + delta
        self._scan_tags(combined, len(self._tail))
        self._tail = combined[-_TAIL_LEN:]

        # --- unclosed-think budget -------------------------------------------
        # Count a delta as "inside" only when the block was open for its whole
        # duration — the delta carrying the open/close marker itself is a
        # boundary, not generated reasoning content.
        if was_open and self.think_open:
            self.think_tokens += 1
        if self.think_open and (self.total - self._think_block_start) > self.think_budget:
            self.verdict = Verdict.TRIP_THINK
            return self.verdict

        # --- repetition loop --------------------------------------------------
        if self._feed_repetition(delta):
            self.verdict = Verdict.TRIP_REPEAT
            return self.verdict

        # --- absolute backstop ------------------------------------------------
        if self.total > self.hard_max:
            self.verdict = Verdict.TRIP_HARD
            return self.verdict

        return Verdict.OK

    def _scan_tags(self, combined: str, tail_len: int) -> None:
        """Apply every ``<think>`` / ``</think>`` occurrence in ``combined`` in
        positional order. A match is acted on only when it ends past ``tail_len``
        — i.e. it extends into the new delta — so a tag already consumed on a
        previous feed (and still lingering in the retained tail) is not counted
        twice."""
        events: list[tuple[int, bool]] = []
        for tag, is_open in ((_THINK_OPEN, True), (_THINK_CLOSE, False)):
            start = 0
            while True:
                p = combined.find(tag, start)
                if p == -1:
                    break
                if p + len(tag) > tail_len:
                    events.append((p, is_open))
                start = p + 1
        events.sort()
        for _pos, is_open in events:
            if is_open:
                if not self.think_open:
                    self.think_open = True
                    self._think_block_start = self.total
            else:
                self.think_open = False

    def _feed_repetition(self, delta: str) -> bool:
        """Slide every char of ``delta`` through the rolling window; return True
        as soon as a shingle recurs more than ``repeat_max`` times in-window."""
        tripped = False
        for ch in delta:
            code = ord(ch)
            self._win.append(code)

            # Extend the rolling hash by one char; drop the oldest once the
            # shingle suffix is full so ``self._h`` is always the hash of the
            # trailing ``shingle_size`` chars.
            self._h = (self._h * _HASH_BASE + code) % _HASH_MOD
            self._codes.append(code)
            if len(self._codes) > self.shingle_size:
                old = self._codes.popleft()
                self._h = (self._h - old * self._bpow) % _HASH_MOD
            if len(self._codes) == self.shingle_size:
                h = self._h
                self._shingles.append(h)
                self._counts[h] = self._counts.get(h, 0) + 1
                if self._counts[h] > self.repeat_max:
                    tripped = True

            # Evict from the left once the window overflows. Each char dropped
            # from the window retires exactly the leftmost shingle (they slide
            # in lockstep once the window is full).
            if len(self._win) > self.window_chars:
                self._win.popleft()
                if self._shingles:
                    gone = self._shingles.popleft()
                    n = self._counts.get(gone, 0) - 1
                    if n <= 0:
                        self._counts.pop(gone, None)
                    else:
                        self._counts[gone] = n
        return tripped
