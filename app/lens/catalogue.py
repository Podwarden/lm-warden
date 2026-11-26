"""Prompt Lens recommendation catalogue (analysis spec 11).

Every finding comes from one of these rules. The copy (message, fix), the
threshold and guard text and the leanprompt ids are the spec's tables 11.1-11.3
verbatim; ``scored`` says whether the rule's term counts in its layer's score
(spec 10.2). A ``Finding`` holds codes and numbers only (A-10): turn numbers,
counts, a tool hash, a cause code and the saving.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

Layer = Literal["E", "C", "R"]
Confidence = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class Rule:
    id: str
    layer: Layer
    #: The leanprompt rule(s) that explain the fix, e.g. "PC-03, PC-02"; None: none.
    lp: str | None
    #: The rule's term counts in its layer's score (spec 10.2).
    scored: bool
    message: str
    fix: str
    #: Spec text, for the docs and the UI's "why": when it fires and what stops it.
    threshold: str = ""
    guard: str = ""


def _r(
    id: str,
    lp: str | None,
    scored: bool,
    message: str,
    fix: str,
    threshold: str,
    guard: str,
) -> Rule:
    layer: Layer = "E" if id[0] == "E" else "C" if id[0] == "C" else "R"
    return Rule(id, layer, lp, scored, message, fix, threshold, guard)


_RULES: tuple[Rule, ...] = (
    # -- 11.1 Efficiency ---------------------------------------------------------
    _r(
        "E-01",
        "PC-03, PC-02",
        True,
        "System prompt line {n} changes every {turn/session} ({class}); "
        "the {m} lines after it are recomputed",
        "make the value stable (a day, not a time; a fixed path) or move it to the "
        "newest user message; re-check answers (A-6)",
        "≥ 2 turns or ≥ 2 sessions, lost ≥ 1,024 tokens",
        "the line is the only change across a day boundary (once a day is expected, v1 9.3); "
        "class `other` → message without class, low confidence",
    ),
    _r(
        "E-02",
        "PC-04",
        True,
        "The same tools were sent in a different order (or with re-ordered JSON keys)",
        "sort tools; serialise schemas deterministically",
        "≥ 1 turn, lost ≥ 1,024",
        "none needed: equal multiset of definition hashes is proof",
    ),
    _r(
        "E-03",
        "PC-05",
        True,
        "Tool {name/#k} was added, removed or edited mid-list; everything after it is recomputed",
        "append new tools at the end; load deferred tools last",
        "≥ 2 turns",
        "a toolset change that coincides with a new prompt version (13.2) is reported as a "
        "version, not a fault",
    ),
    _r(
        "E-04",
        "PC-06",
        True,
        "Message {i}, sent before, came back different ({probe})",
        "append-only history; do not re-serialise old messages",
        "≥ 2 turns",
        "an explicit user edit (fork from a user message) is not counted",
    ),
    _r(
        "E-05",
        "PC-09",
        True,
        "Reasoning is kept on some turns and dropped on others",
        "keep or drop it the same way every turn; fix the thinking setting per session",
        "≥ 2 switches",
        "never advise dropping (A-6)",
    ),
    _r(
        "E-06",
        "PC-01",
        True,
        "Anthropic caches only at breakpoints; this session has none",
        "add `cache_control` marks or turn on automatic marks (#341)",
        "relayed to Anthropic, ≥ 3 turns, prefix ≥ provider minimum",
        "below the provider minimum → E-08 instead",
    ),
    _r(
        "E-07",
        "PC-08",
        True,
        "Compaction also changed the system prompt or tools, so nothing was reused after it",
        "keep sys and tools byte-identical, append the summary",
        "≥ 1",
        "none",
    ),
    _r(
        "E-08",
        "PC-11",
        False,
        "Your prompt was stable; the cache expired during pauses of {p}",
        "keep pauses inside the cache lifetime or expect a cold prefix",
        "≥ 2",
        "relayed turns excluded (`pauses.py`)",
    ),
    _r(
        "E-09",
        "PC-10",
        True,
        "An image or attachment in message {i} changed or moved",
        "stable placement and encoding",
        "≥ 2",
        "none",
    ),
    # -- 11.2 Construction -------------------------------------------------------
    _r(
        "C-01",
        "PC-05",
        True,
        "{k} of {n} tools were never called; their definitions add {t} tokens to every turn",
        "drop them for this app, or load them on demand; LM Warden's prompt diet can strip "
        "them for this key",
        "unused schema ≥ 1,000 tokens per turn or ≥ 5 % of the prompt, session ≥ 10 turns",
        "a tool called in any of the key's last 20 sessions with this toolset is "
        '"rarely used" (info), not unused; agent and MCP toolsets shared by design get the '
        'advice "load on demand", not "remove"',
    ),
    _r(
        "C-02",
        "PC-06",
        True,
        "The same {family} result ({t} tokens) was sent {k} times",
        "keep one copy; tell the model the file is already in context; cache reads in the harness",
        "≥ 2 repeats and ≥ 500 tokens",
        "the content changed between reads (hash differs) is not a repeat; results under 500 "
        "tokens ignored",
    ),
    _r(
        "C-03",
        "PC-02",
        False,
        "The system prompt and tools are {s} % of every request",
        "shorten, or make sure it is cached (if E is high this is info: it costs context room "
        "and first-turn time, not money)",
        "sys + tools ≥ 70 % of the prompt on the median turn, and ≥ 8,000 tokens",
        "severity drops to info when the head is reused on ≥ 90 % of turns",
    ),
    _r(
        "C-04",
        "GE-03",
        False,
        "Tool results are {s} % of the context; the largest ({family}) is {t} tokens",
        "truncate or summarise large results; paginate reads",
        "tool results ≥ 70 % of body growth and one result ≥ 8,000 tokens",
        "results the model then quoted at length are still flagged (we cannot tell use) → "
        "medium confidence at most",
    ),
    _r(
        "C-05",
        "PC-08",
        False,
        "Context grows {g} tokens per turn; the window fills by turn {n} / "
        "your harness compacts at {p} % of the window",
        "trim tool results, compact later or with a cache-aware summary",
        "projected to fill 80 % of `max_model_len` within the session's length, or compaction "
        "before 50 % of the window",
        "sessions under 5 turns get no projection",
    ),
    _r(
        "C-06",
        "PC-03",
        False,
        "Line {n} of {m} changes; the lines after it are stable",
        "as E-01",
        "as E-01",
        "as E-01",
    ),
    _r(
        "C-07",
        "PC-06",
        True,
        "Reminder and environment blocks are {s} % of the prompt",
        "shorten or stop repeating them",
        "`boil` ≥ 10 % of the prompt",
        "often the harness's, not the app's: the message says so",
    ),
    _r(
        "C-08",
        "PC-05",
        False,
        "Tool {name/#k}'s definition is {t} tokens, {x} × the others",
        "shorten the description, move long docs out",
        "one tool ≥ 3 × the median definition and ≥ 2,000 tokens",
        'a tool called often is still flagged but marked "used on {k} turns"',
    ),
    _r(
        "C-09",
        "PC-05",
        False,
        "{n} tools; OpenAI accepts at most 128",
        "split or load on demand",
        "≥ 100 tools to a provider with a 128 limit",
        "only on routes with a modelled limit",
    ),
    # -- 11.3 Response -----------------------------------------------------------
    _r(
        "R-01",
        "GE-03, GE-05",
        True,
        "{k} replies were cut at max_tokens = {m}; the session's p90 reply is {p}",
        "raise the limit for this step, or lower the thinking budget",
        "≥ 2 turns or ≥ 10 % of turns",
        "`max_tokens` ≤ 16 (probes, classifiers) ignored; a cut followed by a `continue` the "
        "client asked for is not a retry",
    ),
    _r(
        "R-02",
        "PC-08",
        True,
        "The prompt filled the context window at turn {n}",
        "see C-05",
        "≥ 1",
        "none",
    ),
    _r(
        "R-03",
        "GE-03",
        True,
        "The warden stopped a {thinking/repetition} loop on turn {n}",
        "lower thinking budget; check the prompt does not ask to repeat",
        "≥ 1",
        "reported even once (it is rare and costly), but recommendation only from 2",
    ),
    _r(
        "R-04",
        "GE-10",
        True,
        "{tool} was called with {code} {k} times of {n}; {cause from 8.2}",
        "add the format and an example to the description; mark required parameters; type "
        "every parameter",
        "≥ 3 calls or ≥ 20 % of that tool's calls",
        "`leaked_markup` is an engine setting (admin), not the prompt: separate admin message",
    ),
    _r(
        "R-05",
        "GE-05",
        True,
        "{tool} was called {k} times with the same arguments and got the same result",
        "make the result say what changed or that the task is done; cap retries in the harness",
        "the same tool and arguments ≥ 3 times with identical results",
        "different results in between (polling, waiting) → not a loop",
    ),
    _r(
        "R-06",
        "GE-05",
        True,
        "{tool} failed {k} times in a row on turns {list}; the error results are {len} characters",
        "return errors that say what to change; describe valid arguments",
        "≥ 2 consecutive failed calls of one tool, in ≥ 2 runs or ≥ 3 total",
        "errors on OpenAI-format results are heuristic → medium confidence at most",
    ),
    _r(
        "R-07",
        "GE-03",
        True,
        "{k} replies had no text and no tool call",
        "per 8.2 cause",
        "≥ 2 or ≥ 5 %",
        "a reply with only tool calls is not empty",
    ),
    _r(
        "R-08",
        "GE-03",
        False,
        "{k} turns thought for {r} tokens to answer in under 40",
        "a lower thinking budget for this step, **with an answer check**",
        "≥ 3 turns",
        "info only, never in R; low confidence (A-6)",
    ),
    _r(
        "R-09",
        None,
        False,
        "{k} values matched rule {rule} in {role}; the model saw placeholders",
        "keep secrets out of the prompt at the source",
        "≥ 1 sanitised value in the prompt",
        "`restored` actions are the guard working, not a finding",
    ),
    _r(
        "R-10",
        "PC-02",
        False,
        "First token took {s} s: {a} s recomputing what a stable prompt would have reused, "
        "{b} s queue",
        "the E finding if recompute dominates; otherwise none (server load, A-11)",
        "median TTFT ≥ 2 s on turns of ≥ 1,000 new tokens",
        "queue share > 50 % → no prompt advice",
    ),
)

RULES: Mapping[str, Rule] = MappingProxyType({r.id: r for r in _RULES})


@dataclass(frozen=True)
class Saving:
    """What fixing a finding saves over the session: tokens always; seconds
    with a learned rate; money only with a price card (spec 5.4)."""

    tokens: int
    prefill_s: float | None
    decode_s: float | None
    money: float | None

    def to_json(self) -> dict[str, Any]:
        return {
            "tokens": int(self.tokens),
            "prefill_s": self.prefill_s,
            "decode_s": self.decode_s,
            "money": self.money,
        }


@dataclass
class Finding:
    """One fired rule. ``turns`` are 1-based turn numbers of the session;
    ``tool`` is a tool-name hash (never a name); ``cause`` a cause code."""

    rule: str
    turns: tuple[int, ...]
    counts: dict[str, int]
    tool: str | None
    cause: str | None
    saving: Saving
    confidence: Confidence
    muted: bool = False

    def to_json(self) -> dict[str, Any]:
        """Codes and numbers only (A-10)."""
        return {
            "rule": self.rule,
            "turns": [int(t) for t in self.turns],
            "counts": {str(k): int(v) for k, v in self.counts.items()},
            "tool": self.tool,
            "cause": self.cause,
            "saving": self.saving.to_json(),
            "confidence": self.confidence,
            "muted": bool(self.muted),
        }
