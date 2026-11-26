"""Seeded sessions for the Prompt Lens analysis tests (Tasks 1.5-1.7).

``seeded_rows(fault)`` builds the ``request_history`` rows of one fake session
with exactly one kind of fault in it, as the hot path would have stored them:
codes, counts, sizes and keyed hashes only (there is no prompt text in a Row).
Every id is obviously fake. Each fault is parameterised (keyword arguments
override the defaults below) so later tasks can tune a seed without breaking
the ones that already pass.

Baseline ("clean"): one lane, a turn every 30 s, the prompt grows by
``growth`` tokens per turn and every continuing turn reuses all of the
previous prompt. Three tools are offered and all of them are called with
different arguments (so nothing is unused, repeated or looping). Shape chars
are 4 per token (tokens per char 0.25).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from app.lens.schema_lint import LINT_NO_DESC
from app.sessions.store import Row

FAULTS = (
    "clean",
    "timestamp_line",
    "path_line",
    "tools_reordered",
    "unused_tools",
    "reread_file",
    "bad_schema",
    "loop",
    "empty_reply_template",
    "truncating_max_tokens",
    "evicted_pauses",
    "no_marks_anthropic",
)

#: Chars per token in the seeded shapes.
CHARS_PER_TOKEN = 4
T0 = 1_700_000_000.0
SESSION = "sess-fake-0001"
LANE = "conv-fake-0001"
TOKEN = "tok-fake-0001"
TOOLSET = "toolset-fake-01"


def tool_hash(i: int) -> str:
    """An obviously fake 8-hex tool-name hash."""
    return f"fa{i:06x}"


def args_hash(i: int) -> str:
    return f"ab{i:06x}"


#: The defaults of every knob; ``seeded_rows(fault, **params)`` overrides them.
DEFAULTS: dict[str, Any] = {
    "base_prompt": 6_000,
    "growth": 400,
    "step_s": 30.0,
    "start": T0,
    "completion": 200,
    "reply_chars": 600,
    "max_tokens": 4_096,
    "sys_chars": 4_000,
    # offered tools: count, how many of them get called, chars per definition
    "tools": 3,
    "called_tools": 3,
    "tool_chars": 1_200,
    "model": "fake-model",
    # timestamp_line / path_line
    "span_class": None,  # fault default: "time" / "path"
    "span_line": None,  # fault default: 2 / 4
    "every": 1,
    "lost": None,  # tokens lost per faulty turn (default: all of the previous prompt)
    # tools_reordered
    "reorder_at": (6, 13, 19),
    # reread_file
    "read_at": (4, 9, 14),
    "result_tokens": 900,
    # bad_schema
    "bad_calls_at": (7, 9, 12, 14),
    "ok_calls_at": (3, 5),
    "bad_code": "missing_arg",
    "bad_lint": LINT_NO_DESC,
    # loop
    "loop_at": (6, 7, 8, 9),
    # empty_reply_template
    "empty_at": (5, 11, 17),
    "reasoning": 300,
    # truncating_max_tokens
    "cut_at": (8, 15, 20),
    "client_max_tokens": 1_024,
    # evicted_pauses
    "evicted_at": (8, 15, 21),
    "pause_s": 900.0,
    # no_marks_anthropic
    "relay_model": "claude-fake-sonnet",
}

FAULT_DEFAULTS: dict[str, dict[str, Any]] = {
    "timestamp_line": {"span_class": "time", "span_line": 2},
    "path_line": {"span_class": "path", "span_line": 4},
    "unused_tools": {"tools": 12, "called_tools": 5, "tool_chars": 1_600},
    "bad_schema": {"tools": 4},
}


def _params(fault: str, over: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(over) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown seed parameters: {sorted(unknown)}")
    return {**DEFAULTS, **FAULT_DEFAULTS.get(fault, {}), **over}


def _prompt(p: Mapping[str, Any], n: int) -> int:
    return int(p["base_prompt"] + p["growth"] * (n - 1))


def _base_row(p: Mapping[str, Any], n: int) -> Row:
    prompt = _prompt(p, n)
    prev = _prompt(p, n - 1) if n > 1 else 0
    first = n == 1
    k = int(p["tools"])
    tools_chars = k * int(p["tool_chars"])
    started = float(p["start"]) + float(p["step_s"]) * (n - 1)
    return Row(
        id=f"req-fake-{n:04d}",
        token_id=TOKEN,
        token_name="fake key",
        model_id="model-fake",
        model=str(p["model"]),
        started_at=started,
        finished_at=started + 2.0,
        duration_s=2.0,
        queued_s=0.01,
        ttft_s=0.3,
        finish_reason="stop",
        prompt_tokens=prompt,
        cached_tokens=0 if first else prev,
        completion_tokens=int(p["completion"]),
        reusable_tokens=0 if first else prev,
        cache_outcome="cold" if first else "hit",
        cache_outcome_own="cold" if first else "hit",
        divergence_cause=None,
        divergence_cause_own=None,
        dp_rank=0,
        session_key=SESSION,
        parent_session_key=None,
        cobs2_session=SESSION,
        cobs2_conv=LANE,
        cobs2_node=f"node-fake-{n:04d}",
        cobs2_parent_node=None if first else f"node-fake-{n - 1:04d}",
        cobs2_parent_req=None if first else f"req-fake-{n - 1:04d}",
        cobs2_edge="new" if first else "continue",
        cobs2_compaction_request=False,
        shape={
            "chars": prompt * CHARS_PER_TOKEN,
            "sys": int(p["sys_chars"]),
            "tools": k,
            "tools_chars": tools_chars,
            "msgs": {"user": 1, "assistant": n - 1, "tool": n - 1},
            "last_user": 200,
            "dw": {"dup": 0, "boil": 0, "reason": 0},
        },
        toolset=TOOLSET,
        tools_out=(),
        relayed=False,
        upstream=None,
        served_gpu_tokens=0 if first else prev,
        served_ram_tokens=0,
        served_nvme_tokens=0,
        served_remote_tokens=0,
        load_s=None,
        node_id="nd-fake",
        reusable_tokens_own=0 if first else prev,
        lens_break=None,
        lens_cause="new_conversation" if first else None,
        lens_span_class=None,
        lens_span_line=None,
        lens_lost_tokens=None if first else 0,
        lens_cf_tokens=None if first else prev,
        lens_other_breaks=0,
        lens_repeat_results=0,
        max_tokens_req=int(p["max_tokens"]),
        reasoning_tokens=0,
        reply_chars=int(p["reply_chars"]),
        calls_n=0,
        calls_bad="[]",
        tools_lint=json.dumps([0] * k),
    )


def _call(r: Row, tool: int, args: int, *, bad: str | None = None) -> Row:
    """The reply of ``r`` is one call of ``tool`` with ``args``."""
    h = tool_hash(tool)
    return replace(
        r,
        finish_reason="tool_calls",
        reply_chars=0,
        calls_n=1,
        tools_out=(("fn", h, args_hash(args)),),
        calls_bad=json.dumps([[0, h, bad]] if bad else []),
    )


def _with_calls(rows: list[Row], p: Mapping[str, Any]) -> list[Row]:
    """Odd turns call one of the ``called_tools`` with fresh arguments; even
    turns answer in text."""
    out = []
    called = max(1, int(p["called_tools"]))
    for r in rows:
        n = int(r.id.rsplit("-", 1)[1])
        out.append(_call(r, ((n - 1) // 2) % called, 1_000 + n) if n % 2 else r)
    return out


def _miss(r: Row, n: int, p: Mapping[str, Any], *, cause: str, brk: str | None, **kw: Any) -> Row:
    """A continuing turn that lost ``p['lost']`` (default: all) of the previous prompt."""
    prev = _prompt(p, n - 1)
    lost = prev if p["lost"] is None else min(int(p["lost"]), prev)
    kept = prev - lost
    return replace(
        r,
        cached_tokens=kept,
        served_gpu_tokens=kept,
        cache_outcome="diverged",
        cache_outcome_own="diverged",
        lens_cause=cause,
        lens_break=brk,
        lens_cf_tokens=prev,
        lens_lost_tokens=lost,
        **kw,
    )


# -- faults -------------------------------------------------------------------


def _sys_line(rows: list[Row], p: Mapping[str, Any]) -> list[Row]:
    every = max(1, int(p["every"]))
    out = []
    for r in rows:
        n = _n(r)
        if n > 1 and (n - 1) % every == 0:
            r = _miss(
                r,
                n,
                p,
                cause="sys_changed",
                brk="sys",
                lens_span_class=p["span_class"],
                lens_span_line=p["span_line"],
            )
        out.append(r)
    return out


def _tools_reordered(rows: list[Row], p: Mapping[str, Any]) -> list[Row]:
    return [
        _miss(r, _n(r), p, cause="tools_reordered", brk="tools")
        if _n(r) in p["reorder_at"] and _n(r) > 1
        else r
        for r in rows
    ]


def _results_in_prompt(n: int, calls_at: tuple[int, ...]) -> int:
    """Copies of one call's result in turn ``n``'s prompt (each reply's result
    is in the next request)."""
    return sum(1 for c in calls_at if c < n)


def _repeated(rows: list[Row], p: Mapping[str, Any], at: tuple[int, ...], tool: int) -> list[Row]:
    """The same call (same tool, same args) on the turns ``at``, with the same
    result each time: ``lens_repeat_results`` counts the extra copies."""
    rt = int(p["result_tokens"])
    out = []
    for r in rows:
        n = _n(r)
        if n in at:
            r = _call(r, tool, 1)
        copies = _results_in_prompt(n, at)
        if copies > 1:
            r = replace(r, lens_repeat_results=(copies - 1) * rt)
        out.append(r)
    return out


def _reread_file(rows: list[Row], p: Mapping[str, Any]) -> list[Row]:
    # Other tools' calls (with other results) sit between the reads: not a loop.
    return _repeated(rows, p, tuple(p["read_at"]), 0)


def _loop(rows: list[Row], p: Mapping[str, Any]) -> list[Row]:
    return _repeated(rows, p, tuple(p["loop_at"]), 1)


def _bad_schema(rows: list[Row], p: Mapping[str, Any]) -> list[Row]:
    bad_tool = int(p["tools"]) - 1
    lint = [0] * int(p["tools"])
    lint[bad_tool] = int(p["bad_lint"])
    out = []
    for r in rows:
        n = _n(r)
        if n in p["bad_calls_at"]:
            r = _call(r, bad_tool, 2_000 + n, bad=str(p["bad_code"]))
        elif n in p["ok_calls_at"]:
            r = _call(r, bad_tool, 2_000 + n)
        out.append(replace(r, tools_lint=json.dumps(lint)))
    return out


def _empty_reply(rows: list[Row], p: Mapping[str, Any]) -> list[Row]:
    return [
        replace(
            r,
            finish_reason="stop",
            completion_tokens=int(p["reasoning"]),
            reasoning_tokens=int(p["reasoning"]),
            reply_chars=0,
            calls_n=0,
            tools_out=(),
            calls_bad="[]",
        )
        if _n(r) in p["empty_at"]
        else r
        for r in rows
    ]


def _truncating(rows: list[Row], p: Mapping[str, Any]) -> list[Row]:
    m = int(p["client_max_tokens"])
    return [
        replace(
            r,
            max_tokens_req=m,
            **(
                {
                    "finish_reason": "length",
                    "completion_tokens": m,
                    "reply_chars": m * CHARS_PER_TOKEN,
                    "calls_n": 0,
                    "tools_out": (),
                    "calls_bad": "[]",
                }
                if _n(r) in p["cut_at"]
                else {}
            ),
        )
        for r in rows
    ]


def _evicted(rows: list[Row], p: Mapping[str, Any]) -> list[Row]:
    shift = 0.0
    out = []
    for r in rows:
        n = _n(r)
        if n in p["evicted_at"] and n > 1:
            shift += float(p["pause_s"])
            prev = _prompt(p, n - 1)
            r = replace(
                r,
                cached_tokens=0,
                served_gpu_tokens=0,
                cache_outcome="lost",
                cache_outcome_own="lost",
                lens_cause="server_evicted",
                lens_break=None,
                lens_cf_tokens=prev,
                lens_lost_tokens=prev,
            )
        out.append(replace(r, started_at=r.started_at + shift, finished_at=r.finished_at + shift))
    return out


def _no_marks(rows: list[Row], p: Mapping[str, Any]) -> list[Row]:
    model = str(p["relay_model"])
    pmin = 2_048 if "haiku" in model.lower() else 1_024
    out = []
    for r in rows:
        n = _n(r)
        relay: dict[str, Any] = {
            "model": model,
            "relayed": True,
            "upstream": "upstream.example",
            "cache_marks": 0,
            "cache_write_tokens": 0,
            "cached_tokens": 0,
            "served_gpu_tokens": None,
            "served_ram_tokens": None,
            "served_nvme_tokens": None,
            "served_remote_tokens": None,
            "dp_rank": None,
            "cache_outcome": "cold",
            "cache_outcome_own": "cold",
            # Anthropic relays carry 2-element tools_out and no verdicts in S1.
            "tools_out": tuple(e[:2] for e in r.tools_out),
            "calls_bad": None,
            "calls_n": None,
            "reply_chars": None,
            "tools_lint": None,
        }
        if n > 1:
            prev = _prompt(p, n - 1)
            relay.update(
                lens_cause="provider_below_min" if prev < pmin else "provider_no_marks",
                lens_break=None,
                lens_cf_tokens=prev,
                lens_lost_tokens=prev,
            )
        out.append(replace(r, **relay))
    return out


def _n(r: Row) -> int:
    return int(r.id.rsplit("-", 1)[1])


_FAULTS: dict[str, Callable[[list[Row], Mapping[str, Any]], list[Row]]] = {
    "clean": lambda rows, p: rows,
    "timestamp_line": _sys_line,
    "path_line": _sys_line,
    "tools_reordered": _tools_reordered,
    "unused_tools": lambda rows, p: rows,
    "reread_file": _reread_file,
    "bad_schema": _bad_schema,
    "loop": _loop,
    "empty_reply_template": _empty_reply,
    "truncating_max_tokens": _truncating,
    "evicted_pauses": _evicted,
    "no_marks_anthropic": _no_marks,
}


def seeded_rows(fault: str, *, turns: int = 24, **params: Any) -> list[Row]:
    """The rows of one fake session with ``fault`` (see ``FAULTS``)."""
    if fault not in _FAULTS:
        raise KeyError(fault)
    p = _params(fault, params)
    rows = _with_calls([_base_row(p, n) for n in range(1, turns + 1)], p)
    return _FAULTS[fault](rows, p)


def seeded_toolsets(fault: str, **params: Any) -> dict[str, list[tuple[str, str, int]]]:
    """``store.toolsets``-shaped entries for the seed: toolset id -> (hash8,
    family, chars) per offered tool, in offer order."""
    p = _params(fault, params)
    k = int(p["tools"])
    return {TOOLSET: [(tool_hash(i), "fn", int(p["tool_chars"])) for i in range(k)]}


def seeded_lint(fault: str, **params: Any) -> dict[str, int]:
    """Tool hash -> lint flags, as Task 1.7's ``response_findings(lint=...)`` takes it."""
    p = _params(fault, params)
    rows = seeded_rows(fault, turns=1, **params)
    flags = json.loads(rows[0].tools_lint or "[]")
    return {tool_hash(i): int(f) for i, f in enumerate(flags[: int(p["tools"])])}
