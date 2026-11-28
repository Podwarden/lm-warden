"""Reasoning budget (#348): bound a model's thinking per model, per key and per request.

Three layers, effective budget = min(every layer that is set). A request can only
lower the budget, never exceed the key's or the model's; unset = no limit there.

What happens when the budget is spent is the *mode* (the key's if set, else the
model's, default ``force_answer``):

* ``force_answer`` -- end the reasoning block and let the model write its reply.
  Both pinned engines can do this per request (vLLM 0.32.0 ``thinking_token_budget``
  with a reasoning parser; llama.cpp b12644 ``reasoning_budget_tokens``), see
  :func:`engine_field`. The proxy still counts, as a backstop for a model whose
  template ignores the field.
* ``stop`` -- end the request at once. No engine does this per request, so the proxy
  does it: it counts reasoning-channel deltas (about one token each) and, past the
  budget, closes the upstream and ends the stream itself (:func:`budget_frames`).

Either way the client is told with ``lmw_stop_reason: "reasoning_budget"`` and the
applied ``lmw_reasoning_budget`` on the final stream chunk or the response body.
"""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, TypeGuard

from app.proxy.alias_overrides import apply_alias_params
from app.proxy.reaggregate import parse_sse_event
from app.utils.sse import delta_reasoning

ReasoningBudgetMode = Literal["force_answer", "stop"]
EffortName = Literal["minimal", "low", "medium", "high"]
REASONING_BUDGET_MODES = ("force_answer", "stop")
DEFAULT_MODE: ReasoningBudgetMode = "force_answer"

STOP_REASON = "reasoning_budget"
#: What ``request_history.finish_reason`` records when the budget was hit.
HISTORY_OUTCOME = "reasoning_budget"

#: ``reasoning_effort`` -> tokens, until a model overrides entries.
DEFAULT_EFFORT_BUDGETS: dict[str, int] = {
    "minimal": 256,
    "low": 1024,
    "medium": 4096,
    "high": 16384,
}
#: Engine-enforced requests get this much slack before the proxy backstop acts.
_GRACE_MIN = 256


def valid_budget(value: Any) -> TypeGuard[int]:
    """A usable budget: a positive int (``bool`` is not one)."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def effort_table(raw: Any) -> dict[str, int]:
    """The effort -> tokens table: the defaults, with the model's valid entries laid over.

    ``raw`` is the model's override: a dict, or the JSON text of one.
    """
    table = dict(DEFAULT_EFFORT_BUDGETS)
    if not raw:
        return table
    over = raw
    if isinstance(raw, str):
        try:
            over = json.loads(raw)
        except ValueError:
            return table
    if not isinstance(over, dict):
        return table
    for effort, tokens in over.items():
        if effort in DEFAULT_EFFORT_BUDGETS and valid_budget(tokens):
            table[effort] = tokens
    return table


def request_budget(body: dict[str, Any], table: dict[str, int]) -> int | None:
    """The request's own budget: ``lmw_reasoning_budget`` (exact), else the effort level.

    Anthropic ``thinking.budget_tokens`` arrives as ``lmw_reasoning_budget``: the
    Messages translator copies it there. The effort is OpenAI's ``reasoning_effort``, or
    ``lmw_reasoning_effort`` where the Responses translator carries ``reasoning.effort``
    (it is not forwarded as ``reasoning_effort``: llama.cpp refuses a level its chat
    template does not know). An invalid value counts as not sent.
    """
    explicit = body.get("lmw_reasoning_budget")
    if valid_budget(explicit):
        return explicit
    for key in ("reasoning_effort", "lmw_reasoning_effort"):
        effort = body.get(key)
        if isinstance(effort, str) and effort in table:
            return table[effort]
    return None


def strip_request_fields(body: dict[str, Any]) -> bool:
    """Remove the warden-only fields before the body goes to an engine."""
    found = False
    for key in ("lmw_reasoning_budget", "lmw_reasoning_effort"):
        if key in body:
            del body[key]
            found = True
    return found


@dataclass(frozen=True)
class Applied:
    budget: int
    mode: str
    #: The layer whose value is the minimum: "model", "key" or "request".
    source: str


def resolve(
    *,
    model_budget: Any = None,
    key_budget: Any = None,
    request_budget: Any = None,
    model_mode: str | None = None,
    key_mode: str | None = None,
) -> Applied | None:
    """min() over the layers that are set; ``None`` when none is."""
    best: tuple[int, str] | None = None
    for source, value in (
        ("model", model_budget),
        ("key", key_budget),
        ("request", request_budget),
    ):
        if valid_budget(value) and (best is None or value <= best[0]):
            best = (value, source)
    if best is None:
        return None
    mode = next((m for m in (key_mode, model_mode) if m in REASONING_BUDGET_MODES), DEFAULT_MODE)
    return Applied(best[0], mode, best[1])


def resolve_for_request(
    model: Any, token: Any, body: dict[str, Any], alias: Any = None
) -> Applied | None:
    """The budget this chat request runs under, or ``None`` when no layer sets one.

    Reads the model's and the key's columns (``getattr``: a row from before the
    migration, or a test double, has none) and the request's own fields. An alias that
    sets ``reasoning_effort`` counts as the request's own, as it is applied to the body
    later on.
    """
    if not isinstance(body.get("messages"), list):
        return None
    probe = {
        k: body[k]
        for k in ("lmw_reasoning_budget", "reasoning_effort", "lmw_reasoning_effort")
        if k in body
    }
    params = getattr(alias, "params", None)
    if isinstance(params, dict) and "reasoning_effort" in params:
        apply_alias_params(probe, {"reasoning_effort": params["reasoning_effort"]})
    table = effort_table(getattr(model, "reasoning_effort_budgets", None))
    return resolve(
        model_budget=getattr(model, "reasoning_budget", None),
        key_budget=getattr(token, "reasoning_budget", None),
        request_budget=request_budget(probe, table),
        model_mode=getattr(model, "reasoning_budget_mode", None),
        key_mode=getattr(token, "reasoning_budget_mode", None),
    )


def _has_reasoning_parser(extra_args: Any) -> bool:
    return isinstance(extra_args, list | tuple) and any(
        isinstance(a, str) and (a == "--reasoning-parser" or a.startswith("--reasoning-parser="))
        for a in extra_args
    )


def engine_field(model: Any, applied: Applied) -> tuple[str, int] | None:
    """The engine's own per-request field for a force_answer budget, or ``None``.

    ``None`` means the proxy has to enforce it (stop mode, MLX, or a vLLM launched
    without ``--reasoning-parser``: vLLM answers 400 to ``thinking_token_budget``
    unless a reasoning config exists, and the parser creates it).
    """
    if applied.mode != "force_answer":
        return None
    backend = getattr(model, "backend", None)
    if backend == "llamacpp":
        return "reasoning_budget_tokens", applied.budget
    if backend == "vllm" and _has_reasoning_parser(getattr(model, "extra_args", None)):
        return "thinking_token_budget", applied.budget
    return None


def _choice0(ev: dict[str, Any]) -> dict[str, Any] | None:
    for c in ev.get("choices") or ():
        if isinstance(c, dict) and c.get("index", 0) == 0:
            return c
    return None


class BudgetGuard:
    """Counts the reasoning channel of one request and says when the budget is spent."""

    def __init__(self, applied: Applied, *, engine_enforced: bool = False) -> None:
        self.applied = applied
        self.engine_enforced = engine_enforced
        self.reasoning = 0
        # An engine that enforces itself gets slack, so the backstop only fires
        # when the engine did not (a template with no thinking tags).
        grace = max(_GRACE_MIN, applied.budget // 4) if engine_enforced else 0
        self.limit = applied.budget + grace

    @property
    def tripped(self) -> bool:
        """The proxy must end the request now."""
        return self.reasoning > self.limit

    @property
    def hit(self) -> bool:
        """The budget decided how this request ended."""
        if self.applied.mode == "stop" and not self.engine_enforced:
            return self.tripped
        return self.reasoning >= self.applied.budget or self.tripped

    def feed_event(self, ev: dict[str, Any]) -> None:
        choice = _choice0(ev)
        if choice is not None:
            delta = choice.get("delta")
            if isinstance(delta, dict) and delta_reasoning(delta):
                self.reasoning += 1
        usage = ev.get("usage")
        details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
        n = details.get("reasoning_tokens") if isinstance(details, dict) else None
        if isinstance(n, int) and not isinstance(n, bool) and n > self.reasoning:
            self.reasoning = n

    def annotation(self) -> dict[str, Any]:
        return {"lmw_stop_reason": STOP_REASON, "lmw_reasoning_budget": self.applied.budget}


def _data_frame(ev: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(ev).encode() + b"\n\n"


def terminal_chunk(guard: BudgetGuard, meta: dict[str, Any]) -> bytes:
    """The stream's last frames after a proxy-enforced stop."""
    ev = {
        "id": meta.get("id") or "chatcmpl-reasoning-budget",
        "object": "chat.completion.chunk",
        "created": meta.get("created") or 0,
        "model": meta.get("model") or "",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
        **guard.annotation(),
    }
    return _data_frame(ev) + b"data: [DONE]\n\n"


async def budget_frames(source: AsyncIterator[bytes], guard: BudgetGuard) -> AsyncIterator[bytes]:
    """Pass an SSE byte stream through the guard.

    Frames are forwarded byte for byte, except the final one (the one carrying a
    ``finish_reason``) when the budget decided the outcome: it gains the ``lmw_*``
    fields. When the guard trips the upstream is closed and a terminal chunk ends the
    stream.
    """
    buf = b""
    meta: dict[str, Any] = {}
    try:
        async for chunk in source:
            buf += chunk
            while b"\n\n" in buf:
                frame, buf = buf.split(b"\n\n", 1)
                ev = parse_sse_event(frame)
                if ev is None:
                    yield frame + b"\n\n"
                    continue
                for k in ("id", "model", "created"):
                    if k not in meta and ev.get(k) is not None:
                        meta[k] = ev[k]
                guard.feed_event(ev)
                if guard.tripped:
                    yield terminal_chunk(guard, meta)
                    return
                choice = _choice0(ev)
                if choice is not None and choice.get("finish_reason") and guard.hit:
                    ev.update(guard.annotation())
                    yield _data_frame(ev)
                else:
                    yield frame + b"\n\n"
        if buf:
            yield buf
    finally:
        aclose = getattr(source, "aclose", None)
        if aclose is not None:
            await aclose()
