"""Reasoning budget (#348): the pure parts -- precedence, request fields, engine mapping, guard."""

import json
from types import SimpleNamespace
from typing import Any

import pytest

from app.proxy.reasoning_budget import (
    DEFAULT_EFFORT_BUDGETS,
    Applied,
    BudgetGuard,
    budget_frames,
    effort_table,
    engine_field,
    request_budget,
    resolve,
    strip_request_fields,
)

# -- precedence ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "key", "req", "want", "source"),
    [
        (None, None, None, None, None),
        (4000, None, None, 4000, "model"),
        (None, 3000, None, 3000, "key"),
        (None, None, 2000, 2000, "request"),
        (4000, 3000, 2000, 2000, "request"),
        (4000, 3000, 9000, 3000, "key"),  # a request can only lower the budget
        (4000, 5000, None, 4000, "model"),
        (4000, None, 9999, 4000, "model"),
        (None, 3000, 9999, 3000, "key"),
        (1000, 1000, 1000, 1000, "request"),
    ],
)
def test_effective_budget_is_the_minimum_of_the_set_layers(model, key, req, want, source):
    got = resolve(model_budget=model, key_budget=key, request_budget=req)
    if want is None:
        assert got is None
    else:
        assert got is not None
        assert (got.budget, got.source) == (want, source)


def test_mode_key_overrides_model_and_defaults_to_force_answer():
    assert resolve(model_budget=10).mode == "force_answer"  # type: ignore[union-attr]
    assert resolve(model_budget=10, model_mode="stop").mode == "stop"  # type: ignore[union-attr]
    got = resolve(model_budget=10, model_mode="stop", key_mode="force_answer")
    assert got is not None and got.mode == "force_answer"
    got = resolve(model_budget=10, model_mode="force_answer", key_mode="stop")
    assert got is not None and got.mode == "stop"
    got = resolve(model_budget=10, model_mode="junk", key_mode="junk")
    assert got is not None and got.mode == "force_answer"


@pytest.mark.parametrize("bad", [0, -5, True, "7", 1.5])
def test_a_non_positive_or_non_integer_layer_counts_as_unset(bad):
    assert resolve(model_budget=bad) is None  # type: ignore[arg-type]


# -- request fields -----------------------------------------------------------


def test_explicit_field_is_taken_as_is():
    assert request_budget({"lmw_reasoning_budget": 1234}, effort_table(None)) == 1234


def test_effort_maps_through_the_default_table():
    table = effort_table(None)
    assert table == DEFAULT_EFFORT_BUDGETS
    for effort, want in DEFAULT_EFFORT_BUDGETS.items():
        assert request_budget({"reasoning_effort": effort}, table) == want


def test_per_model_table_overrides_defaults_and_keeps_the_rest():
    table = effort_table(json.dumps({"low": 100, "high": 9000}))
    assert request_budget({"reasoning_effort": "low"}, table) == 100
    assert request_budget({"reasoning_effort": "high"}, table) == 9000
    assert request_budget({"reasoning_effort": "medium"}, table) == DEFAULT_EFFORT_BUDGETS["medium"]


def test_broken_table_falls_back_to_defaults():
    assert effort_table("not json") == DEFAULT_EFFORT_BUDGETS
    assert effort_table(json.dumps({"low": -1, "medium": "x"})) == DEFAULT_EFFORT_BUDGETS
    assert effort_table(json.dumps([1, 2])) == DEFAULT_EFFORT_BUDGETS


def test_explicit_wins_over_effort():
    body = {"lmw_reasoning_budget": 50, "reasoning_effort": "high"}
    assert request_budget(body, effort_table(None)) == 50


@pytest.mark.parametrize("effort", ["none", "xhigh", "max", "", 3, None])
def test_unmapped_effort_sets_no_budget(effort):
    assert request_budget({"reasoning_effort": effort}, effort_table(None)) is None


@pytest.mark.parametrize("bad", [0, -1, True, "5", 2.5, None])
def test_invalid_explicit_value_is_ignored(bad):
    assert request_budget({"lmw_reasoning_budget": bad}, effort_table(None)) is None


def test_strip_removes_the_warden_field_only():
    body: dict[str, Any] = {"lmw_reasoning_budget": 5, "reasoning_effort": "low", "x": 1}
    assert strip_request_fields(body) is True
    assert body == {"reasoning_effort": "low", "x": 1}
    assert strip_request_fields(body) is False


# -- engine mapping -----------------------------------------------------------


def _model(backend: str, extra: list[str] | None = None) -> Any:
    return SimpleNamespace(backend=backend, extra_args=extra or [])


def test_llamacpp_takes_the_per_request_field():
    got = engine_field(_model("llamacpp"), Applied(300, "force_answer", "key"))
    assert got == ("reasoning_budget_tokens", 300)


def test_vllm_needs_a_reasoning_parser():
    applied = Applied(300, "force_answer", "key")
    assert engine_field(_model("vllm"), applied) is None
    assert engine_field(_model("vllm", ["--reasoning-parser", "qwen3"]), applied) == (
        "thinking_token_budget",
        300,
    )
    assert engine_field(_model("vllm", ["--reasoning-parser=qwen3"]), applied) == (
        "thinking_token_budget",
        300,
    )


def test_stop_mode_and_unsupported_backends_use_no_engine_field():
    assert engine_field(_model("llamacpp"), Applied(300, "stop", "key")) is None
    assert engine_field(_model("mlx"), Applied(300, "force_answer", "key")) is None


# -- guard --------------------------------------------------------------------


def _r(text: str) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": {"reasoning_content": text}}]}


def _c(text: str) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": {"content": text}}]}


def test_guard_counts_reasoning_deltas_of_choice_zero_only():
    g = BudgetGuard(Applied(3, "stop", "key"))
    g.feed_event(_c("x"))
    g.feed_event({"choices": [{"index": 1, "delta": {"reasoning_content": "y"}}]})
    assert g.reasoning == 0
    for _ in range(3):
        g.feed_event(_r("t"))
    assert g.reasoning == 3 and not g.tripped and not g.hit
    g.feed_event(_r("t"))
    assert g.tripped and g.hit


def test_guard_reads_the_reasoning_field_alias_and_usage_tokens():
    g = BudgetGuard(Applied(100, "force_answer", "key"))
    g.feed_event({"choices": [{"index": 0, "delta": {"reasoning": "a"}}]})
    assert g.reasoning == 1
    g.feed_event({"choices": [], "usage": {"completion_tokens_details": {"reasoning_tokens": 100}}})
    assert g.reasoning == 100 and g.hit and not g.tripped


def test_engine_enforced_guard_only_trips_past_the_grace():
    g = BudgetGuard(Applied(1000, "force_answer", "key"), engine_enforced=True)
    assert g.limit == 1000 + 256
    stop = BudgetGuard(Applied(1000, "stop", "key"))
    assert stop.limit == 1000


def test_guard_not_hit_below_budget():
    g = BudgetGuard(Applied(10, "stop", "key"))
    g.feed_event(_r("a"))
    assert not g.hit and not g.tripped


# -- frame wrapper ------------------------------------------------------------


def _frame(ev: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(ev).encode() + b"\n\n"


async def _src(frames: list[bytes]):
    for f in frames:
        yield f


async def _collect(it) -> list[bytes]:
    return [c async for c in it]


def _events(blob: bytes) -> list[Any]:
    out: list[Any] = []
    for part in blob.split(b"\n\n"):
        part = part.strip()
        if not part:
            continue
        payload = part[len(b"data:") :].strip()
        out.append("[DONE]" if payload == b"[DONE]" else json.loads(payload))
    return out


def _finish(reason: str) -> dict[str, Any]:
    return {
        "id": "c1",
        "model": "m",
        "created": 1,
        "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
    }


async def test_stop_mode_cuts_the_stream_with_a_terminal_chunk():
    g = BudgetGuard(Applied(2, "stop", "key"))
    frames = [_frame({"id": "c1", "model": "m", "created": 1, **_r("a")})] + [
        _frame(_r("b")),
        _frame(_r("c")),
        _frame(_c("never")),
        b"data: [DONE]\n\n",
    ]
    closed: list[bool] = []

    async def src():
        try:
            for f in frames:
                yield f
        finally:
            closed.append(True)

    got = await _collect(budget_frames(src(), g))
    evs = _events(b"".join(got))
    assert evs[-1] == "[DONE]"
    last = evs[-2]
    assert last["choices"][0]["finish_reason"] == "length"
    assert last["lmw_stop_reason"] == "reasoning_budget"
    assert last["lmw_reasoning_budget"] == 2
    assert not any(
        e != "[DONE]" and e["choices"] and e["choices"][0]["delta"].get("content") == "never"
        for e in evs
    )
    assert g.tripped and closed == [True]


async def test_force_answer_annotates_the_final_chunk_and_keeps_the_engine_reason():
    g = BudgetGuard(Applied(2, "force_answer", "model"), engine_enforced=True)
    frames = [
        _frame(_r("a")),
        _frame(_r("b")),
        _frame(_c("answer")),
        _frame(_finish("stop")),
        b"data: [DONE]\n\n",
    ]
    evs = _events(b"".join(await _collect(budget_frames(_src(frames), g))))
    fin = [
        e for e in evs if e != "[DONE]" and e["choices"] and e["choices"][0].get("finish_reason")
    ][0]
    assert fin["choices"][0]["finish_reason"] == "stop"
    assert fin["lmw_stop_reason"] == "reasoning_budget"
    assert fin["lmw_reasoning_budget"] == 2
    assert not g.tripped and g.hit


async def test_no_hit_leaves_every_frame_byte_identical():
    g = BudgetGuard(Applied(50, "force_answer", "model"), engine_enforced=True)
    frames = [_frame(_r("a")), _frame(_c("hi")), _frame(_finish("stop")), b"data: [DONE]\n\n"]
    got = await _collect(budget_frames(_src(frames), g))
    assert b"".join(got) == b"".join(frames)


async def test_frames_split_across_chunks_are_reassembled():
    g = BudgetGuard(Applied(1, "stop", "key"))
    blob = _frame(_r("a")) + _frame(_r("b")) + _frame(_r("c"))
    pieces = [blob[:10], blob[10:37], blob[37:]]
    got = await _collect(budget_frames(_src(pieces), g))
    assert g.tripped
    assert _events(b"".join(got))[-1] == "[DONE]"


# -- resolve_for_request ------------------------------------------------------


def test_resolve_for_request_reads_the_rows_and_the_body():
    from app.proxy.reasoning_budget import resolve_for_request

    model = SimpleNamespace(
        reasoning_budget=5000,
        reasoning_budget_mode="stop",
        reasoning_effort_budgets={"low": 300},
    )
    token = SimpleNamespace(reasoning_budget=4000, reasoning_budget_mode=None)
    body = {"messages": [], "reasoning_effort": "low"}
    got = resolve_for_request(model, token, body)
    assert got == Applied(300, "stop", "request")
    assert resolve_for_request(model, token, {"messages": []}) == Applied(4000, "stop", "key")
    # not a chat body: no budget
    assert resolve_for_request(model, token, {"prompt": "x"}) is None


def test_resolve_for_request_counts_an_alias_effort_as_the_requests_own():
    from app.proxy.reasoning_budget import resolve_for_request

    model = SimpleNamespace(reasoning_budget=None, reasoning_effort_budgets=None)
    alias = SimpleNamespace(params={"reasoning_effort": {"value": "low", "mode": "force"}})
    got = resolve_for_request(
        model, SimpleNamespace(), {"messages": [], "reasoning_effort": "high"}, alias
    )
    assert got is not None and got.budget == DEFAULT_EFFORT_BUDGETS["low"]
    plain = resolve_for_request(model, SimpleNamespace(), {"messages": []})
    assert plain is None


def test_rows_from_before_the_migration_have_no_budget():
    from app.proxy.reasoning_budget import resolve_for_request

    assert resolve_for_request(SimpleNamespace(), SimpleNamespace(), {"messages": []}) is None
