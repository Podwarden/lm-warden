"""Prompt Lens layer 1: why a request did not reuse its prefix (spec v1 4.2, 5.1-5.2)."""

import dataclasses
import hashlib
import os

import pytest

from app.cobs2.canonical import canonicalize
from app.cobs2.ids import ids_for
from app.cobs2.keys import NetworkKey
from app.cobs2.lineage import Lineage, Observation, short_msg
from app.lens import causes
from app.lens.causes import APP_FIXABLE, CAUSES, CauseInputs, LensVerdict, evaluate
from app.lens.probes import message_probes

KEY = NetworkKey(epoch=1, key=os.urandom(32))
SALT = b"s"
#: The unattributed verdict: every lens field null, other_breaks too (M2).
NONE = LensVerdict(None, None, None, None, None, None, None)

SYS_TOKENS = 1_000
TOOLS_TOKENS = 2_000
HEAD_TOKENS = SYS_TOKENS + TOOLS_TOKENS
MSG_TOKENS = (100, 200, 300, 400, 500)
PARENT_LEN = 3
#: head + the three messages the parent had: what a stable continuing turn reuses
REUSABLE = HEAD_TOKENS + sum(MSG_TOKENS[:PARENT_LEN])


def _b8(label: str) -> bytes:
    return hashlib.blake2b(label.encode(), digest_size=8).digest()


def D(text: str) -> tuple[bytes, ...]:  # noqa: N802 - the brief's name
    """Fake line digests: one per word."""
    return tuple(_b8("line:" + w) for w in text.split())


def P(tag: str) -> tuple[bytes, bytes, bytes, bytes]:  # noqa: N802
    """Four probes that match nothing else."""
    return tuple(_b8(f"probe:{tag}:{i}") for i in range(4))  # type: ignore[return-value]


PM = tuple(_b8(f"msg{i}") for i in range(PARENT_LEN))
NOW = (*PM, _b8("msg3"), _b8("msg4"))
TH = (_b8("tool-a"), _b8("tool-b"), _b8("tool-c"))


def ci(
    *,
    stable: bool = True,
    loose_differs: bool = False,
    edge: str = "continue",
    obs: Observation | None | str = "auto",
    parent_msgs: tuple[bytes, ...] = PM,
    parent_probes=None,
    **kw,
) -> CauseInputs:
    """A continuing turn that extends a 3-message parent by 2 messages, prompt stable,
    nothing reused yet (``actual_tokens`` 0). Keywords override any CauseInputs field."""
    assert stable or kw
    if obs == "auto":
        obs = Observation(
            edge,  # type: ignore[arg-type]
            "parent-node",
            False,
            "s",
            parent_msgs=parent_msgs,
            parent_probes=parent_probes,
        )
    base = dict(
        edge=edge,
        obs=obs,
        sys_now="sys-1",
        sys_parent="sys-1",
        loose_sys_now="lsys-1",
        loose_sys_parent="lsys-1",
        sys_digests_now=D("a b c"),
        sys_digests_parent=D("a b c"),
        tools_now="tools-1",
        tools_parent="tools-1",
        loose_tools_now="ltools-1",
        loose_tools_parent="ltools-1",
        tool_hashes_now=TH,
        tool_hashes_parent=TH,
        msgs_now=NOW,
        probes_now=None,
        msg_tokens=MSG_TOKENS,
        head_tokens=HEAD_TOKENS,
        tools_tokens=TOOLS_TOKENS,
        own_outcome="hit",
        gap_s=None,
        relayed_to=None,
        cache_marks=None,
        provider_min=None,
        sys_text_line=None,
        actual_tokens=0,
    )
    if loose_differs:
        base.update(sys_now="sys-2", loose_sys_now="lsys-2")
    base.update(kw)
    return CauseInputs(**base)  # type: ignore[arg-type]


def ci_sys_changed(**kw) -> CauseInputs:
    args = dict(
        sys_digests_parent=D("a b c"),
        sys_digests_now=D("a X c"),
        loose_differs=True,
        sys_text_line=lambda n: "Working directory: /w/2",
    )
    return ci(**{**args, **kw})


def _body_break(i: int, parent_msg: dict, now_msg: dict, **kw) -> CauseInputs:
    """Message ``i`` of the parent came back as ``now_msg`` (real probes)."""
    now_msgs = list(NOW)
    now_msgs[i] = _b8("changed")
    return ci(
        msgs_now=tuple(now_msgs),
        parent_probes=message_probes(parent_msg, KEY, SALT),
        probes_now=message_probes(now_msg, KEY, SALT),
        obs=Observation(
            "continue",
            "parent-node",
            True,
            "s",
            parent_msgs=PM,
            parent_probes=message_probes(parent_msg, KEY, SALT),
        ),
        **kw,
    )


# -- the code list ------------------------------------------------------------


def test_causes_are_the_18_codes_in_evaluation_order() -> None:
    assert CAUSES == (
        "compacted",
        "fork",
        "sys_noise",
        "sys_changed",
        "tools_reordered",
        "tools_key_order",
        "tools_added_end",
        "tools_changed",
        "history_rewritten",
        "history_noise",
        "reasoning_dropped",
        "reasoning_kept",
        "media_changed",
        "new_conversation",
        "server_evicted",
        "misrouted",
        "provider_no_marks",
        "provider_below_min",
    )


def test_app_fixable_is_the_spec_10_2_list() -> None:
    assert APP_FIXABLE == frozenset(
        {
            "sys_noise",
            "sys_changed",
            "tools_reordered",
            "tools_key_order",
            "tools_added_end",
            "tools_changed",
            "history_rewritten",
            "history_noise",
            "reasoning_dropped",
            "reasoning_kept",
            "media_changed",
            "provider_no_marks",
            "compacted",
        }
    )
    assert APP_FIXABLE <= set(CAUSES)
    for server_side in ("server_evicted", "misrouted", "provider_below_min", "new_conversation"):
        assert server_side not in APP_FIXABLE


# -- one test per cause -------------------------------------------------------


def test_compacted() -> None:
    v = evaluate(
        ci(edge="compacted_from", msgs_now=(_b8("summary"),), msg_tokens=(800,)),
        render_order="sys_first",
    )
    assert v.cause == "compacted" and v.brk == "node:0"
    # the summary replaces the history: at best the head is reused
    assert (v.counterfactual_tokens, v.lost_tokens, v.other_breaks) == (HEAD_TOKENS, HEAD_TOKENS, 0)


def test_compacted_with_a_changed_head_breaks_at_the_head() -> None:
    v = evaluate(
        ci(edge="compacted_from", loose_differs=True, msgs_now=(_b8("summary"),)),
        render_order="sys_first",
    )
    assert (v.cause, v.brk) == ("compacted", "sys")


def test_fork() -> None:
    v = evaluate(
        ci(edge="fork", parent_msgs=PM[:2], msgs_now=(*PM[:2], _b8("other"))),
        render_order="sys_first",
    )
    assert (v.cause, v.brk) == ("fork", "node:2")
    assert v.counterfactual_tokens == HEAD_TOKENS + 100 + 200
    assert v.lost_tokens == HEAD_TOKENS + 300


def test_sys_noise() -> None:
    v = evaluate(
        ci(
            sys_now="sys-2",
            sys_digests_now=D("a b2 c"),
            sys_text_line=lambda n: "Today's date: 2026-11-22",
        ),
        render_order="sys_first",
    )
    assert (v.brk, v.cause, v.span_line, v.span_class) == ("sys", "sys_noise", 2, "date")
    assert (v.counterfactual_tokens, v.lost_tokens, v.other_breaks) == (REUSABLE, REUSABLE, 0)


def test_sys_changed_locates_line_and_class() -> None:
    v = evaluate(
        ci(
            sys_digests_parent=D("a b c"),
            sys_digests_now=D("a X c"),
            loose_differs=True,
            sys_text_line=lambda n: "Working directory: /w/2",
        ),
        render_order="sys_first",
    )
    assert (v.brk, v.cause, v.span_line, v.span_class) == ("sys", "sys_changed", 2, "path")
    assert v.counterfactual_tokens == REUSABLE


def test_sys_text_line_gets_the_located_line_number() -> None:
    seen: list[int] = []
    evaluate(
        ci_sys_changed(sys_text_line=lambda n: seen.append(n) or "x"), render_order="sys_first"
    )
    assert seen == [2]


def test_tools_reordered() -> None:
    v = evaluate(
        ci(tools_now="tools-2", loose_tools_now="ltools-2", tool_hashes_now=TH[::-1]),
        render_order="sys_first",
    )
    assert (v.brk, v.cause, v.span_line, v.span_class) == ("tools", "tools_reordered", None, None)
    assert v.counterfactual_tokens == REUSABLE


def test_tools_key_order() -> None:
    v = evaluate(
        ci(tools_now="tools-2", tool_hashes_now=(TH[0], _b8("tool-b-keys"), TH[2])),
        render_order="sys_first",
    )
    assert (v.brk, v.cause) == ("tools", "tools_key_order")
    assert v.lost_tokens == REUSABLE


def test_tools_added_end() -> None:
    v = evaluate(
        ci(
            tools_now="tools-2",
            loose_tools_now="ltools-2",
            tool_hashes_now=(*TH, _b8("tool-d")),
        ),
        render_order="sys_first",
    )
    assert (v.brk, v.cause) == ("tools", "tools_added_end")


def test_tools_changed() -> None:
    v = evaluate(
        ci(
            tools_now="tools-2",
            loose_tools_now="ltools-2",
            tool_hashes_now=(TH[0], _b8("tool-x"), TH[2]),
        ),
        render_order="sys_first",
    )
    assert (v.brk, v.cause) == ("tools", "tools_changed")
    assert v.counterfactual_tokens == REUSABLE


def test_tools_removed_is_tools_changed() -> None:
    v = evaluate(
        ci(tools_now="tools-2", loose_tools_now="ltools-2", tool_hashes_now=TH[:2]),
        render_order="sys_first",
    )
    assert v.cause == "tools_changed"


def test_history_rewritten() -> None:
    v = _body_break(
        1,
        {"role": "assistant", "content": "the original answer"},
        {"role": "assistant", "content": "a different answer"},
    )
    out = evaluate(v, render_order="sys_first")
    assert (out.brk, out.cause, out.span_line, out.span_class) == (
        "node:1",
        "history_rewritten",
        None,
        None,
    )
    # had message 1 been kept, messages 0-2 would all have matched
    assert out.counterfactual_tokens == REUSABLE
    assert out.lost_tokens == REUSABLE


def test_history_noise() -> None:
    v = evaluate(
        _body_break(
            0,
            {"role": "user", "content": "x <system-reminder>one</system-reminder> y"},
            {"role": "user", "content": "x <system-reminder>two</system-reminder> y"},
        ),
        render_order="sys_first",
    )
    assert (v.brk, v.cause) == ("node:0", "history_noise")


def test_whitespace_only_change_is_history_noise() -> None:
    v = evaluate(
        _body_break(2, {"role": "user", "content": "a\r\nb"}, {"role": "user", "content": "a\nb"}),
        render_order="sys_first",
    )
    assert (v.brk, v.cause) == ("node:2", "history_noise")


def test_reasoning_dropped() -> None:
    v = evaluate(
        _body_break(
            1,
            {"role": "assistant", "content": "ok", "reasoning_content": "thinking hard"},
            {"role": "assistant", "content": "ok"},
            reasoning_now=False,
        ),
        render_order="sys_first",
    )
    # the noise probe also equalises a reasoning drop: the specific probe wins
    assert (v.brk, v.cause) == ("node:1", "reasoning_dropped")
    assert v.counterfactual_tokens == REUSABLE


def test_reasoning_kept() -> None:
    v = evaluate(
        _body_break(
            1,
            {"role": "assistant", "content": "ok"},
            {"role": "assistant", "content": "ok", "reasoning_content": "thinking hard"},
            reasoning_now=True,
        ),
        render_order="sys_first",
    )
    assert (v.brk, v.cause) == ("node:1", "reasoning_kept")


def test_media_changed() -> None:
    t = {"type": "text", "text": "look"}
    img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
    v = evaluate(
        _body_break(
            0, {"role": "user", "content": [img, t]}, {"role": "user", "content": [t, img]}
        ),
        render_order="sys_first",
    )
    assert (v.brk, v.cause) == ("node:0", "media_changed")


def test_new_conversation() -> None:
    v = evaluate(
        ci(
            edge="new",
            obs=Observation("new", None),
            sys_parent=None,
            loose_sys_parent=None,
            sys_digests_parent=None,
            tools_parent=None,
            loose_tools_parent=None,
            tool_hashes_parent=None,
            msgs_now=(_b8("first-user"),),
            msg_tokens=(50,),
            first_turn=True,
        ),
        render_order="sys_first",
    )
    assert v == LensVerdict(None, "new_conversation", None, None, None, None, 0)


def test_new_conversation_with_the_same_recent_head() -> None:
    v = evaluate(
        ci(edge="new", obs=Observation("new", None), msgs_now=(_b8("u"),), first_turn=True),
        render_order="sys_first",
    )
    # the head could have been reused from the key's earlier session
    assert v == LensVerdict(None, "new_conversation", None, None, HEAD_TOKENS, HEAD_TOKENS, 0)


def test_new_session_whose_recent_head_has_another_exact_sys_is_sys_noise() -> None:
    v = evaluate(
        ci_sys_changed(edge="new", obs=Observation("new", None), loose_sys_now="lsys-1"),
        render_order="sys_first",
    )
    assert (v.brk, v.cause, v.span_line, v.span_class) == ("sys", "sys_noise", 2, "path")
    assert v.counterfactual_tokens == HEAD_TOKENS


def test_server_evicted_when_prompt_stable() -> None:
    v = evaluate(ci(stable=True, own_outcome="lost", gap_s=840), render_order="sys_first")
    assert v.cause == "server_evicted" and v.lost_tokens == REUSABLE
    assert (v.brk, v.span_class, v.span_line, v.counterfactual_tokens) == (
        None,
        None,
        None,
        REUSABLE,
    )


def test_misrouted() -> None:
    v = evaluate(ci(own_outcome="misrouted", actual_tokens=1_000), render_order="sys_first")
    assert (v.cause, v.brk) == ("misrouted", None)
    assert v.lost_tokens == REUSABLE - 1_000


def test_provider_no_marks() -> None:
    v = evaluate(
        ci(relayed_to="anthropic", cache_marks=0, own_outcome="lost", provider_min=1_024),
        render_order="tools_first",
    )
    assert (v.cause, v.brk) == ("provider_no_marks", None)
    assert (v.counterfactual_tokens, v.lost_tokens) == (REUSABLE, REUSABLE)


def test_provider_below_min() -> None:
    v = evaluate(
        ci(
            relayed_to="anthropic",
            cache_marks=2,
            own_outcome="lost",
            provider_min=4_096,
        ),
        render_order="tools_first",
    )
    assert REUSABLE < 4_096
    assert (v.cause, v.brk, v.counterfactual_tokens) == ("provider_below_min", None, REUSABLE)


def test_below_min_wins_over_no_marks() -> None:
    v = evaluate(
        ci(relayed_to="anthropic", cache_marks=0, own_outcome="lost", provider_min=4_096),
        render_order="tools_first",
    )
    assert v.cause == "provider_below_min"


def test_relayed_stable_lost_with_marks_above_min_is_server_evicted() -> None:
    v = evaluate(
        ci(relayed_to="anthropic", cache_marks=2, own_outcome="lost", provider_min=1_024),
        render_order="tools_first",
    )
    assert v.cause == "server_evicted"


def test_every_code_has_a_test() -> None:
    names = {n for n in globals() if n.startswith("test_")}
    for code in CAUSES:
        assert f"test_{code}" in names or any(n.startswith(f"test_{code}_") for n in names), code


# -- render order, counterfactual, cost ----------------------------------------


def test_tools_first_route_keeps_tools_when_only_sys_changed() -> None:
    v = evaluate(ci_sys_changed(), render_order="tools_first")
    assert v.counterfactual_tokens >= TOOLS_TOKENS
    assert (v.brk, v.cause) == ("sys", "sys_changed")


def test_render_order_picks_the_primary_head_break() -> None:
    both = dict(tools_now="tools-2", loose_tools_now="ltools-2", tool_hashes_now=TH[::-1])
    s = evaluate(ci_sys_changed(**both), render_order="sys_first")
    t = evaluate(ci_sys_changed(**both), render_order="tools_first")
    assert (s.brk, s.cause, s.other_breaks) == ("sys", "sys_changed", 1)
    assert (t.brk, t.cause, t.other_breaks) == ("tools", "tools_reordered", 1)
    # with the primary part treated as equal, reuse stops at the other head part
    assert s.counterfactual_tokens == SYS_TOKENS
    assert t.counterfactual_tokens == TOOLS_TOKENS
    assert t.span_line is None and t.span_class is None


def test_counterfactual_stops_at_the_next_differing_message() -> None:
    now = (PM[0], _b8("changed-1"), _b8("changed-2"), _b8("msg3"))
    v = evaluate(
        ci(
            msgs_now=now,
            probes_now=P("now"),
            obs=Observation("continue", "p", True, "s", parent_msgs=PM, parent_probes=P("par")),
        ),
        render_order="sys_first",
    )
    assert (v.brk, v.cause, v.other_breaks) == ("node:1", "history_rewritten", 1)
    assert v.counterfactual_tokens == HEAD_TOKENS + 100 + 200


def test_lost_tokens_is_counterfactual_minus_actual_never_negative() -> None:
    base = ci_sys_changed()
    assert evaluate(base, render_order="sys_first").lost_tokens == REUSABLE
    some = dataclasses.replace(base, actual_tokens=1_000)
    assert evaluate(some, render_order="sys_first").lost_tokens == REUSABLE - 1_000
    more = dataclasses.replace(base, actual_tokens=REUSABLE + 5_000)
    assert evaluate(more, render_order="sys_first").lost_tokens == 0
    unknown = dataclasses.replace(base, actual_tokens=None)
    v = evaluate(unknown, render_order="sys_first")
    assert v.counterfactual_tokens == REUSABLE and v.lost_tokens is None


def test_stable_hit_has_no_cause_but_keeps_the_counterfactual() -> None:
    v = evaluate(ci(actual_tokens=REUSABLE), render_order="sys_first")
    assert v == LensVerdict(None, None, None, None, REUSABLE, 0, 0)


# -- never a false attribution (Review Focus 1) --------------------------------


def test_no_parent_in_memory_is_not_attributed() -> None:  # Review Focus 1
    v = evaluate(ci(edge="continue", obs=None), render_order="sys_first")
    assert v == NONE


def test_continue_without_parent_chain_is_not_attributed() -> None:
    v = evaluate(ci(parent_msgs=()), render_order="sys_first")
    assert v == NONE
    v = evaluate(
        ci(obs=Observation("continue", None, False, "s", parent_msgs=PM)),
        render_order="sys_first",
    )
    assert v == NONE


def test_changed_message_without_probes_is_not_history_rewritten() -> None:
    now = (PM[0], _b8("changed"), PM[2])
    for pp, pn in ((None, P("now")), (P("par"), None), (None, None)):
        v = evaluate(
            ci(
                msgs_now=now,
                probes_now=pn,
                obs=Observation("continue", "p", True, "s", parent_msgs=PM, parent_probes=pp),
            ),
            render_order="sys_first",
        )
        assert v == NONE


def test_loose_match_with_nothing_located_is_not_attributed() -> None:
    # lineage says the exact chain differs, but no part we can see does: unknown head change
    v = evaluate(
        ci(
            own_outcome="lost",
            sys_parent=None,
            tools_parent=None,
            obs=Observation("continue", "p", True, "s", parent_msgs=PM),
        ),
        render_order="sys_first",
    )
    assert v == NONE


def test_new_conversation_only_on_edge_new() -> None:
    for edge in ("continue", "fork", "subagent", None):
        v = evaluate(ci(edge=edge, obs=None), render_order="sys_first")  # type: ignore[arg-type]
        assert v.cause != "new_conversation"


def _multi_turn_new_edge():
    """A 40-message continuing body observed by a Lineage with empty memory
    (a restart, or past the TTL): lineage says ``new``."""
    msgs = [{"role": "system", "content": "be brief"}]
    for i in range(40):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"})
    c = canonicalize({"messages": msgs})
    assert c is not None
    ids = ids_for(c, KEY, SALT)
    obs = Lineage().observe(
        model_id="m",
        scope=b"",
        session_key="s",
        parent_session_key=None,
        ids=ids,
        last_user_text_loose="turn 38",
        first_body_text_loose="turn 0",
    )
    assert (obs.edge, obs.parent, obs.parent_msgs) == ("new", None, ())
    return obs, tuple(short_msg(m) for m in ids.msgs)


def test_new_edge_with_prior_turns_and_no_parent_is_not_attributed() -> None:  # Review Focus 1
    obs, now = _multi_turn_new_edge()
    toks = (10,) * len(now)
    # unknown body, a body with prior turns, with and without the same recent head
    for first_turn in (None, False):
        for recent in (True, False):
            parent = {} if recent else dict(sys_parent=None, tools_parent=None)
            v = evaluate(
                ci(
                    edge=obs.edge,
                    obs=obs,
                    msgs_now=now,
                    msg_tokens=toks,
                    first_turn=first_turn,
                    **parent,
                ),
                render_order="sys_first",
            )
            assert v == NONE, (first_turn, recent)
    # the bare inputs Task 1.3 has with nothing else known
    assert (
        evaluate(CauseInputs(edge=obs.edge, obs=obs, msgs_now=now), render_order="sys_first")
        == NONE
    )


def test_new_edge_with_prior_turns_keeps_a_changed_recent_head() -> None:
    obs, now = _multi_turn_new_edge()
    v = evaluate(
        ci_sys_changed(edge=obs.edge, obs=obs, msgs_now=now, first_turn=False),
        render_order="sys_first",
    )
    assert (v.brk, v.cause) == ("sys", "sys_changed")


def test_subagent_with_the_same_head_is_not_attributed() -> None:
    v = evaluate(
        ci(edge="subagent", obs=Observation("subagent", "p", False, "s")),
        render_order="sys_first",
    )
    assert v == NONE


def test_unknown_edge_is_not_attributed() -> None:
    assert evaluate(ci(edge=None, obs=None), render_order="sys_first") == NONE  # type: ignore[arg-type]


# -- robustness -----------------------------------------------------------------


def test_never_raises_on_missing_optional_inputs() -> None:
    bare = CauseInputs(edge="continue", obs=Observation("continue", "p", parent_msgs=PM))
    v = evaluate(bare, render_order="sys_first")
    assert isinstance(v, LensVerdict)
    assert evaluate(CauseInputs(edge=None, obs=None), render_order="tools_first") == NONE


def test_short_token_list_and_raising_line_reader_do_not_raise() -> None:
    def bad(_n: int) -> str:
        raise ValueError("no text")

    v = evaluate(ci_sys_changed(msg_tokens=(1,), sys_text_line=bad), render_order="sys_first")
    assert (v.cause, v.span_line, v.span_class) == ("sys_changed", 2, None)
    assert v.counterfactual_tokens == HEAD_TOKENS + 1


def test_internal_error_returns_the_empty_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a, **_k):
        raise RuntimeError("bug")

    monkeypatch.setattr(causes, "_evaluate", boom)
    assert causes.evaluate(ci_sys_changed(), render_order="sys_first") == NONE


def test_evaluate_is_a_module_level_function() -> None:
    assert causes.evaluate is evaluate
    assert callable(causes.evaluate)


def test_empty_verdict_is_null_in_every_field() -> None:
    """M2: an unattributed turn stores no lens field at all (not other_breaks 0)."""
    assert causes.EMPTY == NONE
    assert all(getattr(causes.EMPTY, f.name) is None for f in dataclasses.fields(LensVerdict))
