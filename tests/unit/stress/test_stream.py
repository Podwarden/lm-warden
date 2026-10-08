"""SSE accumulation for engine-direct probes (design §9.1, §9.2).

Probes cannot go through the warden's own proxy: PriorityScheduler admits
VW_PROXY_MAX_INFLIGHT (default 16) per engine, so a concurrency run through
/v1 would measure the warden rather than the engine. Going direct means the
harness needs its own SSE client, which `warmup_probe` does not provide -- it
is a single non-streaming POST.

Parsing is separated from transport so all of this is testable by feeding
lines, with no network and no GPU.
"""

from __future__ import annotations

from app.proxy.runaway import RunawayDetector
from app.stress.stream import StreamAccumulator


def _acc(**over) -> StreamAccumulator:
    base = dict(detector=None)
    base.update(over)
    return StreamAccumulator(**base)


def _chunk(**delta) -> str:
    import json

    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]})


# ---- basic accumulation --------------------------------------------------


def test_content_deltas_accumulate_in_order():
    a = _acc()
    a.feed_line(_chunk(content="Hello "))
    a.feed_line(_chunk(content="world"))
    assert a.content == "Hello world"


def test_reasoning_accumulates_separately_from_content():
    """The gates distinguish 'answered nothing' from 'reasoned and never answered'."""
    a = _acc()
    a.feed_line(_chunk(reasoning_content="thinking..."))
    a.feed_line(_chunk(content="42"))
    assert a.reasoning_content == "thinking..."
    assert a.content == "42"


def test_reasoning_field_is_recorded_and_watched():
    """R22: newer vLLM names the channel `reasoning`. The observer must record
    it (the gates distinguish 'reasoned' from 'answered') and feed it to the
    detector (an over-reasoning generation must still trip)."""
    det = RunawayDetector(
        repeat_max=10**9, window_chars=4096, shingle_size=48, think_budget=6, hard_max=10**9
    )
    a = _acc(detector=det)
    phrases = ["pondering", "the", "question", "at", "hand", "very", "carefully"]
    for p in phrases:
        a.feed_line(_chunk(reasoning=p))
    assert a.reasoning_content == "ponderingthequestionathandverycarefully"
    assert a.repetition_tripped is True


def test_only_the_primary_choice_is_accumulated():
    """delta-count ~ token-count only holds for one choice."""
    import json

    a = _acc()
    a.feed_line(
        "data: "
        + json.dumps(
            {
                "choices": [
                    {"index": 0, "delta": {"content": "keep"}},
                    {"index": 1, "delta": {"content": "drop"}},
                ]
            }
        )
    )
    assert a.content == "keep"


def test_finish_reason_is_captured():
    import json

    a = _acc()
    a.feed_line(
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    )
    assert a.finish_reason == "stop"


def test_usage_prompt_tokens_come_from_the_engine():
    """Never from our own tokenizer count.

    TokenizerCache joins message text without the chat template, so role
    tokens, BOS and the generation prompt go uncounted -- it systematically
    undercounts. The engine's own number is the only trustworthy one.
    """
    import json

    a = _acc()
    a.feed_line(
        "data: "
        + json.dumps({"choices": [], "usage": {"prompt_tokens": 24361, "completion_tokens": 12}})
    )
    assert a.prompt_tokens == 24361
    assert a.completion_tokens == 12


# ---- terminal frames -----------------------------------------------------


def test_done_marks_the_stream_complete():
    a = _acc()
    a.feed_line(_chunk(content="hi"))
    a.feed_line("data: [DONE]")
    assert a.saw_done is True


def test_a_stream_without_done_is_not_complete():
    """This is a real state, not a theoretical one.

    The wall-clock reaper and the client-disconnect path both drop the
    connection with no terminal frame, so 'truncated with no explanation' has
    to be distinguishable from 'answered briefly'.
    """
    a = _acc()
    a.feed_line(_chunk(content="hi"))
    assert a.saw_done is False
    assert a.finish_reason is None


def test_blank_lines_and_comments_are_ignored():
    a = _acc()
    a.feed_line("")
    a.feed_line(": keep-alive")
    a.feed_line(_chunk(content="x"))
    assert a.content == "x"


def test_malformed_json_does_not_kill_the_stream():
    """One bad frame must not abort a probe that is otherwise fine."""
    a = _acc()
    a.feed_line("data: {not json")
    a.feed_line(_chunk(content="ok"))
    assert a.content == "ok"
    assert a.malformed_frames == 1


# ---- runaway detector integration ---------------------------------------


def test_the_detector_sees_content():
    det = RunawayDetector(
        repeat_max=3, window_chars=4096, shingle_size=48, think_budget=64, hard_max=256
    )
    a = _acc(detector=det)
    loop = "the quick brown fox jumps over the lazy dog and then repeats itself. "
    for _ in range(12):
        a.feed_line(_chunk(content=loop))
    assert a.repetition_tripped is True


def test_think_tags_are_synthesised_around_reasoning():
    """--reasoning-parser qwen3 strips the literal tags from the stream.

    The detector's unclosed-think signal is purely textual, so without
    synthesising a <think> on the first reasoning delta an over-reasoning
    generation -- exactly the pathology we hunt -- would never open a think
    block and the signal would be dead. This mirrors the proxy's
    _feed_detector, which exists for the same reason.
    """
    seen: list[str] = []

    class _Spy:
        def feed(self, s):
            seen.append(s)

        @property
        def tripped(self):
            return False

    a = _acc(detector=_Spy())
    a.feed_line(_chunk(reasoning_content="pondering"))
    a.feed_line(_chunk(content="answer"))
    assert seen[0] == "<think>"
    assert "</think>" in seen


def test_a_think_block_is_opened_only_once():
    seen: list[str] = []

    class _Spy:
        def feed(self, s):
            seen.append(s)

        @property
        def tripped(self):
            return False

    a = _acc(detector=_Spy())
    a.feed_line(_chunk(reasoning_content="a"))
    a.feed_line(_chunk(reasoning_content="b"))
    assert seen.count("<think>") == 1


def test_no_detector_means_no_repetition_verdict():
    """The gate is skipped for probes whose output is legitimately repetitive."""
    a = _acc(detector=None)
    for _ in range(20):
        a.feed_line(_chunk(content="same same same same same same same same "))
    assert a.repetition_tripped is False


# ---- the observation it produces ----------------------------------------


def test_it_builds_the_observation_the_gates_consume():
    import json

    a = _acc()
    a.feed_line(_chunk(content="Paris"))
    a.feed_line(
        "data: "
        + json.dumps(
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 3},
            }
        )
    )
    a.feed_line("data: [DONE]")
    obs = a.observation(graded_ok=True)
    assert obs.content == "Paris"
    assert obs.finish_reason == "stop"
    assert obs.output_tokens == 3
    assert obs.graded_ok is True


def test_a_truncated_stream_reports_no_finish_reason():
    """Which the ABRUPT gate reads as a defect, correctly."""
    a = _acc()
    a.feed_line(_chunk(content="half an ans"))
    obs = a.observation(graded_ok=None)
    assert obs.finish_reason is None


def test_completion_tokens_fall_back_to_a_delta_count():
    """Some engines omit usage on a stream.

    The SHORT gate needs a length, so an approximation beats None -- but it is
    only ever used for output length, never for the published prompt size.
    """
    a = _acc()
    for _ in range(5):
        a.feed_line(_chunk(content="tok "))
    obs = a.observation(graded_ok=None)
    assert obs.output_tokens == 5
