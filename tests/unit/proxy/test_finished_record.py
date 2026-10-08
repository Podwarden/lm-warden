"""The finished-request record's identity fields.

``model_id`` is what every stats endpoint filters on, and callers supply the
models table's row id there -- ``/api/stats/v2/overview`` validates exactly
that (``_resolve_selection``). The finished ring was being written with the
SERVED model name under that key, so a selection that filtered the overview
correctly matched nothing here whenever the two differ: a silently empty
table, which reads as "no requests" rather than as a bug.

These tests pin the producer, not the ring. The existing ring tests inject
records of their own making, so nothing checked what the proxy actually
writes -- which is how this got shipped.
"""

from app.proxy.request_registry import LiveRequest, finished_record


def _req(**kw: object) -> LiveRequest:
    base = dict(
        id="r1",
        token_id="t1",
        token_name="dev",
        client_ip="127.0.0.1",
        model="vendor-a/model-a-8b",  # served_model_name
        model_row_id="id-model-a-8b",  # models table row id
        path="/v1/chat/completions",
        prompt_tokens=12,
        max_model_len=4096,
        started_monotonic=100.0,
        started_iso="2026-09-06T00:00:00Z",
    )
    base.update(kw)
    return LiveRequest(**base)  # type: ignore[arg-type]


def test_model_id_is_the_row_id_not_the_served_name() -> None:
    rec = finished_record(_req(), now=101.0)
    assert rec["model_id"] == "id-model-a-8b"


def test_the_served_name_is_still_carried_for_display() -> None:
    rec = finished_record(_req(), now=101.0)
    assert rec["model"] == "vendor-a/model-a-8b"


def test_ttft_is_none_when_no_first_token_was_seen() -> None:
    rec = finished_record(_req(), now=101.0)
    assert rec["ttft_s"] is None


def test_ttft_is_measured_from_the_first_streamed_frame() -> None:
    rec = finished_record(_req(first_token_monotonic=100.25), now=101.0)
    assert rec["ttft_s"] == 0.25


def test_duration_is_measured_to_now() -> None:
    rec = finished_record(_req(), now=102.5)
    assert rec["duration_s"] == 2.5


def test_the_token_row_id_is_carried_next_to_the_name() -> None:
    # Names are reused across rotations (migration 0033); only the id says
    # which key made the request.
    rec = finished_record(_req(), now=101.0)
    assert rec["token_id"] == "t1"
    assert rec["token_name"] == "dev"


def test_forest_fields_are_recorded():
    rec = finished_record(
        _req(
            session_key="k1",
            parent_session_key="p1",
            turn_index=2,
            batch_id="b-1",
            tools_in=[["read", 10, False]],
            tools_out=[["edit", "abcd1234"]],
        ),
        now=101.0,
    )
    assert (rec["session_key"], rec["parent_session_key"], rec["turn_index"], rec["batch_id"]) == (
        "k1",
        "p1",
        2,
        "b-1",
    )
    assert rec["tools_in"] == [["read", 10, False]] and rec["tools_out"] == [["edit", "abcd1234"]]


def test_cache_obs_fields_are_null_by_default_and_merged_when_set() -> None:
    keys = (
        "reusable_tokens",
        "reusable_tokens_fleet",
        "reusable_tokens_own",
        "cache_outcome",
        "cache_outcome_own",
        "diverged_at",
        "diverged_at_own",
        "cached_source",
        "cached_ttft_est_tokens",
    )
    rec = finished_record(_req(), now=101.0)
    assert all(rec[k] is None for k in keys)
    rec = finished_record(_req(cache_obs={"cache_outcome": "hit", "reusable_tokens": 7}), now=101.0)
    assert rec["cache_outcome"] == "hit" and rec["reusable_tokens"] == 7
