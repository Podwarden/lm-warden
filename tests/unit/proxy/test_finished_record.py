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
        model="vendor-a/model-a-8b",       # served_model_name
        model_row_id="id-model-a-8b",     # models table row id
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
