"""The live in-flight row carries a session_ref link only when cobs2 gave it a session."""

from app.proxy.request_registry import LiveRequest
from app.stats.live_requests import _serialize


def _req(**kw):
    base = dict(
        id="a" * 32,
        token_id="tok1",
        token_name="k",
        client_ip=None,
        model="m",
        model_row_id="m",
        path="/v1/chat/completions",
        prompt_tokens=1,
        max_model_len=None,
        started_monotonic=0.0,
        started_iso="2026-10-25T10:00:00Z",
    )
    base.update(kw)
    return LiveRequest(**base)


def test_live_row_carries_request_ref_only_with_a_session():
    assert _serialize(_req(cobs2_session="x"), 1.0)["session_ref"] == "r_" + "a" * 32
    assert _serialize(_req(), 1.0)["session_ref"] is None
