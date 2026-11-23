"""A non-streaming request the engine is running is not "waiting".

The proxy sees no token of a non-stream reply until it is complete, so the
row used to read phase "waiting", out 0, TTFT "-" for the whole generation.
Once the body is forwarded the row reads ``running_no_stream``; "waiting"
stays for the moment between admission and forward. vLLM 0.32 exports no
per-request series, so ``engine_progress_tokens`` is null for such a row
(never 0), and is the observed count on a streamed row.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from tests.unit.proxy.test_cache_estimate_forward import _body, _live, _live_rows, _spy_registry
from tests.unit.proxy.test_dp_routing_forward import OK_BODY, _post, _ready, _resp


def test_non_stream_row_is_running_while_the_engine_generates(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _spy_registry(client)
    seen = []
    resp = _resp()

    async def aread():
        seen.append([r.phase for r in client.app.state.request_registry.snapshot()])
        import json

        return json.dumps(OK_BODY).encode()

    resp.aread = AsyncMock(side_effect=aread)
    r, _ = _post(client, _body(5), resp=resp)
    assert r.status_code == 200
    assert seen == [["running_no_stream"]]
    assert _live_rows(client)[0].is_stream is False


def test_serialized_non_stream_row_has_null_engine_progress():
    from app.stats.live_requests import _serialize

    row = _serialize(_live(phase="running_no_stream", is_stream=False), 30.0)
    assert row["phase"] == "running_no_stream"
    assert row["engine_progress_tokens"] is None
    assert row["prefill_slow"] is False


def test_streamed_row_reports_its_observed_tokens_as_progress():
    from app.stats.live_requests import _serialize

    row = _serialize(_live(phase="answering", completion_tokens=42), 5.0)
    assert row["engine_progress_tokens"] == 42
    # Before the first token a streamed row has made no observed progress yet.
    assert _serialize(_live(phase="prefill"), 1.0)["engine_progress_tokens"] == 0


def test_queued_or_waiting_rows_report_no_progress():
    from app.stats.live_requests import _serialize

    for phase in ("queued", "waiting"):
        row = _serialize(_live(phase=phase, is_stream=False), 1.0)
        assert row["engine_progress_tokens"] is None


def test_key_holder_own_row_carries_engine_progress():
    from app.tokens.routes_me_stats import _own_row

    row = _own_row(_live(phase="running_no_stream", is_stream=False), 30.0, [], {})
    assert row["phase"] == "running_no_stream" and row["engine_progress_tokens"] is None
