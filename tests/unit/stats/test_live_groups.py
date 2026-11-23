"""B2: /api/stats/requests group aggregates for the Requests table's Group by.

Groups carry the summed live decode speed of their running requests, and a
key group never mixes models (one group per key and model)."""

from __future__ import annotations

from app.stats.live_requests import _aggregate, _groups


def _row(tid, model, tps, ip="10.0.0.1", phase="answering"):
    return {
        "token_id": tid,
        "token_name": f"n-{tid}",
        "client_ip": ip,
        "model": model,
        "context_tokens": 100,
        "prompt_tokens": 80,
        "completion_tokens": 20,
        "decode_tps_5s": tps,
        "phase": phase,
    }


def test_key_groups_split_by_model_and_sum_live_speed():
    rows = [
        _row("a", "m1", 40.0),
        _row("a", "m1", 10.0),
        _row("a", "m2", 90.0),
        _row("b", "m1", None),
    ]
    g = {(x["key"], x["model"]): x for x in _groups(rows, "key")}
    assert g[("a", "m1")]["in_flight"] == 2 and g[("a", "m1")]["live_tps"] == 50.0
    assert g[("a", "m2")]["live_tps"] == 90.0
    assert g[("b", "m1")]["live_tps"] is None  # nothing decoding: idle, not 0
    ip = _groups(rows, "ip")
    assert len(ip) == 1 and ip[0]["in_flight"] == 4 and ip[0]["model"] is None
    by_model = {x["key"]: x for x in _groups(rows, "model")}
    assert by_model["m1"]["in_flight"] == 3


def test_aggregate_keeps_its_old_shape():
    by_token, by_ip = _aggregate([_row("a", "m1", 1.0)])
    assert by_token[0]["token_id"] == "a" and by_ip[0]["client_ip"] == "10.0.0.1"
