"""RouterState: breakers, counters, ring, snapshot, ruleset cache (#287)."""

from pathlib import Path

import pytest

import app.router.state as state_mod
from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.models import ModelRepo, ModelRow
from app.db.repos.router_rules import RouterRuleRepo
from app.db.repos.settings import SettingsRepo
from app.router.state import Decision, RouterState


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _dec(**over) -> Decision:
    base = dict(
        ts=1000.0,
        path="/v1/messages",
        model_in="claude-haiku-4",
        model_out="qwen",
        rule_id="r1",
        route="local",
        reason=None,
        status=200,
        latency_ms=100,
        ttfb_ms=40,
        stream=True,
        token_name="laptop",
    )
    base.update(over)
    return Decision(**base)


def test_breaker_opens_on_third_consecutive_failure_then_half_opens():
    clk = Clock()
    s = RouterState(now=clk)
    assert s.breaker_allows("m1")
    assert s.record_failure("m1", "status_502", threshold=3, open_s=60) is False
    assert s.record_failure("m1", "status_502", threshold=3, open_s=60) is False
    assert s.breaker_allows("m1")
    assert s.record_failure("m1", "status_502", threshold=3, open_s=60) is True
    assert not s.breaker_allows("m1")
    clk.t += 59
    assert not s.breaker_allows("m1")
    clk.t += 2
    assert s.breaker_allows("m1")  # half-open probe


def test_failure_in_half_open_reopens():
    clk = Clock()
    s = RouterState(now=clk)
    for _ in range(3):
        s.record_failure("m1", "x", threshold=3, open_s=60)
    clk.t += 61
    assert s.breaker_allows("m1")
    assert s.record_failure("m1", "x", threshold=3, open_s=60) is True
    assert not s.breaker_allows("m1")
    clk.t += 61
    assert s.breaker_allows("m1")


def test_success_closes_and_resets():
    clk = Clock()
    s = RouterState(now=clk)
    for _ in range(3):
        s.record_failure("m1", "x", threshold=3, open_s=60)
    clk.t += 61
    s.record_success("m1")
    assert s.breaker_allows("m1")
    # counter was reset: two more failures do not open it
    s.record_failure("m1", "x", threshold=3, open_s=60)
    assert s.record_failure("m1", "x", threshold=3, open_s=60) is False
    assert s.breaker_allows("m1")


def test_success_interrupts_consecutive_run():
    s = RouterState(now=Clock())
    s.record_failure("m1", "x", threshold=3, open_s=60)
    s.record_failure("m1", "x", threshold=3, open_s=60)
    s.record_success("m1")
    assert s.record_failure("m1", "x", threshold=3, open_s=60) is False


def test_breakers_are_per_model():
    s = RouterState(now=Clock())
    for _ in range(3):
        s.record_failure("m1", "x", threshold=3, open_s=60)
    assert not s.breaker_allows("m1")
    assert s.breaker_allows("m2")


def test_record_counts_and_ring():
    s = RouterState(now=Clock())
    s.record(_dec(), rule_pattern="claude-haiku*")
    s.record(_dec(route="fallback", reason="status_502", status=200), rule_pattern="claude-haiku*")
    s.record(_dec(route="passthrough", model_out=None, rule_id=None, reason=None))
    s.record(_dec(route="refused", reason="token_not_allowed", status=403))
    snap = s.snapshot(None)
    assert snap["totals"] == {"local": 1, "passthrough": 1, "fallback": 1, "refused": 1, "error": 0}
    assert snap["by_reason"] == {"status_502": 1, "token_not_allowed": 1}
    assert snap["passthrough"]["requests"] == 1
    assert snap["since"] is not None


def test_by_reason_counts_only_fallback_and_refused():
    s = RouterState(now=Clock())
    s.record(_dec(reason="status_400", status=400))  # local error
    s.record(_dec(route="fallback", reason="status_400", status=200))
    s.record(
        _dec(route="passthrough", model_out=None, rule_id=None, reason="status_529", status=529)
    )
    snap = s.snapshot(None)
    assert snap["by_reason"] == {"status_400": 1}
    assert snap["error_reasons"] == {"status_400": 1, "status_529": 1}
    s.reset()
    assert s.snapshot(None)["error_reasons"] == {}


def test_ring_keeps_200_newest_first():
    s = RouterState(now=Clock())
    for i in range(250):
        s.record(_dec(latency_ms=i))
    out = s.decisions(500)
    assert len(out) == 200
    assert out[0]["latency_ms"] == 249
    assert out[-1]["latency_ms"] == 50
    assert [d["latency_ms"] for d in s.decisions(2)] == [249, 248]


def test_decision_model_in_truncated_and_shape():
    s = RouterState(now=Clock())
    s.record(_dec(model_in="x" * 300))
    d = s.decisions(1)[0]
    assert len(d["model_in"]) == 128
    assert set(d) == {
        "ts",
        "path",
        "model_in",
        "model_out",
        "rule_id",
        "route",
        "reason",
        "status",
        "latency_ms",
        "ttfb_ms",
        "stream",
        "token_name",
    }
    assert d["ts"].endswith("Z")


def test_empty_snapshot_has_every_contract_key():
    snap = RouterState(now=Clock()).snapshot(None)
    assert set(snap) == {
        "enabled",
        "since",
        "totals",
        "by_reason",
        "error_reasons",
        "rules",
        "targets",
        "passthrough",
    }
    assert snap["enabled"] is False
    assert snap["since"] is None
    assert snap["totals"] == {"local": 0, "passthrough": 0, "fallback": 0, "refused": 0, "error": 0}
    assert snap["by_reason"] == {}
    assert snap["rules"] == [] and snap["targets"] == []
    pt = snap["passthrough"]
    assert pt["requests"] == 0 and pt["errors"] == 0
    assert pt["latency_ms"] == {"p50": None, "p95": None}
    assert pt["ttfb_ms"] == {"p50": None, "p95": None}
    assert pt["input_tokens"] == 0 and pt["output_tokens"] == 0


def test_percentiles_from_samples():
    s = RouterState(now=Clock())
    for ms in range(1, 101):
        s.record(_dec(latency_ms=ms, ttfb_ms=ms), rule_pattern="claude-haiku*")
    s.record_failure("m1", "status_502", threshold=3, open_s=60)
    from app.router.rules import RouterConfig, RuleSet, RuleSpec

    rs = RuleSet(
        config=RouterConfig(enabled=True),
        rules=(RuleSpec("r1", 0, "claude-haiku*", "m1", True, True, True, 0),),
        served_names={"m1": "qwen"},
        version=1,
    )
    snap = s.snapshot(rs)
    assert snap["enabled"] is True
    (rule,) = snap["rules"]
    assert rule["rule_id"] == "r1" and rule["target_served_name"] == "qwen"
    assert rule["local"] == 100
    assert rule["latency_ms"]["p50"] in (50, 51)
    assert rule["latency_ms"]["p95"] in (95, 96)
    (tgt,) = snap["targets"]
    assert tgt["model_id"] == "m1" and tgt["served_name"] == "qwen"
    assert tgt["breaker"] == "closed" and tgt["consecutive_failures"] == 1
    assert tgt["failures"] == 1 and tgt["last_reason"] == "status_502"
    assert tgt["open_until"] is None


def test_snapshot_reports_open_breaker():
    clk = Clock()
    s = RouterState(now=clk)
    for _ in range(3):
        s.record_failure("m1", "x", threshold=3, open_s=60)
    from app.router.rules import RouterConfig, RuleSet, RuleSpec

    rs = RuleSet(
        RouterConfig(enabled=True),
        (RuleSpec("r1", 0, "claude-*-x", "m1", True, True, True, 0),),
        {"m1": "qwen"},
        1,
    )
    (tgt,) = s.snapshot(rs)["targets"]
    assert tgt["breaker"] == "open" and tgt["open_until"].endswith("Z")
    clk.t += 61
    (tgt,) = s.snapshot(rs)["targets"]
    assert tgt["breaker"] == "half_open"


def test_reset_clears_everything():
    s = RouterState(now=Clock())
    for _ in range(3):
        s.record_failure("m1", "x", threshold=3, open_s=60)
    s.record(_dec(), rule_pattern="p-*")
    s.reset()
    snap = s.snapshot(None)
    assert snap["since"] is None and snap["totals"]["local"] == 0
    assert s.decisions(10) == []
    assert s.breaker_allows("m1")


def test_record_never_raises():
    s = RouterState(now=Clock())
    s.record(None, rule_pattern=None)  # type: ignore[arg-type]
    s.record(_dec(latency_ms="boom"))  # type: ignore[arg-type]


def _model(**over) -> ModelRow:
    base = dict(
        id="m1",
        served_model_name="qwen",
        hf_repo="a/b",
        hf_revision="main",
        gpu_indices=[0],
        tensor_parallel_size=1,
        dtype="auto",
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        trust_remote_code=False,
        extra_args=[],
        status="loaded",
        pulled_bytes=0,
        pulled_total=None,
        last_error=None,
        extra_env={},
    )
    base.update(over)
    return ModelRow(**base)


@pytest.mark.asyncio
async def test_ruleset_reloads_only_when_dirty_and_joins_served_names(tmp_path, monkeypatch):
    db_path: Path = tmp_path / "t.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await ModelRepo(db).insert(_model())
        await RouterRuleRepo(db).create("claude-haiku*", "m1", True, True, True, 0)
        await RouterRuleRepo(db).create("claude-opus*", "gone", True, False, True, 0)
        await SettingsRepo(db).set("router_enabled", "true")

    opens = 0
    real = state_mod.open_db

    def spy(path):
        nonlocal opens
        opens += 1
        return real(path)

    monkeypatch.setattr(state_mod, "open_db", spy)
    s = RouterState()
    rs1 = await s.ruleset(db_path)
    assert opens == 1
    assert rs1.config.enabled is True
    assert [r.pattern for r in rs1.rules] == ["claude-haiku*", "claude-opus*"]
    assert rs1.served_names == {"m1": "qwen", "gone": None}
    rs2 = await s.ruleset(db_path)
    assert opens == 1 and rs2 is rs1
    s.invalidate()
    rs3 = await s.ruleset(db_path)
    assert opens == 2 and rs3.version > rs1.version


def test_app_wires_router_state_and_shared_http_client(client):
    import httpx

    assert isinstance(client.app.state.router, RouterState)
    http = client.app.state.router_http
    assert isinstance(http, httpx.AsyncClient)
    assert http.follow_redirects is False


# -- review #1: half-open admits exactly one probe ------------------------------


def _open(s: RouterState) -> None:
    for _ in range(3):
        s.record_failure("m1", "x", threshold=3, open_s=60)


def test_half_open_admits_only_the_probe():
    clk = Clock()
    s = RouterState(now=clk)
    assert s.breaker_admit("m1") == 0  # closed: admitted, no probe claim
    _open(s)
    assert s.breaker_admit("m1") is None
    clk.t += 61
    claim = s.breaker_admit("m1")
    assert claim  # the probe
    assert s.breaker_admit("m1") is None  # everyone else keeps falling back
    assert not s.breaker_allows("m1")


def test_probe_release_admits_the_next_one_and_is_per_claim():
    clk = Clock()
    s = RouterState(now=clk)
    _open(s)
    clk.t += 61
    first = s.breaker_admit("m1")
    s.release_probe("m1", 12345)  # somebody else's claim: no effect
    assert s.breaker_admit("m1") is None
    s.release_probe("m1", first)
    second = s.breaker_admit("m1")
    assert second and second != first


def test_probe_outcome_clears_the_claim():
    clk = Clock()
    s = RouterState(now=clk)
    _open(s)
    clk.t += 61
    assert s.breaker_admit("m1")
    s.record_success("m1")
    assert s.breaker_admit("m1") == 0  # closed again
    _open(s)
    clk.t += 61
    assert s.breaker_admit("m1")
    s.record_failure("m1", "x", threshold=3, open_s=60)  # probe failed: re-opened
    assert s.breaker_admit("m1") is None
    clk.t += 61
    assert s.breaker_admit("m1")  # and the next period has a fresh probe


def test_abandoned_probe_claim_expires():
    clk = Clock()
    s = RouterState(now=clk)
    _open(s)
    clk.t += 61
    assert s.breaker_admit("m1")
    assert s.breaker_admit("m1") is None
    clk.t += state_mod.PROBE_TTL_S + 1
    assert s.breaker_admit("m1")


def test_failed_probe_reopens_even_when_the_threshold_was_raised_meanwhile():
    clk = Clock()
    s = RouterState(now=clk)
    _open(s)
    clk.t += 61
    assert s.breaker_admit("m1")
    # breaker_threshold was raised via the settings API while it was open: the
    # failed probe re-opens regardless of the run length.
    assert s.record_failure("m1", "x", threshold=50, open_s=60) is True
    assert not s.breaker_allows("m1")
