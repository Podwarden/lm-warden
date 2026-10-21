"""The stress runner's reload path uses the load stall policy, not a wall clock.

Drives the real ``WardenEngineControl.reload_with`` (app/stress/runner.py): the
restart and warmup seams are faked, but the health wait is the real
``wait_for_health`` polling a real local HTTP server.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.config import Settings
from app.runtime.warmup_probe import ProbeResult
from app.stress import runner as runner_mod


class _HealthServer:
    """A local /health endpoint: 503 until ``healthy_after`` seconds, then 200.
    ``healthy_after=None`` means it never becomes healthy."""

    def __init__(self, healthy_after: float | None) -> None:
        self.started = time.monotonic()
        self.healthy_after = healthy_after
        server = self

        class _H(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                ready = server.healthy_after is not None and (
                    time.monotonic() - server.started >= server.healthy_after
                )
                self.send_response(200 if ready else 503)
                self.end_headers()

            def log_message(self, *args):  # silence test output
                return

        self._httpd = HTTPServer(("127.0.0.1", 0), _H)
        self.port = self._httpd.server_address[1]
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


@dataclass
class _FakeSup:
    """Just the supervisor surface WardenEngineControl.target()/reload_with read."""

    port: int | None = None
    alive: bool = False

    def get_port(self, model_id):
        return self.port

    def get_host(self, model_id):
        return "127.0.0.1"

    def get_overrides(self, model_id):
        return None

    def is_running(self, model_id):
        return self.alive

    engine_id: str | None = None

    def get_engine_model_id(self, model_id):
        return self.engine_id


def _settings(tmp_path: Path, **kw) -> Settings:
    base = dict(
        data_dir=tmp_path,
        hf_cache_dir=tmp_path,
        cookie_secret="x" * 32,
        container_gpu_count=0,
        load_timeout_s=0.05,  # the old strict wall clock, deliberately tiny
        load_max_s=30.0,
        load_stall_warn_s=300.0,
        load_stall_timeout_s=1200.0,
    )
    base.update(kw)
    return Settings(**base)


def _control(settings: Settings, sup: _FakeSup) -> runner_mod.WardenEngineControl:
    row = SimpleNamespace(id="m1", served_model_name="m1")
    app_state = SimpleNamespace(supervisor=sup, restart_budget=None)
    return runner_mod.WardenEngineControl(
        settings,
        app_state,
        row,
        subject=None,
        caps=None,
    )


def _fake_restart(sup: _FakeSup, port: int, *, engine_dies_after: float | None = None):
    """Stands in for watchdog._restart: a fresh engine on ``port``. When
    ``engine_dies_after`` is set, the engine is torn down that many seconds
    later, the way a stall give-up unloads it."""

    async def restart(settings, app_state, model_id, overrides, **_kw):
        sup.port = port
        sup.alive = True
        if engine_dies_after is not None:

            async def _die():
                await asyncio.sleep(engine_dies_after)
                sup.alive = False

            asyncio.get_running_loop().create_task(_die())
        return "loaded"

    return restart


@pytest.mark.asyncio
async def test_reload_waits_past_the_old_wall_clock_for_a_slow_healthy_load(tmp_path):
    """The old reload gave up at ``load_timeout_s``. A slow but progressing
    load that becomes healthy well after that must still count as loaded, and
    the health wait must use the stall policy's cap, not ``load_timeout_s``."""
    server = _HealthServer(healthy_after=0.4)
    sup = _FakeSup()
    settings = _settings(tmp_path)
    control = _control(settings, sup)
    seen_timeouts: list[float] = []

    from app.runtime import supervisor as sup_mod

    real = sup_mod.wait_for_health

    async def spy(**kwargs):
        seen_timeouts.append(kwargs["timeout_s"])
        return await real(**kwargs)

    try:
        with (
            patch("app.runtime.watchdog._restart", new=_fake_restart(sup, server.port)),
            patch.object(sup_mod, "wait_for_health", new=spy),
            patch(
                "app.runtime.warmup_probe.warmup_probe",
                new=_ok_probe,
            ),
        ):
            ok = await control.reload_with(max_model_len=1024)
    finally:
        server.close()
    assert ok is True, "a slow, progressing load must not be reported as won't-load"
    assert seen_timeouts == [settings.load_max_s], (
        f"reload must wait for the stall-policy cap {settings.load_max_s}, "
        f"not the old wall clock {settings.load_timeout_s}; saw {seen_timeouts}"
    )


@pytest.mark.asyncio
async def test_a_stuck_reload_is_won_t_load_via_the_policy_not_the_wall_clock(tmp_path):
    """An engine that never becomes healthy and is torn down by the stall
    give-up is reported as ``won't load`` (False) promptly, once the engine is
    gone. It does not sit out the 30 s cap, and it is not a strict wall clock."""
    server = _HealthServer(healthy_after=None)
    sup = _FakeSup()
    settings = _settings(tmp_path)
    control = _control(settings, sup)
    warmed = []

    async def never_warm(**_kw):
        warmed.append(1)
        return ProbeResult(ok=True, detail=None)

    try:
        with (
            patch(
                "app.runtime.watchdog._restart",
                new=_fake_restart(sup, server.port, engine_dies_after=0.3),
            ),
            patch("app.runtime.warmup_probe.warmup_probe", new=never_warm),
        ):
            started = time.monotonic()
            ok = await control.reload_with(max_model_len=1024)
            elapsed = time.monotonic() - started
    finally:
        server.close()
    assert ok is False, "a reload whose engine never came up must be won't-load"
    assert warmed == [], "a stuck reload must not reach the warmup probe"
    assert elapsed < 5.0, f"won't-load must return once the engine is gone, took {elapsed:.2f}s"


async def _ok_probe(**_kw):
    return ProbeResult(ok=True, detail=None)


@pytest.mark.parametrize(
    ("engine_id", "expect"),
    [("/cache/mlx/Qwen-4bit", "/cache/mlx/Qwen-4bit"), (None, "m1"), ("m1", "m1")],
    ids=["mlx-like", "no-engine-id", "vllm-like"],
)
@pytest.mark.asyncio
async def test_stress_warmup_probes_the_engines_model_id(tmp_path, engine_id, expect):
    sup = _FakeSup()
    sup.engine_id = engine_id
    sup.port = 1
    control = _control(_settings(tmp_path), sup)
    seen: list[dict] = []

    async def spy(**kw):
        seen.append(kw)
        return ProbeResult(True, None)

    with patch("app.runtime.warmup_probe.warmup_probe", new=spy):
        assert await control.warmup() is True
    assert seen[0]["model_id"] == expect
