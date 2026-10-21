"""A broken MLX load fails in seconds, not load_stall_timeout_s (BACKLOG-macos P5).

mlx_lm.server does not exit when its model fails to load: the load runs on a
background generation thread, which logs one ``mlx_lm.server generation thread
died: ...`` line and dies, while the HTTP server keeps answering 503 on
/health. Before the fail-fast the warden waited the full stall timeout
(1200 s) to give such a load up.

This drives the REAL engine on a broken COPY of the smoke model (every file
copied except the weights, which are replaced by random bytes -- the source
dir is only read) and the REAL load-progress watcher on its log, with the
production stall thresholds and the production 5 s tick, then builds
last_error the way the load route's give-up handler does. It proves, on Apple
silicon:

* the engine is still alive and still answering 503 when the load is given
  up -- the engine would never have ended this load on its own;
* the give-up is a ``fatal`` verdict, within 60 s of spawn;
* last_error carries the engine's own fatal line and the log tail.

Same prerequisites and skips as test_mlx_live_smoke.py (darwin/arm64, mlx_lm
importable, LMW_MLX_SMOKE_MODEL a directory of MLX weights). A SKIP is not
green.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.integration.test_mlx_live_smoke import (
    _free_port,
    _launch,
    _log_tail,
    _skip_if_no_mlx,
    _SmokeModel,
    _terminate,
)

pytestmark = [pytest.mark.macos, pytest.mark.live]

_FAIL_FAST_BUDGET_S = 60.0


def _health_status(url: str) -> int:
    """The /health status code, 503 included (the smoke's _http_get folds
    every HTTPError into 0, and 503 is the very answer this test needs)."""
    try:
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, OSError):
        return 0


def _broken_copy(src: Path, dst: Path) -> None:
    """Copy the model's top-level files into ``dst``, replacing every
    ``*.safetensors`` with 4 KiB of random bytes. Never writes under ``src``."""
    dst.mkdir()
    weights = 0
    for f in sorted(src.iterdir()):
        if not f.is_file():
            continue
        if f.suffix == ".safetensors":
            (dst / f.name).write_bytes(os.urandom(4096))
            weights += 1
        else:
            shutil.copy2(f, dst / f.name)
    assert weights, f"{src} has no *.safetensors to corrupt -- not an MLX model dir"


def test_corrupt_weights_fail_fast_with_the_tail(tmp_path, monkeypatch):
    _skip_if_no_mlx()
    from app.models.routes_api import _fatal_load_suffix
    from app.runtime.backends import registry
    from app.runtime.backends.mlx import MlxBackend
    from app.runtime.backends.paths import ResolvedModelPaths
    from app.runtime.load_progress.follow import LogFollower
    from app.runtime.load_progress.policy import StallPolicy, Verdict
    from app.runtime.load_progress.registry import LoadProgressRegistry
    from app.runtime.load_progress.tracker import LoadTracker, VitalsSample
    from app.runtime.load_progress.watcher import watch_load

    # The route resolves the diagnosis grammar through the registry; make sure
    # this process's registry serves the row's backend whatever the profile.
    monkeypatch.setitem(registry._BACKENDS, "mlx", MlxBackend())

    broken = tmp_path / "broken-model"
    _broken_copy(Path(os.environ["LMW_MLX_SMOKE_MODEL"]), broken)

    model = _SmokeModel(id="mlx-fail-fast")
    port = _free_port()
    backend = MlxBackend()
    resolved = ResolvedModelPaths(model_path=str(broken), snapshot_dir=str(broken))
    plan = backend.plan(model, port=port, bind_host="127.0.0.1", resolved=resolved)
    health = backend.health_url("127.0.0.1", port)

    settings = SimpleNamespace(logs_dir=tmp_path / "logs")
    settings.logs_dir.mkdir()
    log_path = settings.logs_dir / f"{model.id}.log"
    # The real policy thresholds -- the fail-fast must not lean on a short one.
    policy = StallPolicy(warn_s=300.0, give_up_s=1200.0, max_s=7200.0)

    given: list[tuple[float, Verdict, bool]] = []

    t0 = time.monotonic()
    proc = _launch(plan.argv, plan.env, log_path)
    try:
        # Built right after the spawn, as the load route builds it.
        follower = LogFollower(log_path)
        tracker = LoadTracker(
            model_id=model.id,
            generation=1,
            backend="mlx",
            started_at=t0,
            started_wall=datetime.now(UTC),
        )
        progress = LoadProgressRegistry()
        progress.start(tracker)

        async def sample() -> VitalsSample:
            return VitalsSample(
                at=time.monotonic(), cpu_seconds=None, gpu_mem_mib=None, read_bytes=None
            )

        async def on_give_up(v: Verdict) -> None:
            given.append((time.monotonic() - t0, v, proc.poll() is None))

        async def run() -> None:
            await asyncio.wait_for(
                watch_load(
                    tracker=tracker,
                    follower=follower,
                    sample=sample,
                    policy=policy,
                    still_current=lambda: True,
                    on_give_up=on_give_up,
                    registry=progress,
                    interval_s=5.0,  # production tick
                ),
                timeout=_FAIL_FAST_BUDGET_S,
            )

        try:
            asyncio.run(run())
        except TimeoutError:
            pytest.fail(
                f"no give-up within {_FAIL_FAST_BUDGET_S:.0f}s of spawn "
                f"(exit={proc.poll()}):\n{_log_tail(log_path)}"
            )
        follower.close()

        assert len(given) == 1, given
        elapsed, verdict, alive = given[0]
        assert verdict.fatal is True
        assert verdict.state == "give_up"
        assert "mlx_lm.server generation thread died:" in (verdict.reason or "")
        assert elapsed < _FAIL_FAST_BUDGET_S
        assert alive, "the engine exited on its own -- the fail-fast proved nothing"
        # The fatal line lands before the HTTP server binds, so /health may not
        # answer yet AT the give-up. Teeth for "the engine would never have
        # ended this load itself": it comes up answering 503 and stays alive.
        code = 0
        deadline = time.monotonic() + 15.0
        while code != 503 and time.monotonic() < deadline:
            code = _health_status(health)
            if code != 503:
                time.sleep(0.5)
        assert code == 503, f"/health answered {code}, expected the load-failed 503"
        assert proc.poll() is None, "the engine exited on its own after the fatal line"

        last_error = (
            f"{verdict.reason} — gave up; engine stopped and GPUs released"
            + _fatal_load_suffix(settings, model.id, "mlx")
        )
        assert "--- engine log tail ---" in last_error
        assert "load_safetensors" in last_error
        print(
            f"mlx fail-fast: pid={proc.pid} port={port} gave_up_after={elapsed:.1f}s "
            f"(policy give_up_s={policy.give_up_s:.0f}s) alive={alive} health={code}\n"
            f"last_error:\n{last_error}"
        )
    finally:
        _terminate(proc)
