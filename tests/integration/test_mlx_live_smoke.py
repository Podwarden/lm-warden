"""Does the MLX engine actually serve on a real Mac? Needs Apple silicon, the
warden's own venv with ``mlx_lm`` in it, and a tiny 4-bit MLX model already
downloaded. Not in CI: CI is amd64 and has no GPU pool to talk to.

This is the metal-design s5 verification that "genuinely needs a Mac" --
``test_llamacpp_smoke.py`` is the template. It drives the SAME launch
``MlxBackend.plan()`` would: the bare ``mlx_lm.server`` name resolved through
the hard-set ``PATH`` the plan emits, and the ``--model`` value the plan picks,
so a plan that emits an argv the real binary rejects fails HERE rather than at
first load.

Point the smoke at a small model with a local directory of MLX weights:

    export LMW_MLX_SMOKE_MODEL=/path/to/mlx-community/....-4bit

A run that SKIPS is not green: mlx_lm missing, not darwin/arm64, or the env
var unset all skip with a reason, and the lane reports them as blocked.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import pytest

pytestmark = [pytest.mark.macos, pytest.mark.live]

# plan() resolves the bare console-script name through the PATH it emits, which
# puts the warden's OWN venv's bin first. The smoke must run from that same
# interpreter for the two to agree.
_HEALTH_DEADLINE_S = 180.0
_BAD_PORT_DEADLINE_S = 30.0
_SIGTERM_GRACE_S = 15.0


@dataclass
class _SmokeModel:
    """A minimal model row for the smoke: ``plan()`` reads only these fields."""

    id: str = "mlx-smoke"
    served_model_name: str = "mlx-smoke"
    gpu_indices: list[int] = field(default_factory=lambda: [0])
    max_model_len: int | None = None
    max_num_seqs: int | None = 4
    extra_args: list[str] = field(default_factory=list)
    extra_env: dict[str, str] = field(default_factory=dict)


def _skip_if_no_mlx() -> None:
    """Skip with a clear reason when the prerequisites are absent.

    ``import mlx_lm`` is imported inside this function, not at module level,
    so that a base without it skips (collects and skips) rather than turning
    this whole file into a collection ImportError that runs nothing.
    """
    if sys.platform != "darwin" or platform.machine() != "arm64":
        pytest.skip(
            f"MLX live smoke needs darwin/arm64; this is " f"{sys.platform}/{platform.machine()}"
        )
    try:
        import mlx_lm  # noqa: F401 -- presence is the whole check
    except ImportError:
        pytest.skip("mlx_lm is not importable in the warden's venv")
    model = os.environ.get("LMW_MLX_SMOKE_MODEL")
    if not model:
        pytest.skip(
            "LMW_MLX_SMOKE_MODEL is not set; export the local path to a tiny "
            "4-bit MLX model directory"
        )
    if not os.path.isdir(model):
        pytest.skip(f"LMW_MLX_SMOKE_MODEL={model!r} is not a directory of MLX weights")


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _http_get(url: str, timeout: float = 2.0) -> int:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status
    except (urllib.error.URLError, OSError):
        return 0


def _poll_until_healthy(
    health_url: str, deadline: float, proc: subprocess.Popen | None = None
) -> bool:
    start = time.monotonic()
    while time.monotonic() - start < deadline:
        if _http_get(health_url) == 200:
            return True
        if proc is not None and proc.poll() is not None:
            return False  # the engine exited: no point waiting out the deadline
        time.sleep(0.5)
    return False


def _launch(argv: list[str], env: dict[str, str], log_path) -> subprocess.Popen:
    # The engine's output goes to a FILE, never a pipe: a server that serves but
    # spews logs would fill an unread pipe and wedge. The file's tail is what a
    # failure message shows, so a red run says why the engine died.
    with open(log_path, "wb") as log:
        return subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT)


def _log_tail(log_path, n: int = 2000) -> str:
    try:
        with open(log_path, "rb") as f:
            return f.read()[-n:].decode(errors="replace")
    except OSError:
        return "<no engine log>"


def _terminate(proc: subprocess.Popen) -> None:
    # Only the PID this test started: no pattern kills, no process-group kills.
    if proc.poll() is not None:
        return
    proc.terminate()  # SIGTERM
    try:
        proc.wait(timeout=_SIGTERM_GRACE_S)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.kill()  # SIGKILL after the grace
    proc.wait()


def _plan():
    """Build the launch plan the way the supervisor would, using the smoke model."""
    # Import inside the test: on a base without p5a this is the red import.
    from app.runtime.backends.mlx import MlxBackend
    from app.runtime.backends.paths import ResolvedModelPaths

    model_path = os.environ["LMW_MLX_SMOKE_MODEL"]
    # The plan -- and therefore the --model value -- IS the local model path:
    # ResolvedModelPaths carries the cache bookkeeping (model_path,
    # snapshot_dir); the smoke is self-contained and points both at the env var.
    resolved = ResolvedModelPaths(model_path=model_path, snapshot_dir=model_path)
    model = _SmokeModel()
    port = _free_port()
    backend = MlxBackend()
    plan = backend.plan(model, port=port, bind_host="127.0.0.1", resolved=resolved)
    # The health URL comes from the backend, not a hand-written path: the smoke
    # checks the URL the supervisor would poll.
    return (
        plan,
        port,
        backend.health_url("127.0.0.1", port),
        backend.engine_model_id(model, resolved),
    )


def test_mlx_server_serves_one_completion(tmp_path):
    _skip_if_no_mlx()
    plan, port, health, model_id = _plan()
    # The proxy forwards engine_model_id; it must be the very --model value.
    assert model_id == plan.argv[plan.argv.index("--model") + 1]

    log = tmp_path / "engine.log"
    t0 = time.monotonic()
    proc = _launch(plan.argv, plan.env, log)
    try:
        assert _poll_until_healthy(health, _HEALTH_DEADLINE_S, proc), (
            f"mlx_lm.server never became healthy on {health} within "
            f"{_HEALTH_DEADLINE_S:.0f}s (exit={proc.poll()}):\n{_log_tail(log)}"
        )
        t_health = time.monotonic() - t0
        body = json.dumps(
            {
                "model": model_id,
                "messages": [{"role": "user", "content": "Say hello."}],
                "max_tokens": 4,
            }
        ).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            assert resp.status == 200
            payload = json.loads(resp.read().decode())
        content = payload["choices"][0]["message"]["content"]
        assert content, "the engine returned an empty completion"
        t_done = time.monotonic() - t0
        # Evidence for the run log (pytest -s / -rA): pid, port, timings, the text.
        print(
            f"mlx smoke: pid={proc.pid} port={port} health={t_health:.1f}s "
            f"completion={t_done - t_health:.1f}s content={content!r}"
        )
    finally:
        _terminate(proc)


def test_bad_port_argv_is_caught(tmp_path):
    _skip_if_no_mlx()
    plan, planned_port, health, _ = _plan()
    other = _free_port()
    while other == planned_port:
        other = _free_port()
    # Point the BINARY at a different port, but poll the PLANNED port (which
    # never has a server): a smoke that "succeeded" by talking to a different
    # endpoint would not catch this. argv entries are str -- Popen refuses an int.
    i = plan.argv.index("--port")
    assert plan.argv[i + 1] == str(planned_port)
    bad_argv = [*plan.argv[: i + 1], str(other), *plan.argv[i + 2 :]]
    log = tmp_path / "engine.log"
    proc = _launch(bad_argv, plan.env, log)
    try:
        assert not _poll_until_healthy(health, _BAD_PORT_DEADLINE_S, proc), (
            f"health reported 200 on the planned port {planned_port} although the "
            "engine was told another one -- the smoke is talking to something that "
            "is not the engine it launched"
        )
        # Teeth, not vacuity: the same engine DID come up -- on the port it was
        # given. A planned-port miss because the engine died proves nothing.
        other_health = health.replace(f":{planned_port}/", f":{other}/")
        assert _poll_until_healthy(other_health, _HEALTH_DEADLINE_S, proc), (
            f"the engine never served on its argv port {other} either "
            f"(exit={proc.poll()}):\n{_log_tail(log)}"
        )
    finally:
        _terminate(proc)
