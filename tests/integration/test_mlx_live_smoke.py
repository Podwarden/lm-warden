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

By default the engine runs from the warden's OWN venv (dev mode). To exercise
the BUNDLED interpreter instead -- so a trimmed bundle is what the smoke proves
-- point the smoke at the bundle's renamed interpreter:

    export LMW_MLX_SMOKE_PYTHON="/path/LM Warden.app/Contents/Resources/python/bin/LM Warden Server"

In that mode the plan's hard-set PATH is recomputed as
``engine_path(<that interpreter>)`` -- the same value the app computes -- so the
bare ``mlx_lm.server`` argv[0] resolves to the bundle's own console script, and
the ``import mlx_lm`` presence skip is not applied (the bundle carries mlx_lm;
the test venv may not). A value that is not an executable file is a REFUSAL,
not a skip: an empty value is treated as unset (dev mode), but a set value that
is missing or not executable is a pytest.fail naming the variable -- a
mis-pointed bundle must never be silently read as "skip".

A run that SKIPS is not green: mlx_lm missing, not darwin/arm64, or the env
var unset all skip with a reason, and the lane reports them as blocked.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import pytest
from _pytest.outcomes import Failed

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


def _skip_platform() -> None:
    """The whole file is the macos+live gate; refuse on any other machine.

    Shared by the live legs and the non-live plan legs so the platform
    contract is pinned in ONE place, and so the refusal-arm legs can assume
    they are on the machine the smoke targets.
    """
    if sys.platform != "darwin" or platform.machine() != "arm64":
        pytest.skip(
            f"MLX live smoke needs darwin/arm64; this is " f"{sys.platform}/{platform.machine()}"
        )


def _smoke_python() -> str | None:
    """The bundled interpreter the smoke should serve from, or None for dev.

    Dev mode (the variable unset or EMPTY) keeps today's behaviour exactly:
    the plan's own venv PATH, the ``import mlx_lm`` presence check, same skips.
    A non-empty value names a BUNDLED interpreter -- ``<app>/Contents/
    Resources/python/bin/LM Warden Server`` -- and must be an executable file:
    a value that is missing, not a file, or not executable is a REFUSAL
    (pytest.fail naming the variable and the reason), never a skip, so a
    mis-pointed bundle is caught instead of silently read as "skip".
    """
    override = os.environ.get("LMW_MLX_SMOKE_PYTHON", "").strip()
    if not override:
        return None
    if not os.path.isfile(override):
        pytest.fail(
            f"LMW_MLX_SMOKE_PYTHON={override!r} does not name an executable file "
            "(no such file); point it at a bundle's "
            "Contents/Resources/python/bin/LM Warden Server"
        )
    if not os.access(override, os.X_OK):
        pytest.fail(
            f"LMW_MLX_SMOKE_PYTHON={override!r} is not executable; point it at a "
            "bundle's Contents/Resources/python/bin/LM Warden Server"
        )
    return override


def _skip_if_no_mlx(bundle: bool = False) -> None:
    """Skip with a clear reason when the prerequisites are absent.

    In bundled mode (``bundle=True``) the ``import mlx_lm`` presence check is
    skipped: the engine is the BUNDLE's interpreter, which carries mlx_lm, and
    the test venv need not import it. In dev mode it is the whole check -- and
    it is imported inside this function, not at module level, so that a base
    without it skips (collects and skips) rather than turning this whole file
    into a collection ImportError that runs nothing.
    """
    _skip_platform()
    if not bundle:
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


def _plan_bundled(interpreter: str):
    """plan() re-pointed at the BUNDLED interpreter's PATH.

    The plan's argv stays EXACTLY what the supervisor would emit (bare
    ``mlx_lm.server``); only the hard-set PATH is replaced with the one the app
    computes for the bundle's interpreter. The smoke then asserts the bare name
    resolves, through THAT PATH, to the bundle's own console script -- the whole
    point of serving from the bundle.
    """
    from app.runtime.backends.mlx import engine_path

    plan, port, health, model_id = _plan()
    # LaunchPlan is a frozen dataclass: build a copy, never assign to it.
    bundled = dataclasses.replace(plan, env={**plan.env, "PATH": engine_path(interpreter)})
    return bundled, port, health, model_id


def test_mlx_server_serves_one_completion(tmp_path):
    override = _smoke_python()
    bundle = override is not None
    _skip_if_no_mlx(bundle=bundle)
    if bundle:
        plan, port, health, model_id = _plan_bundled(override)
    else:
        plan, port, health, model_id = _plan()
    # The proxy forwards engine_model_id; it must be the very --model value.
    assert model_id == plan.argv[plan.argv.index("--model") + 1]
    if bundle:
        # The engine runs from the BUNDLE: the bare argv[0] must resolve, through
        # the re-pointed PATH, to a script INSIDE the bundle's bin -- never a
        # copy the smoke host has on its own PATH.
        resolved = shutil.which("mlx_lm.server", path=plan.env["PATH"])
        assert resolved, f"mlx_lm.server not found on PATH={plan.env['PATH']!r}"
        assert os.path.dirname(os.path.realpath(resolved)) == os.path.dirname(
            os.path.realpath(override)
        ), f"mlx_lm.server resolves to {resolved!r}, not the bundle's own copy"
        assert plan.argv[0] == "mlx_lm.server"  # argv stays the bare console-script name

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
    override = _smoke_python()
    bundle = override is not None
    _skip_if_no_mlx(bundle=bundle)
    if bundle:
        plan, planned_port, health, _ = _plan_bundled(override)
    else:
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


# ---------------------------------------------------------------------------
# Non-live legs: LMW_MLX_SMOKE_PYTHON. No model load, no engine process, no
# live-model lock needed -- so they collect and run in this macos+live file
# under -k smoke_python, and they pin the three refusal arms of the override
# plus the PATH the bundled engine actually resolves through.
# ---------------------------------------------------------------------------


def test_smoke_python_missing_refuses(tmp_path, monkeypatch):
    # A value that names no file is a REFUSAL (fail), never a skip: on the base
    # the variable does not exist, so a missing path is silently ignored and
    # this leg passes -- which is the red-first signal that the override is
    # being read at all.
    _skip_platform()
    monkeypatch.setenv("LMW_MLX_SMOKE_PYTHON", str(tmp_path / "LM Warden Server"))
    with pytest.raises(Failed) as excinfo:  # pytest.fail raises _pytest.outcomes.Failed
        _smoke_python()
    assert "LMW_MLX_SMOKE_PYTHON" in str(excinfo.value)


def test_smoke_python_not_executable_refuses(tmp_path, monkeypatch):
    _skip_platform()
    script = tmp_path / "LM Warden Server"
    script.write_text("# a file, but not executable\n")
    monkeypatch.setenv("LMW_MLX_SMOKE_PYTHON", str(script))
    with pytest.raises(Failed) as excinfo:
        _smoke_python()
    assert "LMW_MLX_SMOKE_PYTHON" in str(excinfo.value)
    assert "not executable" in str(excinfo.value)


def test_smoke_python_empty_is_unset(tmp_path, monkeypatch):
    # An EMPTY value is the arm that means "unset": dev mode, no refusal.
    _skip_platform()
    for empty in ("", "   "):
        monkeypatch.setenv("LMW_MLX_SMOKE_PYTHON", empty)
        assert _smoke_python() is None


def test_smoke_python_resolution(tmp_path, monkeypatch):
    # The positive leg: a real executable makes the bare console-script name
    # resolve, through the re-pointed PATH, to a copy INSIDE that executable's
    # own directory -- and argv stays the bare name. No model, no lock.
    _skip_platform()
    monkeypatch.delenv("LMW_MLX_SMOKE_MODEL", raising=False)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    interpreter = bin_dir / "LM Warden Server"
    # A minimal "interpreter": an executable shell script (os.access X_OK).
    interpreter.write_text("#!/bin/sh\nexit 0\n")
    interpreter.chmod(0o755)
    monkeypatch.setenv("LMW_MLX_SMOKE_PYTHON", str(interpreter))
    override = _smoke_python()
    assert override == str(interpreter)

    from app.runtime.backends.mlx import engine_path  # in-test: red import on base

    # engine_path puts the interpreter's own dir first, then the fixed fallbacks.
    path = engine_path(str(interpreter))
    assert path.split(":")[0] == str(bin_dir)
    assert path.endswith("/usr/bin:/bin")
    # The bare name resolves through that PATH to a script in the SAME dir.
    server = bin_dir / "mlx_lm.server"
    server.write_text("#!/bin/sh\nexit 0\n")
    server.chmod(0o755)
    found = shutil.which("mlx_lm.server", path=path)
    assert found is not None, f"mlx_lm.server not resolvable on PATH={path!r}"
    assert os.path.dirname(os.path.realpath(found)) == os.path.dirname(
        os.path.realpath(str(interpreter))
    )
    # argv[0] stays the bare console-script name; only the PATH carries it.
    assert "mlx_lm.server" in os.listdir(bin_dir)


def test_smoke_python_plan_bundled_repoints_path(tmp_path, monkeypatch):
    # The live legs' bundled plan, built without a model load: _plan_bundled
    # must hand back a LaunchPlan (not a nested tuple) whose PATH is
    # engine_path(<override>) and whose argv, port and other env are exactly
    # plan()'s. plan() is pure, so a tmp dir stands in for the weights.
    _skip_platform()
    model_dir = tmp_path / "weights"
    model_dir.mkdir()
    monkeypatch.setenv("LMW_MLX_SMOKE_MODEL", str(model_dir))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    interpreter = bin_dir / "LM Warden Server"
    interpreter.write_text("#!/bin/sh\nexit 0\n")
    interpreter.chmod(0o755)

    from app.runtime.backends import LaunchPlan
    from app.runtime.backends.mlx import engine_path

    dev, _, _, _ = _plan()
    plan, port, health, model_id = _plan_bundled(str(interpreter))
    assert isinstance(plan, LaunchPlan)
    assert plan.env["PATH"] == engine_path(str(interpreter))
    assert plan.env["PATH"].split(":")[0] == str(bin_dir)
    assert {k: v for k, v in plan.env.items() if k != "PATH"} == {
        k: v for k, v in dev.env.items() if k != "PATH"
    }
    assert plan.argv[0] == "mlx_lm.server"
    assert plan.port == port and str(port) in plan.argv
    assert str(port) in health
    assert model_id == plan.argv[plan.argv.index("--model") + 1] == str(model_dir)
