"""Does the BUNDLED Metal llama-server serve on a real Mac? Not in CI.

The macOS counterpart of ``test_llamacpp_smoke.py`` (the Linux image check),
shaped like ``test_mlx_live_smoke.py``: it drives the launch
``LlamaCppBackend(profile="apple").plan()`` would -- the bare ``llama-server``
argv[0] resolved through the apple engine PATH, with the plan's own flags --
so an argv the real binary rejects, or a binary the PATH cannot find, fails
here rather than at first load.

The plan's PATH is ``dirname(sys.executable):/usr/bin:/bin``. Under pytest
``sys.executable`` is the tree's venv, not the app's interpreter, so the smoke
rebuilds that PATH for the BUNDLED interpreter
(``engine_path(<app>/Contents/Resources/python/bin/LM Warden Server)``) -- the
exact value the server computes inside the app -- and points nothing else at
the binary. Two inputs:

    export LMW_LLAMACPP_SMOKE_BIN_DIR="/path/LM Warden.app/Contents/Resources/python/bin"
    export LMW_LLAMACPP_SMOKE_GGUF=~/models/gguf/qwen2.5-0.5b-instruct-q4_0.gguf

Run it under the live-model lock (scripts/farm/ITERATION.md). A run that SKIPS
is not green: not darwin/arm64 or either env var unset/wrong all skip with a
reason.
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

_HEALTH_DEADLINE_S = 180.0
_SIGTERM_GRACE_S = 15.0
_SERVER_EXE = "LM Warden Server"


@dataclass
class _SmokeModel:
    """A minimal llama.cpp model row: plan() reads only these fields."""

    id: str = "llamacpp-metal-smoke"
    served_model_name: str = "llamacpp-metal-smoke"
    gpu_indices: list[int] = field(default_factory=lambda: [0])
    max_model_len: int | None = 2048
    max_num_seqs: int | None = 2
    n_gpu_layers: int | None = None
    extra_args: list[str] = field(default_factory=list)
    extra_env: dict[str, str] = field(default_factory=dict)


def _inputs() -> tuple[str, str]:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        pytest.skip(
            f"Metal llama.cpp smoke needs darwin/arm64; this is {sys.platform}/{platform.machine()}"
        )
    bin_dir = os.environ.get("LMW_LLAMACPP_SMOKE_BIN_DIR", "")
    gguf = os.environ.get("LMW_LLAMACPP_SMOKE_GGUF", "")
    if not bin_dir or not os.access(os.path.join(bin_dir, "llama-server"), os.X_OK):
        pytest.skip(
            f"LMW_LLAMACPP_SMOKE_BIN_DIR={bin_dir!r} holds no executable llama-server; "
            "point it at a bundle's Contents/Resources/python/bin"
        )
    if not os.path.isfile(gguf):
        pytest.skip(f"LMW_LLAMACPP_SMOKE_GGUF={gguf!r} is not a GGUF file")
    return bin_dir, gguf


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _status(url: str) -> int:
    try:
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, OSError):
        return 0


def _log_tail(path, n: int = 3000) -> str:
    try:
        with open(path, "rb") as f:
            return f.read()[-n:].decode(errors="replace")
    except OSError:
        return "<no engine log>"


def _terminate(proc: subprocess.Popen) -> None:
    # Only the PID this test started: no pattern kills, no process-group kills.
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=_SIGTERM_GRACE_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def test_bundled_llama_server_serves_one_completion_on_metal(tmp_path):
    bin_dir, gguf = _inputs()
    from app.runtime.backends import registry
    from app.runtime.backends.mlx import ENGINE_PATH, engine_path
    from app.runtime.backends.paths import ResolvedModelPaths

    backend = registry.backends_for("apple")["llamacpp"]
    model = _SmokeModel()
    port = _free_port()
    resolved = ResolvedModelPaths(model_path=gguf, snapshot_dir=os.path.dirname(gguf))
    plan = backend.plan(model, port=port, bind_host="127.0.0.1", resolved=resolved)
    assert plan.argv[0] == "llama-server"
    assert plan.env["PATH"] == ENGINE_PATH  # what the server emits on apple
    assert not any(k.startswith("CUDA_") for k in plan.env)
    # The same PATH, computed for the bundled interpreter instead of pytest's.
    env = dict(plan.env, PATH=engine_path(os.path.join(bin_dir, _SERVER_EXE)))
    assert env["PATH"].split(":")[0] == bin_dir
    # The binary that PATH finds is a Metal build and sees the GPU (its default
    # log is too terse to say so; --list-devices is the engine's own answer).
    # The plan never pins it to the CPU: no --device, no zero offload. (One
    # GPU is `--split-mode none --main-gpu 0`, and MTL0 is device 0.)
    devices = subprocess.run(
        [plan.argv[0], "--list-devices"], env=env, capture_output=True, text=True, cwd="/"
    )
    assert "MTL0" in devices.stdout + devices.stderr, devices.stdout + devices.stderr
    assert "--device" not in plan.argv
    for flag in ("-ngl", "--n-gpu-layers"):
        if flag in plan.argv:
            assert plan.argv[plan.argv.index(flag) + 1] != "0", plan.argv

    log = tmp_path / "engine.log"
    t0 = time.monotonic()
    with open(log, "wb") as fh:
        # cwd="/": the binary must need nothing next to it (static, Metal
        # kernels embedded) whatever directory the supervisor runs from.
        proc = subprocess.Popen(plan.argv, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd="/")
    try:
        health = backend.health_url("127.0.0.1", port)
        while time.monotonic() - t0 < _HEALTH_DEADLINE_S:
            if _status(health) == 200 or proc.poll() is not None:
                break
            time.sleep(0.5)
        healthy = _status(health) == 200
        why = f"never healthy on {health} (exit={proc.poll()}):\n{_log_tail(log)}"
        assert healthy, why
        t_health = time.monotonic() - t0
        body = json.dumps(
            {
                "model": backend.engine_model_id(model, resolved),
                "messages": [{"role": "user", "content": "Say hello."}],
                "max_tokens": 8,
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
        # /metrics is on (args.py emits --metrics unconditionally).
        assert _status(f"http://127.0.0.1:{port}/metrics") == 200
        print(
            f"llamacpp metal smoke: pid={proc.pid} port={port} health={t_health:.1f}s "
            f"completion={time.monotonic() - t0 - t_health:.1f}s content={content!r}"
        )
    finally:
        _terminate(proc)
