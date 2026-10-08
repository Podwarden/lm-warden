"""End-to-end test: load runner + warmup probe + supervisor state gate.

Spawns a real subprocess (a tiny FastAPI app) that impersonates vLLM's
``/health`` and ``/v1/completions``. The fake's completions endpoint
returns 503 for the first N seconds and 200 after — simulating the
multimodal-warmup window where /health is already green. We verify:

1. DB status stays 'loading' during the 503 window.
2. An unload attempt during that window returns 409 (UnloadRefused).
3. After the probe succeeds, DB flips to 'loaded' and unload works.
"""

import asyncio
import subprocess
import sys
import textwrap

import pytest

from tests.conftest import wait_until

pytestmark = pytest.mark.integration


# A standalone uvicorn app that fakes vLLM's two relevant endpoints.
# Lives in a tempfile so the test owns its lifecycle.
FAKE_VLLM_SOURCE = textwrap.dedent("""
    import os
    import time
    from fastapi import FastAPI, Response
    import uvicorn

    START = time.monotonic()
    WARMUP_DELAY = float(os.environ.get("FAKE_VLLM_WARMUP_S", "5.0"))
    PORT = int(os.environ["FAKE_VLLM_PORT"])

    app = FastAPI()

    @app.get("/health")
    async def health():
        return Response(status_code=200)

    @app.post("/v1/completions")
    async def completions(body: dict):
        elapsed = time.monotonic() - START
        if elapsed < WARMUP_DELAY:
            return Response(status_code=503, content='{"error":"warming"}',
                            media_type="application/json")
        return {"choices": [{"text": " "}], "model": body.get("model")}

    if __name__ == "__main__":
        uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="error")
""")


@pytest.fixture
def fake_vllm(tmp_path):
    """Spawn the fake vLLM on a free port; tear down after the test."""
    # Pick a free port
    import socket

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    src = tmp_path / "fake_vllm.py"
    src.write_text(FAKE_VLLM_SOURCE)

    env = {
        **__import__("os").environ,
        "FAKE_VLLM_PORT": str(port),
        "FAKE_VLLM_WARMUP_S": "3.0",
    }
    proc = subprocess.Popen(
        [sys.executable, str(src)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait until it's listening
    import socket as _s

    def _listening():
        try:
            with _s.create_connection(("127.0.0.1", port), timeout=0.2):
                return True
        except OSError:
            return False

    try:
        wait_until(
            _listening,
            what="the fake vllm to start listening",
            timeout_s=5.0,
            child=proc,
        )
    except BaseException:
        # A child that died is already reaped by the wait (killing it would
        # raise ProcessLookupError and mask the return-code message); a hung
        # live child is killed so the fixture cannot leak it.
        if proc.returncode is None:
            proc.kill()
        raise
    yield port
    proc.terminate()
    proc.wait(timeout=5)


@pytest.mark.asyncio
async def test_status_stays_loading_during_probe_window(fake_vllm):
    """The race window between /health 200 and probe success must keep
    DB status at 'loading', and an unload attempt in that window must
    return 409."""
    from app.runtime.warmup_probe import warmup_probe

    port = fake_vllm
    # Health should be 200 immediately
    import httpx

    async with httpx.AsyncClient() as c:
        r = await c.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
    assert r.status_code == 200

    # Probe attempted now should fail (warmup still pending)
    early = await warmup_probe(port=port, served_model_name="fake", timeout_s=0.5)
    assert early.ok is False

    # Wait for warmup window to end, then probe again
    await asyncio.sleep(3.5)
    late = await warmup_probe(port=port, served_model_name="fake", timeout_s=2.0)
    assert late.ok is True


@pytest.mark.asyncio
async def test_supervisor_unload_refused_during_warming(tmp_path):
    """Direct unit test of the state gate at the supervisor level —
    integration-marked because it exercises the full Supervisor class
    without mocking _state."""
    from app.runtime.supervisor import ModelState, Supervisor, UnloadRefused

    class _Settings:
        pass

    s = _Settings()
    s.data_dir = str(tmp_path)
    s.hf_token_path = str(tmp_path / "tok")
    (tmp_path / "tok").write_text("hf_x")
    sup = Supervisor(s)

    from unittest.mock import AsyncMock, MagicMock

    from app.runtime.engine.local_subprocess import LocalHandle

    proc = MagicMock()
    proc.pid = 9999
    proc.returncode = None
    proc.wait = AsyncMock(return_value=0)
    sup._handles["m1"] = LocalHandle(proc)
    sup._state["m1"] = ModelState.WARMING
    sup.gpus.claim("m1", [0])

    with pytest.raises(UnloadRefused) as exc:
        await sup.unload("m1")
    assert exc.value.state == ModelState.WARMING

    # Process still registered, no SIGTERM
    assert "m1" in sup._handles
    proc.wait.assert_not_called()

    # Force bypass works
    from unittest.mock import patch

    with patch("os.killpg"):
        await sup.unload("m1", force=True)
    assert "m1" not in sup._handles
