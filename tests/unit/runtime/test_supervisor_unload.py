import asyncio
import signal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.runtime.engine.local_subprocess import LocalHandle
from app.runtime.supervisor import Supervisor


class _Settings:
    pass


@pytest.mark.asyncio
async def test_unload_sigterms_then_releases_gpus(tmp_path):
    s = _Settings()
    s.data_dir = str(tmp_path)
    s.hf_token_path = str(tmp_path / "tok")
    (tmp_path / "tok").write_text("hf_x")
    sup = Supervisor(s)
    proc = MagicMock()
    proc.pid = 4242
    proc.returncode = None

    async def fake_wait():
        proc.returncode = 0
        return 0

    proc.wait = AsyncMock(side_effect=fake_wait)

    sup._handles["m1"] = LocalHandle(proc)
    sup._ports["m1"] = 10001
    sup.gpus.claim("m1", [0, 1])

    with patch("os.killpg") as kp:
        await sup.unload("m1")
    kp.assert_called_with(4242, signal.SIGTERM)
    assert "m1" not in sup._handles
    assert sup.gpus.owner_of(0) is None


@pytest.mark.asyncio
async def test_unload_releases_gpus_even_when_terminate_raises(tmp_path):
    """A teardown that raises must STILL release the in-memory GPU claim.

    Regression for the #166-adjacent leak observed in production: when the unload
    request was cancelled mid-``terminate()`` (client/proxy disconnect) — or
    when the driver's terminate raises any other exception — the old code
    skipped the trailing ``self.gpus.release()`` and the GPUs stayed
    permanently ``already claimed`` by a model that is gone, so no future
    load could use them without a control-plane restart.
    """
    s = _Settings()
    s.data_dir = str(tmp_path)
    s.hf_token_path = str(tmp_path / "tok")
    (tmp_path / "tok").write_text("hf_x")

    driver = MagicMock()
    driver.terminate = AsyncMock(side_effect=RuntimeError("docker daemon hiccup"))
    sup = Supervisor(s, driver=driver)

    handle = MagicMock()
    handle.returncode = None
    sup._handles["m1"] = handle
    sup._ports["m1"] = 10001
    sup._state["m1"] = sup._state.get("m1")  # leave unset → not READY
    sup.gpus.claim("m1", [1, 2])

    with pytest.raises(RuntimeError):
        await sup.unload("m1", force=True)

    # The claim is gone and the lifecycle bookkeeping is cleared, despite the
    # teardown exception propagating to the caller.
    assert sup.gpus.owner_of(1) is None
    assert sup.gpus.owner_of(2) is None
    assert "m1" not in sup._handles
    assert "m1" not in sup._ports
    assert "m1" not in sup._state


@pytest.mark.asyncio
async def test_unload_refusal_does_not_release_gpus(tmp_path):
    """A refused unload (transient state, no force) must NOT release GPUs —
    the model is still live; only an accepted teardown frees the claim."""
    s = _Settings()
    s.data_dir = str(tmp_path)
    s.hf_token_path = str(tmp_path / "tok")
    (tmp_path / "tok").write_text("hf_x")

    from app.runtime.supervisor import ModelState, UnloadRefused

    sup = Supervisor(s)
    handle = MagicMock()
    handle.returncode = None
    sup._handles["m1"] = handle
    sup._state["m1"] = ModelState.LOADING
    sup.gpus.claim("m1", [1])

    with pytest.raises(UnloadRefused):
        await sup.unload("m1")  # transient state, no force → refused

    assert sup.gpus.owner_of(1) == "m1"
    assert "m1" in sup._handles


@pytest.mark.asyncio
async def test_unload_sigkill_after_timeout(tmp_path):
    s = _Settings()
    s.data_dir = str(tmp_path)
    s.hf_token_path = str(tmp_path / "tok")
    (tmp_path / "tok").write_text("hf_x")
    sup = Supervisor(s)
    proc = MagicMock()
    proc.pid = 4242
    proc.returncode = None

    async def hang():
        await asyncio.sleep(60)

    proc.wait = AsyncMock(side_effect=hang)

    sup._handles["m1"] = LocalHandle(proc)
    sup.gpus.claim("m1", [0])

    sent = []

    def fake_killpg(pid, sig):
        sent.append(sig)
        if sig == signal.SIGKILL:
            proc.returncode = -9
            proc.wait = AsyncMock(return_value=-9)

    with patch("os.killpg", side_effect=fake_killpg):
        with patch("app.runtime.supervisor.UNLOAD_GRACE_SECONDS", 0.2):
            await sup.unload("m1")
    assert signal.SIGTERM in sent
    assert signal.SIGKILL in sent
    assert sup.gpus.owner_of(0) is None


@pytest.mark.asyncio
async def test_unload_cancels_the_registered_load_runner_task(tmp_path):
    """#275 root cause — a force-unload of a model stuck in LOADING must
    cancel the load runner (routes_api.start_engine's wait_for_health +
    warmup task) directly, not just the engine process. Left running, that
    task keeps awaiting health on an engine this unload just killed and
    eventually times out on its OWN schedule, writing a terminal status onto
    whatever row now owns model_id -- possibly a newer load that has since
    superseded this one, while its replacement engine keeps running
    unaffected (podwarden/apps/lm-warden#275).
    """
    s = _Settings()
    s.data_dir = str(tmp_path)
    s.hf_token_path = str(tmp_path / "tok")
    (tmp_path / "tok").write_text("hf_x")
    sup = Supervisor(s)
    proc = MagicMock()
    proc.pid = 4242
    proc.returncode = None

    async def fake_wait():
        proc.returncode = 0
        return 0

    proc.wait = AsyncMock(side_effect=fake_wait)

    sup._handles["m1"] = LocalHandle(proc)
    sup._ports["m1"] = 10001
    sup.gpus.claim("m1", [0, 1])

    runner_started = asyncio.Event()

    async def fake_runner():
        runner_started.set()
        await asyncio.sleep(60)

    runner_task = asyncio.create_task(fake_runner())
    sup.register_load_runner("m1", runner_task)
    await runner_started.wait()

    with patch("os.killpg"):
        await sup.unload("m1", force=True)

    assert runner_task.cancelled(), "force-unload must cancel the in-flight load runner"


@pytest.mark.asyncio
async def test_unload_does_not_touch_a_finished_load_runner(tmp_path):
    """A load runner that already completed (its task is done) must not be
    touched -- only a still-running runner needs cancelling."""
    s = _Settings()
    s.data_dir = str(tmp_path)
    s.hf_token_path = str(tmp_path / "tok")
    (tmp_path / "tok").write_text("hf_x")
    sup = Supervisor(s)
    proc = MagicMock()
    proc.pid = 4242
    proc.returncode = None

    async def fake_wait():
        proc.returncode = 0
        return 0

    proc.wait = AsyncMock(side_effect=fake_wait)

    sup._handles["m1"] = LocalHandle(proc)
    sup._ports["m1"] = 10001
    sup.gpus.claim("m1", [0])

    async def finished_runner():
        return None

    runner_task = asyncio.create_task(finished_runner())
    sup.register_load_runner("m1", runner_task)
    await runner_task  # let it actually finish before unload runs

    with patch("os.killpg"):
        await sup.unload("m1", force=True)

    assert runner_task.cancelled() is False
    assert runner_task.exception() is None


@pytest.mark.asyncio
async def test_ensure_unloadable_raises_for_transient_state(tmp_path):
    """#166 — the fast pre-flight check refuses a model in a transient state
    (no force) so the route can return 409 synchronously, without touching the
    engine."""
    from app.runtime.supervisor import ModelState, UnloadRefused

    s = _Settings()
    s.data_dir = str(tmp_path)
    s.hf_token_path = str(tmp_path / "tok")
    (tmp_path / "tok").write_text("hf_x")
    sup = Supervisor(s)
    sup._state["m1"] = ModelState.WARMING

    with pytest.raises(UnloadRefused):
        await sup.ensure_unloadable("m1", force=False)
    # force overrides the refusal
    await sup.ensure_unloadable("m1", force=True)
    # READY state is always unloadable
    sup._state["m1"] = ModelState.READY
    await sup.ensure_unloadable("m1", force=False)
