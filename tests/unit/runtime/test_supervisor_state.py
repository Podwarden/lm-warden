import signal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.runtime.backends import LaunchPlan
from app.runtime.engine.local_subprocess import LocalHandle
from app.runtime.supervisor import ModelState, Supervisor, UnloadRefused


class _Settings:
    pass


def _make_sup(tmp_path):
    s = _Settings()
    s.data_dir = str(tmp_path)
    s.hf_token_path = str(tmp_path / "tok")
    s.hf_cache_dir = str(tmp_path / "hf-cache")
    (tmp_path / "tok").write_text("hf_x")
    return Supervisor(s)


def _register_proc(sup, model_id="m1", state=ModelState.LOADING):
    proc = MagicMock()
    proc.pid = 4242
    proc.returncode = None

    async def fake_wait():
        proc.returncode = 0
        return 0

    proc.wait = AsyncMock(side_effect=fake_wait)

    sup._handles[model_id] = LocalHandle(proc)
    sup._ports[model_id] = 10001
    sup._state[model_id] = state
    sup.gpus.claim(model_id, [0])
    return proc


@pytest.mark.asyncio
async def test_load_sets_state_loading(tmp_path):
    sup = _make_sup(tmp_path)
    proc = MagicMock()
    proc.pid = 1
    proc.returncode = None
    with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)):
        model = MagicMock()
        model.id = "m1"
        model.gpu_indices = [0]
        model.tensor_parallel_size = 1
        # No engine-axis pin — a bare MagicMock would otherwise auto-return a
        # truthy Mock for engine_image, tripping the #177 EnginePinUnsupported
        # guard under the default subprocess driver. Real model rows carry NULL.
        model.engine_image = None
        model.engine_channel = None
        model.engine_vllm_version = None
        # NULL backend, i.e. every pre-0027 row: registry.get(None) decodes it
        # to vLLM (D6). A bare MagicMock would auto-return a truthy Mock here
        # and UnknownBackendError would abort the load.
        model.backend = None
        # Sub-project C: the supervisor resolves the row's pinned files before
        # calling plan(). A bare MagicMock auto-returns a truthy Mock for
        # mmproj_filename, and a set-but-unresolvable projector is a hard error
        # (a vision model launched without one loads, serves, and silently
        # ignores every image). Real vLLM rows carry NULL for both.
        model.mmproj_filename = None
        model.filename = None
        # Sub-project B: the supervisor no longer calls build_vllm_args /
        # build_subprocess_env itself -- it asks the backend for a LaunchPlan.
        # This test is about the state machine, not about launch construction,
        # so stub the plan out at the same seam the two builders used to sit.
        stub_plan = LaunchPlan(
            argv=["vllm", "serve"], env={}, image=None, port=10001, gpu_indices=[0]
        )
        with patch("app.runtime.backends.vllm.VllmBackend.plan", return_value=stub_plan):
            await sup.load(model, port=10001)
    assert sup.get_state("m1") == ModelState.LOADING


@pytest.mark.asyncio
async def test_mark_warming_then_ready_transitions(tmp_path):
    sup = _make_sup(tmp_path)
    _register_proc(sup, state=ModelState.LOADING)
    await sup.mark_warming("m1")
    assert sup.get_state("m1") == ModelState.WARMING
    await sup.mark_ready("m1")
    assert sup.get_state("m1") == ModelState.READY


@pytest.mark.asyncio
async def test_unload_refused_from_loading(tmp_path):
    sup = _make_sup(tmp_path)
    _register_proc(sup, state=ModelState.LOADING)
    with pytest.raises(UnloadRefused) as exc:
        await sup.unload("m1")
    assert "LOADING" in str(exc.value)
    # Process must still be registered — no SIGTERM sent
    assert "m1" in sup._handles


@pytest.mark.asyncio
async def test_unload_refused_from_warming(tmp_path):
    sup = _make_sup(tmp_path)
    _register_proc(sup, state=ModelState.WARMING)
    with pytest.raises(UnloadRefused):
        await sup.unload("m1")
    assert "m1" in sup._handles


@pytest.mark.asyncio
async def test_unload_force_bypasses_state_gate_from_loading(tmp_path):
    sup = _make_sup(tmp_path)
    _register_proc(sup, state=ModelState.LOADING)
    with patch("os.killpg") as kp:
        await sup.unload("m1", force=True)
    kp.assert_called_with(4242, signal.SIGTERM)
    assert "m1" not in sup._handles


@pytest.mark.asyncio
async def test_unload_from_ready_works_without_force(tmp_path):
    sup = _make_sup(tmp_path)
    _register_proc(sup, state=ModelState.READY)
    with patch("os.killpg") as kp:
        await sup.unload("m1")
    kp.assert_called_with(4242, signal.SIGTERM)
    assert "m1" not in sup._handles


@pytest.mark.asyncio
async def test_watch_exit_clears_state(tmp_path):
    sup = _make_sup(tmp_path)
    _register_proc(sup, state=ModelState.WARMING)
    # Trigger natural exit
    await sup._watch_exit("m1", None)
    assert sup.get_state("m1") is None
    assert "m1" not in sup._handles


@pytest.mark.asyncio
async def test_get_state_returns_none_for_unknown(tmp_path):
    sup = _make_sup(tmp_path)
    assert sup.get_state("does-not-exist") is None


@pytest.mark.asyncio
async def test_mark_warming_raises_if_not_loading(tmp_path):
    sup = _make_sup(tmp_path)
    _register_proc(sup, state=ModelState.READY)
    with pytest.raises(RuntimeError):
        await sup.mark_warming("m1")
