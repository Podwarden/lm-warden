import asyncio
import site
import socket
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from app.runtime.supervisor import ModelState, Supervisor, wait_for_health


def _free_port() -> int:
    """A free loopback port (bind-0 handshake). The old hardcoded 18003/18004
    collided with whatever already held them on a busy runner."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class _Settings:
    pass


class _M:
    id = "fake"
    hf_repo = "fake/repo"
    hf_revision = None
    served_model_name = "fake-model"
    gpu_indices = [0]
    tensor_parallel_size = 1
    max_model_len = 4096
    dtype = "auto"
    gpu_memory_utilization = 0.5
    extra_env = {}


@pytest.mark.integration
async def test_supervisor_spawns_real_fake_process(tmp_path):
    """Real Supervisor.load() is called; only the vllm executable is intercepted.

    Asserts that build_subprocess_env ran (CUDA_VISIBLE_DEVICES present in the
    spawned subprocess /proc/<pid>/environ) — regression guard for the
    2026-05-08 production bug (Task 29).
    """
    s = _Settings()
    s.data_dir = str(tmp_path)
    s.hf_cache_dir = str(tmp_path / "hf-cache")
    s.hf_token_path = str(tmp_path / "hf-token")
    (tmp_path / "hf-token").write_bytes(b"fake-token")
    sup = Supervisor(s)

    # Repo root — needed so the child Python can import tests.fakes.fake_vllm.
    repo_root = str(Path(__file__).parent.parent.parent)
    port = _free_port()

    real_create = asyncio.create_subprocess_exec

    async def shim(*args, **kwargs):
        """Intercept 'vllm serve …' and redirect to fake_vllm shim."""
        cmd = list(args)
        if cmd and cmd[0] == "vllm":
            # Extract --port and --served-model-name from the original argv.
            orig_port = str(port)
            orig_served = "fake-model"
            for i, tok in enumerate(cmd):
                if tok == "--port" and i + 1 < len(cmd):
                    orig_port = cmd[i + 1]
                if tok == "--served-model-name" and i + 1 < len(cmd):
                    orig_served = cmd[i + 1]
            cmd = [
                sys.executable,
                "-m",
                "tests.fakes.fake_vllm",
                "--port",
                orig_port,
                "--served-model-name",
                orig_served,
            ]
            # Merge PYTHONPATH into the env dict that build_subprocess_env produced.
            # build_subprocess_env returns a closed env (no HOME, no PYTHONPATH), so
            # the child Python can't find aiohttp via user-site lookup. Inject the
            # current process's site-packages directly onto PYTHONPATH instead.
            env = dict(kwargs.get("env") or {})
            existing_pp = env.get("PYTHONPATH", "")
            site_paths = list(site.getsitepackages()) + [site.getusersitepackages()]
            env["PYTHONPATH"] = ":".join(filter(None, [repo_root, *site_paths, existing_pp]))
            kwargs["env"] = env
        return await real_create(*cmd, **kwargs)

    with patch("app.runtime.supervisor.asyncio.create_subprocess_exec", side_effect=shim):
        await sup.load(_M(), port=port)

    try:
        ok = await wait_for_health(port=port, timeout_s=10, interval_s=0.2)
        assert ok is True

        # Regression assertion: CUDA_VISIBLE_DEVICES must be present in the
        # spawned subprocess environment.  If build_subprocess_env is bypassed
        # (the original bug) this will be absent and the test fails.
        pid = sup._handles["fake"].pid
        with open(f"/proc/{pid}/environ", "rb") as _f:  # noqa: ASYNC230
            environ_bytes = _f.read()
        env_vars = {
            kv.split(b"=", 1)[0]: kv.split(b"=", 1)[1]
            for kv in environ_bytes.split(b"\x00")
            if b"=" in kv
        }
        assert env_vars.get(b"CUDA_VISIBLE_DEVICES") == b"0", (
            f"CUDA_VISIBLE_DEVICES missing or wrong in /proc/{pid}/environ; "
            f"build_subprocess_env may have been bypassed"
        )

        # Mirror the load route (app/models/routes_api.py): a plain unload is
        # refused until the model is READY, and only the route's
        # mark_warming/mark_ready get it there after health + warmup. The
        # 2026-05-23 quarantine (#143) existed because this test unloaded a
        # model still in LOADING — by construction, not a state leak.
        await sup.mark_warming(_M.id)
        await sup.mark_ready(_M.id)
        assert sup.get_state(_M.id) is ModelState.READY

        await sup.unload(_M.id)

    finally:
        # Always kill the child, even when an assertion above failed: a
        # leaked fake engine would hold the port and the GPU claim. force
        # covers the non-READY states a failed run can leave behind; after a
        # successful unload this is a no-op (no handle registered).
        try:
            await sup.unload(_M.id, force=True)
        except Exception:  # noqa: BLE001
            pass

    # After unload, supervisor internal state must be clean.
    assert _M.id not in sup._handles
    assert _M.id not in sup._watchers


@pytest.mark.integration
async def test_supervisor_extra_env_reaches_subprocess(tmp_path):
    """Regression guard: extra_env={"VLLM_USE_V1": "1"} must appear in the
    spawned subprocess /proc/<pid>/environ.  Verifies the env_builder
    allowlist merge path end-to-end.
    """

    class _MWithEnv(_M):
        extra_env = {"VLLM_USE_V1": "1"}

    s = _Settings()
    s.data_dir = str(tmp_path)
    s.hf_cache_dir = str(tmp_path / "hf-cache")
    s.hf_token_path = str(tmp_path / "hf-token")
    (tmp_path / "hf-token").write_bytes(b"fake-token")
    sup = Supervisor(s)

    repo_root = str(Path(__file__).parent.parent.parent)
    port = _free_port()

    real_create = asyncio.create_subprocess_exec

    async def shim(*args, **kwargs):
        cmd = list(args)
        if cmd and cmd[0] == "vllm":
            orig_port = str(port)
            orig_served = "fake-model"
            for i, tok in enumerate(cmd):
                if tok == "--port" and i + 1 < len(cmd):
                    orig_port = cmd[i + 1]
                if tok == "--served-model-name" and i + 1 < len(cmd):
                    orig_served = cmd[i + 1]
            cmd = [
                sys.executable,
                "-m",
                "tests.fakes.fake_vllm",
                "--port",
                orig_port,
                "--served-model-name",
                orig_served,
            ]
            env = dict(kwargs.get("env") or {})
            existing_pp = env.get("PYTHONPATH", "")
            site_paths = list(site.getsitepackages()) + [site.getusersitepackages()]
            env["PYTHONPATH"] = ":".join(filter(None, [repo_root, *site_paths, existing_pp]))
            kwargs["env"] = env
        return await real_create(*cmd, **kwargs)

    model = _MWithEnv()
    with patch("app.runtime.supervisor.asyncio.create_subprocess_exec", side_effect=shim):
        await sup.load(model, port=port)

    try:
        ok = await wait_for_health(port=port, timeout_s=10, interval_s=0.2)
        assert ok is True

        pid = sup._handles[model.id].pid
        with open(f"/proc/{pid}/environ", "rb") as _f:  # noqa: ASYNC230
            environ_bytes = _f.read()
        env_vars = {
            kv.split(b"=", 1)[0]: kv.split(b"=", 1)[1]
            for kv in environ_bytes.split(b"\x00")
            if b"=" in kv
        }
        assert (
            env_vars.get(b"CUDA_VISIBLE_DEVICES") == b"0"
        ), "CUDA_VISIBLE_DEVICES missing or wrong"
        assert (
            b"VLLM_USE_V1" in env_vars
        ), "VLLM_USE_V1 from extra_env not found in subprocess environ"
        assert env_vars[b"VLLM_USE_V1"] == b"1"

        # Mirror the load route: LOADING is not unloadable without force
        # (see test_supervisor_spawns_real_fake_process above).
        await sup.mark_warming(model.id)
        await sup.mark_ready(model.id)
        assert sup.get_state(model.id) is ModelState.READY

        await sup.unload(model.id)

    finally:
        # Always kill the child, even when an assertion above failed.
        try:
            await sup.unload(model.id, force=True)
        except Exception:  # noqa: BLE001
            pass

    assert model.id not in sup._handles
    assert model.id not in sup._watchers
