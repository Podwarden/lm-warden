import pytest

from app.runtime.backends.vllm.env import build_subprocess_env


@pytest.fixture
def model():
    class M:
        id = "qwen3.5-9b"
        gpu_indices = [1, 2]
        tensor_parallel_size = 2

    return M()


def test_cuda_visible_devices_from_model_row_not_env(model, monkeypatch):
    """Regression: 2026-05-08 bug. CUDA_VISIBLE_DEVICES must come from model.gpu_indices,
    NEVER inherited from parent env."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    env = build_subprocess_env(model, hf_token="hf_xxx", hf_cache_dir="/hfcache")
    assert env["CUDA_VISIBLE_DEVICES"] == "1,2"


def test_cuda_visible_devices_serializes_in_gpu_indices_order():
    """Order in the env var matches gpu_indices order. vLLM treats CUDA_VISIBLE_DEVICES
    position as the logical device id."""

    class M:
        id = "m"
        gpu_indices = [3, 0, 2]
        tensor_parallel_size = 3

    env = build_subprocess_env(M(), hf_token="hf_xxx", hf_cache_dir="/hfcache")
    assert env["CUDA_VISIBLE_DEVICES"] == "3,0,2"


def test_env_builder_points_hf_hub_cache_at_cache_dir_not_data(model):
    """Regression: 2026-06-15 ENOSPC crash. The vLLM engine subprocess must
    read/write its model cache at the SAME directory the pull task wrote to
    (``settings.hf_cache_dir``), via ``HF_HUB_CACHE`` — whose layout
    (``<root>/models--org--name/``) matches ``snapshot_download(cache_dir=...)``.

    The old builder hard-locked ``HF_HOME=<data_dir>/hf-cache``, pointing the
    engine at the tiny ``/data`` PVC instead of the large model-cache volume.
    The 57 GB model then re-downloaded onto ``/data``, filling it to 0 bytes
    (ENOSPC) and taking SQLite down with it. ``HF_HOME`` must no longer be set
    here so it can't override ``HF_HUB_CACHE``."""
    env = build_subprocess_env(model, hf_token="hf_secret", hf_cache_dir="/hfcache")
    assert env["HF_HUB_CACHE"] == "/hfcache"
    assert "HF_HOME" not in env
    assert env["HUGGING_FACE_HUB_TOKEN"] == "hf_secret"


def test_env_builder_sets_cuda_device_order_pci_bus_id(model):
    """Heterogeneous-GPU regression: NVML may reorder by compute capability
    unless CUDA_DEVICE_ORDER=PCI_BUS_ID is set, which breaks gpu_indices
    semantics on hosts mixing Quadro/A4000-class cards."""
    env = build_subprocess_env(model, hf_token="hf_xxx", hf_cache_dir="/hfcache")
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_env_builder_sets_pythonunbuffered(model):
    """v2026.05.15.5 regression: subprocess stdout must be line-buffered.

    The supervisor opens the log file with O_WRONLY|O_CREAT|O_APPEND and
    passes the fd as stdout=. With stdout pointing at a file (not a
    tty), CPython defaults to block buffering — so a fast-crashing
    subprocess (rc=1 within <1s) can exit before its print/traceback
    output is flushed, leaving an empty log file. PYTHONUNBUFFERED=1
    forces line buffering and flush-on-exit, guaranteeing the
    crash trace makes it to disk before the process dies.
    """
    env = build_subprocess_env(model, hf_token="hf_xxx", hf_cache_dir="/hfcache")
    assert env["PYTHONUNBUFFERED"] == "1"


def test_cuda_device_order_in_extra_env_raises():
    """Operators cannot override CUDA_DEVICE_ORDER via extra_env."""

    class M:
        id = "m"
        gpu_indices = [0, 1]
        tensor_parallel_size = 2
        extra_env = {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}

    with pytest.raises(ValueError, match="CUDA_DEVICE_ORDER"):
        build_subprocess_env(M(), hf_token="hf_xxx", hf_cache_dir="/hfcache")


def test_pythonunbuffered_in_extra_env_raises():
    """v2026.05.15.5 lockdown: operators cannot disable PYTHONUNBUFFERED via
    extra_env. Today this is also blocked by the allowlist (no PYTHON_ prefix
    is accepted), but the explicit HARD_LOCKED entry is defence-in-depth — a
    future hand adding a PYTHON_ prefix for PYTHONFAULTHANDLER / PYTHONHASHSEED
    would otherwise let PYTHONUNBUFFERED=0 flow through and silently undo the
    fast-crash flush guarantee."""

    class M:
        id = "m"
        gpu_indices = [0, 1]
        tensor_parallel_size = 2
        extra_env = {"PYTHONUNBUFFERED": "0"}

    with pytest.raises(ValueError, match="PYTHONUNBUFFERED"):
        build_subprocess_env(M(), hf_token="hf_xxx", hf_cache_dir="/hfcache")


def test_env_builder_does_not_leak_warden_secrets(model, monkeypatch):
    """VW_COOKIE_SECRET, VW_ADMIN_PASSWORD etc. from parent env must NOT propagate."""
    monkeypatch.setenv("VW_COOKIE_SECRET", "session-key")
    monkeypatch.setenv("VW_ADMIN_PASSWORD", "supersecret")
    env = build_subprocess_env(model, hf_token="hf_xxx", hf_cache_dir="/hfcache")
    assert "VW_COOKIE_SECRET" not in env
    assert "VW_ADMIN_PASSWORD" not in env


def test_env_builder_empty_gpu_indices_raises():
    class M:
        id = "m"
        gpu_indices = []
        tensor_parallel_size = 1

    with pytest.raises(ValueError, match="gpu_indices"):
        build_subprocess_env(M(), hf_token="hf_xxx", hf_cache_dir="/hfcache")


def test_env_builder_tp_mismatch_raises():
    class M:
        id = "m"
        gpu_indices = [0, 1, 2]
        tensor_parallel_size = 2

    with pytest.raises(ValueError, match="tensor_parallel_size"):
        build_subprocess_env(M(), hf_token="hf_xxx", hf_cache_dir="/hfcache")


async def test_env_builder_roundtrips_through_db(tmp_data_dir):
    """Regression: 2026-05-10 bug. Previously ModelRow stored gpu_indices as a
    JSON string, so len() returned the string length (e.g. len('[1, 2]') == 6),
    triggering a spurious tensor_parallel_size mismatch in env_builder.
    This exercises the full insert → get → env_builder path that the live
    wizard uses but no test covered before."""
    from app.db.database import open_db
    from app.db.migrations import apply_migrations
    from app.db.repos.models import ModelRepo, ModelRow

    async with open_db(tmp_data_dir / "vllm-warden.db") as db:
        await apply_migrations(db)
        repo = ModelRepo(db)
        await repo.insert(
            ModelRow(
                id="m1",
                served_model_name="x",
                hf_repo="o/r",
                hf_revision="main",
                gpu_indices=[1, 2],
                tensor_parallel_size=2,
                dtype=None,
                max_model_len=None,
                gpu_memory_utilization=0.9,
                trust_remote_code=False,
                extra_args=[],
                extra_env={},
                status="pulled",
                pulled_bytes=0,
                pulled_total=None,
                last_error=None,
            )
        )
        loaded = await repo.get("m1")
    assert loaded.gpu_indices == [1, 2]
    env = build_subprocess_env(loaded, hf_token="hf_x", hf_cache_dir="/hfcache")
    assert env["CUDA_VISIBLE_DEVICES"] == "1,2"
